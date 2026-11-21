/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CORRECTION_H_INCLUDED
#define CORRECTION_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "bitboard.h"
#include "misc.h"
#include "position.h"
#include "types.h"

namespace Stockfish {

// Correction history with learned keys. A key is a set of piece planes (colour x piece type), each read at one
// of three resolutions (present, count or exact squares), plus a set of recent moves (lags 1 to 6). The
// correction is the weighted sum of one table entry per active key. Entries learn the error between the search
// result and the corrected static evaluation. A few candidate keys learn the same error in the background without
// being used, and each key keeps a running estimate of how much it reduces the combined error. Periodically the
// best candidate is promoted if it reduces the error enough, and active keys that stopped paying are retired.
// The game starts from the keys of the classic correction histories (pawn, minor, non-pawn and continuations).
class CorrectionKeys {
   public:
    static constexpr int MaxActive = 16;
    static constexpr int MaxLag    = 6;

    // A key specification: bits 2p..2p+1 give the resolution of plane p = colour * 6 + piece type - 1
    // (0 off, 1 present, 2 count, 3 squares), bits 24..29 select the lags of the moves taken into account.
    using Spec = u32;

    CorrectionKeys() :
        table(size_t(1) << LogSize) {
        u64 s = 0x9E3779B97F4A7C15ull;
        for (auto& p : countRand)
            for (auto& v : p)
                v = next(s);
        for (auto& v : presentRand)
            v = next(s);
        clear();
    }

    void clear() {
        std::fill(table.begin(), table.end(), i16(0));
        active.clear();
        for (Spec s : start_specs())
            active.push_back(compile(s));
        candidates.clear();
        queue.clear();
        for (Spec s : pool_specs())
        {
            if (is_active(s))
                continue;
            if (int(candidates.size()) < Live)
                candidates.push_back(compile(s));
            else
                queue.push_back(s);
        }
        updates = 0;
        generation++;
    }

    int active_count() const { return int(active.size()); }
    u32 gen() const { return generation; }
    int weight() const {
        return std::min(KeyWeight, TotalWeight / std::max(1, int(active.size())));
    }
    void prefetch_entries(const u32* idx, int n) const {
        for (int k = 0; k < n; ++k)
            prefetch(&table[idx[k]]);
    }

    // Hash of a move (moved piece * 64 + to square), computed once per move and kept on the search stack
    static u64 move_hash(int moveKey) { return (u64(moveKey) + 1) * 0xD6E8FEB86659FD93ull; }

    // Table indices of the active keys at a position; moveHashes[j] is the hash of the move j + 1 plies ago
    void indices(const Position& pos, const u64* moveKeys, u32* out) const {
        for (int k = 0; k < int(active.size()); ++k)
            out[k] = index(pos, moveKeys, active[k]);
    }

    // Correction value in the scale of the classic histories (internal units * 131072)
    int value(const u32* idx) const {
        int sum = 0;
        for (int k = 0; k < int(active.size()); ++k)
            sum += table[idx[k]];
        return sum * weight();
    }

    // delta: searched value - corrected static evaluation; bonus: the scaled update; rate: 0..1024, by depth
    void update(
      const Position& pos, const u64* moveKeys, const u32* idx, int delta, int bonus, int rate) {
        const int    r = std::clamp(delta, -512, 512);
        const int    w = weight();
        const double a = double(rate) / (1024.0 * 8192.0);

        for (int k = 0; k < int(active.size()); ++k)
        {
            i16&      e = table[idx[k]];
            const int c = e * w / 131072;
            active[k].value += (double((r + c) * (r + c) - r * r) - active[k].value) * a;
            active[k].age++;
            learn(e, bonus);
        }
        for (auto& cand : candidates)
        {
            i16&      e = table[index(pos, moveKeys, cand)];
            const int c = e * w / 131072;
            cand.value += (double(r * r - (r - c) * (r - c)) - cand.value) * a;
            cand.age++;
            learn(e, bonus);
        }
        if (++updates % Period == 0)
            manage();
    }

   private:
    static constexpr int    LogSize = 20;
    static constexpr int    Live    = 8;      // candidates learning at a time
    static constexpr int    Period  = 50000;  // updates between promotion decisions
    static constexpr int    MinAge  = 50000;  // updates before a key is judged
    static constexpr double Threshold =
      100;  // error reduction (internal units squared) to earn a place
    static constexpr int KeyWeight = 10630;  // matches the classic histories' mean correction size
    static constexpr int TotalWeight = 7 * KeyWeight;

    struct Key {
        Spec   spec;
        u16    squares, counts, present;  // plane masks
        u8     lags;
        u64    salt;
        double value;
        int    age;
    };

    static u64 next(u64& s) {
        u64 z = (s += 0x9E3779B97F4A7C15ull);
        z     = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z     = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    static Key compile(Spec s) {
        Key k{s, 0, 0, 0, u8((s >> 24) & 63), 0, 0.0, 0};
        for (int p = 0; p < 12; ++p)
            switch ((s >> (2 * p)) & 3)
            {
            case 1 :
                k.present |= 1 << p;
                break;
            case 2 :
                k.counts |= 1 << p;
                break;
            case 3 :
                k.squares |= 1 << p;
                break;
            default :
                break;
            }
        u64 x  = s ^ 0x5EED0000ull;
        k.salt = next(x);
        return k;
    }

    static Spec planes(std::initializer_list<int> ps, u32 res = 3) {
        Spec s = 0;
        for (int p : ps)
            s |= res << (2 * p);
        return s;
    }
    static Spec lags(u32 mask) { return mask << 24; }

    static std::vector<Spec> start_specs() {
        return {
          planes({0, 6}), planes({1, 2, 7, 8}), planes({1, 2, 3, 4, 5}), planes({7, 8, 9, 10, 11}),
          lags(0b000011), lags(0b001001),       lags(0b100001)};
    }

    static std::vector<Spec> pool_specs() {
        std::vector<Spec> v;
        for (int p = 0; p < 12; ++p)
            v.push_back(planes({p}));
        for (Spec s : {planes({0, 6}), planes({1, 2, 7, 8}), planes({1, 2, 3, 4, 5}),
                       planes({7, 8, 9, 10, 11}), planes({3, 4, 9, 10}), planes({5, 11}),
                       planes({0, 5, 6, 11}), planes({0, 1, 2, 3, 4, 5}),
                       planes({6, 7, 8, 9, 10, 11}), planes({0, 1, 2, 3, 4, 6, 7, 8, 9, 10}, 2),
                       planes({4, 10}, 1), planes({0, 6}) | planes({4, 10}, 1)})
            v.push_back(s);
        for (u32 m : {0b000001u, 0b000011u, 0b000101u, 0b001001u, 0b100001u, 0b000111u, 0b000010u})
            v.push_back(lags(m));
        v.push_back(planes({0, 6}) | lags(0b000001));
        return v;
    }

    bool is_active(Spec s) const {
        return std::any_of(active.begin(), active.end(), [s](const Key& k) { return k.spec == s; });
    }

    __attribute__((always_inline)) u32 index(const Position& pos,
                                             const u64*      moveKeys,
                                             const Key&      k) const {
        u64 h = k.salt;
        for (u16 m = k.squares; m; m &= m - 1)
            h ^= pos.plane_key(lsb(m));
        for (u16 m = k.counts; m; m &= m - 1)
        {
            const int p = lsb(m);
            h ^=
              countRand[p][std::min(popcount(pos.pieces(Color(p / 6), PieceType(p % 6 + 1))), 15)];
        }
        for (u16 m = k.present; m; m &= m - 1)
        {
            const int p = lsb(m);
            h ^= pos.pieces(Color(p / 6), PieceType(p % 6 + 1)) ? presentRand[p] : 0;
        }
        for (u8 m = k.lags; m; m &= m - 1)
        {
            const int j = lsb(m);
            h ^= (moveKeys[j] << (7 * j + 1)) | (moveKeys[j] >> (63 - 7 * j));
        }
        h *= 0x9E3779B97F4A7C15ull;
        return u32((h >> (64 - LogSize + 1)) << 1) | u32(pos.side_to_move());
    }

    static int lsb(u32 m) { return __builtin_ctz(m); }

    static void learn(i16& e, int bonus) {
        const int b = std::clamp(bonus, -1024, 1024);
        e           = i16(e + b - e * std::abs(b) / 1024);
    }

    void manage() {
        int best = -1;
        for (int k = 0; k < int(candidates.size()); ++k)
            if (candidates[k].age >= MinAge
                && (best < 0 || candidates[k].value > candidates[best].value))
                best = k;
        if (best >= 0 && candidates[best].value > Threshold && int(active.size()) < MaxActive)
        {
            active.push_back(candidates[best]);
            active.back().value = 0;
            active.back().age   = 0;
            candidates.erase(candidates.begin() + best);
            if (!queue.empty())
            {
                candidates.push_back(compile(queue.front()));
                queue.erase(queue.begin());
            }
            generation++;
        }
        // A candidate that had its trial without being promoted makes room for the next one
        if (!queue.empty())
            for (auto& cand : candidates)
                if (cand.age >= 2 * MinAge)
                {
                    queue.push_back(cand.spec);
                    cand = compile(queue.front());
                    queue.erase(queue.begin());
                }
        int worst = -1;
        for (int k = 0; k < int(active.size()); ++k)
            if (active[k].age >= MinAge && (worst < 0 || active[k].value < active[worst].value))
                worst = k;
        if (worst >= 0 && active.size() > 1 && active[worst].value < Threshold / 2)
        {
            queue.push_back(active[worst].spec);
            active.erase(active.begin() + worst);
            generation++;
        }
    }

    std::vector<i16>  table;
    std::vector<Key>  active, candidates;
    std::vector<Spec> queue;
    u64               countRand[12][16], presentRand[12];
    long long         updates    = 0;
    u32               generation = 0;
};

}  // namespace Stockfish

#endif  // #ifndef CORRECTION_H_INCLUDED
