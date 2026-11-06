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

#include "evaluate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "misc.h"
#include "nnue/network.h"
#include "nnue/nnue_misc.h"
#include "position.h"
#include "types.h"
#include "tune.h"
#include "uci.h"
#include "nnue/nnue_accumulator.h"

namespace Stockfish {

static int simple_eval(const Position& pos) {
    const Color c = pos.side_to_move();
    return PawnValue * (pos.count<PAWN>(c) - pos.count<PAWN>(~c)) + pos.non_pawn_material(c)
         - pos.non_pawn_material(~c);
}

Value scale_evaluation(Value nnue, int optimism, const Position& pos);

// Expected score under uncertainty. With E(v) the expected score of eval v under the WDL model,
// an eval with error spread s is worth about E(v) + s^2 E''(v) / 2, i.e. eval v + s^2 J(v) / 2
// with J = E'' / E'. Since E has two steps (loss to draw, draw to win), J is positive from 0 up
// to the draw/win step and beyond the loss/draw step, and negative elsewhere: spread helps a
// side that is slightly ahead or clearly behind, and hurts one that is slightly behind or
// clearly ahead. Only the difference between positions matters (a transform shared by all
// positions does not change minimax), so the spread is the network's own estimate of its error
// size (a linear head on fc_2's inputs, see nnue_risk_weights.h) relative to its average at
// that eval.
int jensenScale = 0;
TUNE(SetRange(-32, 32), jensenScale);

namespace {

constexpr int JensenMinMaterial = 17, JensenMaxMaterial = 78, JensenStep = 8, JensenRange = 1024;

// J(v) * 2^20 for each WDL material count and eval (steps of 8 internal units)
struct JensenTable {
    i32 j[JensenMaxMaterial - JensenMinMaterial + 1][2 * JensenRange / JensenStep];

    JensenTable() {
        constexpr double as[] = {-142.72052667, 372.35176398, -340.71073572, 415.23490212};
        constexpr double bs[] = {5.93832785, 15.61267078, -30.57816876, 69.63866711};

        for (int mat = JensenMinMaterial; mat <= JensenMaxMaterial; ++mat)
        {
            const double m = mat / 58.0;
            const double a = ((as[0] * m + as[1]) * m + as[2]) * m + as[3];
            const double b = ((bs[0] * m + bs[1]) * m + bs[2]) * m + bs[3];

            auto w  = [&](double x) { return 1 / (1 + std::exp((a - x) / b)); };
            auto w1 = [&](double x) { return w(x) * (1 - w(x)) / b; };
            auto w2 = [&](double x) { return w1(x) * (1 - 2 * w(x)) / b; };

            for (int i = 0; i < 2 * JensenRange / JensenStep; ++i)
            {
                const double v  = i * JensenStep - JensenRange;
                const double e1 = (w1(v) + w1(-v)) / 2;
                const double e2 = (w2(v) - w2(-v)) / 2;
                j[mat - JensenMinMaterial][i] = i32(std::lround(e2 / e1 * (1 << 20)));
            }
        }
    }
};

const JensenTable jensenTable;

// Error variance factor exp(1.70 * x / 64) * 1024 for x in [-96, 96]: the risk head's unit
// (64) is 0.85 in log error size, so 1.70 in log variance. SpreadMean is its average over
// in-search evaluations (bench), so that the correction averages zero.
struct SpreadTable {
    i32 f[193];
    SpreadTable() {
        for (int x = -96; x <= 96; ++x)
            f[x + 96] = i32(std::lround(std::exp(1.70 * x / 64) * 1024));
    }
};
constexpr int SpreadMean = 3451;

const SpreadTable spreadTable;

}  // namespace

Value Eval::jensen_correction(Value v, Value nnue, int risk, const Position& pos) {

    // Only around the WDL steps: in clearly decided positions the expected score hardly moves
    if (std::abs(v) >= JensenRange)
        return VALUE_ZERO;

    const int material = pos.count<PAWN>() + 3 * pos.count<KNIGHT>() + 3 * pos.count<BISHOP>()
                       + 5 * pos.count<ROOK>() + 9 * pos.count<QUEEN>();
    const int mat = std::clamp(material, JensenMinMaterial, JensenMaxMaterial) - JensenMinMaterial;
    const int i   = (v + JensenRange) / JensenStep;

    // Spread relative to its average at this eval, centred so that the correction averages zero
    const int expected = 8 + 26 * std::min(std::abs(nnue), 1500) / 256;
    const int rel      = spreadTable.f[std::clamp(risk - expected, -96, 96) + 96] - SpreadMean;

    // jensenScale 64: s^2 / 2 = 5000 at the average spread (an error of 100 internal units)
    const i64 corr = i64(jensenScale) * 5000 / 64 * rel / 1024 * jensenTable.j[mat][i] / (1 << 20);
    return Value(std::clamp<i64>(corr, -128, 128));
}

Value Eval::evaluate(const Eval::NNUE::Network&     network,
                     const Position&                pos,
                     Eval::NNUE::AccumulatorStack&  accumulators,
                     Eval::NNUE::AccumulatorCaches& caches,
                     int                            optimism) {

    assert(!pos.checkers());
    const auto [nnue, risk] = network.evaluate(pos, accumulators, caches);
    Value v = scale_evaluation(nnue, optimism, pos);

    if (jensenScale)
        v = std::clamp(v + jensen_correction(v, nnue, risk, pos), VALUE_TB_LOSS_IN_MAX_PLY + 1,
                       VALUE_TB_WIN_IN_MAX_PLY - 1);

    return v;
}

// Applies search-dependent scaling (optimism and rule50) to the raw NNUE eval
Value scale_evaluation(Value nnue, int optimism, const Position& pos) {
    Value se = simple_eval(pos);

    // Normalize the raw evaluations to [-1024, 1024] to measure their correlation.
    int se_norm   = (se * 1024) / (std::abs(se) + 1024);
    int nnue_norm = (nnue * 1024) / (std::abs(nnue) + 1024);
    // When NNUE and material agree (positive correlation), the position is straightforward;
    // otherwise (negative correlation) it involves complex compensation. In a representative
    // sample, correl averages -1 or so, i.e. it is well-centered in [-2048, 2048].
    int correl = (se_norm * nnue_norm) / 512;

    // When winning, we boost easy positions, and vice versa
    int v = nnue + (nnue + optimism) * correl / 65536;

    // Scale the combined evaluation by total material
    int material = 521 * pos.count<PAWN>() + pos.non_pawn_material();
    v += i64(material) * v / 90649;

    // Damp down the evaluation linearly when shuffling
    v -= v * pos.rule50_count() / 189;

    // Guarantee that the evaluation does not hit the tablebase range
    v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

    return v;
}

// Like evaluate(), but instead of returning a value, it returns
// a string (suitable for outputting to stdout) that contains the detailed
// descriptions and values of each evaluation term. Useful for debugging.
// Trace scores are from white's point of view
std::string Eval::trace(Position& pos, const Eval::NNUE::Network& network) {

    if (pos.checkers())
        return "Final evaluation: none (in check)";

    auto accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();
    auto caches       = std::make_unique<Eval::NNUE::AccumulatorCaches>(network);

    std::stringstream ss;
    ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
    ss << '\n' << NNUE::trace(pos, network, *caches) << '\n';

    ss << std::showpoint << std::showpos << std::fixed << std::setprecision(2) << std::setw(15);

    Value nnue = network.evaluate(pos, *accumulators, *caches).value;
    Value s_v  = scale_evaluation(nnue, VALUE_ZERO, pos);  // requires stm perspective

    ss << "NNUE evaluation          " << nnue << " (side to move, internal units)\n";

    nnue = pos.side_to_move() == WHITE ? nnue : -nnue;
    s_v  = pos.side_to_move() == WHITE ? s_v : -s_v;

    ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(nnue, pos) << " (white side)\n";
    ss << "Final evaluation      ";
    ss << 0.01 * UCIEngine::to_cp(s_v, pos) << " (white side)";
    ss << " [with scaled NNUE, ...]\n";

    return ss.str();
}

}  // namespace Stockfish
