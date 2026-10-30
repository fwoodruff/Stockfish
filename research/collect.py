#!/usr/bin/env python3
"""Collect (static features, shallow search, deep search) records for the eval-error study.

Each task takes one book position, walks a few plies with fast engine moves, then
  * root mode: records features + an iterative-deepening search to --depth
  * node mode: runs a normal search with SampleNodes on, keeps a few of the nodes the
    search actually visited (with the alpha/beta/correction it saw there), and
    re-searches each of them deeply from scratch.

Output is JSONL; one record per position. All scores are side-to-move relative,
converted to centipawns with that position's own WDL normalisation (cp = 100 * v / a).

Requires the research build (branch research/eval-error) for `features` and SampleNodes.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENGINE = os.path.join(HERE, "..", "src", "stockfish")
DEFAULT_BOOK = os.path.join(HERE, "data/books/UHO_Lichess_4852_v1.epd")
MATE_CP = 10000


class Engine:
    def __init__(self, path):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        self.send("uci")
        self.wait_for("uciok")
        self.send("setoption name Threads value 1")
        self.send("setoption name Hash value 16")
        self.sample_every = 0

    def send(self, cmd):
        self.p.stdin.write(cmd + "\n")

    def lines_until(self, prefix):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            line = line.rstrip("\n")
            yield line
            if line.startswith(prefix):
                return

    def wait_for(self, prefix):
        for _ in self.lines_until(prefix):
            pass

    def sync(self):
        self.send("isready")
        self.wait_for("readyok")

    def set_sampling(self, every):
        # HeadExtra 3 makes every main-search node compute its head disagreement (also at TT
        # hits), so samples carry their own and their parent's; it does not change the search
        if every != self.sample_every:
            self.send(f"setoption name SampleNodes value {every}")
            self.send(f"setoption name HeadExtra value {3 if every else 0}")
            self.sample_every = every

    def new_position(self, fen, moves=(), clear=True):
        if clear:
            self.send("ucinewgame")
        self.send(f"position fen {fen}" + (" moves " + " ".join(moves) if moves else ""))
        self.sync()

    def fen(self):
        self.send("d")
        out = None
        for line in self.lines_until("Checkers:"):
            if line.startswith("Fen: "):
                out = line[5:]
        return out

    def features(self):
        self.send("features")
        tok = next(l for l in self.lines_until("features") if l.startswith("features")).split()[1:]
        f = {}
        for k, v in zip(tok[::2], tok[1::2]):
            f[k] = [int(x) for x in v.split(",")] if k == "heads" else float(v)
        return f

    def go(self, limit):
        """Run a search; return (bestmove, per-depth info, sample lines)."""
        self.send(f"go {limit}")
        depths, samples, best = {}, [], None
        for line in self.lines_until("bestmove"):
            if line.startswith("sample "):
                samples.append(line)
            elif line.startswith("info depth") and " score " in line and " pv " in line:
                t = line.split()
                if "lowerbound" in t or "upperbound" in t:
                    continue
                d = int(t[t.index("depth") + 1])
                kind, val = t[t.index("score") + 1], int(t[t.index("score") + 2])
                cp = val if kind == "cp" else (MATE_CP if val > 0 else -MATE_CP)
                depths[d] = {"cp": cp, "mate": kind == "mate",
                             "nodes": int(t[t.index("nodes") + 1]),
                             "move": t[t.index("pv") + 1]}
            elif line.startswith("bestmove"):
                best = line.split()[1]
        return best, depths, samples

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def parse_sample(line):
    t = line.split()
    i = t.index("fen")
    d = {k: int(v) for k, v in zip(t[1:i:2], t[2:i:2])}
    d["fen"] = " ".join(t[i + 1:])
    return d


def deep_record(eng, fen, depth, extra):
    eng.new_position(fen)
    f = eng.features()
    if f.get("incheck"):
        return None
    _, depths, _ = eng.go(f"depth {depth}")
    if depth not in depths:
        return None
    a = f["a"]
    rec = {"fen": fen, "a": a,
           "static_cp": 100 * f["scaled"] / a,  # what search would see, minus correction history
           "nnue_cp": 100 * f["nnue"] / a,
           "simple_cp": 100 * f["simple"] / a,
           "heads_cp": [100 * h / a for h in f["heads"]],
           "bucket": int(f["bucket"]),
           **{k: int(f[k]) for k in ("legal", "captures", "checks", "pieces", "rule50", "ply")},
           "deep_cp": depths[depth]["cp"], "deep_mate": depths[depth]["mate"],
           "depth_cp": {d: v["cp"] for d, v in depths.items()},
           "depth_move": {d: v["move"] for d, v in depths.items()},
           "depth_nodes": {d: v["nodes"] for d, v in depths.items()}}
    rec.update(extra)
    return rec


def task(args):
    idx, fen, cfg = args
    rng = random.Random(cfg["seed"] * 1000003 + idx)
    eng = Engine(cfg["engine"])
    out = []
    try:
        # Play a few plies like a game (hash and histories kept between moves), so that
        # correction history is warm when we sample, as it would be in real play
        eng.set_sampling(0)
        eng.new_position(fen)
        moves = []
        for _ in range(rng.randint(cfg["min_walk"], cfg["max_walk"])):
            eng.new_position(fen, moves, clear=False)
            best, _, _ = eng.go(f"nodes {cfg['walk_nodes']}")
            if best in (None, "(none)"):
                break
            moves.append(best)
        eng.new_position(fen, moves, clear=False)
        root = eng.fen()

        if cfg["mode"] in ("node", "both"):
            eng.set_sampling(cfg["sample_every"])
            _, _, samples = eng.go(f"nodes {cfg['sample_search_nodes']}")
            eng.set_sampling(0)
            samples = [parse_sample(s) for s in samples]
            rng.shuffle(samples)
            for s in samples[:cfg["samples_per_root"]]:
                extra = {"src": "node", "book_idx": idx, "root_fen": root,
                         **{f"s_{k}": v for k, v in s.items() if k != "fen"}}
                r = deep_record(eng, s["fen"], cfg["depth"], extra)
                if r:
                    a = r["a"]  # node values below are internal units of this same position
                    for k in ("alpha", "beta", "raw", "static", "headdis", "parentheaddis"):
                        if k in s and s[k] != -1:
                            r[f"s_{k}_cp"] = 100 * s[k] / a
                    out.append(r)

        if cfg["mode"] in ("root", "both"):
            r = deep_record(eng, root, cfg["depth"], {"src": "root", "book_idx": idx})
            if r:
                out.append(r)
    finally:
        eng.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500, help="number of book positions")
    ap.add_argument("--mode", choices=["root", "node", "both"], default="both")
    ap.add_argument("--depth", type=int, default=14, help="'deep' search depth")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--book", default=DEFAULT_BOOK)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-walk", type=int, default=4)
    ap.add_argument("--max-walk", type=int, default=24)
    ap.add_argument("--walk-nodes", type=int, default=20000)
    ap.add_argument("--sample-search-nodes", type=int, default=200000)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--samples-per-root", type=int, default=6)
    a = ap.parse_args()

    with open(a.book) as f:
        book = f.read().splitlines()
    rng = random.Random(a.seed)
    picks = rng.sample(range(len(book)), a.n)
    cfg = {"engine": a.engine, "mode": a.mode, "depth": a.depth, "seed": a.seed,
           "min_walk": a.min_walk, "max_walk": a.max_walk, "walk_nodes": a.walk_nodes,
           "sample_search_nodes": a.sample_search_nodes, "sample_every": a.sample_every,
           "samples_per_root": a.samples_per_root}
    meta = {"meta": True, **cfg, "n": a.n, "book": os.path.basename(a.book), "time": time.time()}

    t0, done, recs = time.time(), 0, 0
    with open(a.out, "a") as fo, mp.Pool(a.workers) as pool:
        fo.write(json.dumps(meta) + "\n")
        jobs = [(i, " ".join(book[i].split()[:4]) + " 0 1", cfg) for i in picks]
        for res in pool.imap_unordered(task, jobs):
            for r in res:
                fo.write(json.dumps(r) + "\n")
            fo.flush()
            done += 1
            recs += len(res)
            if done % 20 == 0 or done == a.n:
                el = time.time() - t0
                print(f"{done}/{a.n} roots, {recs} records, {el:.0f}s elapsed, "
                      f"eta {el / done * (a.n - done):.0f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
