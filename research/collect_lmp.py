#!/usr/bin/env python3
"""Collect nodes where late move pruning can apply, from searches with LMP switched off.

For each book position: play a few game-like plies, then search --search-nodes with NoLMP and
SampleLMP on, keeping up to --per-root sampled nodes. Each record has the node's depth, improving,
Stockfish's LMP threshold, the move-order position of the best move and whether it is quiet, and
the order of searched quiet/capture moves. Requires the research build (NoLMP, SampleLMP).
"""

import argparse
import json
import multiprocessing as mp
import random
import sys
import time

from collect import DEFAULT_BOOK, DEFAULT_ENGINE, Engine

INT_KEYS = ("depth", "improving", "threshold", "pvnode", "cutnode", "ttmove", "alpha", "beta",
            "static", "best", "bestidx", "bestquiet")


def parse(line):
    t = line.split()
    i = t.index("fen")
    d = {t[k]: t[k + 1] for k in range(1, i, 2)}
    rec = {k: int(d[k]) for k in INT_KEYS}
    rec["types"] = d["types"]
    rec["fen"] = " ".join(t[i + 1:])
    return rec


def task(args):
    idx, fen, cfg = args
    rng = random.Random(cfg["seed"] * 1000003 + idx)
    eng = Engine(cfg["engine"])
    try:
        eng.new_position(fen)
        moves = []
        for _ in range(rng.randint(cfg["min_walk"], cfg["max_walk"])):
            eng.new_position(fen, moves, clear=False)
            best, _, _ = eng.go(f"nodes {cfg['walk_nodes']}")
            if best in (None, "(none)"):
                break
            moves.append(best)
        eng.new_position(fen, moves, clear=False)
        eng.send("setoption name NoLMP value true")
        eng.send(f"setoption name SampleLMP value {cfg['sample_every']}")
        eng.send(f"go nodes {cfg['search_nodes']}")
        recs = [parse(l) for l in eng.lines_until("bestmove") if l.startswith("lmpnode ")]
        eng.send("setoption name SampleLMP value 0")
        eng.send("setoption name NoLMP value false")
    finally:
        eng.close()
    rng.shuffle(recs)
    return [{"book_idx": idx, **r} for r in recs[:cfg["per_root"]]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--book", default=DEFAULT_BOOK)
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--min-walk", type=int, default=4)
    ap.add_argument("--max-walk", type=int, default=24)
    ap.add_argument("--walk-nodes", type=int, default=20000)
    ap.add_argument("--search-nodes", type=int, default=300000)
    ap.add_argument("--sample-every", type=int, default=300)
    ap.add_argument("--per-root", type=int, default=60)
    a = ap.parse_args()

    book = open(a.book).read().splitlines()
    picks = random.Random(a.seed).sample(range(len(book)), a.n)
    cfg = {k: getattr(a, k) for k in ("engine", "seed", "min_walk", "max_walk", "walk_nodes",
                                      "search_nodes", "sample_every", "per_root")}
    t0, kept = time.time(), 0
    with open(a.out, "w") as fo, mp.Pool(a.workers) as pool:
        jobs = [(i, " ".join(book[i].split()[:4]) + " 0 1", cfg) for i in picks]
        for k, res in enumerate(pool.imap_unordered(task, jobs), 1):
            for r in res:
                fo.write(json.dumps(r) + "\n")
            kept += len(res)
            if k % 100 == 0 or k == a.n:
                print(f"{k}/{a.n} roots, {kept} nodes, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
