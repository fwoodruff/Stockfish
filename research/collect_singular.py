#!/usr/bin/env python3
"""Collect singular extension verification searches (outermost only) from real searches.

For each book position: play a few game-like plies (hash and histories kept), then search
--search-nodes with SampleSingular on. Writes JSONL with one 'search' record per root (total
nodes, nodes spent in outermost verification searches, outcome counts) and up to
--events-per-root randomly chosen 'event' records (outcome, node cost, search context, fen).

Requires the research build (SampleSingular option).
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

from collect import DEFAULT_BOOK, DEFAULT_ENGINE, Engine

KEYS_INT = ("nodes", "depth", "ttdepth", "pvnode", "cutnode", "ttpv", "ttcapture", "improving", "ply",
            "alpha", "beta", "ttvalue", "static", "sbeta", "value", "corr", "ttmovehist")


def parse(line):
    t = line.split()
    i = t.index("fen")
    d = {t[k]: t[k + 1] for k in range(1, i, 2)}
    rec = {k: int(d[k]) for k in KEYS_INT}
    rec["outcome"] = d["outcome"]
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

        eng.send("setoption name SampleSingular value true")
        eng.send(f"go nodes {cfg['search_nodes']}")
        events, total = [], 0
        for line in eng.lines_until("bestmove"):
            if line.startswith("singular "):
                events.append(parse(line))
            elif line.startswith("info depth") and " nodes " in line:
                t = line.split()
                total = int(t[t.index("nodes") + 1])
        eng.send("setoption name SampleSingular value false")
    finally:
        eng.close()

    counts = {o: sum(e["outcome"] == o for e in events) for o in "SMN"}
    search = {"kind": "search", "book_idx": idx, "total_nodes": total, "events": len(events),
              "singular_nodes": sum(e["nodes"] for e in events), **{f"n_{o}": c for o, c in counts.items()}}
    rng.shuffle(events)
    out = [search] + [{"kind": "event", "book_idx": idx, **e} for e in events[:cfg["events_per_root"]]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--book", default=DEFAULT_BOOK)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--min-walk", type=int, default=4)
    ap.add_argument("--max-walk", type=int, default=24)
    ap.add_argument("--walk-nodes", type=int, default=20000)
    ap.add_argument("--search-nodes", type=int, default=500000)
    ap.add_argument("--events-per-root", type=int, default=60)
    a = ap.parse_args()

    with open(a.book) as f:
        book = f.read().splitlines()
    picks = random.Random(a.seed).sample(range(len(book)), a.n)
    cfg = {k: getattr(a, k) for k in ("engine", "seed", "min_walk", "max_walk", "walk_nodes",
                                      "search_nodes", "events_per_root")}
    t0 = time.time()
    with open(a.out, "w") as fo, mp.Pool(a.workers) as pool:
        jobs = [(i, " ".join(book[i].split()[:4]) + " 0 1", cfg) for i in picks]
        for k, res in enumerate(pool.imap_unordered(task, jobs), 1):
            for r in res:
                fo.write(json.dumps(r) + "\n")
            if k % 100 == 0 or k == a.n:
                print(f"{k}/{a.n} roots, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
