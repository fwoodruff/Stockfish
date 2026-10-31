#!/usr/bin/env python3
"""Collect positions where search enters qsearch, with their quietness target and baselines.

For each book position: play a few game-like plies, then search --search-nodes with SampleQsearch
on and keep up to --per-root sampled qsearch entry positions. For each kept position, from a fresh
state: the `features` line (incl. capture counts) and a full-window qsearch (QsearchOnly), giving
gain = qsearch value - static eval (0 = quiet: captures change nothing).

Requires the research build (features with goodcaptures, SampleQsearch, QsearchOnly).
"""

import argparse
import json
import multiprocessing as mp
import random
import sys
import time

from collect import DEFAULT_BOOK, DEFAULT_ENGINE, Engine


def task(args):
    idx, fen, cfg = args
    rng = random.Random(cfg["seed"] * 1000003 + idx)
    eng = Engine(cfg["engine"])
    out = []
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

        eng.send(f"setoption name SampleQsearch value {cfg['sample_every']}")
        eng.send(f"go nodes {cfg['search_nodes']}")
        roots = []
        for line in eng.lines_until("bestmove"):
            if line.startswith("qsroot "):
                t = line.split()
                i = t.index("fen")
                roots.append({"pvnode": int(t[2]), "alpha": int(t[4]), "beta": int(t[6]),
                              "fen": " ".join(t[i + 1:])})
        eng.send("setoption name SampleQsearch value 0")
        rng.shuffle(roots)

        for r in roots[:cfg["per_root"]]:
            eng.new_position(r["fen"])
            f = eng.features()
            if f.get("incheck"):
                continue
            eng.send("setoption name QsearchOnly value true")
            eng.send("go depth 1")
            qs = None
            for line in eng.lines_until("bestmove"):
                if line.startswith("qsvalue"):
                    t = line.split()
                    qs = (int(t[1]), int(t[3]))
            eng.send("setoption name QsearchOnly value false")
            if qs is None:
                continue
            a = f["a"]
            out.append({"book_idx": idx, "fen": r["fen"], "a": a, "pvnode": r["pvnode"],
                        "alpha_cp": 100 * r["alpha"] / a, "beta_cp": 100 * r["beta"] / a,
                        "static_cp": 100 * qs[1] / a, "qs_cp": 100 * qs[0] / a,
                        "gain_cp": 100 * (qs[0] - qs[1]) / a,
                        **{k: int(f[k]) for k in ("legal", "captures", "checks", "goodcaptures",
                                                    "oppgoodcaptures", "pieces")}})
    finally:
        eng.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--book", default=DEFAULT_BOOK)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--min-walk", type=int, default=4)
    ap.add_argument("--max-walk", type=int, default=24)
    ap.add_argument("--walk-nodes", type=int, default=20000)
    ap.add_argument("--search-nodes", type=int, default=300000)
    ap.add_argument("--sample-every", type=int, default=200)
    ap.add_argument("--per-root", type=int, default=40)
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
                print(f"{k}/{a.n} roots, {kept} positions, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
