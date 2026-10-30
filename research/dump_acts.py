#!/usr/bin/env python3
"""Dump the inputs to the final NNUE layer (the "penultimate" activations of the head in use)
for every position in the given collect.py outputs. Writes an .npz with fens, buckets, acts.

Requires the research build (the `acts` UCI command).
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENGINE = os.path.join(HERE, "..", "src", "stockfish")


def work(args):
    engine, fens = args
    p = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    out = []
    for fen in fens:
        p.stdin.write(f"position fen {fen}\nacts\n")
        while True:
            line = p.stdout.readline()
            if line.startswith("acts"):
                break
        t = line.split()
        out.append(None if t[1] == "none" else (int(t[1]), [int(x) for x in t[2].split(",")]))
    p.stdin.write("quit\n")
    p.wait()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    fens = set()
    for f in a.files:
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if not r.get("meta") and "fen" in r:
                    fens.add(r["fen"])
    fens = sorted(fens)
    chunks = [fens[i::a.workers * 4] for i in range(a.workers * 4)]
    with mp.Pool(a.workers) as pool:
        res = pool.map(work, [(a.engine, c) for c in chunks])

    keep_fens, buckets, acts = [], [], []
    for c, r in zip(chunks, res):
        for fen, x in zip(c, r):
            if x is not None:
                keep_fens.append(fen)
                buckets.append(x[0])
                acts.append(x[1])
    np.savez_compressed(a.out, fens=np.array(keep_fens), buckets=np.array(buckets, dtype=np.int16),
                        acts=np.array(acts, dtype=np.int16))
    print(f"{len(keep_fens)} positions, {len(acts[0])} activations each -> {a.out}")


if __name__ == "__main__":
    main()
