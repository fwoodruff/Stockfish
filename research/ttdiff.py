#!/usr/bin/env python3
"""Fixed-node, game-like comparison of TT variants against master.

For each book position, master plays a line of --plies moves at --nodes per move, keeping its
transposition table across moves. Every engine then searches each position of that same line,
also keeping its own table across moves. Fixed nodes make this deterministic, so differences come
from the change, not from timing. Reports per engine:
  differs     share of positions where its best move differs from master's
  continuity  share of positions where it plays the move its own search two plies earlier
              predicted (pv[2]), counted only when the line followed that prediction (pv[0], pv[1])
"""
import argparse
import random
import subprocess
from multiprocessing import Pool


def engine(path, opts, hash_mb):
    p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdin.write(f"setoption name Threads value 1\nsetoption name Hash value {hash_mb}\n")
    for k, v in opts.items():
        p.stdin.write(f"setoption name {k} value {v}\n")
    return p


def search(p, fen, moves, nodes):
    p.stdin.write(f"position fen {fen}" + (" moves " + " ".join(moves) if moves else "") + f"\ngo nodes {nodes}\n")
    pv = []
    for line in p.stdout:
        if " pv " in line and line.startswith("info depth"):
            pv = line.split(" pv ")[1].split()
        if line.startswith("bestmove"):
            return line.split()[1], pv


def run(args):
    fen, cfg = args
    master = engine(cfg["engines"][0][1], {}, cfg["hash"])
    master.stdin.write("ucinewgame\n")
    line, mpv = [], []
    for _ in range(cfg["plies"]):
        best, pv = search(master, fen, line, cfg["nodes"])
        if best in (None, "(none)"):
            break
        mpv.append((best, pv))
        line.append(best)
    master.stdin.write("quit\n")
    master.wait()

    res = {}
    for name, path, opts in cfg["engines"]:
        e = engine(path, opts, cfg["hash"])
        e.stdin.write("ucinewgame\n")
        out = [search(e, fen, line[:t], cfg["nodes"]) for t in range(len(line))]
        e.stdin.write("quit\n")
        e.wait()
        differs = sum(o[0] != m[0] for o, m in zip(out, mpv))
        cont = followed = 0
        for t in range(2, len(line)):
            pv = out[t - 2][1]
            if len(pv) >= 3 and pv[0] == line[t - 2] and pv[1] == line[t - 1]:
                followed += 1
                cont += out[t][0] == pv[2]
        res[name] = (differs, len(line), cont, followed)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--variant-opts", nargs="*", default=[], help="name=value[,name=value] per variant")
    ap.add_argument("--book", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--plies", type=int, default=24)
    ap.add_argument("--nodes", type=int, default=300000)
    ap.add_argument("--hash", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    engines = [("master", a.master, {}), ("master_again", a.master, {})]
    for spec in a.variant_opts:
        opts = dict(kv.split("=") for kv in spec.split(","))
        engines.append((spec, a.variant, opts))
    lines = open(a.book).read().splitlines()
    random.seed(9)
    fens = [" ".join(l.split()[:4]) + " 0 1" for l in random.sample(lines, a.n)]
    cfg = {"engines": engines, "plies": a.plies, "nodes": a.nodes, "hash": a.hash}
    with Pool(a.workers) as pool:
        results = pool.map(run, [(f, cfg) for f in fens])
    print(f"{a.n} lines x {a.plies} plies, {a.nodes} nodes/move, Hash {a.hash} MB")
    print(f"   {'engine':42} {'best move differs from master':>30} {'continuity (own pv[2])':>24}")
    for name, _, _ in engines:
        d = sum(r[name][0] for r in results); n = sum(r[name][1] for r in results)
        c = sum(r[name][2] for r in results); f = sum(r[name][3] for r in results)
        print(f"   {name:42} {d:6d}/{n:<6d} ({100 * d / n:5.1f}%)       {c:5d}/{f:<5d} ({100 * c / max(f, 1):5.1f}%)")


if __name__ == "__main__":
    main()
