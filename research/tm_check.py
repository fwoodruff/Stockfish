#!/usr/bin/env python3
"""Time-management check: does root head disagreement predict that more search changes the answer?

Pretend the engine has finished depth D0 (default 10) and must decide whether to keep thinking.
Targets, using the depth-16 search as what extra time would reveal:
  move_change : best move at 16 != best move at D0
  upside      : score(16) - score(D0) >= +T cp  (more time finds something better)
  downside    : score(16) - score(D0) <= -T cp
Baseline = signals Stockfish's time manager already has at that point: best-move changes so far,
depth since the last change, recent score trend, plus |score|, material and mobility.
"""
import argparse
import sys

import numpy as np
import pandas as pd

import analyze as A


def last_upto(m, d):
    keys = [int(k) for k in m if int(k) <= d]
    return m[str(max(keys))] if keys else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--d0", type=int, default=10)
    ap.add_argument("--swing", type=float, default=30, help="cp threshold T")
    a = ap.parse_args()

    df = A.add_features(A.load(a.files, 1000))
    df = df[df.src == "root"].copy()
    d0 = a.d0

    def seq(m, lo, hi):
        return [last_upto(m, d) for d in range(lo, hi + 1)]

    mv = df.depth_move.map(lambda m: seq(m, 1, d0))
    df["bm_changes"] = mv.map(lambda s: float(sum(x != y for x, y in zip(s, s[1:]) if x and y)))
    df["stable_for"] = mv.map(lambda s: float(next((i for i, (x, y) in enumerate(zip(s[::-1], s[-2::-1])) if x != y), len(s) - 1)))
    sc = lambda d: df.depth_cp.map(lambda m: last_upto(m, d)).astype(float)
    s0, s16 = sc(d0), sc(16)
    df["s0"] = s0
    df["trend"] = s0 - sc(d0 - 2)
    df["abs_trend"] = df.trend.abs()
    df["id_spread"] = pd.concat([sc(d) for d in range(d0 - 3, d0 + 1)], axis=1).std(axis=1)
    df["log_nodes"] = np.log(df.depth_nodes.map(lambda m: last_upto(m, d0)).astype(float))
    m16 = df.depth_move.map(lambda m: last_upto(m, 16))
    m0 = mv.map(lambda s: s[-1])
    df = df[s0.notna() & s16.notna() & (s0.abs() < 600)].copy()

    targets = {
        "move_change": (m16[df.index] != m0[df.index]),
        "upside": (s16[df.index] - s0[df.index]) >= a.swing,
        "downside": (s16[df.index] - s0[df.index]) <= -a.swing,
    }
    base_cols = ["bm_changes", "stable_for", "trend", "abs_trend", "id_spread", "log_nodes"]
    Xb = np.column_stack([df.s0.abs(), df.s0, df.pieces, A.tf(df.legal)] + [df[c] for c in base_cols])
    groups = df.book_idx.values
    print(f"roots: {len(df)} (D0={d0}, swing={a.swing} cp)")
    print("baseline = |score|, score, pieces, legal moves, best-move changes so far, depth since last change,")
    print("           score trend over last 2 depths, spread of last 4 scores, nodes so far")
    for name, y in targets.items():
        y = y.values.astype(float)
        print(f"\n[{name}] event rate {y.mean():.3f}")
        print(f"   {'proxy':14} {'event rate by proxy quintile (low -> high)':44} {'held-out gain':>13} {'90% interval':>18}")
        for p in ["head_nb", "head_spread", "mat_disagree", "captures", "checks"]:
            x = df[p].values
            ok = ~np.isnan(x)
            rates = A.quintile_rates(x[ok], y[ok])
            g, lo, hi = A.cluster_boot_gain(Xb[ok], x[ok], y[ok], groups[ok], reps=200)
            print(f"   {p:14} " + "".join(f"{r:8.3f}" for r in rates).ljust(44) + f" {100*g:12.2f}%  [{100*lo:5.2f}, {100*hi:5.2f}]")
        # for reference: how much the existing signals themselves explain
        b0 = A.logit_loss(np.column_stack([df.s0.abs(), df.s0, df.pieces, A.tf(df.legal)]), y)
        b1 = A.logit_loss(Xb, y)
        print(f"   (existing time-management signals reduce log-loss by {100*(b0-b1)/b0:.1f}% in-sample)")


if __name__ == "__main__":
    main()
