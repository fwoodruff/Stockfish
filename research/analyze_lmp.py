#!/usr/bin/env python3
"""Late move pruning (LMP) offline study.

Mistake: the move that raised alpha is quiet and comes after Stockfish's LMP threshold (moveCount),
so LMP would have pruned it. Pruned: searched quiet moves after the threshold.

1. Among nodes where LMP prunes something, how well do search features vs the network readout
   predict a mistake (held-out by search)?
2. Per-node thresholds T' = T * m(score) with m from each model's predicted mistake log-odds,
   at the same total number of pruned moves as Stockfish's rule: how many mistakes?
"""
import argparse
import json

import numpy as np

import analyze as A


def auc(y, p):
    order = np.argsort(p); r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def counts(recs, thresholds):
    pruned = mist = 0
    for r, t in zip(recs, thresholds):
        pruned += sum(1 for i, c in enumerate(r["types"]) if c == "q" and i + 1 > t)
        mist += r["bestidx"] > 0 and r["bestquiet"] and r["bestidx"] > t
    return pruned, mist


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--acts", required=True)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.data)]
    acts = np.load(a.acts)
    idx = {f: i for i, f in enumerate(acts["fens"])}
    recs = [r for r in recs if r["fen"] in idx]
    g = np.array([r["book_idx"] for r in recs]); ug = np.unique(g)
    trg = set(np.random.default_rng(0).choice(ug, int(0.7 * len(ug)), replace=False))
    tr = np.array([x in trg for x in g]); te = ~tr

    def feats(rs):
        # Only what the search knows before the decision. (The number and kind of moves counted
        # are NOT allowed: the move loop stops at a cutoff, so they encode the outcome.)
        return np.array([[r["depth"], r["improving"], r["threshold"], r["pvnode"], r["cutnode"], r["ttmove"],
                          np.log1p(abs(r["static"] - r["alpha"])), float(r["static"] > r["alpha"])]
                         for r in rs], float)
    S = feats(recs)
    X = acts["acts"][[idx[r["fen"]] for r in recs]].astype(float)
    acting = np.array([any(c == "q" and i + 1 > r["threshold"] for i, c in enumerate(r["types"])) for r in recs])
    y = np.array([r["bestidx"] > 0 and r["bestquiet"] and r["bestidx"] > r["threshold"] for r in recs], float)
    print(f"{len(recs)} nodes; LMP prunes something at {acting.mean():.1%}; mistakes {int(y.sum())} "
          f"({y[acting].mean():.2%} of acting nodes)")

    print("\n1. Predicting a mistake among nodes where LMP acts (held-out by search):")
    models = {}
    for n, F in (("search features", S), ("network readout", X), ("both", np.column_stack([S, X]))):
        m = A.logit_fit(F[tr & acting], y[tr & acting], l2=1e-3)
        mu, sd, w = m
        score = np.column_stack([np.ones(len(F)), (F - mu) / sd]) @ w
        models[n] = score
        print(f"   AUC {auc(y[te & acting], score[te & acting]):.3f}  {n}")

    print("\n2. Per-node thresholds at equal pruning (test nodes):")
    test = [r for r, t in zip(recs, te) if t]
    T = np.array([r["threshold"] for r in test], float)
    p0, m0 = counts(test, T)
    print(f"   Stockfish's rule: pruned {p0} quiet moves, mistakes {m0}")
    for n, score in models.items():
        s = score[te]
        s = (s - s.mean()) / (s.std() + 1e-9)
        best = None
        for lam in (0.25, 0.5, 1.0, 1.5, 2.0):
            mult = np.clip(np.exp(lam * s), 0.25, 4.0)      # risky nodes get a higher threshold
            lo, hi = 0.2, 5.0
            for _ in range(40):                              # scale so pruning matches Stockfish's rule
                mid = (lo + hi) / 2
                p, _ = counts(test, np.maximum(1, T * mult * mid))
                lo, hi = (mid, hi) if p > p0 else (lo, mid)
            p, mm = counts(test, np.maximum(1, T * mult * hi))
            if best is None or mm < best[2]:
                best = (lam, p, mm)
        lam, p, mm = best
        print(f"   {n:16} best lambda {lam}: pruned {p}, mistakes {mm} ({100 * (mm - m0) / max(m0, 1):+.1f}% vs Stockfish)")


if __name__ == "__main__":
    main()
