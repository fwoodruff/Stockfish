#!/usr/bin/env python3
"""Replicate the LMP re-allocation on an independent run: fit on --train, score only on --test.

Rules compared at equal pruning (same number of searched quiet moves beyond the threshold as
Stockfish's rule, matched by a global scale on the test set):
  type rule      : a threshold multiplier per node situation (PV / cut / all with static > alpha /
                   all with static <= alpha), chosen on --train
  search model   : multiplier exp(lambda * z) from a logistic model of search features
  search+network : the same with the 128 final-layer inputs added
Mistake: the move that raised alpha is quiet and lies beyond the threshold.
90% intervals: bootstrap over test searches (the global scale is held fixed).
"""
import argparse
import bisect
import itertools
import json

import numpy as np

import analyze as A


def load(path, acts_path=None):
    recs = [json.loads(l) for l in open(path)]
    X = None
    if acts_path:
        acts = np.load(acts_path)
        idx = {f: i for i, f in enumerate(acts["fens"])}
        recs = [r for r in recs if r["fen"] in idx]
        X = acts["acts"][[idx[r["fen"]] for r in recs]].astype(float)
    return recs, X


def prep(recs):
    qpos = [[i + 1 for i, c in enumerate(r["types"]) if c == "q"] for r in recs]
    T = np.array([r["threshold"] for r in recs], float)
    best = np.array([r["bestidx"] if (r["bestidx"] > 0 and r["bestquiet"]) else 0 for r in recs], float)
    cls = np.array([0 if r["pvnode"] else 1 if r["cutnode"] else 2 if r["static"] > r["alpha"] else 3
                    for r in recs])
    S = np.array([[r["depth"], r["improving"], r["threshold"], r["pvnode"], r["cutnode"], r["ttmove"],
                   np.log1p(abs(r["static"] - r["alpha"])), float(r["static"] > r["alpha"])] for r in recs], float)
    y = (best > T).astype(float)
    g = np.array([r["book_idx"] for r in recs])
    return qpos, T, best, cls, S, y, g


COST = None  # per-node weight of one pruned move (approximate subtree size), set in main


def per_node(qpos, best, thr):
    pruned = np.array([len(q) - bisect.bisect_right(q, t) for q, t in zip(qpos, thr)], float)
    if COST is not None and len(COST) == len(pruned):
        pruned = pruned * COST
    mist = ((best > 0) & (best > thr)).astype(float)
    return pruned, mist


def match_scale(qpos, best, T, mult, target):
    lo, hi = 0.1, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        p, _ = per_node(qpos, best, np.maximum(1, T * mult * mid))
        lo, hi = (mid, hi) if p.sum() > target else (lo, mid)
    return hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--train-acts", required=True)
    ap.add_argument("--test-acts", required=True)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--cost-base", type=float, default=1.0,
                    help="weight a pruned move by cost_base**depth (1 = count moves)")
    a = ap.parse_args()

    tr_recs, Xtr = load(a.train, a.train_acts)
    te_recs, Xte = load(a.test, a.test_acts)
    global COST
    qtr, Ttr, btr, ctr, Str, ytr, gtr = prep(tr_recs)
    qte, Tte, bte, cte, Ste, yte, gte = prep(te_recs)
    cost_tr = a.cost_base ** np.array([r["depth"] for r in tr_recs], float)
    cost_te = a.cost_base ** np.array([r["depth"] for r in te_recs], float)
    COST = cost_tr
    p0tr, _ = per_node(qtr, btr, Ttr)
    COST = cost_te
    p0, m0 = per_node(qte, bte, Tte)
    COST = cost_tr
    print(f"pruning measured as moves x {a.cost_base}^depth")
    print(f"train: {len(tr_recs)} nodes ({len(set(gtr))} searches); test: {len(te_recs)} nodes ({len(set(gte))} searches)")
    print(f"test, Stockfish's rule: pruned {int(p0.sum())} quiet moves, mistakes {int(m0.sum())}")
    names = ["PV", "cut", "all, static > alpha", "all, static <= alpha"]
    print("test nodes by situation: " + ", ".join(f"{n} {np.mean(cte == k):.1%}" for k, n in enumerate(names)))
    print("test mistakes by situation: " + ", ".join(f"{n} {int(m0[cte == k].sum())}" for k, n in enumerate(names)))

    rules = {}
    # type rule: grid of multipliers per situation, fitted on train at equal pruning
    grid = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    best_fit = None
    for m in itertools.product((1.0,), grid, grid, grid):
        mult = np.array(m)[ctr]
        s = match_scale(qtr, btr, Ttr, mult, p0tr.sum())
        _, mm = per_node(qtr, btr, np.maximum(1, Ttr * mult * s))
        if best_fit is None or mm.sum() < best_fit[0]:
            best_fit = (mm.sum(), m)
    rules["type rule " + str(best_fit[1])] = np.array(best_fit[1])[cte]

    acting_tr = np.array([len(q) - bisect.bisect_right(q, t) > 0 for q, t in zip(qtr, Ttr)])
    for name, Ftr, Fte in (("search model", Str, Ste),
                           ("search + network model", np.column_stack([Str, Xtr]), np.column_stack([Ste, Xte]))):
        mu, sd, w = A.logit_fit(Ftr[acting_tr], ytr[acting_tr], l2=1e-3)
        z_tr = np.column_stack([np.ones(len(Ftr)), (Ftr - mu) / sd]) @ w
        z_te = np.column_stack([np.ones(len(Fte)), (Fte - mu) / sd]) @ w
        zm, zs = z_tr.mean(), z_tr.std() + 1e-9
        best_lam = None
        for lam in (0.25, 0.5, 1.0, 1.5, 2.0):
            mult = np.clip(np.exp(lam * (z_tr - zm) / zs), 0.25, 4.0)
            s = match_scale(qtr, btr, Ttr, mult, p0tr.sum())
            _, mm = per_node(qtr, btr, np.maximum(1, Ttr * mult * s))
            if best_lam is None or mm.sum() < best_lam[0]:
                best_lam = (mm.sum(), lam)
        rules[f"{name} (lambda {best_lam[1]})"] = np.clip(np.exp(best_lam[1] * (z_te - zm) / zs), 0.25, 4.0)

    COST = cost_te
    rng = np.random.default_rng(0)
    ug, inv = np.unique(gte, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ug))]
    print("\nOn the test run, at equal pruning (90% interval over searches):")
    for name, mult in rules.items():
        s = match_scale(qte, bte, Tte, mult, p0.sum())
        p1, m1 = per_node(qte, bte, np.maximum(1, Tte * mult * s))
        ratios = []
        for _ in range(a.boot):
            pick = np.concatenate([members[k] for k in rng.integers(0, len(ug), len(ug))])
            ratios.append(m1[pick].sum() / max(m0[pick].sum(), 1))
        lo, hi = np.percentile(ratios, [5, 95])
        print(f"   {name:46} pruned {int(p1.sum())} (vs {int(p0.sum())}), mistakes {int(m1.sum())} vs {int(m0.sum())}: "
              f"{100 * (m1.sum() / m0.sum() - 1):+.1f}% [{100 * (lo - 1):+.1f}%, {100 * (hi - 1):+.1f}%]")


if __name__ == "__main__":
    main()
