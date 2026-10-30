#!/usr/bin/env python3
"""Can the final NNUE layer's inputs predict singular extension verification outcomes, beyond what
the search already knows when it starts the verification? And how many nodes could gating save?

Outcomes: S = singular (value < singularBeta), M = multi-cut (value >= beta), N = neither.
Split by search (book position): fit on --train-frac of searches, score on the rest.
"""

import argparse
import json

import numpy as np

import analyze as A


def load(path):
    ev, se = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            (ev if r["kind"] == "event" else se).append(r)
    return ev, se


def baseline_features(e):
    return [e["depth"], e["ttdepth"] - e["depth"], e["pvnode"], e["cutnode"], e["ttpv"], e["ttcapture"],
            e["improving"], e["ply"], e["ttvalue"] - e["beta"], e["static"] - e["ttvalue"],
            e["ttvalue"] - e["sbeta"], abs(e["corr"]) / 131072, e["ttmovehist"]]


def auc(y, p):
    order = np.argsort(p)
    r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--acts", required=True)
    ap.add_argument("--train-frac", type=float, default=0.7)
    a = ap.parse_args()

    ev, se = load(a.data)
    tot = sum(s["total_nodes"] for s in se)
    sing = sum(s["singular_nodes"] for s in se)
    print(f"{len(se)} searches, {tot / 1e6:.0f}M nodes; outermost verification searches use "
          f"{100 * sing / tot:.1f}% of all nodes")
    print("outcome counts (all events):", {o: sum(s[f'n_{o}'] for s in se) for o in "SMN"})

    acts = np.load(a.acts)
    idx = {f: i for i, f in enumerate(acts["fens"])}
    ev = [e for e in ev if e["fen"] in idx]
    B = np.array([baseline_features(e) for e in ev], dtype=float)
    X = acts["acts"][[idx[e["fen"]] for e in ev]].astype(float)
    nodes = np.array([e["nodes"] for e in ev], dtype=float)
    out = np.array([e["outcome"] for e in ev])
    g = np.array([e["book_idx"] for e in ev])
    ug = np.unique(g)
    rng = np.random.default_rng(0)
    train_groups = set(rng.choice(ug, int(len(ug) * a.train_frac), replace=False))
    tr = np.array([x in train_groups for x in g])
    te = ~tr
    print(f"events with positions: {len(ev)} (train {tr.sum()}, test {te.sum()}); "
          f"node cost per event: median {np.median(nodes):.0f}, mean {nodes.mean():.0f}, "
          f"share of sampled cost in the top 10% of events {100 * np.sort(nodes)[-len(nodes) // 10:].sum() / nodes.sum():.0f}%")

    preds = {}
    for o in "SMN":
        y = (out == o).astype(float)
        res = {}
        for name, F in (("search features", B), ("+ 128 network inputs", np.column_stack([B, X]))):
            m = A.logit_fit(F[tr], y[tr], l2=1e-3)
            mu, sd, w = m
            p = 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(te.sum()), (F[te] - mu) / sd]) @ w, -30, 30)))
            res[name] = (A.logit_eval(m, F[te], y[te]), auc(y[te], p), p)
        base = A.logit_loss(np.zeros((te.sum(), 0)), y[te])
        print(f"\n[{o}] test rate {y[te].mean():.3f}")
        for name, (ll, au, p) in res.items():
            print(f"   {name:22} log-loss {ll:.4f} ({100 * (base - ll) / base:5.1f}% better than base rate)   AUC {au:.3f}")
        preds[o] = res

    # Gating: skip the verification when the prediction is confident; cost-weighted by nodes
    print("\nGating simulation on test events (node-weighted savings vs wrong decisions):")
    print("   skip and EXTEND when P(S) >= t  |  skip and treat as NEITHER when P(N) >= t")
    for o, action in (("S", "extend"), ("N", "neither")):
        for name in ("search features", "+ 128 network inputs"):
            p = preds[o][name][2]
            y = out[te] == o
            row = []
            for t in (0.7, 0.8, 0.9, 0.95):
                skip = p >= t
                saved = nodes[te][skip].sum() / nodes[te].sum()
                wrong = (skip & ~y).sum() / max(skip.sum(), 1)
                row.append(f"t={t}: save {100 * saved:4.1f}% of verif. nodes, {100 * wrong:4.1f}% wrong")
            print(f"   [{o}->{action}] {name:22} " + " | ".join(row))


if __name__ == "__main__":
    main()
