#!/usr/bin/env python3
"""Can a linear readout of the final NNUE layer's inputs predict quietness (qsearch value == static
eval) at positions where search enters qsearch, better than cheap counts Stockfish could compute?

Split by search (book position). Targets: not quiet (gain > t cp) for several t; log1p(gain).
Also a precision table: if we call the top f% most-confidently-quiet positions "quiet", how often
are they really quiet, and how often does qsearch gain a lot there (a costly miss)?
"""
import argparse
import json

import numpy as np

import analyze as A


def auc(y, p):
    order = np.argsort(p)
    r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def predict(model, X):
    mu, sd, w = model
    z = np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ w
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--acts", required=True)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.data)]
    acts = np.load(a.acts)
    idx = {f: i for i, f in enumerate(acts["fens"])}
    recs = [r for r in recs if r["fen"] in idx and r["oppgoodcaptures"] >= 0]
    X = acts["acts"][[idx[r["fen"]] for r in recs]].astype(float)
    C = np.array([[np.log1p(r[k]) for k in ("captures", "goodcaptures", "oppgoodcaptures", "checks", "legal", "pieces")]
                  + [float(r["goodcaptures"] > 0)] for r in recs])
    gain = np.array([r["gain_cp"] for r in recs])
    g = np.array([r["book_idx"] for r in recs])
    ug = np.unique(g)
    tr_g = set(np.random.default_rng(0).choice(ug, int(0.7 * len(ug)), replace=False))
    tr = np.array([x in tr_g for x in g]); te = ~tr
    print(f"{len(recs)} qsearch-entry positions from {len(ug)} searches (train {tr.sum()}, test {te.sum()})")
    print(f"quiet (gain = 0): {np.mean(gain == 0):.3f}; gain > 20 cp: {np.mean(gain > 20):.3f}; "
          f"gain > 100 cp: {np.mean(gain > 100):.3f}")

    feats = {"cheap counts": C, "network readout": X, "both": np.column_stack([C, X])}
    print("\nPredicting 'not quiet' (qsearch gains more than t cp), test set:")
    print(f"   {'t':>5} {'rate':>6}  " + "  ".join(f"{n:>24}" for n in feats))
    probs = {}
    for t in (0, 20, 100):
        y = (gain > t).astype(float)
        row = []
        for n, F in feats.items():
            m = A.logit_fit(F[tr], y[tr], l2=1e-3)
            p = predict(m, F[te])
            if t == 0:
                probs[n] = p
            base = A.logit_loss(np.zeros((te.sum(), 0)), y[te])
            ll = A.logit_eval(m, F[te], y[te])
            row.append(f"AUC {auc(y[te], p):.3f}, LL -{100 * (base - ll) / base:4.1f}%")
        print(f"   {t:5d} {y[te].mean():6.3f}  " + "  ".join(f"{s:>24}" for s in row))

    # simple rule many engines use: no winning capture available -> quiet
    rule = np.array([r["goodcaptures"] == 0 for r in recs])[te]
    gt = gain[te]
    print(f"\nRule 'no winning capture (SEE > 0) available': flags {rule.mean():.1%} as quiet; "
          f"of those {np.mean(gt[rule] == 0):.1%} really quiet, {np.mean(gt[rule] > 100):.2%} gain > 100 cp")

    print("\nCall the most-confidently-quiet f% 'quiet' (test set):")
    print(f"   {'model':16} " + "  ".join(f"{'f=' + str(f) + '%':>30}" for f in (20, 40, 60)))
    for n, p in probs.items():
        cells = []
        for f in (20, 40, 60):
            sel = p <= np.quantile(p, f / 100)
            cells.append(f"quiet {np.mean(gt[sel] == 0):5.1%}, >100cp {np.mean(gt[sel] > 100):5.2%}")
        print(f"   {n:16} " + "  ".join(f"{c:>30}" for c in cells))

    y = np.log1p(gain)
    print("\nSize of the gain, R^2 of log1p(gain) on the test set (ridge):")
    import variance_head as V
    for n, F in feats.items():
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        b, w = V.ridge((F[tr] - mu) / sd, y[tr], 1e-3)
        pred = b + ((F[te] - mu) / sd) @ w
        print(f"   {n:16} {1 - np.mean((y[te] - pred) ** 2) / np.var(y[te]):.3f}")


if __name__ == "__main__":
    main()
