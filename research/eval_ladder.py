#!/usr/bin/env python3
"""Identity ladder: can a linear readout of the final NNUE layer's inputs predict the depth-16
search score better than the network's own static eval? Nested models, fitted on --train and
scored on unseen positions from --test, separately for game positions (root) and search nodes.

  rung 0  static eval unchanged ("multiply by identity")
  rung 1  a * static + b
  rung 2  rung 1 + up/down risk predictions (trained on --train)
  rung 3  rung 1 + all 128 inputs of fc_2 (ridge)
  rung 3c rung 3 fitted only on the same kind of position it is scored on
"""
import argparse

import numpy as np

import analyze as A
import variance_head as V


def fit_ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    b, w = V.ridge(Z, y, lam)
    return lambda Xn: b + ((Xn - mu) / sd) @ w


def choose_lam(X, y, groups, grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), folds=3):
    rng = np.random.default_rng(0)
    ug = np.unique(groups)
    f = dict(zip(ug, rng.integers(0, folds, len(ug))))
    fold = np.array([f[g] for g in groups])
    best = None
    for lam in grid:
        err = []
        for k in range(folds):
            tr, te = fold != k, fold == k
            p = fit_ridge(X[tr], y[tr], lam)(X[te])
            err.append(np.mean((y[te] - p) ** 2))
        if best is None or np.mean(err) < best[0]:
            best = (np.mean(err), lam)
    return best[1]


def metrics(y, p):
    e = y - p
    return np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e)), np.median(np.abs(e)), np.mean(np.abs(e) > 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--acts", required=True)
    a = ap.parse_args()

    acts = np.load(a.acts)
    tr = A.add_features(A.load([a.train], 1000))
    te = A.add_features(A.load([a.test], 1000))
    te = te[~te.fen.isin(set(tr.fen))]
    tr, Xtr, btr = V.attach(tr, acts)
    te, Xte, bte = V.attach(te, acts)

    # risk predictions (shared weights, as in the engine), trained on the training run only
    for k in ("up", "down"):
        y = V.TARGETS[k](tr.err_static.values)
        _, lp, lb = V.choose_lams(Xtr, btr, y, tr.book_idx.values, min_bucket=10 ** 9)
        h = V.VarianceHead(lp, lb, 10 ** 9).fit(Xtr, btr, y)
        tr[k] = h.predict(Xtr, btr)
        te[k] = h.predict(Xte, bte)

    def design(df, X, rung):
        s = df.static_cp.values[:, None]
        if rung == 1:
            return s
        if rung == 2:
            return np.column_stack([s, df.up.values, df.down.values,
                                    s[:, 0] * df.up.values, s[:, 0] * df.down.values])
        return np.column_stack([s, X])

    y_tr = tr.deep_cp.values
    preds = {"0 identity": {}, "1 a*static+b": {}, "2 + up/down": {}, "3 + 128 inputs": {}, "3c same-kind fit": {}}
    models = {}
    for rung, name in ((1, "1 a*static+b"), (2, "2 + up/down"), (3, "3 + 128 inputs")):
        D = design(tr, Xtr, rung)
        lam = choose_lam(D, y_tr, tr.book_idx.values) if rung == 3 else 1e-6
        models[name] = (rung, fit_ridge(D, y_tr, lam), lam)

    print(f"train {len(tr)} positions, test {len(te)} unseen positions; errors in cp vs depth-16 search\n")
    for src in ("root", "node"):
        m_te = (te.src == src).values
        m_tr = (tr.src == src).values
        y = te.deep_cp.values[m_te]
        rows = [("0 identity", te.static_cp.values[m_te])]
        for name, (rung, f, lam) in models.items():
            rows.append((name, f(design(te[m_te], Xte[m_te], rung))))
        D3 = design(tr[m_tr], Xtr[m_tr], 3)
        lam = choose_lam(D3, y_tr[m_tr], tr.book_idx.values[m_tr])
        f3c = fit_ridge(D3, y_tr[m_tr], lam)
        rows.append(("3c same-kind fit", f3c(design(te[m_te], Xte[m_te], 3))))
        base = metrics(y, rows[0][1])
        print(f"[{src}] n={m_te.sum()}")
        print(f"   {'model':20} {'RMSE':>7} {'MAE':>7} {'median':>7} {'>100cp':>7}   {'MAE vs identity':>15}")
        for name, p in rows:
            r = metrics(y, p)
            print(f"   {name:20} {r[0]:7.1f} {r[1]:7.1f} {r[2]:7.1f} {r[3]:7.3f}   {100 * (r[1] - base[1]) / base[1]:+14.1f}%")
        print()
    for name, (rung, f, lam) in models.items():
        if rung == 1:
            print(f"rung 1 fit (all training positions): deep = {f(np.array([[1.0]]))[0] - f(np.array([[0.0]]))[0]:.3f} * static "
                  f"+ {f(np.array([[0.0]]))[0]:.1f}")


if __name__ == "__main__":
    main()
