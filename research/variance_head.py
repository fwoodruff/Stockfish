#!/usr/bin/env python3
"""Route A: can a linear "variance head" on the frozen network's penultimate activations
predict eval error, and does it beat the head-disagreement proxy at the search decisions?

Train on one collect.py run, test on another (positions that also occur in the training run are
dropped from the test set). The head is ridge regression from the 128 inputs of the final layer
of the head in use: a model shared by all buckets, plus per-bucket corrections shrunk towards it
where a bucket has enough data. Targets (cp, deep = depth-16 search, static = scaled static eval):
  var  : log1p |deep - static|              (two-sided spread)
  up   : log1p max(deep - static, 0)        (upside surprise size)
  down : log1p max(static - deep, 0)        (downside surprise size)
"""

import argparse

import numpy as np
import pandas as pd

import analyze as A

TARGETS = {
    "var": lambda e: np.log1p(np.abs(e)),
    "up": lambda e: np.log1p(np.maximum(e, 0)),
    "down": lambda e: np.log1p(np.maximum(-e, 0)),
}


def ridge(X, y, lam):
    """Ridge with an unpenalised intercept; X already standardised."""
    n, d = X.shape
    Xc, yc = X - X.mean(0), y - y.mean()
    w = np.linalg.solve(Xc.T @ Xc + lam * n * np.eye(d), Xc.T @ yc)
    return y.mean() - X.mean(0) @ w, w


class VarianceHead:
    def __init__(self, lam_pool, lam_bucket, min_bucket):
        self.lam_pool, self.lam_bucket, self.min_bucket = lam_pool, lam_bucket, min_bucket

    def fit(self, X, b, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.b0, self.w0 = ridge(Z, y, self.lam_pool)
        r = y - (self.b0 + Z @ self.w0)
        self.per = {}
        for k in np.unique(b):
            m = b == k
            if m.sum() >= self.min_bucket:
                self.per[k] = ridge(Z[m], r[m], self.lam_bucket)
        return self

    def predict(self, X, b):
        Z = (X - self.mu) / self.sd
        p = self.b0 + Z @ self.w0
        for k, (bb, ww) in self.per.items():
            m = b == k
            p[m] += bb + Z[m] @ ww
        return p


def r2(y, p):
    return 1 - np.mean((y - p) ** 2) / np.var(y)


def choose_lams(X, b, y, groups, grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), min_bucket=1000, folds=3, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    fold_of = dict(zip(ug, rng.integers(0, folds, len(ug))))
    f = np.array([fold_of[g] for g in groups])
    best = None
    for lp in grid:
        for lb in grid:
            sc = []
            for k in range(folds):
                tr, te = f != k, f == k
                h = VarianceHead(lp, lb, min_bucket).fit(X[tr], b[tr], y[tr])
                sc.append(r2(y[te], h.predict(X[te], b[te])))
            s = np.mean(sc)
            if best is None or s > best[0]:
                best = (s, lp, lb)
    return best


def attach(df, acts):
    idx = {f: i for i, f in enumerate(acts["fens"])}
    rows = df.fen.map(idx)
    df = df[rows.notna()].copy()
    rows = rows[rows.notna()].astype(int).values
    return df, acts["acts"][rows].astype(float), acts["buckets"][rows].astype(int)


def decision(name, t, y, base_cols, proxies):
    Xb = np.column_stack(base_cols)
    print(f"  {name}: n={len(t)}, event rate {y.mean():.3f}, {t.book_idx.nunique()} searches")
    print(f"     {'proxy':12} {'event rate by proxy quintile (low -> high)':44} {'held-out gain':>13}  {'90% interval':>16}")
    for p in proxies:
        x = t[p].values
        ok = ~np.isnan(x)
        if ok.sum() < 300:
            continue
        rates = A.quintile_rates(x[ok], y[ok])
        # proxies can be negative (log-scale predictions); shift so A.tf's log1p is monotone
        xs = x[ok] - np.nanmin(x[ok])
        g, lo, hi = A.cluster_boot_gain(Xb[ok], xs, y[ok], t.book_idx.values[ok], reps=100)
        print(f"     {p:12} " + "".join(f"{r:8.3f}" for r in rates).ljust(44) + f" {100 * g:12.2f}%  [{100 * lo:5.2f}, {100 * hi:5.2f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--min-bucket", type=int, default=1000)
    a = ap.parse_args()

    acts = np.load(a.acts)
    tr = A.add_features(A.load([a.train], 1000))
    te = A.add_features(A.load([a.test], 1000))
    te = te[~te.fen.isin(set(tr.fen))]
    tr, Xtr, btr = attach(tr, acts)
    te, Xte, bte = attach(te, acts)
    print(f"train {len(tr)} positions ({a.train}), test {len(te)} unseen positions ({a.test})")

    heads = {}
    for k, f in TARGETS.items():
        ytr = f(tr.err_static.values)
        cv, lp, lb = choose_lams(Xtr, btr, ytr, tr.book_idx.values, min_bucket=a.min_bucket)
        h = VarianceHead(lp, lb, a.min_bucket).fit(Xtr, btr, ytr)
        heads[k] = h
        tr[f"{k}_head"] = h.predict(Xtr, btr)
        te[f"{k}_head"] = h.predict(Xte, bte)
        yte = f(te.err_static.values)
        print(f"  head '{k}': lambda pool {lp:g}, bucket {lb:g}, {len(h.per)} bucket corrections; "
              f"R^2 train {r2(ytr, tr[f'{k}_head'].values):.3f}, CV {cv:.3f}, test {r2(yte, te[f'{k}_head'].values):.3f}")

    print("\n=== Error size on the test run: R^2 for log1p|deep - static| (all OLS fits on the test run) ===")
    for src in ("root", "node"):
        s = te[te.src == src]
        y = np.log1p(s.err_static.abs().values)
        c = s.v0.abs()
        B = [c, c ** 2, s.pieces, s.pieces ** 2, c * s.pieces]
        rows = [("baseline (|static|, pieces)", B)]
        if src == "node":
            B = B + [A.tf(s.abs_corr_cp)]
            rows = [("baseline (+|correction|)", B)]
        rows += [("+ head_nb", B + [A.tf(s.head_nb)]),
                 ("+ var_head", B + [s.var_head]),
                 ("+ up_head + down_head", B + [s.up_head, s.down_head]),
                 ("+ heads + head_nb + captures", B + [s.up_head, s.down_head, A.tf(s.head_nb), A.tf(s.captures)])]
        print(f"  [{src}] n={len(s)}")
        for label, cols in rows:
            print(f"     {label:32} R^2 {A.ols_r2(np.column_stack(cols), y):.3f}")

    print("\n=== Decisions on the test run (held-out gain over baseline, cluster bootstrap) ===")
    proxies = ["head_nb", "own_hd", "captures", "var_head", "up_head", "down_head"]
    n = te[te.src == "node"].dropna(subset=["abs_corr_cp"]).copy()
    n["margin"] = n.s_static_cp - n.s_beta_cp

    t = n[(n.margin < 0) & (n.margin >= -150)]
    m = t.margin.abs().values
    decision("UPSIDE flips: static < beta, deep >= beta", t, (t.deep_cp >= t.s_beta_cp).values.astype(float),
             [m, np.sqrt(m), t.pieces, t.s_depth, t.s_improving, A.tf(t.abs_corr_cp)], proxies)
    decision("UPSIDE flips, baseline also has captures + depth-1 search", t, (t.deep_cp >= t.s_beta_cp).values.astype(float),
             [m, np.sqrt(m), t.pieces, t.s_depth, t.s_improving, A.tf(t.abs_corr_cp), A.tf(t.captures),
              (t.depth_cp.map(lambda d: d.get('1', np.nan)) - t.s_beta_cp).fillna(0).values], proxies)

    t = n[(n.s_pvnode == 0) & (n.s_ttpv == 0) & (n.s_depth < 19) & (n.margin >= 0) & (n.margin <= 150)]
    m = t.margin.values
    decision("FUTILITY (downside): non-PV, static >= beta, deep < beta", t, (t.deep_cp < t.s_beta_cp).values.astype(float),
             [m, np.sqrt(m), t.pieces, t.s_depth, t.s_improving, A.tf(t.abs_corr_cp)], proxies)

    r = te[(te.src == "root") & (te.v0 >= -150) & (te.v0 <= -30)]
    decision("ROOT fat tail: static in [-150,-30], deep >= 0", r, (r.deep_cp >= 0).values.astype(float),
             [r.v0.values], ["head_nb", "captures", "var_head", "up_head", "down_head"])


if __name__ == "__main__":
    main()
