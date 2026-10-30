#!/usr/bin/env python3
"""Simulate Stockfish's futility pruning rule (step 9) on sampled nodes, with and without the risk
term, and count prunes and wrong prunes (deep search ends below beta).

The risk head is trained on --train only and applied to --test. Internal units throughout for the
margin; the deep result is compared with beta in centipawns of that position.
Approximation: opponentWorsening is not recorded, so it is taken as 0 (slightly larger margins).
"""
import argparse

import numpy as np

import analyze as A
import variance_head as V


def main():
    ap = argparse.ArgumentParser()
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

    y = V.TARGETS["down"](tr.err_static.values)
    _, lp, lb = V.choose_lams(Xtr, btr, y, tr.book_idx.values)
    h = V.VarianceHead(lp, lb, 1000).fit(Xtr, btr, y)
    ptr = h.predict(Xtr, btr)
    te["risk"] = np.round((h.predict(Xte, bte) - ptr.mean()) / ptr.std() * 64)

    n = te[(te.src == "node") & (te.s_ttpv == 0) & (te.s_depth < 19)].dropna(subset=["s_corr"]).copy()
    d = n.s_depth.values
    mult = np.minimum(45 + d * 4, 85) - 20 * (1 - n.s_tthit.values)
    base_margin = mult * d - (2789 * n.s_improving.values) * mult // 1024 + np.abs(n.s_corr.values) // 198435
    static = n.s_static.values
    beta = n.s_beta.values
    wrong = (n.deep_cp.values < n.s_beta_cp.values)
    risk = np.clip(n.risk.values, -128, 256)
    # In the engine, risk is 0 at TT hits (no fresh evaluation)
    risk_eff = np.where(n.s_tthit.values == 1, 0, risk)
    cand = static >= beta
    print(f"test nodes (non-ttPv, depth<19): {len(n)}, with static >= beta: {cand.sum()}, "
          f"fresh-eval share among those {1 - n.s_tthit.values[cand].mean():.2f}")

    def rule(extra, offset=0):
        prune = cand & (static - (base_margin + extra + offset) >= beta)
        return prune.sum(), (prune & wrong).sum()

    p0, w0 = rule(0)
    print(f"\nmaster rule:              prunes {p0:6d}  wrong {w0:5d}  wrong rate {w0 / p0:.4f}")
    print(f"\n{'risk scale':>10} {'prunes':>8} {'wrong':>6} {'rate':>7} | matched to master's prune count: {'offset':>7} {'prunes':>7} {'wrong':>6} {'vs master':>9}")
    for k in (0.5, 2.0, 4.0, 8.0, 16.0):
        extra = (risk_eff * k).astype(int)
        p, w = rule(extra)
        # shift a constant offset so the number of prunes matches master; compare wrong prunes
        lo, hi = -400, 400
        for _ in range(40):
            mid = (lo + hi) / 2
            if rule(extra, mid)[0] > p0:
                lo = mid
            else:
                hi = mid
        pm, wm = rule(extra, hi)
        print(f"{k:10.2f} {p:8d} {w:6d} {w / p:7.4f} | {'':32} {hi:7.1f} {pm:7d} {wm:6d} {100 * (wm - w0) / w0:+8.1f}%")

    # Bootstrap over searches for the k=0.5 matched comparison
    rng = np.random.default_rng(0)
    g = n.book_idx.values
    ug, inv = np.unique(g, return_inverse=True)
    extra = (risk_eff * 0.5).astype(int)
    diffs = []
    for _ in range(300):
        pick = rng.integers(0, len(ug), len(ug))
        wts = np.bincount(pick, minlength=len(ug))[inv]
        pr0 = cand & (static - base_margin >= beta)
        P0, W0 = (pr0 * wts).sum(), (pr0 & wrong).astype(int) @ wts
        lo, hi = -400, 400
        for _ in range(30):
            mid = (lo + hi) / 2
            pr = cand & (static - (base_margin + extra + mid) >= beta)
            if (pr * wts).sum() > P0:
                lo = mid
            else:
                hi = mid
        pr = cand & (static - (base_margin + extra + hi) >= beta)
        diffs.append(((pr & wrong).astype(int) @ wts - W0) / W0)
    lo5, med, hi95 = np.percentile(diffs, [5, 50, 95])
    print(f"\nk=0.5 (the engine's /2), matched prune count: change in wrong prunes {100 * med:+.1f}% "
          f"[90% interval {100 * lo5:+.1f}%, {100 * hi95:+.1f}%] (bootstrap over searches)")


if __name__ == "__main__":
    main()
