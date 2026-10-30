#!/usr/bin/env python3
"""Does any cheap signal predict how wrong a quick assessment is, beyond what Stockfish already knows?

Two targets ("how wrong"):
  A. static error   = deep - static eval        (what pruning margins have to absorb)
  B. shallow error  = deep - depth-4 search     (what a quick search misses: the "fat tail")

Three questions for each, from general to search-relevant:
  1. Error size: does proxy X predict |error| after controlling for |assessment| and material?
  2. Fat tails: among slightly-worse (or slightly-better) positions, does X pick out the ones
     whose deep score crosses zero?
  3. Pruning flips (node samples, target A only): does X predict when sign(static - beta)
     disagrees with sign(deep - beta), beyond the margin |static - beta| and |correction history|?

Effect sizes are the gain in R^2 / reduction in log-loss over the baseline model, and event rates
by proxy quintile. Proxies are tagged by cost: static (free or nearly), search (needs a depth<=4
search), insearch (already available inside Stockfish's search). Units are centipawns, side to move.
"""

import argparse
import json

import numpy as np
import pandas as pd

CORR_TO_INTERNAL = 1 / 131072  # see to_corrected_static_eval


def load(paths, max_abs):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # file still being written
                    continue
                if r.get("meta") or r.get("deep_mate"):
                    continue
                rows.append(r)
    df = pd.DataFrame(rows)
    return df[df.static_cp.abs() <= max_abs].copy()


def add_features(df):
    d = lambda k: df.depth_cp.map(lambda m: m.get(str(k), np.nan)).astype(float)
    d1, d2, d3, d4 = d(1), d(2), d(3), d(4)

    df["v0"] = df.static_cp
    df["d4"] = d4
    df["err_static"] = df.deep_cp - df.v0
    df["err_shallow"] = df.deep_cp - d4

    # static
    df["mat_disagree"] = (df.nnue_cp - df.simple_cp).abs()

    def head_feats(r):
        h, pc = r.heads_cp, r.bucket // 4
        nb = [abs(h[j] - h[pc]) for j in (pc - 1, pc + 1) if 1 <= j < len(h)]
        return pd.Series({"head_nb": np.mean(nb) if nb else np.nan,
                          "head_spread": np.std(h[max(1, pc - 2): pc + 3])})

    df[["head_nb", "head_spread"]] = df.apply(head_feats, axis=1)
    for k in ("legal", "captures", "checks"):
        df[k] = df[k].astype(float)

    # search (depth <= 4 only, so they are legitimately cheaper than the target)
    df["d1_jump"] = (d1 - df.v0).abs()
    df["d4_jump"] = (d4 - df.v0).abs()
    df["id_instab"] = pd.concat([d1, d2, d3, d4], axis=1).std(axis=1)

    def bm_changes(m):
        mv = [x for x in (m.get(str(k)) for k in range(1, 5)) if x]
        return float(sum(a != b for a, b in zip(mv, mv[1:])))

    df["bm_changes_1_4"] = df.depth_move.map(bm_changes)
    df["log_nodes_d4"] = np.log(df.depth_nodes.map(lambda m: m.get("4", np.nan)).astype(float))

    # insearch
    if "s_corr" in df:
        df["abs_corr_cp"] = df.s_corr.abs() * CORR_TO_INTERNAL * 100 / df.a
    # insearch, single nearest neighbouring head as the engine computes it (HeadExtra), for this
    # node and for its parent (the "one ply up" variant)
    for k, col in (("own_hd", "s_headdis_cp"), ("parent_hd", "s_parentheaddis_cp")):
        df[k] = df[col] if col in df else np.nan
    return df


STATIC = ["mat_disagree", "head_nb", "head_spread", "legal", "captures", "checks"]
SEARCH_A = ["d1_jump", "d4_jump", "id_instab", "bm_changes_1_4", "log_nodes_d4"]
SEARCH_B = ["d1_jump", "d4_jump", "id_instab", "bm_changes_1_4", "log_nodes_d4"]
HEADS = ["own_hd", "parent_hd"]
COST = {**{p: "static" for p in STATIC}, **{p: "search" for p in SEARCH_A}, "abs_corr_cp": "insearch",
        "own_hd": "insearch", "parent_hd": "insearch"}


# ---------- small numpy models ----------

def ols_r2(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - (y - X @ beta).var() / y.var()


def logit_loss(X, y, iters=50, l2=1e-4):
    X = np.column_stack([np.ones(len(y)), X])
    mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-9
    X[:, 1:] = (X[:, 1:] - mu) / sd
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) / len(y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X / len(y) + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    p = np.clip(1 / (1 + np.exp(-X @ w)), 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def rank(x):
    return pd.Series(x).rank().values


def partial_rank_corr(x, y, Z):
    Z = np.column_stack([np.ones(len(x)), Z])
    rx, ry = rank(x), rank(y)
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def tf(x):
    return np.log1p(np.maximum(x, 0))


def quintile_rates(x, event):
    q = pd.qcut(pd.Series(x).rank(method="first"), 5, labels=False).values
    return [event[q == i].mean() for i in range(5)]


def complete(df, cols):
    return df.dropna(subset=cols)


# ---------- reports ----------

def q1_error_size(df, center, err, proxies, extra_base=()):
    s = complete(df, [center, err, *proxies, *extra_base])
    c = s[center].abs()
    B = np.column_stack([c, c ** 2, s.pieces, s.pieces ** 2, c * s.pieces, *[tf(s[e]) for e in extra_base]])
    y = tf(s[err].abs()).values
    r2b = ols_r2(B, y)
    base_desc = ", ".join([f"|{center}|", "pieces", *extra_base])
    print(f"  target |{err}|, baseline ({base_desc}) R^2 = {r2b:.3f}, n={len(s)}")
    print(f"    {'proxy':16} {'cost':9} {'rank corr':>9} {'partial':>8} {'dR^2':>7}")
    for p in proxies:
        x = s[p].values
        rho = np.corrcoef(rank(x), rank(np.abs(s[err].values)))[0, 1]
        part = partial_rank_corr(x, y, B)
        dr2 = ols_r2(np.column_stack([B, tf(x)]), y) - r2b
        print(f"    {p:16} {COST.get(p, ''):9} {rho:9.3f} {part:8.3f} {dr2:7.3f}")
    for label, group in (("ALL static", [p for p in proxies if COST.get(p) == "static"]),
                         ("ALL", proxies)):
        if group:
            dr2 = ols_r2(np.column_stack([B, *[tf(s[p]) for p in group]]), y) - r2b
            print(f"    {label:16} {'':9} {'':9} {'':8} {dr2:7.3f}")


def q2_tails(df, center, proxies, lo, hi):
    s0 = complete(df, [center, *proxies])
    print(f"  assessment = {center}; behind band [{-hi}, {-lo}] event: deep >= 0 | ahead band [{lo}, {hi}] event: deep <= 0")
    for name, band, ev in (("behind", lambda s: (s[center] >= -hi) & (s[center] <= -lo), lambda s: s.deep_cp >= 0),
                           ("ahead", lambda s: (s[center] >= lo) & (s[center] <= hi), lambda s: s.deep_cp <= 0)):
        s = s0[band(s0)]
        y = ev(s).values.astype(float)
        if len(s) < 200 or y.sum() < 10:
            print(f"    [{name}] n={len(s)}, events={int(y.sum())}: too few to say anything")
            continue
        X0 = np.column_stack([s[center]])
        base = logit_loss(X0, y)
        print(f"    [{name}] n={len(s)}, event rate {y.mean():.3f}")
        print(f"      {'proxy':16} {'event rate by proxy quintile (low -> high)':44} {'logloss gain':>12}")
        rows = [(p, [p]) for p in proxies] + [("ALL static", [p for p in proxies if COST.get(p) == "static"]),
                                               ("ALL", proxies)]
        for label, group in rows:
            if not group:
                continue
            X = np.column_stack([X0, *[tf(s[p]) for p in group]])
            gain = (base - logit_loss(X, y)) / base
            rates = quintile_rates(s[group[0]].values, y) if len(group) == 1 else []
            print(f"      {label:16} " + "".join(f"{r:8.3f}" for r in rates).ljust(44) + f" {100 * gain:9.1f}%")


def q3_flips(df, proxies, max_margin):
    s = complete(df[df.src == "node"], ["abs_corr_cp", *proxies]).copy()
    s["margin"] = s.s_static_cp - s.s_beta_cp
    s = s[s.margin.abs() <= max_margin]
    y = (np.sign(s.margin) != np.sign(s.deep_cp - s.s_beta_cp)).values.astype(float)
    m = s.margin.abs().values
    no_corr = np.column_stack([m, np.sqrt(m), s.pieces])
    base_X = np.column_stack([no_corr, tf(s.abs_corr_cp)])
    base = logit_loss(base_X, y)
    print(f"  node samples with |static - beta| <= {max_margin}: n={len(s)}, flip rate {y.mean():.3f}")
    print(f"  |correction| alone, over margin+pieces: "
          f"{100 * (logit_loss(no_corr, y) - base) / logit_loss(no_corr, y):.1f}% log-loss gain; "
          f"flip rate by |corr| quintile: " + " ".join(f"{r:.3f}" for r in quintile_rates(s.abs_corr_cp.values, y)))
    print(f"  baseline = margin + pieces + |correction|, log-loss {base:.4f}")
    print(f"      {'proxy':16} {'flip rate by proxy quintile (low -> high)':44} {'logloss gain':>12}")
    rows = [(p, [p]) for p in proxies] + [("ALL static", [p for p in proxies if COST.get(p) == "static"]),
                                           ("ALL", proxies)]
    for label, group in rows:
        X = np.column_stack([base_X, *[tf(s[p]) for p in group]])
        gain = (base - logit_loss(X, y)) / base
        rates = quintile_rates(s[group[0]].values, y) if len(group) == 1 else []
        print(f"      {label:16} " + "".join(f"{r:8.3f}" for r in rates).ljust(44) + f" {100 * gain:9.1f}%")


def logit_fit(X, y, iters=50, l2=1e-4):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(y)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) / len(y) + l2 * w
        H = (Z * (p * (1 - p))[:, None]).T @ Z / len(y) + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return mu, sd, w


def logit_eval(model, X, y):
    mu, sd, w = model
    Z = np.column_stack([np.ones(len(y)), (X - mu) / sd])
    p = np.clip(1 / (1 + np.exp(-np.clip(Z @ w, -30, 30))), 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def cluster_boot_gain(Xb, x, y, groups, reps=100, seed=0):
    """Out-of-sample relative log-loss gain of adding tf(x) to Xb.

    Each rep fits both models on a bootstrap sample of whole searches and scores them on the
    searches left out, so a useless proxy can come out negative. Returns median and 90% interval.
    """
    Xf = np.column_stack([Xb, tf(x)])
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(ug))]
    gains = []
    for _ in range(reps):
        pick = rng.integers(0, len(ug), len(ug))
        oob = np.setdiff1d(np.arange(len(ug)), pick)
        tr = np.concatenate([members[g] for g in pick])
        te = np.concatenate([members[g] for g in oob])
        if y[tr].min() == y[tr].max() or y[te].min() == y[te].max():
            continue
        b = logit_eval(logit_fit(Xb[tr], y[tr]), Xb[te], y[te])
        f = logit_eval(logit_fit(Xf[tr], y[tr]), Xf[te], y[te])
        gains.append((b - f) / b)
    lo, med, hi = np.percentile(gains, [5, 50, 95])
    return med, lo, hi


def q3_with_ci(df, proxies, max_margin):
    s = complete(df[df.src == "node"], ["abs_corr_cp"]).copy()
    s["margin"] = s.s_static_cp - s.s_beta_cp
    s = s[s.margin.abs() <= max_margin]
    y = (np.sign(s.margin) != np.sign(s.deep_cp - s.s_beta_cp)).values.astype(float)
    m = s.margin.abs().values
    Xb = np.column_stack([m, np.sqrt(m), s.pieces, tf(s.abs_corr_cp)])
    print(f"  all node samples, |static - beta| <= {max_margin}: n={len(s)}, flip rate {y.mean():.3f}, "
          f"{s.book_idx.nunique()} searches")
    print(f"      {'proxy':16} {'logloss gain':>12}  {'90% interval, held-out searches':>32}")
    for p in proxies:
        ok = ~np.isnan(s[p].values)
        if ok.sum() < 300:
            continue
        g, lo, hi = cluster_boot_gain(Xb[ok], s[p].values[ok], y[ok], s.book_idx.values[ok])
        print(f"      {p:16} {100 * g:11.2f}%  [{100 * lo:6.2f}%, {100 * hi:6.2f}%]  n={ok.sum()}")


def q4_futility(df, proxies, max_margin):
    """The exact decision futility pruning makes: non-PV node, static >= beta, prune if the margin
    is big enough. Event = pruning would be wrong (deep < beta)."""
    s = complete(df[df.src == "node"], ["abs_corr_cp"]).copy()
    s["margin"] = s.s_static_cp - s.s_beta_cp
    s = s[(s.s_pvnode == 0) & (s.s_ttpv == 0) & (s.s_depth < 19) & (s.margin >= 0) & (s.margin <= max_margin)]
    y = (s.deep_cp < s.s_beta_cp).values.astype(float)
    m = s.margin.values
    Xb = np.column_stack([m, np.sqrt(m), s.pieces, s.s_depth, s.s_improving, tf(s.abs_corr_cp)])
    print(f"  non-PV, static >= beta, margin <= {max_margin}: n={len(s)}, "
          f"'pruning would be wrong' rate {y.mean():.3f}, {s.book_idx.nunique()} searches")
    print(f"  baseline = margin + pieces + depth + improving + |correction|")
    print(f"      {'proxy':16} {'wrong-prune rate by proxy quintile (low -> high)':48} {'gain':>7}  {'90% held-out':>18}")
    for p in proxies:
        ok = ~np.isnan(s[p].values)
        if ok.sum() < 300:
            continue
        rates = quintile_rates(s[p].values[ok], y[ok])
        g, lo, hi = cluster_boot_gain(Xb[ok], s[p].values[ok], y[ok], s.book_idx.values[ok])
        print(f"      {p:16} " + "".join(f"{r:8.3f}" for r in rates).ljust(48)
              + f" {100 * g:6.2f}%  [{100 * lo:5.2f}%, {100 * hi:5.2f}%]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--max-abs", type=float, default=1000, help="drop |static| above this (cp)")
    ap.add_argument("--band", type=float, nargs=2, default=[30, 150], help="tail band |assessment| range (cp)")
    ap.add_argument("--max-margin", type=float, default=150, help="|static - beta| window for flips (cp)")
    a = ap.parse_args()

    df = add_features(load(a.files, a.max_abs))
    print(f"records: {len(df)} {df.src.value_counts().to_dict()}")
    for src in ("root", "node"):
        s = df[df.src == src]
        if len(s) < 300:
            continue
        print(f"\n[{src}] |static err| p50/p90/p99 cp: {np.percentile(s.err_static.abs(), [50, 90, 99]).round().tolist()}"
              f"   |shallow err|: {np.percentile(s.err_shallow.dropna().abs(), [50, 90, 99]).round().tolist()}")
        ins = (["abs_corr_cp"] + [h for h in HEADS if s[h].notna().sum() > 300]) if src == "node" else []
        print(f"\n=== [{src}] A. static error ===")
        q1_error_size(s, "v0", "err_static", STATIC + ins)
        print()
        q2_tails(s, "v0", STATIC + ins, *a.band)
        print(f"\n=== [{src}] B. shallow-search error ===")
        q1_error_size(s, "d4", "err_shallow", STATIC + SEARCH_B + ins)
        print()
        q2_tails(s, "d4", STATIC + SEARCH_B + ins, *a.band)

    if (df.src == "node").sum() >= 300:
        heads = [h for h in HEADS if df[h].notna().sum() > 300]
        print("\n=== [node] Q3 pruning flips (static vs beta) ===")
        q3_flips(df, STATIC, a.max_margin)
        print("\n=== [node] Q3 with cluster-bootstrap CIs ===")
        q3_with_ci(df, ["mat_disagree", "head_nb", "captures", *heads], a.max_margin)
        print("\n=== [node] Q4 futility decision (would pruning be wrong?) ===")
        q4_futility(df, ["mat_disagree", "head_nb", "captures", *heads], a.max_margin)


if __name__ == "__main__":
    main()
