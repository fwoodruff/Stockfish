#!/usr/bin/env python3
"""Screen four candidate network signals: can a linear readout of the final NNUE layer's inputs
predict what search measures, as well as the cheap heuristic Stockfish uses today?

  1 plausible moves : number of moves within --near cp of the best at depth --mpv-depth
  2 forward improving: static eval two plies down the depth --pv-depth main line > static now
  3 mini policy     : from/to squares of the depth-16 best move (trained on --train, tested here)
  4 null-move safety: null-move search (side to move swapped) overstates the real search by > 50 cp

Collect on a sample of --test positions (which already have network inputs and depth-16 moves),
then --analyze. Uses the research build (features, acts).
"""
import argparse
import json
import random
import sys
from multiprocessing import Pool

import numpy as np

import analyze as A
import variance_head as V
from collect import DEFAULT_ENGINE, Engine

PIECE_VALUE = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}


def sq_index(s):
    return (ord(s[0]) - 97) + 8 * (int(s[1]) - 1)


def board_of(fen):
    rows = fen.split()[0].split("/")
    b = {}
    for r, row in enumerate(rows):
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                b[chr(97 + f) + str(8 - r)] = ch
                f += 1
    return b


def search_lines(eng, fen, depth, multipv):
    eng.send(f"setoption name MultiPV value {multipv}")
    eng.send(f"ucinewgame\nposition fen {fen}\ngo depth {depth}")
    lines = {}
    for l in eng.lines_until("bestmove"):
        if l.startswith(f"info depth {depth} ") and " pv " in l and "bound" not in l:
            t = l.split()
            k = int(t[t.index("multipv") + 1]) if "multipv" in t else 1
            i = t.index("score")
            v = int(t[i + 2]) if t[i + 1] == "cp" else (10000 if int(t[i + 2]) > 0 else -10000)
            lines[k] = (v, t[t.index("pv") + 1:])
    eng.send("setoption name MultiPV value 1")
    return lines


def task(args):
    rec, cfg = args
    fen = rec["fen"]
    e = Engine(cfg["engine"])
    try:
        e.new_position(fen)
        f = e.features()
        if f.get("incheck"):
            return None
        out = {"fen": fen, "book_idx": rec["book_idx"], "a": f["a"], "static": f["scaled"],
               "legal": int(f["legal"]), "captures": int(f["captures"]), "pieces": int(f["pieces"]),
               "checks": int(f["checks"]), "improving": rec.get("s_improving"), "depth": rec.get("s_depth"),
               "deep_best": rec["depth_move"].get("16")}
        # 1: plausible moves
        mp = search_lines(e, fen, cfg["mpv_depth"], 256)
        if not mp:
            return None
        best = max(v for v, _ in mp.values())
        out["near"] = sum(v >= best - cfg["near"] for v, _ in mp.values())  # UCI scores are cp
        out["mpv_best"] = best
        out["legal_list"] = [pv[0] for _, pv in mp.values()]
        # 2: forward improving along the main line
        pvl = search_lines(e, fen, cfg["pv_depth"], 1)
        if 1 in pvl and len(pvl[1][1]) >= 2:
            e.new_position(fen, pvl[1][1][:2])
            f2 = e.features()
            out["static_plus2"] = None if f2.get("incheck") else f2["scaled"]
        # 4: null move (swap side to move, clear en passant)
        t = fen.split()
        t[1] = "b" if t[1] == "w" else "w"
        t[3] = "-"
        nfen = " ".join(t)
        e.new_position(nfen)
        fn = e.features()
        if not fn.get("incheck"):
            nl = search_lines(e, nfen, cfg["mpv_depth"] - 2, 1)
            if 1 in nl:
                out["null_value"] = -nl[1][0]  # from the original side to move
        return out
    finally:
        e.close()


def collect(a):
    recs = [json.loads(l) for l in open(a.test)]
    recs = [r for r in recs if not r.get("meta") and r.get("src") == "node" and "16" in r.get("depth_move", {})]
    random.Random(5).shuffle(recs)
    recs = recs[:a.n]
    cfg = {"engine": a.engine, "mpv_depth": a.mpv_depth, "pv_depth": a.pv_depth, "near": a.near}
    kept = 0
    with open(a.out, "w") as fo, Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(task, [(r, cfg) for r in recs]), 1):
            if r:
                fo.write(json.dumps(r) + "\n"); kept += 1
            if k % 500 == 0:
                print(f"{k}/{len(recs)}, kept {kept}", file=sys.stderr, flush=True)


def fit_eval_binary(Btr, ytr, Bte, yte):
    m = A.logit_fit(Btr, ytr, l2=1e-3)
    mu, sd, w = m
    p = 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(len(Bte)), (Bte - mu) / sd]) @ w, -30, 30)))
    order = np.argsort(p); r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    n1 = yte.sum(); n0 = len(yte) - n1
    return (r[yte == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def r2_ridge(Ftr, ytr, Fte, yte):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    b, w = V.ridge((Ftr - mu) / sd, ytr, 1e-3)
    pred = b + ((Fte - mu) / sd) @ w
    return 1 - np.mean((yte - pred) ** 2) / np.var(yte)


def softmax_fit(X, y, k=64, iters=300, lr=0.5, l2=1e-4):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    W = np.zeros((Z.shape[1], k))
    Y = np.eye(k)[y]
    for _ in range(iters):
        S = Z @ W; S -= S.max(1, keepdims=True)
        P = np.exp(S); P /= P.sum(1, keepdims=True)
        W -= lr * (Z.T @ (P - Y) / len(y) + l2 * W)
    return mu, sd, W


def softmax_logp(m, X):
    mu, sd, W = m
    S = np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ W
    S -= S.max(1, keepdims=True)
    return S - np.log(np.exp(S).sum(1, keepdims=True))


def analyze(a):
    acts = np.load(a.acts)
    idx = {f: i for i, f in enumerate(acts["fens"])}
    rs = [json.loads(l) for l in open(a.out)]
    rs = [r for r in rs if r["fen"] in idx]
    X = acts["acts"][[idx[r["fen"]] for r in rs]].astype(float)
    g = np.array([r["book_idx"] for r in rs]); ug = np.unique(g)
    trg = set(np.random.default_rng(0).choice(ug, int(0.6 * len(ug)), replace=False))
    tr = np.array([x in trg for x in g]); te = ~tr
    cp = lambda r, v: 100 * v / r["a"]
    print(f"{len(rs)} positions (train {tr.sum()}, test {te.sum()}), split by search\n")

    # 1 plausible moves
    y = np.log(np.array([r["near"] for r in rs], float))
    base = np.array([[np.log(r["legal"]), np.log1p(r["captures"]), r["pieces"], abs(cp(r, r["static"])),
                      r["depth"] or 0, r["improving"] or 0] for r in rs], float)
    print("1 plausible moves (log count within near cp of best):  mean count "
          f"{np.mean([r['near'] for r in rs]):.2f}")
    for n, F in (("baseline: legal, captures, pieces, |eval|, depth, improving", base), ("network readout", X),
                 ("both", np.column_stack([base, X]))):
        print(f"   R^2 {r2_ridge(F[tr], y[tr], F[te], y[te]):6.3f}  {n}")

    # 2 forward improving
    m = np.array(["static_plus2" in r and r["static_plus2"] is not None for r in rs])
    y2 = np.array([(r.get("static_plus2") or 0) > r["static"] for r in rs], float)
    imp = np.array([r["improving"] or 0 for r in rs], float)
    print(f"\n2 forward improving (eval two plies down the main line is higher): rate {y2[m].mean():.3f}")
    for n, F in (("Stockfish's improving flag (backward)", imp[:, None]),
                 ("baseline: flag + |eval| + pieces", np.column_stack([imp, base[:, 2:4]])),
                 ("network readout", X), ("both", np.column_stack([imp, base[:, 2:4], X]))):
        print(f"   AUC {fit_eval_binary(F[tr & m], y2[tr & m], F[te & m], y2[te & m]):.3f}  {n}")

    # 4 null-move safety
    m4 = np.array(["null_value" in r for r in rs])
    over = np.array([r.get("null_value", 0) - r["mpv_best"] for r in rs], float)  # both in cp
    y4 = (over > 50).astype(float)
    npm = np.array([[r["pieces"], np.log(r["legal"]), np.log1p(r["captures"]), r["checks"]] for r in rs], float)
    print(f"\n4 null move overstates the real search by > 50 cp: rate {y4[m4].mean():.3f}")
    for n, F in (("baseline: pieces, mobility, captures, checks", npm), ("network readout", X),
                 ("both", np.column_stack([npm, X]))):
        print(f"   AUC {fit_eval_binary(F[tr & m4], y4[tr & m4], F[te & m4], y4[te & m4]):.3f}  {n}")

    # 3 mini policy: train from/to squares on the big training run, test on these positions' legal moves
    trn = [json.loads(l) for l in open(a.train)]
    trn = [r for r in trn if not r.get("meta") and "16" in r.get("depth_move", {}) and r["fen"] in idx]
    Xt = acts["acts"][[idx[r["fen"]] for r in trn]].astype(float)
    mv = [r["depth_move"]["16"] for r in trn]
    mfrom = softmax_fit(Xt, np.array([sq_index(x[:2]) for x in mv]))
    mto = softmax_fit(Xt, np.array([sq_index(x[2:4]) for x in mv]))
    lf, lt = softmax_logp(mfrom, X), softmax_logp(mto, X)
    top1 = {"network from+to readout": 0, "captures first by victim value": 0, "random": 0}
    top3 = dict.fromkeys(top1, 0); n = 0
    rng = random.Random(1)
    for i, r in enumerate(rs):
        moves = r["legal_list"]
        if r["deep_best"] not in moves or len(moves) < 2:
            continue
        n += 1
        b = board_of(r["fen"])
        sc = {"network from+to readout": {mv_: lf[i, sq_index(mv_[:2])] + lt[i, sq_index(mv_[2:4])] for mv_ in moves},
              "captures first by victim value": {mv_: PIECE_VALUE.get(b.get(mv_[2:4], "").lower(), -1) + rng.random() * 0.1
                                                 for mv_ in moves},
              "random": {mv_: rng.random() for mv_ in moves}}
        for k, s in sc.items():
            ranked = sorted(moves, key=lambda x: -s[x])
            top1[k] += ranked[0] == r["deep_best"]
            top3[k] += r["deep_best"] in ranked[:3]
    print(f"\n3 mini policy: depth-16 best move among legal moves (n={n}, mean legal moves "
          f"{np.mean([len(r['legal_list']) for r in rs]):.1f})")
    for k in top1:
        print(f"   top-1 {top1[k] / n:6.1%}   top-3 {top3[k] / n:6.1%}   {k}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--test", default="data/run2_parent_d16.jsonl")
    ap.add_argument("--train", default="data/run1_warm_d16.jsonl")
    ap.add_argument("--acts", default="data/acts_run1_run2.npz")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--mpv-depth", type=int, default=8)
    ap.add_argument("--pv-depth", type=int, default=12)
    ap.add_argument("--near", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    (analyze if a.analyze else collect)(a)


if __name__ == "__main__":
    main()
