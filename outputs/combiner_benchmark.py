"""Combiner search — how best to spend ONE review budget across two models (v0.2-G).

The complementarity run showed LeadX and XGBoost flag largely different
transactions (overlap 0.11–0.12 at ultra imbalance) and that reviewing both
lists at 2× capacity lifts recall +41–59%. But a plain rank-average at *fixed*
capacity captured almost none of that (+21% only in one cell, neutral or
negative elsewhere). So the question is not "is there complementarity" — it is
**which combination rule converts it into catches at the same review budget**.

Combiners evaluated at a fixed budget k (all see the same two score vectors):
  xgb_only        : incumbent baseline
  leadx_only      : challenger alone
  rank_avg        : mean of normalised ranks (already measured, kept as control)
  wavg_0.3        : 0.7 × xgb rank + 0.3 × leadx rank (incumbent-anchored)
  rrf             : reciprocal rank fusion, 1/(60+rank) — standard IR fusion
  split_0.25/0.5  : reserve α·k slots for LeadX's top, fill the rest from XGBoost
                    (dedup) — spends capacity directly on disjoint catches
  cascade_xgb     : XGBoost shortlists 10k candidates, LeadX re-ranks them
  cascade_leadx   : LeadX shortlists 10k candidates, XGBoost re-ranks them

PRE-REGISTERED before the full run:
  data/setting : identical to the complementarity run — IEEE fraud, temporal
                 (train 250k → next 70k), 3 rolling folds × 2 PU draws,
                 natural (~3.5%) and 1:1,000, incumbent = XGBoost-PU (trained on
                 confirmed frauds only, i.e. what a fraud team actually has)
  primary      : Recall@0.1% (tight queue); secondary Recall@0.5%
  analysis     : paired per (ratio, fold, draw) vs `xgb_only`, median + wins
  gate         : a combiner beats `xgb_only` on Recall@0.1% at 1:1,000 with
                 paired median > 0 and ≥5/6 wins, AND does not regress more than
                 5% at natural imbalance. Only such a rule is recommended.

Run: uv run --with xgboost,scipy python outputs/combiner_benchmark.py [--smoke]
"""
from __future__ import annotations

import gc
import math
import time
import warnings

import numpy as np
import polars as pl
from scipy.stats import rankdata

from leadx import LeadX
from leadx.encoder import FeatureEncoder
from temporal_fraud_benchmark import (CAT_COLS, LEADX_KW, NUM_COLS,
                                      downsample_positives, load)

warnings.filterwarnings("ignore")
BUDGETS = (0.001, 0.005)


def _order(score: np.ndarray) -> np.ndarray:
    """indices best-first"""
    return np.argsort(-score, kind="stable")


def _split_pick(ord_a, ord_b, k, alpha):
    """สงวน α·k ช่องให้ list a แล้วเติมจาก b (ข้ามตัวซ้ำ) จนครบ k"""
    ka = int(round(alpha * k))
    picked, seen = [], set()
    for i in ord_a:
        if len(picked) >= ka:
            break
        if i not in seen:
            picked.append(i); seen.add(i)
    for i in ord_b:
        if len(picked) >= k:
            break
        if i not in seen:
            picked.append(i); seen.add(i)
    for i in ord_a:                      # เผื่อ b หมดก่อน (ไม่ค่อยเกิด)
        if len(picked) >= k:
            break
        if i not in seen:
            picked.append(i); seen.add(i)
    return np.asarray(picked[:k])


def combiners(s_leadx: np.ndarray, s_xgb: np.ndarray, k: int) -> dict:
    """คืน dict ชื่อ -> ดัชนีที่เลือกไว้ k ตัว (budget เท่ากันทุกวิธี)"""
    n = len(s_leadx)
    ra, rb = rankdata(s_leadx) / n, rankdata(s_xgb) / n     # สูง = ดี
    oa, ob = _order(s_leadx), _order(s_xgb)
    inv_a, inv_b = np.empty(n), np.empty(n)                 # rank 0-based, ต่ำ = ดี
    inv_a[oa] = np.arange(n); inv_b[ob] = np.arange(n)
    m = min(n, 10 * k)
    out = {
        "xgb_only": ob[:k],
        "leadx_only": oa[:k],
        "rank_avg": _order((ra + rb) / 2.0)[:k],
        "wavg_0.3": _order(0.3 * ra + 0.7 * rb)[:k],
        "rrf": _order(1.0 / (60.0 + inv_a) + 1.0 / (60.0 + inv_b))[:k],
        "split_0.25": _split_pick(oa, ob, k, 0.25),
        "split_0.5": _split_pick(oa, ob, k, 0.5),
    }
    cand = ob[:m]                                            # xgb shortlist
    out["cascade_xgb"] = cand[_order(s_leadx[cand])[:k]]
    cand2 = oa[:m]                                           # leadx shortlist
    out["cascade_leadx"] = cand2[_order(s_xgb[cand2])[:k]]
    return out


def run(train_len=250_000, test_len=70_000, stride=100_000, n_folds=3,
        ratios=(None, 1e-3), seed_frac=0.5, draws=2):
    import xgboost as xgb
    df = load()
    y_all = df["isFraud"].to_numpy().astype(np.int8)
    feats = NUM_COLS + CAT_COLS
    rows = []
    for ratio in ratios:
        keep = downsample_positives(y_all, ratio, np.random.default_rng(4242))
        sub, ysub = df[keep.tolist()], y_all[keep]
        rlabel = "natural" if ratio is None else f"{ratio:.0e}"
        for fold in range(n_folds):
            a = fold * stride
            b, c = a + train_len, a + train_len + test_len
            if c > len(sub):
                continue
            Xtr = sub[np.arange(a, b).tolist()].select(feats)
            Xte = sub[np.arange(b, c).tolist()].select(feats)
            ytr, yte = ysub[a:b], ysub[b:c]
            pos_tr = np.flatnonzero(ytr == 1)
            if len(pos_tr) < 40 or yte.sum() < 3:
                continue
            enc = FeatureEncoder().fit(Xtr)
            Ztr, Zte = enc.transform(Xtr), enc.transform(Xte)
            xkw = dict(tree_method="hist", n_estimators=300, max_depth=6,
                       learning_rate=0.1, n_jobs=4)
            pos_set = set(np.flatnonzero(yte == 1).tolist())
            n_pos = max(1, len(pos_set))
            for d in range(draws):
                rng = np.random.default_rng(1000 * fold + 100 * d + 7)
                seeds = rng.choice(pos_tr, max(20, int(seed_frac * len(pos_tr))),
                                   replace=False)
                leadx = LeadX(**LEADX_KW).fit(Xtr[seeds.tolist()], background=Xtr)
                s_leadx = leadx.score_samples(Xte)
                del leadx; gc.collect()
                y_pu = np.zeros(len(ytr), dtype=np.int8); y_pu[seeds] = 1
                s_xgb = (xgb.XGBClassifier(**xkw,
                         scale_pos_weight=(len(y_pu) - y_pu.sum()) / max(1, y_pu.sum()))
                         .fit(Ztr, y_pu).predict_proba(Zte)[:, 1])
                row = {"ratio": rlabel, "fold": fold, "draw": d,
                       "n_test": len(yte), "n_pos_test": int(yte.sum())}
                for frac in BUDGETS:
                    k = max(1, int(math.ceil(len(yte) * frac)))
                    tag = str(frac * 100).rstrip("0").rstrip(".").replace(".", "_")
                    for name, idx in combiners(s_leadx, s_xgb, k).items():
                        row[f"{name}_at_{tag}pct"] = len(set(idx.tolist()) & pos_set) / n_pos
                rows.append(row)
                print(f"  {rlabel} f{fold}d{d}: " + " ".join(
                    f"{nm}={row[f'{nm}_at_0_1pct']:.3f}"
                    for nm in ("xgb_only", "rank_avg", "rrf", "split_0.25",
                               "cascade_xgb")), flush=True)
            del Ztr, Zte; gc.collect()
    return rows


COMBINERS = ("leadx_only", "rank_avg", "wavg_0.3", "rrf", "split_0.25",
             "split_0.5", "cascade_xgb", "cascade_leadx")


def summarize(raw: pl.DataFrame) -> None:
    for ratio in raw["ratio"].unique(maintain_order=True):
        r = raw.filter(pl.col("ratio") == ratio)
        print(f"\n=== ratio={ratio} (n={r.height}) — paired vs xgb_only ===")
        for tag, lbl in (("0_1", "Recall@0.1%  [PRIMARY]"), ("0_5", "Recall@0.5%")):
            base = r[f"xgb_only_at_{tag}pct"].to_numpy()
            print(f"  {lbl}   xgb_only median = {np.median(base):.4f}")
            scored = []
            for nm in COMBINERS:
                v = r[f"{nm}_at_{tag}pct"].to_numpy()
                d = (v - base) / np.maximum(1e-12, base) * 100
                scored.append((float(np.median(d)), int((d > 0).sum()), nm,
                               float(np.median(v))))
            for med, wins, nm, absmed in sorted(scored, reverse=True):
                flag = " <<<" if (med > 0 and wins >= 5) else ""
                print(f"    {nm:14s} {absmed:.4f}  ({med:+6.1f}%, {wins}/{len(base)}){flag}")


if __name__ == "__main__":
    import sys
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    rows = run(n_folds=1, ratios=(1e-3,), draws=1) if smoke else run()
    raw = pl.DataFrame(rows)
    out = ("outputs/combiner_smoke_raw.csv" if smoke else "outputs/combiner_raw.csv")
    raw.write_csv(out)
    summarize(raw)
    print(f"\n{out}  |  {(time.time() - t0) / 60:.1f} min")
