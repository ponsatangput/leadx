"""Criteo PU ultra-imbalance benchmark (v0.2 workstream B).

The production regime LeadX was built for: ~10-20M rows, ~1-2k confirmed
positives (≈1:10,000), positive-unlabeled. Every earlier Criteo test was at
1:338 with 20-100 seeds — the wrong regime. The ultra-imbalance report touched
1:10,000 but *fully-supervised*; this measures the PU version that matches the
use case, and adds XGBoost.

PRE-REGISTERED before running (locked; git-committed with this file):
  data      : Criteo Uplift v2.1, uniform 10M sample, conversion target
  imbalance : downsample positives to 1:10,000 (and 1:1,000 for contrast)
  splits    : 3 feature-group hash splits (seeds 41001-41003), 70/30 train/test
  seeds(PU) : 50% of train positives confirmed; the other 50% stay UNLABELED in
              the pool (genuine PU — allseeds would make PU-naive == supervised).
              3 seed draws per split. Also reported: seed_frac 1.0 (=supervised
              reference, where pu-naive collapses to oracle)
  models    : leadx_pu (rpforest b10 K80 lift/log hier contrast),
              logreg_pu_naive, xgboost_pu_naive, xgboost_oracle (full labels)
  primary   : Recall@0.5% (broad review queue) ; also PR-AUC, Recall@0.1/0.01%
  analysis  : paired per split (leadx vs each), median + wins/3 (NOT ratio-of-
              medians). 3 splits = replication, not a significance test.
  gate      : leadx_pu beats xgboost_pu_naive on Recall@0.5% AND PR-AUC in >=2/3
              splits at 1:10,000 -> preliminary "wins strong PU at ultra imbalance"

Run: uv run --with xgboost python outputs/criteo_pu_ultra_benchmark.py
"""
from __future__ import annotations

import gc
import time
import warnings

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from criteo_uplift_benchmark import (FEATURES, group_hash_splits, load_criteo,
                                     tie_aware_ranking_metrics)
from leadx import LeadX
from leadx.encoder import FeatureEncoder

warnings.filterwarnings("ignore")
DATA = "/Users/ponpiboonsatangput/jarvis/data/criteo-uplift/criteo-research-uplift-v2.1.csv.gz"
TARGET = "conversion"
LEADX_KW = dict(mode="pu", tune=False, n_bits=10, n_tables=80, score_type="lift",
                agg="log", feature_weighting="contrast", engine="rpforest",
                hier_depths="auto", random_state=0)


def downsample_to_ratio(y: np.ndarray, ratio: float, rng) -> np.ndarray:
    """คืน index ที่เก็บไว้ (negatives ทั้งหมด + positives ที่ downsample ให้ได้ ratio)"""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = int(round(ratio * len(neg) / (1 - ratio)))
    keep_pos = rng.choice(pos, size=min(len(pos), n_pos), replace=False)
    return np.sort(np.concatenate([keep_pos, neg]))


def run(max_rows=10_000_000, ratios=(1e-4, 1e-3), split_seeds=(41001, 41002, 41003),
        seed_frac=0.5, draws=3):
    import xgboost as xgb
    frame, _ = load_criteo(pl.Path(DATA) if hasattr(pl, "Path") else __import__("pathlib").Path(DATA),
                           max_rows, sample_seed=42, population="all")
    y_all = frame[TARGET].to_numpy().astype(np.int8)
    rows = []
    for ratio in ratios:
        keep = downsample_to_ratio(y_all, ratio, np.random.default_rng(2025))
        sub = frame[keep.tolist()]
        ysub = y_all[keep]
        for si, sseed in enumerate(split_seeds):
            split, _ = group_hash_splits(sub, split_seed=sseed)
            tr, ev = split == "train", split == "test"
            Xtr, Xev = sub.filter(pl.Series(tr)).select(FEATURES), sub.filter(pl.Series(ev)).select(FEATURES)
            ytr, yev = ysub[tr], ysub[ev]
            pos_tr = np.flatnonzero(ytr == 1)
            if len(pos_tr) < 40 or yev.sum() < 3:
                print(f"  r={ratio:.1e} s{si}: positives ไม่พอ — ข้าม"); continue
            enc = FeatureEncoder().fit(Xtr)
            Ztr, Zev = enc.transform(Xtr), enc.transform(Xev)
            xkw = dict(tree_method="hist", n_estimators=300, max_depth=6,
                       learning_rate=0.1, n_jobs=4)
            # oracle (label เต็ม) — ไม่ขึ้นกับ draw
            def rec(model, s, draw):
                m = tie_aware_ranking_metrics(yev, s)
                rows.append({"ratio": ratio, "split": si, "draw": draw, "model": model,
                             "n_pos_train": len(pos_tr), "n_eval": len(yev), **m})
                print(f"  r={ratio:.1e} s{si}d{draw} {model:16s} pr={m['pr_auc']:.4f} "
                      f"r@0.5%={m['recall_at_0_5pct']:.3f} r@0.1%={m['recall_at_0_1pct']:.3f}",
                      flush=True)
            rec("xgboost_oracle", xgb.XGBClassifier(**xkw,
                scale_pos_weight=(len(ytr)-ytr.sum())/max(1, ytr.sum()))
                .fit(Ztr, ytr).predict_proba(Zev)[:, 1], -1)
            for d in range(draws):
                rng = np.random.default_rng(1000 * si + 100 * d + 7)
                seed_rows = rng.choice(pos_tr, max(20, int(seed_frac * len(pos_tr))),
                                       replace=False)                  # confirmed subset (PU)
                t0 = time.perf_counter()
                leadx = LeadX(**LEADX_KW).fit(Xtr[seed_rows.tolist()], background=Xtr)
                rec("leadx_pu", leadx.score_samples(Xev), d)
                del leadx; gc.collect()
                y_pu = np.zeros(len(ytr), dtype=np.int8); y_pu[seed_rows] = 1  # rest unlabeled=0
                rec("logreg_pu_naive", LogisticRegression(max_iter=200, class_weight="balanced")
                    .fit(Ztr, y_pu).predict_proba(Zev)[:, 1], d)
                rec("xgboost_pu_naive", xgb.XGBClassifier(**xkw,
                    scale_pos_weight=(len(y_pu)-y_pu.sum())/max(1, y_pu.sum()))
                    .fit(Ztr, y_pu).predict_proba(Zev)[:, 1], d)
            del Ztr, Zev; gc.collect()
    return rows


def summarize(rows):
    raw = pl.DataFrame(rows)
    print("\n=== leadx_pu paired vs xgboost_pu_naive (per (split,draw)) ===")
    for ratio in sorted(raw["ratio"].unique()):
        r = raw.filter(pl.col("ratio") == ratio)
        key = lambda d: (d["split"], d["draw"])
        base = {key(d): d for d in r.filter(pl.col("model") == "xgboost_pu_naive").to_dicts()}
        lxr = {key(d): d for d in r.filter(pl.col("model") == "leadx_pu").to_dicts()}
        common = sorted(set(base) & set(lxr))
        for metric in ("pr_auc", "recall_at_0_5pct", "recall_at_0_1pct"):
            deltas = [(lxr[k][metric] - base[k][metric]) / max(1e-9, base[k][metric]) * 100
                      for k in common]
            if deltas:
                print(f"  {ratio:.1e} {metric:18s} median {float(np.median(deltas)):+6.1f}%  "
                      f"wins {sum(d > 0 for d in deltas)}/{len(deltas)}")
    return raw


if __name__ == "__main__":
    t0 = time.time()
    rows = run()
    raw = summarize(rows)
    raw.write_csv("outputs/criteo_pu_ultra_raw.csv")
    med = (raw.group_by(["ratio", "model"]).agg(
        pl.col("pr_auc").median().alias("pr"),
        pl.col("recall_at_0_5pct").median().alias("r05"),
        pl.col("recall_at_0_1pct").median().alias("r01"))
        .sort(["ratio", "pr"], descending=[False, True]))
    print("\n=== median ==="); print(med)
    print(f"\noutputs/criteo_pu_ultra_raw.csv  |  {(time.time()-t0)/60:.1f} min")
