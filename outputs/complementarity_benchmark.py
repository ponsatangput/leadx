"""Complementarity: does LeadX catch fraud that XGBoost misses? (v0.2-E)

The production question for a team that already runs XGBoost is not "is LeadX
better" but **"does adding LeadX catch cases XGBoost misses"**. Different
inductive biases (density/similarity vs decision boundary) should disagree on
which transactions look suspicious; if the top-k lists overlap little, the union
/ ensemble is worth real money.

Reuses the temporal setup (train past -> score future, IEEE fraud) and measures,
per review budget k:
  overlap@k       : |top-k(leadx) & top-k(xgb)| / k
  unique catches  : frauds in leadx top-k that xgb top-k misses (and vice versa)
  ensemble@k      : recall of a rank-averaged score at the SAME budget k
                    (fair: same review capacity, not a bigger queue)
  union@2k        : recall if you review both lists (double capacity — reference)

Also runs xgboost_oracle (full labels) as the strongest realistic incumbent.

Run: uv run --with xgboost python outputs/complementarity_benchmark.py [--smoke]
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


def topk(score: np.ndarray, k: int) -> np.ndarray:
    return np.argpartition(-score, k - 1)[:k]


def pair_stats(y, s_a, s_b, frac, name_a="leadx", name_b="xgb") -> dict:
    n = len(y)
    k = max(1, int(math.ceil(n * frac)))
    A, B = set(topk(s_a, k).tolist()), set(topk(s_b, k).tolist())
    inter = A & B
    pos = set(np.flatnonzero(y == 1).tolist())
    n_pos = max(1, len(pos))
    # ensemble at the SAME budget: average of normalised ranks
    ra, rb = rankdata(s_a) / n, rankdata(s_b) / n
    ens = topk((ra + rb) / 2.0, k)
    union2k = A | B                       # double capacity reference
    tag = str(frac * 100).rstrip("0").rstrip(".").replace(".", "_")
    return {
        f"overlap_at_{tag}pct": len(inter) / k,
        f"recall_{name_a}_at_{tag}pct": len(A & pos) / n_pos,
        f"recall_{name_b}_at_{tag}pct": len(B & pos) / n_pos,
        f"recall_ensemble_at_{tag}pct": len(set(ens.tolist()) & pos) / n_pos,
        f"recall_union2k_at_{tag}pct": len(union2k & pos) / n_pos,
        f"caught_only_by_{name_a}_at_{tag}pct": len((A - B) & pos),
        f"caught_only_by_{name_b}_at_{tag}pct": len((B - A) & pos),
        f"n_reviewed_at_{tag}pct": k,
    }


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
            s_oracle = (xgb.XGBClassifier(**xkw,
                        scale_pos_weight=(len(ytr) - ytr.sum()) / max(1, ytr.sum()))
                        .fit(Ztr, ytr).predict_proba(Zte)[:, 1])
            for d in range(draws):
                rng = np.random.default_rng(1000 * fold + 100 * d + 7)
                seeds = rng.choice(pos_tr, max(20, int(seed_frac * len(pos_tr))),
                                   replace=False)
                leadx = LeadX(**LEADX_KW).fit(Xtr[seeds.tolist()], background=Xtr)
                s_leadx = leadx.score_samples(Xte)
                del leadx; gc.collect()
                y_pu = np.zeros(len(ytr), dtype=np.int8); y_pu[seeds] = 1
                s_xgbpu = (xgb.XGBClassifier(**xkw,
                           scale_pos_weight=(len(y_pu) - y_pu.sum()) / max(1, y_pu.sum()))
                           .fit(Ztr, y_pu).predict_proba(Zte)[:, 1])
                for opp_name, s_opp in (("xgbpu", s_xgbpu), ("xgboracle", s_oracle)):
                    row = {"ratio": rlabel, "fold": fold, "draw": d,
                           "opponent": opp_name, "n_test": len(yte),
                           "n_pos_test": int(yte.sum())}
                    for frac in BUDGETS:
                        row.update(pair_stats(yte, s_leadx, s_opp, frac,
                                              "leadx", opp_name))
                    rows.append(row)
                    tag = "0_5"
                    print(f"  {rlabel} f{fold}d{d} vs {opp_name:9s} "
                          f"overlap@0.5%={row['overlap_at_0_5pct']:.2f} "
                          f"leadx={row[f'recall_leadx_at_{tag}pct']:.3f} "
                          f"{opp_name}={row[f'recall_{opp_name}_at_{tag}pct']:.3f} "
                          f"ens={row[f'recall_ensemble_at_{tag}pct']:.3f} "
                          f"only_leadx={row[f'caught_only_by_leadx_at_{tag}pct']}",
                          flush=True)
            del Ztr, Zte; gc.collect()
    return rows


def summarize(raw: pl.DataFrame) -> None:
    for ratio in raw["ratio"].unique(maintain_order=True):
        for opp in raw["opponent"].unique(maintain_order=True):
            r = raw.filter((pl.col("ratio") == ratio) & (pl.col("opponent") == opp))
            if not r.height:
                continue
            print(f"\n=== ratio={ratio} — leadx vs {opp} (n={r.height}) ===")
            for tag, lbl in (("0_1", "0.1%"), ("0_5", "0.5%")):
                ov = r[f"overlap_at_{tag}pct"].median()
                rl = r[f"recall_leadx_at_{tag}pct"].to_numpy()
                ro = r[f"recall_{opp}_at_{tag}pct"].to_numpy()
                re_ = r[f"recall_ensemble_at_{tag}pct"].to_numpy()
                ru = r[f"recall_union2k_at_{tag}pct"].to_numpy()
                only_l = r[f"caught_only_by_leadx_at_{tag}pct"].to_numpy()
                only_o = r[f"caught_only_by_{opp}_at_{tag}pct"].to_numpy()
                gain = (re_ - ro) / np.maximum(1e-12, ro) * 100
                print(f"  budget {lbl}: overlap {ov:.2f} | recall leadx "
                      f"{np.median(rl):.3f} / {opp} {np.median(ro):.3f} / "
                      f"ensemble {np.median(re_):.3f} (vs {opp}: "
                      f"{np.median(gain):+.1f}%, {int((gain > 0).sum())}/{len(gain)}) "
                      f"| union@2k {np.median(ru):.3f}")
                print(f"     frauds caught ONLY by leadx: {np.median(only_l):.1f} | "
                      f"ONLY by {opp}: {np.median(only_o):.1f} (median per fold)")


if __name__ == "__main__":
    import sys
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    rows = run(n_folds=1, ratios=(None,), draws=1) if smoke else run()
    raw = pl.DataFrame(rows)
    out = ("outputs/complementarity_smoke_raw.csv" if smoke
           else "outputs/complementarity_raw.csv")
    raw.write_csv(out)
    summarize(raw)
    print(f"\n{out}  |  {(time.time() - t0) / 60:.1f} min")
