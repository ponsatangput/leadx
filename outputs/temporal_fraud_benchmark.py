"""Temporal fraud rehearsal — train on the past, score the future (v0.2-D).

Dress rehearsal for the real production question: given one period of payments
plus the frauds confirmed in it, rank next period's transactions. Every earlier
LeadX benchmark used random / feature-group splits within a single period, so
**temporal generalisation has never been measured** — this closes that gap and
doubles as the second categorical dataset for `cat_engine="minhash"`.

Key design — the counterfactual that isolates drift:
for every fold we run BOTH
  * temporal : train = earlier rows, test = the rows immediately after
  * random   : same row counts drawn at random from the same union of rows
`drift cost = random − temporal` per model. Absolute temporal numbers say
"can it work at all"; the paired difference says "who degrades least when the
world moves", which is the actual production risk.

PRE-REGISTERED before running (locked; committed with this file):
  data      : IEEE-CIS fraud train_transaction.csv, sorted by TransactionDT
  imbalance : natural (~3.5%) and downsampled 0.1% (1:1,000)
  folds     : 3 rolling origins (train 250k rows -> test next 70k), stride 100k
  PU        : 50% of the train window's frauds are "confirmed" seeds, the rest
              stay unlabeled in the pool (genuine PU); 2 seed draws per fold
  models    : leadx_pu (rpforest b12 K40 lift/log contrast + MinHash categorical),
              logreg_pu_naive, xgboost_pu_naive, xgboost_oracle (full labels)
  primary   : Recall@0.5% on the test window; also PR-AUC, Recall@0.1%
  analysis  : paired per (fold, draw, ratio); report temporal, random, and the
              drift cost for each model. 3 folds x 2 draws = replication, not a
              significance test.
  gate      : LeadX stays usable under temporal shift (Recall@0.5% within noise
              of its random-split value) AND its drift cost is no worse than
              XGBoost's -> "temporal-ready for a pilot"; otherwise report the gap.

Run: uv run --with xgboost python outputs/temporal_fraud_benchmark.py [--smoke]
"""
from __future__ import annotations

import gc
import math
import time
import warnings

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from leadx import LeadX
from leadx.encoder import FeatureEncoder

warnings.filterwarnings("ignore")
DATA = "/Users/ponpiboonsatangput/jarvis/data/kaggle/fraud-detection/train_transaction.csv"
CAT_COLS = (["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
             "addr1", "addr2", "P_emaildomain", "R_emaildomain"]
            + [f"M{i}" for i in range(1, 10)])
NUM_COLS = (["TransactionAmt"] + [f"C{i}" for i in range(1, 15)]
            + [f"D{i}" for i in range(1, 16)] + ["dist1", "dist2"])
REVIEW = (0.001, 0.005)
LEADX_KW = dict(mode="pu", tune=False, n_bits=12, n_tables=40, score_type="lift",
                agg="log", feature_weighting="contrast", engine="rpforest",
                cat_engine="minhash", minhash_columns=CAT_COLS, random_state=0)


def load() -> pl.DataFrame:
    cols = ["isFraud", "TransactionDT"] + NUM_COLS + CAT_COLS
    return pl.read_csv(DATA, columns=cols, infer_schema_length=20000,
                       ignore_errors=True).sort("TransactionDT")


def metrics(y, s) -> dict:
    y = np.asarray(y, np.int8)
    s = np.asarray(s, np.float64)
    out = {"pr_auc": float(average_precision_score(y, s)), "base": float(y.mean())}
    order = np.argsort(-s)
    for f in REVIEW:
        k = max(1, int(math.ceil(len(y) * f)))
        tag = str(f * 100).rstrip("0").rstrip(".").replace(".", "_")
        out[f"recall_at_{tag}pct"] = float(y[order[:k]].sum()) / max(1.0, y.sum())
    return out


def downsample_positives(y: np.ndarray, ratio: float | None, rng) -> np.ndarray:
    """index ที่เก็บ (เรียงเวลาเดิม) — ลด positives ให้ได้ ratio โดยสุ่มทั่วทั้งช่วงเวลา"""
    if ratio is None:
        return np.arange(len(y))
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = int(round(ratio * len(neg) / (1 - ratio)))
    keep_pos = rng.choice(pos, size=min(len(pos), n_pos), replace=False)
    return np.sort(np.concatenate([keep_pos, neg]))


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
                print(f"  {rlabel} fold{fold}: rows ไม่พอ — ข้าม"); continue
            idx_tr, idx_te = np.arange(a, b), np.arange(b, c)
            for split_type in ("temporal", "random"):
                if split_type == "random":     # counterfactual: ช่วงเวลาเดียวกัน สลับแถว
                    pool = np.arange(a, c)
                    perm = np.random.default_rng(900 + fold).permutation(pool)
                    itr, ite = np.sort(perm[:train_len]), np.sort(perm[train_len:])
                else:
                    itr, ite = idx_tr, idx_te
                Xtr = sub[itr.tolist()].select(feats)
                Xte = sub[ite.tolist()].select(feats)
                ytr, yte = ysub[itr], ysub[ite]
                pos_tr = np.flatnonzero(ytr == 1)
                if len(pos_tr) < 40 or yte.sum() < 3:
                    print(f"  {rlabel} f{fold} {split_type}: positives ไม่พอ "
                          f"(tr={len(pos_tr)}, te+={int(yte.sum())}) — ข้าม"); continue
                enc = FeatureEncoder().fit(Xtr)
                Ztr, Zte = enc.transform(Xtr), enc.transform(Xte)
                xkw = dict(tree_method="hist", n_estimators=300, max_depth=6,
                           learning_rate=0.1, n_jobs=4)

                def rec(model, s, draw):
                    m = metrics(yte, s)
                    rows.append({"ratio": rlabel, "fold": fold, "split": split_type,
                                 "draw": draw, "model": model,
                                 "n_pos_train": int(len(pos_tr)),
                                 "n_test": len(yte), **m})
                    print(f"  {rlabel} f{fold} {split_type:8s} d{draw} {model:16s} "
                          f"pr={m['pr_auc']:.4f} r@0.5%={m['recall_at_0_5pct']:.3f}",
                          flush=True)

                rec("xgboost_oracle", xgb.XGBClassifier(**xkw,
                    scale_pos_weight=(len(ytr) - ytr.sum()) / max(1, ytr.sum()))
                    .fit(Ztr, ytr).predict_proba(Zte)[:, 1], -1)
                for d in range(draws):
                    rng = np.random.default_rng(1000 * fold + 100 * d + 7)
                    seeds = rng.choice(pos_tr, max(20, int(seed_frac * len(pos_tr))),
                                       replace=False)
                    leadx = LeadX(**LEADX_KW).fit(Xtr[seeds.tolist()], background=Xtr)
                    rec("leadx_pu", leadx.score_samples(Xte), d)
                    del leadx; gc.collect()
                    y_pu = np.zeros(len(ytr), dtype=np.int8); y_pu[seeds] = 1
                    rec("logreg_pu_naive", LogisticRegression(
                        max_iter=200, class_weight="balanced")
                        .fit(Ztr, y_pu).predict_proba(Zte)[:, 1], d)
                    rec("xgboost_pu_naive", xgb.XGBClassifier(**xkw,
                        scale_pos_weight=(len(y_pu) - y_pu.sum()) / max(1, y_pu.sum()))
                        .fit(Ztr, y_pu).predict_proba(Zte)[:, 1], d)
                del Ztr, Zte; gc.collect()
    return rows


def summarize(raw: pl.DataFrame) -> None:
    for ratio in raw["ratio"].unique(maintain_order=True):
        r = raw.filter(pl.col("ratio") == ratio)
        print(f"\n=== ratio={ratio} — temporal vs random (median), drift cost ===")
        for metric in ("pr_auc", "recall_at_0_5pct", "recall_at_0_1pct"):
            print(f"  [{metric}]")
            for model in ("leadx_pu", "logreg_pu_naive", "xgboost_pu_naive",
                          "xgboost_oracle"):
                m = r.filter(pl.col("model") == model)
                t = m.filter(pl.col("split") == "temporal")[metric].to_numpy()
                q = m.filter(pl.col("split") == "random")[metric].to_numpy()
                if not len(t) or not len(q):
                    continue
                tm, qm = float(np.median(t)), float(np.median(q))
                drop = (tm - qm) / max(1e-12, qm) * 100
                print(f"    {model:16s} temporal {tm:.4f} | random {qm:.4f} | "
                      f"drift {drop:+6.1f}%")
        # paired leadx vs each baseline on the temporal split
        tt = r.filter(pl.col("split") == "temporal")
        lx = {(d["fold"], d["draw"]): d for d in tt.filter(pl.col("model") == "leadx_pu").to_dicts()}
        print("  [leadx_pu paired vs baselines — TEMPORAL only]")
        for opp in ("xgboost_pu_naive", "logreg_pu_naive"):
            B = {(d["fold"], d["draw"]): d for d in tt.filter(pl.col("model") == opp).to_dicts()}
            common = sorted(set(lx) & set(B))
            line = f"    vs {opp:16s}"
            for metric in ("pr_auc", "recall_at_0_5pct"):
                d = [(lx[k][metric] - B[k][metric]) / max(1e-12, B[k][metric]) * 100
                     for k in common]
                if d:
                    line += (f"  {metric.replace('recall_at_', 'R@').replace('pr_auc', 'PR')}: "
                             f"{float(np.median(d)):+.0f}%({sum(x > 0 for x in d)}/{len(d)})")
            print(line)


if __name__ == "__main__":
    import sys
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    rows = run(n_folds=1, ratios=(None,), draws=1) if smoke else run()
    raw = pl.DataFrame(rows)
    out = ("outputs/temporal_fraud_smoke_raw.csv" if smoke
           else "outputs/temporal_fraud_raw.csv")
    raw.write_csv(out)
    summarize(raw)
    print(f"\n{out}  |  {(time.time() - t0) / 60:.1f} min")
