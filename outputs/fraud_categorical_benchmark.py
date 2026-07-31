"""Categorical-handling ablation on IEEE-CIS fraud (v0.2, workstream A).

Clean ablation: the numeric geometry is IDENTICAL across LeadX variants (same K
rpforest tables on the numeric columns). We only change what categorical tables
are *appended* (additive across tables, flat lift so `hier` is not a confound):

  numeric_only  : numeric tables only
  cat_onehot    : numeric + rpforest-on-onehot categorical tables   (current way)
  cat_minhash   : numeric + MinHash categorical tables              (new way)
  cat_within    : within-table concat [numeric | MinHash] (production-style, one code)

Comparisons: (onehot|minhash) vs numeric_only = does categorical help;
minhash vs onehot = is MinHash the better handler. card1-6/addr treated as
categorical for the MinHash paths (currently z-scored as numeric).

Baselines: logreg PU-naive, XGBoost PU-naive (native categorical), XGBoost
supervised oracle (full labels — ceiling reference).

Fraud can't hit 1:10,000 with enough positives (590k rows) — this tests the
categorical question at 1:1,000; the pure 1:10,000 imbalance test is on Criteo.

Run: uv run --with xgboost,pyarrow python outputs/fraud_categorical_benchmark.py
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

from leadx.encoder import FeatureEncoder
from leadx.rpforest import RPForest
from leadx.scoring import SeedScorer
from minhash_cat import MinHashCat

warnings.filterwarnings("ignore")
DATA = "/Users/ponpiboonsatangput/jarvis/data/kaggle/fraud-detection/train_transaction.csv"
CAT_COLS = (["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
             "addr1", "addr2", "P_emaildomain", "R_emaildomain"]
            + [f"M{i}" for i in range(1, 10)])
NUM_COLS = (["TransactionAmt"] + [f"C{i}" for i in range(1, 15)]
            + [f"D{i}" for i in range(1, 16)] + ["dist1", "dist2"])
N_TABLES, B_NUM, B_CAT = 40, 12, 8
REVIEW = (0.001, 0.005)


def load() -> pl.DataFrame:
    cols = ["isFraud", "TransactionDT"] + NUM_COLS + CAT_COLS
    return pl.read_csv(DATA, columns=cols, infer_schema_length=20000,
                       ignore_errors=True).sort("TransactionDT")


def metrics(y, s) -> dict:
    y = np.asarray(y, np.int8); s = np.asarray(s, np.float64)
    out = {"pr_auc": float(average_precision_score(y, s)), "base": float(y.mean())}
    order = np.argsort(-s)
    for f in REVIEW:
        k = max(1, int(math.ceil(len(y) * f)))
        tag = str(f * 100).rstrip("0").rstrip(".").replace(".", "_")
        out[f"recall_at_{tag}pct"] = float(y[order[:k]].sum()) / max(1.0, y.sum())
    return out


def flat_lift(codes_tr, codes_ev, seed_rows):
    """flat lift/log per-table groupby averaged — ไม่มี hier (กัน confound)"""
    sc = SeedScorer(mode="pu", score_type="lift", agg="log", n_bits=B_NUM,
                    multi_probe=False)
    sc.fit(codes_tr[seed_rows], np.ones(len(seed_rows)), background_codes=codes_tr)
    return sc.score(codes_ev)


def build_codes(df_tr, df_ev, seed):
    """คืน dict variant -> (codes_tr, codes_ev); numeric tables ใช้ร่วมทุก variant"""
    enc_n = FeatureEncoder().fit(df_tr.select(NUM_COLS))
    Zn_tr, Zn_ev = enc_n.transform(df_tr.select(NUM_COLS)), enc_n.transform(df_ev.select(NUM_COLS))
    fnum = RPForest(N_TABLES, B_NUM, random_state=seed).fit(Zn_tr)
    num_tr, num_ev = fnum.transform(Zn_tr), fnum.transform(Zn_ev)

    enc_c = FeatureEncoder().fit(df_tr.select(CAT_COLS))          # onehot/hash categorical
    Zc_tr, Zc_ev = enc_c.transform(df_tr.select(CAT_COLS)), enc_c.transform(df_ev.select(CAT_COLS))
    foh = RPForest(N_TABLES, B_CAT, random_state=seed + 1).fit(Zc_tr)
    oh_tr, oh_ev = foh.transform(Zc_tr), foh.transform(Zc_ev)

    mh = MinHashCat(CAT_COLS, N_TABLES, B_CAT, random_state=seed)
    mh_tr, mh_ev = mh.transform(df_tr), mh.transform(df_ev)

    cat = lambda a, b: (np.concatenate([num_tr, a], axis=1),
                        np.concatenate([num_ev, b], axis=1))
    within = ((num_tr << np.uint64(B_CAT)) | mh_tr, (num_ev << np.uint64(B_CAT)) | mh_ev)
    return {
        "numeric_only": (num_tr, num_ev),
        "cat_onehot": cat(oh_tr, oh_ev),
        "cat_minhash": cat(mh_tr, mh_ev),
        "cat_within": within,
    }


def run(ratio=0.001, n_seeds=200, draws=5):
    import xgboost as xgb
    df = load()
    y_all = df["isFraud"].to_numpy().astype(np.int8)
    pos, neg = np.flatnonzero(y_all == 1), np.flatnonzero(y_all == 0)
    rows = []
    for d in range(draws):
        rng = np.random.default_rng(100 * d + 7)
        n_keep = max(n_seeds + 80, int(round(ratio * len(neg) / (1 - ratio))))
        keep = np.sort(np.concatenate([rng.choice(pos, min(len(pos), n_keep), replace=False), neg]))
        sub, ysub = df[keep.tolist()], y_all[keep]
        m_tr = np.arange(len(keep)) < int(0.7 * len(keep))
        df_tr, df_ev = sub.filter(pl.Series(m_tr)), sub.filter(pl.Series(~m_tr))
        y_tr, y_ev = ysub[m_tr], ysub[~m_tr]
        pos_tr = np.flatnonzero(y_tr == 1)
        if len(pos_tr) < n_seeds or y_ev.sum() < 3:
            print(f"  d{d}: positives ไม่พอ — ข้าม"); continue
        seed_rows = rng.choice(pos_tr, n_seeds, replace=False)

        def rec(model, s):
            m = metrics(y_ev, s)
            rows.append({"draw": d, "model": model, "ratio": ratio, **m})
            print(f"  d{d} {model:16s} pr={m['pr_auc']:.4f} r@0.5%={m['recall_at_0_5pct']:.3f}", flush=True)

        codes = build_codes(df_tr, df_ev, seed=d)
        for v, (ctr, cev) in codes.items():
            rec(f"leadx_{v}", flat_lift(ctr, cev, seed_rows))
        del codes; gc.collect()

        enc = FeatureEncoder().fit(df_tr)
        Ztr, Zev = enc.transform(df_tr), enc.transform(df_ev)
        y_pu = np.zeros(len(y_tr)); y_pu[seed_rows] = 1
        rec("logreg_pu_naive", LogisticRegression(max_iter=300, class_weight="balanced")
            .fit(Ztr, y_pu).predict_proba(Zev)[:, 1])
        del Ztr, Zev

        # cat -> string + category ที่ align train/eval (xgboost ไม่รับ float cat;
        # ต้องใช้ coding เดียวกัน ไม่งั้น predict พัง)
        import pandas as pd
        cast = [pl.col(c).cast(pl.Utf8).fill_null("NA") for c in CAT_COLS]
        ptr = df_tr.select(NUM_COLS).hstack(df_tr.select(cast)).to_pandas()
        pev = df_ev.select(NUM_COLS).hstack(df_ev.select(cast)).to_pandas()
        for c in CAT_COLS:
            cats = pd.Index(ptr[c].unique())
            ptr[c] = pd.Categorical(ptr[c], categories=cats)
            pev[c] = pd.Categorical(pev[c], categories=cats)   # unseen -> NaN
        xkw = dict(tree_method="hist", enable_categorical=True, n_estimators=200,
                   max_depth=6, learning_rate=0.1, n_jobs=4)
        rec("xgboost_pu_naive", xgb.XGBClassifier(**xkw,
            scale_pos_weight=(len(y_pu) - y_pu.sum()) / max(1, y_pu.sum()))
            .fit(ptr, y_pu).predict_proba(pev)[:, 1])
        rec("xgboost_oracle", xgb.XGBClassifier(**xkw,
            scale_pos_weight=(len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()))
            .fit(ptr, y_tr).predict_proba(pev)[:, 1])
        del ptr, pev; gc.collect()
    return rows


if __name__ == "__main__":
    import sys
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    rows = run(n_seeds=100 if smoke else 200, draws=1 if smoke else 5)
    raw = pl.DataFrame(rows)
    out = ("outputs/fraud_categorical_smoke_raw.csv" if smoke
           else "outputs/fraud_categorical_raw.csv")
    raw.write_csv(out)
    med = (raw.group_by("model").agg(
        pl.col("pr_auc").median().alias("pr"),
        pl.col("recall_at_0_5pct").median().alias("r05"),
        pl.col("recall_at_0_1pct").median().alias("r01")).sort("pr", descending=True))
    print("\n=== median ==="); print(med)
    print(f"\n{out}  |  {(time.time()-t0)/60:.1f} min")
