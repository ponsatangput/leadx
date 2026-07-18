"""Imbalance-focused benchmark — คู่เทียบ "ข้อมูลเท่ากัน" (แก้จุดอ่อนของ audit benchmark)

audit's extreme_imbalance_benchmark เทียบ LeadX PU (เห็นแค่ positives) กับ HistGB
ที่เห็น label เต็ม → วัด "ราคาของการไม่มี negatives" ปนกับ "ทน imbalance"
ที่นี่แยกคู่เทียบให้ตรงคำถาม:

  ที่ information เท่ากัน (seeds 10% ของ positives, ไม่มี negative labels):
    - leadx_pu_10pct     : config deploy จริง (b10 K80 lift/log contrast, parity)
    - histgb_pu_naive    : unlabeled = negative + balanced weights (มาตรฐาน PU-naive)
    - logreg_pu_naive    : linear baseline เดียวกัน
  จุดอ้างอิง:
    - histgb_oracle_full : label เต็มทุกตัว (เพดาน — ไม่มีใครมีข้อมูลนี้ในโจทย์ PU)
    - leadx_pu_allseeds  : positives 100% เป็น seeds (setting ของ audit, config ใหม่)

sweep ratio จาก 3% → 0.1% (fraud temporal split + home-credit) 3 draws
รัน: uv run python outputs/imbalance_focus_benchmark.py
"""
import gc
import time
import warnings

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def hgb(class_weight=None, random_state=0):
    """hyperparameters ตามที่ audit จูนไว้ — l2=1.0 สำคัญมากเมื่อ positives น้อย
    (default l2=0 ทำ oracle พังที่ ratio 0.1%: ROC 0.53 ทุก random_state)"""
    return HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=150, max_leaf_nodes=31,
        l2_regularization=1.0, class_weight=class_weight,
        random_state=random_state)

from extreme_imbalance_benchmark import keep_train_valid_columns, metrics, sample_ratio
from leadx import LeadX
from leadx.encoder import FeatureEncoder
from real_data_benchmark import load_dataset, split_data

warnings.filterwarnings("ignore")
DRAWS = 3
SEED_FRAC = 0.10
LEADX_KW = dict(mode="pu", tune=False, n_bits=10, n_tables=80, score_type="lift",
                agg="log", feature_weighting="contrast", random_state=0)


def leadx_fit_score(X_seed, X_bg, X_te):
    t0 = time.perf_counter()
    clf = LeadX(**LEADX_KW).fit(X_seed, background=X_bg)
    fit_s = time.perf_counter() - t0
    return clf.score_samples(X_te), fit_s


def run_dataset(name, ratios):
    df, target, split_type, max_rows = load_dataset(name)
    train_full, test_full = split_data(df, target, split_type, max_rows)
    rows = []
    for ratio in ratios:
        train = sample_ratio(train_full, target, ratio, 42)
        test = sample_ratio(test_full, target, ratio, 43)
        train, test = keep_train_valid_columns(train, test, target)
        y_tr = train[target].to_numpy().astype(np.float64)
        y_te = test[target].to_numpy().astype(np.float64)
        X_tr, X_te = train.drop(target), test.drop(target)
        pos_idx = np.flatnonzero(y_tr == 1)
        meta = {"dataset": name, "target_ratio": ratio,
                "n_positive_train": len(pos_idx),
                "n_seeds": max(2, int(SEED_FRAC * len(pos_idx)))}

        def record(model, draw, score, fit_s):
            rows.append({**meta, "model": model, "draw": draw,
                         "fit_seconds": round(fit_s, 2), **metrics(y_te, score)})
            print(f"  {name} r={ratio:.3%} {model} d={draw}: "
                  f"pr={rows[-1]['pr_auc']:.4f} roc={rows[-1]['roc_auc']:.3f} "
                  f"({fit_s:.0f}s)", flush=True)

        # ---- จุดอ้างอิง (ไม่ขึ้นกับ draw) ----
        enc = FeatureEncoder().fit(X_tr)
        Ztr, Zte = enc.transform(X_tr), enc.transform(X_te)
        t0 = time.perf_counter()
        hg = hgb().fit(Ztr, y_tr)
        record("histgb_oracle_full", -1, hg.predict_proba(Zte)[:, 1],
               time.perf_counter() - t0)
        del hg
        s, fs = leadx_fit_score(X_tr[pos_idx.tolist()], X_tr, X_te)
        record("leadx_pu_allseeds", -1, s, fs)
        gc.collect()

        # ---- คู่เทียบข้อมูลเท่ากัน: seeds 10% ของ positives ----
        for d in range(DRAWS):
            rd = np.random.default_rng(100 * d + 7)
            seeds = rd.choice(pos_idx, size=meta["n_seeds"], replace=False)
            s, fs = leadx_fit_score(X_tr[seeds.tolist()], X_tr, X_te)
            record("leadx_pu_10pct", d, s, fs)
            y_pu = np.zeros(len(y_tr))
            y_pu[seeds] = 1.0
            t0 = time.perf_counter()
            hg = hgb(class_weight="balanced", random_state=d).fit(Ztr, y_pu)
            record("histgb_pu_naive", d, hg.predict_proba(Zte)[:, 1],
                   time.perf_counter() - t0)
            del hg
            t0 = time.perf_counter()
            lr = LogisticRegression(max_iter=300, class_weight="balanced").fit(
                Ztr, y_pu)
            record("logreg_pu_naive", d, lr.predict_proba(Zte)[:, 1],
                   time.perf_counter() - t0)
            del lr
            gc.collect()
        del Ztr, Zte
        gc.collect()
    return rows


if __name__ == "__main__":
    t0 = time.time()
    rows = run_dataset("fraud", (0.03, 0.01, 0.003, 0.001))
    rows += run_dataset("home_credit", (0.03, 0.01, 0.003))
    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/imbalance_focus_raw.csv")
    med = (raw.group_by(["dataset", "target_ratio", "model"])
           .agg(pr=pl.col("pr_auc").median(), roc=pl.col("roc_auc").median(),
                lift01=pl.col("lift_at_0_1pct").median())
           .sort(["dataset", "target_ratio", "model"], descending=[False, True, False]))
    print("\n=== median across draws ===")
    with pl.Config(tbl_rows=60):
        print(med)
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min → outputs/imbalance_focus_raw.csv")
