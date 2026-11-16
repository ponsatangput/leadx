"""ประมาณ density ratio โดยตรงบน LSH buckets — แทนการหาร density สองตัว

**ทฤษฎี:** `lift` ของ LeadX คืออัตราส่วนของ KDE สองตัวที่ประมาณด้วย LSH (RACE /
Hashing-Based Estimators): ต่อ bucket คำนวณ (n_seed + a) / (E_bg + a) แล้วเฉลี่ย
log ข้าม tables ด้วยน้ำหนักเท่ากัน — โครงสร้างแบบ Naive Bayes ที่ถือว่า tables
อิสระกัน Sugiyama และคณะแสดงว่าการหารด้วย density ที่ประมาณมาขยาย error และการ
ประมาณ ratio โดยตรงแม่นกว่า

**วิธี:** ใช้ bucket ทุกอันของทุก table เป็น one-hot แบบ sparse (80 tables x
<=1024 = <=81,920 คอลัมน์) แล้ว fit logistic regression ระหว่าง seeds (s=1) กับ
background ทั้งก้อน (s=0) — ภายใต้ SCAR, P(s=1|x) เป็น monotone กับ P(y=1|x)
(Elkan & Noto 2008) ranking จึงถูกต้อง และ logit ของ LR คือ log density ratio
ที่ fit แบบ discriminative โดยตรง = โครงสร้างเดียวกับ Facebook 2014 (GBDT leaf +
LR) แต่ใช้ LSH bucket แทน leaf ยังเป็น PU ล้วน (seeds + background เท่านั้น)

สิ่งที่ LR ทำได้แต่ lift ทำไม่ได้: ให้น้ำหนักต่าง table/bucket และชดเชยความซ้ำซ้อน
ระหว่าง tables (ซึ่ง lift ถือว่าอิสระ)

---------------------------------------------------------------------------
PRE-REGISTERED
---------------------------------------------------------------------------
Cell    : fraud @ 0.3%, 61 seeds, rare_pu geometry (rpforest b10 x 80, contrast,
          hier auto) — 6 reps, seed draw + random_state ตรงกับ tables_capacity_ultra
Variants: base       lift ของ LeadX
          lr_c01     L2 logistic, C=0.1, class_weight="balanced"   <- PRIMARY
          lr_c1      L2 logistic, C=1.0, class_weight="balanced"   (รายงานเท่านั้น)
          blend      rank-average ของ lift กับ lr_c01               (รายงานเท่านั้น)
          C ถูกตรึงล่วงหน้า ไม่เลือกจาก test (เลือกจาก test = leak). C=0.1 เป็น primary
          เพราะมี positive 61 ตัวต่อ ~82k features จึงต้อง regularize แรง
Primary : pr_auc paired vs base
GATE    : lr_c01 -> median paired delta pr_auc > +3% AND ชนะ >= 5/6
          AND median paired delta precision@0.1% >= 0 (ห้ามแลกหัวแถว)
หักล้าง : ไม่ผ่าน -> การหารต่อ bucket ไม่ใช่ตัวจำกัด หรือ 61 positives น้อยเกินกว่าจะ
          fit น้ำหนักแบบ discriminative ได้ดีกว่า Naive Bayes (ซึ่งเป็นผลที่รู้กันว่า
          NB ชนะ LR เมื่อข้อมูลน้อย — Ng & Jordan 2001) -> ทดสอบซ้ำเมื่อ seeds เยอะ

รัน: PYTHONPATH=outputs uv run python outputs/direct_ratio_experiment.py
"""
import gc
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from extreme_imbalance_benchmark import keep_train_valid_columns, metrics, sample_ratio
from leadx import LeadX
from real_data_benchmark import load_dataset, split_data

N_REPS = 6
RATIO = 0.003
BASE = dict(mode="pu", tune=False, n_bits=10, n_tables=80, score_type="lift",
            agg="log", engine="rpforest", feature_weighting="contrast",
            hier_depths="auto")


def bucket_onehot(codes_tr: np.ndarray, codes_te: np.ndarray):
    """(table, code) -> คอลัมน์ sparse; bucket ที่ไม่เคยเห็นใน train ถูกทิ้ง"""
    n_tr, n_tab = codes_tr.shape
    n_te = codes_te.shape[0]
    rows_tr, cols_tr, rows_te, cols_te = [], [], [], []
    offset = 0
    for k in range(n_tab):
        uniq, inv = np.unique(codes_tr[:, k], return_inverse=True)
        rows_tr.append(np.arange(n_tr)); cols_tr.append(inv + offset)
        pos = np.searchsorted(uniq, codes_te[:, k])
        pos_c = np.minimum(pos, len(uniq) - 1)
        hit = uniq[pos_c] == codes_te[:, k]
        rows_te.append(np.flatnonzero(hit)); cols_te.append(pos_c[hit] + offset)
        offset += len(uniq)
    def build(r, c, n):
        r, c = np.concatenate(r), np.concatenate(c)
        return sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)),
                             shape=(n, offset))
    return build(rows_tr, cols_tr, n_tr), build(rows_te, cols_te, n_te), offset


def rank01(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x, kind="stable"), kind="stable") / max(1, len(x) - 1)


def main() -> None:
    df, target, split_type, max_rows = load_dataset("fraud")
    trf, tef = split_data(df, target, split_type, max_rows)
    train = sample_ratio(trf, target, RATIO, 42)
    test = sample_ratio(tef, target, RATIO, 43)
    train, test = keep_train_valid_columns(train, test, target)
    y_tr = train[target].to_numpy().astype(np.float64)
    y_te = test[target].to_numpy().astype(np.float64)
    X_tr, X_te = train.drop(target), test.drop(target)
    pos = np.flatnonzero(y_tr == 1)
    n_seeds = max(2, int(0.10 * len(pos)))
    print(f"fraud @ {RATIO:.1%} bg={X_tr.height:,} test={len(y_te):,} "
          f"seeds={n_seeds} reps={N_REPS}\n", flush=True)

    rows = []
    for rep in range(N_REPS):
        t0 = time.perf_counter()
        sd = np.random.default_rng(100 * rep + 7).choice(pos, size=n_seeds, replace=False)
        model = LeadX(**BASE, random_state=rep).fit(X_tr[sd.tolist()], background=X_tr)
        lift = model.score_samples(X_te)
        B_tr, B_te, n_cols = bucket_onehot(model.transform_codes(X_tr),
                                           model.transform_codes(X_te))
        s = np.zeros(X_tr.height)
        s[sd] = 1.0                                  # PU: seeds=1, unlabeled=0
        scores = {"base": lift}
        for name, C in (("lr_c01", 0.1), ("lr_c1", 1.0)):
            lr = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs",
                                    max_iter=2000)
            lr.fit(B_tr, s)
            scores[name] = lr.decision_function(B_te)
        scores["blend"] = rank01(lift) + rank01(scores["lr_c01"])
        for name, sc in scores.items():
            rows.append({"rep": rep, "variant": name, "n_bucket_cols": n_cols,
                         **metrics(y_te, sc)})
        dt = time.perf_counter() - t0
        print(f"  rep={rep} cols={n_cols:,} | " + " | ".join(
            f"{r['variant']} pr={r['pr_auc']:.4f} p01={r['precision_at_0_1pct']:.3f}"
            for r in rows[-4:]) + f" ({dt:.0f}s)", flush=True)
        del model, B_tr, B_te
        gc.collect()

    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/direct_ratio_raw.csv")
    cv = lambda c: (100 * pl.col(c).std() / pl.col(c).mean())   # noqa: E731
    print("\n=== ระดับ ===")
    print(raw.group_by("variant").agg(
        pr=pl.col("pr_auc").median().round(4), cv_pr=cv("pr_auc").round(2),
        p01=pl.col("precision_at_0_1pct").median().round(4),
        p1=pl.col("precision_at_1pct").median().round(4),
        roc=pl.col("roc_auc").median().round(3)).sort("variant"))
    base = raw.filter(pl.col("variant") == "base").select(
        "rep", pl.col("pr_auc").alias("a0"), pl.col("precision_at_0_1pct").alias("z0"),
        pl.col("precision_at_1pct").alias("q0"))
    j = (raw.filter(pl.col("variant") != "base").join(base, on="rep")
         .with_columns(
             (100 * (pl.col("pr_auc") - pl.col("a0")) / pl.col("a0")).alias("dpr"),
             (100 * (pl.col("precision_at_0_1pct") - pl.col("z0"))
              / pl.col("z0")).alias("d01"),
             (100 * (pl.col("precision_at_1pct") - pl.col("q0")) / pl.col("q0")).alias("d1")))
    print("\n=== paired vs base ===")
    print(j.group_by("variant").agg(
        wins=(pl.col("pr_auc") > pl.col("a0")).sum(),
        d_pr=pl.col("dpr").median().round(2),
        d_p01=pl.col("d01").median().round(2),
        d_p1=pl.col("d1").median().round(2))
        .with_columns(gate=(pl.col("variant") == "lr_c01") & (pl.col("d_pr") > 3)
                      & (pl.col("wins") >= 5) & (pl.col("d_p01") >= 0))
        .sort("variant"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n({time.time()-t0:.0f}s)")
