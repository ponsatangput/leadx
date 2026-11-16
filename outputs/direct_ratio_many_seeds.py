"""direct density-ratio (LR บน bucket) เมื่อมี seeds เยอะ — follow-up ที่ระบุไว้ล่วงหน้า

`direct_ratio_experiment.py` (61 seeds, ultra) แพ้หนัก: lr_c01 PR -41.3% (0/6) และ
C=1.0 แย่กว่า (-48.7%) -> regularize แรงยิ่งดี = อาการข้อมูลน้อยเกิน ตรงกับเงื่อนไข
หักล้างที่ pre-register ไว้: Naive Bayes (= lift ข้าม tables) ชนะ logistic regression
เมื่อข้อมูลน้อย และ LR แซงเมื่อข้อมูลมากพอ (Ng & Jordan 2001) docstring เดิมระบุให้
**ทดสอบซ้ำเมื่อ seeds เยอะ** — ซึ่งคือ regime ของโจทย์จริง (30k confirmed)

---------------------------------------------------------------------------
PRE-REGISTERED
---------------------------------------------------------------------------
Cell    : fraud natural (bg 210k), n_seeds=4000, 5 reps — seed draw rng(2000+17*rep)
          + random_state=rep ตรงกับ seed_count_curve / bits_vs_seeds
Geometry: rare_pu (rpforest b10 x 80, contrast, hier auto) เหมือนรอบแรก
Variants: base, lr_c01 (PRIMARY), lr_c1, blend — C ตรึงเหมือนรอบแรก
GATE    : lr_c01 -> median paired delta pr_auc > +3% AND ชนะ >= 4/5
          AND median paired delta precision@0.1% >= 0
ทำนาย  : ช่องว่างกับ base แคบลงมากเทียบกับ -41% ที่ 61 seeds (Ng & Jordan)
หักล้าง : ยังแพ้หนักที่ 4000 seeds -> ปัญหาไม่ใช่จำนวนข้อมูล แต่เป็นตัว formulation
          (เช่น one-hot bucket แบบแบน ทิ้งโครงสร้าง prefix/hier ที่ lift ใช้อยู่)

รัน: PYTHONPATH=outputs uv run python outputs/direct_ratio_many_seeds.py
"""
import gc
import time

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from direct_ratio_experiment import BASE, bucket_onehot, rank01
from extreme_imbalance_benchmark import keep_train_valid_columns, metrics
from leadx import LeadX
from real_data_benchmark import load_dataset, split_data

N_REPS = 5
N_SEEDS = 4000


def main() -> None:
    df, target, split_type, max_rows = load_dataset("fraud")
    train, test = split_data(df, target, split_type, max_rows)
    train, test = keep_train_valid_columns(train, test, target)
    y_tr = train[target].to_numpy().astype(np.float64)
    y_te = test[target].to_numpy().astype(np.float64)
    X_tr, X_te = train.drop(target), test.drop(target)
    pos = np.flatnonzero(y_tr == 1)
    print(f"fraud natural bg={X_tr.height:,} test={len(y_te):,} pos={len(pos):,} "
          f"seeds={N_SEEDS} reps={N_REPS}\n", flush=True)
    rows = []
    for rep in range(N_REPS):
        t0 = time.perf_counter()
        sd = np.random.default_rng(2000 + 17 * rep).choice(pos, size=N_SEEDS, replace=False)
        model = LeadX(**BASE, random_state=rep).fit(X_tr[sd.tolist()], background=X_tr)
        lift = model.score_samples(X_te)
        B_tr, B_te, n_cols = bucket_onehot(model.transform_codes(X_tr),
                                           model.transform_codes(X_te))
        s = np.zeros(X_tr.height)
        s[sd] = 1.0
        scores = {"base": lift}
        for name, C in (("lr_c01", 0.1), ("lr_c1", 1.0)):
            lr = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs",
                                    max_iter=2000).fit(B_tr, s)
            scores[name] = lr.decision_function(B_te)
        scores["blend"] = rank01(lift) + rank01(scores["lr_c01"])
        for name, sc in scores.items():
            rows.append({"rep": rep, "variant": name, **metrics(y_te, sc)})
        print(f"  rep={rep} | " + " | ".join(
            f"{r['variant']} pr={r['pr_auc']:.4f} p01={r['precision_at_0_1pct']:.3f}"
            for r in rows[-4:]) + f" ({time.perf_counter()-t0:.0f}s)", flush=True)
        del model, B_tr, B_te
        gc.collect()

    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/direct_ratio_many_seeds_raw.csv")
    print("\n=== ระดับ ===")
    print(raw.group_by("variant").agg(
        pr=pl.col("pr_auc").median().round(4),
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
                      & (pl.col("wins") >= 4) & (pl.col("d_p01") >= 0))
        .sort("variant"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n({time.time()-t0:.0f}s)")
