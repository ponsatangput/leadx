"""n_bits ที่ดีที่สุดขึ้นกับจำนวน seeds หรือเปล่า — default อาจถูกจูนผิด regime

**ที่มา:** seed_count_curve พบว่า precision/variance ดีขึ้นตาม seeds จนถึงราว
**1.5 seeds/bucket แล้วแบน** (1500->4000 seeds: PR +3.2%, precision@0.1% ไม่ขยับ)
ถ้าจุดอิ่มตัวถูกกำหนดโดย *seeds ต่อ bucket* จริง แสดงว่าเมื่อมี seeds เยอะกว่านั้น
เราควร **เพิ่ม n_bits** เพื่อแลกหลักฐานส่วนเกินเป็น resolution — ไม่ใช่ปล่อยทิ้ง

เลขที่เกี่ยวข้อง (n_bits -> buckets/table):  8->256  10->1024  12->4096

    n_seeds=150  : 0.59 / 0.15 / 0.04  seeds ต่อ bucket
    n_seeds=4000 : 15.6 / 3.9  / 0.98  seeds ต่อ bucket

default ปัจจุบันคือ n_bits=10 ซึ่งถูกเลือกตอน benchmark ใช้ seeds 20-61 ตัว
โจทย์จริง (PromptPay, 30k confirmed) อยู่คนละที่บนแกนนี้โดยสิ้นเชิง

---------------------------------------------------------------------------
PRE-REGISTERED
---------------------------------------------------------------------------
H : n_bits ที่ดีที่สุดเลื่อนขึ้นตามจำนวน seeds (เพราะสิ่งที่กำกับคือ seeds/bucket)
ทำนาย 1 : ที่ n_seeds=4000  -> n_bits=12 ชนะ n_bits=10 (precision@0.1%, PR-AUC)
ทำนาย 2 : ที่ n_seeds=150   -> n_bits=12 **แพ้** n_bits=10  (crossover)
          ทำนาย 2 สำคัญกว่า — ถ้าไม่มี crossover แปลว่า "bits เยอะกว่าดีกว่าเสมอ"
          ซึ่งเป็นคนละข้อสรุปและ default ก็แค่ตั้งต่ำไป
หักล้าง : ไม่มี interaction ระหว่าง n_seeds กับ n_bits (อันดับของ n_bits เหมือนกัน
          ทั้งสอง seed level) -> H ผิด, จูน n_bits แยกจาก seeds ได้ตามเดิม

ตัดสินด้วย **paired ต่อ rep** (rep เดียวกัน = seed draw + random_state เดียวกัน)
เทียบภายใน seed level; 5 reps ต่อ cell. ไม่มี pass/fail gate ตายตัวเพราะเป็นการ
วัด interaction — รายงาน effect size พร้อม CV เป็นกรอบ noise (จาก seed_count_curve:
CV ~10% ที่ 150 seeds, ~1% ที่ 4000)

หมายเหตุ: rpforest จำกัด n_bits <= FOREST_MAX_DEPTH (12) จึงไม่ทดสอบเกินนั้น

รัน: PYTHONPATH=outputs uv run python outputs/bits_vs_seeds_experiment.py
"""
import gc
import time

import numpy as np
import polars as pl

from extreme_imbalance_benchmark import keep_train_valid_columns, metrics
from leadx import LeadX
from real_data_benchmark import load_dataset, split_data

N_REPS = 5
SEED_LEVELS = (150, 4000)
BITS = (8, 10, 12)
BASE = dict(mode="pu", tune=False, n_tables=80, score_type="lift", agg="log",
            engine="rpforest", feature_weighting="contrast", hier_depths="auto")


def main() -> None:
    df, target, split_type, max_rows = load_dataset("fraud")
    train, test = split_data(df, target, split_type, max_rows)
    train, test = keep_train_valid_columns(train, test, target)
    y_tr = train[target].to_numpy().astype(np.float64)
    y_te = test[target].to_numpy().astype(np.float64)
    X_tr, X_te = train.drop(target), test.drop(target)
    pos = np.flatnonzero(y_tr == 1)
    print(f"fraud natural: bg={X_tr.height:,} positives={len(pos):,} "
          f"reps={N_REPS}\n", flush=True)

    rows = []
    for n_seeds in SEED_LEVELS:
        for rep in range(N_REPS):
            # seed draw + random_state ตรงกับ seed_count_curve เป๊ะ -> เทียบข้ามได้
            sd = np.random.default_rng(2000 + 17 * rep).choice(
                pos, size=n_seeds, replace=False)
            for nb in BITS:
                t0 = time.perf_counter()
                clf = LeadX(**BASE, n_bits=nb,
                            random_state=rep).fit(X_tr[sd.tolist()], background=X_tr)
                m = metrics(y_te, clf.score_samples(X_te))
                rows.append({"n_seeds": n_seeds, "n_bits": nb, "rep": rep,
                             "seeds_per_bucket": round(n_seeds / 2 ** nb, 3),
                             "fit_seconds": round(time.perf_counter() - t0, 1), **m})
                print(f"  seeds={n_seeds:>5} bits={nb:>2} "
                      f"({n_seeds/2**nb:>6.2f}/bucket) rep={rep}: "
                      f"pr={m['pr_auc']:.4f} p@0.1%={m['precision_at_0_1pct']:.3f} "
                      f"roc={m['roc_auc']:.3f}", flush=True)
                del clf
                gc.collect()

    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/bits_vs_seeds_raw.csv")
    cv = lambda c: (100 * pl.col(c).std() / pl.col(c).mean())      # noqa: E731
    print("\n=== n_bits x n_seeds ===")
    print(raw.group_by(["n_seeds", "n_bits"]).agg(
        per_bucket=pl.col("seeds_per_bucket").first(),
        pr=pl.col("pr_auc").median().round(4), cv_pr=cv("pr_auc").round(2),
        p01=pl.col("precision_at_0_1pct").median().round(4),
        roc=pl.col("roc_auc").median().round(3),
    ).sort(["n_seeds", "n_bits"]))

    print("\n=== paired เทียบ n_bits=10 (ภายใน rep เดียวกัน) ===")
    base = (raw.filter(pl.col("n_bits") == 10)
            .select("n_seeds", "rep", pl.col("pr_auc").alias("p0"),
                    pl.col("precision_at_0_1pct").alias("q0")))
    j = (raw.filter(pl.col("n_bits") != 10).join(base, on=["n_seeds", "rep"])
         .with_columns(
             (100 * (pl.col("pr_auc") - pl.col("p0")) / pl.col("p0")).alias("d_pr"),
             (100 * (pl.col("precision_at_0_1pct") - pl.col("q0"))
              / pl.col("q0")).alias("d_p01")))
    print(j.group_by(["n_seeds", "n_bits"]).agg(
        n=pl.len(), wins=(pl.col("pr_auc") > pl.col("p0")).sum(),
        d_pr=pl.col("d_pr").median().round(2),
        d_p01=pl.col("d_p01").median().round(2)).sort(["n_seeds", "n_bits"]))
    print("\nH: อันดับของ n_bits ต้อง *สลับ* ระหว่าง seeds=150 กับ seeds=4000")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n({time.time()-t0:.0f}s)")
