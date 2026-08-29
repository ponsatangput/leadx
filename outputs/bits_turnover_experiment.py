"""หา n_bits ที่พลิกกลับ — เพดาน 12 ของ rpforest กั้นคุณภาพอยู่หรือเปล่า

bits_vs_seeds พบว่า n_bits=12 ชนะ 10 ทั้งที่ 150 และ 4000 seeds (5/5 ทั้งคู่,
+2.7%/+3.9% PR-AUC) — **ไม่มี crossover** สมมติฐานเดิม (n_bits ที่ดีสุดผูกกับ
seeds/bucket) จึงถูกหักล้าง: ที่ 4000 seeds n_bits=12 คือ 0.98 seeds/bucket
บางกว่า n_bits=10 สี่เท่า แต่ยังชนะ

คำอธิบายที่เหลืออยู่: hier_depths + multi-probe + lift fallback=1 จัดการ bucket
บางไว้แล้ว -> เพิ่ม resolution ได้โดยไม่จ่ายค่าปรับ ถ้าจริง n_bits ควรไปต่อได้อีก
จนถึงจุดที่กลไกเหล่านั้นเอาไม่อยู่ **rpforest ติดเพดาน 12** จึงต้องใช้ hyperplane

---------------------------------------------------------------------------
PRE-REGISTERED
---------------------------------------------------------------------------
คำถาม : PR-AUC พลิกกลับที่ n_bits เท่าไหร่ (engine="hyperplane", ไม่มีเพดาน)
ทำนาย : มีจุดพลิกกลับที่ n_bits ระหว่าง 14-20 (เกินนั้น bucket บางจนกลไก
        multi-probe/hier เอาไม่อยู่) และจุดนั้น **ไม่ขึ้นกับจำนวน seeds**
        (ต่อเนื่องจากผลที่หักล้าง crossover ไปแล้ว)
หักล้าง : ไม่พลิกเลยถึง n_bits=20 -> เพดาน 12 ของ rpforest กั้นคุณภาพจริง และ
        ควรพิจารณา default เป็น hyperplane ที่ bits สูงสำหรับงาน seeds เยอะ

ตัดสินด้วย paired ต่อ rep เทียบ n_bits=12 (จุดที่ดีที่สุดเท่าที่รู้ตอนนี้)
5 reps, seed draw + random_state ตรงกับการทดลองก่อนหน้าทั้งหมด

รัน: PYTHONPATH=outputs uv run python outputs/bits_turnover_experiment.py
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
BITS = (10, 12, 14, 16, 20)
BASE = dict(mode="pu", tune=False, n_tables=80, score_type="lift", agg="log",
            engine="hyperplane", feature_weighting="contrast", hier_depths="auto")


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
    raw.write_csv("outputs/bits_turnover_raw.csv")
    cv = lambda c: (100 * pl.col(c).std() / pl.col(c).mean())      # noqa: E731
    print("\n=== n_bits x n_seeds ===")
    print(raw.group_by(["n_seeds", "n_bits"]).agg(
        per_bucket=pl.col("seeds_per_bucket").first(),
        pr=pl.col("pr_auc").median().round(4), cv_pr=cv("pr_auc").round(2),
        p01=pl.col("precision_at_0_1pct").median().round(4),
        roc=pl.col("roc_auc").median().round(3),
    ).sort(["n_seeds", "n_bits"]))

    print("\n=== paired เทียบ n_bits=12 (ภายใน rep เดียวกัน) ===")
    base = (raw.filter(pl.col("n_bits") == 12)
            .select("n_seeds", "rep", pl.col("pr_auc").alias("p0"),
                    pl.col("precision_at_0_1pct").alias("q0")))
    j = (raw.filter(pl.col("n_bits") != 12).join(base, on=["n_seeds", "rep"])
         .with_columns(
             (100 * (pl.col("pr_auc") - pl.col("p0")) / pl.col("p0")).alias("d_pr"),
             (100 * (pl.col("precision_at_0_1pct") - pl.col("q0"))
              / pl.col("q0")).alias("d_p01")))
    print(j.group_by(["n_seeds", "n_bits"]).agg(
        n=pl.len(), wins=(pl.col("pr_auc") > pl.col("p0")).sum(),
        d_pr=pl.col("d_pr").median().round(2),
        d_p01=pl.col("d_p01").median().round(2)).sort(["n_seeds", "n_bits"]))
    print("\nหาจุดพลิกกลับ: d_pr ต้องเปลี่ยนจากบวกเป็นลบที่ n_bits สักค่า")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n({time.time()-t0:.0f}s)")
