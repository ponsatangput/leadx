"""seed-count curve — เราตั้งใจ optimize regime ที่ถูกหรือเปล่า

**ทำไมต้องทำก่อนแก้อะไร:** benchmark ทั้งโปรเจกต์ใช้ seeds 20-61 ตัว แต่โจทย์จริง
(PromptPay) มี confirmed fraud **30,000** ที่ n_bits=10 -> 1,024 buckets/table:

    61 seeds    -> 0.06 seeds/bucket   (เกือบทุก bucket ว่าง หรือมี 1)
    30,000      -> 29   seeds/bucket   (ประมาณค่าได้เสถียร)

ต่างกัน ~500 เท่า นี่ไม่ใช่ "เหมือนกันแต่เยอะกว่า" แต่เป็นคนละ regime กลไกที่
สงสัยว่าเป็นตัวจำกัด precision@k (หัวแถวถูกขับด้วย singleton bucket ที่ n_pos_b=1
ได้ lift มหาศาลจาก noise) เป็นฟังก์ชันของ seeds/bucket ล้วนๆ — ถ้ามันหายไปเองที่
seeds เยอะ การไปสร้าง LCB/bootstrap weights จะแก้ปัญหาที่ไม่มีอยู่ในโจทย์จริง

---------------------------------------------------------------------------
PRE-REGISTERED (characterization ไม่ใช่ superiority — ไม่มี pass/fail)
---------------------------------------------------------------------------
H : noise floor (CV 8.1% ที่วัดได้จาก background_sample_control) ถูกขับด้วย
    seeds/bucket << 1
ทำนาย : CV ลดลง monotone กับ n_seeds และที่ >= 1 seed/bucket (n_seeds >= 1024)
        CV <= 3%
หักล้าง : CV ยังคง >= 6% ที่ n_seeds=4000+ -> ความแปรปรวนมาจาก geometry เอง
        ไม่ใช่ความบางของ seeds -> LCB/bootstrap weights ต้องคิดใหม่

setup: fraud, background = train เต็ม (210k แถว, natural imbalance) เพื่อให้มี
positives พอ (7,348) — **seeds ถูกแยกออกจาก imbalance โดยตั้งใจ**: ที่ ultra
imbalance ในเครื่องเดียวจะมี positives ไม่พอไปถึง 4000 seeds (ที่ 0.3% มี 610)
โจทย์จริงมีทั้งสองอย่าง (1:16,700 *และ* 30k seeds) ซึ่ง reproduce ที่นี่ไม่ได้

แต่ละ rep เปลี่ยน **ทั้ง** seed draw และ random_state -> CV ที่ได้คือความแปรปรวน
รวมแบบที่ผู้ใช้เจอจริง ไม่ใช่แยกส่วน

รัน: PYTHONPATH=outputs uv run python outputs/seed_count_curve.py
"""
import gc
import time

import numpy as np
import polars as pl

from extreme_imbalance_benchmark import keep_train_valid_columns, metrics
from leadx import LeadX
from real_data_benchmark import load_dataset, split_data

N_REPS = 5
SEED_GRID = (50, 150, 500, 1500, 4000)
N_BITS = 10
BASE = dict(mode="pu", tune=False, n_bits=N_BITS, n_tables=80, score_type="lift",
            agg="log", engine="rpforest", feature_weighting="contrast",
            hier_depths="auto")


def main() -> None:
    df, target, split_type, max_rows = load_dataset("fraud")
    train, test = split_data(df, target, split_type, max_rows)
    train, test = keep_train_valid_columns(train, test, target)
    y_tr = train[target].to_numpy().astype(np.float64)
    y_te = test[target].to_numpy().astype(np.float64)
    X_tr, X_te = train.drop(target), test.drop(target)
    pos = np.flatnonzero(y_tr == 1)
    n_buckets = 2 ** N_BITS
    print(f"fraud natural: bg={X_tr.height:,} positives={len(pos):,} "
          f"buckets/table={n_buckets} reps={N_REPS}\n", flush=True)

    rows = []
    for n_seeds in SEED_GRID:
        if n_seeds > len(pos):
            print(f"  skip n_seeds={n_seeds} (มี positives {len(pos)})")
            continue
        for rep in range(N_REPS):
            sd = np.random.default_rng(2000 + 17 * rep).choice(
                pos, size=n_seeds, replace=False)
            t0 = time.perf_counter()
            clf = LeadX(**BASE, random_state=rep).fit(X_tr[sd.tolist()],
                                                      background=X_tr)
            m = metrics(y_te, clf.score_samples(X_te))
            rows.append({"n_seeds": n_seeds,
                         "seeds_per_bucket": round(n_seeds / n_buckets, 3),
                         "rep": rep,
                         "fit_seconds": round(time.perf_counter() - t0, 1), **m})
            print(f"  n_seeds={n_seeds:>5} ({n_seeds/n_buckets:>5.2f}/bucket) "
                  f"rep={rep}: pr={m['pr_auc']:.4f} roc={m['roc_auc']:.3f} "
                  f"p@0.1%={m['precision_at_0_1pct']:.3f}", flush=True)
            del clf
            gc.collect()

    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/seed_count_curve_raw.csv")
    cv = lambda c: (100 * pl.col(c).std() / pl.col(c).mean())   # noqa: E731
    print("\n=== seed-count curve (fraud, natural imbalance) ===")
    print(raw.group_by("n_seeds").agg(
        per_bucket=pl.col("seeds_per_bucket").first(),
        pr=pl.col("pr_auc").median().round(4),
        cv_pr=cv("pr_auc").round(2),
        p01=pl.col("precision_at_0_1pct").median().round(4),
        cv_p01=cv("precision_at_0_1pct").round(2),
        roc=pl.col("roc_auc").median().round(3),
        cv_roc=cv("roc_auc").round(2),
        fit_s=pl.col("fit_seconds").median(),
    ).sort("n_seeds"))
    print("\nH ทำนาย: cv_pr ลดลง monotone และ <= 3% ที่ per_bucket >= 1")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n({time.time()-t0:.0f}s)")
