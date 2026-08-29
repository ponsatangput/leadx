"""hyperplane ชนะ rpforest ที่ ultra-imbalance ด้วยหรือเปล่า — ตัวตัดสิน default

**confound ที่ต้องตัด:** bits_vs_seeds / bits_turnover รันที่ natural imbalance
(3.5%) เพราะต้องมี positives พอไปถึง 4000 seeds ผลที่ได้คือ hyperplane@16-20 ชนะ
rpforest@12 บน precision@0.1% ทุกช่อง (+7.9 ถึง +14.9%, 4-5/5) และเร็วกว่า 3.4x

แต่ profile "rare_pu" ตั้ง engine="rpforest" เพราะ **§8f validate ที่ ultra-imbalance**
ยังไม่มีหลักฐานว่า rpforest ผิดใน regime ที่มันถูกเลือกมา — ต่างกันสองแกน
(imbalance และ seeds) ในการทดลองเดียว แยกไม่ออก

การทดลองนี้ **ตรึง seeds ให้น้อยตามปกติของ ultra regime (10% ของ positives) แล้ว
เปลี่ยนแค่ imbalance** -> ถ้าอันดับ engine พลิกกลับเป็น rpforest ที่ ultra แสดงว่า
ตัวแปรคือ imbalance และ default เดิมถูกอยู่แล้วสำหรับสิ่งที่มันเล็งไว้

---------------------------------------------------------------------------
PRE-REGISTERED
---------------------------------------------------------------------------
Primary : precision_at_0_1pct (ตัวชี้วัดปฏิบัติการ — review queue), paired ต่อ rep
          เทียบ rpforest@10 (default ที่ ship อยู่)
Cells   : fraud @ 0.3% (61 seeds) และ 0.1% (20 seeds) x 5 reps
Configs : rpforest@10 (default), rpforest@12, hyperplane@16, hyperplane@20

ทำนาย (จากผล natural) : hyperplane@20 ชนะ rpforest@10 บน precision@0.1% >= 4/5
                        ทั้งสอง ratio
หักล้าง               : rpforest ชนะที่ ultra -> ตัวแปรคือ imbalance ไม่ใช่ engine
                        เก็บ default เดิม และจำกัดข้อสรุป natural ไว้เท่านั้น

รายงาน PR-AUC เป็น secondary — ผลที่ natural แสดงว่า PR-AUC กับ precision@0.1%
ชี้คนละทางได้ (hyperplane@12: PR -2.5% แต่ p@0.1% +7.9%) จึงต้องดูทั้งคู่

noise: จาก background_sample_control CV ~8% ที่ cell 0.3% -> effect < ~8% อ่านจาก
median อย่างเดียวไม่ได้ ต้องดู wins ควบคู่

รัน: PYTHONPATH=outputs uv run python outputs/engine_at_ultra_experiment.py
"""
import gc
import time

import numpy as np
import polars as pl

from extreme_imbalance_benchmark import keep_train_valid_columns, metrics, sample_ratio
from leadx import LeadX
from real_data_benchmark import load_dataset, split_data

N_REPS = 5
RATIOS = (0.003, 0.001)
CONFIGS = (("rpforest", 10), ("rpforest", 12), ("hyperplane", 16), ("hyperplane", 20))
BASE = dict(mode="pu", tune=False, n_tables=80, score_type="lift", agg="log",
            feature_weighting="contrast", hier_depths="auto")


def main() -> None:
    df, target, split_type, max_rows = load_dataset("fraud")
    tr_full, te_full = split_data(df, target, split_type, max_rows)
    rows = []
    for ratio in RATIOS:
        train = sample_ratio(tr_full, target, ratio, 42)
        test = sample_ratio(te_full, target, ratio, 43)
        train, test = keep_train_valid_columns(train, test, target)
        y_tr = train[target].to_numpy().astype(np.float64)
        y_te = test[target].to_numpy().astype(np.float64)
        X_tr, X_te = train.drop(target), test.drop(target)
        pos = np.flatnonzero(y_tr == 1)
        n_seeds = max(2, int(0.10 * len(pos)))
        print(f"\n[fraud r={ratio:.3%}] bg={X_tr.height:,} pos={len(pos)} "
              f"seeds={n_seeds}", flush=True)
        for rep in range(N_REPS):
            sd = np.random.default_rng(100 * rep + 7).choice(pos, size=n_seeds,
                                                             replace=False)
            for eng, nb in CONFIGS:
                t0 = time.perf_counter()
                clf = LeadX(**BASE, engine=eng, n_bits=nb,
                            random_state=rep).fit(X_tr[sd.tolist()], background=X_tr)
                m = metrics(y_te, clf.score_samples(X_te))
                rows.append({"target_ratio": ratio, "engine": eng, "n_bits": nb,
                             "rep": rep, "n_seeds": n_seeds,
                             "fit_seconds": round(time.perf_counter() - t0, 1), **m})
                print(f"  r={ratio:.3%} {eng[:4]}@{nb:<2} rep={rep}: "
                      f"p@0.1%={m['precision_at_0_1pct']:.4f} "
                      f"pr={m['pr_auc']:.4f} roc={m['roc_auc']:.3f}", flush=True)
                del clf
                gc.collect()

    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/engine_at_ultra_raw.csv")
    print("\n=== ระดับ (median ต่อ 5 reps) ===")
    print(raw.group_by(["target_ratio", "engine", "n_bits"]).agg(
        p01=pl.col("precision_at_0_1pct").median().round(4),
        pr=pl.col("pr_auc").median().round(4),
        roc=pl.col("roc_auc").median().round(3),
        fit_s=pl.col("fit_seconds").median()).sort(["target_ratio", "engine", "n_bits"]))

    print("\n=== paired เทียบ rpforest@10 (default ที่ ship อยู่) ===")
    base = (raw.filter((pl.col("engine") == "rpforest") & (pl.col("n_bits") == 10))
            .select("target_ratio", "rep",
                    pl.col("precision_at_0_1pct").alias("q0"),
                    pl.col("pr_auc").alias("p0")))
    j = (raw.filter(~((pl.col("engine") == "rpforest") & (pl.col("n_bits") == 10)))
         .join(base, on=["target_ratio", "rep"])
         .with_columns(
             (100 * (pl.col("precision_at_0_1pct") - pl.col("q0"))
              / pl.col("q0")).alias("d_p01"),
             (100 * (pl.col("pr_auc") - pl.col("p0")) / pl.col("p0")).alias("d_pr")))
    print(j.group_by(["target_ratio", "engine", "n_bits"]).agg(
        n=pl.len(),
        win_p01=(pl.col("precision_at_0_1pct") > pl.col("q0")).sum(),
        d_p01=pl.col("d_p01").median().round(2),
        win_pr=(pl.col("pr_auc") > pl.col("p0")).sum(),
        d_pr=pl.col("d_pr").median().round(2),
    ).sort(["target_ratio", "engine", "n_bits"]))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n({time.time()-t0:.0f}s)")
