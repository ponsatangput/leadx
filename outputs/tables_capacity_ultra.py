"""n_tables > 80 ที่ ultra-imbalance — หลักฐานบวกต่อแถวบางเกินไปหรือเปล่า

**ที่มา (วัดได้ 2026-11-16):** ที่ fraud @ 0.3%, 61 seeds, rpforest b10 x 80 tables
`miss_rate_join = 0.95` -> แต่ละแถวตกใน bucket ที่ **มี seed อยู่จริงแค่ ~4 จาก
80 tables** อีก ~76 tables ให้แค่ค่าปรับตามมวล (n_pos=0) ไม่ใช่หลักฐานบวก
คะแนนที่หัวแถวจึงถูกขับด้วย table จำนวนหยิบมือ ซึ่งอธิบาย geometry CV 8% ได้
(ensemble ที่มีผลจริงเล็กกว่า 80 มาก)

capacity sweep เดิม (`vector_capacity_sweep.py`) กวาดแค่ 10/20/40 tables ที่
natural imbalance — **ยังไม่เคยวัด > 80 และไม่เคยวัดที่ ultra**

---------------------------------------------------------------------------
PRE-REGISTERED
---------------------------------------------------------------------------
Cell    : fraud @ 0.3% (baseline เสถียรที่สุดใน ultra regime), 6 reps
          แต่ละ rep เปลี่ยนทั้ง seed draw และ random_state; paired ภายใน rep
Arms    : n_tables = 80 (default), 160, 320 — config อื่นตรึงตาม rare_pu
Primary : pr_auc paired vs 80
GATE    : 320 vs 80 -> median paired delta > +3% AND ชนะ >= 5/6
Secondary (ทำนาย ไม่ใช่ gate): CV ของ pr_auc ลดลงเมื่อ tables เพิ่ม (~1/sqrt)
          และ precision@0.1% ไม่แย่ลง
หักล้าง : ไม่ผ่าน gate -> หลักฐานบางต่อแถวไม่ใช่ตัวจำกัด (หรือ tables ใน rpforest
          correlated กันจนเพิ่มแล้วไม่ได้ข้อมูลใหม่) -> ไม่เปลี่ยน default

รัน: PYTHONPATH=outputs uv run python outputs/tables_capacity_ultra.py
"""
import gc
import time

import numpy as np
import polars as pl

from extreme_imbalance_benchmark import keep_train_valid_columns, metrics, sample_ratio
from leadx import LeadX
from real_data_benchmark import load_dataset, split_data

N_REPS = 6
RATIO = 0.003
TABLES = (80, 160, 320)
BASE = dict(mode="pu", tune=False, n_bits=10, score_type="lift", agg="log",
            engine="rpforest", feature_weighting="contrast", hier_depths="auto")


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
    print(f"fraud @ {RATIO:.1%} bg={X_tr.height:,} seeds={n_seeds} reps={N_REPS}\n",
          flush=True)
    rows = []
    for rep in range(N_REPS):
        sd = np.random.default_rng(100 * rep + 7).choice(pos, size=n_seeds,
                                                         replace=False)
        for nt in TABLES:
            t0 = time.perf_counter()
            clf = LeadX(**BASE, n_tables=nt, random_state=rep).fit(
                X_tr[sd.tolist()], background=X_tr)
            m = metrics(y_te, clf.score_samples(X_te))
            rows.append({"n_tables": nt, "rep": rep,
                         "miss_rate_join": clf.diagnostics_["scoring"]["miss_rate_join"],
                         "fit_seconds": round(time.perf_counter() - t0, 1), **m})
            print(f"  tables={nt:>3} rep={rep}: pr={m['pr_auc']:.4f} "
                  f"p@0.1%={m['precision_at_0_1pct']:.4f} roc={m['roc_auc']:.3f} "
                  f"({rows[-1]['fit_seconds']:.0f}s)", flush=True)
            del clf
            gc.collect()

    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/tables_capacity_ultra_raw.csv")
    cv = lambda c: (100 * pl.col(c).std() / pl.col(c).mean())   # noqa: E731
    print("\n=== ระดับ ===")
    print(raw.group_by("n_tables").agg(
        pr=pl.col("pr_auc").median().round(4), cv_pr=cv("pr_auc").round(2),
        p01=pl.col("precision_at_0_1pct").median().round(4),
        roc=pl.col("roc_auc").median().round(3),
        fit_s=pl.col("fit_seconds").median()).sort("n_tables"))
    base = raw.filter(pl.col("n_tables") == 80).select(
        "rep", pl.col("pr_auc").alias("p0"), pl.col("precision_at_0_1pct").alias("q0"))
    j = (raw.filter(pl.col("n_tables") != 80).join(base, on="rep")
         .with_columns((100 * (pl.col("pr_auc") - pl.col("p0")) / pl.col("p0")).alias("d"),
                       (100 * (pl.col("precision_at_0_1pct") - pl.col("q0"))
                        / pl.col("q0")).alias("d01")))
    print("\n=== paired vs 80 tables ===")
    print(j.group_by("n_tables").agg(
        wins=(pl.col("pr_auc") > pl.col("p0")).sum(), d_pr=pl.col("d").median().round(2),
        wins01=(pl.col("precision_at_0_1pct") > pl.col("q0")).sum(),
        d_p01=pl.col("d01").median().round(2))
        .with_columns(gate=(pl.col("d_pr") > 3) & (pl.col("wins") >= 5))
        .sort("n_tables"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n({time.time()-t0:.0f}s)")
