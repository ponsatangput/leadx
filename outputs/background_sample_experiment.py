"""background ต้องใหญ่แค่ไหน — ตอบโจทย์ 500M: fit บน sample แล้ว score ทั้งชุด

ทำไมการ sample background ถึงถูกต้องทางสถิติตั้งแต่แรก: สูตร lift ใน scoring.py
ใช้ background เป็น **สัดส่วนมวล** ไม่ใช่ count ดิบ

    lift(b) = (n_seed_b + alpha) / (n_seed_total * n_total_b / n_bg_total + alpha)
                                                   ^^^^^^^^^^^^^^^^^^^^^^
uniform sample ประมาณ n_total_b / n_bg_total ได้แบบ unbiased — ที่เสียคือ
*ความแปรปรวน* ของ bucket ที่มีมวลน้อย (และ geometry/encoder stats ที่ fit จาก
background ตาม parity contract) คำถามคือเสียเท่าไหร่ และพอที่เท่าไหร่

---------------------------------------------------------------------------
PRE-REGISTERED (commit ก่อนรัน full) — นี่คือ EQUIVALENCE test ไม่ใช่ superiority
---------------------------------------------------------------------------
Primary : pr_auc, paired ต่อ (dataset, ratio, draw) เทียบ frac=1.0 (background เต็ม)
Cells   : fraud @ 0.3%, 0.1% ; home_credit @ 0.3% x 3 draws = 9 paired obs
          (draw seed rng(100*d+7) ตรงกับ imbalance_focus_rpf.py และ
          feature_selection_experiment.py — เทียบข้ามการทดลองได้)
Fractions: 0.01, 0.025, 0.05, 0.10, 0.25, 0.50 (1.0 = baseline)

ADEQUATE(f) = median paired delta pr_auc > -5%  AND  จำนวน draw ที่แย่กว่า -15% <= 1
รายงาน f ที่เล็กที่สุดที่ adequate พร้อม **จำนวนแถว background สัมบูรณ์** ที่ f นั้น
— ตัวเลขนั้นคือสิ่งที่เอาไปใช้จริง (ต้องการ N แถว ไม่ใช่ N%)

หมายเหตุ: seeds ใช้ชุดเดิมทุก fraction (production มี confirmed fraud ครบอยู่แล้ว
ที่ต้อง sample คือประชากร unlabeled)

รัน: PYTHONPATH=outputs uv run python outputs/background_sample_experiment.py
"""
import gc
import sys
import time

import numpy as np
import polars as pl

from extreme_imbalance_benchmark import keep_train_valid_columns, metrics, sample_ratio
from leadx import LeadX
from real_data_benchmark import load_dataset, split_data

DRAWS = 3
SEED_FRAC = 0.10
FRACTIONS = (0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.0)
MIN_BG = 2_000            # ต่ำกว่านี้ encoder/geometry ไม่มีข้อมูลพอ — ข้าม cell
RPF_KW = dict(mode="pu", tune=False, n_bits=10, n_tables=80, score_type="lift",
              agg="log", feature_weighting="contrast", engine="rpforest",
              hier_depths="auto", random_state=0)


def run_dataset(name, ratios, rows):
    df, target, split_type, max_rows = load_dataset(name)
    train_full, test_full = split_data(df, target, split_type, max_rows)
    for ratio in ratios:
        train = sample_ratio(train_full, target, ratio, 42)
        test = sample_ratio(test_full, target, ratio, 43)
        train, test = keep_train_valid_columns(train, test, target)
        y_tr = train[target].to_numpy().astype(np.float64)
        y_te = test[target].to_numpy().astype(np.float64)
        X_tr, X_te = train.drop(target), test.drop(target)
        pos_idx = np.flatnonzero(y_tr == 1)
        n_seeds = max(2, int(SEED_FRAC * len(pos_idx)))
        n_tr = X_tr.height
        print(f"  [{name} r={ratio:.3%}] bg_full={n_tr:,} pos={len(pos_idx)} "
              f"seeds={n_seeds}", flush=True)

        for d in range(DRAWS):
            rd = np.random.default_rng(100 * d + 7)
            seeds = rd.choice(pos_idx, size=n_seeds, replace=False)
            for frac in FRACTIONS:
                n_bg = n_tr if frac == 1.0 else int(round(frac * n_tr))
                if n_bg < MIN_BG:
                    print(f"    skip frac={frac} (n_bg={n_bg} < {MIN_BG})", flush=True)
                    continue
                if frac == 1.0:
                    bg = X_tr
                else:
                    # sample แยกต่อ draw (rng ต่างจาก seed draw) — วัดความแปรปรวนจริง
                    rb = np.random.default_rng(9000 + 100 * d + int(frac * 1000))
                    bg = X_tr[rb.choice(n_tr, size=n_bg, replace=False).tolist()]
                t0 = time.perf_counter()
                clf = LeadX(**RPF_KW).fit(X_tr[seeds.tolist()], background=bg)
                score = clf.score_samples(X_te)
                fit_s = time.perf_counter() - t0
                rows.append({"dataset": name, "target_ratio": ratio, "draw": d,
                             "bg_frac": frac, "n_bg": n_bg, "n_seeds": n_seeds,
                             "fit_seconds": round(fit_s, 2),
                             **metrics(y_te, score)})
                print(f"    {name} r={ratio:.3%} frac={frac:<6} n_bg={n_bg:>7,} d={d}: "
                      f"pr={rows[-1]['pr_auc']:.4f} roc={rows[-1]['roc_auc']:.3f} "
                      f"({fit_s:.0f}s)", flush=True)
                del clf, bg
                gc.collect()
    return rows


def summarize(raw: pl.DataFrame) -> pl.DataFrame:
    base = (raw.filter(pl.col("bg_frac") == 1.0)
            .select("dataset", "target_ratio", "draw",
                    pl.col("pr_auc").alias("pr_base"),
                    pl.col("fit_seconds").alias("fit_base")))
    j = (raw.filter(pl.col("bg_frac") != 1.0)
         .join(base, on=["dataset", "target_ratio", "draw"], how="inner")
         .with_columns(
             (100 * (pl.col("pr_auc") - pl.col("pr_base")) / pl.col("pr_base")).alias("d"),
             (pl.col("fit_seconds") / pl.col("fit_base")).alias("speedup")))
    return (j.group_by("bg_frac").agg(
        n=pl.len(), median_d=pl.col("d").median().round(2),
        worst=pl.col("d").min().round(2), best=pl.col("d").max().round(2),
        bad=(pl.col("d") < -15).sum(), med_n_bg=pl.col("n_bg").median().cast(pl.Int64),
        fit_ratio=pl.col("speedup").median().round(2))
        .with_columns(adequate=(pl.col("median_d") > -5) & (pl.col("bad") <= 1))
        .sort("bg_frac"))


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    rows: list[dict] = []
    if smoke:
        globals()["DRAWS"] = 1
        globals()["FRACTIONS"] = (0.05, 1.0)
        run_dataset("fraud", (0.01,), rows)
    else:
        run_dataset("fraud", (0.003, 0.001), rows)
        run_dataset("home_credit", (0.003,), rows)
    raw = pl.DataFrame(rows)
    out = ("outputs/background_sample_smoke.csv" if smoke
           else "outputs/background_sample_raw.csv")
    raw.write_csv(out)
    print(f"\nwrote {out}  ({time.time()-t0:.0f}s total)")
    if not smoke:
        print("\n=== paired vs background เต็ม (primary: pr_auc) ===")
        print(summarize(raw))
