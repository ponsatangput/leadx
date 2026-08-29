"""noise-floor control สำหรับ background_sample_experiment — แยก "sample เล็กไป"
ออกจาก "geometry คนละ realization"

**ทำไมต้องมี:** background_sample_experiment รอบแรกให้ผลที่ไม่ monotone กับ
ขนาด sample เลย (fraud@0.1%: -4.9, +1.9, +14.1, -13.1, +20.2, -1.2 ตาม frac
ที่เพิ่มขึ้น) ซึ่งเป็นไปไม่ได้ถ้าขนาด sample เป็นตัวขับ — แปลว่ามี confound

confound ที่ระบุได้: RPForest fit **บน background** (parity contract) ดังนั้นการ
เปลี่ยน background sample = เปลี่ยน geometry realization ไปด้วย การทดลองนั้นจึงวัด
sample-size effect + geometry-seed noise ปนกัน โดยไม่มีทางแยก

**การทดลองนี้วัด noise floor ตรงๆ** บน cell ที่ baseline เสถียรที่สุด
(fraud @ 0.3% — baseline pr_auc 0.064/0.059/0.050 ข้าม draw):

  A. full   : background เต็ม, random_state 0..5      -> geometry noise ล้วน (ไม่ sample)
  B. sampled: frac=0.05, background seed 6 แบบ, rs=0  -> sampling + geometry noise

ตีความ:
  spread(B) ~ spread(A)  -> การ sample ไม่เพิ่ม noise ที่วัดได้ ที่ frac นี้
  spread(B) >> spread(A) -> sampling ทำร้ายจริง

รัน: PYTHONPATH=outputs uv run python outputs/background_sample_control.py
"""
import gc
import time

import numpy as np
import polars as pl

from extreme_imbalance_benchmark import keep_train_valid_columns, metrics, sample_ratio
from leadx import LeadX
from real_data_benchmark import load_dataset, split_data

N_REPS = 6
FRAC = 0.05
RATIO = 0.003
BASE_KW = dict(mode="pu", tune=False, n_bits=10, n_tables=80, score_type="lift",
               agg="log", feature_weighting="contrast", engine="rpforest",
               hier_depths="auto")


def main() -> None:
    df, target, split_type, max_rows = load_dataset("fraud")
    train_full, test_full = split_data(df, target, split_type, max_rows)
    train = sample_ratio(train_full, target, RATIO, 42)
    test = sample_ratio(test_full, target, RATIO, 43)
    train, test = keep_train_valid_columns(train, test, target)
    y_tr = train[target].to_numpy().astype(np.float64)
    y_te = test[target].to_numpy().astype(np.float64)
    X_tr, X_te = train.drop(target), test.drop(target)
    pos_idx = np.flatnonzero(y_tr == 1)
    n_seeds = max(2, int(0.10 * len(pos_idx)))
    # seeds ตรึงชุดเดียวทุก rep (draw 0 ของการทดลองหลัก) — ตัดตัวแปรนี้ออก
    seeds = np.random.default_rng(7).choice(pos_idx, size=n_seeds, replace=False)
    n_tr, n_bg = X_tr.height, int(round(FRAC * X_tr.height))
    print(f"fraud @ {RATIO:.1%}  bg_full={n_tr:,}  bg_sampled={n_bg:,}  "
          f"seeds={n_seeds}  reps={N_REPS}\n", flush=True)

    rows = []
    for rep in range(N_REPS):
        for arm in ("full", "sampled"):
            if arm == "full":
                bg, rs = X_tr, rep            # geometry เปลี่ยน, ข้อมูลเหมือนเดิม
            else:
                rb = np.random.default_rng(4000 + rep)
                bg = X_tr[rb.choice(n_tr, size=n_bg, replace=False).tolist()]
                rs = 0                        # geometry seed ตรึง, ข้อมูลเปลี่ยน
            t0 = time.perf_counter()
            clf = LeadX(**BASE_KW, random_state=rs).fit(X_tr[seeds.tolist()],
                                                        background=bg)
            m = metrics(y_te, clf.score_samples(X_te))
            rows.append({"arm": arm, "rep": rep, "n_bg": bg.height,
                         "fit_seconds": round(time.perf_counter() - t0, 1), **m})
            print(f"  {arm:<8} rep={rep}: pr={m['pr_auc']:.4f} roc={m['roc_auc']:.3f} "
                  f"({rows[-1]['fit_seconds']:.0f}s)", flush=True)
            del clf, bg
            gc.collect()

    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/background_sample_control_raw.csv")
    print("\n=== noise floor (pr_auc) ===")
    s = (raw.group_by("arm").agg(
        n=pl.len(), median=pl.col("pr_auc").median(),
        lo=pl.col("pr_auc").min(), hi=pl.col("pr_auc").max(),
        cv_pct=(100 * pl.col("pr_auc").std() / pl.col("pr_auc").mean()),
        fit_s=pl.col("fit_seconds").median())
        .with_columns(spread_pct=(100 * (pl.col("hi") - pl.col("lo")) / pl.col("median")))
        .sort("arm"))
    print(s)
    a = raw.filter(pl.col("arm") == "full")["pr_auc"]
    b = raw.filter(pl.col("arm") == "sampled")["pr_auc"]
    print(f"\nmedian sampled vs full: {100*(b.median()-a.median())/a.median():+.1f}%")
    print(f"geometry-only noise (full, min..max): "
          f"{100*(a.max()-a.min())/a.median():.1f}% ของ median")
    print(f"sampling+geometry noise (sampled):    "
          f"{100*(b.max()-b.min())/b.median():.1f}% ของ median")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n({time.time()-t0:.0f}s)")
