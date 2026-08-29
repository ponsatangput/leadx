"""แยกว่า noise floor 8% CV มาจากไหน — ก่อนจะไปแก้ precision/recall

**ปริศนา:** background_sample_control วัดได้ CV 8.1% จากการเปลี่ยน `random_state`
อย่างเดียว ทั้งที่ config นี้รวม **80 tables** เข้าด้วยกัน ถ้า 80 tables เป็นอิสระ
จริง ความแปรปรวนของ ensemble ควรเหลือ ~1/sqrt(80) ของ table เดียว การที่ยังเหลือ 8%
แปลว่ามี **noise ที่ทุก table แชร์กัน** ซึ่งไม่มีทางเฉลี่ยทิ้งได้ด้วยการเพิ่ม tables

ผู้ต้องสงสัยหลัก: `feature_weighting="contrast"` = |mean(seeds) - mean(bg)| คำนวณ
**ครั้งเดียว** จาก seeds ไม่กี่สิบตัว (ที่ cell นี้ = 61) แล้วเอาไปคูณ input ของ
*ทุก* table -> weight ที่เพี้ยนทำให้ทั้ง 80 tables เพี้ยนเหมือนกันหมด

arms (fraud @ 0.3%, cell ที่ baseline เสถียรที่สุด, 6 reps ต่อ arm):

  geometry    seeds ตรึง, weighting=contrast, random_state 0..5
              -> ค่าอ้างอิง (ควรได้ ~CV 8% ตรงกับ control เดิม = sanity check)
  no_weight   seeds ตรึง, weighting="none",   random_state 0..5
              -> ถ้า CV ตกฮวบ = weights คือแหล่ง noise ที่แชร์กัน (ยืนยันสมมติฐาน)
  seed_draw   random_state ตรึง, สุ่ม seeds 6 ชุด, weighting=contrast
              -> ความแปรปรวนจาก "ได้ confirmed positives ชุดไหนมา" (ควบคุมไม่ได้)

การตีความคือการเทียบ **CV** ข้าม arm (scale-free) ไม่ใช่ระดับ pr_auc

รัน: PYTHONPATH=outputs uv run python outputs/variance_decomposition.py
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
BASE = dict(mode="pu", tune=False, n_bits=10, n_tables=80, score_type="lift",
            agg="log", engine="rpforest", hier_depths="auto")


def main() -> None:
    df, target, split_type, max_rows = load_dataset("fraud")
    tr_full, te_full = split_data(df, target, split_type, max_rows)
    train = sample_ratio(tr_full, target, RATIO, 42)
    test = sample_ratio(te_full, target, RATIO, 43)
    train, test = keep_train_valid_columns(train, test, target)
    y_tr = train[target].to_numpy().astype(np.float64)
    y_te = test[target].to_numpy().astype(np.float64)
    X_tr, X_te = train.drop(target), test.drop(target)
    pos = np.flatnonzero(y_tr == 1)
    n_seeds = max(2, int(0.10 * len(pos)))
    fixed_seeds = np.random.default_rng(7).choice(pos, size=n_seeds, replace=False)
    print(f"fraud @ {RATIO:.1%}  bg={X_tr.height:,}  seeds={n_seeds}  "
          f"reps={N_REPS}\n", flush=True)

    rows = []
    for rep in range(N_REPS):
        for arm in ("geometry", "no_weight", "seed_draw"):
            if arm == "seed_draw":
                sd = np.random.default_rng(500 + rep).choice(pos, size=n_seeds,
                                                             replace=False)
                rs, fw = 0, "contrast"
            else:
                sd, rs = fixed_seeds, rep
                fw = "contrast" if arm == "geometry" else "none"
            t0 = time.perf_counter()
            clf = LeadX(**BASE, feature_weighting=fw,
                        random_state=rs).fit(X_tr[sd.tolist()], background=X_tr)
            m = metrics(y_te, clf.score_samples(X_te))
            rows.append({"arm": arm, "rep": rep, "fit_seconds":
                         round(time.perf_counter() - t0, 1), **m})
            print(f"  {arm:<10} rep={rep}: pr={m['pr_auc']:.4f} "
                  f"roc={m['roc_auc']:.3f} lift01={m['lift_at_0_1pct']:.1f}",
                  flush=True)
            del clf
            gc.collect()

    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/variance_decomposition_raw.csv")
    print("\n=== variance decomposition ===")
    print(raw.group_by("arm").agg(
        n=pl.len(),
        median_pr=pl.col("pr_auc").median().round(4),
        cv_pr=(100 * pl.col("pr_auc").std() / pl.col("pr_auc").mean()).round(2),
        cv_roc=(100 * pl.col("roc_auc").std() / pl.col("roc_auc").mean()).round(2),
        spread_pr=(100 * (pl.col("pr_auc").max() - pl.col("pr_auc").min())
                   / pl.col("pr_auc").median()).round(1),
    ).sort("arm"))
    g = raw.filter(pl.col("arm") == "geometry")["pr_auc"]
    w = raw.filter(pl.col("arm") == "no_weight")["pr_auc"]
    cv = lambda x: 100 * x.std() / x.mean()          # noqa: E731
    print(f"\ncontrast weights ทำให้ CV เปลี่ยน: {cv(g):.2f}% -> {cv(w):.2f}% "
          f"เมื่อปิด weighting")
    print(f"ระดับ pr_auc: contrast {g.median():.4f} vs none {w.median():.4f} "
          f"({100*(w.median()-g.median())/g.median():+.1f}%)")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n({time.time()-t0:.0f}s)")
