"""ยืนยัน learned hash tables — hybrid (random + learned) เป็นตัวหลัก

**ที่มา:** `learned_partition_experiment.py` ตัวหลัก `leaf_honest` ไม่ผ่าน gate
(pooled +4.81% แต่ 11/15 และ fraud@0.3% 1/5) แต่สองตัวที่เป็น report-only ดูแรงมาก:
  hybrid     pooled +7.63%, **15/15**, ทุก cell 5/5, p@0.1% +12.3%
  leaf_naive pooled +43.94%, 14/15 (fraud@0.3% +81.4% 4/5)
ผลที่เห็นหลังรันต้องยืนยันบน seed draw ใหม่ (บทเรียน: blend ของ direct-ratio ก็เคย
ดู 5/5 แล้วล้มในรอบยืนยัน)

**ทำไม hybrid เป็นตัวหลัก (ตัดสินก่อนเห็นผลรอบนี้):** learned partition ขึ้นกับ
seeds ของแต่ละโจทย์ จึงทำลายจุดขาย "สร้าง geometry ครั้งเดียว ใช้หลายโจทย์" —
hybrid รักษาไว้บางส่วน (random tables ใช้ซ้ำได้ เพิ่มแค่ learned tables ต่อโจทย์)
leaf_naive รายงานด้วย gate เดียวกันเพื่อข้อมูล

**construction เหมือนรอบแรกทุกประการ** (ยืนยันสิ่งที่วัดไป ไม่ปรับหลังเห็นผล):
  hybrid     = (log(base lift) + leaf_honest) / 2   — leaf ใช้ honest cross-fit
  leaf_naive = 2 forests บน seeds คนละครึ่ง นับด้วยครึ่งเดียวกับที่สร้าง เฉลี่ย

---------------------------------------------------------------------------
PRE-REGISTERED
---------------------------------------------------------------------------
Cells   : fraud@0.3% (61 seeds), fraud natural (4000), home_credit natural (4000)
          5 reps ต่อ cell, **seed draw ใหม่** rng(11000 + 19*rep), random_state=300+rep
Forest  : เหมือนรอบแรก (80 trees, 1024 leaves, balanced, sqrt)
PRIMARY : hybrid
GATE    : pooled 15 obs median paired delta pr_auc > +2% AND ชนะ >= 12/15
          AND median paired delta precision@0.1% >= 0 AND ชนะ >= 3/5 ในทุก cell
leaf_naive: gate เดียวกัน, report-only

รัน: PYTHONPATH=outputs uv run python outputs/learned_partition_confirm.py
"""
import gc
import time

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier

from extreme_imbalance_benchmark import keep_train_valid_columns, metrics, sample_ratio
from leadx import LeadX
from learned_partition_experiment import BASE, CELLS, RF_KW, leaf_log_lift
from real_data_benchmark import load_dataset, split_data

N_REPS = 5


def run_cell(name, ratio, n_seeds_fixed, rows):
    df, target, split_type, max_rows = load_dataset(name)
    train, test = split_data(df, target, split_type, max_rows)
    if ratio is not None:
        train = sample_ratio(train, target, ratio, 42)
        test = sample_ratio(test, target, ratio, 43)
    train, test = keep_train_valid_columns(train, test, target)
    y_tr = train[target].to_numpy().astype(np.float64)
    y_te = test[target].to_numpy().astype(np.float64)
    X_tr, X_te = train.drop(target), test.drop(target)
    pos = np.flatnonzero(y_tr == 1)
    n_seeds = n_seeds_fixed or max(2, int(0.10 * len(pos)))
    cell = f"{name}@{'natural' if ratio is None else f'{ratio:.1%}'}"
    print(f"\n[{cell}] bg={X_tr.height:,} pos={len(pos):,} seeds={n_seeds}", flush=True)
    for rep in range(N_REPS):
        t0 = time.perf_counter()
        rng = np.random.default_rng(11000 + 19 * rep)
        sd = rng.choice(pos, size=n_seeds, replace=False)
        rs = 300 + rep
        model = LeadX(**BASE, random_state=rs).fit(X_tr[sd.tolist()], background=X_tr)
        base = model.score_samples(X_te)
        alpha = float(model.config_["alpha"])
        Z_bg = model.encoder_.transform(model._drop_mh(X_tr))
        Z_te = model.encoder_.transform(model._drop_mh(X_te))
        halves = np.array_split(rng.permutation(sd), 2)
        honest, naive = np.zeros(len(y_te)), np.zeros(len(y_te))
        for f in range(2):
            build, count = halves[f], halves[1 - f]
            s = np.zeros(X_tr.height)
            s[build] = 1.0
            rf = RandomForestClassifier(**RF_KW, random_state=rs * 10 + f).fit(Z_bg, s)
            L_bg, L_te = rf.apply(Z_bg), rf.apply(Z_te)
            honest += leaf_log_lift(L_bg, L_bg[count], L_te, alpha) / 2
            naive += leaf_log_lift(L_bg, L_bg[build], L_te, alpha) / 2
            del rf, L_bg, L_te
        scores = {"base": base, "hybrid": (np.log(base) + honest) / 2,
                  "leaf_naive": naive}
        for v, sc in scores.items():
            rows.append({"cell": cell, "rep": rep, "variant": v, **metrics(y_te, sc)})
        print(f"  rep={rep} | " + " | ".join(
            f"{r['variant']} pr={r['pr_auc']:.4f} p01={r['precision_at_0_1pct']:.3f}"
            for r in rows[-3:]) + f" ({time.perf_counter()-t0:.0f}s)", flush=True)
        del model, Z_bg, Z_te
        gc.collect()


def main() -> None:
    rows: list[dict] = []
    for name, ratio, ns in CELLS:
        run_cell(name, ratio, ns, rows)
    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/learned_partition_confirm_raw.csv")
    base = raw.filter(pl.col("variant") == "base").select(
        "cell", "rep", pl.col("pr_auc").alias("a0"),
        pl.col("precision_at_0_1pct").alias("z0"), pl.col("precision_at_1pct").alias("q0"))
    j = (raw.filter(pl.col("variant") != "base").join(base, on=["cell", "rep"])
         .with_columns(
             (100 * (pl.col("pr_auc") - pl.col("a0")) / pl.col("a0")).alias("dpr"),
             (100 * (pl.col("precision_at_0_1pct") - pl.col("z0"))
              / pl.col("z0")).alias("d01"),
             (100 * (pl.col("precision_at_1pct") - pl.col("q0")) / pl.col("q0")).alias("d1")))
    agg = dict(n=pl.len(), wins=(pl.col("pr_auc") > pl.col("a0")).sum(),
               d_pr=pl.col("dpr").median().round(2), d_p01=pl.col("d01").median().round(2),
               d_p1=pl.col("d1").median().round(2))
    per_cell = j.group_by(["cell", "variant"]).agg(**agg).sort(["cell", "variant"])
    print("\n=== ราย cell (paired vs base) ===")
    print(per_cell)
    mins = per_cell.group_by("variant").agg(min_cell_wins=pl.col("wins").min())
    pooled = (j.group_by("variant").agg(**agg).join(mins, on="variant")
              .with_columns(gate=(pl.col("d_pr") > 2) & (pl.col("wins") >= 12)
                            & (pl.col("d_p01") >= 0) & (pl.col("min_cell_wins") >= 3))
              .sort("variant"))
    print("\n=== pooled (15 obs) — PRIMARY = hybrid ===")
    print(pooled)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n({time.time()-t0:.0f}s)")
