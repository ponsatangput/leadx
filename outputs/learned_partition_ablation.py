"""ablation: กำไรของ learned partition มาจาก LeadX หรือแค่จาก random forest?

**ที่มา:** `learned_partition_confirm.py` ผ่าน gate ทั้งสองตัว (seed draw ใหม่):
  hybrid     pooled +9.95%  (14/15)
  leaf_naive pooled +45.79% (15/15), ทุก cell 5/5
แต่ leaf_naive มีโครงสร้างใกล้ PU random forest ธรรมดามาก: lift ต่อใบที่คำนวณ ≈
สัดส่วน seed ในใบ ซึ่งคือสิ่งที่ `predict_proba` ของ RF ทำอยู่แล้ว ถ้า RF ธรรมดา
ได้เท่ากัน ข้อสรุปที่ซื่อตรงคือ "tree ที่เรียนมาดีกว่า partition สุ่ม" ไม่ใช่
"LeadX ดีขึ้น" — ต้องแยกก่อนเคลมอะไร

และเพราะผู้ใช้จริงใช้ XGBoost จึงเทียบกับ XGBoost แบบ PU ด้วย config เดียวกับที่ใช้
ใน complementarity_benchmark (hist, 300 trees, depth 6, lr 0.1, scale_pos_weight
balanced, input = space ของ encoder ตัวเดียวกัน)

---------------------------------------------------------------------------
PRE-REGISTERED
---------------------------------------------------------------------------
Cells   : fraud@0.3% (61 seeds), fraud natural (4000), home_credit natural (4000)
          5 reps, **seed draw ใหม่** rng(13000 + 23*rep), random_state=400+rep
Variants: base        rare_pu lift
          leaf_naive  construction เดิม (2 forests บนครึ่ง seeds, นับครึ่งเดียวกัน)
          rf_proba    **forests ชุดเดียวกัน** แต่ใช้ predict_proba[:,1] เฉลี่ยสองฝั่ง
                      -> ต่างจาก leaf_naive แค่กฎให้คะแนนบนใบ (ablation เดียวที่สะอาด)
          hybrid      construction เดิม (log base + leaf_honest)/2
          xgb_pu      XGBoost seeds vs background (config ข้างบน, ใช้ seeds ทั้งหมด)

Q1 กฎให้คะแนนของ LeadX ช่วยหรือไม่: leaf_naive vs rf_proba (paired)
   "LeadX scoring ช่วย" ถ้า median delta pr_auc > +2% AND ชนะ >= 12/15
   "ไม่ต่าง/แย่กว่า" ถ้าไม่ผ่าน -> กำไรมาจาก tree ล้วน ต้องเขียน positioning ใหม่
Q2 เทียบผู้เล่นปัจจุบัน: leaf_naive และ hybrid vs xgb_pu (paired)
   "ชนะ XGBoost-PU" ถ้า median delta pr_auc > +2% AND ชนะ >= 12/15 AND ชนะ >= 3/5
   ทุก cell; รายงาน precision@0.1% คู่กัน

รัน: PYTHONPATH=outputs uv run --with xgboost python outputs/learned_partition_ablation.py
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
XGB_KW = dict(tree_method="hist", n_estimators=300, max_depth=6, learning_rate=0.1,
              n_jobs=4)


def run_cell(name, ratio, n_seeds_fixed, rows):
    import xgboost as xgb
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
        rng = np.random.default_rng(13000 + 23 * rep)
        sd = rng.choice(pos, size=n_seeds, replace=False)
        rs = 400 + rep
        model = LeadX(**BASE, random_state=rs).fit(X_tr[sd.tolist()], background=X_tr)
        base = model.score_samples(X_te)
        alpha = float(model.config_["alpha"])
        Z_bg = model.encoder_.transform(model._drop_mh(X_tr))
        Z_te = model.encoder_.transform(model._drop_mh(X_te))
        halves = np.array_split(rng.permutation(sd), 2)
        honest, naive, proba = (np.zeros(len(y_te)) for _ in range(3))
        for f in range(2):
            build, count = halves[f], halves[1 - f]
            s = np.zeros(X_tr.height)
            s[build] = 1.0
            rf = RandomForestClassifier(**RF_KW, random_state=rs * 10 + f).fit(Z_bg, s)
            L_bg, L_te = rf.apply(Z_bg), rf.apply(Z_te)
            honest += leaf_log_lift(L_bg, L_bg[count], L_te, alpha) / 2
            naive += leaf_log_lift(L_bg, L_bg[build], L_te, alpha) / 2
            proba += rf.predict_proba(Z_te)[:, 1] / 2
            del rf, L_bg, L_te
        s_all = np.zeros(X_tr.height)
        s_all[sd] = 1.0
        xgb_pu = (xgb.XGBClassifier(**XGB_KW, random_state=rs,
                                    scale_pos_weight=(len(s_all) - s_all.sum()) / s_all.sum())
                  .fit(Z_bg, s_all).predict_proba(Z_te)[:, 1])
        scores = {"base": base, "leaf_naive": naive, "rf_proba": proba,
                  "hybrid": (np.log(base) + honest) / 2, "xgb_pu": xgb_pu}
        for v, sc in scores.items():
            rows.append({"cell": cell, "rep": rep, "variant": v, **metrics(y_te, sc)})
        print(f"  rep={rep} | " + " | ".join(
            f"{r['variant']} pr={r['pr_auc']:.4f}" for r in rows[-5:])
            + f" ({time.perf_counter()-t0:.0f}s)", flush=True)
        del model, Z_bg, Z_te
        gc.collect()


def paired(raw: pl.DataFrame, ref: str, others: list[str]) -> pl.DataFrame:
    base = raw.filter(pl.col("variant") == ref).select(
        "cell", "rep", pl.col("pr_auc").alias("a0"),
        pl.col("precision_at_0_1pct").alias("z0"))
    j = (raw.filter(pl.col("variant").is_in(others)).join(base, on=["cell", "rep"])
         .with_columns(
             (100 * (pl.col("pr_auc") - pl.col("a0")) / pl.col("a0")).alias("dpr"),
             (100 * (pl.col("precision_at_0_1pct") - pl.col("z0"))
              / pl.col("z0")).alias("d01")))
    agg = dict(n=pl.len(), wins=(pl.col("pr_auc") > pl.col("a0")).sum(),
               d_pr=pl.col("dpr").median().round(2), d_p01=pl.col("d01").median().round(2))
    cells = j.group_by(["cell", "variant"]).agg(**agg).sort(["variant", "cell"])
    mins = cells.group_by("variant").agg(min_cell_wins=pl.col("wins").min())
    pooled = j.group_by("variant").agg(**agg).join(mins, on="variant").sort("variant")
    return cells, pooled


def main() -> None:
    rows: list[dict] = []
    for name, ratio, ns in CELLS:
        run_cell(name, ratio, ns, rows)
    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/learned_partition_ablation_raw.csv")
    print("\n=== ระดับ (median) ===")
    print(raw.group_by(["cell", "variant"]).agg(
        pr=pl.col("pr_auc").median().round(4),
        p01=pl.col("precision_at_0_1pct").median().round(4))
        .sort(["cell", "variant"]))

    cells, pooled = paired(raw, "rf_proba", ["leaf_naive"])
    print("\n=== Q1: leaf_naive vs rf_proba (กฎให้คะแนนของ LeadX ช่วยหรือไม่) ===")
    print(cells)
    print(pooled.with_columns(
        leadx_scoring_helps=(pl.col("d_pr") > 2) & (pl.col("wins") >= 12)))

    cells, pooled = paired(raw, "xgb_pu", ["leaf_naive", "hybrid", "base"])
    print("\n=== Q2: vs xgb_pu ===")
    print(cells)
    print(pooled.with_columns(
        beats_xgb_pu=(pl.col("d_pr") > 2) & (pl.col("wins") >= 12)
        & (pl.col("min_cell_wins") >= 3)))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n({time.time()-t0:.0f}s)")
