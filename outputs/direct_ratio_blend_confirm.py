"""ยืนยัน blend ของ lift (Naive Bayes) + LR บน bucket (discriminative) — confirmatory

**ที่มา:** ในสองรอบก่อน `blend` เป็นแค่ตัวที่ *รายงาน* ไม่ใช่ primary แต่เป็นบวกทั้งคู่:
  61 seeds ultra   : PR +2.68% (4/6)
  4000 seeds natural: PR +4.73% (5/5), p@1% +5.03%
ผลที่เห็นหลังรันจะเชื่อไม่ได้จนกว่าจะยืนยันบน **seeds ชุดใหม่** และ **dataset ที่สอง**

ทฤษฎีที่ทำให้ blend มีเหตุผล: lift คือ Naive Bayes ข้าม tables (เด่นเมื่อข้อมูลน้อย)
ส่วน LR บน bucket คือ discriminative (เด่นเมื่อข้อมูลมาก, ชดเชยความซ้ำซ้อนระหว่าง
tables) — ทั้งคู่ประมาณ log density ratio เดียวกันด้วย bias/variance ต่างกัน

**blend สองแบบ:**
  blend_rank   rank-average ภายใน batch (ซ้ำกับที่เห็นมา) — ขึ้นกับ batch จึง
               ใช้กับ streaming / score ทีละ chunk ไม่ได้
  blend_log    log(lift) + logit_LR — ทั้งสองตัวอยู่บนสเกล log-ratio อยู่แล้ว
               ไม่ขึ้นกับ batch -> **ตัวนี้เท่านั้นที่ ship ได้**

---------------------------------------------------------------------------
PRE-REGISTERED
---------------------------------------------------------------------------
Cells   : (a) fraud natural, 4000 seeds   (b) home_credit natural, 4000 seeds
          (c) fraud @ 0.3%, 10% seeds (61) — regime เป้าหมายของ rare_pu
          5 reps ต่อ cell, **seed draw ใหม่** rng(7000 + 31*rep), random_state=100+rep
          (ไม่ซ้ำกับ draw ใดที่เคยใช้)
LR      : C=0.1, class_weight="balanced" (ตรึงจากรอบแรก ไม่จูนใหม่)
Primary : blend_log, pr_auc paired vs base
GATE    : pooled 15 obs -> median paired delta pr_auc > +2% AND ชนะ >= 12/15
          AND median paired delta precision@0.1% >= 0
          รายงานผลราย cell ด้วย — ถ้าผ่านแบบ pooled แต่ cell (c) ติดลบ แปลว่าช่วยแค่
          regime seeds เยอะ ซึ่งยังมีค่า (คือโจทย์จริง) แต่ต้องเขียนขอบเขตให้ชัด
blend_rank: report-only (ตรวจว่าผลเดิม replicate หรือไม่)

รัน: PYTHONPATH=outputs uv run python outputs/direct_ratio_blend_confirm.py
"""
import gc
import time

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from direct_ratio_experiment import BASE, bucket_onehot, rank01
from extreme_imbalance_benchmark import keep_train_valid_columns, metrics, sample_ratio
from leadx import LeadX
from real_data_benchmark import load_dataset, split_data

N_REPS = 5
CELLS = (("fraud", None, 4000), ("home_credit", None, 4000), ("fraud", 0.003, None))


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
        sd = np.random.default_rng(7000 + 31 * rep).choice(pos, size=n_seeds,
                                                           replace=False)
        model = LeadX(**BASE, random_state=100 + rep).fit(X_tr[sd.tolist()],
                                                          background=X_tr)
        lift = model.score_samples(X_te)
        B_tr, B_te, _ = bucket_onehot(model.transform_codes(X_tr),
                                      model.transform_codes(X_te))
        s = np.zeros(X_tr.height)
        s[sd] = 1.0
        logit = LogisticRegression(C=0.1, class_weight="balanced", solver="lbfgs",
                                   max_iter=2000).fit(B_tr, s).decision_function(B_te)
        scores = {"base": lift,
                  "blend_log": np.log(lift) + logit,
                  "blend_rank": rank01(lift) + rank01(logit)}
        for v, sc in scores.items():
            rows.append({"cell": cell, "rep": rep, "variant": v, **metrics(y_te, sc)})
        print(f"  rep={rep} | " + " | ".join(
            f"{r['variant']} pr={r['pr_auc']:.4f} p01={r['precision_at_0_1pct']:.3f}"
            for r in rows[-3:]) + f" ({time.perf_counter()-t0:.0f}s)", flush=True)
        del model, B_tr, B_te
        gc.collect()


def main() -> None:
    rows: list[dict] = []
    for name, ratio, ns in CELLS:
        run_cell(name, ratio, ns, rows)
    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/direct_ratio_blend_confirm_raw.csv")
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
    print("\n=== ราย cell (paired vs base) ===")
    print(j.group_by(["cell", "variant"]).agg(**agg).sort(["cell", "variant"]))
    print("\n=== pooled (15 obs) ===")
    print(j.group_by("variant").agg(**agg)
          .with_columns(gate=(pl.col("variant") == "blend_log") & (pl.col("d_pr") > 2)
                        & (pl.col("wins") >= 12) & (pl.col("d_p01") >= 0))
          .sort("variant"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n({time.time()-t0:.0f}s)")
