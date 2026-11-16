"""learned hash tables — ใช้ใบของ tree ที่เรียนจาก PU เป็น bucket แทน random partition

**ที่มา:** ทุกบริษัทที่ใช้ LSH/ANN (Spotify, Meta, Uber) ใช้มันเป็น index บน
representation ที่ *เรียนมา* ขณะที่ LeadX ใช้ partition แบบสุ่มบน feature ดิบ
หลักฐานภายในก็ชี้ทางเดียวกัน: ระยะ Euclidean ใน space ดิบแทบใช้ไม่ได้ (kNN re-rank
p@0.1% = 0), ส่วนเดียวที่ "เรียน" (contrast weighting) ให้ +13.4%, และการคัด feature
แบบ univariate ทำร้ายเพราะสัญญาณอยู่ใน *การประกอบกัน* ของ feature — ซึ่ง tree จับได้

**วิธี:** tree หนึ่งต้นคือ hash table หนึ่งตารางที่ *เรียนมา* ใบ = bucket ฝึก random
forest ระหว่าง seeds (s=1) กับ background (s=0) ซึ่งยังเป็น PU ล้วน แล้วให้คะแนน
ด้วยสูตรเดียวกับ LeadX ทุกประการ ต่อใบ:
    lift(leaf) = (n_seed_leaf + a) / (n_seed_total * n_bg_leaf / n_bg_total + a)
รวมข้าม trees ด้วยค่าเฉลี่ย log (NB เหมือน agg="log") — เปลี่ยน *แค่ partition*
จากสุ่มเป็นเรียนมา ตัวให้คะแนนเหมือนเดิม จึงแยกผลของ representation ได้ตรงๆ
ใบที่ RF สร้างแบบ best-first จะลึกรอบ seeds และตื้นที่อื่น = bandwidth ปรับตามข้อมูล
(Jaccard ของชุดใบ = Breiman's RF proximity)

**กัน overfit ด้วย honest trees (Wager & Athey 2018):** ถ้านับ seeds ชุดที่ใช้สร้าง
tree ใบที่ตัดมาแยก seed แต่ละตัวจะได้ lift สูงเกินจริง จึง cross-fit 2 fold: แบ่ง
seeds เป็น S1/S2, forest F1 เรียนจาก S1 แต่ *นับ* numerator ด้วย S2 และกลับกัน
คะแนนสุดท้าย = เฉลี่ยสองฝั่ง ใช้ seeds ครบทุกตัวโดยไม่มีตัวไหนนับตัวเอง

Input ของ tree = space ที่ encoder ของ LeadX แปลงแล้ว (standardize + one-hot +
missing indicator, ไม่คูณ feature weights) เพื่อให้ categorical/missing ถูกจัดการ
เหมือน base ทุกประการ

---------------------------------------------------------------------------
PRE-REGISTERED
---------------------------------------------------------------------------
Cells   : (a) fraud @ 0.3%, 10% seeds (61)  — regime เป้าหมายของ rare_pu
          (b) fraud natural, 4000 seeds
          (c) home_credit natural, 4000 seeds — dataset ที่สองตั้งแต่รอบแรก
          5 reps ต่อ cell, seed draw ใหม่ rng(9000 + 13*rep), random_state=200+rep
Forest  : RandomForestClassifier n_estimators=80 (= n_tables), max_leaf_nodes=1024
          (= 2^n_bits), class_weight="balanced", max_features="sqrt" — ตรึงล่วงหน้า
          ให้ budget เท่ากับ base (80 tables x 1024 buckets) ไม่จูนจาก test
Scoring : alpha = alpha ของ base config, agg = mean log
Variants: base          rare_pu lift
          leaf_honest   learned partition + honest cross-fit          <- PRIMARY
          leaf_naive    นับ seeds ชุดเดียวกับที่สร้าง tree (วัดขนาด overfit) report-only
          hybrid        เฉลี่ย log-lift ของ 80 random tables + 80 learned tables
                        = NB ข้าม 160 tables (สเกลเดียวกันจริงเพราะสูตรเดียวกัน
                        และ alpha เดียวกัน — ต่างจาก blend_log ที่พลาดเพราะผสมคนละ
                        estimator) report-only
GATE    : leaf_honest pooled 15 obs -> median paired delta pr_auc > +2%
          AND ชนะ >= 12/15 AND median paired delta precision@0.1% >= 0
          AND ชนะ >= 3/5 **ในทุก cell** (บทเรียนจาก direct-ratio: ผลที่ pooled
          ดูดีแต่มาจาก dataset เดียว ต้องไม่ผ่าน)
หักล้าง : ไม่ผ่าน -> การเรียน partition จาก seeds ไม่ดีกว่า partition สุ่มแบบ
          equal-mass; ถ้าแพ้แค่ที่ 61 seeds แต่ชนะที่ 4000 = มีค่าเฉพาะ regime seeds เยอะ
          (ซึ่งคือโจทย์จริง) แต่ไม่ผ่าน gate ข้อสุดท้าย ต้องรายงานแยก

รัน: PYTHONPATH=outputs uv run python outputs/learned_partition_experiment.py
"""
import gc
import time

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier

from extreme_imbalance_benchmark import keep_train_valid_columns, metrics, sample_ratio
from leadx import LeadX
from real_data_benchmark import load_dataset, split_data

N_REPS = 5
CELLS = (("fraud", 0.003, None), ("fraud", None, 4000), ("home_credit", None, 4000))
BASE = dict(mode="pu", tune=False, n_bits=10, n_tables=80, score_type="lift",
            agg="log", engine="rpforest", feature_weighting="contrast",
            hier_depths="auto")
RF_KW = dict(n_estimators=80, max_leaf_nodes=1024, class_weight="balanced",
             max_features="sqrt", n_jobs=-1)


def leaf_log_lift(L_bg: np.ndarray, L_cnt: np.ndarray, L_q: np.ndarray,
                  alpha: float) -> np.ndarray:
    """mean log-lift ข้าม trees; L_* = leaf ids (n, T) จาก forest.apply

    L_bg  : leaf ของ background (ตัวหาร)
    L_cnt : leaf ของ seeds ที่ใช้ *นับ* numerator
    L_q   : leaf ของแถวที่จะให้คะแนน
    """
    n_bg, n_seed = L_bg.shape[0], L_cnt.shape[0]
    out = np.zeros(L_q.shape[0])
    for t in range(L_bg.shape[1]):
        size = int(max(L_bg[:, t].max(), L_cnt[:, t].max(), L_q[:, t].max())) + 1
        bg = np.bincount(L_bg[:, t], minlength=size).astype(np.float64)
        sd = np.bincount(L_cnt[:, t], minlength=size).astype(np.float64)
        lift = (sd + alpha) / (n_seed * bg / n_bg + alpha)
        out += np.log(lift[L_q[:, t]])
    return out / L_bg.shape[1]


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
        rng = np.random.default_rng(9000 + 13 * rep)
        sd = rng.choice(pos, size=n_seeds, replace=False)
        rs = 200 + rep
        model = LeadX(**BASE, random_state=rs).fit(X_tr[sd.tolist()], background=X_tr)
        base = model.score_samples(X_te)
        alpha = float(model.config_["alpha"])

        # input ของ tree: space ของ encoder (ไม่คูณ weights)
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
        scores = {"base": base, "leaf_honest": honest, "leaf_naive": naive,
                  "hybrid": (np.log(base) + honest) / 2}
        for v, sc in scores.items():
            rows.append({"cell": cell, "rep": rep, "variant": v, **metrics(y_te, sc)})
        print(f"  rep={rep} | " + " | ".join(
            f"{r['variant']} pr={r['pr_auc']:.4f} p01={r['precision_at_0_1pct']:.3f}"
            for r in rows[-4:]) + f" ({time.perf_counter()-t0:.0f}s)", flush=True)
        del model, Z_bg, Z_te
        gc.collect()


def main() -> None:
    rows: list[dict] = []
    for name, ratio, ns in CELLS:
        run_cell(name, ratio, ns, rows)
    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/learned_partition_raw.csv")
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
    min_cell_wins = (per_cell.filter(pl.col("variant") == "leaf_honest")["wins"].min())
    print("\n=== pooled (15 obs) ===")
    print(j.group_by("variant").agg(**agg)
          .with_columns(gate=(pl.col("variant") == "leaf_honest") & (pl.col("d_pr") > 2)
                        & (pl.col("wins") >= 12) & (pl.col("d_p01") >= 0)
                        & pl.lit(min_cell_wins >= 3))
          .sort("variant"))
    print(f"\nleaf_honest ชนะต่ำสุดใน cell ใดๆ: {min_cell_wins}/5 (gate ต้อง >= 3)")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n({time.time()-t0:.0f}s)")
