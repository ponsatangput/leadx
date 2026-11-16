"""re-rank shortlist ด้วยระยะจริง — แก้หัวแถวที่ bucket หยาบเกินไป

**ที่มา:** ทุกการทดลองวันนี้เจอ pattern เดียว — ROC นิ่ง (CV 0.3-2.7%) แต่ PR/หัวแถว
แกว่ง (5-18%) แปลว่า LeadX จัดอันดับ *ทั้งเส้น* ได้ดี แต่ที่หัวแถว การเป็นสมาชิก
bucket หยาบเกินไป: ที่ fraud @ 0.3% แต่ละแถวมี seed ใน bucket แค่ ~5% ของ tables
(miss_rate_join 0.947) คะแนนหัวแถวจึงมาจากหลักฐานไม่กี่ table

cascade (LeadX shortlist -> XGBoost re-rank) ได้ +22.5% Recall@0.1% มาแล้ว แต่ต้อง
มีโมเดลที่สองและ label สองคลาส การทดลองนี้ถามว่า **re-rank ด้วยระยะจริงใน space
เดียวกับที่ LeadX hash (encoded + feature-weighted)** ได้ผลแบบเดียวกันโดยยัง PU
ล้วนหรือไม่ — ใช้ข้อมูลแค่ seeds + background เหมือนเดิม

ใช้เฉพาะ shortlist (top 10% ของ LeadX) เพราะ kNN ทั้ง population แพง และ LeadX
เก่งเรื่องคัดออกอยู่แล้ว (ROC นิ่ง) — แถวนอก shortlist คงลำดับ lift เดิมไว้ใต้ shortlist

Variants (re-rank ภายใน shortlist เท่านั้น):
  base       ลำดับ lift ของ LeadX (ไม่ re-rank)
  knn_seed   -d_seed(k)                       ใกล้ seed = สูง (ไม่ดู density พื้นหลัง)
  knn_ratio  log d_bg(k) - log d_seed(k)      kNN density-ratio estimator: monotone
             กับ (k/n_s / k/n_b) * (d_bg/d_seed)^D จึงจัดอันดับด้วย log ต่างได้ —
             แนวคิดเดียวกับ lift แต่ต่อเนื่อง ไม่มีขอบ bucket
  blend      rank-average ของ lift กับ knn_ratio

---------------------------------------------------------------------------
PRE-REGISTERED
---------------------------------------------------------------------------
Cell    : fraud @ 0.3%, 61 seeds, rare_pu config (rpforest b10 x 80), 6 reps
          (rep เปลี่ยนทั้ง seed draw + random_state; paired ภายใน rep)
Shortlist: top 10% ของ test ตาม lift; k = 5; background ref = uniform 20k จาก train
Primary : precision_at_1pct (budget 1% < shortlist 10% จึงวัดผลของ re-rank ได้;
          precision@1% มี resolution ดีกว่า @0.1% ที่มี positive แค่ ~4-18 ตัว)
GATE    : variant ผ่านถ้า median paired delta p@1% > +3% AND ชนะ >= 5/6
          AND median paired delta PR-AUC >= -2% (ห้ามแลกหัวแถวกับทั้งเส้น)
Secondary: precision@0.1%, CV ของ base เป็นกรอบ noise
หักล้าง : ไม่มี variant ผ่าน -> ความหยาบของ bucket ไม่ใช่ตัวจำกัดหัวแถว หรือระยะใน
          encoded space (~170 มิติ) concentrate จนไม่มีข้อมูลเพิ่ม

รัน: PYTHONPATH=outputs uv run python outputs/knn_rerank_experiment.py
"""
import gc
import math
import time

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors

from extreme_imbalance_benchmark import keep_train_valid_columns, sample_ratio
from leadx import LeadX
from real_data_benchmark import load_dataset, split_data

N_REPS = 6
RATIO = 0.003
SHORT_FRAC = 0.10
K = 5
N_BG_REF = 20_000
BASE = dict(mode="pu", tune=False, n_bits=10, n_tables=80, score_type="lift",
            agg="log", engine="rpforest", feature_weighting="contrast",
            hier_depths="auto")


def embed(model: LeadX, df: pl.DataFrame) -> np.ndarray:
    """space เดียวกับที่ LeadX hash: encode -> feature weights"""
    Z = model.encoder_.transform(model._drop_mh(df))
    if model.feature_weights_ is not None:
        Z = Z * model.feature_weights_
    return np.ascontiguousarray(Z, dtype=np.float32)


def precision_at(y: np.ndarray, order: np.ndarray, frac: float) -> float:
    k = max(1, int(math.ceil(len(y) * frac)))
    return float(y[order[:k]].mean())


def final_order(base_score: np.ndarray, short_idx: np.ndarray,
                short_score: np.ndarray) -> np.ndarray:
    """shortlist เรียงด้วย short_score อยู่บนสุด แถวที่เหลือคงลำดับ lift"""
    top = short_idx[np.argsort(-short_score, kind="stable")]
    rest_mask = np.ones(len(base_score), dtype=bool)
    rest_mask[short_idx] = False
    rest = np.flatnonzero(rest_mask)
    rest = rest[np.argsort(-base_score[rest], kind="stable")]
    return np.concatenate([top, rest])


def rank01(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x, kind="stable"), kind="stable") / max(1, len(x) - 1)


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
    n_short = int(math.ceil(SHORT_FRAC * len(y_te)))
    print(f"fraud @ {RATIO:.1%} bg={X_tr.height:,} test={len(y_te):,} "
          f"seeds={n_seeds} shortlist={n_short:,} k={K}\n", flush=True)

    rows = []
    for rep in range(N_REPS):
        sd = np.random.default_rng(100 * rep + 7).choice(pos, size=n_seeds, replace=False)
        t0 = time.perf_counter()
        model = LeadX(**BASE, random_state=rep).fit(X_tr[sd.tolist()], background=X_tr)
        lift = model.score_samples(X_te)
        short = np.argsort(-lift, kind="stable")[:n_short]

        Z_seed = embed(model, X_tr[sd.tolist()])
        bg_idx = np.random.default_rng(3000 + rep).choice(X_tr.height, N_BG_REF,
                                                          replace=False)
        Z_bg = embed(model, X_tr[bg_idx.tolist()])
        Z_q = embed(model, X_te[short.tolist()])
        d_seed = NearestNeighbors(n_neighbors=K).fit(Z_seed).kneighbors(Z_q)[0][:, -1]
        d_bg = NearestNeighbors(n_neighbors=K).fit(Z_bg).kneighbors(Z_q)[0][:, -1]
        eps = 1e-9
        knn_ratio = np.log(d_bg + eps) - np.log(d_seed + eps)

        variants = {
            "base": lift[short],
            "knn_seed": -d_seed,
            "knn_ratio": knn_ratio,
            "blend": rank01(lift[short]) + rank01(knn_ratio),
        }
        for name, sc in variants.items():
            order = final_order(lift, short, sc)
            # PR-AUC ของลำดับสุดท้าย: แปลงตำแหน่งเป็นคะแนน (สูง = อยู่บน)
            pseudo = np.empty(len(y_te))
            pseudo[order] = np.arange(len(y_te), 0, -1)
            rows.append({"rep": rep, "variant": name,
                         "p_at_0_1pct": precision_at(y_te, order, 0.001),
                         "p_at_1pct": precision_at(y_te, order, 0.01),
                         "pr_auc": float(average_precision_score(y_te, pseudo)),
                         "seconds": round(time.perf_counter() - t0, 1)})
        b = rows[-4]
        print(f"  rep={rep} base p@1%={b['p_at_1pct']:.4f} | " + " | ".join(
            f"{r['variant']} {r['p_at_1pct']:.4f}" for r in rows[-3:]), flush=True)
        del model
        gc.collect()

    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/knn_rerank_raw.csv")
    cv = lambda c: (100 * pl.col(c).std() / pl.col(c).mean())   # noqa: E731
    print("\n=== ระดับ ===")
    print(raw.group_by("variant").agg(
        p1=pl.col("p_at_1pct").median().round(4), cv_p1=cv("p_at_1pct").round(2),
        p01=pl.col("p_at_0_1pct").median().round(4),
        pr=pl.col("pr_auc").median().round(4)).sort("variant"))
    base = raw.filter(pl.col("variant") == "base").select(
        "rep", pl.col("p_at_1pct").alias("q0"), pl.col("pr_auc").alias("a0"),
        pl.col("p_at_0_1pct").alias("z0"))
    j = (raw.filter(pl.col("variant") != "base").join(base, on="rep")
         .with_columns(
             (100 * (pl.col("p_at_1pct") - pl.col("q0")) / pl.col("q0")).alias("d1"),
             (100 * (pl.col("pr_auc") - pl.col("a0")) / pl.col("a0")).alias("dpr"),
             (100 * (pl.col("p_at_0_1pct") - pl.col("z0")) / pl.col("z0")).alias("d01")))
    print("\n=== paired vs base ===")
    print(j.group_by("variant").agg(
        wins=(pl.col("p_at_1pct") > pl.col("q0")).sum(),
        d_p1=pl.col("d1").median().round(2),
        d_pr=pl.col("dpr").median().round(2),
        d_p01=pl.col("d01").median().round(2))
        .with_columns(gate=(pl.col("d_p1") > 3) & (pl.col("wins") >= 5)
                      & (pl.col("d_pr") >= -2))
        .sort("variant"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n({time.time()-t0:.0f}s)")
