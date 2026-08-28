"""feature-selection headroom — ตัด features ก่อนเข้า LeadX ช่วยที่ ultra-imbalance มั้ย?

คำถาม: SNR ต่อ projection ของ random-hyperplane/rpTree ≈ sqrt(k/(D-k)) เมื่อ k =
จำนวน feature ที่มีสัญญาณจริง, D = จำนวน feature ทั้งหมด (fraud D=153,
home_credit D=120). ถ้า junk features เจือจางสัญญาณจริง การคัด feature ก่อน fit
ควรช่วย — แต่ยังไม่เคยวัด

ออกแบบเป็นสองชั้นเพื่อให้ *ตัดสินใจได้* ด้วยการรันครั้งเดียว:

  oracle_ig_k : IG จาก label จริงทั้ง training set  -> upper bound ของ feature
                selection ทุกวิธี (ทำจริงไม่ได้ใน PU — ใช้วัด headroom เท่านั้น)
  pu_ig_k     : IG จาก seeds-vs-background          -> PU-legal, ship ได้จริง

ถ้า oracle ไม่ผ่าน แปลว่า *ไม่มี* วิธีคัด feature ใดช่วยได้ ปิดประเด็นทันที
โดยไม่ต้องเขียน selection จริง

---------------------------------------------------------------------------
PRE-REGISTERED GATE (commit ก่อนรัน full)
---------------------------------------------------------------------------
Primary metric : pr_auc, paired ต่อ (dataset, ratio, draw) เทียบ variant "all"
Cells          : fraud @ 0.003, 0.001 ; home_credit @ 0.003  (3 cells x 3 draws
                 = 9 paired observations ต่อ variant, draw seed = rng(100*d+7)
                 ตรงกับ imbalance_focus_rpf.py)
PASS ของ variant = median paired delta% > +5%  AND  ชนะ >= 7/9

Decision rule:
  - ไม่มี oracle_ig_k ผ่าน  -> feature selection ไม่ใช่คอขวด. ไม่สร้าง.
                              บันทึกเป็น rejected idea ใน DESIGN_NOTES 5.4
  - oracle ผ่าน แต่ pu_ig ไม่ผ่าน -> มี headroom แต่เอื้อมไม่ถึงแบบ PU-legal.
                              ไม่ ship, บันทึกไว้
  - pu_ig ผ่าน            -> สร้างจริงเป็น feature-selection API

Secondary (รายงานอย่างเดียว ไม่ใช้ตัดสิน): lift@0.1%, roc_auc, fit_seconds
---------------------------------------------------------------------------

รัน: PYTHONPATH=outputs uv run python outputs/feature_selection_experiment.py
"""
import gc
import sys
import time
import warnings

import numpy as np
import polars as pl

from extreme_imbalance_benchmark import keep_train_valid_columns, metrics, sample_ratio
from leadx import LeadX
from real_data_benchmark import load_dataset, split_data

warnings.filterwarnings("ignore")

DRAWS = 3
SEED_FRAC = 0.10
K_GRID = (16, 48)
N_BINS = 10
# config เดียวกับ imbalance_focus_rpf.py (เงื่อนไขที่ LeadX ชนะอยู่แล้ว)
RPF_KW = dict(mode="pu", tune=False, n_bits=10, n_tables=80, score_type="lift",
              agg="log", feature_weighting="contrast", engine="rpforest",
              hier_depths="auto", random_state=0)


def _binned(col: pl.Series, n_bins: int = N_BINS) -> np.ndarray:
    """numeric -> quantile deciles, categorical -> integer codes. null = bin ของตัวเอง"""
    if col.dtype.is_numeric():
        s = col.cast(pl.Float64).fill_nan(None)
        qs = sorted({float(s.quantile(q / n_bins, "nearest") or 0.0)
                     for q in range(1, n_bins)})
        v = s.to_numpy()
        out = np.digitize(np.nan_to_num(v, nan=np.inf), qs).astype(np.int32)
        out[~np.isfinite(v)] = len(qs) + 1          # missing = bin แยก
        return out
    return (col.cast(pl.Utf8).fill_null("__missing__").cast(pl.Categorical)
               .to_physical().to_numpy().astype(np.int32))


def _entropy(counts: np.ndarray) -> float:
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts[counts > 0] / tot
    return float(-(p * np.log2(p)).sum())


def information_gain(X: pl.DataFrame, y: np.ndarray) -> np.ndarray:
    """IG(feature; y) = H(y) - sum_b p(b) H(y|b) ต่อคอลัมน์ (bits)"""
    y = y.astype(np.int64)
    h_y = _entropy(np.bincount(y, minlength=2))
    out = np.zeros(X.width)
    n = len(y)
    for j, name in enumerate(X.columns):
        b = _binned(X[name])
        nb = int(b.max()) + 1
        joint = np.bincount(b * 2 + y, minlength=nb * 2).reshape(nb, 2)
        marg = joint.sum(axis=1)
        cond = sum((marg[i] / n) * _entropy(joint[i]) for i in range(nb) if marg[i])
        out[j] = h_y - cond
    return out


def top_k_columns(X: pl.DataFrame, y: np.ndarray, k: int) -> list[str]:
    ig = information_gain(X, y)
    order = np.argsort(-ig, kind="stable")[:k]
    return [X.columns[i] for i in sorted(order)]


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

        # oracle IG: label จริงทั้ง train set — คงที่ทุก draw (ไม่ขึ้นกับ seeds)
        t0 = time.perf_counter()
        oracle_cols = {k: top_k_columns(X_tr, y_tr.astype(np.int64), k)
                       for k in K_GRID}
        print(f"  [{name} r={ratio:.3%}] oracle IG {time.perf_counter()-t0:.0f}s "
              f"| D={X_tr.width} pos={len(pos_idx)} seeds={n_seeds}", flush=True)

        for d in range(DRAWS):
            rd = np.random.default_rng(100 * d + 7)   # draw เดียวกับ benchmark เดิม
            seeds = rd.choice(pos_idx, size=n_seeds, replace=False)

            # PU-legal IG: seeds (class 1) vs background ทั้งก้อน (class 0)
            y_pu = np.zeros(len(y_tr), dtype=np.int64)
            y_pu[seeds] = 1
            pu_cols = {k: top_k_columns(X_tr, y_pu, k) for k in K_GRID}

            variants = {"all": None}
            variants |= {f"oracle_ig_{k}": oracle_cols[k] for k in K_GRID}
            variants |= {f"pu_ig_{k}": pu_cols[k] for k in K_GRID}

            for vname, cols in variants.items():
                xtr = X_tr if cols is None else X_tr.select(cols)
                xte = X_te if cols is None else X_te.select(cols)
                t0 = time.perf_counter()
                clf = LeadX(**RPF_KW).fit(xtr[seeds.tolist()], background=xtr)
                score = clf.score_samples(xte)
                fit_s = time.perf_counter() - t0
                rows.append({"dataset": name, "target_ratio": ratio,
                             "n_positive_train": len(pos_idx), "n_seeds": n_seeds,
                             "n_features": xtr.width, "variant": vname, "draw": d,
                             "fit_seconds": round(fit_s, 2),
                             **metrics(y_te, score)})
                print(f"    {name} r={ratio:.3%} {vname:<14} d={d}: "
                      f"pr={rows[-1]['pr_auc']:.4f} roc={rows[-1]['roc_auc']:.3f} "
                      f"({fit_s:.0f}s)", flush=True)
                del clf
                gc.collect()
    return rows


def summarize(raw: pl.DataFrame) -> pl.DataFrame:
    base = (raw.filter(pl.col("variant") == "all")
            .select("dataset", "target_ratio", "draw",
                    pl.col("pr_auc").alias("pr_base"),
                    pl.col("lift_at_0_1pct").alias("lift_base")))
    j = raw.filter(pl.col("variant") != "all").join(
        base, on=["dataset", "target_ratio", "draw"], how="inner")
    j = j.with_columns(
        (100 * (pl.col("pr_auc") - pl.col("pr_base")) / pl.col("pr_base")).alias("d_pr"),
        (pl.col("pr_auc") > pl.col("pr_base")).alias("win"))
    return (j.group_by("variant").agg(
        n=pl.len(), wins=pl.col("win").sum(),
        median_d_pr=pl.col("d_pr").median().round(2),
        mean_d_pr=pl.col("d_pr").mean().round(2),
        worst=pl.col("d_pr").min().round(2), best=pl.col("d_pr").max().round(2),
        fit_s=pl.col("fit_seconds").median())
        .with_columns(gate=(pl.col("median_d_pr") > 5) & (pl.col("wins") >= 7))
        .sort("variant"))


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    rows: list[dict] = []
    if smoke:
        globals()["DRAWS"] = 1
        run_dataset("fraud", (0.01,), rows)
    else:
        run_dataset("fraud", (0.003, 0.001), rows)
        run_dataset("home_credit", (0.003,), rows)
    raw = pl.DataFrame(rows)
    out = "outputs/feature_selection_smoke.csv" if smoke else \
          "outputs/feature_selection_raw.csv"
    raw.write_csv(out)
    print(f"\nwrote {out}  ({time.time()-t0:.0f}s total)")
    if not smoke:
        print("\n=== paired vs all-features (primary: pr_auc) ===")
        print(summarize(raw))
