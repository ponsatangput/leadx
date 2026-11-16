"""learned partitions vs XGBoost-PU ที่จูนอย่างยุติธรรม — หลาย ultra cells

**ที่มา:** `learned_partition_ablation.py` พบว่า leaf_naive (ใบของ PU random forest +
lift ของ LeadX) ชนะ XGBoost-PU ที่ fraud@0.3% +81.3% PR (5/5), p@0.1% +71.4% และเสมอ
ที่ natural แต่มีข้อจำกัดสองข้อที่ต้องปิดก่อน ship:
  1. ultra cell เดียว
  2. XGBoost ใช้ config ตายตัว (300 trees, depth 6, scale_pos_weight balanced ~3,300)
     ที่อาจ overfit เมื่อมี positive 61 ตัว — อาจทำให้ XGBoost ดูแย่เกินจริง

**การจูน XGBoost แบบ PU ที่ทำได้จริง (ไม่แตะ test):** 3-fold CV บน seeds — กัน seeds
หนึ่ง fold ไว้ แล้ววัด average precision ของ seeds ที่กันไว้เทียบกับ background
ที่กันไว้ (ใช้แค่ข้อมูลที่ทีมจริงมี: confirmed positives + unlabeled) background ที่ใช้
เลือก config subsample 60k แถวเพื่อความเร็ว แล้ว refit config ที่ชนะบนข้อมูลเต็ม
grid: max_depth {2, 4, 6} x scale_pos_weight {balanced, sqrt(balanced)}, 300 trees, lr 0.1
leaf_naive **ไม่จูน** (ใช้ config เดิมที่ตรึงไว้: 80 trees x 1024 leaves) -> การเทียบนี้
เอื้อ XGBoost โดยตั้งใจ

---------------------------------------------------------------------------
PRE-REGISTERED
---------------------------------------------------------------------------
Cells (4 x 5 reps = 20 paired obs, seed draw ใหม่ทั้งหมด rng(15000 + 29*rep),
random_state = 500 + rep):
  fraud@0.3%       10% seeds (~61)  — replicate กับ XGBoost ที่จูนแล้ว
  fraud@0.1%       10% seeds (~20)  — ultra ที่ยากกว่า
  home_credit@1%   10% seeds (~97)  — dataset ที่สองใน regime rare
  criteo@1e-4      uniform 4M rows, group-hash split (seed 41001) train/test,
                   50% ของ train positives เป็น seeds — regime เรือธงของ LeadX
Variants: base (rare_pu), leaf_naive, hybrid (report-only), xgb_fixed (config เดิม),
          xgb_tuned (CV บน seeds)
PRIMARY : leaf_naive vs xgb_tuned
GATE    : pooled 20 obs median paired delta pr_auc > +2% AND ชนะ >= 14/20
          AND median paired delta precision@0.1% >= 0 AND ชนะ >= 3/5 ทุก cell
Secondary (รายงาน): leaf_naive vs base, xgb_tuned vs xgb_fixed (การจูนช่วย XGB แค่ไหน)

รัน: PYTHONPATH=outputs uv run --with xgboost python outputs/learned_vs_tuned_xgb.py
"""
import gc
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from criteo_uplift_benchmark import FEATURES, group_hash_splits, load_criteo
from extreme_imbalance_benchmark import keep_train_valid_columns, metrics, sample_ratio
from leadx import LeadX
from learned_partition_experiment import BASE, RF_KW, leaf_log_lift
from real_data_benchmark import load_dataset, split_data

N_REPS = 5
XGB_BASE = dict(tree_method="hist", n_estimators=300, learning_rate=0.1, n_jobs=8)
XGB_GRID = [(d, w) for d in (2, 4, 6) for w in ("bal", "sqrt")]
N_SELECT_BG = 60_000
CRITEO = Path("/Users/ponpiboonsatangput/jarvis/data/criteo-uplift/"
              "criteo-research-uplift-v2.1.csv.gz")


def prep_tabular(name, ratio):
    df, target, split_type, max_rows = load_dataset(name)
    train, test = split_data(df, target, split_type, max_rows)
    train = sample_ratio(train, target, ratio, 42)
    test = sample_ratio(test, target, ratio, 43)
    train, test = keep_train_valid_columns(train, test, target)
    y_tr = train[target].to_numpy().astype(np.float64)
    y_te = test[target].to_numpy().astype(np.float64)
    pos = np.flatnonzero(y_tr == 1)
    return (train.drop(target), y_tr, test.drop(target), y_te,
            max(2, int(0.10 * len(pos))))


def prep_criteo(ratio=1e-4, max_rows=4_000_000):
    frame, _ = load_criteo(CRITEO, max_rows, sample_seed=42, population="all")
    y = frame["conversion"].to_numpy().astype(np.int8)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = int(round(ratio * len(neg) / (1 - ratio)))
    keep = np.sort(np.concatenate([
        np.random.default_rng(2025).choice(pos, size=min(len(pos), n_pos), replace=False),
        neg]))
    sub, ysub = frame[keep.tolist()], y[keep]
    split, _ = group_hash_splits(sub, split_seed=41001)
    tr, te = split == "train", split == "test"
    X_tr = sub.filter(pl.Series(tr)).select(FEATURES)
    X_te = sub.filter(pl.Series(te)).select(FEATURES)
    y_tr, y_te = ysub[tr].astype(np.float64), ysub[te].astype(np.float64)
    return X_tr, y_tr, X_te, y_te, max(20, int(0.5 * (y_tr == 1).sum()))


def spw(n_pos, n_neg, mode):
    r = n_neg / max(1, n_pos)
    return r if mode == "bal" else float(np.sqrt(r))


def tune_xgb(Z, seeds, rng, xgb):
    """3-fold CV บน seeds: AP ของ seeds ที่กันไว้ เทียบกับ background ที่กันไว้"""
    is_seed = np.zeros(Z.shape[0], dtype=bool)
    is_seed[seeds] = True
    unl = np.flatnonzero(~is_seed)
    unl = rng.choice(unl, size=min(N_SELECT_BG, len(unl)), replace=False)
    seed_folds = np.array_split(rng.permutation(seeds), 3)
    unl_folds = np.array_split(rng.permutation(unl), 3)
    scores = {}
    for d, w in XGB_GRID:
        aps = []
        for f in range(3):
            tr_p = np.concatenate([seed_folds[g] for g in range(3) if g != f])
            tr_u = np.concatenate([unl_folds[g] for g in range(3) if g != f])
            idx = np.concatenate([tr_p, tr_u])
            lab = np.concatenate([np.ones(len(tr_p)), np.zeros(len(tr_u))])
            m = xgb.XGBClassifier(**XGB_BASE, max_depth=d,
                                  scale_pos_weight=spw(len(tr_p), len(tr_u), w))
            m.fit(Z[idx], lab)
            v_idx = np.concatenate([seed_folds[f], unl_folds[f]])
            v_lab = np.concatenate([np.ones(len(seed_folds[f])), np.zeros(len(unl_folds[f]))])
            aps.append(average_precision_score(v_lab, m.predict_proba(Z[v_idx])[:, 1]))
        scores[(d, w)] = float(np.mean(aps))
    return max(scores, key=scores.get), scores


def run_cell(cell, X_tr, y_tr, X_te, y_te, n_seeds, rows):
    import xgboost as xgb
    pos = np.flatnonzero(y_tr == 1)
    print(f"\n[{cell}] bg={X_tr.height:,} test={len(y_te):,} pos_train={len(pos):,} "
          f"pos_test={int(y_te.sum())} seeds={n_seeds}", flush=True)
    for rep in range(N_REPS):
        t0 = time.perf_counter()
        rng = np.random.default_rng(15000 + 29 * rep)
        sd = rng.choice(pos, size=n_seeds, replace=False)
        rs = 500 + rep
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

        s_all = np.zeros(X_tr.height)
        s_all[sd] = 1.0
        n_p, n_u = int(s_all.sum()), int(len(s_all) - s_all.sum())
        xgb_fixed = (xgb.XGBClassifier(**XGB_BASE, max_depth=6, random_state=rs,
                                       scale_pos_weight=spw(n_p, n_u, "bal"))
                     .fit(Z_bg, s_all).predict_proba(Z_te)[:, 1])
        (d_best, w_best), _ = tune_xgb(Z_bg, sd, rng, xgb)
        xgb_tuned = (xgb.XGBClassifier(**XGB_BASE, max_depth=d_best, random_state=rs,
                                       scale_pos_weight=spw(n_p, n_u, w_best))
                     .fit(Z_bg, s_all).predict_proba(Z_te)[:, 1])

        scores = {"base": base, "leaf_naive": naive, "hybrid": (np.log(base) + honest) / 2,
                  "xgb_fixed": xgb_fixed, "xgb_tuned": xgb_tuned}
        for v, sc in scores.items():
            rows.append({"cell": cell, "rep": rep, "variant": v,
                         "xgb_choice": f"d{d_best}-{w_best}", **metrics(y_te, sc)})
        print(f"  rep={rep} xgb_tuned=d{d_best}/{w_best} | " + " | ".join(
            f"{r['variant']} pr={r['pr_auc']:.4f}" for r in rows[-5:])
            + f" ({time.perf_counter()-t0:.0f}s)", flush=True)
        del model, Z_bg, Z_te
        gc.collect()


def paired(raw, ref, others):
    b = raw.filter(pl.col("variant") == ref).select(
        "cell", "rep", pl.col("pr_auc").alias("a0"), pl.col("precision_at_0_1pct").alias("z0"))
    j = (raw.filter(pl.col("variant").is_in(others)).join(b, on=["cell", "rep"])
         .with_columns((100 * (pl.col("pr_auc") - pl.col("a0")) / pl.col("a0")).alias("dpr"),
                       (100 * (pl.col("precision_at_0_1pct") - pl.col("z0"))
                        / pl.col("z0")).alias("d01")))
    agg = dict(n=pl.len(), wins=(pl.col("pr_auc") > pl.col("a0")).sum(),
               d_pr=pl.col("dpr").median().round(2), d_p01=pl.col("d01").median().round(2))
    cells = j.group_by(["variant", "cell"]).agg(**agg).sort(["variant", "cell"])
    mins = cells.group_by("variant").agg(min_cell_wins=pl.col("wins").min())
    return cells, j.group_by("variant").agg(**agg).join(mins, on="variant").sort("variant")


def main() -> None:
    rows: list[dict] = []
    run_cell("fraud@0.3%", *prep_tabular("fraud", 0.003), rows)
    run_cell("fraud@0.1%", *prep_tabular("fraud", 0.001), rows)
    run_cell("home_credit@1%", *prep_tabular("home_credit", 0.01), rows)
    run_cell("criteo@1e-4", *prep_criteo(), rows)
    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/learned_vs_tuned_xgb_raw.csv")

    print("\n=== ระดับ PR-AUC (median) ===")
    print(raw.group_by(["cell", "variant"]).agg(pr=pl.col("pr_auc").median().round(4))
          .pivot(on="variant", index="cell", values="pr"))
    print("\nXGB config ที่ CV เลือก:")
    print(raw.filter(pl.col("variant") == "xgb_tuned").group_by(["cell", "xgb_choice"])
          .len().sort(["cell", "xgb_choice"]))

    cells, pooled = paired(raw, "xgb_tuned", ["leaf_naive", "hybrid", "base"])
    print("\n=== PRIMARY: vs xgb_tuned ===")
    print(cells)
    print(pooled.with_columns(gate=(pl.col("variant") == "leaf_naive") & (pl.col("d_pr") > 2)
                              & (pl.col("wins") >= 14) & (pl.col("d_p01") >= 0)
                              & (pl.col("min_cell_wins") >= 3)))
    _, pooled = paired(raw, "base", ["leaf_naive"])
    print("\n=== secondary: leaf_naive vs base ===")
    print(pooled)
    _, pooled = paired(raw, "xgb_fixed", ["xgb_tuned"])
    print("\n=== secondary: xgb_tuned vs xgb_fixed (การจูนช่วย XGB แค่ไหน) ===")
    print(pooled)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n({time.time()-t0:.0f}s)")
