"""imbalance-focus rerun ด้วย engine="rpforest" — อัพเดท positioning หลัง §8f

รันเฉพาะเงื่อนไข leadx (rpforest+hier) บน cells เดิมของ imbalance_focus_benchmark
(draw seeds เดียวกันเป๊ะ: rng(100*d+7)) แล้วรวมตารางกับ raw เดิม
(imbalance_focus_raw.csv) เพื่อเทียบ rpf vs bank vs PU-naive vs oracle

รัน: PYTHONPATH=outputs uv run python outputs/imbalance_focus_rpf.py
"""
import gc
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
RPF_KW = dict(mode="pu", tune=False, n_bits=10, n_tables=80, score_type="lift",
              agg="log", feature_weighting="contrast", engine="rpforest",
              hier_depths="auto", random_state=0)


def run_dataset(name, ratios):
    df, target, split_type, max_rows = load_dataset(name)
    train_full, test_full = split_data(df, target, split_type, max_rows)
    rows = []
    for ratio in ratios:
        train = sample_ratio(train_full, target, ratio, 42)
        test = sample_ratio(test_full, target, ratio, 43)
        train, test = keep_train_valid_columns(train, test, target)
        y_tr = train[target].to_numpy().astype(np.float64)
        y_te = test[target].to_numpy().astype(np.float64)
        X_tr, X_te = train.drop(target), test.drop(target)
        pos_idx = np.flatnonzero(y_tr == 1)
        n_seeds = max(2, int(SEED_FRAC * len(pos_idx)))

        def record(model, draw, score, fit_s):
            rows.append({"dataset": name, "target_ratio": ratio,
                         "n_positive_train": len(pos_idx), "n_seeds": n_seeds,
                         "model": model, "draw": draw,
                         "fit_seconds": round(fit_s, 2), **metrics(y_te, score)})
            print(f"  {name} r={ratio:.3%} {model} d={draw}: "
                  f"pr={rows[-1]['pr_auc']:.4f} roc={rows[-1]['roc_auc']:.3f} "
                  f"({fit_s:.0f}s)", flush=True)

        t0 = time.perf_counter()
        clf = LeadX(**RPF_KW).fit(X_tr[pos_idx.tolist()], background=X_tr)
        record("leadx_rpf_allseeds", -1, clf.score_samples(X_te),
               time.perf_counter() - t0)
        del clf
        gc.collect()
        for d in range(DRAWS):
            rd = np.random.default_rng(100 * d + 7)   # draw เดียวกับ benchmark เดิม
            seeds = rd.choice(pos_idx, size=n_seeds, replace=False)
            t0 = time.perf_counter()
            clf = LeadX(**RPF_KW).fit(X_tr[seeds.tolist()], background=X_tr)
            record("leadx_rpf_10pct", d, clf.score_samples(X_te),
                   time.perf_counter() - t0)
            del clf
            gc.collect()
    return rows


if __name__ == "__main__":
    t0 = time.time()
    rows = run_dataset("fraud", (0.03, 0.01, 0.003, 0.001))
    rows += run_dataset("home_credit", (0.03, 0.01, 0.003))
    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/imbalance_focus_rpf_raw.csv")
    old = pl.read_csv("outputs/imbalance_focus_raw.csv")
    both = pl.concat([old, raw.select(old.columns)], how="vertical_relaxed")
    med = (both.group_by(["dataset", "target_ratio", "model"])
           .agg(pr=pl.col("pr_auc").median(), roc=pl.col("roc_auc").median(),
                lift01=pl.col("lift_at_0_1pct").median())
           .sort(["dataset", "target_ratio", "model"],
                 descending=[False, True, False]))
    print("\n=== median (รวม raw เดิม) ===")
    with pl.Config(tbl_rows=60):
        print(med)
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min → outputs/imbalance_focus_rpf_raw.csv")
