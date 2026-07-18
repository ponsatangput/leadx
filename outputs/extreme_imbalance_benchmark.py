"""Extreme-imbalance Fraud benchmark with fair full-label HistGB baselines."""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from leadx import LeadXBackbone
from leadx.encoder import FeatureEncoder
from real_data_benchmark import load_dataset, split_data


OUT = Path(__file__).resolve().parent
RATIOS = (0.03, 0.01, 0.003, 0.001)
SEED = 42


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    base = float(y.mean())
    out = {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "base_rate": base,
    }
    for frac in (0.001, 0.01):
        k = max(1, int(math.ceil(len(y) * frac)))
        idx = np.argpartition(score, -k)[-k:]
        precision = float(y[idx].mean())
        label = "0_1pct" if frac == 0.001 else "1pct"
        out[f"precision_at_{label}"] = precision
        out[f"lift_at_{label}"] = precision / base
    return out


def sample_ratio(df: pl.DataFrame, target: str, ratio: float, seed: int) -> pl.DataFrame:
    neg = df.filter(pl.col(target) == 0)
    pos = df.filter(pl.col(target) == 1)
    n_pos = min(len(pos), int(round(ratio * len(neg) / (1.0 - ratio))))
    pos = pos.sample(n=n_pos, shuffle=True, seed=seed)
    return pl.concat([neg, pos]).sample(fraction=1.0, shuffle=True, seed=seed)


def keep_train_valid_columns(train: pl.DataFrame, test: pl.DataFrame, target: str):
    keep = [c for c in train.columns if c == target or (
        train[c].null_count() < len(train) and train[c].n_unique() > 1
    )]
    return train.select(keep), test.select(keep)


def main() -> None:
    df, target, split_type, max_rows = load_dataset("fraud")
    train_full, test_full = split_data(df, target, split_type, max_rows)
    rows = []
    for ratio in RATIOS:
        train = sample_ratio(train_full, target, ratio, SEED)
        test = sample_ratio(test_full, target, ratio, SEED + 1)
        train, test = keep_train_valid_columns(train, test, target)
        y_train = train[target].to_numpy().astype(np.float64)
        y_test = test[target].to_numpy().astype(np.float64)
        X_train, X_test = train.drop(target), test.drop(target)
        meta = {
            "target_ratio": ratio, "n_train": len(train), "n_test": len(test),
            "n_positive_train": int(y_train.sum()), "n_positive_test": int(y_test.sum()),
            "n_features": X_train.width,
        }
        print("DATASET", json.dumps(meta), flush=True)

        def record(model: str, fit_s: float, pred_s: float, score: np.ndarray):
            row = {**meta, "model": model, "fit_seconds": fit_s,
                   "predict_seconds": pred_s, **metrics(y_test, score)}
            rows.append(row)
            print("RESULT", json.dumps(row), flush=True)

        # PU: only positives are labels; the full population is background.
        t0 = time.perf_counter()
        bb = LeadXBackbone(n_bits=10, n_tables=40, hash_mode="median",
                           cat_bit_frac="auto", random_state=SEED).fit(X_train)
        pu = bb.fit_task(mode="pu", seeds=X_train.filter(pl.Series(y_train == 1)),
                         background=X_train, score_type="lift", alpha=2.0, agg="log")
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        score = pu.score_samples(X_test)
        pred_s = time.perf_counter() - t0
        record("leadx_pu_lift_10x40", fit_s, pred_s, score)
        del bb, pu, score
        gc.collect()

        # Encode once; both HistGB variants receive all positive and negative labels.
        t0 = time.perf_counter()
        enc = FeatureEncoder().fit(X_train)
        Z_train = enc.transform(X_train)
        Z_test = enc.transform(X_test)
        encode_s = time.perf_counter() - t0
        for class_weight in (None, "balanced"):
            t0 = time.perf_counter()
            hgb = HistGradientBoostingClassifier(
                learning_rate=0.08, max_iter=150, max_leaf_nodes=31,
                l2_regularization=1.0, class_weight=class_weight, random_state=SEED,
            ).fit(Z_train, y_train)
            fit_s = encode_s + time.perf_counter() - t0
            t0 = time.perf_counter()
            score = hgb.predict_proba(Z_test)[:, 1]
            pred_s = time.perf_counter() - t0
            suffix = "balanced" if class_weight else "plain"
            record(f"hist_gradient_boosting_{suffix}", fit_s, pred_s, score)
            del hgb, score
            gc.collect()
        del enc, Z_train, Z_test, train, test, X_train, X_test
        gc.collect()

    pl.DataFrame(rows).write_csv(OUT / "extreme_imbalance_fraud.csv")
    (OUT / "extreme_imbalance_fraud.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
