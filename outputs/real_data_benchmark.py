"""Reproducible real-data pilot benchmark for LeadX.

Source CSVs are read-only. Results are written beside this script.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from leadx import LeadX, LeadXBackbone
from leadx.encoder import FeatureEncoder


ROOT = Path("/Users/ponpiboonsatangput/jarvis")
OUT = Path(__file__).resolve().parent
SEED = 42


def _drop_all_null_and_constant(df: pl.DataFrame, target: str) -> pl.DataFrame:
    keep = [target]
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        if s.null_count() < df.height and s.n_unique() > 1:
            keep.append(c)
    return df.select(keep)


def load_dataset(name: str) -> tuple[pl.DataFrame, str, str, int]:
    """Return frame, target, split type, max rows."""
    if name == "telco":
        path = ROOT / "data/benchmarks/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        df = pl.read_csv(path, null_values=["", " "]).with_columns(
            pl.col("TotalCharges").cast(pl.Float64, strict=False),
            (pl.col("Churn") == "Yes").cast(pl.Int8).alias("target"),
        ).drop("Churn", "customerID")
        return _drop_all_null_and_constant(df, "target"), "target", "stratified", 20_000

    if name == "adult":
        path = ROOT / "data/benchmarks/adult.csv"
        df = pl.read_csv(path, null_values=["", "?"]).with_columns(
            (pl.col("income").str.strip_chars() == ">50K").cast(pl.Int8).alias("target")
        ).drop("income")
        return _drop_all_null_and_constant(df, "target"), "target", "stratified", 50_000

    if name == "hotel":
        path = ROOT / "data/benchmarks/hotel_bookings.csv"
        months = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12,
        }
        df = pl.read_csv(path, null_values=["", "NA"]).with_columns(
            pl.col("is_canceled").cast(pl.Int8).alias("target"),
            pl.col("arrival_date_month").replace_strict(months).cast(pl.Int8).alias("_month"),
        ).with_columns(
            pl.date("arrival_date_year", "_month", "arrival_date_day_of_month").alias("_split_date")
        ).sort("_split_date")
        # reservation_status and its date are direct outcome leakage.
        df = df.drop("is_canceled", "reservation_status", "reservation_status_date", "_month")
        return _drop_all_null_and_constant(df, "target"), "target", "temporal", 150_000

    if name == "home_credit":
        path = ROOT / "data/kaggle/home-credit/application_train.csv"
        df = pl.read_csv(path, null_values=["", "NA"]).rename({"TARGET": "target"})
        df = df.drop("SK_ID_CURR")
        return _drop_all_null_and_constant(df, "target"), "target", "stratified", 150_000

    if name == "fraud":
        path = ROOT / "data/kaggle/fraud-detection/train_transaction.csv"
        schema_names = pl.scan_csv(path).collect_schema().names()
        base = [
            "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
            "card1", "card2", "card3", "card4", "card5", "card6",
            "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain",
        ]
        base += [f"C{i}" for i in range(1, 15)]
        base += [f"D{i}" for i in range(1, 16)]
        base += [f"M{i}" for i in range(1, 10)]
        base += [f"V{i}" for i in range(1, 101)]
        cols = [c for c in base if c in schema_names]
        df = pl.read_csv(path, columns=cols, null_values=["", "NA"]).rename({"isFraud": "target"})
        df = df.sort("TransactionDT")
        return _drop_all_null_and_constant(df, "target"), "target", "temporal", 300_000

    raise ValueError(f"unknown dataset: {name}")


def split_data(df: pl.DataFrame, target: str, split_type: str, max_rows: int):
    rng = np.random.default_rng(SEED)
    y = df[target].to_numpy().astype(np.int8)
    if split_type == "temporal":
        if len(df) > max_rows:
            idx = np.linspace(0, len(df) - 1, max_rows, dtype=np.int64)
            df = df[idx]
        cut = int(0.70 * len(df))
        return df[:cut], df[cut:]

    selected_train, selected_test = [], []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        if len(df) > max_rows:
            keep = max(2, int(round(max_rows * len(idx) / len(df))))
            idx = idx[:keep]
        cut = int(0.70 * len(idx))
        selected_train.append(idx[:cut])
        selected_test.append(idx[cut:])
    tr = np.concatenate(selected_train)
    te = np.concatenate(selected_test)
    rng.shuffle(tr)
    rng.shuffle(te)
    return df[tr], df[te]


def ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    base = float(y.mean())
    out = {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "base_rate": base,
    }
    for frac in (0.01, 0.05):
        k = max(1, int(math.ceil(len(y) * frac)))
        idx = np.argpartition(score, -k)[-k:]
        precision = float(y[idx].mean())
        out[f"precision_at_{int(frac * 100)}pct"] = precision
        out[f"lift_at_{int(frac * 100)}pct"] = precision / base if base else float("nan")
    return out


def evaluate(name: str) -> list[dict]:
    df, target, split_type, max_rows = load_dataset(name)
    train, test = split_data(df, target, split_type, max_rows)
    y_train = train[target].to_numpy().astype(np.float64)
    y_test = test[target].to_numpy().astype(np.float64)
    X_train, X_test = train.drop(target), test.drop(target)
    bits = 10 if len(train) < 30_000 else 12
    tables = 10
    common = dict(tune=False, n_bits=bits, n_tables=tables, strategy="unsupervised",
                  hash_mode="median", cat_bit_frac="auto", random_state=SEED)
    rows: list[dict] = []
    meta = {
        "dataset": name, "split": split_type, "n_train": len(train), "n_test": len(test),
        "n_features_raw": X_train.width, "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()), "n_bits": bits, "n_tables": tables,
    }
    print("DATASET", json.dumps(meta), flush=True)

    def record(model: str, fit_seconds: float, predict_seconds: float, score: np.ndarray):
        row = {**meta, "model": model, "fit_seconds": fit_seconds,
               "predict_seconds": predict_seconds, **ranking_metrics(y_test, score)}
        rows.append(row)
        print("RESULT", json.dumps(row), flush=True)

    for score_type, agg in (("prop", "mean"), ("lift", "log")):
        t0 = time.perf_counter()
        clf = LeadX(mode="two_class", score_type=score_type, alpha=2.0, agg=agg, **common)
        clf.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        score = clf.score_samples(X_test)
        pred_s = time.perf_counter() - t0
        record(f"leadx_two_class_{score_type}", fit_s, pred_s, score)
        del clf, score
        gc.collect()

    # PU simulation: only confirmed positives are labels; full train is unlabeled background.
    t0 = time.perf_counter()
    bb = LeadXBackbone(n_bits=bits, n_tables=tables, hash_mode="median",
                       cat_bit_frac="auto", random_state=SEED).fit(X_train)
    pu = bb.fit_task(mode="pu", seeds=X_train.filter(pl.Series(y_train == 1)),
                     background=X_train, score_type="lift", alpha=2.0, agg="log")
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    score = pu.score_samples(X_test)
    pred_s = time.perf_counter() - t0
    record("leadx_pu_lift", fit_s, pred_s, score)
    del bb, pu, score
    gc.collect()

    # Strong tabular baseline on exactly the same encoded feature representation.
    t0 = time.perf_counter()
    enc = FeatureEncoder().fit(X_train)
    Z_train = enc.transform(X_train)
    Z_test = enc.transform(X_test)
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=150, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=SEED,
    ).fit(Z_train, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    score = hgb.predict_proba(Z_test)[:, 1]
    pred_s = time.perf_counter() - t0
    record("hist_gradient_boosting", fit_s, pred_s, score)
    del enc, Z_train, Z_test, hgb, score, train, test, df
    gc.collect()
    return rows


def evaluate_pu_sensitivity(name: str) -> list[dict]:
    """Measure PU ranking as only a fraction of positives are confirmed seeds."""
    df, target, split_type, max_rows = load_dataset(name)
    train, test = split_data(df, target, split_type, max_rows)
    y_train = train[target].to_numpy().astype(np.float64)
    y_test = test[target].to_numpy().astype(np.float64)
    X_train, X_test = train.drop(target), test.drop(target)
    bits = 10 if len(train) < 30_000 else 12
    tables = 10
    rng = np.random.default_rng(SEED)
    positives = np.flatnonzero(y_train == 1)
    rng.shuffle(positives)

    t0 = time.perf_counter()
    bb = LeadXBackbone(n_bits=bits, n_tables=tables, hash_mode="median",
                       cat_bit_frac="auto", random_state=SEED).fit(X_train)
    backbone_fit_s = time.perf_counter() - t0
    rows = []
    for frac in (0.10, 0.25, 0.50, 1.00):
        n_seed = max(20, int(round(len(positives) * frac)))
        seed_rows = positives[:n_seed]
        t0 = time.perf_counter()
        pu = bb.fit_task(mode="pu", seeds=X_train[seed_rows], background=X_train,
                         score_type="lift", alpha=2.0, agg="log")
        task_fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        score = pu.score_samples(X_test)
        pred_s = time.perf_counter() - t0
        row = {
            "dataset": name, "split": split_type, "confirmed_fraction": frac,
            "n_confirmed_seeds": n_seed, "n_train": len(train), "n_test": len(test),
            "positive_rate_test": float(y_test.mean()), "backbone_fit_seconds": backbone_fit_s,
            "task_fit_seconds": task_fit_s, "predict_seconds": pred_s,
            **ranking_metrics(y_test, score),
        }
        rows.append(row)
        print("PU_SENSITIVITY", json.dumps(row), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pu-sensitivity", action="store_true")
    parser.add_argument("datasets", nargs="*", default=["telco", "adult", "hotel", "home_credit", "fraud"])
    args = parser.parse_args()
    all_rows = []
    for name in args.datasets:
        try:
            if args.pu_sensitivity:
                all_rows.extend(evaluate_pu_sensitivity(name))
            else:
                all_rows.extend(evaluate(name))
        except Exception as exc:
            error = {"dataset": name, "error": f"{type(exc).__name__}: {exc}"}
            all_rows.append(error)
            print("ERROR", json.dumps(error), flush=True)
    prefix = "pu_sensitivity" if args.pu_sensitivity else "results"
    suffix = "_".join(args.datasets)
    pl.DataFrame(all_rows, infer_schema_length=None).write_csv(OUT / f"{prefix}_{suffix}.csv")
    (OUT / f"{prefix}_{suffix}.json").write_text(json.dumps(all_rows, indent=2))


if __name__ == "__main__":
    main()
