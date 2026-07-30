"""Fully-supervised Criteo arena at 1:1,000 and 1:10,000 prevalence.

The experiment keeps every negative row and down-samples only positives after
feature-group splitting.  All models receive the same complete labels; this is
an imbalance-only test, not a PU benchmark.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from criteo_supervised_imbalance_benchmark import _hgb
from criteo_uplift_benchmark import (
    DEFAULT_DATA,
    EXPECTED_SHA256,
    FEATURES,
    _git_state,
    _sha256,
    _source_tree_sha256,
    _write_json_atomic,
    group_hash_splits,
    load_criteo,
)
from leadx import LeadX
from leadx.encoder import FeatureEncoder


TARGET_RATIOS = (0.001, 0.0001)
REVIEW_FRACTIONS = (0.0001, 0.001, 0.005)
LEADX_MODEL = "leadx_two_class_rpforest_corr"
BASELINES = ("logreg_full_balanced", "histgb_full_plain", "histgb_full_balanced")


def downsample_positives(
    frame: pl.DataFrame, target: str, ratio: float, seed: int
) -> pl.DataFrame:
    """Keep every negative and select a deterministic nested positive subset."""
    if not 0.0 < ratio < 1.0:
        raise ValueError("target ratio must be in (0, 1)")
    y = frame[target].to_numpy().astype(np.int8)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    wanted = int(round(ratio * len(negative) / (1.0 - ratio)))
    if wanted < 1:
        raise ValueError(f"target ratio leaves only {wanted} positives")
    if wanted > len(positive):
        raise ValueError(
            f"target ratio needs {wanted} positives but only {len(positive)} exist"
        )
    rng = np.random.default_rng(seed)
    selected_positive = rng.permutation(positive)[:wanted]
    selected = np.concatenate([negative, selected_positive])
    selected.sort()
    return frame[selected]


def ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Tie-aware rare-event metrics at operationally small review budgets."""
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    positives = int(y.sum())
    if positives == 0 or positives == len(y) or not np.isfinite(score).all():
        raise ValueError("metrics require finite scores and both classes")
    base = positives / len(y)
    result = {
        "base_rate": base,
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
    }
    for fraction in REVIEW_FRACTIONS:
        k = max(1, int(math.ceil(len(y) * fraction)))
        threshold = float(np.partition(score, len(score) - k)[len(score) - k])
        above = score > threshold
        tied = score == threshold
        remaining = k - int(above.sum())
        expected_hits = float(y[above].sum())
        if remaining:
            expected_hits += remaining * float(y[tied].mean())
        tag = str(fraction * 100).rstrip("0").rstrip(".").replace(".", "_")
        precision = expected_hits / k
        result[f"precision_at_{tag}pct"] = precision
        result[f"recall_at_{tag}pct"] = expected_hits / positives
        result[f"lift_at_{tag}pct"] = precision / base
    return result


def paired_summary(raw: pl.DataFrame, bootstrap_resamples: int = 100_000) -> pl.DataFrame:
    metrics = ("pr_auc", "recall_at_0_01pct", "recall_at_0_1pct", "recall_at_0_5pct")
    rows = []
    for ratio in sorted(raw["target_ratio"].unique().to_list(), reverse=True):
        cell = raw.filter(pl.col("target_ratio") == ratio)
        for metric_no, metric in enumerate(metrics):
            wide = cell.select("split_seed", "model", metric).pivot(
                on="model", index="split_seed", values=metric
            ).sort("split_seed")
            missing = sorted(set((LEADX_MODEL, *BASELINES)) - set(wide.columns))
            if missing:
                raise ValueError(f"ratio={ratio}/{metric} missing models: {missing}")
            baseline = np.maximum.reduce([wide[name].to_numpy() for name in BASELINES])
            leadx = wide[LEADX_MODEL].to_numpy()
            if metric == "pr_auc":
                effect = 100.0 * (leadx / baseline - 1.0)
                effect_unit = "relative_pct"
            else:
                # Recall can legitimately be zero at the tightest budget;
                # percentage-point differences stay defined in that case.
                effect = 100.0 * (leadx - baseline)
                effect_unit = "percentage_points"
            rng = np.random.default_rng(20_260_721 + 1000 * metric_no + int(ratio * 1e6))
            indices = rng.integers(
                0, len(effect), size=(bootstrap_resamples, len(effect))
            )
            boot = np.median(effect[indices], axis=1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            rows.append({
                "target_ratio": ratio,
                "imbalance_ratio": f"1:{round(1 / ratio):,}",
                "metric": metric,
                "model": LEADX_MODEL,
                "n_splits": len(effect),
                "effect_unit": effect_unit,
                "paired_median_effect": float(np.median(effect)),
                "bootstrap_lo_pct": float(lo),
                "bootstrap_hi_pct": float(hi),
                "wins": int((leadx > baseline).sum()),
                "ties": int((leadx == baseline).sum()),
            })
    return pl.DataFrame(rows, infer_schema_length=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-rows", type=int, default=10_000_000)
    parser.add_argument("--target-ratios", default="0.001,0.0001")
    parser.add_argument("--split-seeds", default="51001,51002,51003")
    parser.add_argument("--n-bits", type=int, default=10)
    parser.add_argument("--n-tables", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/criteo_ultra_imbalance_raw.csv"),
    )
    parser.add_argument("--skip-source-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = [float(value.strip()) for value in args.target_ratios.split(",") if value.strip()]
    split_seeds = [int(value.strip()) for value in args.split_seeds.split(",") if value.strip()]
    if not ratios or len(ratios) != len(set(ratios)):
        raise ValueError("target ratios must be non-empty and unique")
    if not split_seeds or len(split_seeds) != len(set(split_seeds)):
        raise ValueError("split seeds must be non-empty and unique")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"result exists: {args.output}; pass --overwrite")
    source_sha256 = None
    checksum_status = "skipped"
    if not args.skip_source_check:
        source_sha256 = _sha256(args.data)
        if args.data.resolve() == DEFAULT_DATA.resolve():
            if source_sha256 != EXPECTED_SHA256:
                raise ValueError(f"source SHA-256 mismatch: {source_sha256}")
            checksum_status = "matched_expected"
        else:
            checksum_status = "recorded_no_expected"

    started = time.time()
    frame, load_meta = load_criteo(args.data, args.max_rows, 42, "all")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_name(output.stem.replace("_raw", "") + "_metadata.json")
    summary_path = output.with_name(output.stem.replace("_raw", "") + "_summary.csv")
    rows: list[dict] = []
    metadata = {
        **load_meta,
        "data": str(args.data.resolve()),
        "source_sha256": source_sha256,
        "source_checksum_status": checksum_status,
        "command_argv": list(sys.argv),
        "benchmark_script_sha256": _sha256(Path(__file__).resolve()),
        "leadx_source_sha256": _source_tree_sha256(
            Path(__file__).resolve().parents[1] / "src/leadx"
        ),
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scikit_learn": sklearn.__version__,
        },
        **_git_state(Path(__file__).resolve().parents[1]),
        "estimand": "fully supervised ultra-rare ranking; not PU",
        "target": "conversion",
        "target_ratios": ratios,
        "split_seeds": split_seeds,
        "n_bits": args.n_bits,
        "n_tables": args.n_tables,
        "models": [LEADX_MODEL, *BASELINES],
        "created_unix": started,
        "cells": {},
    }
    _write_json_atomic(metadata_path, metadata)

    def checkpoint() -> None:
        tmp = output.with_suffix(output.suffix + ".tmp")
        pl.DataFrame(rows, infer_schema_length=None).write_csv(tmp)
        tmp.replace(output)

    for split_seed in split_seeds:
        split, split_meta = group_hash_splits(frame, split_seed)
        base_train = frame.filter(pl.Series(split == "train"))
        base_test = frame.filter(pl.Series(split == "test"))
        for ratio in ratios:
            train = downsample_positives(base_train, "conversion", ratio, split_seed + 11)
            test = downsample_positives(base_test, "conversion", ratio, split_seed + 29)
            x_train, x_test = train.select(FEATURES), test.select(FEATURES)
            y_train = train["conversion"].to_numpy().astype(np.int8)
            y_test = test["conversion"].to_numpy().astype(np.int8)
            common = {
                "dataset": "criteo_uplift_v2.1",
                "estimand": "fully_supervised_ultra_imbalance",
                "target_ratio": ratio,
                "imbalance_ratio": f"1:{round(1 / ratio):,}",
                "split_seed": split_seed,
                "n_train": len(train),
                "n_test": len(test),
                "n_positive_train": int(y_train.sum()),
                "n_positive_test": int(y_test.sum()),
                "positive_rate_train": float(y_train.mean()),
                "positive_rate_test": float(y_test.mean()),
            }
            metadata["cells"][f"{split_seed}:{ratio}"] = {
                "split": split_meta,
                **{key: common[key] for key in (
                    "n_train", "n_test", "n_positive_train", "n_positive_test",
                    "positive_rate_train", "positive_rate_test",
                )},
            }
            _write_json_atomic(metadata_path, metadata)

            def record(model: str, score: np.ndarray, fit_s: float, score_s: float) -> None:
                row = {
                    **common,
                    "model": model,
                    "fit_seconds": fit_s,
                    "score_seconds": score_s,
                    **ranking_metrics(y_test, score),
                }
                rows.append(row)
                checkpoint()
                print(
                    f"RESULT ratio={ratio:g} split={split_seed} {model} "
                    f"pr={row['pr_auc']:.6f} r@0.01%={row['recall_at_0_01pct']:.4f} "
                    f"r@0.1%={row['recall_at_0_1pct']:.4f} fit={fit_s:.1f}s",
                    flush=True,
                )

            gc.collect()
            t0 = time.perf_counter()
            model = LeadX(
                mode="two_class", tune=False, n_bits=args.n_bits,
                n_tables=args.n_tables, m=10.0, score_type="prop",
                feature_weighting="corr", engine="rpforest",
                hash_mode="median", cat_bit_frac=0.0,
                chunk_size=args.chunk_size, random_state=0,
            ).fit(x_train, y_train)
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            score = model.score_samples(x_test)
            record(LEADX_MODEL, score, fit_s, time.perf_counter() - t0)
            del model, score

            gc.collect()
            t0 = time.perf_counter()
            encoder = FeatureEncoder().fit(x_train)
            z_train = encoder.transform(x_train)
            z_test = encoder.transform(x_test)
            preprocess_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            model = LogisticRegression(
                max_iter=300, class_weight="balanced", solver="lbfgs", random_state=0
            ).fit(z_train, y_train)
            fit_s = preprocess_s + time.perf_counter() - t0
            t0 = time.perf_counter()
            score = model.predict_proba(z_test)[:, 1]
            record("logreg_full_balanced", score, fit_s, time.perf_counter() - t0)
            del model, score
            for class_weight, suffix in ((None, "plain"), ("balanced", "balanced")):
                t0 = time.perf_counter()
                model = _hgb(class_weight, split_seed).fit(z_train, y_train)
                fit_s = preprocess_s + time.perf_counter() - t0
                t0 = time.perf_counter()
                score = model.predict_proba(z_test)[:, 1]
                record(
                    f"histgb_full_{suffix}", score, fit_s,
                    time.perf_counter() - t0,
                )
                del model, score
            del encoder, z_train, z_test, train, test, x_train, x_test, y_train, y_test
            gc.collect()
        del base_train, base_test, split
        gc.collect()

    raw = pl.DataFrame(rows, infer_schema_length=None)
    summary = paired_summary(raw)
    summary.write_csv(summary_path)
    metadata["completed_unix"] = time.time()
    metadata["elapsed_seconds"] = time.time() - started
    metadata["n_result_rows"] = len(rows)
    _write_json_atomic(metadata_path, metadata)
    print("\n=== MEDIANS ===")
    print(
        raw.group_by(["target_ratio", "imbalance_ratio", "model"])
        .agg(
            pl.col("pr_auc").median().alias("pr_auc"),
            pl.col("recall_at_0_01pct").median().alias("recall_0_01pct"),
            pl.col("recall_at_0_1pct").median().alias("recall_0_1pct"),
            pl.col("recall_at_0_5pct").median().alias("recall_0_5pct"),
            pl.col("fit_seconds").median().alias("fit_seconds"),
        )
        .sort(["target_ratio", "pr_auc"], descending=[True, True])
    )
    print("\n=== PAIRED VS STRONGEST SUPERVISED BASELINE ===")
    print(summary)
    print(f"elapsed {metadata['elapsed_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
