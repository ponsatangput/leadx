"""Pure supervised-imbalance arena on Criteo conversion.

Every model receives the same complete positive and negative labels.  This
deliberately removes PU/label-scarcity as a confound and asks whether LeadX has
an advantage from extreme class imbalance alone.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

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
    tie_aware_ranking_metrics,
)
from leadx import LeadX
from leadx.encoder import FeatureEncoder


LEADX_MODELS = (
    "leadx_two_class_hyperplane_corr",
    "leadx_two_class_rpforest_corr",
)
BASELINE_MODELS = (
    "logreg_full_balanced",
    "histgb_full_plain",
    "histgb_full_balanced",
)
METRICS = ("pr_auc", "recall_at_0_1pct", "recall_at_0_5pct")


def _hgb(class_weight, random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        class_weight=class_weight,
        early_stopping=False,
        random_state=random_state,
    )


def paired_summary(raw: pl.DataFrame, bootstrap_resamples: int = 100_000) -> pl.DataFrame:
    """Compare each LeadX arm with the strongest supervised baseline per split."""
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    needed = {"split_seed", "model", *METRICS}
    missing = sorted(needed - set(raw.columns))
    if missing:
        raise ValueError(f"raw columns missing: {missing}")
    duplicates = raw.group_by(["split_seed", "model"]).len().filter(pl.col("len") != 1)
    if duplicates.height:
        raise ValueError("raw must contain exactly one row per split/model")
    pivoted = {
        metric: raw.select("split_seed", "model", metric).pivot(
            on="model", index="split_seed", values=metric
        ).sort("split_seed")
        for metric in METRICS
    }
    rows = []
    for metric_no, metric in enumerate(METRICS):
        wide = pivoted[metric]
        available = set(wide.columns) - {"split_seed"}
        missing_models = sorted(set((*LEADX_MODELS, *BASELINE_MODELS)) - available)
        if missing_models:
            raise ValueError(f"{metric} missing models: {missing_models}")
        baseline = np.maximum.reduce([wide[name].to_numpy() for name in BASELINE_MODELS])
        for model_no, model in enumerate(LEADX_MODELS):
            values = wide[model].to_numpy()
            relative = 100.0 * (values / baseline - 1.0)
            rng = np.random.default_rng(20_260_721 + 100 * metric_no + model_no)
            indices = rng.integers(
                0, len(relative), size=(bootstrap_resamples, len(relative))
            )
            boot = np.median(relative[indices], axis=1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            rows.append({
                "metric": metric,
                "model": model,
                "n_splits": len(relative),
                "paired_relative_median_pct": float(np.median(relative)),
                "bootstrap_lo_pct": float(lo),
                "bootstrap_hi_pct": float(hi),
                "wins": int((values > baseline).sum()),
                "ties": int((values == baseline).sum()),
            })
    return pl.DataFrame(rows, infer_schema_length=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    parser.add_argument("--split-seeds", default="41001,41002,41003,41004,41005")
    parser.add_argument("--n-bits", type=int, default=10)
    parser.add_argument("--n-tables", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/criteo_supervised_imbalance_raw.csv"),
    )
    parser.add_argument("--skip-source-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_seeds = [int(value.strip()) for value in args.split_seeds.split(",") if value.strip()]
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
    rows: list[dict] = []
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_name(output.stem.replace("_raw", "") + "_metadata.json")
    summary_path = output.with_name(output.stem.replace("_raw", "") + "_summary.csv")
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
        "estimand": "fully supervised rare-event ranking; not PU",
        "target": "conversion",
        "split_seeds": split_seeds,
        "n_bits": args.n_bits,
        "n_tables": args.n_tables,
        "models": [*LEADX_MODELS, *BASELINE_MODELS],
        "created_unix": started,
        "split_details": {},
    }
    _write_json_atomic(metadata_path, metadata)

    def checkpoint() -> None:
        tmp = output.with_suffix(output.suffix + ".tmp")
        pl.DataFrame(rows, infer_schema_length=None).write_csv(tmp)
        tmp.replace(output)

    for split_seed in split_seeds:
        split, split_meta = group_hash_splits(frame, split_seed)
        train = frame.filter(pl.Series(split == "train"))
        test = frame.filter(pl.Series(split == "test"))
        x_train, x_test = train.select(FEATURES), test.select(FEATURES)
        y_train = train["conversion"].to_numpy().astype(np.int8)
        y_test = test["conversion"].to_numpy().astype(np.int8)
        common = {
            "dataset": "criteo_uplift_v2.1",
            "estimand": "fully_supervised_imbalance",
            "split_seed": split_seed,
            "n_train": len(train),
            "n_test": len(test),
            "n_positive_train": int(y_train.sum()),
            "n_positive_test": int(y_test.sum()),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
        }
        metadata["split_details"][str(split_seed)] = {
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
                **tie_aware_ranking_metrics(y_test, score),
            }
            rows.append(row)
            checkpoint()
            print(
                f"RESULT split={split_seed} {model} pr={row['pr_auc']:.6f} "
                f"r@0.1%={row['recall_at_0_1pct']:.4f} "
                f"r@0.5%={row['recall_at_0_5pct']:.4f} fit={fit_s:.2f}s",
                flush=True,
            )

        for engine in ("hyperplane", "rpforest"):
            gc.collect()
            t0 = time.perf_counter()
            model = LeadX(
                mode="two_class",
                tune=False,
                n_bits=args.n_bits,
                n_tables=args.n_tables,
                m=10.0,
                score_type="prop",
                feature_weighting="corr",
                engine=engine,
                hash_mode="median",
                cat_bit_frac=0.0,
                chunk_size=args.chunk_size,
                random_state=0,
            ).fit(x_train, y_train)
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            score = model.score_samples(x_test)
            score_s = time.perf_counter() - t0
            record(f"leadx_two_class_{engine}_corr", score, fit_s, score_s)
            del model, score

        gc.collect()
        t0 = time.perf_counter()
        encoder = FeatureEncoder().fit(x_train)
        z_train = encoder.transform(x_train)
        z_test = encoder.transform(x_test)
        preprocess_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        logreg = LogisticRegression(
            max_iter=300, class_weight="balanced", solver="lbfgs", random_state=0
        ).fit(z_train, y_train)
        fit_s = preprocess_s + time.perf_counter() - t0
        t0 = time.perf_counter()
        score = logreg.predict_proba(z_test)[:, 1]
        score_s = time.perf_counter() - t0
        record("logreg_full_balanced", score, fit_s, score_s)
        del logreg, score

        for class_weight, suffix in ((None, "plain"), ("balanced", "balanced")):
            t0 = time.perf_counter()
            model = _hgb(class_weight, split_seed).fit(z_train, y_train)
            fit_s = preprocess_s + time.perf_counter() - t0
            t0 = time.perf_counter()
            score = model.predict_proba(z_test)[:, 1]
            score_s = time.perf_counter() - t0
            record(f"histgb_full_{suffix}", score, fit_s, score_s)
            del model, score
        del encoder, z_train, z_test, train, test, x_train, x_test, y_train, y_test
        gc.collect()

    summary = paired_summary(pl.DataFrame(rows, infer_schema_length=None))
    summary.write_csv(summary_path)
    metadata["completed_unix"] = time.time()
    metadata["elapsed_seconds"] = time.time() - started
    metadata["n_result_rows"] = len(rows)
    _write_json_atomic(metadata_path, metadata)
    print("\n=== MEDIAN BY MODEL ===")
    print(
        pl.DataFrame(rows, infer_schema_length=None)
        .group_by("model")
        .agg(
            pl.col("pr_auc").median().alias("pr_auc"),
            pl.col("recall_at_0_1pct").median().alias("recall_0_1pct"),
            pl.col("recall_at_0_5pct").median().alias("recall_0_5pct"),
            pl.col("fit_seconds").median().alias("fit_seconds"),
            pl.col("score_seconds").median().alias("score_seconds"),
        )
        .sort("pr_auc", descending=True)
    )
    print("\n=== PAIRED VS STRONGEST SUPERVISED BASELINE PER SPLIT ===")
    print(summary)
    print(f"elapsed {metadata['elapsed_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
