"""Paired analysis for Criteo rare-outcome benchmark raw artifacts.

Each LeadX draw is compared with the stronger of the two linear PU baselines
from the same seed draw.  This deliberately avoids ratio-of-medians, which can
pair values produced by different draws and inflate an apparent effect.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import polars as pl


DEFAULT_LEADX_MODELS = (
    "leadx_frozen_hyperplane",
    "leadx_frozen_rpforest",
    "leadx_contrast_hyperplane",
    "leadx_contrast_rpforest",
)
DEFAULT_BASELINES = ("logreg_pu_naive", "bagging_logreg_pu")
DEFAULT_METRICS = ("pr_auc", "recall_at_0_5pct")


def paired_relative_summary(
    raw: pl.DataFrame,
    leadx_models: tuple[str, ...] = DEFAULT_LEADX_MODELS,
    baselines: tuple[str, ...] = DEFAULT_BASELINES,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    bootstrap_resamples: int = 100_000,
    bootstrap_seed: int = 20_260_718,
) -> pl.DataFrame:
    """Return paired median relative deltas and draw-bootstrap intervals."""
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    needed = {"target", "n_seeds", "model", "draw", *metrics}
    missing = sorted(needed - set(raw.columns))
    if missing:
        raise ValueError(f"raw columns missing: {missing}")
    if not baselines:
        raise ValueError("at least one baseline is required")
    pair_columns = ["split_seed", "draw"] if "split_seed" in raw.columns else ["draw"]
    duplicate_keys = (
        raw.group_by(["target", "n_seeds", *pair_columns, "model"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicate_keys.height:
        raise ValueError("raw contains duplicate target/budget/draw/model rows")
    paired = raw.filter(pl.col("draw") >= 0)
    rows: list[dict] = []
    groups = paired.select(["target", "n_seeds"]).unique().sort(
        ["target", "n_seeds"]
    )
    for target, budget in groups.iter_rows():
        cell = paired.filter(
            (pl.col("target") == target) & (pl.col("n_seeds") == budget)
        )
        available = set(cell["model"].unique().to_list())
        if not set(baselines) <= available:
            missing_baselines = sorted(set(baselines) - available)
            raise ValueError(
                f"{target}/{budget} missing paired baselines: {missing_baselines}"
            )
        baseline_pairs = [
            cell.filter(pl.col("model") == model)
            .select(pair_columns)
            .sort(pair_columns)
            .rows()
            for model in baselines
        ]
        if any(pairs != baseline_pairs[0] for pairs in baseline_pairs[1:]):
            unit = "pair" if "split_seed" in pair_columns else "draw"
            raise ValueError(f"{target}/{budget} baselines have different {unit} sets")
        n_pairs = len(baseline_pairs[0])
        n_splits = (
            cell["split_seed"].n_unique() if "split_seed" in pair_columns else 1
        )
        if n_pairs < 5:
            unit = "paired observations" if "split_seed" in pair_columns else "paired draws"
            warnings.warn(
                f"{target}/{budget} has only {n_pairs} {unit}; "
                "bootstrap interval is descriptive, not reliable inference",
                stacklevel=2,
            )
        if "split_seed" in pair_columns and n_splits < 3:
            warnings.warn(
                f"{target}/{budget} has only {n_splits} independent splits; "
                "hierarchical interval is descriptive",
                stacklevel=2,
            )
        for metric_no, metric in enumerate(metrics):
            baseline_values = [
                cell.filter(pl.col("model") == model)
                .sort(pair_columns)[metric]
                .to_numpy()
                for model in baselines
            ]
            baseline = np.maximum.reduce(baseline_values)
            if not np.isfinite(baseline).all() or np.any(baseline <= 0):
                raise ValueError(f"{target}/{budget}/{metric} invalid baseline values")
            for model_no, model in enumerate(leadx_models):
                if model not in available:
                    continue
                model_cell = cell.filter(pl.col("model") == model).sort(pair_columns)
                model_pairs = model_cell.select(pair_columns).rows()
                if model_pairs != baseline_pairs[0]:
                    raise ValueError(
                        f"{target}/{budget}/{model} has incomplete paired observations"
                    )
                values = model_cell[metric].to_numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f"{target}/{budget}/{metric}/{model} is non-finite")
                relative = 100.0 * (values / baseline - 1.0)
                rng = np.random.default_rng(
                    bootstrap_seed + 100 * metric_no + model_no
                )
                split_wins = None
                if "split_seed" in pair_columns:
                    split_values = model_cell["split_seed"].to_numpy()
                    unique_splits, split_counts = np.unique(
                        split_values, return_counts=True
                    )
                    if len(set(split_counts.tolist())) != 1:
                        raise ValueError(
                            f"{target}/{budget}/{model} has unequal draws per split"
                        )
                    draws_per_split = int(split_counts[0])
                    relative_by_split = relative.reshape(len(unique_splits), draws_per_split)
                    selected_splits = rng.integers(
                        0, len(unique_splits),
                        size=(bootstrap_resamples, len(unique_splits)),
                    )
                    selected = relative_by_split[selected_splits]
                    selected_draws = rng.integers(
                        0, draws_per_split,
                        size=(bootstrap_resamples, len(unique_splits), draws_per_split),
                    )
                    boot = np.median(
                        np.take_along_axis(selected, selected_draws, axis=2),
                        axis=(1, 2),
                    )
                    split_medians = np.median(relative_by_split, axis=1)
                    split_wins = int((split_medians > 0).sum())
                else:
                    indices = rng.integers(
                        0, len(relative), size=(bootstrap_resamples, len(relative))
                    )
                    boot = np.median(relative[indices], axis=1)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                rows.append({
                    "target": target,
                    "n_seeds": budget,
                    "metric": metric,
                    "model": model,
                    "n_draws": len(relative),
                    "n_splits": int(n_splits),
                    "paired_relative_median_pct": float(np.median(relative)),
                    "bootstrap_lo_pct": float(lo),
                    "bootstrap_hi_pct": float(hi),
                    "wins": int((values > baseline).sum()),
                    "ties": int((values == baseline).sum()),
                    "split_wins": split_wins,
                })
    if not rows:
        raise ValueError("no LeadX models are comparable with the paired baselines")
    return pl.DataFrame(rows, infer_schema_length=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, nargs="+")
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument(
        "--baselines", default=",".join(DEFAULT_BASELINES),
        help="comma-separated baseline model names; strongest is selected per pair",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baselines = tuple(item.strip() for item in args.baselines.split(",") if item.strip())
    for path in args.raw:
        result = paired_relative_summary(
            pl.read_csv(path), baselines=baselines,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        print(f"\n=== {path} ===")
        with pl.Config(tbl_rows=200, tbl_width_chars=220):
            print(result)


if __name__ == "__main__":
    main()
