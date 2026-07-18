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
    baselines: tuple[str, str] = DEFAULT_BASELINES,
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
    duplicate_keys = (
        raw.group_by(["target", "n_seeds", "draw", "model"])
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
        baseline_draws = [
            cell.filter(pl.col("model") == model)["draw"].sort().to_list()
            for model in baselines
        ]
        if baseline_draws[0] != baseline_draws[1]:
            raise ValueError(f"{target}/{budget} baselines have different draw sets")
        if len(baseline_draws[0]) < 5:
            warnings.warn(
                f"{target}/{budget} has only {len(baseline_draws[0])} paired draws; "
                "bootstrap interval is descriptive, not reliable inference",
                stacklevel=2,
            )
        for metric_no, metric in enumerate(metrics):
            baseline_values = [
                cell.filter(pl.col("model") == model).sort("draw")[metric].to_numpy()
                for model in baselines
            ]
            baseline = np.maximum(
                baseline_values[0], baseline_values[1]
            )
            if not np.isfinite(baseline).all() or np.any(baseline <= 0):
                raise ValueError(f"{target}/{budget}/{metric} invalid baseline values")
            for model_no, model in enumerate(leadx_models):
                if model not in available:
                    continue
                model_cell = cell.filter(pl.col("model") == model).sort("draw")
                model_draws = model_cell["draw"].to_list()
                if model_draws != baseline_draws[0]:
                    raise ValueError(
                        f"{target}/{budget}/{model} has incomplete paired draws"
                    )
                values = model_cell[metric].to_numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f"{target}/{budget}/{metric}/{model} is non-finite")
                relative = 100.0 * (values / baseline - 1.0)
                rng = np.random.default_rng(
                    bootstrap_seed + 100 * metric_no + model_no
                )
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
                    "paired_relative_median_pct": float(np.median(relative)),
                    "bootstrap_lo_pct": float(lo),
                    "bootstrap_hi_pct": float(hi),
                    "wins": int((values > baseline).sum()),
                    "ties": int((values == baseline).sum()),
                })
    if not rows:
        raise ValueError("no LeadX models are comparable with the paired baselines")
    return pl.DataFrame(rows, infer_schema_length=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, nargs="+")
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in args.raw:
        result = paired_relative_summary(
            pl.read_csv(path), bootstrap_resamples=args.bootstrap_resamples
        )
        print(f"\n=== {path} ===")
        with pl.Config(tbl_rows=200, tbl_width_chars=220):
            print(result)


if __name__ == "__main__":
    main()
