import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
import pytest

SCRIPT = Path(__file__).parents[1] / "outputs/criteo_uplift_benchmark.py"
SPEC = importlib.util.spec_from_file_location("criteo_uplift_benchmark", SCRIPT)
BENCHMARK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCHMARK)

SUMMARY_SCRIPT = Path(__file__).parents[1] / "outputs/summarize_criteo_uplift.py"
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "summarize_criteo_uplift", SUMMARY_SCRIPT
)
SUMMARY = importlib.util.module_from_spec(SUMMARY_SPEC)
assert SUMMARY_SPEC.loader is not None
SUMMARY_SPEC.loader.exec_module(SUMMARY)

FEATURES = BENCHMARK.FEATURES
exact_nearest_seed_score = BENCHMARK.exact_nearest_seed_score
bagging_logreg_pu_score = BENCHMARK.bagging_logreg_pu_score
group_hash_splits = BENCHMARK.group_hash_splits
make_seed_draws = BENCHMARK.make_seed_draws
tie_aware_ranking_metrics = BENCHMARK.tie_aware_ranking_metrics
benchmark_summary = BENCHMARK._summary
ResultWriter = BENCHMARK.ResultWriter
paired_relative_summary = SUMMARY.paired_relative_summary
require_unique = BENCHMARK._require_unique


def _frame(x: np.ndarray) -> pl.DataFrame:
    data = {name: x[:, i] for i, name in enumerate(FEATURES)}
    n = len(x)
    data.update({
        "treatment": np.zeros(n, dtype=np.int8),
        "conversion": np.zeros(n, dtype=np.int8),
        "visit": np.zeros(n, dtype=np.int8),
        "exposure": np.zeros(n, dtype=np.int8),
    })
    return pl.DataFrame(data)


def test_group_hash_split_keeps_exact_feature_duplicates_together():
    rng = np.random.default_rng(0)
    unique = rng.normal(size=(60, len(FEATURES)))
    x = np.repeat(unique, 3, axis=0)
    split, meta = group_hash_splits(_frame(x), split_seed=2025)
    for start in range(0, len(x), 3):
        assert len(set(split[start:start + 3])) == 1
    assert sum(meta["counts"].values()) == len(x)
    assert meta["duplicate_row_fraction"] > 0.6


def test_seed_draws_are_nested_and_reused_by_key():
    y = np.zeros(100, dtype=np.int8)
    y[:60] = 1
    draws = make_seed_draws({"conversion": y}, ["conversion"], [20, 40], 2)
    for draw in range(2):
        small = draws[("conversion", 20, draw)]
        large = draws[("conversion", 40, draw)]
        np.testing.assert_array_equal(small, large[:20])
        assert y[large].all()


def test_seed_draws_do_not_depend_on_target_argument_order():
    conversion = np.zeros(120, dtype=np.int8)
    conversion[:70] = 1
    visit = np.zeros(120, dtype=np.int8)
    visit[30:110] = 1
    labels = {"conversion": conversion, "visit": visit}
    forward = make_seed_draws(labels, ["conversion", "visit"], [20], 2)
    reverse = make_seed_draws(labels, ["visit", "conversion"], [20], 2)
    for target in labels:
        for draw in range(2):
            np.testing.assert_array_equal(
                forward[(target, 20, draw)], reverse[(target, 20, draw)]
            )


def test_tie_aware_metrics_do_not_use_arbitrary_row_order():
    y = np.array([1, 0, 1, 0], dtype=np.int8)
    score = np.ones(4)
    metrics = tie_aware_ranking_metrics(y, score)
    # Every review budget rounds up to one row.  With a four-way tie, expected
    # precision is prevalence rather than whichever row happens to come first.
    assert metrics["precision_at_0_5pct"] == 0.5
    assert metrics["lift_at_0_5pct"] == 1.0
    assert metrics["recall_at_0_5pct"] == 0.25
    assert metrics["tie_share_at_0_5pct"] == 1.0


def test_exact_nearest_seed_score_orders_by_closest_distance():
    z_eval = np.array([[0.0, 0.0], [1.0, 1.0], [4.5, 4.5]], dtype=np.float32)
    z_seed = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float32)
    score = exact_nearest_seed_score(z_eval, z_seed, chunk_size=2)
    assert score[0] > score[1]
    assert score[2] > score[1]


def test_bagging_logreg_pu_score_is_deterministic_and_finite():
    rng = np.random.default_rng(2)
    z_train = rng.normal(size=(200, 4)).astype(np.float32)
    z_eval = rng.normal(size=(30, 4)).astype(np.float32)
    seeds = np.arange(20)
    first = bagging_logreg_pu_score(z_train, z_eval, seeds, 3, 40, 9)[0]
    second = bagging_logreg_pu_score(z_train, z_eval, seeds, 3, 40, 9)[0]
    np.testing.assert_allclose(first, second)
    assert np.isfinite(first).all()
    assert ((first >= 0) & (first <= 1)).all()


def test_summary_keeps_full_label_reference_row():
    common = {
        "target": "conversion",
        "pr_auc": 0.1,
        "recall_at_0_5pct": 0.2,
        "lift_at_0_5pct": 3.0,
        "task_fit_seconds": 1.0,
        "score_seconds": 2.0,
    }
    summary = benchmark_summary([
        {**common, "n_seeds": 20, "model": "leadx", "draw": 0},
        {**common, "n_seeds": 1000, "model": "oracle", "draw": -1},
    ])
    assert set(summary["model"].to_list()) == {"leadx", "oracle"}


def test_result_writer_overwrite_removes_stale_raw_immediately(tmp_path):
    output = tmp_path / "raw.csv"
    output.write_text("stale")
    ResultWriter(output, overwrite=True)
    assert not output.exists()


def test_paired_summary_uses_stronger_baseline_within_each_draw():
    rows = []
    leadx = [0.30, 0.20, 0.45]
    first = [0.20, 0.25, 0.30]
    second = [0.25, 0.20, 0.50]
    for draw in range(3):
        common = {
            "target": "conversion", "n_seeds": 20, "draw": draw,
            "recall_at_0_5pct": 0.1,
        }
        rows.extend([
            {**common, "model": "leadx_frozen_hyperplane", "pr_auc": leadx[draw]},
            {**common, "model": "logreg_pu_naive", "pr_auc": first[draw]},
            {**common, "model": "bagging_logreg_pu", "pr_auc": second[draw]},
        ])
    with pytest.warns(UserWarning, match="only 3 paired draws"):
        result = paired_relative_summary(
            pl.DataFrame(rows), metrics=("pr_auc",), bootstrap_resamples=100
        ).row(0, named=True)
    # Per-draw stronger baselines are [.25, .25, .50], so relative deltas are
    # [+20%, -20%, -10%] and the paired median is -10%.
    assert np.isclose(result["paired_relative_median_pct"], -10.0)
    assert result["wins"] == 1


def test_paired_summary_rejects_incomplete_or_duplicate_draws():
    rows = []
    for draw in range(5):
        for model in (
            "leadx_frozen_hyperplane", "logreg_pu_naive", "bagging_logreg_pu"
        ):
            rows.append({
                "target": "conversion", "n_seeds": 20, "draw": draw,
                "model": model, "pr_auc": 0.2, "recall_at_0_5pct": 0.1,
            })
    incomplete = pl.DataFrame(rows[:-1])
    with pytest.raises(ValueError, match="baselines have different draw sets"):
        paired_relative_summary(incomplete, bootstrap_resamples=10)
    duplicated = pl.DataFrame([*rows, rows[0]])
    with pytest.raises(ValueError, match="duplicate"):
        paired_relative_summary(duplicated, bootstrap_resamples=10)


def test_benchmark_defaults_include_paired_summary_baselines(monkeypatch):
    monkeypatch.setattr(BENCHMARK.sys, "argv", ["criteo_uplift_benchmark.py"])
    args = BENCHMARK.parse_args()
    assert "logreg_pu" in args.baseline_models
    assert "bagging_logreg_pu" in args.baseline_models


def test_duplicate_cli_components_are_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        require_unique("engines", ["rpforest", "rpforest"])
