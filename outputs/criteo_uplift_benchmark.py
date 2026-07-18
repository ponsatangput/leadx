"""Criteo Uplift rare-outcome + reusable-backbone benchmark.

This is deliberately *not* an uplift/ITE benchmark.  It uses only the twelve
pre-treatment anonymised features and asks a positive-only retrieval question:
given a handful of confirmed outcomes, rank held-out users by conversion/visit.

The benchmark separates two LeadX claims that must not be conflated:

1. reusable geometry: build/hash once, then fit many cheap task heads;
2. task-specific accuracy: contrast-weighted geometry rebuilt for every task.

Default command is the 1M-row pilot on validation groups::

    uv run python outputs/criteo_uplift_benchmark.py --overwrite

Quick wiring check::

    uv run python outputs/criteo_uplift_benchmark.py \
      --max-rows 50000 --draws 1 --seed-budgets 20 \
      --n-tables 10 --output outputs/criteo_uplift_smoke_raw.csv --overwrite

The source CSV is read-only.  The raw result CSV is checkpointed atomically
after every completed model so an interrupted long run still leaves valid data.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from leadx import LeadX, LeadXBackbone
from leadx.encoder import FeatureEncoder


ROOT = Path("/Users/ponpiboonsatangput/jarvis")
DEFAULT_DATA = ROOT / "data/criteo-uplift/criteo-research-uplift-v2.1.csv.gz"
EXPECTED_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
FEATURES = tuple(f"f{i}" for i in range(12))
LABELS = ("treatment", "conversion", "visit", "exposure")
REQUIRED = FEATURES + LABELS
REVIEW_FRACTIONS = (0.001, 0.005, 0.01)
TARGET_SEED_OFFSETS = {"conversion": 0, "visit": 10_000}


def _csv_values(value: str, cast=str) -> list:
    if not value.strip():
        return []
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_tree_sha256(path: Path) -> str:
    """Hash Python source paths + contents so dirty-worktree runs stay auditable."""
    digest = hashlib.sha256()
    for source in sorted(path.rglob("*.py")):
        digest.update(source.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _git_state(path: Path) -> dict[str, str | bool | None]:
    """Best-effort code revision provenance; never blocks a benchmark run."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "."], cwd=path, check=True,
            capture_output=True, text=True,
        ).stdout
        return {"git_revision": revision, "leadx_worktree_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": None, "leadx_worktree_dirty": None}


def _validate_frame(df: pl.DataFrame) -> None:
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Criteo columns missing: {missing}")
    nulls = df.select(pl.all().null_count()).row(0)
    bad_nulls = {name: count for name, count in zip(df.columns, nulls) if count}
    if bad_nulls:
        raise ValueError(f"Criteo contains nulls: {bad_nulls}")
    for column in LABELS:
        values = set(df[column].unique().to_list())
        if not values <= {0, 1}:
            raise ValueError(f"{column} must be binary, got {sorted(values)}")


def _require_unique(name: str, values: list) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"{name} must not contain duplicates: {duplicates}")


def load_criteo(
    path: Path,
    max_rows: int | None,
    sample_seed: int,
    population: str,
) -> tuple[pl.DataFrame, dict]:
    """Load source and take a uniform row sample without label resampling."""
    schema = {column: pl.Float32 for column in FEATURES}
    schema.update({column: pl.Int8 for column in LABELS})
    if path.suffix == ".parquet":
        df = pl.read_parquet(path, columns=list(REQUIRED))
    else:
        df = pl.read_csv(path, columns=list(REQUIRED), schema_overrides=schema)
    _validate_frame(df)
    source_rows = df.height
    source_rates = {target: float(df[target].mean()) for target in LABELS}
    if population == "treated":
        df = df.filter(pl.col("treatment") == 1)
    elif population != "all":
        raise ValueError(f"unknown population: {population}")
    population_rows = df.height
    population_rates = {target: float(df[target].mean()) for target in LABELS}
    sampled = bool(max_rows and max_rows < df.height)
    if sampled:
        # Uniform over rows and independent of every label.  We do not force the
        # rare-event rate to equal the source rate.
        df = df.sample(n=int(max_rows), shuffle=True, seed=sample_seed)
    df = df.rechunk()
    meta = {
        "source_rows": source_rows,
        "source_rates": source_rates,
        "population": population,
        "population_rows": population_rows,
        "population_rates": population_rates,
        "sampled": sampled,
        "sample_rows": df.height,
        "sample_rates": {target: float(df[target].mean()) for target in LABELS},
        "sample_seed": sample_seed,
    }
    return df, meta


def group_hash_splits(df: pl.DataFrame, split_seed: int) -> tuple[np.ndarray, dict]:
    """Assign identical feature vectors to the same 60/20/20 split."""
    hashes = df.select(FEATURES).hash_rows(seed=split_seed).to_numpy()
    fold = hashes % np.uint64(10)
    split = np.full(df.height, "test", dtype="U5")
    split[fold < 6] = "train"
    split[(fold >= 6) & (fold < 8)] = "valid"
    counts = {name: int((split == name).sum()) for name in ("train", "valid", "test")}
    meta = {
        "split_seed": split_seed,
        "split_method": "feature-row hash 60/20/20; exact feature duplicates stay together",
        "counts": counts,
        "unique_feature_hashes": int(np.unique(hashes).size),
        "duplicate_row_fraction": float(1.0 - np.unique(hashes).size / len(hashes)),
    }
    return split, meta


def make_seed_draws(
    y_train: dict[str, np.ndarray],
    targets: Iterable[str],
    budgets: Iterable[int],
    draws: int,
) -> dict[tuple[str, int, int], np.ndarray]:
    """Nested fixed-budget seeds; identical indices are reused by every model."""
    budgets = sorted(set(int(budget) for budget in budgets))
    if not budgets or budgets[0] < 20:
        raise ValueError("seed budgets must be >=20 for parity with LeadX facade")
    out: dict[tuple[str, int, int], np.ndarray] = {}
    for target in targets:
        positives = np.flatnonzero(y_train[target] == 1)
        if len(positives) < budgets[-1]:
            raise ValueError(
                f"{target} train positives={len(positives)} < max seed budget={budgets[-1]}"
            )
        for draw in range(draws):
            rng = np.random.default_rng(TARGET_SEED_OFFSETS[target] + 100 * draw + 7)
            ordered = rng.permutation(positives)
            for budget in budgets:
                out[(target, budget, draw)] = ordered[:budget]
    return out


def tie_aware_ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """PR metrics plus expected top-k metrics under random tie-breaking."""
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    if y.ndim != 1 or score.shape != y.shape:
        raise ValueError(f"metric shape mismatch: y={y.shape}, score={score.shape}")
    if not np.isfinite(score).all():
        raise ValueError("scores contain NaN/inf")
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        raise ValueError("evaluation target must contain both classes")
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
        n_above, n_tied = int(above.sum()), int(tied.sum())
        remaining = k - n_above
        expected_hits = float(y[above].sum())
        if remaining:
            expected_hits += remaining * float(y[tied].mean())
        precision = expected_hits / k
        tag = str(fraction * 100).rstrip("0").rstrip(".").replace(".", "_")
        result.update({
            f"precision_at_{tag}pct": precision,
            f"recall_at_{tag}pct": expected_hits / positives,
            f"lift_at_{tag}pct": precision / base,
            f"tie_share_at_{tag}pct": n_tied / len(y),
            f"tie_fill_at_{tag}pct": remaining / n_tied if n_tied else 0.0,
        })
    return result


def exact_nearest_seed_score(
    z_eval: np.ndarray,
    z_seed: np.ndarray,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Negative squared distance to the closest confirmed seed, chunked."""
    z_eval = np.asarray(z_eval, dtype=np.float32)
    z_seed = np.asarray(z_seed, dtype=np.float32)
    seed_norm = np.einsum("ij,ij->i", z_seed, z_seed)
    out = np.empty(z_eval.shape[0], dtype=np.float32)
    for start in range(0, z_eval.shape[0], chunk_size):
        stop = min(start + chunk_size, z_eval.shape[0])
        block = z_eval[start:stop]
        distances = (
            np.einsum("ij,ij->i", block, block)[:, None]
            + seed_norm[None, :]
            - 2.0 * (block @ z_seed.T)
        )
        out[start:stop] = -distances.min(axis=1)
    return out


def bagging_logreg_pu_score(
    z_train: np.ndarray,
    z_eval: np.ndarray,
    seed_rows: np.ndarray,
    n_estimators: int,
    unlabeled_size: int,
    random_state: int,
) -> tuple[np.ndarray, float, float]:
    """Bagging-PU adaptation: positives vs random unlabeled subsamples.

    Mordelet & Vert (2014) proposed aggregating classifiers trained on all known
    positives and small random subsets of the unlabeled pool.  Their base learner
    was an SVM; this dependency-free adaptation uses balanced logistic regression
    and reports its name explicitly rather than claiming to be the original model.
    """
    seed_rows = np.asarray(seed_rows, dtype=np.int64)
    if n_estimators < 1 or unlabeled_size < 1:
        raise ValueError("bagging estimators/unlabeled_size must be positive")
    is_seed = np.zeros(len(z_train), dtype=bool)
    is_seed[seed_rows] = True
    unlabeled = np.flatnonzero(~is_seed)
    sample_size = min(int(unlabeled_size), len(unlabeled))
    rng = np.random.default_rng(random_state)
    models = []
    t0 = time.perf_counter()
    y_sub = np.concatenate([
        np.ones(len(seed_rows), dtype=np.int8),
        np.zeros(sample_size, dtype=np.int8),
    ])
    for estimator in range(n_estimators):
        negative_rows = rng.choice(unlabeled, size=sample_size, replace=False)
        train_rows = np.concatenate([seed_rows, negative_rows])
        model = LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state + estimator,
        ).fit(z_train[train_rows], y_sub)
        models.append(model)
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    score = np.zeros(len(z_eval), dtype=np.float64)
    for model in models:
        score += model.predict_proba(z_eval)[:, 1]
    score /= len(models)
    score_seconds = time.perf_counter() - t0
    return score, fit_seconds, score_seconds


class ResultWriter:
    def __init__(self, path: Path, overwrite: bool):
        self.path = path
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"result exists: {path}; pass --overwrite")
            # Do not leave an old raw file paired with metadata from a new run
            # if the process fails before the first checkpoint.
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def append(self, row: dict) -> None:
        self.rows.append(row)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        pl.DataFrame(self.rows, infer_schema_length=None).write_csv(tmp)
        tmp.replace(self.path)
        print(
            f"RESULT {row['model']} {row['target']} seeds={row['n_seeds']} "
            f"draw={row['draw']} pr={row['pr_auc']:.6f} "
            f"r@0.5%={row['recall_at_0_5pct']:.4f} "
            f"fit={row['task_fit_seconds']:.2f}s score={row['score_seconds']:.2f}s",
            flush=True,
        )


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _base_row(
    run_meta: dict,
    target: str,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    model: str,
    scope: str,
    engine: str,
    n_seeds: int,
    draw: int,
) -> dict:
    return {
        "dataset": "criteo_uplift_v2.1",
        "population": run_meta["population"],
        "sample_rows": run_meta["sample_rows"],
        "split": run_meta["eval_split"],
        "n_train": len(y_train),
        "n_eval": len(y_eval),
        "target": target,
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_eval": float(y_eval.mean()),
        "n_positive_train": int(y_train.sum()),
        "n_positive_eval": int(y_eval.sum()),
        "model": model,
        "geometry_scope": scope,
        "engine": engine,
        "n_bits": run_meta["n_bits"],
        "n_tables": run_meta["n_tables"],
        "n_seeds": n_seeds,
        "draw": draw,
        "shared_fit_seconds": 0.0,
        "shared_transform_seconds": 0.0,
        "shared_background_seconds": 0.0,
        "preprocess_seconds": 0.0,
        "task_fit_seconds": 0.0,
        "score_seconds": 0.0,
    }


def _record(
    writer: ResultWriter,
    base: dict,
    y_eval: np.ndarray,
    score: np.ndarray,
    **timings: float,
) -> None:
    row = {**base, **timings, **tie_aware_ranking_metrics(y_eval, score)}
    writer.append(row)


def run_frozen_engine(
    engine: str,
    x_train: pl.DataFrame,
    x_eval: pl.DataFrame,
    y_train: dict[str, np.ndarray],
    y_eval: dict[str, np.ndarray],
    targets: list[str],
    budgets: list[int],
    draws: int,
    seed_draws: dict,
    args: argparse.Namespace,
    run_meta: dict,
    writer: ResultWriter,
) -> None:
    gc.collect()
    t0 = time.perf_counter()
    backbone = LeadXBackbone(
        n_bits=args.n_bits,
        n_tables=args.n_tables,
        engine=engine,
        hash_mode="median",
        whiten=False,
        cat_bit_frac=0.0,
        chunk_size=args.chunk_size,
        random_state=0,
    ).fit(x_train)
    shared_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    eval_codes = backbone.transform(x_eval)
    shared_transform = time.perf_counter() - t0

    # Build task-independent background statistics once, outside per-task timing.
    warm_target, warm_budget = targets[0], budgets[0]
    t0 = time.perf_counter()
    backbone.fit_task(
        mode="pu",
        seed_rows=seed_draws[(warm_target, warm_budget, 0)],
        background="cached",
        score_type="lift",
        alpha=2.0,
        agg="log",
        hier_depths="auto",
    )
    shared_background = time.perf_counter() - t0

    for target in targets:
        for budget in budgets:
            for draw in range(draws):
                rows = seed_draws[(target, budget, draw)]
                t0 = time.perf_counter()
                task = backbone.fit_task(
                    mode="pu",
                    seed_rows=rows,
                    background="cached",
                    score_type="lift",
                    alpha=2.0,
                    agg="log",
                    hier_depths="auto",
                )
                task_fit = time.perf_counter() - t0
                t0 = time.perf_counter()
                score = task.score_codes(eval_codes)
                score_seconds = time.perf_counter() - t0
                base = _base_row(
                    run_meta, target, y_train[target], y_eval[target],
                    f"leadx_frozen_{engine}", "reusable", engine, budget, draw,
                )
                _record(
                    writer, base, y_eval[target], score,
                    shared_fit_seconds=shared_fit,
                    shared_transform_seconds=shared_transform,
                    shared_background_seconds=shared_background,
                    task_fit_seconds=task_fit,
                    score_seconds=score_seconds,
                )
                del task, score
    del eval_codes, backbone
    gc.collect()


def run_task_specific(
    engine: str,
    x_train: pl.DataFrame,
    x_eval: pl.DataFrame,
    y_train: dict[str, np.ndarray],
    y_eval: dict[str, np.ndarray],
    targets: list[str],
    budgets: list[int],
    draws: int,
    seed_draws: dict,
    args: argparse.Namespace,
    run_meta: dict,
    writer: ResultWriter,
) -> None:
    for target in targets:
        for budget in budgets:
            for draw in range(draws):
                rows = seed_draws[(target, budget, draw)]
                gc.collect()
                t0 = time.perf_counter()
                model = LeadX(
                    mode="pu",
                    tune=False,
                    n_bits=args.n_bits,
                    n_tables=args.n_tables,
                    score_type="lift",
                    alpha=2.0,
                    agg="log",
                    feature_weighting="contrast",
                    engine=engine,
                    hash_mode="median",
                    whiten=False,
                    cat_bit_frac=0.0,
                    hier_depths="auto",
                    chunk_size=args.chunk_size,
                    random_state=0,
                ).fit(x_train[rows.tolist()], background=x_train)
                task_fit = time.perf_counter() - t0
                t0 = time.perf_counter()
                score = model.score_samples(x_eval)
                score_seconds = time.perf_counter() - t0
                base = _base_row(
                    run_meta, target, y_train[target], y_eval[target],
                    f"leadx_contrast_{engine}", "task_specific", engine, budget, draw,
                )
                _record(
                    writer, base, y_eval[target], score,
                    task_fit_seconds=task_fit,
                    score_seconds=score_seconds,
                )
                del model, score
                gc.collect()


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


def run_baselines(
    x_train: pl.DataFrame,
    x_eval: pl.DataFrame,
    y_train: dict[str, np.ndarray],
    y_eval: dict[str, np.ndarray],
    targets: list[str],
    budgets: list[int],
    draws: int,
    seed_draws: dict,
    models: list[str],
    args: argparse.Namespace,
    run_meta: dict,
    writer: ResultWriter,
) -> None:
    gc.collect()
    t0 = time.perf_counter()
    encoder = FeatureEncoder().fit(x_train)
    z_train = encoder.transform(x_train)
    z_eval = encoder.transform(x_eval)
    preprocess = time.perf_counter() - t0

    for target in targets:
        if "histgb_oracle" in models:
            t0 = time.perf_counter()
            oracle = _hgb(class_weight="balanced", random_state=0).fit(
                z_train, y_train[target]
            )
            fit_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            score = oracle.predict_proba(z_eval)[:, 1]
            score_seconds = time.perf_counter() - t0
            base = _base_row(
                run_meta, target, y_train[target], y_eval[target],
                "histgb_oracle_full_balanced", "full_label_reference", "tree",
                int(y_train[target].sum()), -1,
            )
            _record(
                writer, base, y_eval[target], score,
                preprocess_seconds=preprocess,
                task_fit_seconds=fit_seconds,
                score_seconds=score_seconds,
            )
            del oracle, score

        for budget in budgets:
            for draw in range(draws):
                rows = seed_draws[(target, budget, draw)]
                if "exact_seed" in models:
                    t0 = time.perf_counter()
                    score = exact_nearest_seed_score(z_eval, z_train[rows])
                    score_seconds = time.perf_counter() - t0
                    base = _base_row(
                        run_meta, target, y_train[target], y_eval[target],
                        "exact_nearest_seed", "seed_similarity", "exact", budget, draw,
                    )
                    _record(
                        writer, base, y_eval[target], score,
                        preprocess_seconds=preprocess,
                        score_seconds=score_seconds,
                    )
                    del score

                y_pu = np.zeros(len(z_train), dtype=np.int8)
                y_pu[rows] = 1
                if "logreg_pu" in models:
                    t0 = time.perf_counter()
                    model = LogisticRegression(
                        max_iter=300,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=draw,
                    ).fit(z_train, y_pu)
                    fit_seconds = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    score = model.predict_proba(z_eval)[:, 1]
                    score_seconds = time.perf_counter() - t0
                    base = _base_row(
                        run_meta, target, y_train[target], y_eval[target],
                        "logreg_pu_naive", "pu_naive", "linear", budget, draw,
                    )
                    _record(
                        writer, base, y_eval[target], score,
                        preprocess_seconds=preprocess,
                        task_fit_seconds=fit_seconds,
                        score_seconds=score_seconds,
                    )
                    del model, score

                if "bagging_logreg_pu" in models:
                    score, fit_seconds, score_seconds = bagging_logreg_pu_score(
                        z_train,
                        z_eval,
                        rows,
                        n_estimators=args.bagging_estimators,
                        unlabeled_size=args.bagging_unlabeled_size,
                        random_state=draw,
                    )
                    base = _base_row(
                        run_meta, target, y_train[target], y_eval[target],
                        "bagging_logreg_pu", "bagging_pu", "linear", budget, draw,
                    )
                    _record(
                        writer, base, y_eval[target], score,
                        preprocess_seconds=preprocess,
                        task_fit_seconds=fit_seconds,
                        score_seconds=score_seconds,
                    )
                    del score

                if "histgb_pu" in models:
                    t0 = time.perf_counter()
                    model = _hgb(class_weight="balanced", random_state=draw).fit(
                        z_train, y_pu
                    )
                    fit_seconds = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    score = model.predict_proba(z_eval)[:, 1]
                    score_seconds = time.perf_counter() - t0
                    base = _base_row(
                        run_meta, target, y_train[target], y_eval[target],
                        "histgb_pu_naive", "pu_naive", "tree", budget, draw,
                    )
                    _record(
                        writer, base, y_eval[target], score,
                        preprocess_seconds=preprocess,
                        task_fit_seconds=fit_seconds,
                        score_seconds=score_seconds,
                    )
                    del model, score
                del y_pu
                gc.collect()
    del encoder, z_train, z_eval
    gc.collect()


def _summary(rows: list[dict]) -> pl.DataFrame:
    raw = pl.DataFrame(rows, infer_schema_length=None)
    return (
        raw.group_by(["target", "n_seeds", "model"])
        .agg(
            pl.col("pr_auc").median().alias("pr_median"),
            pl.col("recall_at_0_5pct").median().alias("recall05_median"),
            pl.col("lift_at_0_5pct").median().alias("lift05_median"),
            pl.col("task_fit_seconds").median().alias("task_fit_s_median"),
            pl.col("score_seconds").median().alias("score_s_median"),
        )
        .sort(["target", "n_seeds", "pr_median"], descending=[False, False, True])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/criteo_uplift_pilot_raw.csv"))
    parser.add_argument("--max-rows", type=int, default=1_000_000,
                        help="uniform sample size; 0 means all rows")
    parser.add_argument("--population", choices=("all", "treated"), default="all")
    parser.add_argument("--eval-split", choices=("valid", "test"), default="valid")
    parser.add_argument("--targets", default="conversion,visit")
    parser.add_argument("--seed-budgets", default="20,100")
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--n-bits", type=int, default=10)
    parser.add_argument("--n-tables", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--frozen-engines", default="hyperplane,rpforest")
    parser.add_argument("--task-specific-engines", default="hyperplane,rpforest")
    parser.add_argument("--task-specific-targets", default="conversion")
    parser.add_argument("--baseline-models",
                        default=("exact_seed,logreg_pu,bagging_logreg_pu,"
                                 "histgb_pu,histgb_oracle"))
    parser.add_argument("--baseline-targets", default="conversion")
    parser.add_argument("--bagging-estimators", type=int, default=30)
    parser.add_argument("--bagging-unlabeled-size", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=2025)
    parser.add_argument("--skip-source-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.draws < 1:
        raise ValueError("draws must be >=1")
    max_rows = None if args.max_rows == 0 else args.max_rows
    targets = _csv_values(args.targets)
    budgets = sorted(set(_csv_values(args.seed_budgets, int)))
    frozen_engines = _csv_values(args.frozen_engines)
    task_engines = _csv_values(args.task_specific_engines)
    task_targets = _csv_values(args.task_specific_targets)
    baseline_models = _csv_values(args.baseline_models)
    baseline_targets = _csv_values(args.baseline_targets)
    if not targets or not set(targets) <= {"conversion", "visit"}:
        raise ValueError("targets must be conversion and/or visit")
    for name, values in (
        ("targets", targets),
        ("frozen engines", frozen_engines),
        ("task-specific engines", task_engines),
        ("task-specific targets", task_targets),
        ("baseline models", baseline_models),
        ("baseline targets", baseline_targets),
    ):
        _require_unique(name, values)
    unknown_task_targets = set(task_targets) - set(targets)
    if unknown_task_targets:
        raise ValueError(f"task-specific targets not in --targets: {unknown_task_targets}")
    unknown_baseline_targets = set(baseline_targets) - set(targets)
    if unknown_baseline_targets:
        raise ValueError(f"baseline targets not in --targets: {unknown_baseline_targets}")
    if not set(frozen_engines + task_engines) <= {"hyperplane", "rpforest"}:
        raise ValueError("engines must be hyperplane/rpforest")
    known_baselines = {
        "exact_seed", "logreg_pu", "bagging_logreg_pu", "histgb_pu", "histgb_oracle"
    }
    if not set(baseline_models) <= known_baselines:
        raise ValueError(f"unknown baseline: {set(baseline_models) - known_baselines}")
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    source_sha256 = None
    source_checksum_status = "skipped"
    if not args.skip_source_check:
        source_sha256 = _sha256(args.data)
        if args.data.resolve() == DEFAULT_DATA.resolve():
            if source_sha256 != EXPECTED_SHA256:
                raise ValueError(f"source SHA-256 mismatch: {source_sha256}")
            source_checksum_status = "matched_expected"
        else:
            source_checksum_status = "recorded_no_expected"

    started = time.time()
    frame, load_meta = load_criteo(args.data, max_rows, args.sample_seed, args.population)
    split, split_meta = group_hash_splits(frame, args.split_seed)
    train_mask, eval_mask = split == "train", split == args.eval_split
    train = frame.filter(pl.Series(train_mask))
    evaluation = frame.filter(pl.Series(eval_mask))
    if not len(train) or not len(evaluation):
        raise ValueError("empty train/evaluation split")
    x_train, x_eval = train.select(FEATURES), evaluation.select(FEATURES)
    y_train = {target: train[target].to_numpy().astype(np.int8) for target in targets}
    y_eval = {target: evaluation[target].to_numpy().astype(np.int8) for target in targets}
    seed_draws = make_seed_draws(y_train, targets, budgets, args.draws)
    run_meta = {
        **load_meta,
        "data": str(args.data.resolve()),
        "source_sha256": source_sha256,
        "source_checksum_status": source_checksum_status,
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
        "eval_split": args.eval_split,
        "split": split_meta,
        "split_rates": {
            name: {target: float(frame[target].to_numpy()[split == name].mean())
                   for target in targets}
            for name in ("train", "valid", "test")
        },
        "features": list(FEATURES),
        "excluded_columns": list(LABELS),
        "estimand": "rare-outcome retrieval; not uplift/ITE",
        "targets": targets,
        "seed_budgets": budgets,
        "draws": args.draws,
        "n_bits": args.n_bits,
        "n_tables": args.n_tables,
        "frozen_engines": frozen_engines,
        "task_specific_engines": task_engines,
        "task_specific_targets": task_targets,
        "baseline_models": baseline_models,
        "baseline_targets": baseline_targets,
        "bagging_estimators": args.bagging_estimators,
        "bagging_unlabeled_size": args.bagging_unlabeled_size,
        "n_train": len(train),
        "n_eval": len(evaluation),
        "created_unix": started,
    }
    output = args.output.resolve()
    writer = ResultWriter(output, args.overwrite)
    meta_path = output.with_name(output.stem.replace("_raw", "") + "_metadata.json")
    if meta_path.exists() and not args.overwrite:
        raise FileExistsError(f"metadata exists: {meta_path}; pass --overwrite")
    _write_json_atomic(meta_path, run_meta)
    print("DATASET", json.dumps(run_meta, sort_keys=True), flush=True)

    for engine in frozen_engines:
        run_frozen_engine(
            engine, x_train, x_eval, y_train, y_eval, targets, budgets, args.draws,
            seed_draws, args, run_meta, writer,
        )
    for engine in task_engines:
        run_task_specific(
            engine, x_train, x_eval, y_train, y_eval, task_targets, budgets, args.draws,
            seed_draws, args, run_meta, writer,
        )
    if baseline_targets and baseline_models:
        run_baselines(
            x_train, x_eval, y_train, y_eval, baseline_targets, budgets, args.draws,
            seed_draws, baseline_models, args, run_meta, writer,
        )

    run_meta["completed_unix"] = time.time()
    run_meta["elapsed_seconds"] = time.time() - started
    run_meta["n_result_rows"] = len(writer.rows)
    _write_json_atomic(meta_path, run_meta)
    print("\n=== MEDIAN ACROSS PAIRED DRAWS ===")
    with pl.Config(tbl_rows=100, tbl_width_chars=180):
        print(_summary(writer.rows))
    print(f"\nraw -> {output}")
    print(f"metadata -> {meta_path}")
    print(f"elapsed {run_meta['elapsed_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
