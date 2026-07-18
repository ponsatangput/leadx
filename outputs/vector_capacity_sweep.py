"""Fraud PU capacity sweep: vectors per set (bits) × number of sets (tables)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from leadx import LeadXBackbone
from leadx.hashing import mask_codes
from leadx.scoring import SeedScorer
from real_data_benchmark import load_dataset, ranking_metrics, split_data


OUT = Path(__file__).resolve().parent
SEED = 42
BITS_GRID = (8, 10, 12, 14, 16)
TABLES_GRID = (10, 20, 40)
SEED_FRACTIONS = (0.10, 0.25)


def main() -> None:
    df, target, split_type, max_rows = load_dataset("fraud")
    train, test = split_data(df, target, split_type, max_rows)
    y_train = train[target].to_numpy().astype(np.float64)
    y_test = test[target].to_numpy().astype(np.float64)
    X_train, X_test = train.drop(target), test.drop(target)

    t0 = time.perf_counter()
    bb = LeadXBackbone(
        n_bits=max(BITS_GRID), n_tables=max(TABLES_GRID), hash_mode="median",
        cat_bit_frac="auto", random_state=SEED,
    ).fit(X_train)
    backbone_fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    test_codes_full = bb.transform(X_test)
    test_hash_s = time.perf_counter() - t0
    train_codes_full = bb._seed_codes

    rng = np.random.default_rng(SEED)
    positive_idx = np.flatnonzero(y_train == 1)
    rng.shuffle(positive_idx)
    rows = []
    for seed_frac in SEED_FRACTIONS:
        n_seed = int(round(len(positive_idx) * seed_frac))
        seed_idx = positive_idx[:n_seed]
        for bits in BITS_GRID:
            train_codes_b = mask_codes(train_codes_full, bits, max(BITS_GRID))
            test_codes_b = mask_codes(test_codes_full, bits, max(BITS_GRID))
            for tables in TABLES_GRID:
                bg_codes = train_codes_b[:, :tables]
                seed_codes = bg_codes[seed_idx]
                test_codes = test_codes_b[:, :tables]
                t0 = time.perf_counter()
                scorer = SeedScorer(
                    mode="pu", n_bits=bits, score_type="lift", alpha=2.0, agg="log",
                ).fit(seed_codes, np.ones(n_seed), background_codes=bg_codes)
                task_fit_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                score = scorer.score(test_codes)
                score_s = time.perf_counter() - t0
                row = {
                    "dataset": "fraud", "confirmed_fraction": seed_frac,
                    "n_confirmed_seeds": n_seed, "n_bits": bits, "n_tables": tables,
                    "backbone_fit_seconds_max_16x40": backbone_fit_s,
                    "test_hash_seconds_max_16x40": test_hash_s,
                    "task_fit_seconds": task_fit_s, "score_lookup_seconds": score_s,
                    "miss_rate_join": scorer.diagnostics_["miss_rate_join"],
                    "miss_rate_final": scorer.diagnostics_["miss_rate_final"],
                    **ranking_metrics(y_test, score),
                }
                rows.append(row)
                print("SWEEP", json.dumps(row), flush=True)

    result = pl.DataFrame(rows)
    result.write_csv(OUT / "vector_capacity_sweep_fraud.csv")
    (OUT / "vector_capacity_sweep_fraud.json").write_text(json.dumps(rows, indent=2))
    best = result.sort("pr_auc", descending=True).group_by("confirmed_fraction", maintain_order=True).first()
    print("BEST_BY_PR_AUC")
    print(best.select(
        "confirmed_fraction", "n_bits", "n_tables", "roc_auc", "pr_auc",
        "precision_at_1pct", "lift_at_1pct", "miss_rate_final",
    ))


if __name__ == "__main__":
    main()
