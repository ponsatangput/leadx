import time

import numpy as np
import polars as pl
import pytest

from leadx import LeadX


@pytest.mark.slow
def test_one_million_rows_under_60s():
    rng = np.random.default_rng(0)
    d = 50
    n_seed, n_pred = 20_000, 1_000_000
    X_seed = rng.standard_normal((n_seed, d))
    y = (X_seed[:, 0] + 0.5 * rng.standard_normal(n_seed) > 0).astype(np.float64)
    df_seed = pl.DataFrame({f"f{i}": X_seed[:, i] for i in range(d)})
    X_pred = rng.standard_normal((n_pred, d))
    df_pred = pl.DataFrame({f"f{i}": X_pred[:, i] for i in range(d)})

    t0 = time.perf_counter()
    clf = LeadX(mode="two_class", tune=False, n_bits=14, n_tables=10, random_state=0)
    clf.fit(df_seed, y)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = clf.predict_proba(df_pred)
    pred_s = time.perf_counter() - t0

    assert proba.shape == (n_pred, 2)
    assert not np.isnan(proba).any()
    print(f"\nfit: {fit_s:.1f}s, predict 1M rows: {pred_s:.1f}s")
    assert fit_s < 60.0
    assert pred_s < 60.0
