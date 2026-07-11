import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from leadx import LeadX


def _blobs_df(n_per=400, d=6, sep=5.0, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n_per, d))
    X1 = rng.standard_normal((n_per, d)) + sep / np.sqrt(d)
    X = np.vstack([X0, X1])
    df = pl.DataFrame({f"f{i}": X[:, i] for i in range(d)})
    y = np.concatenate([np.zeros(n_per), np.ones(n_per)])
    return df, y


def test_fit_predict_separable_blobs():
    df, y = _blobs_df()
    clf = LeadX(mode="two_class", tune=False, n_bits=8, n_tables=5, random_state=0)
    clf.fit(df, y)
    proba = clf.predict_proba(df)
    assert proba.shape == (len(y), 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert roc_auc_score(y, proba[:, 1]) > 0.95


def test_fit_with_tune_records_results():
    df, y = _blobs_df(n_per=200)
    grid = {"n_bits": [4, 8], "n_tables": [2, 4], "m": [10.0],
            "weights": ["mean"], "strategy": ["unsupervised"]}
    clf = LeadX(mode="two_class", tune=True, grid=grid, random_state=0)
    clf.fit(df, y)
    assert clf.tuning_results_ is not None
    assert set(clf.config_) == {"strategy", "n_bits", "n_tables", "m", "weights"}
    assert "projections" in clf.diagnostics_


def test_pu_score_samples_normalized():
    rng = np.random.default_rng(1)
    pos = rng.standard_normal((300, 5)) + 2.0
    df_pos = pl.DataFrame({f"f{i}": pos[:, i] for i in range(5)})
    all_x = np.vstack([rng.standard_normal((500, 5)), pos[:100]])
    df_all = pl.DataFrame({f"f{i}": all_x[:, i] for i in range(5)})
    clf = LeadX(mode="pu", tune=False, n_bits=6, n_tables=4, random_state=0)
    clf.fit(df_pos)  # PU: y ไม่ต้องส่ง (all-positive)
    s = clf.score_samples(df_all)
    assert s.min() >= 0.0 and s.max() <= 1.0
    # แถวท้าย (มาจากบริเวณ positive) ควรได้ score สูงกว่าแถวแรกโดยเฉลี่ย
    assert s[500:].mean() > s[:500].mean()


def test_errors():
    df, y = _blobs_df(n_per=5)  # 10 แถว < 20
    with pytest.raises(ValueError, match="20"):
        LeadX(tune=False).fit(df, y)
    df2, y2 = _blobs_df(n_per=50)
    clf = LeadX(mode="pu", tune=False, n_bits=4, n_tables=2).fit(df2)
    with pytest.raises(ValueError, match="two_class"):
        clf.predict_proba(df2)
    clf2 = LeadX(mode="two_class", tune=False, n_bits=4, n_tables=2).fit(df2, y2)
    with pytest.raises(ValueError, match="columns mismatch"):
        clf2.predict_proba(df2.rename({"f0": "wrong"}))


def test_save_load_roundtrip(tmp_path):
    df, y = _blobs_df(n_per=100)
    clf = LeadX(mode="two_class", tune=False, n_bits=6, n_tables=3, random_state=0)
    clf.fit(df, y)
    expected = clf.predict_proba(df)
    clf.save(tmp_path / "model")
    loaded = LeadX.load(tmp_path / "model")
    np.testing.assert_allclose(loaded.predict_proba(df), expected, rtol=1e-6)
    assert loaded.config_ == clf.config_
