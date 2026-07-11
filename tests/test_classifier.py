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


def test_feature_weighting_downweights_noise_and_helps():
    # 6 signal + 54 noise features — weighting ควรกู้ signal ที่ถูก noise กลบ
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=6000, n_features=60, n_informative=6,
                               n_redundant=0, n_repeated=0, class_sep=1.5,
                               flip_y=0.01, shuffle=False, random_state=0)
    Xtr, Xte, ytr, yte = X[:4000], X[4000:], y[:4000], y[4000:]
    dtr = pl.DataFrame({f"f{i}": Xtr[:, i] for i in range(60)})
    dte = pl.DataFrame({f"f{i}": Xte[:, i] for i in range(60)})
    base = LeadX(mode="two_class", tune=False, n_bits=10, n_tables=20,
                 random_state=0).fit(dtr, ytr.astype(float))
    wtd = LeadX(mode="two_class", tune=False, n_bits=10, n_tables=20,
                feature_weighting="corr", random_state=0).fit(dtr, ytr.astype(float))
    auc_base = roc_auc_score(yte, base.predict_proba(dte)[:, 1])
    auc_wtd = roc_auc_score(yte, wtd.predict_proba(dte)[:, 1])
    assert auc_wtd > auc_base + 0.03  # weighting กู้ signal ที่ noise กลบ
    # signal features (6 ตัวแรก, shuffle=False) ควรได้น้ำหนักเฉลี่ยสูงกว่า noise
    w = wtd.feature_weights_
    assert w[:6].mean() > 3 * w[6:].mean()


def test_feature_weighting_persists_through_save_load(tmp_path):
    df, y = _blobs_df(n_per=100)
    clf = LeadX(mode="two_class", tune=False, n_bits=6, n_tables=3,
                feature_weighting="corr", random_state=0).fit(df, y)
    expected = clf.predict_proba(df)
    clf.save(tmp_path / "m")
    loaded = LeadX.load(tmp_path / "m")
    np.testing.assert_allclose(loaded.predict_proba(df), expected, rtol=1e-6)


def test_pu_scores_batch_invariant():
    # bug: min-max normalize ด้วยสถิติของ batch → แถวเดียวกันได้ score ต่างกันตาม batch
    rng = np.random.default_rng(2)
    pos = rng.standard_normal((200, 5)) + 2.0
    df_pos = pl.DataFrame({f"f{i}": pos[:, i] for i in range(5)})
    Xq = np.vstack([rng.standard_normal((300, 5)), pos[:50]])
    df_q = pl.DataFrame({f"f{i}": Xq[:, i] for i in range(5)})
    clf = LeadX(mode="pu", tune=False, n_bits=6, n_tables=4, random_state=0).fit(df_pos)
    whole = clf.score_samples(df_q)
    halves = np.concatenate([clf.score_samples(df_q[:100]), clf.score_samples(df_q[100:])])
    np.testing.assert_allclose(whole, halves, rtol=1e-9)


def test_two_class_rejects_single_class_y():
    df, _ = _blobs_df(n_per=30)
    with pytest.raises(ValueError, match="class"):
        LeadX(mode="two_class", tune=False, n_bits=4, n_tables=2).fit(df, np.ones(60))
