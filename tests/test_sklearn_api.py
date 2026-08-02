"""LeadXClassifier — XGBoost-shaped interface (fit(X,y), any frame type,
sklearn compat, PU semantics, background policy, persistence)"""
import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from leadx import LeadX, LeadXClassifier
from leadx.frames import to_polars


def _pu_data(n=3000, seed=0, confirmed_frac=0.5):
    """y_true = จริง; y = confirmed subset (ที่เหลือ unlabeled) — PU แท้"""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y_true = (X[:, 0] + X[:, 1] > 2.0).astype(int)
    y = y_true.copy()
    y[rng.random(n) > confirmed_frac] = 0
    df = pl.DataFrame({f"f{i}": X[:, i] for i in range(5)})
    return df, X, y, y_true


FAST = dict(profile="fast", n_tables=10, n_bits=6)


# ------------------------------------------------------------- adapter ---
def test_to_polars_accepts_common_frames():
    X = np.arange(12, dtype=float).reshape(4, 3)
    cols = ["a", "b", "c"]
    ref = pl.DataFrame(X, schema=cols)
    for obj in (ref, ref.lazy(), {c: X[:, i] for i, c in enumerate(cols)}):
        assert to_polars(obj).equals(ref)
    assert to_polars(X, columns=cols).equals(ref)
    assert to_polars(X).columns == ["f0", "f1", "f2"]        # default names


def test_to_polars_validation_and_reorder():
    ref = pl.DataFrame({"a": [1.0], "b": [2.0]})
    assert to_polars(ref, columns=["b", "a"]).columns == ["b", "a"]   # จัดลำดับตาม fit
    with pytest.raises(ValueError, match="ขาดคอลัมน์"):
        to_polars(ref, columns=["a", "zzz"])
    with pytest.raises(ValueError, match="2 มิติ"):
        to_polars(np.zeros(5))


# ----------------------------------------------------------- PU semantics ---
def test_fit_xy_pu_learns_signal_and_ignores_nothing():
    df, _, y, y_true = _pu_data()
    m = LeadXClassifier(**FAST).fit(df, y)
    p = m.predict_proba(df)[:, 1]
    assert p.shape == (df.height,) and 0.0 <= p.min() and p.max() <= 1.0
    assert roc_auc_score(y_true, p) > 0.85          # เจอ positive ที่ยังไม่ยืนยันด้วย


def test_core_leadx_rejects_silently_ignored_labels():
    """footgun: LeadX(mode='pu').fit(X, y) เคยรันผ่านแล้วทิ้ง y เงียบๆ"""
    df, _, y, _ = _pu_data(n=400)
    with pytest.raises(ValueError, match="unlabeled|LeadXClassifier"):
        LeadX(mode="pu", tune=False, n_bits=6, n_tables=4).fit(df, y=y)


def test_fit_validation_errors():
    df, _, y, _ = _pu_data(n=1200)
    with pytest.raises(ValueError, match="อย่างน้อย 20"):
        LeadXClassifier(**FAST).fit(df, np.zeros(df.height))
    with pytest.raises(ValueError, match="unlabeled"):
        LeadXClassifier(**FAST).fit(df, np.ones(df.height))
    with pytest.raises(ValueError, match="y มี"):
        LeadXClassifier(**FAST).fit(df, y[:10])
    with pytest.raises(ValueError, match="0/1"):
        LeadXClassifier(**FAST).fit(df, np.full(df.height, 2))
    with pytest.raises(ValueError, match="profile"):
        LeadXClassifier(profile="nope").fit(df, y)


def test_two_class_task():
    df, _, _, y_true = _pu_data(n=1500)
    m = LeadXClassifier(task="two_class", **FAST).fit(df, y_true)
    assert roc_auc_score(y_true, m.predict_proba(df)[:, 1]) > 0.85


# ----------------------------------------------------------- frame types ---
def test_numpy_and_polars_give_same_scores():
    df, X, y, _ = _pu_data(n=1200)
    a = LeadXClassifier(**FAST).fit(df, y).decision_function(df)
    b = LeadXClassifier(**FAST).fit(X, y).decision_function(X)
    np.testing.assert_allclose(a, b, rtol=1e-6)


# --------------------------------------------------------- sklearn compat ---
def test_sklearn_estimator_protocol():
    df, _, y, _ = _pu_data(n=800)
    m = LeadXClassifier(**FAST)
    assert set(FAST) <= set(m.get_params())
    clone(m)                                          # ต้อง clone ได้ (get/set_params)
    m.fit(df, y)
    assert list(m.classes_) == [0, 1]
    assert m.n_features_in_ == 5 and m.feature_names_in_[0] == "f0"
    assert m.predict(df).shape == (df.height,)
    assert set(np.unique(m.predict(df))) <= {0, 1}


def test_review_queue_and_importances():
    df, _, y, y_true = _pu_data(n=2000)
    m = LeadXClassifier(**FAST).fit(df, y)
    q = m.review_queue(df, frac=0.02)
    assert len(q) == 40 and y_true[q].mean() > y_true.mean()     # คิวเข้มข้นกว่า base
    assert len(m.review_queue(df, k=17)) == 17
    with pytest.raises(ValueError, match="k หรือ frac"):
        m.review_queue(df)
    imp = m.feature_importances_
    assert imp.shape == (5,) and np.nanargmax(imp) in (0, 1)     # signal อยู่ f0/f1


def test_prior_gives_probability_scale():
    df, _, y, _ = _pu_data(n=1000)
    m = LeadXClassifier(prior=0.01, **FAST).fit(df, y)
    p = m.predict_proba(df)[:, 1]
    assert 0.0 <= p.min() and p.max() <= 1.0


# ------------------------------------------------------- eval_set / grid ---
def test_eval_set_selects_config_and_records_results():
    df, _, y, y_true = _pu_data(n=3000)
    tr = np.arange(df.height) < 2000
    Xtr, Xva = df.filter(pl.Series(tr)), df.filter(pl.Series(~tr))
    m = LeadXClassifier(**FAST).fit(Xtr, y[tr], eval_set=(Xva, y_true[~tr]),
                                    param_grid={"n_bits": [6, 8]})
    assert len(m.evals_result_) == 2
    assert m.best_params_["n_bits"] in (6, 8) and m.best_score_ is not None


def test_param_grid_requires_eval_set():
    df, _, y, _ = _pu_data(n=1200)
    with pytest.raises(ValueError, match="eval_set"):
        LeadXClassifier(**FAST).fit(df, y, param_grid={"n_bits": [6]})


# ---------------------------------------------------- background policy ---
def test_background_refresh_scores_and_differs_from_frozen():
    df, _, y, y_true = _pu_data(n=2500)
    tr = np.arange(df.height) < 1500
    Xtr, Xva = df.filter(pl.Series(tr)), df.filter(pl.Series(~tr))
    frozen = LeadXClassifier(**FAST).fit(Xtr, y[tr]).decision_function(Xva)
    refresh = LeadXClassifier(background_policy="refresh", **FAST) \
        .fit(Xtr, y[tr]).decision_function(Xva)
    assert refresh.shape == frozen.shape
    assert roc_auc_score(y_true[~tr], refresh) > 0.8
    assert not np.allclose(frozen, refresh)          # คนละตัวหารจริง
    with pytest.raises(ValueError, match="background_policy"):
        LeadXClassifier(background_policy="nope", **FAST).fit(df, y)


# ---------------------------------------------------------- persistence ---
def test_save_load_roundtrip(tmp_path):
    df, _, y, _ = _pu_data(n=1500)
    m = LeadXClassifier(**FAST).fit(df, y)
    s1 = m.predict_proba(df)[:, 1]
    m.save_model(tmp_path / "m")
    m2 = LeadXClassifier.load_model(tmp_path / "m")
    np.testing.assert_allclose(s1, m2.predict_proba(df)[:, 1], rtol=1e-6)
    np.testing.assert_array_equal(m.predict(df), m2.predict(df))
    assert m2.feature_names_in_ == m.feature_names_in_


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        LeadXClassifier().decision_function(pl.DataFrame({"a": [1.0]}))
