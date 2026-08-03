"""LeadXClassifier — XGBoost-shaped interface (fit(X,y), any frame type,
sklearn compat, PU semantics, background policy, persistence)"""
import warnings

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


# =========================== v0.4 additions ===============================
# ------------------------------------------- all-numeric categorical fallback ---
def test_rare_pu_profile_works_on_all_numeric_input():
    """profile='rare_pu' ตั้ง cat_engine='minhash' — ต้องไม่พังเมื่อ input เป็น
    numeric ล้วน (dataset จริงมัก encode categorical มาแล้ว)"""
    _, X, y, y_true = _pu_data(n=2000)
    with pytest.warns(UserWarning, match="no categorical columns found"):
        m = LeadXClassifier(profile="rare_pu", n_tables=10, n_bits=8).fit(X, y)
    assert m.best_params_["cat_engine"] == "onehot"          # fallback เกิดจริง
    assert roc_auc_score(y_true, m.predict_proba(X)[:, 1]) > 0.85


def test_explicit_missing_categorical_columns_still_raises():
    _, X, y, _ = _pu_data(n=1200)
    with pytest.raises(ValueError, match="minhash_columns|ไม่มีใน"):
        LeadXClassifier(profile="rare_pu", categorical_columns=["nope"],
                        n_tables=8, n_bits=6).fit(X, y)


def test_real_categorical_uses_minhash_without_warning():
    df, _, y, _ = _pu_data(n=1500)
    df = df.with_columns(pl.Series("grp", (np.arange(df.height) % 7).astype(str)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = LeadXClassifier(profile="rare_pu", n_tables=8, n_bits=6).fit(df, y)
    assert not [w for w in caught if "no categorical columns" in str(w.message)]
    assert m.best_params_["cat_engine"] == "minhash"
    assert m.best_params_["hier_depths"] is None      # profile ปรับให้เข้ากันเอง


# -------------------------------------------------------------- sample_weight ---
def test_sample_weight_none_matches_previous_behaviour():
    df, _, y, _ = _pu_data(n=1500)
    a = LeadXClassifier(**FAST).fit(df, y).decision_function(df)
    b = LeadXClassifier(**FAST).fit(df, y, sample_weight=None).decision_function(df)
    c = LeadXClassifier(**FAST).fit(
        df, y, sample_weight=np.ones(df.height)).decision_function(df)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, c, rtol=1e-9)     # uniform weights == unweighted


def test_high_weight_on_a_seed_group_shifts_ranking():
    """ยกน้ำหนักให้ seeds ในมุมหนึ่งของ space -> แถวที่คล้ายมุมนั้นต้องขยับขึ้น"""
    rng = np.random.default_rng(3)
    n = 3000
    X = rng.normal(size=(n, 4))
    y = np.zeros(n, dtype=int)
    left = np.flatnonzero(X[:, 0] < -1.5)[:60]      # seed cluster A
    right = np.flatnonzero(X[:, 0] > 1.5)[:60]      # seed cluster B
    y[left] = 1
    y[right] = 1
    df = pl.DataFrame({f"f{i}": X[:, i] for i in range(4)})
    probe = np.flatnonzero(X[:, 0] > 1.0)           # แถวที่อยู่ฝั่ง B
    base = LeadXClassifier(**FAST).fit(df, y).decision_function(df)
    w = np.ones(n)
    w[right] = 20.0                                  # ให้ cluster B หนักกว่ามาก
    heavy = LeadXClassifier(**FAST).fit(df, y, sample_weight=w).decision_function(df)
    assert heavy[probe].mean() / heavy.mean() > base[probe].mean() / base.mean()


def test_sample_weight_validation():
    df, _, y, _ = _pu_data(n=1200)
    m = LeadXClassifier(**FAST)
    with pytest.raises(ValueError, match="sample_weight"):
        m.fit(df, y, sample_weight=np.ones(5))
    with pytest.raises(ValueError, match="ไม่ติดลบ|non-negative"):
        w = np.ones(df.height); w[0] = -1.0
        m.fit(df, y, sample_weight=w)
    with pytest.raises(ValueError, match="positive seed|> 0"):
        m.fit(df, y, sample_weight=(y == 0).astype(float))   # seeds น้ำหนัก 0 หมด


# ------------------------------------------------------------- metric parser ---
@pytest.mark.parametrize("metric", ["pr_auc", "roc_auc", "f1", "recall@100",
                                    "precision@100", "recall@0.5%",
                                    "precision@0.5%"])
def test_metric_parser_supported_formats(metric):
    from leadx.sklearn_api import _score_metric
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(int)
    s = y + rng.normal(scale=0.5, size=500)
    v = _score_metric(y, s, metric)
    assert 0.0 <= v <= 1.0


def test_metric_parser_rejects_unknown_with_helpful_message():
    from leadx.sklearn_api import _score_metric
    y = np.array([0, 1, 0, 1]); s = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="Supported"):
        _score_metric(y, s, "nonsense")
    with pytest.raises(ValueError, match="Supported"):
        _score_metric(y, s, "auc@10")


def test_eval_metric_flows_through_fit():
    df, _, y, y_true = _pu_data(n=3000)
    tr = np.arange(df.height) < 2000
    m = LeadXClassifier(**FAST).fit(
        df.filter(pl.Series(tr)), y[tr],
        eval_set=(df.filter(pl.Series(~tr)), y_true[~tr]),
        param_grid={"n_bits": [6, 8]}, eval_metric="precision@100")
    assert m.eval_metric_ == "precision@100"
    assert all("precision@100" in row for row in m.evals_result_)


# --------------------------------------------------- persistence metadata ---
def test_save_load_preserves_metadata_and_behaviour(tmp_path):
    df, _, y, y_true = _pu_data(n=3000)
    tr = np.arange(df.height) < 2000
    m = LeadXClassifier(**FAST).fit(
        df.filter(pl.Series(tr)), y[tr],
        eval_set=(df.filter(pl.Series(~tr)), y_true[~tr]),
        param_grid={"n_bits": [6, 8]}, eval_metric="recall@0.5%")
    m.save_model(tmp_path / "m")
    m2 = LeadXClassifier.load_model(tmp_path / "m")
    for key in ("leadx_version", "numpy_version", "polars_version",
                "sklearn_version"):
        assert m2.saved_versions_[key]
    assert m2.best_score_ == m.best_score_
    assert m2.eval_metric_ == "recall@0.5%"
    assert len(m2.evals_result_) == len(m.evals_result_)
    assert m2.best_params_ == m.best_params_
    assert m2.n_features_in_ == m.n_features_in_ == 5
    np.testing.assert_allclose(m.predict_proba(df)[:, 1],
                               m2.predict_proba(df)[:, 1], rtol=1e-6)
    np.testing.assert_allclose(m.decision_function(df), m2.decision_function(df),
                               rtol=1e-6)
    np.testing.assert_array_equal(m.review_queue(df, k=20),
                                  m2.review_queue(df, k=20))
    np.testing.assert_allclose(m.feature_importances_, m2.feature_importances_,
                               rtol=1e-6, equal_nan=True)


def test_determinism_under_random_state():
    df, _, y, _ = _pu_data(n=1500)
    a = LeadXClassifier(random_state=7, **FAST).fit(df, y).decision_function(df)
    b = LeadXClassifier(random_state=7, **FAST).fit(df, y).decision_function(df)
    np.testing.assert_array_equal(a, b)
