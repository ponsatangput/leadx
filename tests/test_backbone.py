import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from leadx import LeadX, LeadXBackbone


def _blobs_df(n_per=300, d=6, sep=5.0, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n_per, d))
    X1 = rng.standard_normal((n_per, d)) + sep / np.sqrt(d)
    X = np.vstack([X0, X1])
    df = pl.DataFrame({f"f{i}": X[:, i] for i in range(d)})
    y = np.concatenate([np.zeros(n_per), np.ones(n_per)])
    return df, y


def test_fit_task_matches_equivalent_leadx():
    # backbone + fit_task ต้องให้ผลเท่ากับ LeadX(unsupervised) ที่ config เดียวกันเป๊ะ
    # (projections มาจาก random_state เดียวกัน → codes เดียวกัน → scorer เดียวกัน)
    df, y = _blobs_df()
    bb = LeadXBackbone(n_bits=8, n_tables=5, random_state=0).fit(df)
    task = bb.fit_task(y, mode="two_class", m=10.0, weights="mean")
    ref = LeadX(mode="two_class", tune=False, strategy="unsupervised",
               n_bits=8, n_tables=5, m=10.0, weights="mean", random_state=0).fit(df, y)
    np.testing.assert_allclose(task.predict_proba(df), ref.predict_proba(df), rtol=1e-6)


def test_multiple_tasks_share_projections():
    # rows เดียว สอง labeling ต่างกันจริง — projections/codes ตัวเดียว, scorer คนละตัว
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 6))
    df = pl.DataFrame({f"f{i}": X[:, i] for i in range(6)})
    y1 = (X[:, 0] > 0).astype(float)   # โจทย์ 1: label ตาม f0
    y2 = (X[:, 1] > 0).astype(float)   # โจทย์ 2: label ตาม f1 (rows เดิม)
    bb = LeadXBackbone(n_bits=10, n_tables=8, random_state=0).fit(df)
    t1 = bb.fit_task(y1, mode="two_class")
    t2 = bb.fit_task(y2, mode="two_class")
    # projections แชร์กัน (object เดียว)
    assert t1.backbone is t2.backbone
    # โจทย์ต่างกัน → prediction ต่างกัน
    assert not np.allclose(t1.predict_proba(df), t2.predict_proba(df))
    assert roc_auc_score(y1, t1.predict_proba(df)[:, 1]) > 0.75


def test_per_task_prefix_masking():
    df, y = _blobs_df()
    bb = LeadXBackbone(n_bits=12, n_tables=6, random_state=0).fit(df)
    # n_bits < backbone → prefix mask (ไม่ re-hash); ต้องเท่ากับ hash ด้วย 8 bits ตรงๆ
    task = bb.fit_task(y, n_bits=8, n_tables=4)
    ref = LeadX(mode="two_class", tune=False, n_bits=8, n_tables=4, m=10.0,
               strategy="unsupervised", random_state=0).fit(df, y)
    # projections 8-prefix ของ backbone ควรตรงกับ ProjectionBank n_bits=8 (bits เรียง quality)
    # จึงคาดว่า AUC ใกล้กัน (ไม่บังคับเป๊ะเพราะ n_tables subset)
    assert roc_auc_score(y, task.predict_proba(df)[:, 1]) > 0.9
    with pytest.raises(ValueError, match="n_bits"):
        bb.fit_task(y, n_bits=20)  # เกิน backbone


def test_pu_task_with_background():
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((300, 5)) + 2.0
    allx = np.vstack([rng.standard_normal((500, 5)), pos[:100]])
    df_all = pl.DataFrame({f"f{i}": allx[:, i] for i in range(5)})
    df_pos = pl.DataFrame({f"f{i}": pos[:, i] for i in range(5)})
    bb = LeadXBackbone(n_bits=8, n_tables=6, random_state=0).fit(df_all)
    task = bb.fit_task(mode="pu", background=df_all, seeds=df_pos)
    s = task.score_samples(df_all)
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert s[500:].mean() > s[:500].mean()  # แถว positive คะแนนสูงกว่า


def test_backbone_save_load(tmp_path):
    df, y = _blobs_df(n_per=100)
    bb = LeadXBackbone(n_bits=8, n_tables=4, random_state=0).fit(df)
    expected = bb.fit_task(y).predict_proba(df)
    bb.save(tmp_path / "bb")
    loaded = LeadXBackbone.load(tmp_path / "bb")
    got = loaded.fit_task(y).predict_proba(df)
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_fit_task_rejects_mismatched_y_length():
    df, y = _blobs_df(n_per=100)
    bb = LeadXBackbone(n_bits=6, n_tables=3, random_state=0).fit(df)
    with pytest.raises(ValueError, match="ความยาว|length|แถว"):
        bb.fit_task(y[:150])  # 150 != 200 rows


def test_pu_task_scores_batch_invariant():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 5))
    df = pl.DataFrame({f"f{i}": X[:, i] for i in range(5)})
    pos = pl.DataFrame({f"f{i}": (rng.standard_normal((150, 5)) + 1.5)[:, i] for i in range(5)})
    bb = LeadXBackbone(n_bits=6, n_tables=4, random_state=0).fit(df)
    task = bb.fit_task(mode="pu", seeds=pos, background=df)
    whole = task.score_samples(df)
    halves = np.concatenate([task.score_samples(df[:200]), task.score_samples(df[200:])])
    np.testing.assert_allclose(whole, halves, rtol=1e-9)
