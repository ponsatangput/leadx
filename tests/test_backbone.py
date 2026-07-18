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


def test_fit_task_lift():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 6))
    df = pl.DataFrame({f"f{i}": X[:, i] for i in range(6)})
    y = (X[:, 0] > 0.8).astype(float)  # imbalanced ~20%
    bb = LeadXBackbone(n_bits=10, n_tables=8, random_state=0).fit(df)
    task = bb.fit_task(y, score_type="lift", alpha=2.0, agg="log")
    s = task.score_samples(df)
    assert roc_auc_score(y, s) > 0.75
    assert s.max() > 1.0  # lift scale


def _multitask_noisy(n=12000, d=100, n_sig=8, seed=0):
    """หลายโจทย์แชร์ signal features ชุดเดียว (คนละ combination) + noise หนัก"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    df = pl.DataFrame({f"f{i}": X[:, i] for i in range(d)})
    ys = []
    for _ in range(3):
        w = np.zeros(d)
        w[:n_sig] = rng.standard_normal(n_sig)
        ys.append((X @ w + rng.standard_normal(n) * 0.5 > 0).astype(float))
    # โจทย์ disjoint: signal อยู่ features คนละชุด
    w2 = np.zeros(d)
    w2[n_sig:2 * n_sig] = rng.standard_normal(n_sig)
    y_disjoint = (X @ w2 + rng.standard_normal(n) * 0.5 > 0).astype(float)
    return df, ys, y_disjoint


def test_shared_weights_lift_related_tasks_and_hurt_disjoint():
    # เรียน weights จากโจทย์อ้างอิง → โจทย์ที่แชร์ signal features ได้ประโยชน์มาก (OOS)
    # แต่โจทย์ที่ signal คนละชุดเสียหาย — trade-off ที่ต้องเข้าใจก่อนใช้
    df, ys, y_dis = _multitask_noisy()
    tr, te = slice(0, 8000), slice(8000, 12000)
    plain = LeadXBackbone(n_bits=10, n_tables=10, random_state=0).fit(df[tr])
    weighted = LeadXBackbone(n_bits=10, n_tables=10, random_state=0).fit(
        df[tr], y_ref=ys[0][tr])
    w = weighted.feature_weights_
    assert w[:8].mean() > 3 * w[8:].mean()  # จับ signal features ถูกชุด
    for y in ys[1:]:  # โจทย์ related (ไม่ใช่ ref เอง)
        auc_p = roc_auc_score(y[te], plain.fit_task(y[tr]).predict_proba(df[te])[:, 1])
        auc_w = roc_auc_score(y[te], weighted.fit_task(y[tr]).predict_proba(df[te])[:, 1])
        assert auc_w > auc_p + 0.10  # กำไรชัดเจน (วัดจริง ~+0.25)
    # downside: โจทย์ disjoint แย่ลง — ต้องเลือก y_ref ที่ represent กลุ่มโจทย์จริง
    auc_p = roc_auc_score(y_dis[te], plain.fit_task(y_dis[tr]).predict_proba(df[te])[:, 1])
    auc_w = roc_auc_score(y_dis[te], weighted.fit_task(y_dis[tr]).predict_proba(df[te])[:, 1])
    assert auc_w < auc_p


def test_shared_weights_persist_save_load(tmp_path):
    df, ys, _ = _multitask_noisy(n=1000)
    bb = LeadXBackbone(n_bits=8, n_tables=4, random_state=0).fit(df, y_ref=ys[0])
    expected = bb.fit_task(ys[1]).predict_proba(df)
    bb.save(tmp_path / "bb")
    loaded = LeadXBackbone.load(tmp_path / "bb")
    np.testing.assert_allclose(loaded.fit_task(ys[1]).predict_proba(df), expected, rtol=1e-6)


def test_backbone_save_load_preserves_config(tmp_path):
    # #4: backbone ต้องรักษา hash_mode + cat_bit_frac ข้าม save/load
    df, y = _blobs_df(n_per=80)
    bb = LeadXBackbone(n_bits=6, n_tables=3, hash_mode="origin",
                       cat_bit_frac=0.0, random_state=0).fit(df)
    expected = bb.fit_task(y).predict_proba(df)
    bb.save(tmp_path / "bb")
    loaded = LeadXBackbone.load(tmp_path / "bb")
    assert loaded.hash_mode == "origin"
    assert loaded.cat_bit_frac == 0.0
    np.testing.assert_allclose(loaded.fit_task(y).predict_proba(df), expected, rtol=1e-6)


def test_backbone_legacy_inference_era1_origin(tmp_path):
    # era-1 backbone: ไม่มีทั้ง hash_mode และ thresholds.npy → infer origin
    import json, os
    df, y = _blobs_df(n_per=80)
    bb = LeadXBackbone(n_bits=6, n_tables=3, hash_mode="origin", random_state=0).fit(df)
    expected = bb.fit_task(y).predict_proba(df)
    bb.save(tmp_path / "bb")
    meta = json.loads((tmp_path / "bb" / "meta.json").read_text())
    meta.pop("hash_mode", None)
    (tmp_path / "bb" / "meta.json").write_text(json.dumps(meta))
    os.remove(tmp_path / "bb" / "thresholds.npy")
    loaded = LeadXBackbone.load(tmp_path / "bb")
    assert loaded.hash_mode == "origin"
    np.testing.assert_allclose(loaded.fit_task(y).predict_proba(df), expected, rtol=1e-6)


def test_fit_task_seed_rows_and_cached_background_equivalence():
    """seed_rows + background='cached' ต้องให้ผลเท่า path ที่ re-hash df เดิมเป๊ะ"""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 8))
    df = pl.DataFrame({f"f{i}": X[:, i] for i in range(8)})
    y = (X[:, 0] > 1.0).astype(float)
    rows = np.flatnonzero(y == 1)
    bb = LeadXBackbone(n_bits=10, n_tables=6, random_state=0).fit(df)
    cheap = bb.fit_task(mode="pu", seed_rows=rows, background="cached",
                        score_type="lift", agg="log")
    rehash = bb.fit_task(mode="pu", seeds=df[rows.tolist()], background=df,
                         score_type="lift", agg="log")
    s_cheap = cheap.score_cached()
    s_rehash = rehash.score_samples(df)
    np.testing.assert_allclose(s_cheap, s_rehash, rtol=1e-6)
    # score_cached ต้องเท่ากับ score_samples(df เดิม) ของ task เดียวกันด้วย
    np.testing.assert_allclose(s_cheap, cheap.score_samples(df), rtol=1e-6)


def test_fit_task_seed_rows_validation():
    rng = np.random.default_rng(4)
    df = pl.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    bb = LeadXBackbone(n_bits=8, n_tables=4, random_state=0).fit(df)
    with pytest.raises(ValueError, match="อย่างเดียว"):
        bb.fit_task(mode="pu", seeds=df[:5], seed_rows=[0, 1, 2])
    with pytest.raises(ValueError, match="cached"):
        bb.fit_task(mode="pu", seed_rows=[0, 1, 2], background="cachd")


def test_background_stats_cache_equivalence_and_reuse():
    """path stats cache (lift) ต้องให้ผลเท่า path bg codes เป๊ะ และ cache ถูก reuse"""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(600, 6))
    df = pl.DataFrame({f"f{i}": X[:, i] for i in range(6)})
    y = (X[:, 0] + X[:, 1] > 1.5).astype(float)
    rows = np.flatnonzero(y == 1)
    bb = LeadXBackbone(n_bits=10, n_tables=5, random_state=0).fit(df)
    kw = dict(mode="pu", score_type="lift", agg="log", hier_depths="auto")
    t1 = bb.fit_task(seed_rows=rows, background="cached", **kw)
    t2 = bb.fit_task(seeds=df[rows.tolist()], background=df, **kw)
    np.testing.assert_allclose(t1.score_cached(), t2.score_samples(df), rtol=1e-6)
    assert len(bb._bg_stats_cache) == 1
    bb.fit_task(seed_rows=rows[:10], background="cached", **kw)  # โจทย์ที่สอง
    assert len(bb._bg_stats_cache) == 1  # reuse ไม่สร้างใหม่
    bb.fit_task(seed_rows=rows, background="cached", mode="pu",
                score_type="lift", agg="log")  # ไม่มี hier → คนละ key
    assert len(bb._bg_stats_cache) == 2
