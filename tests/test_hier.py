"""hier_depths — hierarchical shrinkage ข้าม prefix depths (วัดจริงก่อน ship:
outputs/hier_shrinkage_experiment.py — +6-41% rel PR-AUC ที่ imbalance 1%-0.1%)"""
import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from leadx import LeadX
from leadx.scoring import SeedScorer, resolve_hier_depths


def _rand_codes(n, K=5, b=8, seed=0):
    return np.random.default_rng(seed).integers(0, 2 ** b, size=(n, K),
                                                dtype=np.uint64)


def _pu_data(n=400, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + X[:, 1] > 1.2).astype(float)
    return pl.DataFrame({f"f{i}": X[:, i] for i in range(d)}), y


def test_resolve_hier_depths():
    assert resolve_hier_depths(None, 10) is None
    assert resolve_hier_depths("auto", 10) == (4, 6, 8, 10)
    assert resolve_hier_depths("auto", 4) == (4,)
    assert resolve_hier_depths("auto", 3) == (3,)
    assert resolve_hier_depths([4, 8], 8) == (4, 8)


def test_hier_validation_errors():
    with pytest.raises(ValueError, match="lift"):
        SeedScorer(score_type="prop", n_bits=8, hier_depths=(4, 8))
    with pytest.raises(ValueError, match="n_bits"):
        SeedScorer(score_type="lift", hier_depths=(4, 8))
    with pytest.raises(ValueError, match="n_bits"):
        SeedScorer(score_type="lift", n_bits=8, hier_depths=(4, 6))
    with pytest.raises(ValueError, match="เรียง"):
        SeedScorer(score_type="lift", n_bits=8, hier_depths=(8, 4))


def test_hier_single_depth_equals_plain_lift():
    """chain ที่ depth เดียว (n_bits) ต้องเท่า flat lift (ไม่มี multi-probe) เป๊ะ
    — ยืนยันว่า hier เป็น generalization ไม่ใช่สูตรใหม่"""
    seeds = _rand_codes(50, seed=1)
    bg = _rand_codes(500, seed=2)
    q = _rand_codes(200, seed=3)
    kw = dict(mode="pu", score_type="lift", agg="log", n_bits=8)
    base = SeedScorer(multi_probe=False, **kw).fit(
        seeds, np.ones(50), background_codes=bg)
    hier = SeedScorer(hier_depths=(8,), **kw).fit(
        seeds, np.ones(50), background_codes=bg)
    np.testing.assert_allclose(base.score(q), hier.score(q), rtol=1e-10)


def test_hier_parent_inheritance():
    """bucket ลึกที่ว่างต้อง inherit หลักฐานจาก bucket แม่ — ไม่ตกลง fallback=1"""
    # seeds กระจุกใน prefix 0xA (bucket แม่เดียว) แต่เว้น 0xA1 ไว้ที่ depth 8
    seeds = np.array([[0xA0 + i] for i in range(0, 16, 2)], dtype=np.uint64)
    bg = np.arange(256, dtype=np.uint64)[:, None]
    sc = SeedScorer(mode="pu", score_type="lift", agg="log", n_bits=8,
                    hier_depths=(4, 8)).fit(seeds, np.ones(len(seeds)),
                                            background_codes=bg)
    s = sc.score(np.array([[0xA1], [0x37]], dtype=np.uint64))
    assert s[0] > 1.0  # แม่มี seeds หนาแน่น → ลูกว่างยังได้ lift > กลาง
    assert s[0] > s[1]  # แม่ไม่มี seeds เลย → ต่ำกว่าชัดเจน
    assert s[1] < 1.0


def test_hier_facade_end_to_end_and_save_roundtrip(tmp_path):
    df, y = _pu_data()
    seeds = df.filter(pl.Series(y == 1))
    clf = LeadX(mode="pu", tune=False, n_bits=8, n_tables=6, score_type="lift",
                agg="log", hier_depths="auto", random_state=0).fit(
        seeds, background=df)
    assert clf.scorer_.hier_depths == (4, 6, 8)
    s1 = clf.score_samples(df)
    assert roc_auc_score(y, s1) > 0.7
    clf.save(tmp_path / "m")
    s2 = LeadX.load(tmp_path / "m").score_samples(df)
    np.testing.assert_allclose(s1, s2, rtol=1e-6)


def test_hier_disabled_for_prop_with_warning():
    df, y = _pu_data()
    seeds = df.filter(pl.Series(y == 1))
    with pytest.warns(UserWarning, match="hier_depths"):
        clf = LeadX(mode="pu", tune=False, n_bits=8, n_tables=4,
                    score_type="prop", hier_depths="auto", random_state=0).fit(
            seeds, background=df)
    assert clf.scorer_.hier_depths is None


def test_hier_backbone_fit_task():
    df, y = _pu_data(seed=5)
    from leadx import LeadXBackbone
    bb = LeadXBackbone(n_bits=8, n_tables=6, random_state=0).fit(df)
    task = bb.fit_task(mode="pu", seeds=df.filter(pl.Series(y == 1)),
                       background=df, score_type="lift", agg="log",
                       hier_depths="auto")
    assert roc_auc_score(y, task.score_samples(df)) > 0.7
