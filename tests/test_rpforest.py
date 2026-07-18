"""RPForest — balanced local projection forest (วัดก่อน ship:
outputs/rpforest_experiment.py — +39-42% ที่ 0.3%, +6% natural, gate ผ่าน)"""
import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from leadx import LeadX, LeadXBackbone
from leadx.rpforest import RPForest


def _data(n=4000, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + X[:, 1] > 1.5).astype(float)
    return pl.DataFrame({f"f{i}": X[:, i] for i in range(d)}), y, X


def test_equal_mass_buckets():
    """คุณสมบัติหลัก: ทุก leaf มีสมาชิก ~n/2^b — ไม่มี mega-bucket โดยโครงสร้าง"""
    _, _, X = _data(n=8192)
    codes = RPForest(n_tables=4, n_bits=6, random_state=0).fit(
        X.astype(np.float32)).transform(X.astype(np.float32))
    for k in range(4):
        _, cnt = np.unique(codes[:, k], return_counts=True)
        assert len(cnt) == 64
        assert cnt.max() <= 8192 // 64 + 2   # balanced ยกเว้น tie เล็กน้อย
        assert cnt.min() >= 8192 // 64 - 2


def test_transform_deterministic_and_chunked():
    _, _, X = _data()
    Z = X.astype(np.float32)
    f = RPForest(n_tables=3, n_bits=5, random_state=1).fit(Z)
    c1 = f.transform(Z)
    c2 = f.transform(Z, chunk_size=333)
    np.testing.assert_array_equal(c1, c2)


def test_prefix_bits_are_tree_path():
    """bit ที่ level l อยู่ที่ MSB — prefix ของ code = node บรรพบุรุษ (จำเป็นต่อ
    mask_codes/hier): แถวที่แชร์ prefix b บิตแรกต้องอยู่ node เดียวกันที่ level b"""
    _, _, X = _data(n=2000)
    Z = X.astype(np.float32)
    f = RPForest(n_tables=1, n_bits=6, random_state=2).fit(Z)
    codes = f.transform(Z)[:, 0]
    # เดิน traversal เอง 3 level แรกแล้วเทียบกับ prefix ของ code
    node = np.zeros(len(Z), dtype=np.int64)
    for lvl in range(3):
        base = 2 ** lvl - 1
        G = f.dirs_[0, base + node]
        bit = np.einsum("ij,ij->i", Z, G) > f.thr_[0, base + node]
        node = 2 * node + bit
    np.testing.assert_array_equal(node, (codes >> np.uint64(3)).astype(np.int64))


def test_facade_rpforest_end_to_end_and_roundtrip(tmp_path):
    df, y, _ = _data()
    seeds = df.filter(pl.Series(y == 1))
    clf = LeadX(mode="pu", tune=False, n_bits=8, n_tables=8, score_type="lift",
                agg="log", hier_depths="auto", engine="rpforest",
                random_state=0).fit(seeds, background=df)
    s1 = clf.score_samples(df)
    assert roc_auc_score(y, s1) > 0.75
    clf.save(tmp_path / "m")
    clf2 = LeadX.load(tmp_path / "m")
    assert clf2.engine == "rpforest"
    np.testing.assert_allclose(s1, clf2.score_samples(df), rtol=1e-6)


def test_backbone_rpforest_cached_paths_and_roundtrip(tmp_path):
    df, y, _ = _data(seed=3)
    rows = np.flatnonzero(y == 1)
    bb = LeadXBackbone(n_bits=8, n_tables=6, engine="rpforest",
                       random_state=0).fit(df)
    task = bb.fit_task(mode="pu", seed_rows=rows, background="cached",
                       score_type="lift", agg="log", hier_depths="auto")
    s1 = task.score_cached()
    assert roc_auc_score(y, s1) > 0.75
    bb.save(tmp_path / "bb")
    bb2 = LeadXBackbone.load(tmp_path / "bb")
    task2 = bb2.fit_task(mode="pu", seed_rows=rows, background="cached",
                         score_type="lift", agg="log", hier_depths="auto")
    np.testing.assert_allclose(s1, task2.score_cached(), rtol=1e-6)


def test_rpforest_tune_not_supported():
    with pytest.raises(ValueError, match="rpforest"):
        LeadX(mode="pu", tune=True, engine="rpforest")
    with pytest.raises(ValueError, match="engine"):
        LeadX(mode="pu", tune=False, engine="rptree")
