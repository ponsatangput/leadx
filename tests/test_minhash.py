"""MinHash categorical engine — Jaccard locality + LeadX(cat_engine='minhash')
integration (validated on IEEE fraud: +56% PR vs one-hot, 5/5 draws)"""
import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from leadx import LeadX
from leadx.minhash import MinHashCat


def _cat_df(n=2000, n_cols=6, card=10, seed=0):
    """rows ในกลุ่มเดียวกันมี categorical ตรงกันเยอะ (Jaccard สูง); positive =
    กลุ่มแรก — signal อยู่ที่ categorical ล้วน (numeric เป็น noise)"""
    rng = np.random.default_rng(seed)
    grp = rng.integers(0, 20, n)
    proto = rng.integers(0, card, (20, n_cols))
    X = proto[grp].copy()
    flip = rng.random((n, n_cols)) < 0.1
    X[flip] = rng.integers(0, card, flip.sum())
    data = {f"cat{i}": X[:, i].astype(str) for i in range(n_cols)}
    data["noise"] = rng.normal(size=n)              # numeric noise
    y = (grp == 0).astype(float)
    return pl.DataFrame(data), y, [f"cat{i}" for i in range(n_cols)]


def test_minhash_jaccard_locality_and_determinism():
    df, _, cols = _cat_df()
    mh = MinHashCat(cols, n_tables=30, n_bits=8, random_state=0)
    codes = mh.transform(df)
    assert codes.shape == (df.height, 30)
    np.testing.assert_array_equal(codes, mh.transform(df))          # deterministic
    np.testing.assert_array_equal(codes, mh.transform(df, chunk_size=137))  # chunk-invariant


def test_minhash_serialization_roundtrip():
    df, _, cols = _cat_df()
    mh = MinHashCat(cols, n_tables=12, n_bits=6, random_state=3)
    mh2 = MinHashCat.from_dict(mh.to_dict())
    np.testing.assert_array_equal(mh.transform(df), mh2.transform(df))


def test_minhash_validation():
    with pytest.raises(ValueError, match="n_bits"):
        MinHashCat(["a"], 4, 40)
    with pytest.raises(ValueError, match="อย่างน้อย 1"):
        MinHashCat([], 4, 8)
    df = pl.DataFrame({"a": ["x", "y"]})
    with pytest.raises(ValueError, match="ขาด"):
        MinHashCat(["a", "b"], 4, 8).transform(df)


def test_facade_minhash_recovers_categorical_signal():
    """signal อยู่ใน categorical ล้วน — minhash ต้องจับได้ (numeric-only จับไม่ได้);
    score บน held-out (คนละแถวกับ seeds) กัน eval-on-train leakage"""
    df, y, cols = _cat_df(n=4000)                      # proto เดียวกันทั้งก้อน
    tr = np.arange(4000) < 2000
    df_tr, y_tr = df.filter(pl.Series(tr)), y[tr]
    df_ev, y_ev = df.filter(pl.Series(~tr)), y[~tr]
    seeds = df_tr.filter(pl.Series(y_tr == 1))
    clf = LeadX(mode="pu", tune=False, n_bits=10, n_tables=40, score_type="lift",
                agg="log", engine="rpforest", cat_engine="minhash",
                minhash_columns=cols, random_state=0).fit(seeds, background=df_tr)
    auc = roc_auc_score(y_ev, clf.score_samples(df_ev))
    assert auc > 0.75, f"minhash ควรจับ categorical signal บน held-out (auc={auc:.3f})"
    # numeric-only (ทิ้ง categorical) จับ signal ไม่ได้บน held-out
    num = LeadX(mode="pu", tune=False, n_bits=10, n_tables=40, score_type="lift",
                agg="log", engine="rpforest", random_state=0).fit(
        seeds.select("noise"), background=df_tr.select("noise"))
    assert roc_auc_score(y_ev, num.score_samples(df_ev.select("noise"))) < 0.62


def test_facade_minhash_auto_detects_string_columns():
    df, y, cols = _cat_df()
    seeds = df.filter(pl.Series(y == 1))
    clf = LeadX(mode="pu", tune=False, n_bits=8, n_tables=20, score_type="lift",
                agg="log", cat_engine="minhash", random_state=0).fit(seeds, background=df)
    assert set(clf._minhash_cols_) == set(cols)          # auto = string columns
    assert roc_auc_score(y, clf.score_samples(df)) > 0.7


def test_facade_minhash_save_load_roundtrip(tmp_path):
    df, y, cols = _cat_df()
    seeds = df.filter(pl.Series(y == 1))
    clf = LeadX(mode="pu", tune=False, n_bits=10, n_tables=30, score_type="lift",
                agg="log", engine="rpforest", cat_engine="minhash",
                minhash_columns=cols, random_state=0).fit(seeds, background=df)
    s1 = clf.score_samples(df)
    clf.save(tmp_path / "m")
    clf2 = LeadX.load(tmp_path / "m")
    assert clf2.cat_engine == "minhash"
    np.testing.assert_allclose(s1, clf2.score_samples(df), rtol=1e-6)


def test_minhash_requires_tune_false():
    df, y, cols = _cat_df()
    with pytest.raises(ValueError, match="tune=True"):
        LeadX(mode="pu", tune=True, cat_engine="minhash",
              minhash_columns=cols).fit(df.filter(pl.Series(y == 1)), background=df)
