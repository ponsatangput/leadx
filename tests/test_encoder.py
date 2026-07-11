import numpy as np
import polars as pl
import pytest

from leadx.encoder import FeatureEncoder


def test_numeric_standardized():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    Z = FeatureEncoder().fit_transform(df)
    assert Z.dtype == np.float32
    assert Z.shape == (5, 2)
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(Z.std(axis=0, ddof=1), 1.0, atol=1e-5)


def test_outlier_clipped():
    vals = [1.0] * 999 + [1_000_000.0]  # outlier ตัวเดียว
    df = pl.DataFrame({"a": vals})
    Z = FeatureEncoder(clip_percentiles=(0.5, 99.5)).fit_transform(df)
    # ถ้าไม่ clip, std ถูก outlier ลาก → ค่า z ของ outlier จะ ~31; clip แล้วต้องเล็กกว่ามาก
    assert abs(Z[-1, 0]) < 10.0


def test_missing_fills_median_and_adds_indicator():
    df = pl.DataFrame({"a": [1.0, 2.0, None, 4.0, 100.0]})
    enc = FeatureEncoder()
    Z = enc.fit_transform(df)
    assert Z.shape == (5, 2)  # ค่า + missing indicator
    assert Z[2, 1] == 1.0 and Z[0, 1] == 0.0
    # แถว null ถูกเติม median (3.0) ก่อน standardize → ค่าไม่ใช่ NaN
    assert not np.isnan(Z).any()


def test_transform_rejects_mismatched_columns():
    enc = FeatureEncoder().fit(pl.DataFrame({"a": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="columns mismatch"):
        enc.transform(pl.DataFrame({"z": [1.0]}))


def test_nan_treated_as_missing_like_null():
    # numpy NaN (จาก DataFrame ที่สร้างจาก numpy array) ต้องถูกจัดการเหมือน null
    # ไม่งั้น NaN จะทะลุ dot product แล้ว bits ทั้งหมด collapse (regression bug)
    df = pl.DataFrame({"a": [1.0, 2.0, float("nan"), 4.0, 100.0]})
    enc = FeatureEncoder()
    Z = enc.fit_transform(df)
    assert Z.shape == (5, 2)  # ค่า + missing indicator
    assert not np.isnan(Z).any()
    assert Z[2, 1] == 1.0 and Z[0, 1] == 0.0  # NaN row ถูกทำเครื่องหมาย missing


def test_one_hot_low_cardinality():
    df = pl.DataFrame({"cat": ["a", "b", "c", "a"]})
    enc = FeatureEncoder()
    Z = enc.fit_transform(df)
    assert Z.shape == (4, 3)  # 3 categories
    # ค่า nonzero = 1/sqrt(3) และแถวละหนึ่งตำแหน่ง
    np.testing.assert_allclose(Z[Z > 0], 1.0 / np.sqrt(3), atol=1e-6)
    assert (np.count_nonzero(Z, axis=1) == 1).all()
    # แถว 0 กับแถว 3 เป็น "a" เหมือนกัน → เวกเตอร์เท่ากัน
    np.testing.assert_array_equal(Z[0], Z[3])


def test_one_hot_unseen_category_is_all_zero():
    enc = FeatureEncoder().fit(pl.DataFrame({"cat": ["a", "b"]}))
    Z = enc.transform(pl.DataFrame({"cat": ["zzz"]}))
    np.testing.assert_array_equal(Z, np.zeros((1, 2), dtype=np.float32))


def test_high_cardinality_uses_hashing():
    vals = [f"user_{i}" for i in range(100)]
    df = pl.DataFrame({"cat": vals})
    enc = FeatureEncoder(one_hot_max=30, hash_dims=64)
    Z = enc.fit_transform(df)
    assert Z.shape == (100, 64)
    # deterministic: transform ซ้ำได้ค่าเดิม
    np.testing.assert_array_equal(Z, enc.transform(df))
    # ค่าเดียวกันได้เวกเตอร์เดียวกัน / ต่างกันส่วนใหญ่ต้องต่างกัน
    Z2 = enc.transform(pl.DataFrame({"cat": ["user_0", "user_0", "user_1"]}))
    np.testing.assert_array_equal(Z2[0], Z2[1])
    assert not np.array_equal(Z2[0], Z2[2])


def test_mixed_numeric_categorical():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    Z = FeatureEncoder().fit_transform(df)
    assert Z.shape == (3, 1 + 2)
    assert not np.isnan(Z).any()
