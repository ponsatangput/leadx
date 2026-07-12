import numpy as np
import pytest

from leadx.hashing import HashEnsemble, mask_codes


def _naive_codes(Z, P):
    """reference implementation แบบ loop ตรงไปตรงมา — bit j อยู่ตำแหน่ง MSB"""
    K, D, B = P.shape
    n = Z.shape[0]
    out = np.zeros((n, K), dtype=np.uint64)
    for i in range(n):
        for k in range(K):
            code = 0
            for j in range(B):
                bit = 1 if Z[i] @ P[k, :, j] > 0 else 0
                code |= bit << (B - 1 - j)
            out[i, k] = code
    return out


def test_packing_matches_naive():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((50, 10)).astype(np.float32)
    P = rng.standard_normal((3, 10, 8)).astype(np.float32)
    np.testing.assert_array_equal(HashEnsemble(P).transform(Z), _naive_codes(Z, P))


def test_chunked_equals_unchunked():
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((1000, 10)).astype(np.float32)
    P = rng.standard_normal((2, 10, 12)).astype(np.float32)
    h = HashEnsemble(P)
    np.testing.assert_array_equal(h.transform(Z, chunk_size=100), h.transform(Z))


def test_prefix_masking_is_nested():
    """mask เหลือ b bits ต้องเท่ากับ hash ด้วย projection b คอลัมน์แรกตรงๆ"""
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((200, 10)).astype(np.float32)
    P = rng.standard_normal((2, 10, 16)).astype(np.float32)
    full = HashEnsemble(P).transform(Z)
    for b in (4, 8, 12):
        direct = HashEnsemble(P[:, :, :b]).transform(Z)
        np.testing.assert_array_equal(mask_codes(full, b, 16), direct)


def test_rejects_over_64_bits():
    with pytest.raises(ValueError):
        HashEnsemble(np.zeros((1, 10, 65), dtype=np.float32))


def test_thresholds_offset_hashing():
    """HashEnsemble รองรับ per-bit thresholds: bit = (z@p > t) — LSH ไม่ต้องผ่าน origin"""
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((300, 8)).astype(np.float32)
    P = rng.standard_normal((2, 8, 6)).astype(np.float32)
    t = np.stack([np.median(Z @ P[k], axis=0) for k in range(2)]).astype(np.float32)
    h = HashEnsemble(P, thresholds=t)
    codes = h.transform(Z)
    # เทียบ naive
    powers = np.uint64(1) << np.arange(5, -1, -1, dtype=np.uint64)
    for k in range(2):
        naive = ((Z @ P[k]) > t[k]).astype(np.uint64) @ powers
        np.testing.assert_array_equal(codes[:, k], naive)
    # ทุก bit ควร balanced ~50/50 (threshold = median)
    for k in range(2):
        bits = (Z @ P[k]) > t[k]
        assert np.abs(bits.mean(axis=0) - 0.5).max() < 0.05


def test_thresholds_default_zero_backward_compat():
    rng = np.random.default_rng(4)
    Z = rng.standard_normal((100, 8)).astype(np.float32)
    P = rng.standard_normal((2, 8, 6)).astype(np.float32)
    np.testing.assert_array_equal(
        HashEnsemble(P).transform(Z),
        HashEnsemble(P, thresholds=np.zeros((2, 6), np.float32)).transform(Z))
