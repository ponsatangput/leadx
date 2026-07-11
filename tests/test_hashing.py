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
