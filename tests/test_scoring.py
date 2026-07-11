import numpy as np
import pytest

from leadx.scoring import SeedScorer


def test_m_estimate_matches_hand_computation():
    # bucket 5: seeds 4 ตัว (3 pos) | bucket 9: seeds 2 ตัว (0 pos) | prior = 3/6 = 0.5, m = 2
    codes = np.array([[5], [5], [5], [5], [9], [9]], dtype=np.uint64)
    y = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)
    scorer = SeedScorer(mode="two_class", m=2.0, multi_probe=False, n_bits=4).fit(codes, y)
    s = scorer.score(np.array([[5], [9]], dtype=np.uint64))
    np.testing.assert_allclose(s[0], (3 + 2 * 0.5) / (4 + 2))  # = 4/6
    np.testing.assert_allclose(s[1], (0 + 2 * 0.5) / (2 + 2))  # = 1/4


def test_unseen_bucket_gets_prior():
    codes = np.array([[5], [9]], dtype=np.uint64)
    y = np.array([1.0, 0.0])
    scorer = SeedScorer(mode="two_class", m=1.0, multi_probe=False, n_bits=4).fit(codes, y)
    s = scorer.score(np.array([[7]], dtype=np.uint64))  # bucket ไม่เคยเห็น
    np.testing.assert_allclose(s[0], 0.5)  # prior
    assert scorer.diagnostics_["miss_rate_final"] == 1.0


def test_averages_across_tables():
    # 2 tables: แถว query ตก bucket ที่ pure-pos ใน table 0 และ pure-neg ใน table 1
    codes = np.array([[1, 1], [1, 1], [2, 2], [2, 2]], dtype=np.uint64)
    y = np.array([1, 1, 0, 0], dtype=np.float64)
    scorer = SeedScorer(mode="two_class", m=0.0, multi_probe=False, n_bits=4).fit(codes, y)
    q = np.array([[1, 2]], dtype=np.uint64)  # table0→bucket1 (score 1.0), table1→bucket2 (score 0.0)
    np.testing.assert_allclose(scorer.score(q)[0], 0.5)


def test_rejects_non_binary_y():
    with pytest.raises(ValueError, match="0/1"):
        SeedScorer(mode="two_class").fit(np.array([[1]], dtype=np.uint64), np.array([2.0]))


def test_multi_probe_uses_hamming1_neighbor():
    # seeds ทั้งหมดอยู่ bucket 0b0001; query ตก 0b0011 (ต่าง 1 bit) → ได้ score เพื่อนบ้าน ไม่ใช่ prior
    codes = np.array([[0b0001]] * 4, dtype=np.uint64)
    y = np.array([1, 1, 1, 1], dtype=np.float64)
    scorer = SeedScorer(mode="two_class", m=0.0, multi_probe=True, n_bits=4).fit(codes, y)
    s = scorer.score(np.array([[0b0011]], dtype=np.uint64))
    np.testing.assert_allclose(s[0], 1.0)
    assert scorer.diagnostics_["miss_rate_join"] == 1.0   # join ตรงไม่เจอ
    assert scorer.diagnostics_["miss_rate_final"] == 0.0  # แต่ probe เจอ


def test_multi_probe_weights_neighbors_by_count():
    # เพื่อนบ้าน 2 bucket: 0b0000 (3 seeds, pure pos) และ 0b0011 (1 seed, pure neg)
    # query 0b0001 ต่าง 1 bit จากทั้งคู่ → weighted = (1.0*3 + 0.0*1)/4
    codes = np.array([[0b0000]] * 3 + [[0b0011]], dtype=np.uint64)
    y = np.array([1, 1, 1, 0], dtype=np.float64)
    scorer = SeedScorer(mode="two_class", m=0.0, multi_probe=True, n_bits=4).fit(codes, y)
    s = scorer.score(np.array([[0b0001]], dtype=np.uint64))
    np.testing.assert_allclose(s[0], 3.0 / 4.0)


def test_hamming2_still_misses_then_prior():
    codes = np.array([[0b0000]] * 2, dtype=np.uint64)
    y = np.array([1, 0], dtype=np.float64)
    scorer = SeedScorer(mode="two_class", m=0.0, multi_probe=True, n_bits=4).fit(codes, y)
    s = scorer.score(np.array([[0b0011]], dtype=np.uint64))  # ต่าง 2 bits
    np.testing.assert_allclose(s[0], 0.5)  # prior
    assert scorer.diagnostics_["miss_rate_final"] == 1.0


def test_coverage_weights_downweight_missing_table():
    codes = np.array([[1, 1], [1, 1], [2, 2], [2, 2]], dtype=np.uint64)
    y = np.array([1, 1, 0, 0], dtype=np.float64)  # prior = 0.5
    q = np.array([[1, 99]], dtype=np.uint64)  # table0→score 1.0, table1→miss→prior 0.5
    s_mean = SeedScorer(mode="two_class", m=0.0, multi_probe=False, n_bits=8,
                        weights="mean").fit(codes, y).score(q)
    s_cov = SeedScorer(mode="two_class", m=0.0, multi_probe=False, n_bits=8,
                       weights="coverage").fit(codes, y).score(q)
    np.testing.assert_allclose(s_mean[0], 0.75)  # (1.0 + 0.5)/2
    np.testing.assert_allclose(s_cov[0], 1.0)    # hit = [1, 0] → weight ทั้งหมดไป table 0


def test_pu_log_density_without_background():
    # bucket 1: 3 seeds, bucket 2: 1 seed → score = log(1+n)
    codes = np.array([[1], [1], [1], [2]], dtype=np.uint64)
    y = np.ones(4)
    scorer = SeedScorer(mode="pu", multi_probe=False, n_bits=4).fit(codes, y)
    s = scorer.score(np.array([[1], [2], [7]], dtype=np.uint64))
    np.testing.assert_allclose(s[0], np.log(4.0))
    np.testing.assert_allclose(s[1], np.log(2.0))
    np.testing.assert_allclose(s[2], 0.0)  # miss → prior 0 (ranking ต่ำสุด)
    assert s[0] > s[1] > s[2]


def test_pu_with_background_uses_seed_rate():
    # bucket 1: 2 seeds / 4 background, bucket 2: 1 seed / 100 background
    # แม้ bucket 2 มี seed แต่ rate ต่ำกว่า bucket 1 มาก
    seed_codes = np.array([[1], [1], [2]], dtype=np.uint64)
    bg_codes = np.array([[1]] * 4 + [[2]] * 100, dtype=np.uint64)
    scorer = SeedScorer(mode="pu", m=1.0, multi_probe=False, n_bits=4).fit(
        np.asarray(seed_codes), np.ones(3), background_codes=bg_codes
    )
    s = scorer.score(np.array([[1], [2]], dtype=np.uint64))
    rho = 3 / 104
    np.testing.assert_allclose(s[0], (2 + 1 * rho) / (4 + 1))
    np.testing.assert_allclose(s[1], (1 + 1 * rho) / (100 + 1))
    assert s[0] > s[1]
