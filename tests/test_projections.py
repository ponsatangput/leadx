import numpy as np
import pytest

from leadx.projections import ProjectionBank


def _anisotropic_data(n=5000, d=20, seed=0):
    """ข้อมูลเบ้: มี direction เด่นหนึ่งแกน + offset — vector สุ่มดิบๆ มักได้ bit ไม่ balance"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)) * 0.3
    X[:, 0] = X[:, 0] * 5.0 + 2.0
    return X.astype(np.float32)


def test_shape_and_selected_bits_pass_balance():
    Z = _anisotropic_data()
    bank = ProjectionBank(n_tables=3, n_bits=8, random_state=0).fit(Z)
    assert bank.P_.shape == (3, 20, 8)
    diag = bank.diagnostics()
    for t in diag["tables"]:
        assert all(b >= 0.40 for b in t["balance"]), t["balance"]


def test_redundancy_bounded():
    Z = _anisotropic_data()
    bank = ProjectionBank(n_tables=2, n_bits=8, redundancy_max=0.30, random_state=1).fit(Z)
    for t in bank.diagnostics()["tables"]:
        assert t["max_redundancy"] < 0.30 + 1e-9


def test_bits_ordered_by_quality_descending():
    """จำเป็นต่อ prefix masking — bit แรกต้องคุณภาพดีสุด"""
    Z = _anisotropic_data()
    bank = ProjectionBank(n_tables=1, n_bits=8, random_state=2).fit(Z)
    bal = bank.diagnostics()["tables"][0]["balance"]
    assert all(bal[i] >= bal[i + 1] - 1e-9 for i in range(len(bal) - 1))


def test_diagnostics_keys():
    Z = _anisotropic_data(n=2000)
    bank = ProjectionBank(n_tables=3, n_bits=6, random_state=3).fit(Z)
    diag = bank.diagnostics()
    assert diag["strategy"] == "unsupervised"
    assert 0.0 <= diag["diversity_nmi"] <= 1.0
    for t in diag["tables"]:
        assert set(t) == {"balance", "max_redundancy", "n_eff"}
        assert t["n_eff"] > 1.0


def _labeled_data(n=2000, d=10, seed=0):
    """signal อยู่ที่แกน 0 แกนเดียว — supervised ควรจับ direction นี้ได้"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float64)
    return X, y


def test_supervised_top_bit_aligns_with_signal():
    X, y = _labeled_data()
    bank = ProjectionBank(n_tables=1, n_bits=4, strategy="supervised", random_state=0).fit(X, y)
    assert bank.strategy_ == "supervised"
    # bit แรก (quality ดีสุด) ต้องชี้ใกล้แกน 0 มากกว่า bits อื่น
    alignments = np.abs(bank.P_[0, 0, :])  # |component แกน 0| ของแต่ละ bit vector
    assert alignments[0] == alignments.max()
    assert alignments[0] > 0.5


def test_supervised_guard_falls_back_when_seeds_too_few():
    X, y = _labeled_data(n=60)  # ~30/class < min 50
    with pytest.warns(UserWarning, match="fallback"):
        bank = ProjectionBank(n_tables=1, n_bits=4, strategy="supervised", random_state=0).fit(X, y)
    assert bank.strategy_ == "unsupervised"
