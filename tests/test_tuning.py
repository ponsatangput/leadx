import numpy as np
import pytest

from leadx.tuning import Tuner, stratified_folds


def _blobs(n_per=300, d=8, sep=4.0, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n_per, d))
    X1 = rng.standard_normal((n_per, d)) + sep / np.sqrt(d)
    X = np.vstack([X0, X1])
    # Tuner รับ Z ที่ standardize แล้ว (contract เดียวกับ output ของ FeatureEncoder)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = np.concatenate([np.zeros(n_per), np.ones(n_per)])
    return X.astype(np.float32), y


SMALL_GRID = {
    "n_bits": [4, 8],
    "n_tables": [2, 4],
    "m": [1.0, 10.0],
    "weights": ["mean"],
    "strategy": ["unsupervised", "supervised"],
}


def test_stratified_folds_balanced():
    y = np.array([0] * 10 + [1] * 50, dtype=np.float64)
    folds = stratified_folds(y, 5, np.random.default_rng(0))
    for f in range(5):
        assert (y[folds == f] == 0).sum() == 2
        assert (y[folds == f] == 1).sum() == 10


def test_tuner_finds_good_config_on_easy_data():
    X, y = _blobs()
    tuner = Tuner(grid=SMALL_GRID, n_folds=3, mode="two_class", random_state=0)
    best = tuner.tune(X, y)
    assert set(best) == {"strategy", "n_bits", "n_tables", "m", "weights", "score"}
    assert best["score"] > 0.9  # blobs แยกชัด — OOF AUC ต้องสูง
    # results_ ครอบคลุมทั้ง grid (2*2*2*1 = 8 configs ต่อ strategy ที่รอด)
    assert tuner.results_.height >= 8


def test_tuner_results_sorted_descending():
    X, y = _blobs(n_per=150)
    tuner = Tuner(grid=SMALL_GRID, n_folds=3, mode="two_class", random_state=1)
    tuner.tune(X, y)
    scores = tuner.results_["score"].to_list()
    assert scores == sorted(scores, reverse=True)


def test_pu_tuning_requires_background():
    X, y = _blobs(n_per=100)
    with pytest.raises(ValueError, match="background"):
        Tuner(grid=SMALL_GRID, mode="pu").tune(X, np.ones(len(X)))


def test_pu_tuning_with_background_runs():
    rng = np.random.default_rng(3)
    pos = rng.standard_normal((200, 8)) + 3.0 / np.sqrt(8)
    bg = rng.standard_normal((500, 8))
    # standardize ด้วยสถิติของ seeds — ตรงกับที่ facade ทำ (encoder fit บน seeds)
    mu, sd = pos.mean(axis=0), pos.std(axis=0)
    pos = ((pos - mu) / sd).astype(np.float32)
    bg = ((bg - mu) / sd).astype(np.float32)
    grid = {**SMALL_GRID, "strategy": ["unsupervised"]}
    best = Tuner(grid=grid, n_folds=3, mode="pu", random_state=0).tune(
        pos, np.ones(200), Z_background=bg
    )
    assert best["score"] > 0.4  # AP ต้องดีกว่า random baseline (200/700 ≈ 0.29) ชัดเจน
