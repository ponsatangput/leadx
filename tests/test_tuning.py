import numpy as np
import polars as pl
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
    assert {"strategy", "n_bits", "n_tables", "m", "weights", "score",
            "score_type", "alpha", "agg"} <= set(best)
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


def test_tuner_raises_on_tiny_minority_class():
    X, _ = _blobs(n_per=50)
    y = np.zeros(100)
    y[0] = 1.0  # minority มีตัวเดียว — เดิม n_folds=1 → train fold ว่าง → NaN crash ลึกๆ
    with pytest.raises(ValueError, match="2"):
        Tuner(grid=SMALL_GRID, mode="two_class").tune(X, y)


def test_tuner_evaluates_lift_and_prop():
    X, y = _blobs()
    grid = {**SMALL_GRID, "score_type": ["prop", "lift"], "alpha": [1.0, 5.0],
            "agg": ["mean", "log"]}
    tuner = Tuner(grid=grid, n_folds=3, mode="two_class", random_state=0)
    best = tuner.tune(X, y)
    assert {"score_type", "alpha", "agg"} <= set(best)
    assert set(tuner.results_["score_type"].unique()) == {"prop", "lift"}
    assert best["score"] > 0.9
    # dedupe: prop ไม่ควรถูกประเมินซ้ำต่อ alpha/agg (alpha/agg ไม่มีผลกับ prop)
    n_prop = tuner.results_.filter(pl.col("score_type") == "prop").height
    n_lift = tuner.results_.filter(pl.col("score_type") == "lift").height
    assert n_lift == n_prop * 2 * 2 / 2  # lift: ×alpha×agg แต่ไม่คูณ m (2 ค่าใน grid)


def test_tuner_respects_hash_mode(monkeypatch):
    # #2: LeadX(hash_mode="origin", tune=True) ต้องจูนด้วย origin banks ไม่ใช่ median
    from leadx.projections import ProjectionBank
    seen = []
    orig_init = ProjectionBank.__init__

    def spy(self, *a, **kw):
        seen.append(kw.get("hash_mode", "median"))
        orig_init(self, *a, **kw)

    monkeypatch.setattr(ProjectionBank, "__init__", spy)
    X, y = _blobs(n_per=150)
    grid = {**SMALL_GRID, "strategy": ["unsupervised"], "score_type": ["prop"],
            "alpha": [2.0], "agg": ["mean"], "cat_bit_frac": [0.0]}
    Tuner(grid=grid, n_folds=3, mode="two_class", hash_mode="origin",
          random_state=0).tune(X, y)
    assert seen and set(seen) == {"origin"}


def test_tuner_exposes_winning_bank_prefix():
    # #1: bank ที่ชนะถูกเก็บไว้ให้ facade ใช้ (ไม่ refit ใหม่)
    X, y = _blobs(n_per=200)
    grid = {**SMALL_GRID, "strategy": ["unsupervised"], "score_type": ["lift"],
            "alpha": [2.0], "agg": ["log"], "cat_bit_frac": [0.0]}
    tuner = Tuner(grid=grid, n_folds=3, mode="two_class", random_state=0)
    best = tuner.tune(X, y)
    bank = tuner.best_bank_
    assert bank.P_.shape[0] == max(grid["n_tables"])
    assert bank.P_.shape[2] == max(grid["n_bits"])
    assert bank.strategy_ == best["strategy"]


def test_default_grid_includes_deep_tables():
    # Phase-1 locality headroom: K=40 ให้ +15-24% rel PR-AUC บน PU จริง (home-credit)
    from leadx.tuning import DEFAULT_GRID
    assert 40 in DEFAULT_GRID["n_tables"]


def test_tuner_covers_all_strategy_whiten_combos():
    # regression (external audit P0): dedupe เดิมตัด supervised ทิ้งบน numeric-only
    # และ supervised+whiten หายทุกกรณี — ทุก (strategy, whiten) ที่ขอต้องถูกประเมิน
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 10)).astype(np.float32)
    y = (X[:, 0] > 0).astype(float)  # seeds เพียงพอ ไม่ trigger supervised fallback
    grid = {"n_bits": [6], "n_tables": [4], "m": [10.0], "weights": ["mean"],
            "strategy": ["unsupervised", "supervised"], "score_type": ["prop"],
            "alpha": [2.0], "agg": ["mean"], "cat_bit_frac": [0.0, "auto"],
            "whiten": [False, True]}
    t = Tuner(grid=grid, n_folds=3, mode="two_class", random_state=0)
    t.tune(X, y)
    combos = set(zip(t.results_["strategy"].to_list(), t.results_["whiten"].to_list()))
    assert combos == {("unsupervised", False), ("unsupervised", True),
                      ("supervised", False), ("supervised", True)}


def test_oof_no_leak_on_pure_noise_with_weighting():
    # audit P0: เดิม weights คำนวณจาก y ทั้งชุดก่อน tune → OOF optimistic บน noise ล้วน
    # หลัง cross-fit ต่อ fold: best score บน noise ต้อง ~0.5 (ไม่มีอะไรให้เรียนจริง)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((800, 60)).astype(np.float32)   # noise ล้วน มิติเยอะ
    y = (rng.random(800) < 0.5).astype(float)               # y สุ่ม ไม่เกี่ยวกับ X
    grid = {"n_bits": [8], "n_tables": [8], "m": [10.0], "weights": ["mean"],
            "strategy": ["unsupervised"], "score_type": ["lift"], "alpha": [2.0],
            "agg": ["log"], "cat_bit_frac": [0.0], "whiten": [False]}
    t = Tuner(grid=grid, n_folds=4, mode="two_class", random_state=0)
    best = t.tune(X, y, feature_weighting="corr")
    assert best["score"] < 0.57  # ไม่ inflate (leak เดิมดันเกินนี้ได้)


def test_pu_background_split_fit_vs_eval(monkeypatch):
    # audit P0: background ห้ามถูกใช้ทั้งสร้างตารางและเป็น negative proxy พร้อมกัน
    from leadx.scoring import SeedScorer
    seen_bg_sizes = []
    orig_fit = SeedScorer.fit

    def spy(self, codes, y, background_codes=None):
        if background_codes is not None:
            seen_bg_sizes.append(len(background_codes))
        return orig_fit(self, codes, y, background_codes=background_codes)

    monkeypatch.setattr(SeedScorer, "fit", spy)
    rng = np.random.default_rng(1)
    pos = (rng.standard_normal((150, 6)) + 2).astype(np.float32)
    bg = rng.standard_normal((1000, 6)).astype(np.float32)
    grid = {"n_bits": [6], "n_tables": [4], "m": [10.0], "weights": ["mean"],
            "strategy": ["unsupervised"], "score_type": ["lift"], "alpha": [2.0],
            "agg": ["log"], "cat_bit_frac": [0.0], "whiten": [False]}
    Tuner(grid=grid, n_folds=3, mode="pu", random_state=0).tune(
        pos, np.ones(150), Z_background=bg)
    # ทุก scorer ตอน tune ต้องเห็น background แค่ครึ่ง fit (500) ไม่ใช่เต็มพัน
    assert seen_bg_sizes and all(s == 500 for s in seen_bg_sizes)


def test_supervised_geometry_cross_fitted_per_fold(monkeypatch):
    # supervised bank ต้องถูก fit ต่อ fold (เห็นเฉพาะ y ของ train fold)
    from leadx.projections import ProjectionBank
    fit_sizes = []
    orig_fit = ProjectionBank.fit

    def spy(self, Z, y=None, cat_mask=None):
        if self.strategy == "supervised":
            fit_sizes.append(len(Z))
        return orig_fit(self, Z, y=y, cat_mask=cat_mask)

    monkeypatch.setattr(ProjectionBank, "fit", spy)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((900, 8)).astype(np.float32)
    y = (X[:, 0] > 0).astype(float)
    grid = {"n_bits": [6], "n_tables": [4], "m": [10.0], "weights": ["mean"],
            "strategy": ["supervised"], "score_type": ["prop"], "alpha": [2.0],
            "agg": ["mean"], "cat_bit_frac": [0.0], "whiten": [False]}
    Tuner(grid=grid, n_folds=3, mode="two_class", random_state=0).tune(X, y)
    # 3 folds → 3 bank fits แต่ละครั้งบน train fold (~2/3 ของ 900) ไม่ใช่เต็ม 900
    assert len(fit_sizes) == 3
    assert all(590 <= s <= 610 for s in fit_sizes), fit_sizes
