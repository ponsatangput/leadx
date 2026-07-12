import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from leadx import LeadX


def _blobs_df(n_per=400, d=6, sep=5.0, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n_per, d))
    X1 = rng.standard_normal((n_per, d)) + sep / np.sqrt(d)
    X = np.vstack([X0, X1])
    df = pl.DataFrame({f"f{i}": X[:, i] for i in range(d)})
    y = np.concatenate([np.zeros(n_per), np.ones(n_per)])
    return df, y


def test_fit_predict_separable_blobs():
    df, y = _blobs_df()
    clf = LeadX(mode="two_class", tune=False, n_bits=8, n_tables=5, random_state=0)
    clf.fit(df, y)
    proba = clf.predict_proba(df)
    assert proba.shape == (len(y), 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert roc_auc_score(y, proba[:, 1]) > 0.95


def test_fit_with_tune_records_results():
    df, y = _blobs_df(n_per=200)
    grid = {"n_bits": [4, 8], "n_tables": [2, 4], "m": [10.0],
            "weights": ["mean"], "strategy": ["unsupervised"]}
    clf = LeadX(mode="two_class", tune=True, grid=grid, random_state=0)
    clf.fit(df, y)
    assert clf.tuning_results_ is not None
    assert {"strategy", "n_bits", "n_tables", "m", "weights",
            "score_type", "alpha", "agg"} <= set(clf.config_)
    assert "projections" in clf.diagnostics_


def test_pu_score_samples_normalized():
    rng = np.random.default_rng(1)
    pos = rng.standard_normal((300, 5)) + 2.0
    df_pos = pl.DataFrame({f"f{i}": pos[:, i] for i in range(5)})
    all_x = np.vstack([rng.standard_normal((500, 5)), pos[:100]])
    df_all = pl.DataFrame({f"f{i}": all_x[:, i] for i in range(5)})
    clf = LeadX(mode="pu", tune=False, n_bits=6, n_tables=4, random_state=0)
    clf.fit(df_pos)  # PU: y ไม่ต้องส่ง (all-positive)
    s = clf.score_samples(df_all)
    assert s.min() >= 0.0 and s.max() <= 1.0
    # แถวท้าย (มาจากบริเวณ positive) ควรได้ score สูงกว่าแถวแรกโดยเฉลี่ย
    assert s[500:].mean() > s[:500].mean()


def test_errors():
    df, y = _blobs_df(n_per=5)  # 10 แถว < 20
    with pytest.raises(ValueError, match="20"):
        LeadX(tune=False).fit(df, y)
    df2, y2 = _blobs_df(n_per=50)
    clf = LeadX(mode="pu", tune=False, n_bits=4, n_tables=2).fit(df2)
    with pytest.raises(ValueError, match="two_class"):
        clf.predict_proba(df2)
    clf2 = LeadX(mode="two_class", tune=False, n_bits=4, n_tables=2).fit(df2, y2)
    with pytest.raises(ValueError, match="columns mismatch"):
        clf2.predict_proba(df2.rename({"f0": "wrong"}))


def test_save_load_roundtrip(tmp_path):
    df, y = _blobs_df(n_per=100)
    clf = LeadX(mode="two_class", tune=False, n_bits=6, n_tables=3, random_state=0)
    clf.fit(df, y)
    expected = clf.predict_proba(df)
    clf.save(tmp_path / "model")
    loaded = LeadX.load(tmp_path / "model")
    np.testing.assert_allclose(loaded.predict_proba(df), expected, rtol=1e-6)
    assert loaded.config_ == clf.config_


def test_feature_weighting_downweights_noise_and_helps():
    # 6 signal + 54 noise features — weighting ควรกู้ signal ที่ถูก noise กลบ
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=6000, n_features=60, n_informative=6,
                               n_redundant=0, n_repeated=0, class_sep=1.5,
                               flip_y=0.01, shuffle=False, random_state=0)
    Xtr, Xte, ytr, yte = X[:4000], X[4000:], y[:4000], y[4000:]
    dtr = pl.DataFrame({f"f{i}": Xtr[:, i] for i in range(60)})
    dte = pl.DataFrame({f"f{i}": Xte[:, i] for i in range(60)})
    base = LeadX(mode="two_class", tune=False, n_bits=10, n_tables=20,
                 random_state=0).fit(dtr, ytr.astype(float))
    wtd = LeadX(mode="two_class", tune=False, n_bits=10, n_tables=20,
                feature_weighting="corr", random_state=0).fit(dtr, ytr.astype(float))
    auc_base = roc_auc_score(yte, base.predict_proba(dte)[:, 1])
    auc_wtd = roc_auc_score(yte, wtd.predict_proba(dte)[:, 1])
    assert auc_wtd > auc_base + 0.03  # weighting กู้ signal ที่ noise กลบ
    # signal features (6 ตัวแรก, shuffle=False) ควรได้น้ำหนักเฉลี่ยสูงกว่า noise
    w = wtd.feature_weights_
    assert w[:6].mean() > 3 * w[6:].mean()


def test_feature_weighting_persists_through_save_load(tmp_path):
    df, y = _blobs_df(n_per=100)
    clf = LeadX(mode="two_class", tune=False, n_bits=6, n_tables=3,
                feature_weighting="corr", random_state=0).fit(df, y)
    expected = clf.predict_proba(df)
    clf.save(tmp_path / "m")
    loaded = LeadX.load(tmp_path / "m")
    np.testing.assert_allclose(loaded.predict_proba(df), expected, rtol=1e-6)


def test_pu_scores_batch_invariant():
    # bug: min-max normalize ด้วยสถิติของ batch → แถวเดียวกันได้ score ต่างกันตาม batch
    rng = np.random.default_rng(2)
    pos = rng.standard_normal((200, 5)) + 2.0
    df_pos = pl.DataFrame({f"f{i}": pos[:, i] for i in range(5)})
    Xq = np.vstack([rng.standard_normal((300, 5)), pos[:50]])
    df_q = pl.DataFrame({f"f{i}": Xq[:, i] for i in range(5)})
    clf = LeadX(mode="pu", tune=False, n_bits=6, n_tables=4, random_state=0).fit(df_pos)
    whole = clf.score_samples(df_q)
    halves = np.concatenate([clf.score_samples(df_q[:100]), clf.score_samples(df_q[100:])])
    np.testing.assert_allclose(whole, halves, rtol=1e-9)


def test_two_class_rejects_single_class_y():
    df, _ = _blobs_df(n_per=30)
    with pytest.raises(ValueError, match="class"):
        LeadX(mode="two_class", tune=False, n_bits=4, n_tables=2).fit(df, np.ones(60))


def test_lift_facade_end_to_end_and_batch_invariant():
    df, y = _blobs_df(n_per=200)
    clf = LeadX(mode="two_class", tune=False, n_bits=8, n_tables=5,
                score_type="lift", alpha=2.0, agg="log", random_state=0).fit(df, y)
    s = clf.score_samples(df)
    assert roc_auc_score(y, s) > 0.95
    assert s.max() > 1.0 and s.min() < 1.0  # สเกล lift จริง ไม่ถูก squash เป็น [0,1]
    # batch invariance
    halves = np.concatenate([clf.score_samples(df[:150]), clf.score_samples(df[150:])])
    np.testing.assert_allclose(s, halves, rtol=1e-9)


def test_lift_pu_facade_keeps_lift_scale(tmp_path):
    rng = np.random.default_rng(3)
    pos = rng.standard_normal((300, 5)) + 2.0
    df_pos = pl.DataFrame({f"f{i}": pos[:, i] for i in range(5)})
    allx = np.vstack([rng.standard_normal((700, 5)), pos[:100]])
    df_all = pl.DataFrame({f"f{i}": allx[:, i] for i in range(5)})
    clf = LeadX(mode="pu", tune=False, n_bits=6, n_tables=4,
                score_type="lift", random_state=0).fit(df_pos, background=df_all)
    s = clf.score_samples(df_all)
    assert s[700:].mean() > 1.0 > s[:200].mean()  # positive เข้มข้น >1, ไกลๆ <1
    # save/load รักษา config lift
    clf.save(tmp_path / "m")
    loaded = LeadX.load(tmp_path / "m")
    np.testing.assert_allclose(loaded.score_samples(df_all), s, rtol=1e-6)


def test_categorical_signal_end_to_end():
    # signal อยู่ใน categorical ล้วน — เดิมพังแทบสุ่ม (0.53) type-aware bits ต้องกู้ได้
    rng = np.random.default_rng(0)
    n = 12000
    num = rng.standard_normal((n, 20))  # noise ล้วน
    ids = [rng.integers(0, 8, n) for _ in range(3)]
    effects = [rng.standard_normal(8) for _ in range(3)]
    logit = sum(e[i] for e, i in zip(effects, ids))
    y = (logit + rng.standard_normal(n) * 0.7 > np.median(logit)).astype(float)
    df = pl.DataFrame({**{f"n{i}": num[:, i] for i in range(20)},
                       **{f"c{j}": [f"v{v}" for v in ids[j]] for j in range(3)}})
    tr, te = slice(0, 8000), slice(8000, n)
    clf = LeadX(mode="two_class", tune=False, n_bits=12, n_tables=15,
                score_type="lift", agg="log", random_state=0).fit(df[tr], y[tr])
    auc = roc_auc_score(y[te], clf.score_samples(df[te]))
    assert auc > 0.80  # เดิม ~0.53


def test_final_bank_is_winning_tuned_prefix():
    # #1: โมเดลสุดท้ายต้องใช้ prefix ของ bank ที่ Tuner ประเมินจริง (ไม่ refit ใหม่)
    from leadx.encoder import FeatureEncoder
    from leadx.tuning import Tuner
    df, y = _blobs_df(n_per=200)
    grid = {"n_bits": [4, 8], "n_tables": [2, 4], "m": [10.0], "weights": ["mean"],
            "strategy": ["unsupervised"], "score_type": ["prop"], "alpha": [2.0],
            "agg": ["mean"], "cat_bit_frac": [0.0]}
    clf = LeadX(mode="two_class", tune=True, grid=grid, random_state=0).fit(df, y)
    b, k = clf.config_["n_bits"], clf.config_["n_tables"]
    # จำลอง tuner run เดียวกัน (encoder + seed เดียวกัน) แล้วเทียบ prefix
    Z = FeatureEncoder().fit(df).transform(df)
    tuner = Tuner(grid=grid, n_folds=5, mode="two_class", random_state=0)
    tuner.tune(Z, y)
    np.testing.assert_array_equal(clf.hasher_.P, tuner.best_bank_.P_[:k, :, :b])
    np.testing.assert_array_equal(clf.hasher_.thresholds,
                                  tuner.best_bank_.thresholds_[:k, :b])


def test_save_load_preserves_hash_mode_and_cat_bit_frac(tmp_path):
    # #4: config identity หลัง load
    df, y = _blobs_df(n_per=60)
    clf = LeadX(mode="two_class", tune=False, n_bits=6, n_tables=3,
                hash_mode="origin", cat_bit_frac=0.0, random_state=0).fit(df, y)
    clf.save(tmp_path / "m")
    loaded = LeadX.load(tmp_path / "m")
    assert loaded.hash_mode == "origin"
    assert loaded.cat_bit_frac == 0.0
    np.testing.assert_allclose(loaded.predict_proba(df), clf.predict_proba(df), rtol=1e-6)


def test_load_old_artifact_without_new_meta(tmp_path):
    # #4: artifact เก่าไม่มี hash_mode/cat_bit_frac ใน meta → fallback + predictions เดิม
    import json
    df, y = _blobs_df(n_per=60)
    clf = LeadX(mode="two_class", tune=False, n_bits=6, n_tables=3, random_state=0).fit(df, y)
    expected = clf.predict_proba(df)
    clf.save(tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "meta.json").read_text())
    meta.pop("hash_mode", None)
    meta["config"].pop("cat_bit_frac", None)
    (tmp_path / "m" / "meta.json").write_text(json.dumps(meta))
    loaded = LeadX.load(tmp_path / "m")
    assert loaded.hash_mode == "median"       # fallback
    assert loaded.cat_bit_frac == "auto"      # fallback
    np.testing.assert_allclose(loaded.predict_proba(df), expected, rtol=1e-6)


def _strip_meta(path, remove_thresholds):
    import json, os
    meta = json.loads((path / "meta.json").read_text())
    meta.pop("hash_mode", None)
    (path / "meta.json").write_text(json.dumps(meta))
    if remove_thresholds:
        os.remove(path / "thresholds.npy")


def test_legacy_inference_era1_origin(tmp_path):
    # era-1: ไม่มีทั้ง hash_mode และ thresholds.npy → ต้อง infer เป็น origin
    df, y = _blobs_df(n_per=60)
    clf = LeadX(mode="two_class", tune=False, n_bits=6, n_tables=3,
                hash_mode="origin", random_state=0).fit(df, y)
    expected = clf.predict_proba(df)
    clf.save(tmp_path / "m")
    _strip_meta(tmp_path / "m", remove_thresholds=True)
    loaded = LeadX.load(tmp_path / "m")
    assert loaded.hash_mode == "origin"
    np.testing.assert_allclose(loaded.predict_proba(df), expected, rtol=1e-6)


def test_legacy_inference_era2_origin_zero_thresholds(tmp_path):
    # era-2 origin: มี thresholds.npy แต่เป็นศูนย์ล้วน → ต้อง infer เป็น origin
    df, y = _blobs_df(n_per=60)
    clf = LeadX(mode="two_class", tune=False, n_bits=6, n_tables=3,
                hash_mode="origin", random_state=0).fit(df, y)
    expected = clf.predict_proba(df)
    clf.save(tmp_path / "m")
    _strip_meta(tmp_path / "m", remove_thresholds=False)
    loaded = LeadX.load(tmp_path / "m")
    assert loaded.hash_mode == "origin"
    np.testing.assert_allclose(loaded.predict_proba(df), expected, rtol=1e-6)


def test_legacy_inference_era2_median_nonzero_thresholds(tmp_path):
    # era-2 median: มี thresholds ไม่เป็นศูนย์ → infer เป็น median
    df, y = _blobs_df(n_per=60)
    clf = LeadX(mode="two_class", tune=False, n_bits=6, n_tables=3,
                hash_mode="median", random_state=0).fit(df, y)
    expected = clf.predict_proba(df)
    clf.save(tmp_path / "m")
    _strip_meta(tmp_path / "m", remove_thresholds=False)
    loaded = LeadX.load(tmp_path / "m")
    assert loaded.hash_mode == "median"
    np.testing.assert_allclose(loaded.predict_proba(df), expected, rtol=1e-6)


def test_tuned_diagnostics_match_deployed_model():
    # #2: diagnostics ต้องสะท้อน prefix ที่ deploy จริง ไม่ใช่ bank เต็ม B_max/K_max
    df, y = _blobs_df(n_per=50)  # seeds น้อย → n_bits ใหญ่ (24) ควรแพ้ → best < max
    grid = {"n_bits": [4, 24], "n_tables": [2, 6], "m": [10.0], "weights": ["mean"],
            "strategy": ["unsupervised"], "score_type": ["prop"], "alpha": [2.0],
            "agg": ["mean"], "cat_bit_frac": [0.0]}
    clf = LeadX(mode="two_class", tune=True, grid=grid, random_state=0).fit(df, y)
    diag = clf.diagnostics_["projections"]
    # invariant หลัก: โครงสร้าง diagnostics == โครงสร้างของ hasher ที่ deploy
    assert len(diag["tables"]) == clf.hasher_.n_tables == clf.config_["n_tables"]
    for t in diag["tables"]:
        assert len(t["balance"]) == clf.hasher_.n_bits == clf.config_["n_bits"]
        assert t["n_eff"] <= 2 ** clf.config_["n_bits"] + 1  # n_eff จาก codes ของ prefix จริง
    # ต้อง exercise slicing path จริง (ถ้า best == max ทั้งคู่ test นี้ไม่ได้ทดสอบอะไร)
    assert (clf.config_["n_bits"] < 24) or (clf.config_["n_tables"] < 6)
