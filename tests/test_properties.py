"""Property tests ตามทฤษฎี LSH + integration tests (spec §10)"""
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from leadx import LeadX
from leadx.hashing import HashEnsemble


def test_collision_probability_matches_theory():
    """P(bit ตรงกัน) = 1 - θ/π สำหรับ random hyperplane hashing (Goemans–Williamson)"""
    rng = np.random.default_rng(0)
    d = 20
    for theta in (np.pi / 6, np.pi / 3, np.pi / 2):
        # สร้างคู่เวกเตอร์ทำมุม theta ใน random planes หลายคู่
        agrees = 0
        trials = 4000
        for _ in range(trials):
            u = rng.standard_normal(d)
            u /= np.linalg.norm(u)
            v_perp = rng.standard_normal(d)
            v_perp -= (v_perp @ u) * u
            v_perp /= np.linalg.norm(v_perp)
            v = np.cos(theta) * u + np.sin(theta) * v_perp
            p = rng.standard_normal(d)
            agrees += (u @ p > 0) == (v @ p > 0)
        expected = 1.0 - theta / np.pi
        assert abs(agrees / trials - expected) < 0.03, f"theta={theta}"


def test_similar_rows_collide_more_via_hash_ensemble():
    rng = np.random.default_rng(1)
    d = 16
    base = rng.standard_normal((200, d)).astype(np.float32)
    near = base + 0.1 * rng.standard_normal((200, d)).astype(np.float32)
    far = rng.standard_normal((200, d)).astype(np.float32)
    P = rng.standard_normal((5, d, 10)).astype(np.float32)
    h = HashEnsemble(P)
    c_base, c_near, c_far = h.transform(base), h.transform(near), h.transform(far)
    near_rate = (c_base == c_near).mean()
    far_rate = (c_base == c_far).mean()
    assert near_rate > far_rate + 0.2


def _blobs_df(n_per, d, sep, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n_per, d))
    X1 = rng.standard_normal((n_per, d)) + sep / np.sqrt(d)
    X = np.vstack([X0, X1])
    df = pl.DataFrame({f"f{i}": X[:, i] for i in range(d)})
    y = np.concatenate([np.zeros(n_per), np.ones(n_per)])
    return df, y, X


def test_overlapping_blobs_auc_near_theory():
    # ระยะห่างระหว่าง class means = 2.0 → Bayes AUC = Φ(2/√2) ≈ 0.921
    df, y, _ = _blobs_df(n_per=2000, d=8, sep=2.0, seed=2)
    clf = LeadX(mode="two_class", tune=False, n_bits=8, n_tables=10, random_state=0)
    clf.fit(df, y)
    auc = roc_auc_score(y, clf.predict_proba(df)[:, 1])
    assert 0.80 < auc <= 0.95  # ใกล้ Bayes limit, ไม่ทะลุแบบผิดปกติ


def test_direction_agrees_with_knn():
    df, y, X = _blobs_df(n_per=500, d=6, sep=3.0, seed=3)
    clf = LeadX(mode="two_class", tune=False, n_bits=8, n_tables=8, random_state=0)
    clf.fit(df, y)
    lsh_scores = clf.predict_proba(df)[:, 1]
    knn = KNeighborsClassifier(n_neighbors=25).fit(X, y)
    knn_scores = knn.predict_proba(X)[:, 1]
    rho, _ = spearmanr(lsh_scores, knn_scores)
    assert rho > 0.5  # ไปทางเดียวกัน ไม่ต้องเท่า
