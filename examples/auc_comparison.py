"""AUC comparison: leadx เทียบ baselines หลายตัว บนหลาย dataset

รัน: uv run python examples/auc_comparison.py

วัด AUC ด้วย stratified 5-fold CV (mean ± std) เพื่อให้เทียบได้ยุติธรรม:
- leadx เปิด auto-tune (หา n_bits/n_tables/m/strategy เอง)
- kNN และ LogisticRegression ใส่ StandardScaler ให้ (leadx standardize ในตัว)
- HistGradientBoosting = sklearn เวอร์ชัน LightGBM-like (baseline ที่แข็งสุด)

ทดสอบบน dataset ที่มีลักษณะต่างกัน — linear-separable, non-linear, real-world —
เพื่อดูว่า leadx แข็งตรงไหน อ่อนตรงไหน (ไม่มี method ไหนชนะทุกแบบ)
"""
import warnings

import numpy as np
import polars as pl
from sklearn.datasets import load_breast_cancer, make_classification, make_moons
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from leadx import LeadX

# grid เล็กสำหรับ auto-tune ให้รันไว (default grid ใหญ่กว่า)
LEADX_GRID = {
    "n_bits": [8, 12, 16], "n_tables": [5, 10],
    "m": [1.0, 10.0], "weights": ["mean"],
    "strategy": ["unsupervised", "supervised"],
}


def leadx_cv_auc(X, y, n_splits=5):
    """leadx ต้องใช้ Polars DataFrame — wrap เองใน CV loop"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    aucs = []
    for tr, te in skf.split(X, y):
        df_tr = pl.DataFrame({f"f{i}": X[tr, i] for i in range(X.shape[1])})
        df_te = pl.DataFrame({f"f{i}": X[te, i] for i in range(X.shape[1])})
        clf = LeadX(mode="two_class", tune=True, grid=LEADX_GRID, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(df_tr, y[tr].astype(float))
        aucs.append(roc_auc_score(y[te], clf.predict_proba(df_te)[:, 1]))
    return np.mean(aucs), np.std(aucs)


def sklearn_cv_auc(estimator, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    aucs = []
    for tr, te in skf.split(X, y):
        est = estimator()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], est.predict_proba(X[te])[:, 1]))
    return np.mean(aucs), np.std(aucs)


def make_datasets():
    ds = {}

    # 1) real-world — breast cancer (569 × 30, แยกได้ดีแบบ linear)
    bc = load_breast_cancer()
    ds["breast_cancer (real)"] = (bc.data, bc.target)

    # 2) linear-ish synthetic — informative features เยอะ
    X, y = make_classification(
        n_samples=4000, n_features=20, n_informative=10, n_redundant=5,
        class_sep=1.2, random_state=0)
    ds["synthetic linear"] = (X, y)

    # 3) non-linear — two moons (linear model ควรแพ้ตรงนี้)
    X, y = make_moons(n_samples=4000, noise=0.25, random_state=0)
    ds["two_moons (non-linear)"] = (X, y)

    # 4) hard synthetic — class ซ้อนกันมาก (ทุก method ควรได้ AUC ปานกลาง)
    X, y = make_classification(
        n_samples=4000, n_features=30, n_informative=8, n_redundant=10,
        class_sep=0.6, flip_y=0.1, random_state=0)
    ds["synthetic hard"] = (X, y)

    return ds


def main():
    methods = {
        "leadx (tuned)": None,  # จัดการแยกเพราะใช้ Polars
        "HistGradBoost": lambda: HistGradientBoostingClassifier(random_state=0),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=0),
        "kNN (scaled)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(25)),
        "LogReg (scaled)": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }

    datasets = make_datasets()
    name_w = max(len(m) for m in methods) + 2

    for ds_name, (X, y) in datasets.items():
        X = X.astype(np.float64)
        print(f"\n{'=' * 60}")
        print(f"{ds_name}  —  {X.shape[0]} rows × {X.shape[1]} features")
        print(f"{'=' * 60}")
        print(f"{'method':<{name_w}}{'AUC (5-fold CV)':>18}")
        print("-" * (name_w + 18))
        results = []
        for m_name, factory in methods.items():
            if m_name == "leadx (tuned)":
                mean, std = leadx_cv_auc(X, y)
            else:
                mean, std = sklearn_cv_auc(factory, X, y)
            results.append((m_name, mean, std))
        # เรียงจาก AUC สูงไปต่ำ, mark ตัวที่ดีสุด
        results.sort(key=lambda r: -r[1])
        best = results[0][1]
        for m_name, mean, std in results:
            mark = " ★" if mean == best else ""
            leadx_mark = " ←" if m_name.startswith("leadx") else ""
            print(f"{m_name:<{name_w}}{f'{mean:.3f} ± {std:.3f}':>18}{mark}{leadx_mark}")

    print(f"\n{'=' * 60}")
    print("★ = ดีสุดของ dataset นั้น | ← = leadx")
    print("อ่านผล: leadx มักใกล้ tree-based บน dataset ที่ signal กระจาย")
    print("แต่แพ้บน non-linear ที่ต้องการ boundary โค้ง (เช่น two_moons)")


if __name__ == "__main__":
    main()
