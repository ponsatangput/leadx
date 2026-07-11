"""Benchmark: leadx เทียบกับ baselines (sklearn kNN, LightGBM ถ้ามี)

รัน: uv run python examples/benchmark.py

วัด AUC + เวลา fit/predict หลายขนาดข้อมูล เพื่อให้เห็นจุดคุ้ม/ไม่คุ้มของ leadx:
- leadx เร็วขึ้นแบบ linear และคงที่ต่อ predict — คู่แข่งที่ scale ไม่ดีจะช้าลงเร็ว
- ความแม่นมักต่ำกว่า gradient boosting เล็กน้อย แต่แลกกับความเร็วมหาศาลบนข้อมูลใหญ่

หมายเหตุ: นี่คือ synthetic benchmark เพื่อดู scaling/latency — ตัวเลข AUC บนข้อมูล
จริงขึ้นกับ dataset; อย่าตีความว่าเป็นอันดับความแม่นสากล
"""
import time
import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

import polars as pl

from leadx import LeadX

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def make_data(n_train, n_test, d=30, seed=0):
    X, y = make_classification(
        n_samples=n_train + n_test, n_features=d, n_informative=12,
        n_redundant=6, class_sep=1.0, random_state=seed,
    )
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]
    df_tr = pl.DataFrame({f"f{i}": Xtr[:, i] for i in range(d)})
    df_te = pl.DataFrame({f"f{i}": Xte[:, i] for i in range(d)})
    return (Xtr, ytr, Xte, yte, df_tr, df_te)


def bench_leadx(Xtr, ytr, Xte, yte, df_tr, df_te):
    def fit():
        clf = LeadX(mode="two_class", tune=False,
                            n_bits=16, n_tables=10, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return clf.fit(df_tr, ytr.astype(float))
    clf, t_fit = timed(fit)
    proba, t_pred = timed(lambda: clf.predict_proba(df_te)[:, 1])
    return roc_auc_score(yte, proba), t_fit, t_pred


def bench_knn(Xtr, ytr, Xte, yte, df_tr, df_te):
    clf, t_fit = timed(lambda: KNeighborsClassifier(n_neighbors=25).fit(Xtr, ytr))
    proba, t_pred = timed(lambda: clf.predict_proba(Xte)[:, 1])
    return roc_auc_score(yte, proba), t_fit, t_pred


def bench_lgbm(Xtr, ytr, Xte, yte, df_tr, df_te):
    clf, t_fit = timed(lambda: LGBMClassifier(n_estimators=100, verbose=-1).fit(Xtr, ytr))
    proba, t_pred = timed(lambda: clf.predict_proba(Xte)[:, 1])
    return roc_auc_score(yte, proba), t_fit, t_pred


def main():
    methods = [("leadx", bench_leadx), ("sklearn-kNN", bench_knn)]
    if HAS_LGBM:
        methods.append(("LightGBM", bench_lgbm))
    else:
        print("note: LightGBM ไม่ได้ติดตั้ง — ข้าม (uv add lightgbm --group dev เพื่อเทียบ)\n")

    sizes = [(5_000, 50_000), (50_000, 200_000), (200_000, 1_000_000)]

    header = f"{'method':<14}{'train':>10}{'test':>12}{'AUC':>8}{'fit(s)':>9}{'pred(s)':>9}"
    print(header)
    print("-" * len(header))
    for n_train, n_test in sizes:
        data = make_data(n_train, n_test)
        for name, fn in methods:
            # kNN predict บน 1M เพื่อนบ้านคือ O(n_test·n_train) — ข้ามถ้าใหญ่เกิน
            if name == "sklearn-kNN" and n_train * n_test > 2e10:
                print(f"{name:<14}{n_train:>10,}{n_test:>12,}{'skip':>8}{'':>9}{'':>9}"
                      "   (predict O(n_train·n_test))")
                continue
            auc, t_fit, t_pred = fn(*data)
            print(f"{name:<14}{n_train:>10,}{n_test:>12,}"
                  f"{auc:>8.3f}{t_fit:>9.2f}{t_pred:>9.2f}")
        print()


if __name__ == "__main__":
    main()
