"""Large + imbalanced benchmark: leadx เทียบ baselines ที่ scale ได้

รัน: uv run python examples/large_imbalanced.py

โจทย์ที่ leadx ออกแบบมารองรับ — ข้อมูลใหญ่ (หลักแสน–ล้านแถว) + positive หายาก
(1–2%) แบบงาน lead scoring / fraud / churn วัดทั้ง:
- ROC-AUC — ความสามารถแยกโดยรวม
- Average Precision (PR-AUC) — สำคัญกว่าบน imbalanced (สะท้อนการจัดอันดับ positive)
- เวลา fit / predict — จุดขายหลักของ leadx บนข้อมูลใหญ่

เทียบเฉพาะ method ที่ scale ถึงระดับนี้ได้: HistGradientBoosting, LogisticRegression
(kNN/RandomForest predict ช้าเกินบนข้อมูลล้านแถว จึงไม่รวม)
"""
import time
import warnings

import numpy as np
import polars as pl
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from leadx import LeadX

# grid เน้น bucket หยาบ (n_bits ต่ำ) — positive หายากทำให้ bucket ละเอียดว่างเปล่า
# strategy=unsupervised อย่างเดียว (supervised ช้าและไม่ช่วยบน imbalanced นี้)
LEADX_GRID = {
    "n_bits": [8, 10, 12], "n_tables": [10, 20],
    "m": [1.0, 10.0], "weights": ["mean", "coverage"],
    "strategy": ["unsupervised"],
}


def make_imbalanced(n_train, n_test, pos_rate, d=40, seed=0):
    X, y = make_classification(
        n_samples=n_train + n_test, n_features=d, n_informative=15, n_redundant=10,
        weights=[1 - pos_rate, pos_rate], class_sep=1.0, flip_y=0.01, random_state=seed)
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]
    df_tr = pl.DataFrame({f"f{i}": Xtr[:, i] for i in range(d)})
    df_te = pl.DataFrame({f"f{i}": Xte[:, i] for i in range(d)})
    return Xtr, ytr, Xte, yte, df_tr, df_te


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def eval_leadx(Xtr, ytr, Xte, yte, df_tr, df_te):
    def fit():
        clf = LeadX(mode="two_class", tune=True, grid=LEADX_GRID, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return clf.fit(df_tr, ytr.astype(float))
    clf, t_fit = timed(fit)
    proba, t_pred = timed(lambda: clf.predict_proba(df_te)[:, 1])
    return proba, t_fit, t_pred


def eval_sklearn(factory, Xtr, ytr, Xte, yte, df_tr, df_te):
    def fit():
        est = factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return est.fit(Xtr, ytr)
    est, t_fit = timed(fit)
    proba, t_pred = timed(lambda: est.predict_proba(Xte)[:, 1])
    return proba, t_fit, t_pred


def main():
    methods = {
        "leadx (tuned)": None,
        "HistGradBoost": lambda: HistGradientBoostingClassifier(random_state=0),
        "LogReg (scaled)": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
    }

    configs = [
        (100_000, 500_000, 0.02),
        (500_000, 1_000_000, 0.01),
    ]

    for n_train, n_test, pos_rate in configs:
        data = make_imbalanced(n_train, n_test, pos_rate)
        n_pos = int(data[1].sum())
        print(f"\n{'=' * 72}")
        print(f"train {n_train:,} ({n_pos:,} pos = {pos_rate:.0%}) | test {n_test:,}")
        print(f"{'=' * 72}")
        print(f"{'method':<18}{'ROC-AUC':>10}{'PR-AUC':>10}{'fit(s)':>10}{'pred(s)':>10}")
        print("-" * 58)
        rows = []
        for m_name, factory in methods.items():
            if m_name == "leadx (tuned)":
                proba, t_fit, t_pred = eval_leadx(*data)
            else:
                proba, t_fit, t_pred = eval_sklearn(factory, *data)
            auc = roc_auc_score(data[3], proba)
            ap = average_precision_score(data[3], proba)
            rows.append((m_name, auc, ap, t_fit, t_pred))
        for m_name, auc, ap, t_fit, t_pred in rows:
            mark = " ←" if m_name.startswith("leadx") else ""
            print(f"{m_name:<18}{auc:>10.3f}{ap:>10.3f}{t_fit:>10.1f}{t_pred:>10.2f}{mark}")
        print(f"\nbaseline PR-AUC (random) = positive rate = {pos_rate:.3f}")

    # ---- leadx niche จริง: PU (มีแต่ positive + unlabeled, ไม่มี labeled negative) ----
    print(f"\n{'=' * 72}")
    print("leadx niche — PU learning: rank positives จาก positive-only + unlabeled")
    print("(HistGradBoost/LogReg รันไม่ได้เลยเพราะไม่มี labeled negative)")
    print(f"{'=' * 72}")
    Xtr, ytr, Xte, yte, df_tr, df_te = make_imbalanced(500_000, 1_000_000, 0.01)
    df_pos = df_tr.filter(pl.Series(ytr == 1))  # seeds = positive เท่านั้น
    def fit_pu():
        clf = LeadX(mode="pu", tune=False, n_bits=10, n_tables=20, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return clf.fit(df_pos, background=df_tr)  # ทั้งชุด = unlabeled background
    clf, t_fit = timed(fit_pu)
    scores, t_pred = timed(lambda: clf.score_samples(df_te))
    ap = average_precision_score(yte, scores)
    print(f"\nleadx PU: PR-AUC={ap:.3f} (baseline {yte.mean():.3f}) | "
          f"fit={t_fit:.1f}s pred={t_pred:.2f}s | seeds={int(ytr.sum()):,} positives")

    print(f"\n{'=' * 72}")
    print("สรุปตามจริง:")
    print("• บน imbalanced ที่มี labeled negative → gradient boosting ชนะขาด")
    print("  (PR-AUC สูงกว่า ~3x และ fit เร็วกว่า) — leadx ไม่ใช่เครื่องมือที่เหมาะ")
    print("• leadx คุ้มเมื่อ (1) เป็น PU ไม่มี labeled negative ให้ boosting ใช้ หรือ")
    print("  (2) ต้อง deploy เป็น hash table ธรรมดา ไม่มี model server")
    print("• predict เร็วจริงแต่ 'เร็วกว่า kNN' ไม่ใช่ 'เร็วกว่า boosting'")


if __name__ == "__main__":
    main()
