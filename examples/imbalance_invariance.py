"""ทดสอบ imbalance-invariance ของ score_type="lift"

รัน: uv run python examples/imbalance_invariance.py

สมมุติฐาน: lift = density ratio ที่ class prior ไม่อยู่ในสูตร → ตรึงจำนวน positive
สัมบูรณ์คงที่ แล้วเพิ่ม negative ให้ imbalance สุดโต่งขึ้นเรื่อยๆ ranking quality
(ROC-AUC — prevalence-invariant) ควร "แบนราบ" ไม่ทรุดตาม ratio

setup: X|pos ~ N(mu, I), X|neg ~ N(0, I) (Bayes boundary คงที่ทุก ratio)
       train: 2,000 positives คงที่ + negatives แปรผัน | test: 5k/5k balanced
"""
import warnings

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from leadx import LeadX

D = 20
MU = 2.0  # ||mu|| — ระยะระหว่าง class means


def gen(n_pos, n_neg, seed):
    rng = np.random.default_rng(seed)
    mu = np.zeros(D)
    mu[:4] = MU / 2.0  # signal ใน 4 มิติแรก, ||mu||=2
    Xp = rng.standard_normal((n_pos, D)) + mu
    Xn = rng.standard_normal((n_neg, D))
    X = np.vstack([Xp, Xn])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    return X, y


def to_df(X):
    return pl.DataFrame({f"f{i}": X[:, i] for i in range(D)})


def main():
    n_pos_train = 2_000
    Xte, yte = gen(5_000, 5_000, seed=99)  # balanced holdout — ROC-AUC ไม่ขึ้นกับ prevalence
    dte = to_df(Xte)

    configs = dict(tune=False, n_bits=10, n_tables=15, random_state=0)
    print(f"positive train คงที่ {n_pos_train:,} ตัว | test 5k/5k balanced | ROC-AUC")
    print(f"\n{'pos rate':>9}{'n_train':>12}{'lift(mean)':>12}{'lift(log)':>11}"
          f"{'prop':>8}{'HistGB':>9}")
    print("-" * 61)
    for pos_rate in [0.10, 0.03, 0.01, 0.003, 0.001]:
        n_neg = int(n_pos_train * (1 - pos_rate) / pos_rate)
        Xtr, ytr = gen(n_pos_train, n_neg, seed=0)
        dtr = to_df(Xtr)
        row = [f"{pos_rate:>9.1%}", f"{len(ytr):>12,}"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for st, ag in [("lift", "mean"), ("lift", "log"), ("prop", "mean")]:
                clf = LeadX(mode="two_class", score_type=st, agg=ag, alpha=2.0,
                            m=10.0, **configs).fit(dtr, ytr)
                auc = roc_auc_score(yte, clf.score_samples(dte))
                row.append(f"{auc:>11.3f}" if st == "lift" and ag == "mean"
                           else f"{auc:>10.3f}" if st == "lift"
                           else f"{auc:>7.3f}")
            gb = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
            row.append(f"{roc_auc_score(yte, gb.predict_proba(Xte)[:, 1]):>8.3f}")
        print("".join(row))

    print("\nอ่านผล: คอลัมน์ lift ควรแบนราบ (prior ไม่อยู่ในสูตร) — ratio เปลี่ยน 100 เท่า")
    print("ranking ไม่สะเทือน ตราบใดที่จำนวน positive สัมบูรณ์คงที่")


if __name__ == "__main__":
    main()
