"""End-to-end worked example ของ leadx บนข้อมูลที่มี categorical + missing values

รัน: uv run python examples/worked_example.py

สาธิต workflow เต็ม: สร้างข้อมูลสังเคราะห์ → fit + auto-tune → predict →
อ่าน diagnostics → save/load — บนข้อมูลที่มีทั้ง numeric, categorical และ NaN
"""
import warnings

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from leadx import LeadX


def make_dataset(n: int, seed: int) -> tuple[pl.DataFrame, np.ndarray]:
    """ข้อมูลลูกค้าสังเคราะห์: signal มาจาก numeric 2 ตัว + categorical 'plan'

    - age, income  : numeric (income มี missing 5% + outlier)
    - region       : categorical cardinality ต่ำ (one-hot)
    - user_id      : categorical cardinality สูง (feature hashing) — noise ล้วน
    - plan         : categorical ที่สัมพันธ์กับ label
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 12, n)
    income = rng.lognormal(10.5, 0.5, n)
    region = rng.choice(["north", "south", "central", "east", "west"], n)
    user_id = np.array([f"u{rng.integers(0, n * 5)}" for _ in range(n)])
    plan = rng.choice(["free", "pro", "enterprise"], n, p=[0.6, 0.3, 0.1])

    # label: churn สูงเมื่ออายุน้อย + income ต่ำ + plan free
    logit = (
        -0.05 * (age - 40)
        - 1.2e-5 * (income - 36000)
        + np.where(plan == "free", 0.8, -0.6)
        + rng.normal(0, 0.5, n)
    )
    y = (logit > 0).astype(np.float64)

    # ใส่ missing + outlier ใน income
    income[rng.random(n) < 0.05] = np.nan
    income[rng.random(n) < 0.002] = 1e9

    df = pl.DataFrame({
        "age": age, "income": income,
        "region": region, "user_id": user_id, "plan": plan,
    })
    return df, y


def main() -> None:
    print("=" * 64)
    print("leadx worked example — churn prediction (two-class)")
    print("=" * 64)

    df_train, y_train = make_dataset(20_000, seed=0)
    df_test, y_test = make_dataset(5_000, seed=1)
    print(f"\nTrain: {df_train.height:,} rows | Test: {df_test.height:,} rows")
    print(f"Positive rate: {y_train.mean():.1%}")
    print(f"Schema: {dict(df_train.schema)}")
    n_missing = int((df_train["income"].is_null() | df_train["income"].is_nan()).sum())
    print(f"Missing income (null or NaN): {n_missing} rows")

    # ---- fit + auto-tune -------------------------------------------------
    # grid เล็กลงเพื่อให้ตัวอย่างรันเร็ว (default grid ใหญ่กว่านี้)
    grid = {
        "n_bits": [10, 16], "n_tables": [5, 10],
        "m": [1.0, 10.0], "weights": ["mean"],
        "strategy": ["unsupervised", "supervised"],
    }
    clf = LeadX(mode="two_class", tune=True, grid=grid, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(df_train, y_train)

    print("\n--- Auto-tuned config -------------------------------------------")
    print(clf.config_)

    print("\n--- Top 5 configs (out-of-fold AUC) -----------------------------")
    print(clf.tuning_results_.head(5))

    # ---- predict ---------------------------------------------------------
    proba = clf.predict_proba(df_test)
    auc = roc_auc_score(y_test, proba[:, 1])
    print("\n--- Test performance --------------------------------------------")
    print(f"AUC: {auc:.3f}")

    # ---- diagnostics -----------------------------------------------------
    print("\n--- Projection diagnostics --------------------------------------")
    proj = clf.diagnostics_["projections"]
    print(f"strategy: {proj['strategy']} | table diversity (NMI): {proj['diversity_nmi']:.3f}")
    t0 = proj["tables"][0]
    print(f"table 0: n_eff={t0['n_eff']:.0f} buckets, "
          f"max bit-redundancy={t0['max_redundancy']:.2f}, "
          f"balance range=[{min(t0['balance']):.2f}, {max(t0['balance']):.2f}]")
    print(f"scoring miss rate: {clf.diagnostics_['scoring']['miss_rate_final']:.1%}")

    # ---- save / load -----------------------------------------------------
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "churn_model"
        clf.save(path)
        reloaded = LeadX.load(path)
        proba2 = reloaded.predict_proba(df_test)
        assert np.allclose(proba, proba2), "reloaded model ให้ผลไม่ตรง!"
        print("\n--- Save/load ---------------------------------------------------")
        print(f"saved to {path.name}/ → reloaded → predictions identical ✓")

    print("\n" + "=" * 64)
    print("PU example — เดาว่าใครน่าจะ churn จาก 'confirmed churners' อย่างเดียว")
    print("=" * 64)
    # ใช้เฉพาะ positive ที่ยืนยันแล้วเป็น seeds + ประชากรทั้งหมดเป็น background
    pos_mask = y_train == 1.0
    clf_pu = LeadX(mode="pu", tune=False, n_bits=12, n_tables=8, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_pu.fit(df_train.filter(pos_mask), background=df_train)
    scores = clf_pu.score_samples(df_test)
    ap = average_precision_score(y_test, scores)
    print(f"\nPU ranking — average precision: {ap:.3f} "
          f"(baseline positive rate {y_test.mean():.3f})")
    print(f"score range: [{scores.min():.2f}, {scores.max():.2f}]")


if __name__ == "__main__":
    main()
