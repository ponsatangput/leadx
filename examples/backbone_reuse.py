"""Backbone reuse: hash ครั้งเดียว แล้วใช้กับหลายโจทย์บน rows เดียว (scenario A)

รัน: uv run python examples/backbone_reuse.py

จุดแข็งเชิงสถาปัตยกรรมของ leadx ที่ boosting เลียนแบบไม่ได้: projections เป็น
y-independent → encode+project+hash ครั้งเดียว (แพง) แล้วแต่ละ labeling ใหม่เหลือแค่
SeedScorer group-by บน codes ที่ cache ไว้ (~30ms) — boosting ต้องเทรนใหม่ทั้งต้นทุกครั้ง
ที่ y เปลี่ยน เพราะ tree splits ขึ้นกับ y
"""
import time
import warnings

import numpy as np
import polars as pl
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from leadx import LeadXBackbone


def main():
    rng = np.random.default_rng(0)
    n, d = 200_000, 40
    X, _ = make_classification(n_samples=n, n_features=d, n_informative=20,
                               n_redundant=10, random_state=0)
    df = pl.DataFrame({f"f{i}": X[:, i] for i in range(d)})

    # 6 โจทย์บน rows เดียวกัน (label ขึ้นกับ feature กลุ่มต่างกัน)
    tasks = {}
    for k in range(6):
        w = rng.standard_normal(d) * (rng.random(d) < 0.3)
        tasks[f"task{k}"] = (X @ w + rng.standard_normal(n) * 0.5 > 0).astype(float)

    tr, te = slice(0, 150_000), slice(150_000, n)

    print("=" * 60)
    print("leadx backbone — hash ครั้งเดียว, reuse หลายโจทย์")
    print("=" * 60)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bb = LeadXBackbone(n_bits=16, n_tables=15, random_state=0).fit(df[tr])
    t_backbone = time.perf_counter() - t0
    print(f"\nbackbone.fit (encode+project+hash ครั้งเดียว): {t_backbone:.2f}s")

    print(f"\n{'task':<8}{'leadx fit_task':>16}{'leadx AUC':>11}{'HistGB fit':>12}{'HistGB AUC':>12}")
    print("-" * 59)
    lx_total = 0.0
    for name, y in tasks.items():
        t0 = time.perf_counter()
        task = bb.fit_task(y[tr], mode="two_class")
        t_lx = time.perf_counter() - t0
        lx_total += t_lx
        auc_lx = roc_auc_score(y[te], task.predict_proba(df[te])[:, 1])

        t0 = time.perf_counter()
        gb = HistGradientBoostingClassifier(random_state=0).fit(X[tr], y[tr])
        t_gb = time.perf_counter() - t0
        auc_gb = roc_auc_score(y[te], gb.predict_proba(X[te])[:, 1])

        print(f"{name:<8}{t_lx:>15.3f}s{auc_lx:>11.3f}{t_gb:>11.2f}s{auc_gb:>12.3f}")

    print("-" * 59)
    print(f"\nleadx: backbone {t_backbone:.1f}s + {len(tasks)} โจทย์ {lx_total:.2f}s "
          f"= {t_backbone + lx_total:.1f}s รวม")
    print(f"       (marginal {lx_total/len(tasks)*1000:.0f}ms/โจทย์)")
    print("HistGB: ต้อง fit เต็มทุกโจทย์ — ต้นทุนโตเชิงเส้นตามจำนวนโจทย์")
    print("\nอ่านผล: leadx แม่นน้อยกว่า HistGB ต่อโจทย์ แต่ marginal cost ต่อโจทย์")
    print("แทบเป็นศูนย์ — คุ้มเมื่อมีโจทย์เยอะบน feature space เดียว (multi-label,")
    print("ทดลองหลายนิยาม target, re-score รายวันด้วย label ที่อัปเดต)")


if __name__ == "__main__":
    main()
