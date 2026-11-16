"""แปลงผลการทดลองเป็น precision / recall ที่งบรีวิวคงที่ (top 0.1% และ top 1%)

PR-AUC สรุปทั้งเส้นแต่ทีมรีวิวถามว่า "รีวิว k รายแล้วจับได้กี่ราย" — สคริปต์นี้แปลง
precision_at_* ใน raw CSV เป็นจำนวนที่จับได้, precision และ recall ต่อ cell
(median ข้าม reps) ขนาด test ต่อ cell ตรงกับ prep ใน learned_vs_tuned_xgb.py

รัน: uv run python outputs/budget_table.py <raw.csv> [variant ...]
"""
import math
import sys

import polars as pl

TEST_SIZE = {   # (n_test, n_positive_test) จาก log ของ prep เดียวกัน
    "fraud@0.3%": (87_166, 261),
    "fraud@0.1%": (86_992, 87),
    "home_credit@1%": (41_786, 418),
    "criteo@1e-4": (798_494, 82),
}
BUDGETS = ((0.001, "precision_at_0_1pct"), (0.01, "precision_at_1pct"))


def budget_table(raw: pl.DataFrame, variants=None) -> pl.DataFrame:
    rows = []
    for d in raw.iter_rows(named=True):
        if d["cell"] not in TEST_SIZE or (variants and d["variant"] not in variants):
            continue
        n, n_pos = TEST_SIZE[d["cell"]]
        for frac, col in BUDGETS:
            k = math.ceil(n * frac)
            hits = round(d[col] * k)
            rows.append({"cell": d["cell"], "budget": f"top {frac:.1%}",
                         "variant": d["variant"], "reviewed": k, "caught": hits,
                         "precision": hits / k, "recall": hits / n_pos})
    return (pl.DataFrame(rows).group_by(["cell", "budget", "variant"])
            .agg(reviewed=pl.col("reviewed").first(), caught=pl.col("caught").median(),
                 precision=pl.col("precision").median().round(3),
                 recall=pl.col("recall").median().round(3))
            .sort(["cell", "budget", "variant"]))


if __name__ == "__main__":
    pl.Config.set_tbl_rows(60)
    print(budget_table(pl.read_csv(sys.argv[1]), sys.argv[2:] or None))
