"""วัดตัวที่จะ ship โดยตรง: LeadX(engine="learned") vs XGBoost-PU ที่จูนแล้ว

**ที่มา:** `learned_engine_parity.py` ไม่ผ่านเกณฑ์ (|delta| <= 5%) — library ดีกว่า
โครงสร้างของการทดลองอย่างสม่ำเสมอที่ seeds น้อย (fraud@0.3%: +9.05% median, 5/5
reps สูงกว่า) และเท่ากันที่ seeds เยอะ (+0.40%) ความต่างมาจากการ train แบบ
case-control และการแบ่ง fold ด้วย rng ของ library ตัวเลข +32.5% เหนือ XGBoost เดิม
จึงเป็นของโครงสร้างการทดลอง ไม่ใช่ของโค้ดที่ ship — ต้องวัดโค้ดที่ ship เอง

setup เหมือน `learned_vs_tuned_xgb.py` ทุกประการ (4 cells, การจูน XGBoost ด้วย CV
บน seeds) ต่างแค่ arm ของเราเป็น `LeadX(engine="learned")` จาก library จริง

---------------------------------------------------------------------------
PRE-REGISTERED
---------------------------------------------------------------------------
Cells   : fraud@0.3%, fraud@0.1%, home_credit@1%, criteo@1e-4 (4M, 50% seeds)
          5 reps, **seed draw ใหม่** rng(19000 + 41*rep), random_state = 700 + rep
Arms    : base (rare_pu), learned_lib (engine="learned", n_bits=10, n_tables=80,
          contrast), xgb_tuned
PRIMARY : learned_lib vs xgb_tuned
GATE    : pooled 20 obs median paired delta pr_auc > +2% AND ชนะ >= 14/20
          AND median paired delta precision@0.1% >= 0 AND ชนะ >= 3/5 ทุก cell
ผ่าน -> ship engine="learned" โดยใช้ตัวเลขจากรอบนี้ใน README
ไม่ผ่าน -> ไม่ ship

รัน: PYTHONPATH=outputs uv run --with xgboost python outputs/learned_engine_vs_tuned_xgb.py
"""
import gc
import time

import numpy as np
import polars as pl

from extreme_imbalance_benchmark import metrics
from leadx import LeadX
from learned_partition_experiment import BASE
from learned_vs_tuned_xgb import (XGB_BASE, paired, prep_criteo, prep_tabular, spw,
                                  tune_xgb)

N_REPS = 5
LIB_KW = dict(mode="pu", tune=False, engine="learned", n_bits=10, n_tables=80,
              score_type="lift", agg="log", feature_weighting="contrast")


def run_cell(cell, X_tr, y_tr, X_te, y_te, n_seeds, rows):
    import xgboost as xgb
    pos = np.flatnonzero(y_tr == 1)
    print(f"\n[{cell}] bg={X_tr.height:,} test={len(y_te):,} pos_test={int(y_te.sum())} "
          f"seeds={n_seeds}", flush=True)
    for rep in range(N_REPS):
        t0 = time.perf_counter()
        rng = np.random.default_rng(19000 + 41 * rep)
        sd = rng.choice(pos, size=n_seeds, replace=False)
        rs = 700 + rep
        model = LeadX(**BASE, random_state=rs).fit(X_tr[sd.tolist()], background=X_tr)
        base = model.score_samples(X_te)
        Z_bg = model.encoder_.transform(model._drop_mh(X_tr))
        Z_te = model.encoder_.transform(model._drop_mh(X_te))
        lib = LeadX(**LIB_KW, random_state=rs).fit(X_tr[sd.tolist()], background=X_tr)
        learned = lib.score_samples(X_te, chunk_rows=200_000)
        s_all = np.zeros(X_tr.height)
        s_all[sd] = 1.0
        n_p, n_u = int(s_all.sum()), int(len(s_all) - s_all.sum())
        (d_best, w_best), _ = tune_xgb(Z_bg, sd, rng, xgb)
        xgb_tuned = (xgb.XGBClassifier(**XGB_BASE, max_depth=d_best, random_state=rs,
                                       scale_pos_weight=spw(n_p, n_u, w_best))
                     .fit(Z_bg, s_all).predict_proba(Z_te)[:, 1])
        for v, sc in (("base", base), ("learned_lib", learned), ("xgb_tuned", xgb_tuned)):
            rows.append({"cell": cell, "rep": rep, "variant": v,
                         "xgb_choice": f"d{d_best}-{w_best}", **metrics(y_te, sc)})
        print(f"  rep={rep} xgb=d{d_best}/{w_best} | " + " | ".join(
            f"{r['variant']} pr={r['pr_auc']:.4f}" for r in rows[-3:])
            + f" ({time.perf_counter()-t0:.0f}s)", flush=True)
        del model, lib, Z_bg, Z_te
        gc.collect()


def main() -> None:
    rows: list[dict] = []
    run_cell("fraud@0.3%", *prep_tabular("fraud", 0.003), rows)
    run_cell("fraud@0.1%", *prep_tabular("fraud", 0.001), rows)
    run_cell("home_credit@1%", *prep_tabular("home_credit", 0.01), rows)
    run_cell("criteo@1e-4", *prep_criteo(), rows)
    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/learned_engine_vs_tuned_xgb_raw.csv")
    print("\n=== ระดับ PR-AUC (median) ===")
    print(raw.group_by(["cell", "variant"]).agg(pr=pl.col("pr_auc").median().round(4))
          .pivot(on="variant", index="cell", values="pr"))
    print(raw.group_by(["cell", "variant"])
          .agg(p01=pl.col("precision_at_0_1pct").median().round(4))
          .pivot(on="variant", index="cell", values="p01"))
    cells, pooled = paired(raw, "xgb_tuned", ["learned_lib", "base"])
    print("\n=== PRIMARY: vs xgb_tuned ===")
    print(cells)
    print(pooled.with_columns(gate=(pl.col("variant") == "learned_lib") & (pl.col("d_pr") > 2)
                              & (pl.col("wins") >= 14) & (pl.col("d_p01") >= 0)
                              & (pl.col("min_cell_wins") >= 3)))
    cells, pooled = paired(raw, "base", ["learned_lib"])
    print("\n=== secondary: learned_lib vs base ===")
    print(cells)
    print(pooled)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n({time.time()-t0:.0f}s)")
