"""parity: engine="learned" ใน library ให้ผลเท่ากับ leaf_naive ที่ผ่าน gate หรือไม่

ข้อแตกต่างเดียวที่ตั้งใจ: การทดลอง (`learned_partition_*`) ติด label 1 ให้ seeds
*ตรงตำแหน่งใน background* แต่ core API ของ LeadX ไม่รู้ว่าแถวไหนของ background คือ
seed จึง train forest แบบ case-control (ต่อ seeds เป็น label 1 เพิ่มเข้าไป seed จึง
ปรากฏทั้งเป็น 1 และเป็น 0) ซึ่งเป็นสูตร PU มาตรฐาน ส่วนการแบ่ง fold ของ seeds ก็ใช้
rng ของ library เอง — ทั้งสองอย่างทำให้ผลไม่เท่ากันเป๊ะ จึงต้องวัดว่าใกล้พอหรือไม่

---------------------------------------------------------------------------
PRE-REGISTERED
---------------------------------------------------------------------------
Cells   : fraud@0.3% (61 seeds) และ fraud natural (4000 seeds), 5 reps
          seed draw rng(17000 + 37*rep), random_state = 600 + rep (ใช้ร่วมกันทุก arm)
Arms    : base (rare_pu), experiment (leaf_naive construction เดิม),
          library (LeadX(engine="learned"), n_bits=10, n_tables=80, contrast)
PARITY  : ทุก cell -> median paired |delta pr_auc| ของ library vs experiment <= 5%
          AND library ชนะ base >= 4/5 ที่ fraud@0.3%
ถ้าไม่ผ่าน -> ไม่ ship engine จนกว่าจะแก้ความต่างของการ train

รัน: PYTHONPATH=outputs uv run python outputs/learned_engine_parity.py
"""
import gc
import time

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier

from extreme_imbalance_benchmark import keep_train_valid_columns, metrics, sample_ratio
from leadx import LeadX
from learned_partition_experiment import BASE, RF_KW, leaf_log_lift
from real_data_benchmark import load_dataset, split_data

N_REPS = 5
CELLS = (("fraud", 0.003, None), ("fraud", None, 4000))
LIB_KW = dict(mode="pu", tune=False, engine="learned", n_bits=10, n_tables=80,
              score_type="lift", agg="log", feature_weighting="contrast")


def run_cell(name, ratio, n_seeds_fixed, rows):
    df, target, split_type, max_rows = load_dataset(name)
    train, test = split_data(df, target, split_type, max_rows)
    if ratio is not None:
        train = sample_ratio(train, target, ratio, 42)
        test = sample_ratio(test, target, ratio, 43)
    train, test = keep_train_valid_columns(train, test, target)
    y_tr = train[target].to_numpy().astype(np.float64)
    y_te = test[target].to_numpy().astype(np.float64)
    X_tr, X_te = train.drop(target), test.drop(target)
    pos = np.flatnonzero(y_tr == 1)
    n_seeds = n_seeds_fixed or max(2, int(0.10 * len(pos)))
    cell = f"{name}@{'natural' if ratio is None else f'{ratio:.1%}'}"
    print(f"\n[{cell}] seeds={n_seeds}", flush=True)
    for rep in range(N_REPS):
        t0 = time.perf_counter()
        rng = np.random.default_rng(17000 + 37 * rep)
        sd = rng.choice(pos, size=n_seeds, replace=False)
        rs = 600 + rep
        model = LeadX(**BASE, random_state=rs).fit(X_tr[sd.tolist()], background=X_tr)
        base = model.score_samples(X_te)
        alpha = float(model.config_["alpha"])
        Z_bg = model.encoder_.transform(model._drop_mh(X_tr))
        Z_te = model.encoder_.transform(model._drop_mh(X_te))
        halves = np.array_split(rng.permutation(sd), 2)
        exp_score = np.zeros(len(y_te))
        for f in range(2):
            s = np.zeros(X_tr.height)
            s[halves[f]] = 1.0
            rf = RandomForestClassifier(**RF_KW, random_state=rs * 10 + f).fit(Z_bg, s)
            L_bg, L_te = rf.apply(Z_bg), rf.apply(Z_te)
            exp_score += leaf_log_lift(L_bg, L_bg[halves[f]], L_te, alpha) / 2
            del rf, L_bg, L_te
        lib = LeadX(**LIB_KW, random_state=rs).fit(X_tr[sd.tolist()], background=X_tr)
        lib_score = lib.score_samples(X_te, chunk_rows=50_000)
        for v, sc in (("base", base), ("experiment", exp_score), ("library", lib_score)):
            rows.append({"cell": cell, "rep": rep, "variant": v, **metrics(y_te, sc)})
        print(f"  rep={rep} | " + " | ".join(
            f"{r['variant']} pr={r['pr_auc']:.4f}" for r in rows[-3:])
            + f" ({time.perf_counter()-t0:.0f}s)", flush=True)
        del model, lib, Z_bg, Z_te
        gc.collect()


def main() -> None:
    rows: list[dict] = []
    for name, ratio, ns in CELLS:
        run_cell(name, ratio, ns, rows)
    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/learned_engine_parity_raw.csv")
    p = raw.pivot(on="variant", index=["cell", "rep"], values="pr_auc")
    p = p.with_columns(
        d_lib_exp=(100 * (pl.col("library") - pl.col("experiment")) / pl.col("experiment")),
        lib_beats_base=pl.col("library") > pl.col("base"))
    print(p)
    s = p.group_by("cell").agg(
        median_abs_d=pl.col("d_lib_exp").abs().median().round(2),
        median_d=pl.col("d_lib_exp").median().round(2),
        lib_wins_vs_base=pl.col("lib_beats_base").sum()).sort("cell")
    print(s)
    ok = (s["median_abs_d"] <= 5).all() and \
        s.filter(pl.col("cell") == "fraud@0.3%")["lib_wins_vs_base"][0] >= 4
    print(f"\nPARITY: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n({time.time()-t0:.0f}s)")
