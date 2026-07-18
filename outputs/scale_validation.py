"""Scale validation 1M-10M แถว × 100 features — พิสูจน์จุดเด่น "train ครั้งเดียว
ใช้หลายโจทย์" ที่ scale ตั้งต้นของโจทย์จริง (เดิม benchmark ทั้งหมดวัดที่ 200-300k)

วัดต่อ scale (synthetic, 95 numeric f32 + 5 categorical, 5 PU tasks ปลูก signal
คนละกลุ่ม features, positives ~1%):
  1. backbone.fit        — encode + projections + hash (จ่ายครั้งเดียว)
  2. fit_task ×5         — seed_rows + background="cached" (ไม่ re-hash เลย)
  3. score_cached        — จัดอันดับประชากรทั้งก้อน (task 0) + ROC ทุก task
  4. HistGB tuned        — เพดานอ้างอิงราคา retrain ต่อโจทย์ (n <= 3M)
  5. peak RSS ต่อ stage

รัน: PYTHONPATH=outputs uv run python outputs/scale_validation.py
เครื่องอ้างอิง: Apple Silicon 24GB RAM
"""
import gc
import resource
import time
import warnings

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from leadx import LeadXBackbone
from leadx.hashing import mask_codes

warnings.filterwarnings("ignore")
SCALES = [1_000_000, 3_000_000, 10_000_000]
N_NUM, N_CAT, N_TASKS = 95, 5, 5
TASK_KW = dict(mode="pu", n_bits=10, score_type="lift", agg="log",
               hier_depths="auto", background="cached")


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30


def gen_data(n, seed=0):
    """สร้าง df + labels 5 tasks — task j ใช้ cols (2j, 2j+1) เป็น signal"""
    rng = np.random.default_rng(seed)
    cols, ys = {}, []
    for i in range(N_NUM):
        cols[f"f{i}"] = rng.normal(size=n).astype(np.float32)
    for j in range(N_TASKS):
        sig = cols[f"f{2 * j}"] + cols[f"f{2 * j + 1}"] \
            + 0.8 * rng.normal(size=n).astype(np.float32)
        thr = np.quantile(sig, 0.99)
        ys.append((sig > thr).astype(np.int8))
    for i in range(N_CAT):
        cols[f"cat{i}"] = rng.integers(0, 12, size=n)
    df = pl.DataFrame(cols).with_columns(
        [pl.col(f"cat{i}").cast(pl.Utf8) for i in range(N_CAT)])
    return df, ys


def run_scale(n, rows):
    print(f"\n===== n = {n:,} =====", flush=True)
    df, ys = gen_data(n)
    gc.collect()
    print(f"  data ready (rss {peak_gb():.1f} GB)", flush=True)

    t0 = time.perf_counter()
    bb = LeadXBackbone(n_bits=16, n_tables=40, random_state=0).fit(df)
    t_backbone = time.perf_counter() - t0
    rss_backbone = peak_gb()
    print(f"  backbone.fit: {t_backbone:.1f}s (rss {rss_backbone:.1f} GB)", flush=True)

    # HistGB อ้างอิง (ราคา retrain ต่อโจทย์, label เต็ม) — ข้ามที่ 10M กัน swap บน 24GB
    t_histgb = None
    if n <= 3_000_000:
        Z = bb.encoder_.transform(df)
        t0 = time.perf_counter()
        HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=150, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=0).fit(Z, ys[0])
        t_histgb = time.perf_counter() - t0
        print(f"  histgb (1 task, label เต็ม): {t_histgb:.1f}s", flush=True)
        del Z
    del df
    gc.collect()

    rng = np.random.default_rng(1)
    tasks, t_tasks = [], []
    for j in range(N_TASKS):
        pos = np.flatnonzero(ys[j] == 1)
        seed_rows = rng.choice(pos, size=max(20, len(pos) // 10), replace=False)
        t0 = time.perf_counter()
        tasks.append(bb.fit_task(seed_rows=seed_rows, **TASK_KW))
        t_tasks.append(time.perf_counter() - t0)
    print(f"  fit_task x{N_TASKS}: {[f'{t:.2f}s' for t in t_tasks]}", flush=True)

    t0 = time.perf_counter()
    s0 = tasks[0].score_cached()
    t_score = time.perf_counter() - t0
    rocs = [roc_auc_score(ys[0], s0)]
    # tasks ที่เหลือ: ROC บน subsample 300k (คุมเวลา — join เต็มวัดแล้วที่ task 0)
    codes_b10 = mask_codes(bb._seed_codes, 10, 16)
    sub = np.random.default_rng(2).choice(n, size=min(n, 300_000), replace=False)
    for j in range(1, N_TASKS):
        rocs.append(roc_auc_score(ys[j][sub], tasks[j].scorer_.score(codes_b10[sub])))
    print(f"  score_cached (ทั้งประชากร): {t_score:.1f}s | "
          f"ROC {[f'{r:.3f}' for r in rocs]} (rss {peak_gb():.1f} GB)", flush=True)

    rows.append({
        "n_rows": n, "backbone_fit_s": round(t_backbone, 1),
        "task_fit_median_s": round(float(np.median(t_tasks)), 2),
        "task_fit_max_s": round(max(t_tasks), 2),
        "score_population_s": round(t_score, 1),
        "histgb_one_task_s": None if t_histgb is None else round(t_histgb, 1),
        "roc_median": round(float(np.median(rocs)), 4),
        "roc_min": round(min(rocs), 4),
        "peak_rss_gb": round(peak_gb(), 1),
    })
    del bb, tasks, codes_b10, s0, ys
    gc.collect()


if __name__ == "__main__":
    t_all = time.time()
    rows = []
    for n in SCALES:
        run_scale(n, rows)
    out = pl.DataFrame(rows)
    out.write_csv("outputs/scale_validation_raw.csv")
    print("\n=== summary ===")
    with pl.Config(tbl_width_chars=160):
        print(out)
    print(f"\ntotal {(time.time() - t_all) / 60:.1f} min → outputs/scale_validation_raw.csv")
