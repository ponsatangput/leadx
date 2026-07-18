"""Hierarchical shrinkage ข้าม prefix depths — โจมตี variance ที่ seeds น้อยมาก

ปัจจุบัน lift shrink เข้าหา 1 (เป็นกลาง): L = (n_pos + α)/(E + α) — ที่ seeds 20-60
ตัว bucket ลึกมี n_pos ∈ {0,1} → variance คือตัวจำกัด ไอเดีย: shrink เข้าหา lift
ของ *bucket แม่* แทน (empirical-Bayes ลูกโซ่ จากหยาบไปลึก):

    L_4 = (n_pos_4 + α·1) / (E_4 + α)          # depth หยาบสุด shrink เข้าหา 1
    L_d = (n_pos_d + α·L_{d-2}) / (E_d + α)    # depth ลึกขึ้น shrink เข้าหาแม่

แถวใน bucket ว่าง inherit หลักฐานจากแม่ (แทน fallback=1/multi-probe)
ต่างจาก adaptive-depth ที่ reject ไป: อันนั้น "เลือก" depth เดียวต่อแถว อันนี้
"รวม" หลักฐานทุก depth — ไม่ทิ้งอะไรเลย

เงื่อนไข: base = SeedScorer ปัจจุบัน (lift/log α=2 multi-probe) vs hier α∈{2,8}
depths (4,6,8,10) บน codes ชุดเดียวกันเป๊ะ
Cells: โซนธง (fraud 0.3%/0.1%, home-credit 1%/0.3%) + natural-rate regression
check (home-credit 8%, 10% seeds) — gate: ชนะ ≥3/4 extreme cells + natural
ไม่ regress เกิน 2% rel

รัน: PYTHONPATH=outputs uv run python outputs/hier_shrinkage_experiment.py
"""
import time
import warnings

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from extreme_imbalance_benchmark import keep_train_valid_columns, sample_ratio
from leadx.encoder import FeatureEncoder
from leadx.hashing import HashEnsemble
from leadx.projections import ProjectionBank
from leadx.scoring import SeedScorer
from real_data_benchmark import load_dataset, split_data

warnings.filterwarnings("ignore")
B, K = 10, 80
DEPTHS = (4, 6, 8, 10)
ALPHAS = (2.0, 8.0)
DRAWS = 3


def lookup_counts(train_codes: np.ndarray, query_codes: np.ndarray) -> np.ndarray:
    """นับสมาชิก bucket ของ train ที่ตำแหน่ง query (numpy ล้วน — เร็วกว่า join)"""
    uniq, cnt = np.unique(train_codes, return_counts=True)
    pos = np.clip(np.searchsorted(uniq, query_codes), 0, len(uniq) - 1)
    return np.where(uniq[pos] == query_codes, cnt[pos], 0).astype(np.float64)


def hier_scores(c_seed, c_bg, c_te, alphas=ALPHAS, depths=DEPTHS, b_max=B):
    """คืน {alpha: score} — chain shrinkage ต่อ table แล้ว geometric mean ข้าม K"""
    n_seeds, n_bg = len(c_seed), len(c_bg)
    logL = {a: np.zeros(len(c_te)) for a in alphas}
    for k in range(c_te.shape[1]):
        L = {a: np.ones(len(c_te)) for a in alphas}
        for d in depths:
            sh = np.uint64(b_max - d)
            npos = lookup_counts(c_seed[:, k] >> sh, c_te[:, k] >> sh)
            nbg = lookup_counts(c_bg[:, k] >> sh, c_te[:, k] >> sh)
            E = n_seeds * nbg / n_bg
            for a in alphas:
                L[a] = (npos + a * L[a]) / (E + a)
        for a in alphas:
            logL[a] += np.log(L[a])
    return {a: logL[a] / c_te.shape[1] for a in alphas}


def eval_cell(name, tag, X_bg, X_te, y_te, pos_idx, seed_frac, rows):
    """pipeline เดียวกับ deployed PU: encoder fit บน background (parity) + contrast"""
    enc = FeatureEncoder().fit(X_bg)
    Z_bg, Z_te = enc.transform(X_bg), enc.transform(X_te)
    for d in range(DRAWS):
        rd = np.random.default_rng(100 * d + 7)
        n_seeds = max(2, int(seed_frac * len(pos_idx)))
        seeds = rd.choice(pos_idx, size=n_seeds, replace=False)
        w = np.abs(Z_bg[seeds].mean(axis=0) - Z_bg.mean(axis=0)) + 1e-3
        w = (w / w.max()).astype(np.float32)
        bank = ProjectionBank(n_tables=K, n_bits=B, strategy="unsupervised",
                              hash_mode="median", cat_bit_frac="auto",
                              random_state=0).fit(Z_bg * w, cat_mask=enc.cat_mask_)
        h = HashEnsemble(bank.P_, bank.thresholds_)
        c_seed, c_bg, c_te = (h.transform(Z_bg[seeds] * w), h.transform(Z_bg * w),
                              h.transform(Z_te * w))
        out = {}
        base = SeedScorer(mode="pu", score_type="lift", alpha=2.0, agg="log",
                          n_bits=B).fit(c_seed, np.ones(n_seeds),
                                        background_codes=c_bg)
        out["base"] = average_precision_score(y_te, base.score(c_te))
        for a, s in hier_scores(c_seed, c_bg, c_te).items():
            out[f"hier_a{a:g}"] = average_precision_score(y_te, s)
        rows.append({"cell": name, "tag": tag, "n_seeds": n_seeds, "draw": d, **out})
        print(f"  {name} d={d}: " + " ".join(f"{k}={v:.4f}" for k, v in out.items()),
              flush=True)


def ratio_cells(rows):
    for name, ratios in [("fraud", (0.003, 0.001)), ("home_credit", (0.01, 0.003))]:
        df, target, split_type, max_rows = load_dataset(name)
        train_full, test_full = split_data(df, target, split_type, max_rows)
        for ratio in ratios:
            train = sample_ratio(train_full, target, ratio, 42)
            test = sample_ratio(test_full, target, ratio, 43)
            train, test = keep_train_valid_columns(train, test, target)
            y_tr = train[target].to_numpy().astype(np.float64)
            y_te = test[target].to_numpy().astype(np.float64)
            eval_cell(f"{name}@{ratio:.3%}", "extreme", train.drop(target),
                      test.drop(target), y_te, np.flatnonzero(y_tr == 1), 0.10, rows)


def natural_cell(rows):
    rng = np.random.default_rng(7)
    df = pl.read_csv(
        "/Users/ponpiboonsatangput/jarvis/data/kaggle/home-credit/application_train.csv",
        infer_schema_length=5000, ignore_errors=True)
    y = df["TARGET"].cast(pl.Float64).to_numpy()
    df = df.drop(["SK_ID_CURR", "TARGET"])
    idx = rng.permutation(df.height)
    n_te = int(df.height * 0.2)
    te = idx[:n_te][rng.permutation(n_te)[:20_000]]
    tr = idx[n_te:]
    eval_cell("home_credit@natural8%", "regression_check", df[tr.tolist()],
              df[te.tolist()], y[te], np.flatnonzero(y[tr] == 1), 0.10, rows)


if __name__ == "__main__":
    t0 = time.time()
    rows = []
    ratio_cells(rows)
    natural_cell(rows)
    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/hier_shrinkage_raw.csv")
    cols = [c for c in raw.columns if c.startswith(("base", "hier"))]
    med = (raw.group_by(["cell", "tag"]).agg([pl.col(c).median() for c in cols])
           .sort("cell"))
    print("\n=== median across draws ===")
    with pl.Config(tbl_rows=20, tbl_width_chars=140):
        print(med)
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min → outputs/hier_shrinkage_raw.csv")
