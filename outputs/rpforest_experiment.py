"""Balanced Local Projection Forest (rpTree) — P2 architecture experiment

สมมติฐาน: global hyperplanes + global median ยังให้ bucket โตในย่านหนาแน่น/ว่างใน
ย่านโปร่ง — rpTree แบ่ง recursive ด้วย *ทิศสุ่มใหม่ต่อ node + median ท้องถิ่นของ
node* → ทุก leaf มีสมาชิกเท่ากันโดยโครงสร้าง (equal-mass partition)

ต่างจากที่เคย reject: pair-difference = ทิศ global, adaptive-depth = เลือก depth
ต่อแถวบน global prefix — อันนี้ทั้งทิศและจุดตัด *ปรับต่อ node* จริง

ความเข้ากันได้: bit ที่ level l วางที่ MSB → prefix ของ leaf code = บรรพบุรุษใน tree
→ SeedScorer / mask_codes / hier_depths ใช้ได้ทั้งชุดไม่ต้องแก้ (hier = chain
ตาม tree path ตรงตัว)

เงื่อนไข (codes ต่าง แต่ scoring stack เดียวกัน, contrast weights เดียวกัน):
  bank_hier : ProjectionBank b10 K80 + hier (baseline ดีสุดปัจจุบัน — 8d)
  rpf_hier  : RPForest depth10 K80 + hier
  rpf_flat  : RPForest + flat lift (แยก contribution ของ hier ใน forest)

Gate (ตั้งก่อนเห็นผล): rpf_hier ชนะ bank_hier >=3/4 extreme cells (margin >2% rel)
และ natural 8% ไม่ regress เกิน 2% rel — ไม่ผ่าน = reject การลงทุน v3 นี้

รัน: PYTHONPATH=outputs uv run python outputs/rpforest_experiment.py
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
DRAWS = 3


def _node_threshold(proj: np.ndarray) -> float:
    """tie-aware balanced cut (ตรรกะเดียวกับ _balanced_thresholds แบบคอลัมน์เดียว)"""
    srt = np.sort(proj)
    cuts = np.nonzero(srt[:-1] < srt[1:])[0]
    if len(cuts) == 0:
        return float(srt[0])
    frac = (cuts + 1) / len(srt)
    best = cuts[int(np.argmin(np.abs(frac - 0.5)))]
    return float((srt[best] + srt[best + 1]) / 2.0)


class RPForest:
    """forest ของ balanced rpTrees — ทิศสุ่มใหม่ + median ท้องถิ่นต่อ node

    leaf code: bit ของ level l ที่ตำแหน่ง MSB (depth-1-l) → nested prefix
    เหมือน HashEnsemble ทุกประการ
    """

    def __init__(self, n_trees=K, depth=B, random_state=0):
        self.n_trees = n_trees
        self.depth = depth
        self.random_state = random_state

    def fit(self, Z: np.ndarray) -> "RPForest":
        rng = np.random.default_rng(self.random_state)
        Z = np.ascontiguousarray(Z, dtype=np.float32)
        n, d = Z.shape
        n_nodes = 2 ** self.depth - 1
        self.dirs_ = np.zeros((self.n_trees, n_nodes, d), dtype=np.float32)
        self.thr_ = np.zeros((self.n_trees, n_nodes), dtype=np.float32)
        for t in range(self.n_trees):
            node_of = np.zeros(n, dtype=np.int64)
            for lvl in range(self.depth):
                base = 2 ** lvl - 1
                order = np.argsort(node_of, kind="stable")
                bounds = np.searchsorted(node_of[order], np.arange(2 ** lvl + 1))
                new_node = np.empty(n, dtype=np.int64)
                for node in range(2 ** lvl):
                    idx = order[bounds[node]:bounds[node + 1]]
                    g = rng.standard_normal(d).astype(np.float32)
                    g /= np.linalg.norm(g) + 1e-12
                    self.dirs_[t, base + node] = g
                    if len(idx) == 0:
                        continue
                    proj = Z[idx] @ g
                    thr = _node_threshold(proj)
                    self.thr_[t, base + node] = thr
                    new_node[idx] = 2 * node + (proj > thr)
                node_of = new_node
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        Z = np.ascontiguousarray(Z, dtype=np.float32)
        n = Z.shape[0]
        codes = np.zeros((n, self.n_trees), dtype=np.uint64)
        for t in range(self.n_trees):
            node_of = np.zeros(n, dtype=np.int64)
            for lvl in range(self.depth):
                base = 2 ** lvl - 1
                G = self.dirs_[t, base + node_of]            # (n, d) gather
                proj = np.einsum("ij,ij->i", Z, G)
                bit = proj > self.thr_[t, base + node_of]
                codes[:, t] |= bit.astype(np.uint64) << np.uint64(self.depth - 1 - lvl)
                node_of = 2 * node_of + bit
        return codes


def occupancy(codes_col: np.ndarray) -> dict:
    _, cnt = np.unique(codes_col, return_counts=True)
    return {"max_share": float(cnt.max() / cnt.sum()),
            "p99_size": float(np.quantile(cnt, 0.99)), "n_buckets": len(cnt)}


def eval_cell(name, tag, X_bg, X_te, y_te, pos_idx, seed_frac, rows, diag_done=[False]):
    enc = FeatureEncoder().fit(X_bg)
    Z_bg, Z_te = enc.transform(X_bg), enc.transform(X_te)
    for d in range(DRAWS):
        rd = np.random.default_rng(100 * d + 7)
        n_seeds = max(2, int(seed_frac * len(pos_idx)))
        seeds = rd.choice(pos_idx, size=n_seeds, replace=False)
        w = np.abs(Z_bg[seeds].mean(axis=0) - Z_bg.mean(axis=0)) + 1e-3
        w = (w / w.max()).astype(np.float32)
        Zbg_w, Zte_w, Zs_w = Z_bg * w, Z_te * w, Z_bg[seeds] * w
        out, times = {}, {}
        # --- baseline: global bank + hier ---
        t0 = time.perf_counter()
        bank = ProjectionBank(n_tables=K, n_bits=B, strategy="unsupervised",
                              hash_mode="median", cat_bit_frac="auto",
                              random_state=0).fit(Zbg_w, cat_mask=enc.cat_mask_)
        h = HashEnsemble(bank.P_, bank.thresholds_)
        cb_s, cb_bg, cb_te = h.transform(Zs_w), h.transform(Zbg_w), h.transform(Zte_w)
        times["bank"] = time.perf_counter() - t0
        sc = SeedScorer(mode="pu", score_type="lift", agg="log", n_bits=B,
                        hier_depths=DEPTHS).fit(cb_s, np.ones(n_seeds),
                                                background_codes=cb_bg)
        out["bank_hier"] = average_precision_score(y_te, sc.score(cb_te))
        # --- rpforest ---
        t0 = time.perf_counter()
        rpf = RPForest(n_trees=K, depth=B, random_state=0).fit(Zbg_w)
        cf_s, cf_bg, cf_te = (rpf.transform(Zs_w), rpf.transform(Zbg_w),
                              rpf.transform(Zte_w))
        times["rpf"] = time.perf_counter() - t0
        for cond, hd in [("rpf_hier", DEPTHS), ("rpf_flat", None)]:
            sc = SeedScorer(mode="pu", score_type="lift", agg="log", n_bits=B,
                            hier_depths=hd).fit(cf_s, np.ones(n_seeds),
                                                background_codes=cf_bg)
            out[cond] = average_precision_score(y_te, sc.score(cf_te))
        if not diag_done[0]:
            print(f"  [occupancy@{name} bg] bank={occupancy(cb_bg[:, 0])} "
                  f"rpf={occupancy(cf_bg[:, 0])}", flush=True)
            diag_done[0] = True
        rows.append({"cell": name, "tag": tag, "n_seeds": n_seeds, "draw": d,
                     "t_bank_s": round(times["bank"], 1),
                     "t_rpf_s": round(times["rpf"], 1), **out})
        print(f"  {name} d={d}: " + " ".join(f"{c}={v:.4f}" for c, v in out.items())
              + f" (bank {times['bank']:.0f}s, rpf {times['rpf']:.0f}s)", flush=True)


def main():
    rows = []
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
    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/rpforest_raw.csv")
    med = (raw.group_by(["cell", "tag"])
           .agg([pl.col(c).median() for c in
                 ["bank_hier", "rpf_hier", "rpf_flat", "t_bank_s", "t_rpf_s"]])
           .sort("cell"))
    print("\n=== median across draws ===")
    with pl.Config(tbl_rows=20, tbl_width_chars=150):
        print(med)
    print("raw → outputs/rpforest_raw.csv")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {(time.time() - t0) / 60:.1f} min")
