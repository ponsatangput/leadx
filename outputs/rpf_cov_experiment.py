"""สองการวัดต่อจาก rpforest ship:

A) anomaly fraud 0.1% allseeds (rpf .029 < bank .044) — ตารางเดิม confound:
   bank allseeds วัดแบบ *ไม่มี hier* แต่ rpf มี hier → แยก {engine} x {hier} x
   {forest seed} ให้ชัดว่าอะไรคือสาเหตุ

B) covariance-sampled directions: ทิศต่อ node ~ N(0, Sigma_node) ผ่าน
   g = normalize((Z_node - mu).T @ r), r ~ N(0,1)^n — "PCA-flavored แต่สุ่ม":
   E[gg^T] ~ Sigma_node จึงปรับตาม local structure โดยไม่ทำให้ trees ซ้ำกัน
   (top-PC deterministic จะทำ K trees identical) และแพงกว่า random แค่ ~2x
   gate: ชนะ rpf_hier (จาก rpforest_raw.csv, draw seeds เดียวกัน) >=3/4
   extreme cells margin >2% rel และ natural ไม่ regress >2%

รัน: PYTHONPATH=outputs uv run python outputs/rpf_cov_experiment.py
"""
import time
import warnings

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from extreme_imbalance_benchmark import keep_train_valid_columns, sample_ratio
from leadx.encoder import FeatureEncoder
from leadx.rpforest import RPForest, _node_threshold
from leadx.scoring import SeedScorer
from real_data_benchmark import load_dataset, split_data

warnings.filterwarnings("ignore")
B, K = 10, 80
DEPTHS = (4, 6, 8, 10)
DRAWS = 3


class CovRPForest(RPForest):
    """ทิศต่อ node สุ่มจาก covariance ท้องถิ่น: g ~ N(0, Sigma_node)"""

    def fit(self, Z: np.ndarray) -> "CovRPForest":
        rng = np.random.default_rng(self.random_state)
        Z = np.ascontiguousarray(Z, dtype=np.float32)
        if Z.shape[0] > self.sample_size:
            Z = Z[rng.permutation(Z.shape[0])[: self.sample_size]]
        n, d = Z.shape
        n_nodes = 2 ** self.n_bits - 1
        self.dim = d
        self.dirs_ = np.zeros((self.n_tables, n_nodes, d), dtype=np.float32)
        self.thr_ = np.zeros((self.n_tables, n_nodes), dtype=np.float32)
        for t in range(self.n_tables):
            node_of = np.zeros(n, dtype=np.int64)
            for lvl in range(self.n_bits):
                base = 2 ** lvl - 1
                order = np.argsort(node_of, kind="stable")
                bounds = np.searchsorted(node_of[order], np.arange(2 ** lvl + 1))
                new_node = np.empty(n, dtype=np.int64)
                for node in range(2 ** lvl):
                    idx = order[bounds[node]:bounds[node + 1]]
                    if len(idx) >= 4:
                        sub = Z[idx]
                        r = rng.standard_normal(len(idx)).astype(np.float32)
                        g = (sub - sub.mean(axis=0)).T @ r
                    else:
                        g = rng.standard_normal(d).astype(np.float32)
                    nrm = np.linalg.norm(g)
                    if nrm < 1e-6:
                        g = rng.standard_normal(d).astype(np.float32)
                        nrm = np.linalg.norm(g)
                    g = (g / nrm).astype(np.float32)
                    self.dirs_[t, base + node] = g
                    if len(idx) == 0:
                        continue
                    proj = Z[idx] @ g
                    thr = _node_threshold(proj)
                    self.thr_[t, base + node] = thr
                    new_node[idx] = 2 * node + (proj > thr)
                node_of = new_node
        return self


def contrast_w(Z_bg, seeds):
    w = np.abs(Z_bg[seeds].mean(axis=0) - Z_bg.mean(axis=0)) + 1e-3
    return (w / w.max()).astype(np.float32)


def score_codes(c_seed, c_bg, c_te, y_te, hier):
    sc = SeedScorer(mode="pu", score_type="lift", agg="log", n_bits=B,
                    hier_depths=hier).fit(c_seed, np.ones(len(c_seed)),
                                          background_codes=c_bg)
    return average_precision_score(y_te, sc.score(c_te))


def part_a_anomaly():
    """fraud 0.1% allseeds: {bank, rpf} x {hier, flat} x rs{0,1,2}"""
    from leadx.hashing import HashEnsemble
    from leadx.projections import ProjectionBank
    print("=== A) fraud 0.1% allseeds decomposition ===", flush=True)
    df, target, split_type, max_rows = load_dataset("fraud")
    train_full, test_full = split_data(df, target, split_type, max_rows)
    train = sample_ratio(train_full, target, 0.001, 42)
    test = sample_ratio(test_full, target, 0.001, 43)
    train, test = keep_train_valid_columns(train, test, target)
    y_tr = train[target].to_numpy().astype(np.float64)
    y_te = test[target].to_numpy().astype(np.float64)
    enc = FeatureEncoder().fit(train.drop(target))
    Z_bg, Z_te = enc.transform(train.drop(target)), enc.transform(test.drop(target))
    seeds = np.flatnonzero(y_tr == 1)
    w = contrast_w(Z_bg, seeds)
    Zbg_w, Zte_w, Zs_w = Z_bg * w, Z_te * w, Z_bg[seeds] * w
    rows = []
    for rs in (0, 1, 2):
        bank = ProjectionBank(n_tables=K, n_bits=B, strategy="unsupervised",
                              hash_mode="median", cat_bit_frac="auto",
                              random_state=rs).fit(Zbg_w, cat_mask=enc.cat_mask_)
        h = HashEnsemble(bank.P_, bank.thresholds_)
        cs, cbg, cte = h.transform(Zs_w), h.transform(Zbg_w), h.transform(Zte_w)
        rpf = RPForest(n_tables=K, n_bits=B, random_state=rs).fit(Zbg_w)
        fs, fbg, fte = rpf.transform(Zs_w), rpf.transform(Zbg_w), rpf.transform(Zte_w)
        for eng, (a, b, c) in [("bank", (cs, cbg, cte)), ("rpf", (fs, fbg, fte))]:
            for hier in (DEPTHS, None):
                pr = score_codes(a, b, c, y_te, hier)
                tag = f"{eng}_{'hier' if hier else 'flat'}"
                rows.append({"cond": tag, "rs": rs, "pr": pr})
                print(f"  {tag} rs={rs}: {pr:.4f}", flush=True)
    med = (pl.DataFrame(rows).group_by("cond").agg(pr=pl.col("pr").median())
           .sort("cond"))
    print(med, flush=True)
    return rows


def part_b_cov():
    print("\n=== B) covariance-sampled directions ===", flush=True)
    rows = []

    def eval_cell(name, tag, X_bg, X_te, y_te, pos_idx, seed_frac):
        enc = FeatureEncoder().fit(X_bg)
        Z_bg, Z_te = enc.transform(X_bg), enc.transform(X_te)
        for d in range(DRAWS):
            rd = np.random.default_rng(100 * d + 7)
            n_seeds = max(2, int(seed_frac * len(pos_idx)))
            seeds = rd.choice(pos_idx, size=n_seeds, replace=False)
            w = contrast_w(Z_bg, seeds)
            Zbg_w, Zte_w, Zs_w = Z_bg * w, Z_te * w, Z_bg[seeds] * w
            t0 = time.perf_counter()
            f = CovRPForest(n_tables=K, n_bits=B, random_state=0).fit(Zbg_w)
            cs, cbg, cte = (f.transform(Zs_w), f.transform(Zbg_w),
                            f.transform(Zte_w))
            t_fit = time.perf_counter() - t0
            out = {"cov_hier": score_codes(cs, cbg, cte, y_te, DEPTHS),
                   "cov_flat": score_codes(cs, cbg, cte, y_te, None)}
            rows.append({"cell": name, "tag": tag, "draw": d,
                         "t_cov_s": round(t_fit, 1), **out})
            print(f"  {name} d={d}: " +
                  " ".join(f"{k}={v:.4f}" for k, v in out.items()) +
                  f" ({t_fit:.0f}s)", flush=True)

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
                      test.drop(target), y_te, np.flatnonzero(y_tr == 1), 0.10)
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
              df[te.tolist()], y[te], np.flatnonzero(y[tr] == 1), 0.10)
    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/rpf_cov_raw.csv")
    # เทียบกับ rpf random (draw seeds เดียวกัน) จาก rpforest_raw.csv
    prev = (pl.read_csv("outputs/rpforest_raw.csv")
            .group_by("cell").agg(rpf_hier=pl.col("rpf_hier").median()))
    med = (raw.group_by(["cell", "tag"])
           .agg(cov_hier=pl.col("cov_hier").median(),
                cov_flat=pl.col("cov_flat").median(),
                t_cov_s=pl.col("t_cov_s").median())
           .join(prev, on="cell").sort("cell"))
    print("\n=== median (cov vs rpf random เดิม) ===")
    with pl.Config(tbl_rows=20, tbl_width_chars=150):
        print(med)


if __name__ == "__main__":
    t0 = time.time()
    a = part_a_anomaly()
    pl.DataFrame(a).write_csv("outputs/rpf_anomaly_raw.csv")
    part_b_cov()
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min")
