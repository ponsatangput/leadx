"""Phase 1 — Locality headroom benchmark (home-credit PU proxy)

คำถาม: partition geometry ของ leadx ยังมี headroom ไหม?
Oracle = exact kNN + lift scorer เดียวกัน — เพดานของทุกวิธีที่อิง locality
บน encoder/metric ปัจจุบัน ถ้า oracle ไม่ชนะ → คอขวดคือ representation ไม่ใช่ hashing

Adaptations จาก spec ต้นทาง (จดใน report):
- fraud dataset → home-credit PU sim (credit default, ไม่มีคอลัมน์เวลา → stratified split)
- test subsample 20k คงที่ (ทุก method/draw ใช้ชุดเดียวกัน) เพื่อให้ exact kNN คำนวณไหว
- best-k ของ kNN รายงานแบบ per-draw-best (ให้ oracle ได้เปรียบ — gate จะ conservative
  ไปทาง "มี headroom" ซึ่งเป็นทิศที่ปลอดภัย)

รัน: uv run python outputs/locality_headroom_benchmark.py
ผลดิบ: outputs/locality_headroom_raw.csv | สรุป: outputs/LOCALITY_HEADROOM_REPORT.md
"""
import time
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score

from leadx.encoder import FeatureEncoder
from leadx.projections import ProjectionBank
from leadx.hashing import HashEnsemble
from leadx.scoring import SeedScorer

RNG = np.random.default_rng(7)
N_DRAWS = 10
FRACS = [0.10, 0.25]
ALPHA = 2.0
K_GRID = [20, 50, 100, 200, 500]
TEST_N = 20_000

# ---------- data ----------
df = pl.read_csv("/Users/ponpiboonsatangput/jarvis/data/kaggle/home-credit/application_train.csv",
                 infer_schema_length=5000, ignore_errors=True)
y = df["TARGET"].cast(pl.Float64).to_numpy()
df = df.drop(["SK_ID_CURR", "TARGET"])
idx = RNG.permutation(df.height)
n_te = int(df.height * 0.2)
te_full, tr = idx[:n_te], idx[n_te:]
te = te_full[RNG.permutation(n_te)[:TEST_N]]
df_tr, ytr = df[tr.tolist()], y[tr]
df_te, yte = df[te.tolist()], y[te]
enc = FeatureEncoder().fit(df_tr)
Ztr, Zte = enc.transform(df_tr), enc.transform(df_te)
pos_idx = np.nonzero(ytr == 1)[0]
n_tr = len(ytr)
print(f"train {n_tr:,} (pos {int(ytr.sum()):,}) | test {len(yte):,} (pos rate {yte.mean():.2%}) | dims {Ztr.shape[1]}")

# ---------- precompute: leadx codes (y-independent — ใช้ร่วมทุก draw) ----------
CONFIGS = {"leadx_b12k15": (12, 15), "leadx_b10k40": (10, 40),
           "leadx_b10k80": (10, 80)}
CODES = {}
for name, (b, k) in CONFIGS.items():
    t0 = time.perf_counter()
    bank = ProjectionBank(n_tables=k, n_bits=b, random_state=0).fit(Ztr, cat_mask=enc.cat_mask_)
    h = HashEnsemble(bank.P_, bank.thresholds_)
    CODES[name] = (h.transform(Ztr), h.transform(Zte), b)
    print(f"{name}: hashed in {time.perf_counter()-t0:.1f}s")

# ---------- precompute: exact kNN top-500 neighbors (cos + euc) ----------
t0 = time.perf_counter()
Xtr = Ztr.astype(np.float32)
Xte = Zte.astype(np.float32)
tr_norm = np.linalg.norm(Xtr, axis=1) + 1e-12
te_norm = np.linalg.norm(Xte, axis=1) + 1e-12
tr_sq = (Xtr ** 2).sum(1)
KMAX = max(K_GRID)
nbr = {"cos": np.empty((len(yte), KMAX), np.int32),
       "euc": np.empty((len(yte), KMAX), np.int32)}
CH = 1000
for s in range(0, len(yte), CH):
    dot = Xte[s:s+CH] @ Xtr.T                                    # (ch, n_tr)
    cos = dot / (te_norm[s:s+CH, None] * tr_norm[None, :])
    top = np.argpartition(-cos, KMAX, axis=1)[:, :KMAX]
    row_order = np.take_along_axis(cos, top, 1).argsort(1)[:, ::-1]
    nbr["cos"][s:s+CH] = np.take_along_axis(top, row_order, 1)
    euc = tr_sq[None, :] - 2 * dot                                # + te_sq (คงที่ต่อแถว — ไม่กระทบ ranking)
    top = np.argpartition(euc, KMAX, axis=1)[:, :KMAX]
    row_order = np.take_along_axis(euc, top, 1).argsort(1)
    nbr["euc"][s:s+CH] = np.take_along_axis(top, row_order, 1)
print(f"exact kNN neighbors: {time.perf_counter()-t0:.0f}s")

# ---------- metrics ----------
def metrics(scores):
    ap = average_precision_score(yte, scores)
    order = np.argsort(-scores)
    out = {"pr_auc": ap}
    for frac, label in [(0.001, "p_at_0.1pct"), (0.01, "p_at_1pct")]:
        kk = max(1, int(frac * len(yte)))
        out[label] = float(yte[order[:kk]].mean())
    kk = int(0.05 * len(yte))                                     # review budget 5%
    out["recall_at_5pct"] = float(yte[order[:kk]].sum() / yte.sum())
    return out

# ---------- methods (ต่อ draw) ----------
def leadx_score(name, seed_rows):
    ctr, cte, b = CODES[name]
    sc = SeedScorer(mode="pu", score_type="lift", agg="log", alpha=ALPHA,
                    n_bits=b).fit(ctr[seed_rows], np.ones(len(seed_rows)),
                                  background_codes=ctr)
    return sc.score(cte)

def knn_scores(seed_rows, metric):
    """lift ต่อ k ทั้ง grid จาก neighbor list เดียว"""
    is_seed = np.zeros(n_tr, bool); is_seed[seed_rows] = True
    hits = is_seed[nbr[metric]]                                   # (n_te, KMAX)
    cum = np.cumsum(hits, axis=1)
    rho = len(seed_rows) / n_tr
    return {kk: (cum[:, kk-1] + ALPHA) / (kk * rho + ALPHA) for kk in K_GRID}

def mode_aware_score(name, seed_rows, n_modes=4, draw_seed=0):
    ctr, cte, b = CODES[name]
    km = KMeans(n_clusters=n_modes, n_init=3, random_state=draw_seed).fit(Ztr[seed_rows])
    per_mode = []
    for m in range(n_modes):
        rows = seed_rows[km.labels_ == m]
        if len(rows) < 20:
            continue
        sc = SeedScorer(mode="pu", score_type="lift", agg="log", alpha=ALPHA,
                        n_bits=b).fit(ctr[rows], np.ones(len(rows)),
                                      background_codes=ctr)
        per_mode.append(sc.score(cte))
    return np.max(per_mode, axis=0)

def refined_score(seed_rows, tau=500, max_levels=4, draw_seed=0):
    """PU supervised refinement: แตก mega-bucket ด้วยทิศ seed-vs-background ท้องถิ่น"""
    ctr, cte, b = CODES["leadx_b12k15"]
    rng = np.random.default_rng(draw_seed)
    is_seed = np.zeros(n_tr, bool); is_seed[seed_rows] = True
    r_tr = ctr.copy(); r_te = cte.copy()
    for k in range(ctr.shape[1]):
        c_tr, c_te = r_tr[:, k], r_te[:, k]
        for _ in range(max_levels):
            dfc = pl.DataFrame({"code": c_tr, "s": is_seed}).with_row_index("i")
            g = dfc.group_by("code").agg(n=pl.len(), ns=pl.col("s").sum(),
                                         rows=pl.col("i"))
            big = g.filter((pl.col("n") > tau) & (pl.col("ns") >= 5))
            if big.height == 0:
                break
            bit_tr = np.zeros(n_tr, bool)
            te_map = pl.DataFrame({"code": c_te}).with_row_index("i").group_by(
                "code").agg(rows=pl.col("i"))
            te_rows = {r["code"]: np.array(r["rows"], np.int64)
                       for r in te_map.to_dicts()}
            bit_te = np.zeros(len(yte), bool)
            for r in big.to_dicts():
                rows = np.array(r["rows"], np.int64)
                sm = is_seed[rows]
                w = Ztr[rows[sm]].mean(0) - Ztr[rows[~sm]].mean(0)
                w = (w / (np.linalg.norm(w) + 1e-12)).astype(np.float32)
                p = Ztr[rows] @ w
                t = np.median(p)
                bit_tr[rows] = p > t
                trow = te_rows.get(r["code"])
                if trow is not None:
                    bit_te[trow] = Zte[trow] @ w > t
            c_tr = c_tr * np.uint64(2) + bit_tr.astype(np.uint64)
            c_te = c_te * np.uint64(2) + bit_te.astype(np.uint64)
        r_tr[:, k], r_te[:, k] = c_tr, c_te
    sc = SeedScorer(mode="pu", score_type="lift", agg="log", alpha=ALPHA,
                    multi_probe=False).fit(r_tr[seed_rows], np.ones(len(seed_rows)),
                                           background_codes=r_tr)
    return sc.score(r_te)

# ---------- run ----------
rows = []
t_start = time.perf_counter()
for frac in FRACS:
    for draw in range(N_DRAWS):
        rd = np.random.default_rng(1000 * draw + int(frac * 100))
        seed_rows = rd.choice(pos_idx, size=int(frac * len(pos_idx)), replace=False)
        for name in CONFIGS:
            rows.append({"frac": frac, "draw": draw, "method": name,
                         **metrics(leadx_score(name, seed_rows))})
        for metric in ["cos", "euc"]:
            for kk, s in knn_scores(seed_rows, metric).items():
                rows.append({"frac": frac, "draw": draw,
                             "method": f"knn_{metric}_k{kk}", **metrics(s)})
        rows.append({"frac": frac, "draw": draw, "method": "mode_aware_b12k15",
                     **metrics(mode_aware_score("leadx_b12k15", seed_rows,
                                                draw_seed=draw))})
        rows.append({"frac": frac, "draw": draw, "method": "refined_b12k15",
                     **metrics(refined_score(seed_rows, draw_seed=draw))})
    print(f"frac {frac:.0%} done ({time.perf_counter()-t_start:.0f}s)")

out = pl.DataFrame(rows)
out.write_csv("outputs/locality_headroom_raw.csv")

# ---------- aggregate ----------
print("\n=== median ข้าม 10 draws ===")
agg = (out.group_by(["frac", "method"])
       .agg(pl.col("pr_auc").median(), pl.col("p_at_1pct").median(),
            pl.col("recall_at_5pct").median())
       .sort(["frac", "pr_auc"], descending=[False, True]))
with pl.Config(tbl_rows=60, tbl_cols=10):
    print(agg)

# gate: best kNN (per-draw best-k, oracle-favoring) vs best leadx
print("\n=== decision gate ===")
for frac in FRACS:
    sub = out.filter(pl.col("frac") == frac)
    knn_best = (sub.filter(pl.col("method").str.starts_with("knn_"))
                .group_by("draw").agg(pl.col("pr_auc").max().alias("knn")))
    lx = (sub.filter(pl.col("method").is_in(list(CONFIGS)))
          .group_by("draw").agg(pl.col("pr_auc").max().alias("lx")))
    j = knn_best.join(lx, on="draw")
    wins = int((j["knn"] > j["lx"]).sum())
    rel = float((j["knn"].median() / j["lx"].median() - 1) * 100)
    print(f"frac {frac:.0%}: kNN-oracle ชนะ {wins}/10 draws | "
          f"median PR-AUC diff {rel:+.1f}% rel "
          f"({'ผ่าน gate → geometry มี headroom' if wins >= 8 and rel >= 10 else 'ไม่ผ่าน gate → คอขวดคือ representation'})")
