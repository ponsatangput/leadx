"""Phase 2B — Learned geometry (home-credit PU, harness เดียวกับ Phase 1)

วิธีที่ทดสอบ (ทุกตัวส่ง binary codes เข้า SeedScorer เดิม — แยก geometry จาก scorer):
  unsupervised-reusable:
    - baseline_b10k80      random proj + tie-aware median (แชมป์ Phase 1)
    - pca_rr_b10k80        PCA-whiten 64 มิติ → random rotation ต่อ table → median bits
    - itq_b10k80           PCA-10 → ITQ rotation ต่อ table (30 iters, เรียนบน 50k)
    - agh_b10k80           anchor embedding (200 anchors, RBF) → random proj → median
  task-aware (ใช้ seeds ของ draw — จดว่าปิด reuse):
    - pu_diag_b10k80       ถ่วง features ด้วย |mean(seed)−mean(bg)| ก่อน hash
    - pu_guided_b10k80     แทน 1 bit/table ด้วยทิศ seed-contrast + noise

gate (เทียบ baseline): median PR-AUC +5% rel หรือ P@1% +3pp และชนะ ≥8/10 draws
รัน: uv run python outputs/phase2b_learned_geometry.py
"""
import time
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import polars as pl
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score

from leadx.encoder import FeatureEncoder
from leadx.projections import ProjectionBank
from leadx.hashing import HashEnsemble
from leadx.scoring import SeedScorer

RNG = np.random.default_rng(7)
N_DRAWS, FRACS, ALPHA = 10, [0.10, 0.25], 2.0
B, K = 10, 80
TEST_N = 20_000

# ---------- data (identical seeds/split กับ Phase 1) ----------
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
Ztr = enc.transform(df_tr).astype(np.float32)
Zte = enc.transform(df_te).astype(np.float32)
pos_idx = np.nonzero(ytr == 1)[0]
n_tr, d = Ztr.shape
print(f"train {n_tr:,} | test {len(yte):,} | dims {d}")

POWERS = np.uint64(1) << np.arange(B - 1, -1, -1, dtype=np.uint64)


def hash_median(Xtr, Xte, P_list):
    """P_list: list ของ (d', B) — median thresholds จาก train"""
    ctr = np.empty((len(Xtr), len(P_list)), np.uint64)
    cte = np.empty((len(Xte), len(P_list)), np.uint64)
    for k, P in enumerate(P_list):
        ptr, pte = Xtr @ P, Xte @ P
        t = np.median(ptr, axis=0)
        ctr[:, k] = ((ptr > t).astype(np.uint64) @ POWERS)
        cte[:, k] = ((pte > t).astype(np.uint64) @ POWERS)
    return ctr, cte


def metrics(scores):
    order = np.argsort(-scores)
    out = {"pr_auc": average_precision_score(yte, scores)}
    kk = max(1, int(0.01 * len(yte)))
    out["p_at_1pct"] = float(yte[order[:kk]].mean())
    out["recall_at_5pct"] = float(yte[order[:int(0.05*len(yte))]].sum() / yte.sum())
    return out


def pu_score(ctr, cte, seed_rows):
    sc = SeedScorer(mode="pu", score_type="lift", agg="log", alpha=ALPHA,
                    multi_probe=False).fit(ctr[seed_rows], np.ones(len(seed_rows)),
                                           background_codes=ctr)
    return sc.score(cte)


# ---------- unsupervised geometries (สร้างครั้งเดียว) ----------
UNSUP = {}
t0 = time.perf_counter()
bank = ProjectionBank(n_tables=K, n_bits=B, random_state=0).fit(Ztr, cat_mask=enc.cat_mask_)
h = HashEnsemble(bank.P_, bank.thresholds_)
UNSUP["baseline_b10k80"] = (h.transform(Ztr), h.transform(Zte))
print(f"baseline: {time.perf_counter()-t0:.0f}s")

t0 = time.perf_counter()
sub = RNG.permutation(n_tr)[:50_000]
pca64 = PCA(n_components=64, whiten=True, svd_solver="randomized",
            random_state=0).fit(Ztr[sub])
Wtr, Wte = pca64.transform(Ztr).astype(np.float32), pca64.transform(Zte).astype(np.float32)
rng = np.random.default_rng(1)
P_list = [np.linalg.qr(rng.standard_normal((64, B)))[0].astype(np.float32) for _ in range(K)]
UNSUP["pca_rr_b10k80"] = hash_median(Wtr, Wte, P_list)
print(f"pca_rr: {time.perf_counter()-t0:.0f}s")

t0 = time.perf_counter()
pca10 = PCA(n_components=B, whiten=True, svd_solver="randomized",
            random_state=0).fit(Ztr[sub])
Vtr, Vte = pca10.transform(Ztr).astype(np.float32), pca10.transform(Zte).astype(np.float32)
Vsub = Vtr[sub[:50_000]]
P_list = []
for k in range(K):
    R = np.linalg.qr(np.random.default_rng(k).standard_normal((B, B)))[0]
    for _ in range(30):                                     # ITQ iterations
        Bm = np.sign(Vsub @ R)
        U, _, Vt = np.linalg.svd(Bm.T @ Vsub, full_matrices=False)
        R = (U @ Vt).T
    P_list.append(R.astype(np.float32))
UNSUP["itq_b10k80"] = hash_median(Vtr, Vte, P_list)
print(f"itq: {time.perf_counter()-t0:.0f}s")

t0 = time.perf_counter()
km = MiniBatchKMeans(n_clusters=200, random_state=0, n_init=3).fit(Ztr[sub])
A = km.cluster_centers_.astype(np.float32)
def anchor_embed(X, chunk=20_000):
    out = np.empty((len(X), 200), np.float32)
    a_sq = (A ** 2).sum(1)
    sigma2 = None
    for s in range(0, len(X), chunk):
        d2 = ((X[s:s+chunk] ** 2).sum(1)[:, None] + a_sq[None, :]
              - 2 * X[s:s+chunk] @ A.T)
        if sigma2 is None:
            sigma2 = np.median(d2)
        out[s:s+chunk] = np.exp(-d2 / sigma2)
    return out
Etr, Ete = anchor_embed(Ztr), anchor_embed(Zte)
rng = np.random.default_rng(2)
P_list = [np.linalg.qr(rng.standard_normal((200, B)))[0].astype(np.float32) for _ in range(K)]
UNSUP["agh_b10k80"] = hash_median(Etr, Ete, P_list)
print(f"agh: {time.perf_counter()-t0:.0f}s")

# ---------- task-aware (ต่อ draw) ----------
def pu_diag(seed_rows, draw_seed):
    w = np.abs(Ztr[seed_rows].mean(0) - Ztr.mean(0)) + 1e-3
    w = (w / w.max()).astype(np.float32)
    rng = np.random.default_rng(draw_seed)
    P_list = [np.linalg.qr(rng.standard_normal((d, B)))[0].astype(np.float32)
              for _ in range(K)]
    return hash_median(Ztr * w, Zte * w, P_list)

def pu_guided(seed_rows, draw_seed):
    g = Ztr[seed_rows].mean(0) - Ztr.mean(0)
    g = (g / (np.linalg.norm(g) + 1e-12)).astype(np.float32)
    rng = np.random.default_rng(draw_seed)
    P_list = []
    for _ in range(K):
        P = np.linalg.qr(rng.standard_normal((d, B)))[0].astype(np.float32)
        noise = rng.standard_normal(d).astype(np.float32) * 0.3
        gd = g + noise
        P[:, 0] = gd / np.linalg.norm(gd)                   # bit แรก = ทิศ seed-contrast
        P_list.append(P)
    return hash_median(Ztr, Zte, P_list)

# ---------- run ----------
rows = []
for frac in FRACS:
    for draw in range(N_DRAWS):
        rd = np.random.default_rng(1000 * draw + int(frac * 100))
        seed_rows = rd.choice(pos_idx, size=int(frac * len(pos_idx)), replace=False)
        for name, (ctr, cte) in UNSUP.items():
            rows.append({"frac": frac, "draw": draw, "method": name,
                         **metrics(pu_score(ctr, cte, seed_rows))})
        for name, fn in [("pu_diag_b10k80", pu_diag), ("pu_guided_b10k80", pu_guided)]:
            ctr, cte = fn(seed_rows, draw)
            rows.append({"frac": frac, "draw": draw, "method": name,
                         **metrics(pu_score(ctr, cte, seed_rows))})
    print(f"frac {frac:.0%} done")

out = pl.DataFrame(rows)
out.write_csv("outputs/phase2b_raw.csv")

print("\n=== median ข้าม 10 draws ===")
agg = (out.group_by(["frac", "method"])
       .agg(pl.col("pr_auc").median().round(4), pl.col("p_at_1pct").median().round(3))
       .sort(["frac", "pr_auc"], descending=[False, True]))
with pl.Config(tbl_rows=40):
    print(agg)

print("\n=== gate vs baseline_b10k80 ===")
for frac in FRACS:
    sub = out.filter(pl.col("frac") == frac)
    base = sub.filter(pl.col("method") == "baseline_b10k80").sort("draw")
    for m in [x for x in sub["method"].unique() if x != "baseline_b10k80"]:
        cand = sub.filter(pl.col("method") == m).sort("draw")
        wins = int((cand["pr_auc"].to_numpy() > base["pr_auc"].to_numpy()).sum())
        rel = float((cand["pr_auc"].median() / base["pr_auc"].median() - 1) * 100)
        dpp = float((cand["p_at_1pct"].median() - base["p_at_1pct"].median()) * 100)
        mark = " ← ผ่าน gate" if (wins >= 8 and (rel >= 5 or dpp >= 3)) else ""
        print(f"frac {frac:.0%} {m:<18} {rel:+6.1f}% rel | P@1% {dpp:+5.1f}pp | ชนะ {wins}/10{mark}")
