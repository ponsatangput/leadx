"""Mutual-information feature weighting — attack noise dilution at the source (v0.2-F).

`feature_weighting="contrast"` scores a feature by |mean(seeds) − mean(background)|:
a **first-moment** statistic. It cannot see a feature whose seeds differ in shape
rather than location (bimodal amounts, variance-only shifts, non-monotone
patterns) — common in fraud.

Mutual information between a feature and "is this row a confirmed seed" captures
**any** distributional difference, needs no negatives (PU-native), and — unlike
XGBoost importance — measures marginal relevance, which is what LeadX's linear
projections can actually exploit. LeadX already uses cross-fitted information
gain, but at the *bit* level; once noise is mixed into a projection no bit
selection can unmix it, so the feature level is the gap.

Conditions (numeric features only, so the weighting effect is not diluted by
categorical tables):
  none        : no weighting (baseline)
  contrast    : |mean(seeds) − mean(bg)|            (current library default)
  mi          : cross-fitted MI(feature-bin ; is-seed)
  mi_mrmr     : mi with a soft redundancy penalty w/(1+λ·r), r = w-weighted
                mean |corr| with the other features (fraud feature sets are
                highly redundant: count_1h/6h/24h are one concept, and random
                projections count each copy separately)

Every condition is also run with **+60 injected pure-noise features** — the
direct test of dilution resistance.

PRE-REGISTERED before the full run:
  data     : IEEE fraud, sorted by TransactionDT, numeric columns only
  setting  : temporal (train 250k rows -> next 70k), 3 rolling folds, PU with
             50% of the train window's frauds confirmed, 2 draws, 1:1,000
  primary  : Recall@0.5%; also PR-AUC, Recall@0.1%
  analysis : paired per (fold, draw, noise setting), median + wins
  gate     : `mi` beats `contrast` on Recall@0.5% (paired median > 0, >=4/6 wins)
             in the +noise setting AND does not regress >5% without noise.
             `mi_mrmr` is reported but judged separately.

Run: uv run python outputs/mi_weighting_experiment.py [--smoke]
"""
from __future__ import annotations

import gc
import math
import time
import warnings

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from leadx.encoder import FeatureEncoder
from leadx.rpforest import RPForest
from leadx.scoring import SeedScorer
from temporal_fraud_benchmark import NUM_COLS, downsample_positives, load

warnings.filterwarnings("ignore")
N_TABLES, N_BITS = 40, 12
REVIEW = (0.001, 0.005)


# ----------------------------------------------------------------- weights ---
def _mi_per_feature(Z: np.ndarray, mask: np.ndarray, edges: list) -> np.ndarray:
    """MI(feature-bin ; mask) ต่อ feature — nats"""
    n, D = Z.shape
    p1 = float(mask.mean())
    p0 = 1.0 - p1
    out = np.zeros(D)
    if p1 <= 0 or p1 >= 1:
        return out
    for j in range(D):
        b = np.searchsorted(edges[j], Z[:, j])
        nb = len(edges[j]) + 1
        c_all = np.bincount(b, minlength=nb).astype(np.float64)
        c_1 = np.bincount(b[mask], minlength=nb).astype(np.float64)
        c_0 = c_all - c_1
        pb, pb1, pb0 = c_all / n, c_1 / n, c_0 / n
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(pb1 > 0, pb1 * np.log(pb1 / np.maximum(pb * p1, 1e-300)), 0.0)
            t0 = np.where(pb0 > 0, pb0 * np.log(pb0 / np.maximum(pb * p0, 1e-300)), 0.0)
        out[j] = float(np.sum(t1) + np.sum(t0))
    return out


def mi_weights(Z: np.ndarray, seed_rows: np.ndarray, n_bins=8, rng=None,
               redundancy_lambda: float | None = None) -> np.ndarray:
    """cross-fitted MI weights; ครึ่ง seeds สองชุด แล้วเอา min (กัน MI พองจาก
    seeds น้อย/bin เยอะ) — optional soft mRMR redundancy penalty"""
    rng = rng or np.random.default_rng(0)
    n, D = Z.shape
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = [np.unique(np.quantile(Z[:, j], qs)) for j in range(D)]
    halves = np.array_split(rng.permutation(np.asarray(seed_rows)), 2)
    est = []
    for h in halves:
        m = np.zeros(n, dtype=bool)
        m[h] = True
        est.append(_mi_per_feature(Z, m, edges))
    w = np.maximum(np.minimum(est[0], est[1]), 0.0)      # conservative estimate
    if redundancy_lambda:
        idx = rng.permutation(n)[:20_000]
        C = np.abs(np.corrcoef(Z[idx].T))
        np.fill_diagonal(C, 0.0)
        denom = max(w.sum(), 1e-12)
        r = (C @ w) / denom                               # w-weighted redundancy
        w = w / (1.0 + redundancy_lambda * r)
    w = w / (w.max() + 1e-12) + 1e-3
    return w.astype(np.float32)


def contrast_weights(Z: np.ndarray, seed_rows: np.ndarray) -> np.ndarray:
    w = np.abs(Z[seed_rows].mean(axis=0) - Z.mean(axis=0)) + 1e-3
    return (w / w.max()).astype(np.float32)


# ----------------------------------------------------------------- scoring ---
def metrics(y, s) -> dict:
    y = np.asarray(y, np.int8)
    s = np.asarray(s, np.float64)
    out = {"pr_auc": float(average_precision_score(y, s))}
    order = np.argsort(-s)
    for f in REVIEW:
        k = max(1, int(math.ceil(len(y) * f)))
        tag = str(f * 100).rstrip("0").rstrip(".").replace(".", "_")
        out[f"recall_at_{tag}pct"] = float(y[order[:k]].sum()) / max(1.0, y.sum())
    return out


def leadx_score(Ztr, Zte, seed_rows, w, seed):
    Ztr_w = Ztr if w is None else Ztr * w
    Zte_w = Zte if w is None else Zte * w
    f = RPForest(N_TABLES, N_BITS, random_state=seed).fit(Ztr_w)
    ctr, cte = f.transform(Ztr_w), f.transform(Zte_w)
    sc = SeedScorer(mode="pu", score_type="lift", agg="log", n_bits=N_BITS,
                    multi_probe=False)
    sc.fit(ctr[seed_rows], np.ones(len(seed_rows)), background_codes=ctr)
    return sc.score(cte)


def run(train_len=250_000, test_len=70_000, stride=100_000, n_folds=3,
        ratio=1e-3, seed_frac=0.5, draws=2, noise_settings=(0, 60)):
    df = load()
    y_all = df["isFraud"].to_numpy().astype(np.int8)
    keep = downsample_positives(y_all, ratio, np.random.default_rng(4242))
    sub, ysub = df[keep.tolist()], y_all[keep]
    rows = []
    for fold in range(n_folds):
        a = fold * stride
        b, c = a + train_len, a + train_len + test_len
        if c > len(sub):
            continue
        Xtr = sub[np.arange(a, b).tolist()].select(NUM_COLS)
        Xte = sub[np.arange(b, c).tolist()].select(NUM_COLS)
        ytr, yte = ysub[a:b], ysub[b:c]
        pos_tr = np.flatnonzero(ytr == 1)
        if len(pos_tr) < 40 or yte.sum() < 3:
            continue
        enc = FeatureEncoder().fit(Xtr)
        Z0_tr, Z0_te = enc.transform(Xtr), enc.transform(Xte)
        for n_noise in noise_settings:
            if n_noise:
                rn = np.random.default_rng(555 + fold)
                Ztr = np.hstack([Z0_tr, rn.normal(size=(len(Z0_tr), n_noise)).astype(np.float32)])
                Zte = np.hstack([Z0_te, rn.normal(size=(len(Z0_te), n_noise)).astype(np.float32)])
            else:
                Ztr, Zte = Z0_tr, Z0_te
            for d in range(draws):
                rng = np.random.default_rng(1000 * fold + 100 * d + 7)
                seeds = rng.choice(pos_tr, max(20, int(seed_frac * len(pos_tr))),
                                   replace=False)
                wmap = {
                    "none": None,
                    "contrast": contrast_weights(Ztr, seeds),
                    "mi": mi_weights(Ztr, seeds, rng=np.random.default_rng(d)),
                    "mi_mrmr": mi_weights(Ztr, seeds, rng=np.random.default_rng(d),
                                          redundancy_lambda=1.0),
                }
                for name, w in wmap.items():
                    s = leadx_score(Ztr, Zte, seeds, w, seed=fold)
                    m = metrics(yte, s)
                    rows.append({"fold": fold, "draw": d, "n_noise": n_noise,
                                 "weighting": name, "n_dims": Ztr.shape[1], **m})
                    print(f"  f{fold}d{d} noise={n_noise:3d} {name:9s} "
                          f"pr={m['pr_auc']:.4f} r@0.5%={m['recall_at_0_5pct']:.3f}",
                          flush=True)
                gc.collect()
            del Ztr, Zte
            gc.collect()
    return rows


def summarize(raw: pl.DataFrame) -> None:
    for n_noise in sorted(raw["n_noise"].unique().to_list()):
        r = raw.filter(pl.col("n_noise") == n_noise)
        print(f"\n=== +{n_noise} noise features (dims={r['n_dims'][0]}) ===")
        for metric in ("pr_auc", "recall_at_0_5pct", "recall_at_0_1pct"):
            med = {w: float(np.median(r.filter(pl.col("weighting") == w)[metric].to_numpy()))
                   for w in ("none", "contrast", "mi", "mi_mrmr")}
            print(f"  {metric:18s} " + "  ".join(f"{k}={v:.4f}" for k, v in med.items()))
        base = {(d["fold"], d["draw"]): d for d in
                r.filter(pl.col("weighting") == "contrast").to_dicts()}
        for cond in ("mi", "mi_mrmr", "none"):
            A = {(d["fold"], d["draw"]): d for d in
                 r.filter(pl.col("weighting") == cond).to_dicts()}
            common = sorted(set(A) & set(base))
            line = f"  paired {cond:8s} vs contrast:"
            for metric in ("pr_auc", "recall_at_0_5pct", "recall_at_0_1pct"):
                dl = [(A[k][metric] - base[k][metric]) / max(1e-12, base[k][metric]) * 100
                      for k in common]
                if dl:
                    tag = metric.replace("recall_at_", "R@").replace("pr_auc", "PR")
                    line += f"  {tag} {float(np.median(dl)):+.1f}%({sum(x > 0 for x in dl)}/{len(dl)})"
            print(line)


if __name__ == "__main__":
    import sys
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    rows = run(n_folds=1, draws=1) if smoke else run()
    raw = pl.DataFrame(rows)
    out = ("outputs/mi_weighting_smoke_raw.csv" if smoke
           else "outputs/mi_weighting_raw.csv")
    raw.write_csv(out)
    summarize(raw)
    print(f"\n{out}  |  {(time.time() - t0) / 60:.1f} min")
