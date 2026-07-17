"""Real-data validation: telco / adult / home-credit — leadx vs HistGB

หมายเหตุ: paths เป็น local ของเครื่องที่รัน — แก้ให้ชี้ dataset ของคุณเอง
(telco: Kaggle WA_Fn-UseC_ Telco Churn, adult: UCI census income,
home-credit: Kaggle Home Credit application_train)
ผลสรุปบันทึกใน DESIGN_NOTES §5.5"""
import time
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

from leadx import LeadX

RNG = np.random.default_rng(0)
GRID = {"n_bits": [10, 14, 18], "n_tables": [8, 15], "m": [10.0],
        "weights": ["mean"], "strategy": ["unsupervised", "supervised"],
        "score_type": ["prop", "lift"], "alpha": [1.0, 5.0], "agg": ["mean", "log"],
        "cat_bit_frac": [0.0, "auto"]}


def load_telco():
    df = pl.read_csv("/Users/ponpiboonsatangput/jarvis/data/benchmarks/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    y = (df["Churn"] == "Yes").cast(pl.Float64).to_numpy()
    df = df.drop(["customerID", "Churn"]).with_columns(
        pl.col("TotalCharges").cast(pl.Float64, strict=False))
    return df, y


def load_adult():
    df = pl.read_csv("/Users/ponpiboonsatangput/jarvis/data/benchmarks/adult.csv")
    y = df["income"].str.contains(">50K").cast(pl.Float64).to_numpy()
    return df.drop("income"), y


def load_homecredit():
    df = pl.read_csv("/Users/ponpiboonsatangput/jarvis/data/kaggle/home-credit/application_train.csv",
                     infer_schema_length=5000, ignore_errors=True)
    y = df["TARGET"].cast(pl.Float64).to_numpy()
    return df.drop(["SK_ID_CURR", "TARGET"]), y


def split(df, y, test_frac=0.2):
    n = df.height
    idx = RNG.permutation(n)
    n_te = int(n * test_frac)
    te, tr = idx[:n_te], idx[n_te:]
    return df[tr.tolist()], y[tr], df[te.tolist()], y[te]


def histgb_xy(df_tr, df_te):
    cat_cols = [c for c, d in zip(df_tr.columns, df_tr.dtypes) if not d.is_numeric()]
    num_cols = [c for c in df_tr.columns if c not in cat_cols]
    Xn_tr = df_tr[num_cols].cast(pl.Float64).to_numpy()
    Xn_te = df_te[num_cols].cast(pl.Float64).to_numpy()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                             encoded_missing_value=-1)
        Xc_tr = enc.fit_transform(df_tr[cat_cols].fill_null("__na__").to_numpy())
        Xc_te = enc.transform(df_te[cat_cols].fill_null("__na__").to_numpy())
        X_tr = np.hstack([Xn_tr, Xc_tr]); X_te = np.hstack([Xn_te, Xc_te])
        mask = np.zeros(X_tr.shape[1], bool); mask[len(num_cols):] = True
    else:
        X_tr, X_te, mask = Xn_tr, Xn_te, None
    return X_tr, X_te, mask


def bench(name, loader):
    df, y = loader()
    df_tr, ytr, df_te, yte = split(df, y)
    print(f"\n=== {name}: train {df_tr.height:,} / test {df_te.height:,} | pos {ytr.mean():.1%} ===")

    t0 = time.perf_counter()
    clf = LeadX(mode="two_class", tune=True, grid=GRID, random_state=0).fit(df_tr, ytr)
    t_lx = time.perf_counter() - t0
    s = clf.score_samples(df_te)
    a1, p1 = roc_auc_score(yte, s), average_precision_score(yte, s)
    cfg = clf.config_
    print(f"leadx(tuned)      AUC {a1:.3f}  PR-AUC {p1:.3f}  fit {t_lx:5.1f}s  "
          f"cfg: {cfg['score_type']}/{cfg['agg']} b={cfg['n_bits']} k={cfg['n_tables']} "
          f"strat={cfg['strategy'][:5]} catf={cfg['cat_bit_frac']}")

    t0 = time.perf_counter()
    clf2 = LeadX(mode="two_class", tune=True, grid=GRID, feature_weighting="corr",
                 random_state=0).fit(df_tr, ytr)
    t2 = time.perf_counter() - t0
    s2 = clf2.score_samples(df_te)
    print(f"leadx(+weighting) AUC {roc_auc_score(yte, s2):.3f}  PR-AUC {average_precision_score(yte, s2):.3f}  fit {t2:5.1f}s")

    X_tr, X_te, mask = histgb_xy(df_tr, df_te)
    t0 = time.perf_counter()
    gb = HistGradientBoostingClassifier(random_state=0, categorical_features=mask).fit(X_tr, ytr)
    t_gb = time.perf_counter() - t0
    pg = gb.predict_proba(X_te)[:, 1]
    print(f"HistGB            AUC {roc_auc_score(yte, pg):.3f}  PR-AUC {average_precision_score(yte, pg):.3f}  fit {t_gb:5.1f}s")
    return df_tr, ytr, df_te, yte


for name, loader in [("telco", load_telco), ("adult", load_adult)]:
    bench(name, loader)

# ---------- home credit: two-class + PU sim + rarity sweep ----------
df_tr, ytr, df_te, yte = bench("home-credit", load_homecredit)

print("\n--- PU simulation (home-credit): เห็น positive แค่ 30% ไม่มี negative ---")
pos_idx = np.nonzero(ytr == 1)[0]
seed_idx = RNG.choice(pos_idx, size=int(0.3 * len(pos_idx)), replace=False)
df_seeds = df_tr[seed_idx.tolist()]
t0 = time.perf_counter()
pu = LeadX(mode="pu", tune=False, n_bits=12, n_tables=15, score_type="lift",
           agg="log", alpha=2.0, random_state=0).fit(df_seeds, background=df_tr)
s_pu = pu.score_samples(df_te)
t_pu = time.perf_counter() - t0
ap = average_precision_score(yte, s_pu)
k = int(0.1 * len(yte))
top = np.argsort(-s_pu)[:k]
rec10 = yte[top].sum() / yte.sum()
print(f"leadx PU ({len(seed_idx):,} seeds): AUC {roc_auc_score(yte, s_pu):.3f}  "
      f"PR-AUC {ap:.3f} (baseline {yte.mean():.3f})  recall@top10% {rec10:.1%}  "
      f"({t_pu:.0f}s) — HistGB รันไม่ได้ (ไม่มี negative)")

print("\n--- rarity sweep (home-credit จริง): ลด positive ลง — leadx-lift vs HistGB ---")
neg_idx = np.nonzero(ytr == 0)[0]
print(f"{'pos rate':>9}{'n_pos':>8}{'leadx-lift AUC':>16}{'HistGB AUC':>12}")
for frac in [1.0, 0.12, 0.04]:  # 8% → ~1% → ~0.33%
    n_keep = int(len(pos_idx) * frac)
    keep = RNG.choice(pos_idx, size=n_keep, replace=False)
    sub = np.concatenate([neg_idx, keep])
    d_sub, y_sub = df_tr[sub.tolist()], ytr[sub]
    lx = LeadX(mode="two_class", tune=False, n_bits=12, n_tables=15,
               score_type="lift", agg="log", alpha=2.0, random_state=0).fit(d_sub, y_sub)
    a_lx = roc_auc_score(yte, lx.score_samples(df_te))
    Xs, Xt, mask = histgb_xy(d_sub, df_te)
    gb = HistGradientBoostingClassifier(random_state=0, categorical_features=mask).fit(Xs, y_sub)
    a_gb = roc_auc_score(yte, gb.predict_proba(Xt)[:, 1])
    print(f"{y_sub.mean():>9.2%}{n_keep:>8,}{a_lx:>16.3f}{a_gb:>12.3f}")
