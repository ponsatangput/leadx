"""Margin-aware multi-probe — วัดก่อน ship (audit P1 ตัวสุดท้าย)

สมมติฐาน: ตอน fallback ควร flip เฉพาะ bits ที่ margin |z - threshold| แคบสุด
(bits ที่ "เกือบพลิก") แทนการ flip ทุก bit — neighbor ที่ได้ควรคล้าย query
กว่า → คะแนน fallback แม่นกว่า โดยเฉพาะโซน coverage-cliff (bits สูง, seeds น้อย)

เงื่อนไข: none (prior fallback) / all (Hamming-1 ทุก bit — ปัจจุบัน) /
m2, m4 (margin-aware T bits) / m4p (T=4 + pair flip ของ 2 bits แคบสุด)

รัน: uv run python outputs/margin_multiprobe_experiment.py  (จาก dir leadx)
"""
import time
import warnings

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from leadx.encoder import FeatureEncoder
from leadx.hashing import HashEnsemble, mask_codes
from leadx.projections import ProjectionBank
from leadx.scoring import SeedScorer

warnings.filterwarnings("ignore")
DATA = "/Users/ponpiboonsatangput/jarvis/data"
K, B_MAX = 40, 20


def margin_score(scorer, codes, Zq, P, thr, b, T, pair=False):
    """เหมือน SeedScorer.score แต่ fallback ใช้ margin-aware probe (T bits แคบสุด)

    codes = masked codes ที่ b bits; projection j อยู่ที่ bit position b-1-j
    """
    n, n_tables = codes.shape
    per = np.empty((n, n_tables))
    for k in range(n_tables):
        col = codes[:, k]
        joined = (
            pl.DataFrame({"code": col}).with_row_index("_i")
            .join(scorer.tables_[k], on="code", how="left").sort("_i")
        )
        s = joined["score"].to_numpy().astype(np.float64).copy()
        miss = np.nonzero(joined["score"].is_null().to_numpy())[0]
        still = miss
        if len(miss):
            marg = np.abs(Zq[miss] @ P[k][:, :b] - thr[k, :b])
            order = np.argsort(marg, axis=1)[:, :T].astype(np.uint64)
            flips = np.uint64(1) << (np.uint64(b - 1) - order)
            neigh = col[miss][:, None] ^ flips
            if pair and T >= 2:
                pmask = flips[:, 0] | flips[:, 1]
                neigh = np.concatenate([neigh, (col[miss] ^ pmask)[:, None]], axis=1)
            m, t = neigh.shape
            agg = (
                pl.DataFrame({"i": np.repeat(np.arange(m, dtype=np.uint32), t),
                              "code": neigh.ravel()})
                .join(scorer.tables_[k], on="code", how="inner").group_by("i")
                .agg(s=(pl.col("score") * pl.col("n_total")).sum()
                       / pl.col("n_total").sum())
            )
            out = np.full(m, np.nan)
            out[agg["i"].to_numpy()] = agg["s"].to_numpy()
            found = ~np.isnan(out)
            s[miss[found]] = out[found]
            still = miss[~found]
        s[still] = scorer.prior_
        per[:, k] = s
    w = np.full(n_tables, 1.0 / n_tables)
    if scorer.score_type == "lift" and scorer.agg == "log":
        return np.exp(np.log(per) @ w)
    return per @ w


def run_conditions(bank, c_seed, c_bg, c_te, Zte_w, y_te, b, mode, metric):
    cs = mask_codes(c_seed, b, B_MAX)
    ct = mask_codes(c_te, b, B_MAX)
    out = {}
    for cond in ["none", "all", "m2", "m4", "m4p"]:
        scorer = SeedScorer(mode=mode, score_type="lift", alpha=2.0, agg="log",
                            n_bits=b, multi_probe=(cond == "all"))
        if mode == "pu":
            scorer.fit(cs, np.ones(len(cs)),
                       background_codes=mask_codes(c_bg, b, B_MAX))
        else:
            scorer.fit(cs, c_bg)  # c_bg = y_seed สำหรับ two_class
        if cond in ("none", "all"):
            s = scorer.score(ct)
            out["miss"] = scorer.diagnostics_["miss_rate_join"]
        else:
            T = int(cond[1])
            s = margin_score(scorer, ct, Zte_w, bank.P_, bank.thresholds_, b,
                             T=T, pair=cond.endswith("p"))
        out[cond] = metric(y_te, s)
    return out


def home_credit_pu(draws=3, bits=(10, 14, 20)):
    rng = np.random.default_rng(7)
    df = pl.read_csv(f"{DATA}/kaggle/home-credit/application_train.csv",
                     infer_schema_length=5000, ignore_errors=True)
    y = df["TARGET"].cast(pl.Float64).to_numpy()
    df = df.drop(["SK_ID_CURR", "TARGET"])
    idx = rng.permutation(df.height)
    n_te = int(df.height * 0.2)
    te = idx[:n_te][rng.permutation(n_te)[:20_000]]
    tr = idx[n_te:]
    df_tr, df_te, y_te = df[tr.tolist()], df[te.tolist()], y[te]
    enc = FeatureEncoder().fit(df_tr)               # parity: fit จาก population
    Z_bg, Z_te = enc.transform(df_tr), enc.transform(df_te)
    pos_idx = np.nonzero(y[tr] == 1)[0]
    rows = []
    for d in range(draws):
        rd = np.random.default_rng(1000 * d + 10)
        seeds = rd.choice(pos_idx, size=int(0.10 * len(pos_idx)), replace=False)
        w = np.abs(Z_bg[seeds].mean(axis=0) - Z_bg.mean(axis=0)) + 1e-3
        w = (w / w.max()).astype(np.float32)
        Zbg_w, Zte_w, Zs_w = Z_bg * w, Z_te * w, Z_bg[seeds] * w
        bank = ProjectionBank(n_tables=K, n_bits=B_MAX, strategy="unsupervised",
                              hash_mode="median", cat_bit_frac=0.0,
                              random_state=0).fit(Zbg_w)
        h = HashEnsemble(bank.P_, bank.thresholds_)
        c_seed, c_bg, c_te = h.transform(Zs_w), h.transform(Zbg_w), h.transform(Zte_w)
        for b in bits:
            r = run_conditions(bank, c_seed, c_bg, c_te, Zte_w, y_te, b, "pu",
                               average_precision_score)
            rows.append({"dataset": "home-credit-pu", "b": b, "draw": d, **r})
            print(f"  hc b={b} d={d}: miss={r['miss']:.3f} " +
                  " ".join(f"{c}={r[c]:.4f}" for c in
                           ["none", "all", "m2", "m4", "m4p"]))
    return rows


def adult_two_class(draws=3, bits=(14, 20), n_seed_per_class=30):
    rng = np.random.default_rng(7)
    df = pl.read_csv(f"{DATA}/benchmarks/adult.csv", infer_schema_length=5000,
                     ignore_errors=True)
    y = (df["income"].str.contains(">50K")).cast(pl.Float64).to_numpy()
    df = df.drop("income")
    idx = rng.permutation(df.height)
    n_te = int(df.height * 0.2)
    te, tr = idx[:n_te], idx[n_te:]
    df_tr, y_tr, df_te, y_te = df[tr.tolist()], y[tr], df[te.tolist()], y[te]
    enc = FeatureEncoder().fit(df_tr)
    Z_tr, Z_te = enc.transform(df_tr), enc.transform(df_te)
    rows = []
    for d in range(draws):
        rd = np.random.default_rng(1000 * d + 10)
        seeds = np.concatenate([
            rd.choice(np.nonzero(y_tr == c)[0], size=n_seed_per_class,
                      replace=False) for c in (0.0, 1.0)])
        bank = ProjectionBank(n_tables=K, n_bits=B_MAX, strategy="unsupervised",
                              hash_mode="median", cat_bit_frac="auto",
                              random_state=0).fit(Z_tr, cat_mask=enc.cat_mask_)
        h = HashEnsemble(bank.P_, bank.thresholds_)
        c_seed, c_te = h.transform(Z_tr[seeds]), h.transform(Z_te)
        for b in bits:
            r = run_conditions(bank, c_seed, y_tr[seeds], c_te, Z_te, y_te, b,
                               "two_class", roc_auc_score)
            rows.append({"dataset": "adult-2c-30seed", "b": b, "draw": d, **r})
            print(f"  adult b={b} d={d}: miss={r['miss']:.3f} " +
                  " ".join(f"{c}={r[c]:.4f}" for c in
                           ["none", "all", "m2", "m4", "m4p"]))
    return rows


if __name__ == "__main__":
    t0 = time.time()
    rows = home_credit_pu() + adult_two_class()
    raw = pl.DataFrame(rows)
    raw.write_csv("outputs/margin_multiprobe_raw.csv")
    med = (raw.group_by(["dataset", "b"])
           .agg([pl.col(c).median() for c in
                 ["miss", "none", "all", "m2", "m4", "m4p"]])
           .sort(["dataset", "b"]))
    print("\n=== median across draws ===")
    print(med)
    print(f"\ntotal {time.time() - t0:.0f}s — raw → outputs/margin_multiprobe_raw.csv")
