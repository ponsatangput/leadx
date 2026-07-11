"""SeedScorer — ตาราง bucket→score (Polars group-by/join) + multi-probe fallback (spec §6)"""
import numpy as np
import polars as pl


class SeedScorer:
    """ให้คะแนน bucket จาก seeds แล้ว join กลับให้ข้อมูลใหม่ (Polars) (spec §6)

    เก็บเฉพาะสถิติต่อ bucket (n_pos, n_total) ไม่เก็บรายชื่อสมาชิก → เล็กมาก.
    two_class ใช้ m-estimate smoothing `(n_pos + m·π)/(n_total + m)`; pu ใช้
    seed-rate (มี background) หรือ log-density (ไม่มี). แถวที่ตก bucket ไม่มี seed
    → multi-probe (Hamming-1 neighbors) → prior fallback. รวมข้าม tables ด้วย mean
    หรือ coverage-weighted.

    Parameters
    ----------
    mode : {"two_class", "pu"}
    m : float
        smoothing strength (m-estimate)
    weights : {"mean", "coverage"}
        วิธีรวม score ข้าม tables
    multi_probe : bool
        เปิด Hamming-1 fallback สำหรับแถวที่ join ไม่เจอ (ต้องระบุ n_bits)
    n_bits : int | None
        จำนวน bits (จำเป็นเมื่อ multi_probe=True — ใช้กำหนดตำแหน่ง flip)
    """

    def __init__(self, mode="two_class", m=10.0, weights="mean",
                 multi_probe=True, n_bits=None):
        if mode not in ("two_class", "pu"):
            raise ValueError(f"mode ไม่รู้จัก: {mode}")
        if weights not in ("mean", "coverage"):
            raise ValueError(f"weights ไม่รู้จัก: {weights}")
        self.mode = mode
        self.m = m
        self.weights = weights
        self.multi_probe = multi_probe
        self.n_bits = n_bits

    def fit(self, codes: np.ndarray, y: np.ndarray, background_codes=None) -> "SeedScorer":
        codes = np.asarray(codes, dtype=np.uint64)
        y = np.asarray(y, dtype=np.float64)
        self.n_tables_ = codes.shape[1]
        self.tables_ = []
        if self.mode == "two_class":
            if not set(np.unique(y)) <= {0.0, 1.0}:
                raise ValueError("two_class ต้องการ y เป็น 0/1")
            self.prior_ = float(y.mean())
            for k in range(self.n_tables_):
                g = (
                    pl.DataFrame({"code": codes[:, k], "y": y})
                    .group_by("code")
                    .agg(n_pos=pl.col("y").sum(), n_total=pl.len().cast(pl.Float64))
                    .with_columns(
                        score=(pl.col("n_pos") + self.m * self.prior_)
                        / (pl.col("n_total") + self.m)
                    )
                )
                self.tables_.append(g.select("code", "score", "n_total"))
        elif background_codes is not None:
            # PU + background: smoothed seed rate ต่อ bucket (spec §6)
            background_codes = np.asarray(background_codes, dtype=np.uint64)
            rho = codes.shape[0] / background_codes.shape[0]
            self.prior_ = float(rho)
            for k in range(self.n_tables_):
                seed_g = (
                    pl.DataFrame({"code": codes[:, k]})
                    .group_by("code").agg(n_seed=pl.len().cast(pl.Float64))
                )
                bg_g = (
                    pl.DataFrame({"code": background_codes[:, k]})
                    .group_by("code").agg(n_total=pl.len().cast(pl.Float64))
                )
                g = (
                    bg_g.join(seed_g, on="code", how="full", coalesce=True)
                    # fill เฉพาะ count columns — fill_null ทั้ง frame จะ cast `code` u64→f64 แล้ว join พัง
                    .with_columns(
                        pl.col("n_seed").fill_null(0.0),
                        pl.col("n_total").fill_null(0.0),
                    )
                    .with_columns(
                        score=(pl.col("n_seed") + self.m * rho) / (pl.col("n_total") + self.m)
                    )
                )
                self.tables_.append(g.select("code", "score", "n_total"))
        else:
            # PU ไม่มี background: log-density (ranking score เท่านั้น — facade จะ normalize)
            self.prior_ = 0.0
            for k in range(self.n_tables_):
                g = (
                    pl.DataFrame({"code": codes[:, k]})
                    .group_by("code")
                    .agg(n_total=pl.len().cast(pl.Float64))
                    .with_columns(score=(pl.col("n_total") + 1.0).log())
                )
                self.tables_.append(g.select("code", "score", "n_total"))
        return self

    def score(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes, dtype=np.uint64)
        n = codes.shape[0]
        per_table = np.empty((n, self.n_tables_))
        hit = np.empty(self.n_tables_)
        n_final_miss = 0
        for k in range(self.n_tables_):
            s, h, fm = self._score_table(codes[:, k], self.tables_[k])
            per_table[:, k], hit[k] = s, h
            n_final_miss += fm
        self.diagnostics_ = {
            "miss_rate_join": float(1.0 - hit.mean()),
            "miss_rate_final": n_final_miss / (n * self.n_tables_),
        }
        if self.weights == "coverage" and hit.sum() > 0:
            return per_table @ (hit / hit.sum())
        return per_table.mean(axis=1)

    def _score_table(self, col: np.ndarray, tbl: pl.DataFrame):
        joined = (
            pl.DataFrame({"code": col})
            .with_row_index("_i")
            .join(tbl, on="code", how="left")
            .sort("_i")  # join ไม่การันตีลำดับแถว — ต้อง sort กลับ
        )
        score = joined["score"].to_numpy().astype(np.float64).copy()
        miss_idx = np.nonzero(joined["score"].is_null().to_numpy())[0]
        hit_frac = 1.0 - len(miss_idx) / len(col)
        still = miss_idx
        if len(miss_idx) and self.multi_probe:
            if self.n_bits is None:
                raise ValueError("multi_probe=True ต้องระบุ n_bits")
            probe, found = self._multi_probe(col[miss_idx], tbl)
            score[miss_idx[found]] = probe[found]
            still = miss_idx[~found]
        score[still] = self.prior_
        return score, hit_frac, len(still)

    def _multi_probe(self, miss_codes: np.ndarray, tbl: pl.DataFrame):
        """รหัส Hamming-1 (flip ทีละ bit ใน b bits ล่าง) → รวม stats ถ่วงน้ำหนัก n_total"""
        b = self.n_bits
        m = len(miss_codes)
        flips = np.uint64(1) << np.arange(b, dtype=np.uint64)
        neigh = (miss_codes[:, None] ^ flips[None, :]).ravel()
        df = pl.DataFrame({
            "i": np.repeat(np.arange(m, dtype=np.uint32), b),
            "code": neigh,
        })
        agg = (
            df.join(tbl, on="code", how="inner")
            .group_by("i")
            .agg(s=(pl.col("score") * pl.col("n_total")).sum() / pl.col("n_total").sum())
        )
        out = np.full(m, np.nan)
        out[agg["i"].to_numpy()] = agg["s"].to_numpy()
        found = ~np.isnan(out)
        return out, found
