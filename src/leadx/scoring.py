"""SeedScorer — ตาราง bucket→score (Polars group-by/join) + multi-probe fallback (spec §6)"""
import numpy as np
import polars as pl


def pu_score_range(scorer: "SeedScorer", *code_arrays) -> tuple[float, float]:
    """คำนวณ (lo, span) ของ raw PU score จาก codes ตอน fit

    ใช้เป็นค่าคงที่ normalize ตอน predict — ห้าม normalize ด้วย min/max ของ batch
    ที่กำลัง score เพราะจะทำให้แถวเดียวกันได้ score ต่างกันตาม batch (batch-dependent)
    """
    parts = [scorer.score(c) for c in code_arrays if c is not None and len(c)]
    s = np.concatenate(parts)
    lo = float(s.min())
    span = float(s.max()) - lo
    return lo, (span if span > 0 else 1.0)


def normalize_pu(s: np.ndarray, lo: float, span: float) -> np.ndarray:
    """normalize PU score ด้วยค่าคงที่จาก fit → [0, 1] (clip ค่าที่เกินช่วง train)"""
    return np.clip((s - lo) / span, 0.0, 1.0)


class SeedScorer:
    """ให้คะแนน bucket จาก seeds แล้ว join กลับให้ข้อมูลใหม่ (Polars) (spec §6)

    เก็บเฉพาะสถิติต่อ bucket (n_pos, n_total) ไม่เก็บรายชื่อสมาชิก → เล็กมาก.
    two_class ใช้ m-estimate smoothing `(n_pos + m·π)/(n_total + m)`; pu ใช้
    seed-rate (มี background) หรือ log-density (ไม่มี). แถวที่ตก bucket ไม่มี seed
    → multi-probe (Hamming-1 neighbors) → prior fallback. รวมข้าม tables ด้วย mean
    หรือ coverage-weighted.

    score_type="lift" (imbalance-invariant): score ต่อ bucket = smoothed lift
    `(n_pos + α) / (E[n_pos|neutral] + α)` — เทียบความเข้มข้นของ positives กับ
    reference population (two_class: seeds ทั้งหมด, pu: background ที่ตรึงตอน fit)
    class prior ไม่อยู่ในสูตร → ranking ไม่สะเทือนตามความ imbalance. fallback = 1
    (เป็นกลาง). กล่องหลักฐานน้อยถูก shrink เข้า 1 (กัน outlier กล่องจิ๋ว lift พุ่ง)

    Parameters
    ----------
    mode : {"two_class", "pu"}
    m : float
        smoothing strength (m-estimate) — ใช้กับ score_type="prop"
    weights : {"mean", "coverage"}
        วิธีถ่วงน้ำหนักข้าม tables
    multi_probe : bool
        เปิด Hamming-1 fallback สำหรับแถวที่ join ไม่เจอ (ต้องระบุ n_bits)
    n_bits : int | None
        จำนวน bits (จำเป็นเมื่อ multi_probe=True — ใช้กำหนดตำแหน่ง flip)
    score_type : {"prop", "lift"}
        "prop" — สัดส่วน smoothed (พฤติกรรมเดิม); "lift" — density ratio เทียบ reference
    alpha : float
        shrink-toward-1 strength ของ lift (ใช้กับ score_type="lift")
    agg : {"mean", "log"}
        รวม lift ข้าม tables แบบ arithmetic หรือ geometric (log-domain, NB-style)
        — ใช้กับ score_type="lift" เท่านั้น
    """

    def __init__(self, mode="two_class", m=10.0, weights="mean",
                 multi_probe=True, n_bits=None,
                 score_type="prop", alpha=2.0, agg="mean"):
        if mode not in ("two_class", "pu"):
            raise ValueError(f"mode ไม่รู้จัก: {mode}")
        if weights not in ("mean", "coverage"):
            raise ValueError(f"weights ไม่รู้จัก: {weights}")
        if score_type not in ("prop", "lift"):
            raise ValueError(f"score_type ไม่รู้จัก: {score_type}")
        if agg not in ("mean", "log"):
            raise ValueError(f"agg ไม่รู้จัก: {agg}")
        if score_type == "lift" and alpha <= 0:
            raise ValueError("lift ต้องการ alpha > 0 (กัน log(0) และกล่องหลักฐานว่าง)")
        self.mode = mode
        self.m = m
        self.weights = weights
        self.multi_probe = multi_probe
        self.n_bits = n_bits
        self.score_type = score_type
        self.alpha = alpha
        self.agg = agg

    def fit(self, codes: np.ndarray, y: np.ndarray, background_codes=None) -> "SeedScorer":
        codes = np.asarray(codes, dtype=np.uint64)
        y = np.asarray(y, dtype=np.float64)
        self.n_tables_ = codes.shape[1]
        self.tables_ = []
        if self.mode == "two_class":
            if not set(np.unique(y)) <= {0.0, 1.0}:
                raise ValueError("two_class ต้องการ y เป็น 0/1")
            n_pos_total, n_total_all = float(y.sum()), float(len(y))
            self.prior_ = 1.0 if self.score_type == "lift" else float(y.mean())
            for k in range(self.n_tables_):
                g = (
                    pl.DataFrame({"code": codes[:, k], "y": y})
                    .group_by("code")
                    .agg(n_pos=pl.col("y").sum(), n_total=pl.len().cast(pl.Float64))
                )
                if self.score_type == "lift":
                    # e(b) = จำนวน positive ที่คาดใน bucket ถ้าเป็นกลาง; shrink เข้า 1
                    g = g.with_columns(
                        score=(pl.col("n_pos") + self.alpha)
                        / (n_pos_total * pl.col("n_total") / n_total_all + self.alpha)
                    )
                else:
                    g = g.with_columns(
                        score=(pl.col("n_pos") + self.m * self.prior_)
                        / (pl.col("n_total") + self.m)
                    )
                self.tables_.append(g.select("code", "score", "n_total"))
        elif background_codes is not None:
            # PU + background (reference ตรึงตอน fit): prop = smoothed seed rate,
            # lift = smoothed density ratio เทียบ background
            background_codes = np.asarray(background_codes, dtype=np.uint64)
            n_seed_total = float(codes.shape[0])
            n_bg_total = float(background_codes.shape[0])
            rho = n_seed_total / n_bg_total
            self.prior_ = 1.0 if self.score_type == "lift" else float(rho)
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
                )
                if self.score_type == "lift":
                    g = g.with_columns(
                        score=(pl.col("n_seed") + self.alpha)
                        / (n_seed_total * pl.col("n_total") / n_bg_total + self.alpha)
                    )
                else:
                    g = g.with_columns(
                        score=(pl.col("n_seed") + self.m * rho) / (pl.col("n_total") + self.m)
                    )
                self.tables_.append(g.select("code", "score", "n_total"))
        else:
            # PU ไม่มี background: log-density (ranking score เท่านั้น — facade จะ normalize)
            if self.score_type == "lift":
                raise ValueError(
                    "score_type='lift' ใน pu mode ต้องการ background (reference population "
                    "เป็นตัวหารของ density ratio) — ส่ง background_codes"
                )
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
            w = hit / hit.sum()
        else:
            w = np.full(self.n_tables_, 1.0 / self.n_tables_)
        if self.score_type == "lift" and self.agg == "log":
            # geometric mean = คูณ likelihood ratio ข้าม tables (NB-style); lift > 0 เสมอ
            # เพราะ alpha > 0 และ fallback = 1
            return np.exp(np.log(per_table) @ w)
        return per_table @ w

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
