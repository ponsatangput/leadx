"""ProjectionBank — คัดชุด projection vectors ตามเกณฑ์วัดได้ (spec §4)"""
import warnings

import numpy as np
from sklearn.metrics import normalized_mutual_info_score


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _info_gain(bits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """IG ของแต่ละ bit เทียบ y — bits: (n, C) bool, y: (n,) 0/1 → (C,)"""
    y = y.astype(bool)
    p1 = bits.mean(axis=0)
    n1 = np.maximum(bits.sum(axis=0), 1)
    n0 = np.maximum((~bits).sum(axis=0), 1)
    pos1 = (bits & y[:, None]).sum(axis=0) / n1
    pos0 = ((~bits) & y[:, None]).sum(axis=0) / n0
    return _binary_entropy(np.float64(y.mean())) - (
        p1 * _binary_entropy(pos1) + (1 - p1) * _binary_entropy(pos0)
    )


class ProjectionBank:
    """คัดชุด random-hyperplane projections คุณภาพดีตามเกณฑ์วัดได้ (spec §4)

    แก้ปัญหา "สุ่มหลายรอบกว่าจะได้ชุดดี" ด้วยการสุ่มเกิน → orthogonalize (QR) →
    คัดด้วย bit balance + redundancy บนข้อมูลจริง → resample ให้เองจนพอ. bits ถูก
    เรียงจากคุณภาพสูงสุดลงมา (จำเป็นต่อ prefix-masking ของ Tuner). supervised mode
    เพิ่มเกณฑ์ cross-fitted information gain (มี guard fallback เมื่อ seeds น้อย).

    Contract: input ต้อง centered/standardized แล้ว (ผ่าน FeatureEncoder) — ไม่งั้น
    เกณฑ์ balance จะ anti-select ทิศทางที่มี offset (มี warning เตือน).

    Parameters
    ----------
    n_tables, n_bits : int
        จำนวน hash tables และ bits ต่อ table
    strategy : {"unsupervised", "supervised"}
        supervised ใช้ label คัด bits (cross-fitted IG); fallback ถ้า seeds < min
    hash_mode : {"median", "origin"}
        "median" (default) — threshold = median(w·x): ทุก bit balanced โดยโครงสร้าง
        ไม่ต้องคัด/resample; "origin" — ตัดที่ 0 + balance-select machinery ดั้งเดิม
    cat_bit_frac : "auto" | float
        (median mode) สัดส่วน bits ที่ project เฉพาะ categorical dims ตาม cat_mask
        ที่ส่งเข้า fit — cat bit + median threshold = balanced bipartition ของ
        categories; 0 = ปิด; supervised จัดสรรเองผ่าน IG
    oversample : int
        สุ่ม candidates เกิน oversample × n_bits เท่า ก่อนคัด
    balance_min : float
        สัดส่วน minority ขั้นต่ำที่ bit จะผ่าน (default 0.40)
    redundancy_max : float
        |correlation| สูงสุดระหว่าง bits ที่ยอมรับ (default 0.30)
    sample_size : int
        ขนาด sample ที่ใช้วัดเกณฑ์ (ไม่ต้องใช้ทั้งชุด)
    min_seeds_per_class : int
        seeds ขั้นต่ำต่อ class สำหรับ supervised (ต่ำกว่านี้ fallback)
    random_state : int
    """

    def __init__(self, n_tables=10, n_bits=16, strategy="unsupervised",
                 hash_mode="median", cat_bit_frac="auto", oversample=10,
                 balance_min=0.40, redundancy_max=0.30, sample_size=100_000,
                 min_seeds_per_class=50, random_state=0):
        if strategy not in ("unsupervised", "supervised"):
            raise ValueError(f"strategy ไม่รู้จัก: {strategy}")
        if hash_mode not in ("median", "origin"):
            raise ValueError(f"hash_mode ไม่รู้จัก: {hash_mode}")
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.strategy = strategy
        self.hash_mode = hash_mode
        self.cat_bit_frac = cat_bit_frac
        self.oversample = oversample
        self.balance_min = balance_min
        self.redundancy_max = redundancy_max
        self.sample_size = sample_size
        self.min_seeds_per_class = min_seeds_per_class
        self.random_state = random_state

    def fit(self, Z: np.ndarray, y=None, cat_mask: np.ndarray | None = None) -> "ProjectionBank":
        rng = np.random.default_rng(self.random_state)
        Z = np.asarray(Z, dtype=np.float32)
        n = Z.shape[0]
        # type-aware bits: cat_mask ระบุ output columns ที่มาจาก categorical blocks
        # → จัดสรร bits แยกชนิด (hyperplane สำหรับ numeric, bipartition สำหรับ cat)
        self._cat_mask = None
        if cat_mask is not None and np.any(cat_mask) and self.cat_bit_frac != 0:
            if self.hash_mode == "origin":
                warnings.warn("cat_bit_frac ใช้ได้เฉพาะ hash_mode='median' — ข้าม")
            else:
                self._cat_mask = np.asarray(cat_mask, dtype=bool)
        if self.hash_mode == "origin":
            # Contract (origin mode): input ต้อง centered — hyperplane ผ่าน origin
            # ถ้า mean เบ้แรง เกณฑ์ balance จะคัดทิ้ง bits ที่ align กับทิศ offset
            # (median mode ไม่มีปัญหานี้ — threshold เลื่อนตามข้อมูลเอง)
            col_mean = Z.mean(axis=0)
            col_std = Z.std(axis=0) + 1e-12
            if np.abs(col_mean / col_std).max() > 0.5:
                warnings.warn(
                    "input ไม่ centered (|mean| > 0.5*std ในบาง column) — origin mode "
                    "คาดหวังข้อมูลที่ standardize แล้ว ไม่งั้นเกณฑ์ balance จะ anti-select "
                    "ทิศทางที่มี offset; พิจารณา hash_mode='median' (default)"
                )
        self.strategy_ = self.strategy
        if self.strategy == "supervised":
            counts = np.bincount(np.asarray(y).astype(int)) if y is not None else np.array([0])
            if y is None or len(counts) < 2 or counts.min() < self.min_seeds_per_class:
                warnings.warn(
                    f"supervised selection ต้องการ seeds >= {self.min_seeds_per_class}/class "
                    "— fallback เป็น unsupervised"
                )
                self.strategy_ = "unsupervised"
        idx = rng.permutation(n)[: min(n, self.sample_size)]
        S = Z[idx]
        y_s = np.asarray(y, dtype=np.float64)[idx] if self.strategy_ == "supervised" else None
        tables, thresholds, self._table_diag = [], [], []
        for _ in range(self.n_tables):
            if self.hash_mode == "median":
                C_k, t_k = self._fit_table_median(S, y_s, rng)
            else:
                C_k = self._fit_table(S, y_s, rng)
                t_k = np.zeros(self.n_bits, dtype=np.float32)
            tables.append(C_k)
            thresholds.append(t_k)
        self.P_ = np.stack(tables).astype(np.float32)          # (K, D, B)
        self.thresholds_ = np.stack(thresholds).astype(np.float32)  # (K, B)
        self._diversity = self._compute_diversity(S[:20_000])
        return self

    def _typed_candidates(self, d: int, rng) -> np.ndarray:
        """สร้าง candidates แยกชนิด: numeric bits project เฉพาะ numeric dims,
        cat bits เฉพาะ cat dims (บน one-hot = random bipartition ของ categories)
        interleave สองชนิดเพื่อให้ prefix masking ของ Tuner ยังเหลือทั้งคู่

        supervised: สร้าง oversample เท่า (ครึ่ง num ครึ่ง cat) ให้ IG ranking
        เป็นผู้จัดสรรชนิดตาม signal จริง"""
        cm = self._cat_mask
        n_cat_dims = int(cm.sum())
        if self.strategy_ == "supervised":
            n_total = self.oversample * self.n_bits
            n_cat = n_total // 2
            is_cat = np.zeros(n_total, dtype=bool)
            is_cat[:n_cat] = True
            rng.shuffle(is_cat)
        else:
            frac = (n_cat_dims / d) if self.cat_bit_frac == "auto" else float(self.cat_bit_frac)
            n_cat = int(round(self.n_bits * frac))
            if 0 < frac:
                n_cat = max(n_cat, 1)
            if frac < 1 and n_cat_dims < d:
                n_cat = min(n_cat, self.n_bits - 1)
            # interleave แบบ Bresenham — cat bits กระจายทั่วทุก prefix
            j = np.arange(self.n_bits)
            is_cat = ((j + 1) * n_cat) // self.n_bits - (j * n_cat) // self.n_bits > 0
        C = rng.standard_normal((d, len(is_cat)))
        C[np.ix_(cm, np.nonzero(~is_cat)[0])] = 0.0    # numeric bits: ปิด cat dims
        C[np.ix_(~cm, np.nonzero(is_cat)[0])] = 0.0    # cat bits: ปิด numeric dims
        C /= np.linalg.norm(C, axis=0, keepdims=True) + 1e-12
        return C

    def _fit_table_median(self, S, y_s, rng):
        """median-threshold: ทุก bit balanced 50/50 โดยโครงสร้าง — ไม่ต้องคัด balance/
        resample เลย. unsupervised = สุ่ม B ทิศ (QR); supervised = oversample แล้วจัด
        อันดับด้วย cross-fitted IG (เรียงมาก→น้อย เพื่อ prefix masking)"""
        d = S.shape[1]
        if self._cat_mask is not None:
            C = self._typed_candidates(d, rng)
        elif self.strategy_ == "supervised":
            C = self._candidates(d, rng)                     # oversample × n_bits
        else:
            Q, _ = np.linalg.qr(rng.standard_normal((d, min(d, self.n_bits))))
            C = Q[:, : self.n_bits]
            while C.shape[1] < self.n_bits:                  # กรณี n_bits > d
                Q2, _ = np.linalg.qr(rng.standard_normal((d, min(d, self.n_bits - C.shape[1]))))
                C = np.hstack([C, Q2[:, : self.n_bits - C.shape[1]]])
        proj = S @ C
        t = np.median(proj, axis=0)
        bits = proj > t
        if self.strategy_ == "supervised":
            perm = rng.permutation(len(S))
            h1, h2 = perm[: len(S) // 2], perm[len(S) // 2 :]
            quality = np.minimum(_info_gain(bits[h1], y_s[h1]), _info_gain(bits[h2], y_s[h2]))
            sel = np.argsort(-quality)[: self.n_bits]
            C, t, bits = C[:, sel], t[sel], bits[:, sel]
        minority = np.minimum(bits.mean(axis=0), 1.0 - bits.mean(axis=0))
        Bc = bits.astype(np.float32)
        Bc = (Bc - Bc.mean(axis=0)) / (Bc.std(axis=0) + np.float32(1e-7))
        corr_m = np.abs(Bc.T @ Bc) / len(S)
        np.fill_diagonal(corr_m, 0.0)
        self._table_diag.append({
            "balance": [float(b) for b in minority],
            "max_redundancy": float(corr_m.max()),
            "n_eff": self._n_eff(_pack_bits(bits)),
        })
        return C, t.astype(np.float32)

    def _candidates(self, d: int, rng) -> np.ndarray:
        """สุ่มเกิน oversample เท่า, QR orthogonalize เป็น block ละ <= d คอลัมน์"""
        n_cand = self.oversample * self.n_bits
        blocks, remaining = [], n_cand
        while remaining > 0:
            m = min(d, remaining)
            Q, _ = np.linalg.qr(rng.standard_normal((d, m)))
            blocks.append(Q[:, :m])
            remaining -= m
        return np.hstack(blocks)  # (d, n_cand) unit-norm columns

    def _fit_table(self, S, y_s, rng) -> np.ndarray:
        # สุ่มเพิ่มเป็นรอบๆ จนมี candidates ผ่านเกณฑ์ balance พอ (แก้ปัญหา "สุ่มหลายรอบแล้วลุ้น"
        # ของข้อมูล anisotropic — ระบบ resample ให้เองแทนผู้ใช้)
        # Memory bound: batch แรกเก็บเต็ม (เป็น pool ของ relax path) batch ถัดไปเก็บเฉพาะ
        # ตัวที่ผ่าน balance — เดิม pathological case (15 batches × n_bits สูง) สะสม
        # candidates ทั้งหมดจน matrix บวมระดับ GB
        C_parts, bits_parts, n_ok = [], [], 0
        for i in range(15):
            C_b = self._candidates(S.shape[1], rng)
            bits_b = (S @ C_b) > 0
            minority_b = np.minimum(bits_b.mean(axis=0), 1.0 - bits_b.mean(axis=0))
            ok_b = minority_b >= self.balance_min
            n_ok += int(ok_b.sum())
            if i == 0:
                C_parts.append(C_b)
                bits_parts.append(bits_b)
            else:
                C_parts.append(C_b[:, ok_b])
                bits_parts.append(bits_b[:, ok_b])
            if n_ok >= 3 * self.n_bits:
                break
        C = np.hstack(C_parts)
        bits = np.hstack(bits_parts)  # (n_s, n_cand)
        minority = np.minimum(bits.mean(axis=0), 1.0 - bits.mean(axis=0))
        ok = np.nonzero(minority >= self.balance_min)[0]
        if len(ok) < self.n_bits:
            warnings.warn(
                f"candidates ผ่านเกณฑ์ balance เพียง {len(ok)}/{self.n_bits} หลัง resample "
                "— ผ่อนเกณฑ์ balance สำหรับ bits ที่เหลือ (ข้อมูลอาจเบ้มาก ตรวจ diagnostics)"
            )
            ok = np.argsort(-minority)[: max(3 * self.n_bits, len(ok))]
        if self.strategy_ == "supervised":
            # cross-fitted IG: คัดจากครึ่งหนึ่งต้อง generalize ไปอีกครึ่ง → กัน noise bits
            perm = rng.permutation(len(S))
            h1, h2 = perm[: len(S) // 2], perm[len(S) // 2 :]
            quality = np.minimum(_info_gain(bits[h1], y_s[h1]), _info_gain(bits[h2], y_s[h2]))
        else:
            quality = minority
        order = ok[np.argsort(-quality[ok])]
        Bc = bits.astype(np.float32)  # float32 พอสำหรับ corr เกณฑ์ 0.3 — ลด memory ครึ่งหนึ่ง
        Bc = (Bc - Bc.mean(axis=0)) / (Bc.std(axis=0) + np.float32(1e-7))
        selected: list[int] = []
        for i in order:
            if len(selected) == self.n_bits:
                break
            if selected:
                corr = np.abs(Bc[:, i] @ Bc[:, selected]) / len(S)
                if corr.max() >= self.redundancy_max:
                    continue
            selected.append(int(i))
        if len(selected) < self.n_bits:
            warnings.warn("candidates ผ่านเกณฑ์ไม่พอ — ผ่อนเกณฑ์ redundancy สำหรับ bits ที่เหลือ")
            for i in order:
                if len(selected) == self.n_bits:
                    break
                if int(i) not in selected:
                    selected.append(int(i))
        # selected เรียงตาม quality มาก→น้อย (จำเป็นต่อ prefix masking ของ Tuner)
        sel_std = Bc[:, selected]
        corr_m = np.abs(sel_std.T @ sel_std) / len(S)
        np.fill_diagonal(corr_m, 0.0)
        self._table_diag.append({
            "balance": [float(minority[i]) for i in selected],
            "max_redundancy": float(corr_m.max()),
            "n_eff": self._n_eff(_pack_bits(bits[:, selected])),
        })
        return C[:, selected]

    @staticmethod
    def _n_eff(codes: np.ndarray) -> float:
        _, counts = np.unique(codes, return_counts=True)
        q = counts / counts.sum()
        return float(np.exp(-(q * np.log(q)).sum()))

    def _compute_diversity(self, S: np.ndarray) -> float:
        if self.n_tables < 2:
            return 0.0
        b = min(8, self.n_bits)  # ใช้ prefix 8 bits — NMI บน partition ละเอียดเกินจะ saturate
        parts = [
            _pack_bits((S @ self.P_[k, :, :b]) > self.thresholds_[k, :b])
            for k in range(self.n_tables)
        ]
        nmis = [
            normalized_mutual_info_score(parts[i], parts[j])
            for i in range(len(parts)) for j in range(i + 1, len(parts))
        ]
        return float(np.mean(nmis))

    def diagnostics(self) -> dict:
        return {
            "strategy": self.strategy_,
            "tables": self._table_diag,
            "diversity_nmi": self._diversity,
        }


def _pack_bits(bits: np.ndarray) -> np.ndarray:
    powers = np.uint64(1) << np.arange(bits.shape[1] - 1, -1, -1, dtype=np.uint64)
    return bits.astype(np.uint64) @ powers
