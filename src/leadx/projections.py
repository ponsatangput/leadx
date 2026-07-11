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
                 oversample=10, balance_min=0.40, redundancy_max=0.30,
                 sample_size=100_000, min_seeds_per_class=50, random_state=0):
        if strategy not in ("unsupervised", "supervised"):
            raise ValueError(f"strategy ไม่รู้จัก: {strategy}")
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.strategy = strategy
        self.oversample = oversample
        self.balance_min = balance_min
        self.redundancy_max = redundancy_max
        self.sample_size = sample_size
        self.min_seeds_per_class = min_seeds_per_class
        self.random_state = random_state

    def fit(self, Z: np.ndarray, y=None) -> "ProjectionBank":
        rng = np.random.default_rng(self.random_state)
        Z = np.asarray(Z, dtype=np.float32)
        n = Z.shape[0]
        # Contract: input ต้อง centered (FeatureEncoder ทำให้แล้ว) — hyperplane ผ่าน origin
        # ถ้า mean เบ้แรง เกณฑ์ balance จะคัดทิ้ง bits ที่ align กับทิศ offset อย่างเป็นระบบ
        col_mean = Z.mean(axis=0)
        col_std = Z.std(axis=0) + 1e-12
        if np.abs(col_mean / col_std).max() > 0.5:
            warnings.warn(
                "input ไม่ centered (|mean| > 0.5*std ในบาง column) — ProjectionBank คาดหวัง "
                "ข้อมูลที่ standardize แล้ว (ผ่าน FeatureEncoder) ไม่งั้นเกณฑ์ balance จะ "
                "anti-select ทิศทางที่มี offset และ score จะเสียหายรุนแรง"
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
        tables, self._table_diag = [], []
        for _ in range(self.n_tables):
            tables.append(self._fit_table(S, y_s, rng))
        self.P_ = np.stack(tables).astype(np.float32)  # (K, D, B)
        self._diversity = self._compute_diversity(S[:20_000])
        return self

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
        C_parts, bits_parts, n_ok = [], [], 0
        for _ in range(15):
            C_b = self._candidates(S.shape[1], rng)
            bits_b = (S @ C_b) > 0
            minority_b = np.minimum(bits_b.mean(axis=0), 1.0 - bits_b.mean(axis=0))
            C_parts.append(C_b)
            bits_parts.append(bits_b)
            n_ok += int((minority_b >= self.balance_min).sum())
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
        Bc = bits.astype(np.float64)
        Bc = (Bc - Bc.mean(axis=0)) / (Bc.std(axis=0) + 1e-12)
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
        parts = [_pack_bits((S @ self.P_[k, :, :b]) > 0) for k in range(self.n_tables)]
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
