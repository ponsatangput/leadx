"""HashEnsemble — Z @ P → sign → bit-pack เป็นรหัส bucket uint64 (K tables)"""
import numpy as np


class HashEnsemble:
    """เข้ารหัสข้อมูลเป็น bucket codes: (Z @ P > t) → bit-pack เป็น uint64 (K tables)

    bit ลำดับ j (คุณภาพอันดับ j) วางที่ตำแหน่ง MSB เพื่อให้ `mask_codes` ตัด bits
    ท้ายออกได้ partition ที่หยาบลงแบบ nested โดยไม่ต้อง re-hash (ใช้โดย Tuner).

    Parameters
    ----------
    P : numpy.ndarray, shape (K, D, B)
        projection tensor — K tables, D dims, B bits ต่อ table (B <= 64)
    thresholds : numpy.ndarray (K, B) | None
        จุดตัดต่อ bit — LSH ไม่จำเป็นต้องผ่าน origin; median-threshold ให้ bit
        balanced โดยโครงสร้าง. None = ตัดที่ 0 (พฤติกรรมเดิม)
    """

    def __init__(self, P: np.ndarray, thresholds: np.ndarray | None = None):
        self.P = np.asarray(P, dtype=np.float32)  # (K, D, B)
        if self.P.ndim != 3:
            raise ValueError(f"P ต้องเป็น (K, D, B) ได้ {self.P.shape}")
        self.n_tables, self.dim, self.n_bits = self.P.shape
        if self.n_bits > 64:
            raise ValueError(f"n_bits สูงสุด 64 (ได้ {self.n_bits})")
        if thresholds is None:
            thresholds = np.zeros((self.n_tables, self.n_bits), dtype=np.float32)
        self.thresholds = np.asarray(thresholds, dtype=np.float32)
        if self.thresholds.shape != (self.n_tables, self.n_bits):
            raise ValueError(
                f"thresholds ต้องเป็น (K, B) = {(self.n_tables, self.n_bits)} "
                f"(ได้ {self.thresholds.shape})"
            )
        # bit j (quality อันดับ j) → MSB เพื่อรองรับ prefix masking
        self._powers = np.uint64(1) << np.arange(self.n_bits - 1, -1, -1, dtype=np.uint64)

    def transform(self, Z: np.ndarray, chunk_size: int = 1_000_000) -> np.ndarray:
        Z = np.ascontiguousarray(Z, dtype=np.float32)
        n = Z.shape[0]
        codes = np.empty((n, self.n_tables), dtype=np.uint64)
        for start in range(0, n, chunk_size):
            chunk = Z[start : start + chunk_size]
            for k in range(self.n_tables):
                bits = (chunk @ self.P[k]) > self.thresholds[k]
                codes[start : start + chunk.shape[0], k] = bits.astype(np.uint64) @ self._powers
        return codes


def mask_codes(codes: np.ndarray, n_bits_keep: int, n_bits_total: int) -> np.ndarray:
    """เก็บ b bits แรก (คุณภาพดีสุด) — partition หยาบลงโดยไม่ต้อง re-hash"""
    return codes >> np.uint64(n_bits_total - n_bits_keep)
