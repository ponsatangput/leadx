"""RPForest — Balanced Local Projection Forest (rpTree ensemble)

ทางเลือกของ ProjectionBank+HashEnsemble: แต่ละ tree แบ่งข้อมูล recursive ด้วย
*ทิศสุ่มใหม่ต่อ node + median ท้องถิ่น (tie-aware) ของจุดใน node* → ทุก leaf มี
สมาชิก ~n/2^depth เท่ากันโดยโครงสร้าง (equal-mass) — ฆ่า mega-bucket ที่ global
hyperplanes สร้างในย่านหนาแน่น (วัดจริง fraud: bucket เดียวกิน 6.3% ของประชากร
p99=2,509 แถว vs forest แคป 0.098% p99=199 — outputs/rpforest_raw.csv)

ผลวัด (PU b10 K80 + contrast + hier, median 3 draws): ชนะ bank ที่ fraud 0.3%
+39%, home-credit 0.3% +42%, natural 8% +6% (0.196→0.208 — best ปัจจุบัน);
ราคา fit+transform ~2-4.5x ของ single-matmul

Compatibility: bit ของ level l วางที่ MSB (depth-1-l) → leaf code มี prefix =
บรรพบุรุษใน tree — mask_codes / SeedScorer / hier_depths / Backbone ใช้ได้ทั้งชุด
(hier = chain ตาม tree path ตรงตัว) — interface เดียวกับ HashEnsemble.transform
"""
import numpy as np


def _node_threshold(proj: np.ndarray) -> float:
    """tie-aware balanced cut — ตรรกะเดียวกับ projections._balanced_thresholds
    แบบคอลัมน์เดียว (median ธรรมดาพังบน discrete atoms — bit ตายฝั่งเดียว)"""
    srt = np.sort(proj)
    cuts = np.nonzero(srt[:-1] < srt[1:])[0]
    if len(cuts) == 0:
        return float(srt[0])
    frac = (cuts + 1) / len(srt)
    best = cuts[int(np.argmin(np.abs(frac - 0.5)))]
    return float((srt[best] + srt[best + 1]) / 2.0)


class RPForest:
    """forest ของ balanced rpTrees — API ล้อ HashEnsemble (duck-type ใน facade)

    Parameters
    ----------
    n_tables : int
        จำนวน trees (ชื่อ n_tables เพื่อความสอดคล้องกับ bank/facade)
    n_bits : int
        ความลึกของ tree (= จำนวน bits ต่อ leaf code, <= 64)
    sample_size : int
        เพดานจำนวนแถวที่ใช้ *สร้าง* โครง tree (ทิศ+threshold) — โครงจาก sample
        2 แสนแถวเพียงพอทางสถิติ และกัน fit เป็น O(n log n) เต็มที่ scale ใหญ่
    random_state : int

    Attributes
    ----------
    dirs_ : (n_tables, 2^n_bits - 1, dim) float32 — ทิศต่อ node (BFS order)
    thr_ : (n_tables, 2^n_bits - 1) float32 — จุดตัดต่อ node
    """

    def __init__(self, n_tables=40, n_bits=10, sample_size=200_000,
                 random_state=0):
        if n_bits > 64:
            raise ValueError(f"n_bits สูงสุด 64 (ได้ {n_bits})")
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(self, Z: np.ndarray) -> "RPForest":
        """สร้างโครง forest — vectorized ต่อ level: สุ่มทิศทั้ง level เป็น batch,
        project ด้วย gather+einsum ครั้งเดียว, threshold ต่อ node จาก lexsort
        (per-node loop เหลือแค่ ops ถูกๆ บน slice ที่เรียงแล้ว — ไม่มี matvec/sort
        ต่อ node; จำเป็นสำหรับ tuner ที่ fit forest ต่อ fold)"""
        rng = np.random.default_rng(self.random_state)
        Z = np.ascontiguousarray(Z, dtype=np.float32)
        if Z.shape[0] > self.sample_size:
            Z = Z[rng.permutation(Z.shape[0])[: self.sample_size]]
        n, d = Z.shape
        n_nodes = 2 ** self.n_bits - 1
        self.dim = d
        self.dirs_ = np.empty((self.n_tables, n_nodes, d), dtype=np.float32)
        self.thr_ = np.zeros((self.n_tables, n_nodes), dtype=np.float32)
        for t in range(self.n_tables):
            node_of = np.zeros(n, dtype=np.int64)
            for lvl in range(self.n_bits):
                base, m = 2 ** lvl - 1, 2 ** lvl
                G = rng.standard_normal((m, d)).astype(np.float32)
                G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-12
                self.dirs_[t, base:base + m] = G
                proj = np.einsum("ij,ij->i", Z, G[node_of])
                order = np.lexsort((proj, node_of))
                sp = proj[order]
                bounds = np.searchsorted(node_of[order], np.arange(m + 1))
                thr_lvl = np.zeros(m, dtype=np.float32)
                for node in range(m):
                    lo, hi = bounds[node], bounds[node + 1]
                    if hi <= lo:
                        continue  # node ว่าง (tie บังคับ) — ทิศสุ่ม/thr 0 พอ
                    col = sp[lo:hi]
                    cuts = np.nonzero(col[:-1] < col[1:])[0]
                    if len(cuts) == 0:
                        thr_lvl[node] = col[0]
                        continue
                    frac = (cuts + 1) / (hi - lo)
                    best = cuts[int(np.argmin(np.abs(frac - 0.5)))]
                    thr_lvl[node] = (col[best] + col[best + 1]) / 2.0
                self.thr_[t, base:base + m] = thr_lvl
                node_of = 2 * node_of + (proj > thr_lvl[node_of])
        return self

    def prefix(self, n_bits: int, n_tables: int) -> "RPForest":
        """forest ย่อยที่ depth/trees น้อยลง — nodes ของ b levels แรกคือ slice
        `dirs_[:k, :2^b-1]` ตรงๆ (BFS order) → deploy prefix ที่ tuner ประเมิน
        ได้โดยไม่ refit (สิ่งที่ deploy = สิ่งที่วัด)"""
        if n_bits > self.n_bits or n_tables > self.n_tables:
            raise ValueError(
                f"prefix ({n_bits}, {n_tables}) เกิน forest ({self.n_bits}, {self.n_tables})"
            )
        n_nodes = 2 ** n_bits - 1
        return RPForest.from_arrays(
            np.ascontiguousarray(self.dirs_[:n_tables, :n_nodes]),
            np.ascontiguousarray(self.thr_[:n_tables, :n_nodes]),
            sample_size=self.sample_size, random_state=self.random_state,
        )

    def transform(self, Z: np.ndarray, chunk_size: int = 1_000_000) -> np.ndarray:
        """traverse ทุก tree level-by-level → leaf codes (n, n_tables) uint64"""
        Z = np.ascontiguousarray(Z, dtype=np.float32)
        n = Z.shape[0]
        codes = np.empty((n, self.n_tables), dtype=np.uint64)
        for start in range(0, n, chunk_size):
            chunk = Z[start:start + chunk_size]
            m = chunk.shape[0]
            for t in range(self.n_tables):
                node_of = np.zeros(m, dtype=np.int64)
                c = np.zeros(m, dtype=np.uint64)
                for lvl in range(self.n_bits):
                    base = 2 ** lvl - 1
                    G = self.dirs_[t, base + node_of]
                    proj = np.einsum("ij,ij->i", chunk, G)
                    bit = proj > self.thr_[t, base + node_of]
                    c |= bit.astype(np.uint64) << np.uint64(self.n_bits - 1 - lvl)
                    node_of = 2 * node_of + bit
                codes[start:start + m, t] = c
        return codes

    def to_arrays(self) -> dict:
        return {"dirs": self.dirs_, "thr": self.thr_}

    @classmethod
    def from_arrays(cls, dirs: np.ndarray, thr: np.ndarray,
                    sample_size=200_000, random_state=0) -> "RPForest":
        n_tables, n_nodes, d = dirs.shape
        n_bits = int(np.log2(n_nodes + 1))
        f = cls(n_tables=n_tables, n_bits=n_bits, sample_size=sample_size,
                random_state=random_state)
        f.dirs_ = np.asarray(dirs, dtype=np.float32)
        f.thr_ = np.asarray(thr, dtype=np.float32)
        f.dim = d
        return f
