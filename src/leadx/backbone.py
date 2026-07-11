"""LeadXBackbone — hash ครั้งเดียว (unsupervised) แล้ว reuse กับหลายโจทย์ (scenario A)

จุดแข็งเชิงสถาปัตยกรรมของ leadx: projections เป็น y-independent → encode + project +
hash ครั้งเดียว (แพง) แล้วแต่ละโจทย์ใหม่ (y ต่างกัน) เหลือแค่ SeedScorer group-by บน
codes ที่ cache ไว้ (~ถูกมาก) — โมเดลต้นทุนที่ boosting เลียนแบบไม่ได้ (tree splits ขึ้นกับ y)

    bb = LeadXBackbone(n_bits=16, n_tables=15).fit(X)   # แพงครั้งเดียว
    churn = bb.fit_task(y_churn)                         # ~ถูก
    fraud = bb.fit_task(y_fraud)
    churn.predict_proba(X_new)
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

from .encoder import FeatureEncoder
from .hashing import HashEnsemble, mask_codes
from .projections import ProjectionBank
from .scoring import SeedScorer, normalize_pu, pu_score_range


class LeadXBackbone:
    """Reusable hashing backbone — projections + cached seed codes (unsupervised)

    Parameters
    ----------
    n_bits, n_tables : int
        ขนาด hash สูงสุดของ backbone; แต่ละ task ใช้ b<=n_bits (prefix mask) และ
        k<=n_tables (subset) ได้โดยไม่ต้อง re-hash
    random_state : int
    """

    def __init__(self, n_bits=16, n_tables=15, random_state=0, chunk_size=1_000_000):
        self.n_bits = n_bits
        self.n_tables = n_tables
        self.random_state = random_state
        self.chunk_size = chunk_size

    def fit(self, df: pl.DataFrame) -> "LeadXBackbone":
        """เข้ารหัส + สุ่ม/คัด projections (unsupervised) + cache seed codes"""
        self.encoder_ = FeatureEncoder().fit(df)
        Z = self.encoder_.transform(df)
        bank = ProjectionBank(
            n_tables=self.n_tables, n_bits=self.n_bits, strategy="unsupervised",
            random_state=self.random_state,
        ).fit(Z)
        self.hasher_ = HashEnsemble(bank.P_)
        self.diagnostics_ = bank.diagnostics()
        self._seed_codes = self.hasher_.transform(Z, chunk_size=self.chunk_size)
        return self

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        """hash rows ใหม่ผ่าน projections เดิม → codes (n × n_tables) uint64"""
        return self.hasher_.transform(self.encoder_.transform(df), chunk_size=self.chunk_size)

    def _resolve(self, n_bits, n_tables):
        b = self.n_bits if n_bits is None else n_bits
        k = self.n_tables if n_tables is None else n_tables
        if b > self.n_bits:
            raise ValueError(f"n_bits ({b}) เกิน backbone ({self.n_bits})")
        if k > self.n_tables:
            raise ValueError(f"n_tables ({k}) เกิน backbone ({self.n_tables})")
        return b, k

    def fit_task(self, y=None, mode="two_class", m=10.0, weights="mean",
                 n_bits=None, n_tables=None, seeds: pl.DataFrame | None = None,
                 background: pl.DataFrame | None = None) -> "TaskModel":
        """สร้างโมเดลต่อโจทย์จาก codes ที่ cache ไว้ (ถูกมาก — แค่ group-by)

        Parameters
        ----------
        y : array-like of {0,1} | None
            label; two_class ต้องส่ง (เรียงตาม rows ที่ fit หรือ seeds), pu ไม่ต้อง
        mode : {"two_class", "pu"}
        m, weights : ดู SeedScorer
        n_bits, n_tables : int | None
            ปรับ granularity ต่อโจทย์ (b<=backbone via prefix mask, k<=backbone via subset)
        seeds : polars.DataFrame | None
            ถ้า seed อยู่นอก rows ที่ fit backbone (เช่น positive แยกใน pu) — จะ hash ให้
            ถ้า None ใช้ cached codes ของ rows ที่ fit
        background : polars.DataFrame | None
            pu: ประชากร unlabeled เป็น negative proxy
        """
        b, k = self._resolve(n_bits, n_tables)
        if seeds is None:
            seed_codes = self._seed_codes
        else:
            seed_codes = self.transform(seeds)
        seed_codes = mask_codes(seed_codes, b, self.n_bits)[:, :k]
        n = seed_codes.shape[0]
        if mode == "two_class" and y is None:
            raise ValueError("two_class ต้องการ y")
        y = np.ones(n) if (mode == "pu" and y is None) else np.asarray(y, dtype=np.float64)
        if len(y) != n:
            raise ValueError(
                f"ความยาว y ({len(y)}) ไม่ตรงกับจำนวนแถวของ seeds ({n}) — "
                "ถ้า seeds=None y ต้องเรียงตาม rows ที่ใช้ fit backbone"
            )
        if mode == "two_class" and len(np.unique(y)) < 2:
            raise ValueError(
                "two_class ต้องมี y ครบทั้งสอง class — ถ้ามีแต่ positive ใช้ mode='pu'"
            )
        bg_codes = None
        if background is not None:
            bg_codes = mask_codes(self.transform(background), b, self.n_bits)[:, :k]
        scorer = SeedScorer(mode=mode, m=m, weights=weights, n_bits=b).fit(
            seed_codes, y, background_codes=bg_codes)
        # PU calibration จาก fit (batch-invariant) — ดู scoring.pu_score_range
        pu_lo, pu_span = (
            pu_score_range(scorer, seed_codes, bg_codes) if mode == "pu" else (None, None)
        )
        return TaskModel(self, scorer, b, k, mode, pu_lo, pu_span)

    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "projections.npy", self.hasher_.P)
        np.save(path / "seed_codes.npy", self._seed_codes)
        meta = {
            "n_bits": self.n_bits, "n_tables": self.n_tables,
            "random_state": self.random_state, "chunk_size": self.chunk_size,
            "encoder": self.encoder_.to_dict(), "polars_version": pl.__version__,
        }
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path) -> "LeadXBackbone":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        bb = cls(n_bits=meta["n_bits"], n_tables=meta["n_tables"],
                 random_state=meta["random_state"], chunk_size=meta["chunk_size"])
        bb.encoder_ = FeatureEncoder.from_dict(meta["encoder"])
        bb.hasher_ = HashEnsemble(np.load(path / "projections.npy"))
        bb._seed_codes = np.load(path / "seed_codes.npy")
        bb.diagnostics_ = {"note": "loaded — projection diagnostics ไม่ถูกบันทึก"}
        return bb


class TaskModel:
    """โมเดลต่อโจทย์ที่ผูกกับ backbone — hash rows ใหม่ผ่าน projections ร่วม แล้ว score"""

    def __init__(self, backbone: LeadXBackbone, scorer: SeedScorer,
                 n_bits: int, n_tables: int, mode: str,
                 pu_lo: float | None = None, pu_span: float | None = None):
        self.backbone = backbone
        self.scorer_ = scorer
        self.n_bits = n_bits
        self.n_tables = n_tables
        self.mode = mode
        self._pu_lo = pu_lo
        self._pu_span = pu_span

    def _codes(self, df: pl.DataFrame) -> np.ndarray:
        codes = self.backbone.transform(df)
        return mask_codes(codes, self.n_bits, self.backbone.n_bits)[:, :self.n_tables]

    def score_samples(self, df: pl.DataFrame) -> np.ndarray:
        s = self.scorer_.score(self._codes(df))
        self.diagnostics_ = self.scorer_.diagnostics_
        if self.mode == "pu":
            s = normalize_pu(s, self._pu_lo, self._pu_span)
        return s

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        if self.mode != "two_class":
            raise ValueError("predict_proba ใช้ได้เฉพาะ two_class — PU ใช้ score_samples")
        p = self.score_samples(df)
        return np.column_stack([1.0 - p, p])
