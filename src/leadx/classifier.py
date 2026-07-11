"""LeadX — facade ต่อท่อ encoder → projections → hashing → scoring (+tuning)

ไม่มี logic ของตัวเอง — ทุก logic อยู่ใน component และทดสอบที่นั่น (spec §2)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from .encoder import FeatureEncoder
from .hashing import HashEnsemble
from .projections import ProjectionBank
from .scoring import SeedScorer
from .tuning import Tuner


class LeadX:
    """Fast LSH ensemble classifier — seed-based bucket scoring บนข้อมูลขนาดใหญ่

    เข้ารหัสข้อมูลด้วย random hyperplane (SimHash) เป็นรหัส bucket, ให้คะแนนแต่ละ
    bucket จากสัดส่วน seeds ที่ตกลงไป, แล้วเฉลี่ยข้าม K hash tables — fit/predict
    เป็น batch operation ล้วน scale ได้ถึงหลักสิบล้านแถว

    Parameters
    ----------
    mode : {"two_class", "pu"}
        "two_class" — seeds มี label 0/1, `predict_proba` คืน probability-like
        "pu" — positive-unlabeled, seeds เป็น positive ล้วน, `score_samples` คืน
        ranking score (ไม่ใช่ probability)
    tune : bool
        True → หา (strategy, n_bits, n_tables, m, weights) ที่ดีที่สุดจาก out-of-fold
        validation; False → ใช้ค่าที่กำหนดผ่าน parameters ด้านล่างตรงๆ
    n_bits : int
        จำนวน bits ต่อ hash table (ขนาด bucket code) — ใช้เมื่อ tune=False
    n_tables : int
        จำนวน hash tables ใน ensemble — ใช้เมื่อ tune=False
    m : float
        m-estimate smoothing strength — ยิ่งสูง bucket ที่ seed น้อยยิ่งถูกดึงเข้า prior
    strategy : {"unsupervised", "supervised"}
        วิธีคัด projection vectors; supervised ใช้ label ช่วย (cross-fitted IG)
    weights : {"mean", "coverage"}
        วิธีรวม score ข้าม tables; "coverage" ให้ table ที่ miss เยอะพูดเบาลง
    grid : dict | None
        override grid สำหรับ auto-tune (คีย์: n_bits, n_tables, m, weights, strategy)
    chunk_size : int
        ขนาด chunk ตอน predict — คุม peak memory บนข้อมูลใหญ่
    random_state : int
        seed สำหรับการสุ่ม projections และ fold

    Attributes
    ----------
    config_ : dict
        config ที่ใช้จริงหลัง fit
    tuning_results_ : polars.DataFrame | None
        ทุก config เรียงตาม OOF score (None ถ้า tune=False)
    diagnostics_ : dict
        projection quality, config, และ scoring miss rate (หลัง predict/score)

    Examples
    --------
    >>> clf = LeadX(mode="two_class")
    >>> clf.fit(X_seed, y)              # doctest: +SKIP
    >>> proba = clf.predict_proba(X)    # doctest: +SKIP
    """

    def __init__(self, mode="two_class", tune=True, n_bits=16, n_tables=10, m=10.0,
                 strategy="unsupervised", weights="mean", grid=None,
                 chunk_size=1_000_000, random_state=0):
        if mode not in ("two_class", "pu"):
            raise ValueError(f"mode ไม่รู้จัก: {mode}")
        self.mode = mode
        self.tune = tune
        self.n_bits = n_bits
        self.n_tables = n_tables
        self.m = m
        self.strategy = strategy
        self.weights = weights
        self.grid = grid
        self.chunk_size = chunk_size
        self.random_state = random_state

    def fit(self, df: pl.DataFrame, y=None, background: pl.DataFrame | None = None):
        """Train บน seeds (ข้อมูลที่มี label)

        Parameters
        ----------
        df : polars.DataFrame
            seed features (numeric และ/หรือ categorical); ต้องมี >= 20 แถว
        y : array-like of {0, 1} | None
            label ต่อแถว; two_class ต้องส่ง, pu ไม่ต้อง (ถือว่า positive ทั้งหมด)
        background : polars.DataFrame | None
            เฉพาะ pu mode — ประชากร unlabeled ใช้เป็น negative proxy ให้ score
            เป็น seed-rate ต่อ bucket (ดีกว่า log-density); auto-tune ใน pu ต้องมี

        Returns
        -------
        self
        """
        n = df.height
        if n < 20:
            raise ValueError(
                f"ต้องการ seeds อย่างน้อย 20 แถว (ได้ {n}) — น้อยกว่านี้ score จะเป็น noise"
            )
        if self.mode == "two_class" and y is None:
            raise ValueError("two_class ต้องการ y")
        y = np.ones(n) if (self.mode == "pu" and y is None) else np.asarray(y, dtype=np.float64)
        self.encoder_ = FeatureEncoder().fit(df)
        Z = self.encoder_.transform(df)
        Z_bg = self.encoder_.transform(background) if background is not None else None
        if self.tune:
            tuner = Tuner(grid=self.grid, mode=self.mode, random_state=self.random_state)
            best = tuner.tune(Z, y, Z_background=Z_bg)
            self.tuning_results_ = tuner.results_
            cfg = {k: best[k] for k in ("strategy", "n_bits", "n_tables", "m", "weights")}
        else:
            self.tuning_results_ = None
            cfg = {"strategy": self.strategy, "n_bits": self.n_bits,
                   "n_tables": self.n_tables, "m": self.m, "weights": self.weights}
        self.config_ = cfg
        bank = ProjectionBank(
            n_tables=cfg["n_tables"], n_bits=cfg["n_bits"], strategy=cfg["strategy"],
            random_state=self.random_state,
        )
        bank.fit(Z, y if self.mode == "two_class" else None)
        self.hasher_ = HashEnsemble(bank.P_)
        codes = self.hasher_.transform(Z)
        bg_codes = self.hasher_.transform(Z_bg) if Z_bg is not None else None
        self.scorer_ = SeedScorer(
            mode=self.mode, m=cfg["m"], weights=cfg["weights"], n_bits=cfg["n_bits"],
        ).fit(codes, y, background_codes=bg_codes)
        self.diagnostics_ = {"projections": bank.diagnostics(), "config": cfg}
        return self

    def score_samples(self, df: pl.DataFrame) -> np.ndarray:
        """คืน score ต่อแถว (ใช้ได้ทั้งสอง mode)

        two_class → smoothed P(positive) ต่อ bucket เฉลี่ยข้าม tables
        pu → ranking score normalize เป็น [0, 1] (ยิ่งสูงยิ่งใกล้ positive)

        Parameters
        ----------
        df : polars.DataFrame
            features (columns ต้องตรงกับตอน fit); ประมวลผลทีละ chunk_size แถว

        Returns
        -------
        numpy.ndarray, shape (n,)
        """
        Z = self.encoder_.transform(df)
        codes = self.hasher_.transform(Z, chunk_size=self.chunk_size)
        s = self.scorer_.score(codes)
        self.diagnostics_["scoring"] = self.scorer_.diagnostics_
        if self.scorer_.diagnostics_["miss_rate_final"] > 0.30:
            warnings.warn(
                "miss rate > 30% — สาเหตุที่เป็นไปได้: n_bits มากไป หรือ seeds น้อยไป "
                "ลองลด n_bits หรือเพิ่ม seeds"
            )
        if self.mode == "pu":
            span = s.max() - s.min()
            s = (s - s.min()) / span if span > 0 else np.zeros_like(s)
        return s

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """คืน class probability (two_class เท่านั้น)

        Parameters
        ----------
        df : polars.DataFrame
            features (columns ต้องตรงกับตอน fit)

        Returns
        -------
        numpy.ndarray, shape (n, 2)
            คอลัมน์ 0 = P(negative), คอลัมน์ 1 = P(positive)

        Raises
        ------
        ValueError
            ถ้า mode="pu" (ใช้ `score_samples` แทน)
        """
        if self.mode != "two_class":
            raise ValueError("predict_proba ใช้ได้เฉพาะ two_class — PU ใช้ score_samples")
        p = self.score_samples(df)
        return np.column_stack([1.0 - p, p])

    def save(self, path) -> None:
        """บันทึกเป็น directory: projections.npy + tables.parquet + meta.json

        Caveat: hashed categorical ใช้ polars hash — โหลดข้าม polars major version
        อาจได้ hash ต่างกัน (บันทึก polars_version ไว้ใน meta)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "projections.npy", self.hasher_.P)
        tables = pl.concat([
            t.with_columns(table=pl.lit(k).cast(pl.Int32))
            for k, t in enumerate(self.scorer_.tables_)
        ])
        tables.write_parquet(path / "tables.parquet")
        meta = {
            "mode": self.mode,
            "config": self.config_,
            "prior": self.scorer_.prior_,
            "chunk_size": self.chunk_size,
            "encoder": self.encoder_.to_dict(),
            "polars_version": pl.__version__,
        }
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path) -> "LeadX":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        cfg = meta["config"]
        clf = cls(
            mode=meta["mode"], tune=False, n_bits=cfg["n_bits"],
            n_tables=cfg["n_tables"], m=cfg["m"], strategy=cfg["strategy"],
            weights=cfg["weights"], chunk_size=meta["chunk_size"],
        )
        clf.config_ = cfg
        clf.tuning_results_ = None
        clf.encoder_ = FeatureEncoder.from_dict(meta["encoder"])
        clf.hasher_ = HashEnsemble(np.load(path / "projections.npy"))
        scorer = SeedScorer(
            mode=meta["mode"], m=cfg["m"], weights=cfg["weights"], n_bits=cfg["n_bits"],
        )
        all_tables = pl.read_parquet(path / "tables.parquet")
        scorer.tables_ = [
            all_tables.filter(pl.col("table") == k).drop("table")
            for k in range(cfg["n_tables"])
        ]
        scorer.n_tables_ = cfg["n_tables"]
        scorer.prior_ = meta["prior"]
        clf.scorer_ = scorer
        clf.diagnostics_ = {"config": cfg}
        return clf
