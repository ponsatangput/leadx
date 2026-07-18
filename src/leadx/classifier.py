"""LeadX — facade ต่อท่อ encoder → projections → hashing → scoring (+tuning)

ไม่มี logic ของตัวเอง — ทุก logic อยู่ใน component และทดสอบที่นั่น (spec §2)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from .encoder import FeatureEncoder
from .hashing import HashEnsemble, infer_hash_mode
from .projections import ProjectionBank, projection_diagnostics
from .rpforest import RPForest
from .scoring import SeedScorer, normalize_pu, pu_score_range, resolve_hier_depths
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
    feature_weighting : {"none", "corr"}
        "corr" ถ่วง feature ตาม |corr กับ y| ก่อน hash (แก้ noise dilution — feature
        ที่ไม่เกี่ยวได้น้ำหนักต่ำ ไม่ปน hash). two_class เท่านั้น. ใช้ y → codes กลาย
        เป็น task-specific จึงปิดการ reuse projections ข้ามโจทย์ (default "none")
    score_type : {"prop", "lift"}
        "prop" — สัดส่วน smoothed (m-estimate); "lift" — density ratio เทียบ reference
        (imbalance-invariant: prior ไม่อยู่ในสูตร; สเกลตีความได้; fallback = 1)
    alpha : float
        shrink-toward-1 strength ของ lift — กล่องหลักฐานน้อยถูกดึงเข้า 1
    agg : {"mean", "log"}
        รวม lift ข้าม tables: arithmetic / geometric (NB-style)
    hash_mode : {"median", "origin"}
        จุดตัด hyperplane: median ของ w·x (balanced โดยโครงสร้าง, fit เร็ว, robust —
        default) / ตัดที่ 0 (ดั้งเดิม + balance-select machinery)
    cat_bit_frac : "auto" | float
        สัดส่วน bits ที่ project เฉพาะ categorical blocks (แก้ hyperplane bits มองไม่
        เห็น one-hot); "auto" = ตามสัดส่วนมิติ; อยู่ใน tuner grid — tuner เลือก 0
        เองเมื่อ categorical เป็น noise
    engine : {"auto", "hyperplane", "rpforest"}
        "hyperplane" — global random hyperplanes + median (single-matmul);
        "rpforest" — balanced rpTree forest: ทิศสุ่มต่อ node + median ท้องถิ่น →
        bucket ขนาดเท่ากันโดยโครงสร้าง (ฆ่า mega-bucket — วัดจริง PU: fraud 0.3%
        +39%, home-credit natural +6% เทียบ hyperplane+hier; fit แพงกว่า ~2-4x,
        depth <= 12 ใน tuner); "auto" (default) — tune=True ให้ tuner ประเมินทั้ง
        สอง engine แล้วเลือกจาก OOF, tune=False = "hyperplane"
    hier_depths : None | "auto" | sequence[int]
        hierarchical shrinkage ข้าม prefix depths (score_type="lift" เท่านั้น):
        "auto" = (4, 6, ..., n_bits) — bucket ลึกที่หลักฐานบาง inherit lift ของ
        bucket แม่ ลด variance เมื่อ seeds น้อยมาก (วัดจริง: +6-41% rel PR-AUC
        ที่ imbalance 1%-0.1%, natural rate ไม่ regress); default None = ปิด
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
                 feature_weighting="none", score_type="prop", alpha=2.0, agg="mean",
                 hash_mode="median", cat_bit_frac="auto", whiten=False,
                 hier_depths=None, engine="auto",
                 chunk_size=1_000_000, random_state=0):
        if mode not in ("two_class", "pu"):
            raise ValueError(f"mode ไม่รู้จัก: {mode}")
        if engine not in ("auto", "hyperplane", "rpforest"):
            raise ValueError(f"engine ไม่รู้จัก: {engine}")
        if feature_weighting not in ("none", "corr", "contrast"):
            raise ValueError(f"feature_weighting ไม่รู้จัก: {feature_weighting}")
        self.mode = mode
        self.tune = tune
        self.n_bits = n_bits
        self.n_tables = n_tables
        self.m = m
        self.strategy = strategy
        self.weights = weights
        self.grid = grid
        self.feature_weighting = feature_weighting
        self.score_type = score_type
        self.alpha = alpha
        self.agg = agg
        self.hash_mode = hash_mode
        self.cat_bit_frac = cat_bit_frac
        self.whiten = whiten
        self.hier_depths = hier_depths
        self.engine = engine
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
        if self.mode == "two_class" and len(np.unique(y)) < 2:
            raise ValueError(
                "two_class ต้องมี y ครบทั้งสอง class — ถ้ามีแต่ positive ใช้ mode='pu'"
            )
        # Parity contract (audit P0): ใน PU ที่มี background — encoder และ geometry
        # (projections/thresholds/whiten) fit จาก background *population*
        # (y-independent, สถิติเสถียรกว่า, vocab ครบกว่า) ไม่ใช่จาก seeds ไม่กี่ตัว —
        # ให้ production API ตรงกับ protocol ของ benchmark
        enc_source = background if (self.mode == "pu" and background is not None) else df
        self.encoder_ = FeatureEncoder().fit(enc_source)
        Z = self.encoder_.transform(df)
        # feature weighting (หลัง standardize มิฉะนั้นถูก z-score หักล้าง) — แก้ noise
        # dilution: ถ่วง feature ตาม |corr กับ y| ก่อน hash. ใช้ y → codes task-specific
        # (ปิด reuse ข้ามโจทย์) จึง opt-in default "none"
        Z_bg = self.encoder_.transform(background) if background is not None else None
        if self.tune:
            # ส่ง Z ที่ *ยังไม่ถ่วงน้ำหนัก* — tuner cross-fit weights ต่อ fold เอง
            # (ปิด OOF leak: validation fold ห้ามเห็น y ตัวเองผ่าน weights — audit P0)
            Z_geom_raw = Z_bg if (self.mode == "pu" and Z_bg is not None) else Z
            tuner_grid = dict(self.grid or {})
            if self.engine != "auto" and "engine" not in tuner_grid:
                tuner_grid["engine"] = [self.engine]
            tuner = Tuner(grid=tuner_grid, mode=self.mode,
                          cat_bit_frac=self.cat_bit_frac, hash_mode=self.hash_mode,
                          whiten=self.whiten, random_state=self.random_state)
            best = tuner.tune(Z, y, Z_background=Z_bg, cat_mask=self.encoder_.cat_mask_,
                              Z_geom=Z_geom_raw, feature_weighting=self.feature_weighting)
            self.tuning_results_ = tuner.results_
            cfg = {k: best[k] for k in ("strategy", "n_bits", "n_tables", "m", "weights",
                                        "score_type", "alpha", "agg")}
            cfg["engine"] = best.get("engine", "hyperplane")
            fr = best.get("cat_bit_frac", str(self.cat_bit_frac))
            cfg["cat_bit_frac"] = "auto" if fr == "auto" else float(fr)
            cfg["whiten"] = bool(best.get("whiten", self.whiten))
        else:
            self.tuning_results_ = None
            cfg = {"strategy": self.strategy, "n_bits": self.n_bits,
                   "n_tables": self.n_tables, "m": self.m, "weights": self.weights,
                   "score_type": self.score_type, "alpha": self.alpha, "agg": self.agg,
                   "cat_bit_frac": self.cat_bit_frac, "whiten": self.whiten,
                   "engine": "hyperplane" if self.engine == "auto" else self.engine}
        # weights เต็ม (จาก seeds ทั้งหมด) สำหรับ deploy — คำนวณหลัง tune เพื่อไม่ leak
        # เข้าการประเมิน; ตอน deploy ใช้ข้อมูลทั้งหมดได้โดยชอบธรรม
        self.feature_weights_ = self._compute_weights(Z, y, Z_bg)
        if self.feature_weights_ is not None:
            Z = Z * self.feature_weights_
            if Z_bg is not None:
                Z_bg = Z_bg * self.feature_weights_
        Z_geom = Z_bg if (self.mode == "pu" and Z_bg is not None) else Z
        self.config_ = cfg
        if self.tune and cfg["engine"] == "rpforest":
            forest = tuner.best_bank_
            if forest is None:
                # label-dependent winner (weighted — cross-fit ตอน tune) → refit เต็ม
                self.hasher_ = RPForest(
                    n_tables=cfg["n_tables"], n_bits=cfg["n_bits"],
                    random_state=self.random_state).fit(Z_geom)
            else:
                # prefix ของ forest ที่ tuner ประเมินจริง — สิ่งที่ deploy = สิ่งที่วัด
                self.hasher_ = forest.prefix(cfg["n_bits"], cfg["n_tables"])
            proj_diag = {"engine": "rpforest"}
        elif self.tune:
            # ใช้ prefix ของ bank ที่ Tuner ประเมินจริง — สิ่งที่ deploy = สิ่งที่วัด
            # (prefix slice ถูกต้องเพราะ codes ที่ประเมิน = mask_codes ของ bank นี้เอง)
            bank = tuner.best_bank_
            if bank is None:
                # winning combo เป็น label-dependent (supervised/weighted — cross-fit
                # ต่อ fold ตอน tune) → refit เต็มบนข้อมูลทั้งหมดสำหรับ deploy
                bank = ProjectionBank(
                    n_tables=cfg["n_tables"], n_bits=cfg["n_bits"],
                    strategy=cfg["strategy"], hash_mode=self.hash_mode,
                    cat_bit_frac=cfg["cat_bit_frac"], whiten=cfg["whiten"],
                    random_state=self.random_state,
                )
                bank.fit(Z_geom if cfg["strategy"] != "supervised" else Z,
                         y if self.mode == "two_class" else None,
                         cat_mask=self.encoder_.cat_mask_)
                self.hasher_ = HashEnsemble(bank.P_, bank.thresholds_)
                proj_diag = bank.diagnostics()
            else:
                P = np.ascontiguousarray(bank.P_[: cfg["n_tables"], :, : cfg["n_bits"]])
                thr = np.ascontiguousarray(bank.thresholds_[: cfg["n_tables"], : cfg["n_bits"]])
                self.hasher_ = HashEnsemble(P, thr)
                # diagnostics ต้องสะท้อน prefix ที่ deploy จริง ไม่ใช่ bank เต็มของ tuner
                proj_diag = projection_diagnostics(P, thr, Z, strategy=cfg["strategy"])
        elif cfg["engine"] == "rpforest":
            # local recursion: ทิศ+threshold ต่อ node → equal-mass buckets; leaf
            # codes มี prefix structure เดียวกับ bank → scorer/hier ใช้ร่วมได้หมด
            self.hasher_ = RPForest(
                n_tables=cfg["n_tables"], n_bits=cfg["n_bits"],
                random_state=self.random_state,
            ).fit(Z_geom)
            proj_diag = {"engine": "rpforest"}
        else:
            bank = ProjectionBank(
                n_tables=cfg["n_tables"], n_bits=cfg["n_bits"], strategy=cfg["strategy"],
                hash_mode=self.hash_mode, cat_bit_frac=cfg["cat_bit_frac"],
                whiten=cfg["whiten"], random_state=self.random_state,
            )
            bank.fit(Z_geom, y if self.mode == "two_class" else None,
                     cat_mask=self.encoder_.cat_mask_)
            self.hasher_ = HashEnsemble(bank.P_, bank.thresholds_)
            proj_diag = bank.diagnostics()
        codes = self.hasher_.transform(Z)
        bg_codes = self.hasher_.transform(Z_bg) if Z_bg is not None else None
        hier = resolve_hier_depths(self.hier_depths, cfg["n_bits"])
        if hier is not None and cfg["score_type"] != "lift":
            warnings.warn(
                "hier_depths ใช้ได้กับ score_type='lift' เท่านั้น — ปิด hierarchical "
                f"shrinkage สำหรับ config นี้ (score_type={cfg['score_type']!r})"
            )
            hier = None
        self.scorer_ = SeedScorer(
            mode=self.mode, m=cfg["m"], weights=cfg["weights"], n_bits=cfg["n_bits"],
            score_type=cfg["score_type"], alpha=cfg["alpha"], agg=cfg["agg"],
            hier_depths=hier,
        ).fit(codes, y, background_codes=bg_codes)
        self._pi = float(y.mean()) if self.mode == "two_class" else None
        # PU calibration: normalize ด้วยค่าคงที่จาก fit (batch-invariant) — spec ห้าม
        # ใช้ min/max ของ batch ที่กำลัง score
        if self.mode == "pu" and cfg["score_type"] != "lift":
            self._pu_lo, self._pu_span = pu_score_range(self.scorer_, codes, bg_codes)
        else:
            self._pu_lo, self._pu_span = None, None
        self.diagnostics_ = {"projections": proj_diag, "config": cfg}
        return self

    def _compute_weights(self, Z: np.ndarray, y: np.ndarray, Z_bg=None):
        """น้ำหนัก feature (Z standardize แล้ว):
        "corr"     — |point-biserial corr กับ y| (two_class เท่านั้น)
        "contrast" — |mean(seeds) − mean(background)| (pu เท่านั้น — วัดจริง
                     home-credit PU: +26-30% rel PR-AUC; task-aware ปิด reuse)
        """
        if self.feature_weighting == "none":
            return None
        if self.feature_weighting == "corr":
            if self.mode != "two_class":
                return None
            yc = y - y.mean()
            denom = np.sqrt((Z ** 2).sum(axis=0)) * np.sqrt((yc ** 2).sum()) + 1e-12
            w = np.abs((Z * yc[:, None]).sum(axis=0) / denom)
            return w.astype(np.float32)
        # contrast
        if self.mode != "pu" or Z_bg is None:
            raise ValueError(
                "feature_weighting='contrast' ใช้กับ mode='pu' + ต้องส่ง background "
                "(weights = |mean(seeds) − mean(background)|)"
            )
        w = np.abs(Z.mean(axis=0) - Z_bg.mean(axis=0)) + 1e-3
        return (w / w.max()).astype(np.float32)

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
        if self.feature_weights_ is not None:
            Z = Z * self.feature_weights_
        codes = self.hasher_.transform(Z, chunk_size=self.chunk_size)
        s = self.scorer_.score(codes)
        self.diagnostics_["scoring"] = self.scorer_.diagnostics_
        if self.scorer_.diagnostics_["miss_rate_final"] > 0.30:
            warnings.warn(
                "miss rate > 30% — สาเหตุที่เป็นไปได้: n_bits มากไป หรือ seeds น้อยไป "
                "ลองลด n_bits หรือเพิ่ม seeds"
            )
        if self.mode == "pu" and self.config_["score_type"] != "lift":
            s = normalize_pu(s, self._pu_lo, self._pu_span)
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
        if self.config_["score_type"] == "lift":
            # SCAR: P(y=1|x) ~= pi * lift(x) — แปลงเป็น probability หยาบ (ranking คงเดิม)
            p = np.clip(self._pi * p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def save(self, path) -> None:
        """บันทึกเป็น directory: projections.npy + tables.parquet + meta.json

        Caveat: hashed categorical ใช้ polars hash — โหลดข้าม polars major version
        อาจได้ hash ต่างกัน (บันทึก polars_version ไว้ใน meta)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if isinstance(self.hasher_, RPForest):
            arrs = self.hasher_.to_arrays()
            np.save(path / "rpf_dirs.npy", arrs["dirs"])
            np.save(path / "rpf_thr.npy", arrs["thr"])
        else:
            np.save(path / "projections.npy", self.hasher_.P)
            np.save(path / "thresholds.npy", self.hasher_.thresholds)
        tables = pl.concat([
            t.with_columns(table=pl.lit(k).cast(pl.Int32))
            for k, t in enumerate(self.scorer_.tables_)
        ])
        tables.write_parquet(path / "tables.parquet")
        hier = getattr(self.scorer_, "hier_depths", None)
        if hier is not None:
            frames = []
            for k, per_depth in enumerate(self.scorer_.hier_tables_):
                for d, (pu_, pc_, ru_, rc_) in zip(hier, per_depth):
                    for kind, u, c in (("pos", pu_, pc_), ("ref", ru_, rc_)):
                        frames.append(pl.DataFrame({
                            "table": np.full(len(u), k, dtype=np.int32),
                            "depth": np.full(len(u), d, dtype=np.int32),
                            "kind": [kind] * len(u), "code": u, "cnt": c,
                        }))
            pl.concat(frames).write_parquet(path / "hier_tables.parquet")
        meta = {
            "mode": self.mode,
            "config": self.config_,
            "prior": self.scorer_.prior_,
            "chunk_size": self.chunk_size,
            "encoder": self.encoder_.to_dict(),
            "feature_weighting": self.feature_weighting,
            "hash_mode": self.hash_mode,
            "whiten": self.config_.get("whiten", self.whiten),
            "pu_lo": self._pu_lo,
            "pu_span": self._pu_span,
            "pi": self._pi,
            "feature_weights": None if self.feature_weights_ is None
            else self.feature_weights_.tolist(),
            "engine": self.config_.get("engine", self.engine),
            "hier_depths": None if hier is None else list(hier),
            "hier_pos_total": getattr(self.scorer_, "hier_pos_total_", None),
            "hier_ref_total": getattr(self.scorer_, "hier_ref_total_", None),
            "polars_version": pl.__version__,
        }
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path) -> "LeadX":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        cfg = meta["config"]
        cfg.setdefault("score_type", "prop")
        cfg.setdefault("alpha", 2.0)
        cfg.setdefault("agg", "mean")
        cfg.setdefault("cat_bit_frac", "auto")
        engine = meta.get("engine", "hyperplane")
        thr_file = path / "thresholds.npy"
        thresholds = np.load(thr_file) if thr_file.exists() else None
        clf = cls(
            mode=meta["mode"], tune=False, n_bits=cfg["n_bits"],
            n_tables=cfg["n_tables"], m=cfg["m"], strategy=cfg["strategy"],
            weights=cfg["weights"], feature_weighting=meta.get("feature_weighting", "none"),
            score_type=cfg["score_type"], alpha=cfg["alpha"], agg=cfg["agg"],
            hash_mode=infer_hash_mode(meta.get("hash_mode"), thresholds),
            whiten=meta.get("whiten", False),
            cat_bit_frac=cfg["cat_bit_frac"],
            hier_depths=meta.get("hier_depths"),
            engine=engine,
            chunk_size=meta["chunk_size"],
        )
        clf._pi = meta.get("pi")
        clf.config_ = cfg
        clf.tuning_results_ = None
        clf._pu_lo = meta.get("pu_lo")
        clf._pu_span = meta.get("pu_span")
        if meta["mode"] == "pu" and clf._pu_lo is None:
            clf._pu_lo, clf._pu_span = 0.0, 1.0  # model เก่าก่อน batch-invariant fix
        fw = meta.get("feature_weights")
        clf.feature_weights_ = None if fw is None else np.asarray(fw, dtype=np.float32)
        clf.encoder_ = FeatureEncoder.from_dict(meta["encoder"])
        if engine == "rpforest":
            clf.hasher_ = RPForest.from_arrays(np.load(path / "rpf_dirs.npy"),
                                               np.load(path / "rpf_thr.npy"))
        else:
            clf.hasher_ = HashEnsemble(np.load(path / "projections.npy"), thresholds)
        hier = meta.get("hier_depths")
        scorer = SeedScorer(
            mode=meta["mode"], m=cfg["m"], weights=cfg["weights"], n_bits=cfg["n_bits"],
            score_type=cfg["score_type"], alpha=cfg["alpha"], agg=cfg["agg"],
            hier_depths=None if hier is None else tuple(hier),
        )
        all_tables = pl.read_parquet(path / "tables.parquet")
        scorer.tables_ = [
            all_tables.filter(pl.col("table") == k).drop("table")
            for k in range(cfg["n_tables"])
        ]
        scorer.n_tables_ = cfg["n_tables"]
        scorer.prior_ = meta["prior"]
        if hier is not None:
            ht = pl.read_parquet(path / "hier_tables.parquet").sort("code")
            scorer.hier_pos_total_ = meta["hier_pos_total"]
            scorer.hier_ref_total_ = meta["hier_ref_total"]
            scorer.hier_tables_ = []
            for k in range(cfg["n_tables"]):
                per_depth = []
                for d in hier:
                    sub = ht.filter((pl.col("table") == k) & (pl.col("depth") == d))
                    parts = {
                        kind: sub.filter(pl.col("kind") == kind)
                        for kind in ("pos", "ref")
                    }
                    per_depth.append((
                        parts["pos"]["code"].to_numpy().astype(np.uint64),
                        parts["pos"]["cnt"].to_numpy().astype(np.float64),
                        parts["ref"]["code"].to_numpy().astype(np.uint64),
                        parts["ref"]["cnt"].to_numpy().astype(np.float64),
                    ))
                scorer.hier_tables_.append(per_depth)
        clf.scorer_ = scorer
        clf.diagnostics_ = {"config": cfg}
        return clf
