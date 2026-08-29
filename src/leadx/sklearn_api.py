"""LeadXClassifier — scikit-learn / XGBoost-shaped interface over LeadX.

The core `LeadX` class exposes the PU contract literally: you hand it the
confirmed positives and, separately, the population to compare them against.
That is faithful to the maths but unfamiliar — and dangerous, because a user
coming from XGBoost writes `fit(X, y)` and would have silently trained on the
wrong thing.

This wrapper speaks the interface people already know:

    m = LeadXClassifier(profile="rare_pu")
    m.fit(X, y)                       # y = 1 confirmed positive, 0 UNLABELED
    scores = m.predict_proba(X_next)[:, 1]

`y == 0` means *unlabeled*, not *negative*: with ``task="pu"`` (the default) the
whole of ``X`` becomes the background population, which is exactly the
positive-unlabeled setting fraud/lead problems live in. Pass
``task="two_class"`` when zeros really are confirmed negatives.
"""
from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (average_precision_score,
                             precision_recall_curve, roc_auc_score)

from .classifier import LeadX
from .frames import to_polars
from .scoring import SeedScorer, resolve_hier_depths

#: named starting points; every field can still be overridden per-argument
PROFILES = {
    # measured flagship for rare + PU (see outputs/*_REPORT.md)
    "rare_pu": dict(n_bits=10, n_tables=80, score_type="lift", agg="log",
                    feature_weighting="contrast", engine="rpforest",
                    cat_engine="minhash", hier_depths="auto"),
    # cheaper: single-matmul geometry, one-hot categoricals, no weighting
    "fast": dict(n_bits=12, n_tables=40, score_type="lift", agg="log",
                 feature_weighting="none", engine="hyperplane",
                 cat_engine="onehot", hier_depths=None),
}
_TUNABLE = ("n_bits", "n_tables", "score_type", "agg", "feature_weighting",
            "engine", "cat_engine", "hier_depths")


SUPPORTED_METRICS = (
    "pr_auc", "roc_auc", "f1",
    "recall@K (top-K rows, e.g. recall@1000)", "recall@P% (e.g. recall@0.5%)",
    "precision@K (e.g. precision@1000)", "precision@P% (e.g. precision@0.5%)",
)


def _top_k(y: np.ndarray, s: np.ndarray, spec: str) -> tuple[np.ndarray, int]:
    """spec = '1000' (top-K rows) หรือ '0.5%' (top fraction) -> (hit labels, k)"""
    spec = spec.strip()
    if spec.endswith("%"):
        frac = float(spec[:-1]) / 100.0
        if not 0 < frac <= 1:
            raise ValueError(f"percentage ต้องอยู่ใน (0, 100] — ได้ {spec}")
        k = int(np.ceil(len(y) * frac))
    else:
        k = int(float(spec))
        if k <= 0:
            raise ValueError(f"k ต้อง > 0 — ได้ {spec}")
    k = max(1, min(k, len(y)))
    idx = np.argpartition(-s, k - 1)[:k]
    return y[idx], k


def _score_metric(y: np.ndarray, s: np.ndarray, metric: str) -> float:
    """metric string -> score (สูง = ดี). รูปแบบที่รองรับดู SUPPORTED_METRICS"""
    name = str(metric).strip().lower()
    if name == "pr_auc":
        return float(average_precision_score(y, s))
    if name == "roc_auc":
        return float(roc_auc_score(y, s))
    if name == "f1":
        # best achievable F1 over the ranking (threshold-free — คะแนน lift ไม่ใช่
        # probability จึงไม่มี threshold 0.5 ที่มีความหมาย)
        prec, rec, _ = precision_recall_curve(y, s)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        return float(np.nanmax(f1))
    if "@" in name:
        kind, spec = name.split("@", 1)
        if kind in ("recall", "precision"):
            hits, k = _top_k(y, s, spec)
            n_hit = float(hits.sum())
            if kind == "recall":
                return n_hit / max(1.0, float(y.sum()))
            return n_hit / k
    raise ValueError(
        f"eval_metric ไม่รู้จัก / unknown eval_metric: {metric!r}. "
        f"Supported: {', '.join(SUPPORTED_METRICS)}")


class LeadXClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost-shaped classifier for rare-event / positive-unlabeled retrieval.

    Parameters
    ----------
    profile : {"rare_pu", "fast"}
        starting configuration; any explicit argument below overrides it
    task : {"pu", "two_class"}
        ``"pu"`` (default): ``y == 1`` are confirmed positives, ``y == 0`` are
        **unlabeled**, and all rows form the background population.
        ``"two_class"``: zeros are genuine negatives.
    n_bits, n_tables, engine, cat_engine, feature_weighting, hier_depths,
    score_type, agg : optional
        see :class:`~leadx.LeadX`; ``None`` means "take it from the profile"
    categorical_columns : sequence[str] | None
        columns to route through MinHash; ``None`` = auto-detect non-numeric
    background_sample : int | None
        PU only. Fit the geometry and the density denominator on a uniform
        sample of this many rows instead of all of ``X``. The lift formula uses
        the background as a *mass fraction* (``n_total_b / n_bg_total``), so a
        uniform sample estimates it without bias; measured at 5% (10k rows,
        ``n_bits=10``) the loss is inside the seed-to-seed noise floor while fit
        is 5.9x faster — see ``outputs/BACKGROUND_SAMPLE_REPORT.md``. Rows per
        bucket is what matters, so keep it well above ``2 ** n_bits``.
    background_policy : {"frozen", "refresh"}
        ``"frozen"`` (default): the density denominator is fixed at fit time, so
        scores are comparable across batches. ``"refresh"``: recompute the
        denominator from each scored batch — adapts to traffic shifts but the
        scores are transductive and not comparable between batches.
    contamination : "auto" | float
        expected positive fraction, used only to place :meth:`predict`'s
        threshold; ``"auto"`` uses the confirmed-positive rate seen at fit
    prior : float | None
        if given, ``predict_proba`` returns ``clip(prior * lift, 0, 1)``
        (SCAR assumption). Otherwise it returns the ranking score rescaled to
        [0, 1] with the fit-time range — a ranking, not a calibrated probability.

    Attributes
    ----------
    feature_names_in_, n_features_in_, classes_
    n_background_ : rows used as the PU denominator (None for two_class)
    best_params_, evals_result_ : populated when ``eval_set`` is used
    feature_importances_ : per input column (NaN for MinHash-routed columns)
    """

    def __init__(self, profile="rare_pu", task="pu", n_bits=None, n_tables=None,
                 engine=None, cat_engine=None, feature_weighting=None,
                 hier_depths=None, score_type=None, agg=None,
                 categorical_columns=None, background_sample=None,
                 background_policy="frozen",
                 contamination="auto", prior=None, chunk_size=1_000_000,
                 random_state=0):
        self.profile = profile
        self.task = task
        self.n_bits = n_bits
        self.n_tables = n_tables
        self.engine = engine
        self.cat_engine = cat_engine
        self.feature_weighting = feature_weighting
        self.hier_depths = hier_depths
        self.score_type = score_type
        self.agg = agg
        self.categorical_columns = categorical_columns
        self.background_sample = background_sample
        self.background_policy = background_policy
        self.contamination = contamination
        self.prior = prior
        self.chunk_size = chunk_size
        self.random_state = random_state

    # ------------------------------------------------------------- config ---
    def _resolve(self, overrides: dict | None = None) -> dict:
        if self.profile not in PROFILES:
            raise ValueError(f"profile ไม่รู้จัก: {self.profile} "
                             f"(มี {sorted(PROFILES)})")
        cfg = dict(PROFILES[self.profile])
        for key in _TUNABLE:
            val = getattr(self, key)
            if val is not None:
                cfg[key] = val
        cfg.update(overrides or {})
        return cfg

    def _resolve_cat_engine(self, cfg: dict, Xp: pl.DataFrame) -> dict:
        """rare_pu defaults to MinHash, but plenty of real feature tables are
        all-numeric (categoricals already encoded upstream). Fall back instead
        of failing — unless the user named the columns explicitly, in which case
        a missing column is still an error."""
        if cfg["cat_engine"] != "minhash":
            return cfg
        # profile-level coherence: hier needs a single bit-width, MinHash tables
        # have their own — drop hier when it came from the profile (an explicit
        # hier_depths= still reaches LeadX and gets its warning there)
        if self.hier_depths is None and cfg.get("hier_depths") is not None:
            cfg = {**cfg, "hier_depths": None}
        if self.categorical_columns is not None:
            return cfg                      # explicit -> LeadX validates & raises
        if not any(not Xp[c].dtype.is_numeric() for c in Xp.columns):
            warnings.warn(
                "no categorical columns found; falling back to "
                "cat_engine='onehot' (numeric-only input). Pass "
                "categorical_columns=[...] if some numeric columns are really "
                "categorical IDs.", UserWarning, stacklevel=3)
            cfg = {**cfg, "cat_engine": "onehot"}
        return cfg

    def _build(self, cfg: dict) -> LeadX:
        return LeadX(
            mode="pu" if self.task == "pu" else "two_class", tune=False,
            n_bits=cfg["n_bits"], n_tables=cfg["n_tables"],
            score_type=cfg["score_type"], agg=cfg["agg"],
            feature_weighting=cfg["feature_weighting"], engine=cfg["engine"],
            cat_engine=cfg["cat_engine"], hier_depths=cfg["hier_depths"],
            minhash_columns=self.categorical_columns,
            chunk_size=self.chunk_size, random_state=self.random_state,
        )

    def _fit_one(self, X: pl.DataFrame, y: np.ndarray, cfg: dict,
                 sample_weight=None) -> LeadX:
        model = self._build(cfg)
        if self.task != "pu":
            self.n_background_ = None
            return model.fit(X, y=y, sample_weight=sample_weight)
        sw = None if sample_weight is None else sample_weight[y == 1]
        bg = X
        if self.background_sample is not None:
            n = int(self.background_sample)
            if n < 1:
                raise ValueError(f"background_sample ต้อง >= 1 (ได้ {n})")
            if n < X.height:
                # ทุก seed ยังอยู่ครบ — ที่ sample คือ *ตัวหาร* (ประชากร unlabeled)
                bg = X.sample(n, seed=self.random_state, shuffle=False)
        self.n_background_ = bg.height
        return model.fit(X.filter(pl.Series(y == 1)), background=bg,
                         sample_weight=sw)

    # ---------------------------------------------------------------- fit ---
    def fit(self, X, y, eval_set=None, param_grid=None, eval_metric="pr_auc",
            sample_weight=None):
        """Fit on ``(X, y)``; ``y == 1`` confirmed positive, ``y == 0`` unlabeled.

        ``eval_set=(X_val, y_val)`` scores the fitted model on a held-out set —
        use a *later time window* for temporal problems, which internal random
        cross-validation would flatter. Combined with ``param_grid`` (dict of
        lists) it also selects the configuration by ``eval_metric`` on that set.
        """
        if self.task not in ("pu", "two_class"):
            raise ValueError(f"task ไม่รู้จัก: {self.task}")
        if self.background_policy not in ("frozen", "refresh"):
            raise ValueError(f"background_policy ไม่รู้จัก: {self.background_policy}")
        Xp = to_polars(X)
        y = np.asarray(y).ravel()
        if len(y) != Xp.height:
            raise ValueError(f"y มี {len(y)} แถว แต่ X มี {Xp.height}")
        if not set(np.unique(y)) <= {0, 1}:
            raise ValueError("y ต้องเป็น 0/1 (1 = positive ที่ยืนยันแล้ว, "
                             "0 = unlabeled สำหรับ task='pu')")
        n_pos = int(y.sum())
        if n_pos < 20:
            raise ValueError(f"ต้องมี y==1 อย่างน้อย 20 แถว (ได้ {n_pos})")
        if n_pos == len(y):
            raise ValueError("y เป็น 1 ทั้งหมด — ต้องมีประชากร unlabeled ให้เทียบด้วย")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()
            if sample_weight.shape != (len(y),):
                raise ValueError(
                    f"sample_weight ต้องมี {len(y)} ค่า (ได้ {sample_weight.shape}) / "
                    f"sample_weight must have one value per row of X")
            if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight < 0):
                raise ValueError("sample_weight ต้องเป็นค่าจำกัดและไม่ติดลบ / "
                                 "must be finite and non-negative")
            if self.task == "pu" and sample_weight[y == 1].sum() <= 0:
                raise ValueError("น้ำหนักรวมของ positive seeds ต้อง > 0 / "
                                 "positive seed weights must sum to > 0")
        self.feature_names_in_ = list(Xp.columns)
        self.n_features_in_ = Xp.width
        self.classes_ = np.array([0, 1])

        candidates = [self._resolve_cat_engine(self._resolve(), Xp)]
        if param_grid:
            keys = sorted(param_grid)
            candidates = [self._resolve_cat_engine(self._resolve(dict(zip(keys, combo))), Xp)
                          for combo in itertools.product(*(param_grid[k] for k in keys))]
            if eval_set is None:
                raise ValueError("param_grid ต้องมาคู่กับ eval_set "
                                 "(เลือก config จาก held-out ไม่ใช่จาก train)")
        self.evals_result_ = []
        best, best_cfg, best_score = None, None, -np.inf
        Xv = yv = None
        if eval_set is not None:
            Xv_raw, yv = eval_set
            Xv = to_polars(Xv_raw, columns=self.feature_names_in_)
            yv = np.asarray(yv).ravel()
        for cfg in candidates:
            model = self._fit_one(Xp, y, cfg, sample_weight)
            if Xv is None:
                best, best_cfg = model, cfg
                break
            s = model.score_samples(Xv)
            score = _score_metric(yv, s, eval_metric)
            self.evals_result_.append({**{k: cfg[k] for k in _TUNABLE},
                                       eval_metric: score})
            if score > best_score:
                best, best_cfg, best_score = model, cfg, score
        self.model_ = best
        self.best_params_ = best_cfg
        self.best_score_ = None if Xv is None else best_score
        self.eval_metric_ = eval_metric if Xv is not None else None

        # fit-time reference scores: batch-invariant [0,1] mapping + threshold
        ref = self.model_.score_samples(Xp)
        self._ref_lo = float(np.quantile(ref, 0.001))
        self._ref_hi = float(np.quantile(ref, 0.999))
        q = (n_pos / len(y)) if self.contamination == "auto" else float(self.contamination)
        self._threshold = float(np.quantile(ref, 1.0 - min(max(q, 1e-6), 0.5)))
        if self.background_policy == "refresh":
            self._seed_codes = self.model_.transform_codes(
                Xp.filter(pl.Series(y == 1)))
        self._imp_dims = self._reporting_weights(Xp, y)
        return self

    def _reporting_weights(self, Xp: pl.DataFrame, y: np.ndarray) -> np.ndarray:
        """น้ำหนักต่อ encoded dim สำหรับรายงาน — ถ้าโมเดลใช้ weighting อยู่แล้วก็ใช้
        ตัวนั้น ไม่งั้นคำนวณ contrast/corr เพิ่ม (ถูก) เพื่อให้ feature_importances_
        มีให้ใช้เสมอแบบ XGBoost"""
        m = self.model_
        if m.feature_weights_ is not None:
            return np.asarray(m.feature_weights_, dtype=np.float64)
        Z = m.encoder_.transform(m._drop_mh(Xp))
        if self.task == "pu":
            w = np.abs(Z[y == 1].mean(axis=0) - Z.mean(axis=0))
        else:
            yc = y.astype(np.float64) - y.mean()
            denom = np.sqrt((Z ** 2).sum(axis=0)) * np.sqrt((yc ** 2).sum()) + 1e-12
            w = np.abs((Z * yc[:, None]).sum(axis=0) / denom)
        return np.asarray(w, dtype=np.float64)

    # -------------------------------------------------------------- score ---
    def _stream_frames(self, X, chunk_rows):
        """chunk ของ X ที่จัดคอลัมน์ให้ตรงกับตอน fit แล้ว — ไม่ collect ทั้งชุด"""
        if self.background_policy != "frozen":
            raise ValueError(
                "chunk_rows ใช้กับ background_policy='frozen' เท่านั้น — 'refresh' "
                "ประเมินตัวหารใหม่จาก batch ทั้งก้อนที่กำลัง score (transductive) "
                "จึงต้องเห็นทุกแถวพร้อมกัน"
            )
        for frame in self.model_._iter_frames(X, chunk_rows):
            yield to_polars(frame, columns=self.feature_names_in_)

    def decision_function(self, X, chunk_rows: int | None = None) -> np.ndarray:
        """raw LeadX score (lift scale) — use this for cascades / ensembling

        ``chunk_rows`` จำกัด peak memory โดยประมวลผลทีละ chunk (ผลเท่าเดิม);
        ถ้า population ใหญ่จนถือ score ทั้งชุดไม่ไหว ใช้ ``review_queue(...,
        chunk_rows=...)`` ซึ่งเก็บแค่ top-k
        """
        if not hasattr(self, "model_"):
            raise RuntimeError("ต้อง fit ก่อน")
        if chunk_rows is not None:
            parts = [self.model_.score_samples(f)
                     for f in self._stream_frames(X, chunk_rows)]
            return np.concatenate(parts) if parts else np.empty(0)
        Xp = to_polars(X, columns=self.feature_names_in_)
        if self.background_policy == "frozen":
            return self.model_.score_samples(Xp)
        # refresh: ประเมินตัวหารใหม่จาก batch ที่กำลัง score (transductive)
        codes = self.model_.transform_codes(Xp)
        cfg = self.model_.config_
        sc = SeedScorer(mode=self.model_.mode, m=cfg["m"], weights=cfg["weights"],
                        n_bits=cfg["n_bits"], score_type=cfg["score_type"],
                        alpha=cfg["alpha"], agg=cfg["agg"],
                        hier_depths=self.model_.scorer_.hier_depths,
                        multi_probe=(self.model_.minhash_ is None))
        sc.fit(self._seed_codes, np.ones(len(self._seed_codes)),
               background_codes=codes)
        return sc.score(codes)

    def predict_proba(self, X) -> np.ndarray:
        """(n, 2). Column 1 is a **ranking** score in [0, 1] unless ``prior`` is
        set, in which case it is ``clip(prior * lift, 0, 1)`` under SCAR."""
        s = self.decision_function(X)
        if self.prior is not None:
            p = np.clip(float(self.prior) * s, 0.0, 1.0)
        else:
            span = max(self._ref_hi - self._ref_lo, 1e-12)
            p = np.clip((s - self._ref_lo) / span, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        """flag the ``contamination`` fraction, using the fit-time threshold"""
        return (self.decision_function(X) >= self._threshold).astype(int)

    def review_queue(self, X, k=None, frac=None,
                     chunk_rows: int | None = None) -> np.ndarray:
        """indices of the top-k (or top-``frac``) rows — the operational output

        Parameters
        ----------
        chunk_rows : int | None
            ถ้าระบุ จะ stream ทีละ chunk และเก็บแค่ top-k ระหว่างทาง (memory
            O(k + chunk) ไม่ใช่ O(n)) — ทำให้ shortlist ของ cascade ใช้ได้กับ
            ประชากรที่ใหญ่กว่า RAM เช่น ``pl.scan_parquet("week=*/*.parquet")``
            ต้องใช้ ``background_policy="frozen"`` (ค่า default)
        """
        if (k is None) == (frac is None):
            raise ValueError("ระบุ k หรือ frac อย่างใดอย่างหนึ่ง")
        if chunk_rows is None:
            s = self.decision_function(X)
            kk = int(k) if k is not None else max(1, int(np.ceil(len(s) * float(frac))))
            return np.argsort(-s, kind="stable")[: min(kk, len(s))]
        if k is not None:
            kk = int(k)
        else:
            n = self._row_count(X)
            if n is None:
                raise ValueError(
                    "frac ต้องรู้จำนวนแถวล่วงหน้า — source นี้นับไม่ได้โดยไม่อ่านทั้งชุด "
                    "ให้ระบุ k (จำนวนที่ทีมรีวิวไหว) แทน"
                )
            kk = max(1, int(np.ceil(n * float(frac))))
        idx, _ = self.model_.top_k(self._stream_frames(X, chunk_rows), kk,
                                   chunk_rows=None)
        return idx

    @staticmethod
    def _row_count(X) -> int | None:
        """จำนวนแถวถ้ารู้ได้ถูก (parquet เก็บไว้ใน metadata) — None ถ้าต้องอ่านทั้งชุด"""
        if isinstance(X, pl.LazyFrame):
            return int(X.select(pl.len()).collect().item())
        if isinstance(X, pl.DataFrame):
            return X.height
        return len(X) if hasattr(X, "__len__") else None

    # --------------------------------------------------------- importances ---
    @property
    def feature_importances_(self) -> np.ndarray:
        """per input column, aggregated from the encoded dimensions.
        NaN for MinHash-routed categorical columns (they bypass the weighting)."""
        if not hasattr(self, "model_"):
            raise RuntimeError("ต้อง fit ก่อน")
        m = self.model_
        w = getattr(self, "_imp_dims", None)
        out = np.full(self.n_features_in_, np.nan)
        if w is None:
            return out
        enc, j = m.encoder_, 0
        span: dict[str, tuple[int, int]] = {}
        for c in enc.numeric_cols_:
            width = 1 + int(enc.stats_[c]["has_missing"])
            span[c], j = (j, j + width), j + width
        for c in enc.cat_cols_:
            cats = enc.cat_maps_[c]
            width = len(cats) if cats is not None else enc.hash_dims
            span[c], j = (j, j + width), j + width
        for i, name in enumerate(self.feature_names_in_):
            if name in span:
                lo, hi = span[name]
                out[i] = float(np.max(w[lo:hi]))
        total = np.nansum(out)
        if total > 0:
            out = out / total
        return out

    # ---------------------------------------------------------- persistence ---
    def save_model(self, path) -> None:
        path = Path(path)
        self.model_.save(path / "leadx")
        import sklearn
        from . import __version__ as _leadx_version
        meta = {"params": self.get_params(),
                "feature_names_in": self.feature_names_in_,
                "n_features_in": self.n_features_in_,
                "best_params": self.best_params_,
                "best_score": self.best_score_,
                "evals_result": self.evals_result_,
                "eval_metric": getattr(self, "eval_metric_", None),
                "ref_lo": self._ref_lo, "ref_hi": self._ref_hi,
                "threshold": self._threshold,
                "imp_dims": np.asarray(self._imp_dims).tolist(),
                "leadx_version": _leadx_version,
                "numpy_version": np.__version__,
                "polars_version": pl.__version__,
                "sklearn_version": sklearn.__version__}
        (path / "wrapper.json").write_text(json.dumps(meta))
        if self.background_policy == "refresh":
            np.save(path / "seed_codes.npy", self._seed_codes)

    @classmethod
    def load_model(cls, path) -> "LeadXClassifier":
        path = Path(path)
        meta = json.loads((path / "wrapper.json").read_text())
        obj = cls(**meta["params"])
        obj.model_ = LeadX.load(path / "leadx")
        obj.feature_names_in_ = meta["feature_names_in"]
        obj.n_features_in_ = meta.get("n_features_in", len(obj.feature_names_in_))
        obj.classes_ = np.array([0, 1])
        obj.best_params_ = meta["best_params"]
        obj.evals_result_ = meta.get("evals_result", [])
        obj.best_score_ = meta.get("best_score")
        obj.eval_metric_ = meta.get("eval_metric")
        obj.saved_versions_ = {k: meta.get(k) for k in
                               ("leadx_version", "numpy_version",
                                "polars_version", "sklearn_version")}
        obj._ref_lo, obj._ref_hi = meta["ref_lo"], meta["ref_hi"]
        obj._threshold = meta["threshold"]
        obj._imp_dims = np.asarray(meta.get("imp_dims", []), dtype=np.float64)
        seeds = path / "seed_codes.npy"
        if seeds.exists():
            obj._seed_codes = np.load(seeds)
        return obj
