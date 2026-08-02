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
from sklearn.metrics import average_precision_score

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


def _recall_at(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    k = max(1, int(np.ceil(len(y) * frac)))
    idx = np.argpartition(-s, k - 1)[:k]
    return float(y[idx].sum()) / max(1.0, float(y.sum()))


def _score_metric(y: np.ndarray, s: np.ndarray, metric: str) -> float:
    if metric == "pr_auc":
        return float(average_precision_score(y, s))
    if metric.startswith("recall@"):
        return _recall_at(y, s, float(metric.split("@")[1].rstrip("%")) / 100.0)
    raise ValueError(f"eval_metric ไม่รู้จัก: {metric} (ใช้ 'pr_auc' หรือ 'recall@0.5%')")


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
    best_params_, evals_result_ : populated when ``eval_set`` is used
    feature_importances_ : per input column (NaN for MinHash-routed columns)
    """

    def __init__(self, profile="rare_pu", task="pu", n_bits=None, n_tables=None,
                 engine=None, cat_engine=None, feature_weighting=None,
                 hier_depths=None, score_type=None, agg=None,
                 categorical_columns=None, background_policy="frozen",
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

    def _fit_one(self, X: pl.DataFrame, y: np.ndarray, cfg: dict) -> LeadX:
        model = self._build(cfg)
        if self.task == "pu":
            return model.fit(X.filter(pl.Series(y == 1)), background=X)
        return model.fit(X, y=y)

    # ---------------------------------------------------------------- fit ---
    def fit(self, X, y, eval_set=None, param_grid=None, eval_metric="pr_auc"):
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
        self.feature_names_in_ = list(Xp.columns)
        self.n_features_in_ = Xp.width
        self.classes_ = np.array([0, 1])

        candidates = [self._resolve()]
        if param_grid:
            keys = sorted(param_grid)
            candidates = [self._resolve(dict(zip(keys, combo)))
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
            model = self._fit_one(Xp, y, cfg)
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
    def decision_function(self, X) -> np.ndarray:
        """raw LeadX score (lift scale) — use this for cascades / ensembling"""
        if not hasattr(self, "model_"):
            raise RuntimeError("ต้อง fit ก่อน")
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

    def review_queue(self, X, k=None, frac=None) -> np.ndarray:
        """indices of the top-k (or top-``frac``) rows — the operational output"""
        s = self.decision_function(X)
        if (k is None) == (frac is None):
            raise ValueError("ระบุ k หรือ frac อย่างใดอย่างหนึ่ง")
        kk = int(k) if k is not None else max(1, int(np.ceil(len(s) * float(frac))))
        kk = min(kk, len(s))
        return np.argsort(-s, kind="stable")[:kk]

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
        meta = {"params": self.get_params(),
                "feature_names_in": self.feature_names_in_,
                "best_params": self.best_params_,
                "ref_lo": self._ref_lo, "ref_hi": self._ref_hi,
                "threshold": self._threshold,
                "imp_dims": np.asarray(self._imp_dims).tolist()}
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
        obj.n_features_in_ = len(obj.feature_names_in_)
        obj.classes_ = np.array([0, 1])
        obj.best_params_ = meta["best_params"]
        obj.evals_result_ = []
        obj.best_score_ = None
        obj._ref_lo, obj._ref_hi = meta["ref_lo"], meta["ref_hi"]
        obj._threshold = meta["threshold"]
        obj._imp_dims = np.asarray(meta.get("imp_dims", []), dtype=np.float64)
        seeds = path / "seed_codes.npy"
        if seeds.exists():
            obj._seed_codes = np.load(seeds)
        return obj
