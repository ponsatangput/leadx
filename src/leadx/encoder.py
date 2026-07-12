"""FeatureEncoder — แปลง Polars DataFrame (numeric + categorical) เป็น float32 matrix"""
import zlib

import numpy as np
import polars as pl

_MISSING = "__missing__"


class FeatureEncoder:
    """แปลง Polars DataFrame → float32 matrix พร้อมสำหรับ random-hyperplane hashing

    Numeric: clip outliers → standardize (z-score) → fill missing ด้วย median พร้อม
    missing-indicator column. Categorical เลือกอัตโนมัติตาม cardinality — one-hot
    (cardinality ต่ำ, คูณ 1/√k กันน้ำหนักเกิน) หรือ feature hashing (cardinality สูง).
    Standardization จำเป็นเพราะ hyperplane ผ่าน origin; NaN ถูกจัดการเหมือน null.

    Parameters
    ----------
    categorical : {"auto", "one_hot"}
        "auto" เลือก one-hot/hashing ตาม one_hot_max; "one_hot" บังคับ one-hot
    one_hot_max : int
        cardinality สูงสุดที่ยังใช้ one-hot (เกินกว่านี้ → feature hashing)
    hash_dims : int
        จำนวนมิติของ feature hashing (คงที่ไม่ว่า cardinality เท่าไหร่)
    clip_percentiles : tuple(float, float)
        percentile ล่าง/บนสำหรับ clip outliers ก่อน standardize
    """

    def __init__(self, categorical="auto", one_hot_max=30, hash_dims=64,
                 clip_percentiles=(0.5, 99.5)):
        self.categorical = categorical
        self.one_hot_max = one_hot_max
        self.hash_dims = hash_dims
        self.clip_percentiles = clip_percentiles

    def fit(self, df: pl.DataFrame) -> "FeatureEncoder":
        self.columns_ = list(df.columns)
        self.numeric_cols_ = [c for c in df.columns if df[c].dtype.is_numeric()]
        self.cat_cols_ = [c for c in df.columns if c not in self.numeric_cols_]
        lo_p, hi_p = self.clip_percentiles
        self.stats_ = {}
        for c in self.numeric_cols_:
            # polars แยก NaN (IEEE) จาก null (missing) — numpy/pandas รวมเป็นอันเดียว
            # normalize NaN→null ที่ต้นทาง เพื่อให้ logic missing ครอบคลุมทั้งสองกรณี
            s = df[c].cast(pl.Float64).fill_nan(None)
            lo = float(s.quantile(lo_p / 100.0, "nearest"))
            hi = float(s.quantile(hi_p / 100.0, "nearest"))
            clipped = s.clip(lo, hi)
            std = clipped.std()
            self.stats_[c] = {
                "lo": lo, "hi": hi,
                "mean": float(clipped.mean()),
                "std": float(std) if std and std > 0 else 1.0,
                "median": float(s.median()),
                "has_missing": bool(s.is_null().any()),
            }
        self.cat_maps_ = {}
        for c in self.cat_cols_:
            s = df[c].cast(pl.Utf8).fill_null(_MISSING)
            if self.categorical == "one_hot" or (
                self.categorical == "auto" and s.n_unique() <= self.one_hot_max
            ):
                self.cat_maps_[c] = sorted(s.unique().to_list())
            else:
                self.cat_maps_[c] = None  # feature hashing — stateless
        self._build_cat_mask()
        return self

    def _build_cat_mask(self) -> None:
        """boolean mask ต่อ output column: True = มาจาก categorical block
        (ใช้โดย ProjectionBank จัดสรร bits แยกชนิด — hyperplane bits ปกติแทบมองไม่เห็น
        one-hot เพราะ contribution เล็กเทียบมิติ numeric รวมกัน)"""
        dims: list[bool] = []
        for c in self.numeric_cols_:
            dims.append(False)
            if self.stats_[c]["has_missing"]:
                dims.append(False)
        for c in self.cat_cols_:
            k = len(self.cat_maps_[c]) if self.cat_maps_[c] is not None else self.hash_dims
            dims.extend([True] * k)
        self.cat_mask_ = np.array(dims, dtype=bool)

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        missing = set(self.columns_) - set(df.columns)
        extra = set(df.columns) - set(self.columns_)
        if missing or extra:
            raise ValueError(
                f"columns mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        blocks = []
        for c in self.numeric_cols_:
            st = self.stats_[c]
            s = df[c].cast(pl.Float64).fill_nan(None)  # NaN→null (ดู fit)
            is_miss = s.is_null().to_numpy()
            x = s.fill_null(st["median"]).clip(st["lo"], st["hi"]).to_numpy()
            blocks.append(((x - st["mean"]) / st["std"])[:, None])
            if st["has_missing"]:
                blocks.append(is_miss.astype(np.float64)[:, None])
        for c in self.cat_cols_:
            s = df[c].cast(pl.Utf8).fill_null(_MISSING)
            cats = self.cat_maps_[c]
            if cats is not None:
                blocks.append(self._one_hot(s.to_numpy(), cats))
            else:
                blocks.append(self._hashed(s, c))
        return np.hstack(blocks).astype(np.float32)

    def fit_transform(self, df: pl.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @staticmethod
    def _one_hot(vals: np.ndarray, cats: list) -> np.ndarray:
        k = len(cats)
        cats_arr = np.asarray(cats)
        pos = np.searchsorted(cats_arr, vals)
        pos_c = np.clip(pos, 0, k - 1)
        known = cats_arr[pos_c] == vals  # category ที่ไม่เคยเห็น → แถว all-zero
        block = np.zeros((len(vals), k))
        block[np.nonzero(known)[0], pos_c[known]] = 1.0 / np.sqrt(k)
        return block

    def to_dict(self) -> dict:
        return {
            "categorical": self.categorical,
            "one_hot_max": self.one_hot_max,
            "hash_dims": self.hash_dims,
            "clip_percentiles": list(self.clip_percentiles),
            "columns": self.columns_,
            "numeric_cols": self.numeric_cols_,
            "cat_cols": self.cat_cols_,
            "stats": self.stats_,
            "cat_maps": self.cat_maps_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureEncoder":
        enc = cls(
            categorical=d["categorical"], one_hot_max=d["one_hot_max"],
            hash_dims=d["hash_dims"], clip_percentiles=tuple(d["clip_percentiles"]),
        )
        enc.columns_ = d["columns"]
        enc.numeric_cols_ = d["numeric_cols"]
        enc.cat_cols_ = d["cat_cols"]
        enc.stats_ = d["stats"]
        enc.cat_maps_ = d["cat_maps"]
        enc._build_cat_mask()
        return enc

    def _hashed(self, s: pl.Series, col: str) -> np.ndarray:
        # polars hash เสถียรภายใน polars version เดียวกัน — caveat ระบุไว้ที่ save/load
        seed = zlib.crc32(col.encode()) & 0xFFFF
        h = s.hash(seed=seed).to_numpy()
        idx = (h % np.uint64(self.hash_dims)).astype(np.int64)
        sign = ((h >> np.uint64(32)) & np.uint64(1)).astype(np.float64) * 2.0 - 1.0
        block = np.zeros((len(idx), self.hash_dims))
        block[np.arange(len(idx)), idx] = sign / np.sqrt(self.hash_dims)
        return block
