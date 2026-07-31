"""MinHashCat — Jaccard-locality hashing for categorical columns.

Turns each row's categorical values into a set of `{column=value}` tokens and
emits K tables of bucket codes whose collision probability tracks the Jaccard
overlap (the fraction of categorical columns two rows agree on). Handles
arbitrary cardinality with no one-hot blow-up, and ID-like integer columns are
treated as tokens instead of being z-scored as ordinals.

Measured on IEEE fraud (PU, 1:1,000, 5 paired draws): appended MinHash tables
beat rpforest-on-one-hot categorical tables by +56% PR-AUC / +37% Recall@0.5%
(5/5 draws), and recover value that one-hot *destroys* (one-hot lost to
numeric-only 0/5) — see outputs/CATEGORICAL_MINHASH_REPORT.md.

Per table k / bit j the bit is `high_bit(minhash_winner_token * c_kj)`, so two
rows agree on a bit with probability 0.5*(1 + Jaccard) — monotone in Jaccard,
independent bits (trailing bits droppable), y-independent (seeds only → composes
with the reusable backbone). Produces bucket codes exactly like HashEnsemble, so
SeedScorer / mask_codes / backbone consume them unchanged.
"""
from __future__ import annotations

import numpy as np
import polars as pl

_HI = np.uint64(63)
_ONE = np.uint64(1)


class MinHashCat:
    """MinHash bucket-code generator for a fixed set of categorical columns.

    Parameters
    ----------
    cat_cols : sequence[str]
        columns to hash as categorical tokens (cast to string; ID-like integers
        welcome). Order is fixed at construction.
    n_tables : int
        number of hash tables (each an independent code column)
    n_bits : int
        bits per table (<= 32); collision resolution of the categorical code
    random_state : int
    """

    def __init__(self, cat_cols, n_tables: int, n_bits: int, random_state: int = 0):
        if n_bits > 32:
            raise ValueError(f"n_bits (categorical) <= 32 (ได้ {n_bits})")
        if not len(cat_cols):
            raise ValueError("MinHashCat ต้องการ categorical column อย่างน้อย 1")
        self.cat_cols = list(cat_cols)
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.random_state = random_state
        rng = np.random.default_rng(random_state)
        shape = (n_tables, n_bits)
        self.a_ = rng.integers(0, 2 ** 63, shape, dtype=np.uint64) | np.uint64(1)
        self.b_ = rng.integers(0, 2 ** 63, shape, dtype=np.uint64)
        self.c_ = rng.integers(0, 2 ** 63, shape, dtype=np.uint64) | np.uint64(1)

    def _base(self, df: pl.DataFrame) -> np.ndarray:
        cols = []
        for ci, c in enumerate(self.cat_cols):
            h = df[c].cast(pl.Utf8).fill_null("\x00NA").hash(seed=1_000 + ci)
            cols.append(h.to_numpy().astype(np.uint64))
        return np.stack(cols, axis=1)

    def transform(self, df: pl.DataFrame, chunk_size: int = 1_000_000) -> np.ndarray:
        """(n, n_tables) uint64 — categorical bits in the low `n_bits` positions"""
        missing = [c for c in self.cat_cols if c not in df.columns]
        if missing:
            raise ValueError(f"ขาด categorical columns: {missing}")
        n = df.height
        out = np.empty((n, self.n_tables), dtype=np.uint64)
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            base = self._base(df[start:stop])            # (m, C)
            m = base.shape[0]
            idx = np.arange(m)
            with np.errstate(over="ignore"):             # uint64 multiply wraps by design
                for k in range(self.n_tables):
                    code = np.zeros(m, dtype=np.uint64)
                    for j in range(self.n_bits):
                        perm = base * self.a_[k, j] + self.b_[k, j]
                        win_tok = base[idx, np.argmin(perm, axis=1)]
                        bit = ((win_tok * self.c_[k, j]) >> _HI) & _ONE
                        code |= bit << np.uint64(j)
                    out[start:stop, k] = code
        return out

    def to_dict(self) -> dict:
        return {
            "cat_cols": self.cat_cols, "n_tables": self.n_tables,
            "n_bits": self.n_bits, "random_state": self.random_state,
            "a": self.a_.tolist(), "b": self.b_.tolist(), "c": self.c_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MinHashCat":
        mh = cls(d["cat_cols"], d["n_tables"], d["n_bits"], d["random_state"])
        mh.a_ = np.asarray(d["a"], dtype=np.uint64)
        mh.b_ = np.asarray(d["b"], dtype=np.uint64)
        mh.c_ = np.asarray(d["c"], dtype=np.uint64)
        return mh
