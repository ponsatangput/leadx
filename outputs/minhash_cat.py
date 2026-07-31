"""MinHash hashing for fixed-schema categorical columns (v0.2 prototype).

Motivation: the current FeatureEncoder turns categoricals into one-hot (<=30) or
feature-hashing (64d) floats, then random hyperplanes score *cosine* locality in
that sparse space. High-cardinality / ID-like columns collide or get diluted.

MinHash instead gives **Jaccard locality** over the set of `{column=value}`
tokens per row: two rows collide in proportion to the fraction of categorical
columns on which they agree. Arbitrary cardinality, no one-hot blow-up, no
z-scoring of ID-like integers.

Bit construction (per table k, per bit j):
  1. permutation hash of each token: perm = base_hash * a_kj + b_kj  (uint64,
     wrapping — multiplicative hashing, non-monotone so argmin reorders)
  2. minhash = the column whose token has the smallest perm value (the "winner")
  3. bit = high bit of (winner_token_hash * c_kj)  -> {0,1}
Two rows agree on bit j with probability 0.5*(1 + Jaccard) — monotone in Jaccard,
and bits are independent so trailing bits can be dropped (prefix-mask / nesting).

y-independent (random seeds only) -> composes with the reusable backbone.

Run self-test:  uv run python outputs/minhash_cat.py
"""
from __future__ import annotations

import numpy as np
import polars as pl

_HI = np.uint64(63)
_ONE = np.uint64(1)


class MinHashCat:
    def __init__(self, cat_cols, n_tables: int, n_bits: int, random_state: int = 0):
        if n_bits > 32:
            raise ValueError("n_bits (categorical) <= 32 ต่อ table")
        self.cat_cols = list(cat_cols)
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.random_state = random_state
        rng = np.random.default_rng(random_state)
        # per (table, bit): odd multiplier a, offset b (permutation) + odd mult c (bit)
        shape = (n_tables, n_bits)
        self.a_ = (rng.integers(0, 2 ** 63, shape, dtype=np.uint64) | np.uint64(1))
        self.b_ = rng.integers(0, 2 ** 63, shape, dtype=np.uint64)
        self.c_ = (rng.integers(0, 2 ** 63, shape, dtype=np.uint64) | np.uint64(1))

    def _base(self, df: pl.DataFrame) -> np.ndarray:
        """(n, n_cols) uint64 — stable per-token hash of each categorical value"""
        cols = []
        for ci, c in enumerate(self.cat_cols):
            h = df[c].cast(pl.Utf8).fill_null("\x00NA").hash(seed=1_000 + ci)
            cols.append(h.to_numpy().astype(np.uint64))
        return np.stack(cols, axis=1)

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        """(n, n_tables) uint64 — categorical bits in the low `n_bits` positions"""
        base = self._base(df)                       # (n, C)
        n = base.shape[0]
        codes = np.zeros((n, self.n_tables), dtype=np.uint64)
        with np.errstate(over="ignore"):            # uint64 multiply wraps by design
            for k in range(self.n_tables):
                code = np.zeros(n, dtype=np.uint64)
                for j in range(self.n_bits):
                    perm = base * self.a_[k, j] + self.b_[k, j]   # (n, C) hash
                    win = np.argmin(perm, axis=1)                 # (n,) winning col
                    win_tok = base[np.arange(n), win]             # (n,) winner hash
                    bit = ((win_tok * self.c_[k, j]) >> _HI) & _ONE
                    code |= bit << np.uint64(j)
                codes[:, k] = code
        return codes


# ---------------------------------------------------------------- self-test ---
def _synthetic(n_groups=40, per_group=200, n_cols=8, cardinality=12, seed=0):
    """rows in the same group agree on most categorical columns (high Jaccard);
    across groups they mostly disagree — lets us check locality is recovered."""
    rng = np.random.default_rng(seed)
    proto = rng.integers(0, cardinality, size=(n_groups, n_cols))
    rows, grp = [], []
    for g in range(n_groups):
        block = np.tile(proto[g], (per_group, 1))
        flip = rng.random((per_group, n_cols)) < 0.15          # 15% noise per cell
        block[flip] = rng.integers(0, cardinality, size=flip.sum())
        rows.append(block)
        grp.append(np.full(per_group, g))
    X = np.vstack(rows)
    df = pl.DataFrame({f"c{i}": X[:, i].astype(str) for i in range(n_cols)})
    return df, np.concatenate(grp)


def _main():
    df, grp = _synthetic()
    mh = MinHashCat([f"c{i}" for i in range(8)], n_tables=40, n_bits=8, random_state=0)
    codes = mh.transform(df)
    n = df.height
    rng = np.random.default_rng(1)
    same = rng.integers(0, n, (5000, 2))
    same = same[grp[same[:, 0]] == grp[same[:, 1]]]
    diff = rng.integers(0, n, (5000, 2))
    diff = diff[grp[diff[:, 0]] != grp[diff[:, 1]]]

    def collide_rate(pairs):
        return float((codes[pairs[:, 0]] == codes[pairs[:, 1]]).mean())

    within, across = collide_rate(same), collide_rate(diff)
    print(f"per-table bucket collision — within-group {within:.3f} vs "
          f"across-group {across:.3f}")
    assert within > 5 * across, "MinHash ไม่แยก within/across — ผิด"

    # bit-agreement should track Jaccard: within-group Jaccard high -> agree ~near 1
    def bit_agree(pairs):
        xa, xb = codes[pairs[:, 0], 0], codes[pairs[:, 1], 0]
        bits = np.array([((xa >> j) & 1) == ((xb >> j) & 1) for j in range(8)])
        return float(bits.mean())

    print(f"bit-agreement (table 0) — within {bit_agree(same):.3f} vs "
          f"across {bit_agree(diff):.3f}  (theory: 0.5*(1+J))")
    assert bit_agree(same) > bit_agree(diff) > 0.45
    # determinism
    assert np.array_equal(codes, mh.transform(df)), "ไม่ deterministic"
    print("SELF-TEST OK")


if __name__ == "__main__":
    _main()
