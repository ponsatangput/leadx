"""Input adapter — accept the frame types people actually have, work in Polars.

LeadX's internals are Polars (group-by/join on Arrow memory is what lets it hold
10M rows in ~12GB). Rather than support every frame natively, convert once at the
API boundary and keep the Polars fast path everywhere inside.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def to_polars(X, columns=None) -> pl.DataFrame:
    """polars / pandas / pyarrow / numpy / dict / LazyFrame -> polars.DataFrame

    Parameters
    ----------
    X : frame-like
        polars DataFrame or LazyFrame, pandas DataFrame, pyarrow Table,
        2-D numpy array, or dict of 1-D arrays
    columns : sequence[str] | None
        column names for array input (default ``f0..f{D-1}``); for named frames
        it is validated against the actual columns
    """
    if isinstance(X, pl.DataFrame):
        out = X
    elif isinstance(X, pl.LazyFrame):
        out = X.collect()
    elif isinstance(X, dict):
        out = pl.DataFrame(X)
    else:
        module = type(X).__module__.split(".")[0]
        if module == "pandas":
            out = pl.from_pandas(X)
        elif module == "pyarrow":
            out = pl.from_arrow(X)
            if isinstance(out, pl.Series):          # ChunkedArray -> Series
                out = out.to_frame()
        else:
            arr = np.asarray(X)
            if arr.ndim != 2:
                raise ValueError(
                    f"array input ต้องเป็น 2 มิติ (n_rows, n_features) — ได้ shape={arr.shape}"
                )
            names = (list(columns) if columns is not None
                     else [f"f{i}" for i in range(arr.shape[1])])
            if len(names) != arr.shape[1]:
                raise ValueError(
                    f"columns มี {len(names)} ชื่อ แต่ array มี {arr.shape[1]} คอลัมน์")
            out = pl.DataFrame(arr, schema=names)
    if columns is not None and list(out.columns) != list(columns):
        missing = [c for c in columns if c not in out.columns]
        if missing:
            raise ValueError(f"ขาดคอลัมน์ที่เคยเห็นตอน fit: {missing}")
        out = out.select(list(columns))             # จัดลำดับให้ตรงกับตอน fit
    return out
