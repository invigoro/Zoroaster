"""Heuristic Stage 1 baselines: rank pages by simple activity statistics."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def lexicographic_score(*keys: np.ndarray) -> np.ndarray:
    """Dense ranks that order rows by `keys`, first key most significant.

    Rows with equal keys get equal scores, so ties stay ties (the metrics
    break them randomly) instead of being settled by row order.
    """
    stacked = np.stack([np.asarray(k, dtype=np.float64) for k in keys], axis=1)
    _, inverse = np.unique(stacked, axis=0, return_inverse=True)
    return inverse.ravel().astype(np.float64)


def _column(table: pa.Table, name: str) -> np.ndarray:
    return pc.cast(table[name], pa.float64()).to_numpy(zero_copy_only=False)


def _recency(table: pa.Table) -> np.ndarray:
    # Fewer days since the last edit ranks higher; never-edited pages rank last.
    return np.nan_to_num(-_column(table, "days_since_last_edit"), nan=-1e9)


BASELINES: dict[str, Callable[[pa.Table], np.ndarray]] = {
    "edits yesterday": lambda t: lexicographic_score(
        _column(t, "edits_1d"), _column(t, "edits_30d"), _column(t, "edits_365d")
    ),
    "edits last 30 days": lambda t: lexicographic_score(_column(t, "edits_30d"), _column(t, "edits_365d")),
    "most recent edit": lambda t: lexicographic_score(_recency(t), _column(t, "edits_365d")),
    "edits last year": lambda t: lexicographic_score(_column(t, "edits_365d")),
    # Persistence, for the burst target: the pages bursting hardest yesterday.
    "burst z yesterday": lambda t: lexicographic_score(_column(t, "burst_z_1d"), _column(t, "edits_1d")),
}
