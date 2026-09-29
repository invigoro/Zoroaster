"""Stage 1 feature sets and model inputs.

The three nested feature sets are the project's core ablation. Each asks
whether a signal adds anything beyond a page's own editing habits:

- `habits`: the page's recent and long-run activity, plus calendar and
  site-wide context. `site_edits_1d` controls for how busy the whole wiki
  was yesterday, so co-burst gets credit only for more than that.
- `habits+burst`: adds the page's own causal burst features.
- `habits+burst+co_burst`: adds how many *other* pages were bursting.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from src.stage1.panel import CO_BURST_COLUMNS

HABIT_FEATURES = (
    "page_age_days",
    "days_since_last_edit",
    "edits_1d",
    "edits_7d",
    "edits_30d",
    "edits_365d",
    "editors_1d",
    "editors_7d",
    "reverted_edits_7d",
    "reverts_7d",
    "bot_edits_30d",
)
CONTEXT_FEATURES = ("day_of_week", "site_edits_1d")
BURST_FEATURES = ("burst_z_1d", "is_burst_1d", "burst_days_30d")

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "habits": HABIT_FEATURES + CONTEXT_FEATURES,
    "habits+burst": HABIT_FEATURES + CONTEXT_FEATURES + BURST_FEATURES,
    "habits+burst+co_burst": HABIT_FEATURES + CONTEXT_FEATURES + BURST_FEATURES + CO_BURST_COLUMNS,
}
CATEGORICAL_FEATURES = ("day_of_week",)


def site_edit_totals(daily: pa.Table) -> dict[int, int]:
    """{day as days-since-epoch: all pages' `edits` that day} from the daily activity table."""
    totals = daily.group_by("date").aggregate([("edits", "sum")])
    days = pc.cast(totals["date"], pa.int32()).to_pylist()
    return dict(zip(days, totals["edits_sum"].to_pylist()))


def add_context_columns(table: pa.Table, site_totals: Mapping[int, int]) -> pa.Table:
    """Add `day_of_week` (of D, Monday = 0) and `site_edits_1d` (all pages'
    `edits` on D-1, known by the end of D-1)."""
    epoch_days = pc.cast(table["date"], pa.int32()).to_numpy()
    lo, hi = min(site_totals), max(site_totals)
    dense = np.zeros(hi - lo + 1, dtype=np.int64)
    for day, total in site_totals.items():
        dense[day - lo] = total
    prev = epoch_days - 1 - lo
    in_range = (prev >= 0) & (prev < len(dense))
    site = np.where(in_range, dense[np.clip(prev, 0, len(dense) - 1)], 0)
    table = table.append_column("day_of_week", pc.day_of_week(table["date"]))
    return table.append_column("site_edits_1d", pa.array(site, pa.int64()))


def feature_matrix(table: pa.Table, columns: Sequence[str]) -> np.ndarray:
    """float32 matrix of `columns`; nulls become NaN, booleans 0/1."""
    out = np.empty((table.num_rows, len(columns)), dtype=np.float32)
    for j, name in enumerate(columns):
        column = table[name]
        if pa.types.is_boolean(column.type):
            column = pc.cast(column, pa.int8())
        out[:, j] = pc.cast(column, pa.float32()).to_numpy(zero_copy_only=False)
    return out
