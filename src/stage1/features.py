"""Stage 1 feature sets and model inputs.

The feature sets are the project's core ablation. Each extends a parent
(FEATURE_SET_PARENTS) and asks whether a signal adds anything beyond it:

- `habits`: the page's recent and long-run activity, plus calendar and
  site-wide context. `site_edits_1d` controls for how busy the whole wiki
  was yesterday, so cross-page signals get credit only for more than that.
- `habits+burst`: adds the page's own causal burst features.
- `habits+burst+co_burst`: adds how many *other* pages were bursting,
  site-wide.
- `habits+burst+degrees`: adds the page's own in/out link counts, the
  control for the next two.
- `habits+burst+links`: adds how many of the page's *link neighbors* were
  bursting (`src.features.neighbors`), a page-specific cross-page signal.
- `habits+burst+links_excl_mass`: the same, but neighbor bursts ignore
  mass-editing sessions (`src.features.activity.mass_editor_days`).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from src.features.neighbors import DEGREE_FEATURES, NEIGHBOR_BURST_FEATURES, NeighborBursts
from src.stage1.panel import CO_BURST_COLUMNS

HABIT_FEATURES = (
    "page_age_days",
    "page_bytes",
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
EXCL_MASS_SUFFIX = "_excl_mass"
LINK_FEATURES = DEGREE_FEATURES + NEIGHBOR_BURST_FEATURES
LINK_FEATURES_EXCL_MASS = DEGREE_FEATURES + tuple(f + EXCL_MASS_SUFFIX for f in NEIGHBOR_BURST_FEATURES)

_HABITS = HABIT_FEATURES + CONTEXT_FEATURES
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "habits": _HABITS,
    "habits+burst": _HABITS + BURST_FEATURES,
    "habits+burst+co_burst": _HABITS + BURST_FEATURES + CO_BURST_COLUMNS,
    "habits+burst+degrees": _HABITS + BURST_FEATURES + DEGREE_FEATURES,
    "habits+burst+links": _HABITS + BURST_FEATURES + LINK_FEATURES,
    "habits+burst+links_excl_mass": _HABITS + BURST_FEATURES + LINK_FEATURES_EXCL_MASS,
}
# The link sets extend `habits+burst+degrees`, so their comparison isolates the
# neighbor-burst signal from the page's own link counts. Those counts are
# static and measured at the dump date: a size/centrality proxy, not a
# "something is happening" signal, and partly shaped by the edits being
# predicted.
FEATURE_SET_PARENTS = {
    "habits+burst": "habits",
    "habits+burst+co_burst": "habits+burst",
    "habits+burst+degrees": "habits+burst",
    "habits+burst+links": "habits+burst+degrees",
    "habits+burst+links_excl_mass": "habits+burst+degrees",
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


def add_neighbor_columns(table: pa.Table, neighbors: NeighborBursts, suffix: str = "") -> pa.Table:
    """Add link-neighbor burst features for each (page_id, date) row; the
    degree columns are added only once across calls."""
    return add_neighbor_columns_to([table], neighbors, suffix)[0]


def add_neighbor_columns_to(tables: list[pa.Table], neighbors: NeighborBursts, suffix: str = "") -> list[pa.Table]:
    """`add_neighbor_columns` for several tables in one `features` call, which
    builds each count table once instead of once per table."""
    pages = np.concatenate([pc.cast(t["page_id"], pa.int64()).to_numpy() for t in tables])
    days = np.concatenate([pc.cast(t["date"], pa.int32()).to_numpy() for t in tables])
    bounds = np.cumsum([0] + [t.num_rows for t in tables])
    values = neighbors.features(pages, days, suffix)
    out = []
    for i, table in enumerate(tables):
        for name, column in values.items():
            if name not in table.schema.names:
                table = table.append_column(name, pa.array(column[bounds[i] : bounds[i + 1]]))
        out.append(table)
    return out


def feature_matrix(table: pa.Table, columns: Sequence[str]) -> np.ndarray:
    """float32 matrix of `columns`; nulls become NaN, booleans 0/1."""
    out = np.empty((table.num_rows, len(columns)), dtype=np.float32)
    for j, name in enumerate(columns):
        column = table[name]
        if pa.types.is_boolean(column.type):
            column = pc.cast(column, pa.int8())
        out[:, j] = pc.cast(column, pa.float32()).to_numpy(zero_copy_only=False)
    return out
