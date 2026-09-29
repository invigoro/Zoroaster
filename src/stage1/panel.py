"""The Stage 1 row contract, shared by the panel and eval-set builders.

One row per (page, prediction day D): point-in-time features that read only
days before D (see `src.features.activity`), the co-burst context for D-1,
and the label y = "the page got a kept edit on day D".
"""

from __future__ import annotations

from datetime import date

import pyarrow as pa

from src.features.activity import FEATURE_NAMES, PageActivity
from src.features.bursts import CoBurstCounter

# Label-side columns: derived from day D itself or from later revisions, so
# they must never be used as model inputs.
LABEL_COLUMNS = ("y", "kept_edits", "sample_weight", "label_is_final")
CO_BURST_COLUMNS = ("co_burst_1d", "co_burst_multi_editor_1d")

_FEATURE_TYPES = {
    "days_since_last_edit": pa.int32(),
    "burst_z_1d": pa.float64(),
    "is_burst_1d": pa.bool_(),
}
PANEL_BASE_SCHEMA = pa.schema(
    [
        ("page_id", pa.int64()),
        ("date", pa.date32()),
        ("y", pa.bool_()),
        ("kept_edits", pa.int32()),
        ("sample_weight", pa.float64()),
        ("label_is_final", pa.bool_()),
    ]
    + [(name, _FEATURE_TYPES.get(name, pa.int32())) for name in FEATURE_NAMES]
)
PANEL_SCHEMA = pa.schema(list(PANEL_BASE_SCHEMA) + [(name, pa.int32()) for name in CO_BURST_COLUMNS])


def panel_row(page: PageActivity, day: int, sample_weight: float, last_final_day: int) -> dict:
    """Features and label for `page` on prediction day `day` (a date ordinal)."""
    kept = page.counts["kept_edits"][day]
    row = page.features(day)
    row.update(
        page_id=page.page_id,
        date=date.fromordinal(day),
        y=kept > 0,
        kept_edits=kept,
        sample_weight=sample_weight,
        label_is_final=day <= last_final_day,
    )
    return row


def add_co_burst(row: dict, co_burst: CoBurstCounter, day: int) -> dict:
    """Add the other-pages-bursting-on-D-1 columns to a `panel_row`."""
    row["co_burst_1d"], row["co_burst_multi_editor_1d"] = co_burst.others(
        day - 1, row["is_burst_1d"], row["editors_1d"]
    )
    return row
