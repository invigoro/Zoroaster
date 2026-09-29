"""Build the Stage 1 full-day evaluation set: every page that exists on each
of a sample of test days, with the same point-in-time features as the panel.

The panel keeps every positive day but samples negative days. That's right
for training, but it makes per-day ranking metrics ("of the pages ranked
highest for tomorrow, how many get a kept edit?") awkward to estimate.
Scoring every existing page on a handful of days makes them exact. The days
are every 13th day of the test window (`src.stage1.splits`), so weekdays
are covered evenly.

Streams the labels file one page at a time, and reads the co-burst table
and panel metadata written by `build_test_features.py`. Writes
`data/processed/simplewiki_test_stage1_eval_days.parquet` with the panel's
schema. `sample_weight` is 1, since nothing is sampled.

Usage:
    python scripts/build_test_eval_days.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq

from scripts.build_test_features import CO_BURST_PATH, INPUT_PATH, PANEL_META_PATH, ROW_GROUP_SIZE, iter_pages
from src.features.activity import PageActivity
from src.features.bursts import CoBurstCounter
from src.parquet_io import RowGroupWriter
from src.stage1.panel import PANEL_SCHEMA, add_co_burst, panel_row
from src.stage1.splits import EVAL_DAY_STRIDE, make_splits

OUTPUT_PATH = Path("data/processed/simplewiki_test_stage1_eval_days.parquet")


def main() -> int:
    start_time = time.monotonic()
    labels_final_through = date.fromisoformat(json.loads(PANEL_META_PATH.read_text())["labels_final_through"])
    last_final_day = labels_final_through.toordinal()
    eval_days = [d.toordinal() for d in make_splits(labels_final_through).test.days(EVAL_DAY_STRIDE)]
    print(f"{len(eval_days)} eval days: {date.fromordinal(eval_days[0])} .. {date.fromordinal(eval_days[-1])}")

    co_burst_table = pq.read_table(CO_BURST_PATH)
    co_burst = CoBurstCounter.from_counts(
        zip(
            (d.toordinal() for d in co_burst_table["date"].to_pylist()),
            co_burst_table["pages_bursting"].to_pylist(),
            co_burst_table["pages_bursting_multi_editor"].to_pylist(),
        )
    )

    positives: Counter[int] = Counter()
    with RowGroupWriter(OUTPUT_PATH, PANEL_SCHEMA, ROW_GROUP_SIZE) as out:
        for pages, revisions in enumerate(iter_pages(INPUT_PATH), start=1):
            page = PageActivity(revisions)
            for day in eval_days:
                if day > page.first_day:  # the page exists at prediction time
                    row = add_co_burst(panel_row(page, day, 1.0, last_final_day), co_burst, day)
                    positives[day] += row["y"]
                    out.append(row)
            if pages % 100_000 == 0:
                print(f"  {pages:,} pages ({time.monotonic() - start_time:,.0f}s)")

    per_day = [positives[d] for d in eval_days]
    print(
        f"Wrote {out.rows_written:,} rows ({sum(per_day):,} positive; {min(per_day)}-{max(per_day)} "
        f"per day) -> {OUTPUT_PATH} in {time.monotonic() - start_time:,.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
