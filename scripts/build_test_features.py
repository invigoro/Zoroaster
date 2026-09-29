"""Build point-in-time Stage 1 features, burst/co-burst signals, and a
stratified page sample from the test revert-label dataset.

Does not download anything: it streams the local
`data/processed/simplewiki_test_revert_labels.parquet` produced by
`build_test_revert_labels.py`, one page at a time, so memory is bounded by
the largest single page history. Outputs, all in `data/processed/`:

- `simplewiki_test_daily_activity.parquet`: one row per page per UTC day
  with any revision: channel counts (see `src.features.activity`), distinct
  editors, and the causal burst z-score/flag.
- `simplewiki_test_co_burst.parquet`: per day, how many pages were bursting,
  and how many of those had at least two distinct editors.
- `simplewiki_test_stage1_panel.parquet`: Stage 1 rows, one per (page,
  prediction day D), with point-in-time features and the label y = "the
  page got a kept edit on day D". Every positive day is kept, plus a uniform
  random sample of negative days across each page's lifetime (most edits
  land on dormant pages, PLAN.md §5); `sample_weight` undoes the sampling.
  `label_is_final` is False within the revert window of the dump date,
  where a later revert could still flip y.
- `simplewiki_test_stage1_panel.json`: panel metadata.
- `simplewiki_test_sample_manifest.json`: stratified page sample for the
  Stage 2 diff fetch (`build_test_diffs.py`).

Usage:
    python scripts/build_test_features.py
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import Counter
from datetime import date
from itertools import groupby
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.features.activity import CHANNELS, FEATURE_NAMES, PageActivity, day_ordinal
from src.features.bursts import CoBurstCounter
from src.ingest.revert_detect import TIME_WINDOW
from src.ingest.sampling import sample_pages, stratify, summarize_pages
from src.parquet_io import RowGroupWriter

INPUT_PATH = Path("data/processed/simplewiki_test_revert_labels.parquet")
DAILY_PATH = Path("data/processed/simplewiki_test_daily_activity.parquet")
CO_BURST_PATH = Path("data/processed/simplewiki_test_co_burst.parquet")
PANEL_PATH = Path("data/processed/simplewiki_test_stage1_panel.parquet")
PANEL_META_PATH = Path("data/processed/simplewiki_test_stage1_panel.json")
MANIFEST_PATH = Path("data/processed/simplewiki_test_sample_manifest.json")

NEGATIVE_SAMPLE_RATE = 0.005
PANEL_SEED = 1234
PER_STRATUM_SAMPLE_SIZE = 25
SAMPLE_SEED = 1234
ROW_GROUP_SIZE = 500_000

INPUT_COLUMNS = [
    "page_id", "page_title", "revision_id", "timestamp", "user_text", "is_anon",
    "is_reverted", "reverted_by_revision_id", "is_revert",
]

DAILY_SCHEMA = pa.schema(
    [("page_id", pa.int64()), ("date", pa.date32())]
    + [(name, pa.int32()) for name in CHANNELS]
    + [("editors", pa.int32()), ("burst_z", pa.float64()), ("is_burst", pa.bool_())]
)
CO_BURST_SCHEMA = pa.schema(
    [
        ("date", pa.date32()),
        ("pages_bursting", pa.int32()),
        ("pages_bursting_multi_editor", pa.int32()),
    ]
)
FEATURE_TYPES = {"days_since_last_edit": pa.int32(), "burst_z_1d": pa.float64(), "is_burst_1d": pa.bool_()}
PANEL_BASE_SCHEMA = pa.schema(
    [
        ("page_id", pa.int64()),
        ("date", pa.date32()),
        ("y", pa.bool_()),
        ("kept_edits", pa.int32()),
        ("sample_weight", pa.float64()),
        ("label_is_final", pa.bool_()),
    ]
    + [(name, FEATURE_TYPES.get(name, pa.int32())) for name in FEATURE_NAMES]
)
PANEL_SCHEMA = PANEL_BASE_SCHEMA.append(pa.field("co_burst_1d", pa.int32())).append(
    pa.field("co_burst_multi_editor_1d", pa.int32())
)


def iter_pages(path: Path) -> Iterator[list[dict]]:
    """Yield one page's labeled revisions at a time (the file is page-contiguous)."""
    batches = pq.ParquetFile(path).iter_batches(columns=INPUT_COLUMNS, batch_size=ROW_GROUP_SIZE)
    rows = (row for batch in batches for row in batch.to_pylist())
    for _, page_rows in groupby(rows, key=lambda r: r["page_id"]):
        yield list(page_rows)


def sample_days(rng: random.Random, start: int, end: int, rate: float) -> Iterator[int]:
    """Days in `start`..`end` inclusive, each kept independently with
    probability `rate` (geometric skipping, so O(days kept), not O(days))."""
    log_skip = math.log1p(-rate)
    day = start - 1
    while True:
        day += 1 + int(math.log(1.0 - rng.random()) / log_skip)
        if day > end:
            return
        yield day


def main() -> int:
    start_time = time.monotonic()
    max_timestamp = pc.max(pq.read_table(INPUT_PATH, columns=["timestamp"])["timestamp"]).as_py()
    last_full_day = day_ordinal(max_timestamp) - 1
    last_final_day = last_full_day - TIME_WINDOW.days
    print(
        f"Data ends {max_timestamp}: panel days run through {date.fromordinal(last_full_day)}, "
        f"labels final through {date.fromordinal(last_final_day)}"
    )

    co_burst = CoBurstCounter()
    summaries: dict[int, dict] = {}
    stats: Counter[str] = Counter()
    panel_tmp = PANEL_PATH.with_name(PANEL_PATH.stem + ".pass1.parquet")

    with RowGroupWriter(DAILY_PATH, DAILY_SCHEMA, ROW_GROUP_SIZE) as daily_out, RowGroupWriter(
        panel_tmp, PANEL_BASE_SCHEMA, ROW_GROUP_SIZE
    ) as panel_out:
        for revisions in iter_pages(INPUT_PATH):
            page = PageActivity(revisions)
            page_id = page.page_id

            for day in page.days:
                row = {name: page.counts[name][day] for name in CHANNELS}
                row.update(
                    page_id=page_id,
                    date=date.fromordinal(day),
                    editors=page.editor_count(day),
                    burst_z=page.burst_zscore(day),
                    is_burst=day in page.burst_days,
                )
                daily_out.append(row)
            for day in page.burst_days:
                co_burst.add(day, page.editor_count(day))

            # Panel: prediction days after the page's first day, up to the
            # last complete day of the dump.
            first, last = page.first_day + 1, last_full_day
            kept = page.counts["kept_edits"]
            positives = [d for d in kept if first <= d <= last]
            rng = random.Random(f"{PANEL_SEED}:{page_id}")
            negatives = [d for d in sample_days(rng, first, last, NEGATIVE_SAMPLE_RATE) if d not in kept]
            for day in sorted(positives + negatives):
                row = page.features(day)
                row.update(
                    page_id=page_id,
                    date=date.fromordinal(day),
                    y=day in kept,
                    kept_edits=kept[day],
                    sample_weight=1.0 if day in kept else 1.0 / NEGATIVE_SAMPLE_RATE,
                    label_is_final=day <= last_final_day,
                )
                panel_out.append(row)

            stats["pages"] += 1
            stats["lifetime_days"] += max(last - first + 1, 0)
            stats["positives"] += len(positives)
            stats["negatives"] += len(negatives)
            stats["burst_days"] += len(page.burst_days)
            if page.burst_days:
                stats["pages_with_burst"] += 1
                if len(page.channels["edits"].days) < 10:
                    stats["pages_with_burst_under_10_active_days"] += 1

            retained = [r for r in revisions if not r["is_reverted"] and not r["is_revert"]]
            if retained:
                summaries.update(summarize_pages(retained, {page_id} if page.burst_days else set()))

            if stats["pages"] % 50_000 == 0:
                print(f"  {stats['pages']:,} pages ({time.monotonic() - start_time:,.0f}s)")

    print(
        f"{stats['pages']:,} pages: {daily_out.rows_written:,} page-days with activity, "
        f"{stats['burst_days']:,} burst page-days across {stats['pages_with_burst']:,} pages "
        f"({stats['pages_with_burst_under_10_active_days']:,} of them with <10 active days)"
    )

    # Pass 2: co-burst (other pages bursting on D-1) needs every page's
    # bursts, so it's joined onto the pass-1 panel afterwards.
    with RowGroupWriter(PANEL_PATH, PANEL_SCHEMA) as out:
        for batch in pq.ParquetFile(panel_tmp).iter_batches(batch_size=ROW_GROUP_SIZE):
            others = [
                co_burst.others(d.toordinal() - 1, bursting, editors)
                for d, bursting, editors in zip(
                    batch.column("date").to_pylist(),
                    batch.column("is_burst_1d").to_pylist(),
                    batch.column("editors_1d").to_pylist(),
                )
            ]
            table = pa.Table.from_batches([batch])
            table = table.append_column(PANEL_SCHEMA.field("co_burst_1d"), pa.array([o[0] for o in others], pa.int32()))
            table = table.append_column(
                PANEL_SCHEMA.field("co_burst_multi_editor_1d"), pa.array([o[1] for o in others], pa.int32())
            )
            out.write_table(table)
    panel_tmp.unlink()
    print(
        f"Panel: {out.rows_written:,} rows ({stats['positives']:,} positive, {stats['negatives']:,} "
        f"sampled negative of {stats['lifetime_days']:,} page-days) -> {PANEL_PATH}"
    )

    days = sorted(co_burst.pages)
    pq.write_table(
        pa.table(
            {
                "date": [date.fromordinal(d) for d in days],
                "pages_bursting": [co_burst.pages[d] for d in days],
                "pages_bursting_multi_editor": [co_burst.multi_editor[d] for d in days],
            },
            schema=CO_BURST_SCHEMA,
        ),
        CO_BURST_PATH,
        compression="zstd",
    )
    for label, counts in [("", co_burst.pages), (" with >=2 editors", co_burst.multi_editor)]:
        print(f"Top co-burst dates (pages bursting{label}):")
        for day, count in counts.most_common(10):
            print(f"  {date.fromordinal(day)}: {count} pages")

    PANEL_META_PATH.write_text(
        json.dumps(
            {
                "source": str(INPUT_PATH),
                "data_end": max_timestamp,
                "last_panel_day": date.fromordinal(last_full_day).isoformat(),
                "labels_final_through": date.fromordinal(last_final_day).isoformat(),
                "negative_sample_rate": NEGATIVE_SAMPLE_RATE,
                "seed": PANEL_SEED,
                "rows": out.rows_written,
                "positives": stats["positives"],
                "negatives": stats["negatives"],
                "lifetime_page_days": stats["lifetime_days"],
            },
            indent=2,
        )
    )

    strata = stratify(summaries)
    print("\nStratum sizes:")
    for key, page_ids in sorted(strata.items(), key=lambda kv: kv[0]):
        print(f"  {key}: {len(page_ids)} pages")
    sample = sample_pages(strata, per_stratum=PER_STRATUM_SAMPLE_SIZE, seed=SAMPLE_SEED)
    manifest = {
        "source": str(INPUT_PATH),
        "seed": SAMPLE_SEED,
        "per_stratum_sample_size": PER_STRATUM_SAMPLE_SIZE,
        "strata": sample,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    total_sampled = sum(len(v) for v in sample.values())
    print(f"\nSampled {total_sampled} pages across {len(sample)} strata -> {MANIFEST_PATH}")
    print(f"Done in {time.monotonic() - start_time:,.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
