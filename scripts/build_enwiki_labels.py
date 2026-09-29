"""Label English Wikipedia revisions, one page bucket at a time.

Reads every month's file for a bucket (`build_enwiki_revisions.py`) and
sorts by page and revision. It then runs the same identity-revert detection
as the test corpus (`src.ingest.revert_detect`) on the content hashes, so
both wikis use one definition (PLAN.md §2). The history window starts a year
before the first prediction day, so reverts to states from before the
window can only affect revisions in that lookback year, never a label.
Wikimedia's own flags (`mwh_is_revert` / `mwh_is_reverted`, which have no
revert window) are kept alongside for comparison.

Each bucket also records how many distinct pages each non-bot editor made
non-revert edits to, per day. `build_enwiki_features.py` sums those across
buckets to find mass-editing days, since a page lives in exactly one
bucket.

Outputs:
- `data/processed/enwiki/labels/part-NNN.parquet`
- `data/processed/enwiki/editor_days/part-NNN.parquet`

Usage:
    python scripts/build_enwiki_labels.py [--workers N]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from itertools import groupby
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.build_enwiki_revisions import N_BUCKETS, SCHEMA, bucket_files
from src.features.activity import page_editor_days
from src.ingest.revert_detect import detect_page_reverts
from src.parquet_io import RowGroupWriter

LABELS_DIR = Path("data/processed/enwiki/labels")
EDITOR_DAYS_DIR = Path("data/processed/enwiki/editor_days")
LABEL_SCHEMA = pa.schema(
    list(SCHEMA)
    + [("is_reverted", pa.bool_()), ("reverted_by_revision_id", pa.int64()), ("is_revert", pa.bool_())]
)
EDITOR_DAY_SCHEMA = pa.schema([("user_text", pa.string()), ("day", pa.int32()), ("pages", pa.int32())])


def labels_path(bucket: int) -> Path:
    return LABELS_DIR / f"part-{bucket:03d}.parquet"


def iter_sorted_pages(table: pa.Table) -> Iterator[list[dict]]:
    """One page's rows at a time, from a table in any order."""
    table = table.sort_by([("page_id", "ascending"), ("revision_id", "ascending")])
    rows = (row for batch in table.to_batches(max_chunksize=200_000) for row in batch.to_pylist())
    for _, page_rows in groupby(rows, key=lambda r: r["page_id"]):
        yield list(page_rows)


def label_bucket(bucket: int) -> Counter[str]:
    stats: Counter[str] = Counter()
    if labels_path(bucket).exists():
        return stats
    table = pa.concat_tables(pq.read_table(f) for f in bucket_files(bucket))
    editor_days: Counter[tuple[str, int]] = Counter()
    with RowGroupWriter(labels_path(bucket), LABEL_SCHEMA) as out:
        for page in iter_sorted_pages(table):
            labeled = detect_page_reverts(page)
            for r in labeled:
                out.append(r)
                stats["revisions"] += 1
                stats["reverted"] += r["is_reverted"]
                stats["reverts"] += r["is_revert"]
                stats["mwh_reverted"] += r["mwh_is_reverted"]
                stats["mwh_reverts"] += r["mwh_is_revert"]
                stats["both_reverted"] += r["is_reverted"] and r["mwh_is_reverted"]
                stats["both_reverts"] += r["is_revert"] and r["mwh_is_revert"]
            editor_days.update(page_editor_days(labeled))
            stats["pages"] += 1
    users, days, pages = zip(*((u, d, n) for (u, d), n in editor_days.items())) if editor_days else ((), (), ())
    EDITOR_DAYS_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"user_text": users, "day": days, "pages": pages}, schema=EDITOR_DAY_SCHEMA),
        EDITOR_DAYS_DIR / f"part-{bucket:03d}.parquet",
        compression="zstd",
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    start = time.monotonic()
    totals: Counter[str] = Counter()
    with Pool(args.workers) as pool:
        for done, stats in enumerate(pool.imap_unordered(label_bucket, range(N_BUCKETS)), start=1):
            totals.update(stats)
            if done % 16 == 0:
                print(f"  {done}/{N_BUCKETS} buckets ({time.monotonic() - start:,.0f}s)", flush=True)
    n = totals["revisions"]
    print(f"Labeled {n:,} revisions across {totals['pages']:,} pages in {time.monotonic() - start:,.0f}s")
    if n:
        print(f"  reverted {totals['reverted']:,} ({totals['reverted'] / n:.2%}); reverts {totals['reverts']:,} "
              f"({totals['reverts'] / n:.2%})")
        print(f"  Wikimedia's flags: reverted {totals['mwh_reverted']:,}, reverts {totals['mwh_reverts']:,}; "
              f"agreement on reverts {totals['both_reverts'] / max(totals['reverts'], 1):.1%} of ours, "
              f"on reverted {totals['both_reverted'] / max(totals['reverted'], 1):.1%} of ours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
