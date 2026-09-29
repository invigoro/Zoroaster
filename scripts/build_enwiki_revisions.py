"""Extract English Wikipedia mainspace revisions into page-bucketed Parquet.

The MediaWiki history dump is split by time, one file per month, but every
later step works one page at a time. So each month's revisions (see
`src.ingest.mediawiki_history`) are split by `page_id % N_BUCKETS` into
`data/processed/enwiki/revisions/month=YYYY-MM/bucket=NNN.parquet`. A
bucket's files across all months then hold every revision of its pages.

Months are independent, so they run in parallel. A month is written to a
temporary directory and renamed when done, and finished months are
skipped. The script can therefore run while the download is still going
and again once it finishes; it only handles months whose download is
complete.

Usage:
    python scripts/build_enwiki_revisions.py [--workers N]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.download_enwiki_history import month_path, months
from src.ingest.mediawiki_history import iter_revisions

N_BUCKETS = 128
OUT_DIR = Path("data/processed/enwiki/revisions")
BATCH_ROWS = 500_000

SCHEMA = pa.schema(
    [
        ("page_id", pa.int64()),
        ("page_title", pa.string()),
        ("revision_id", pa.int64()),
        ("parent_id", pa.int64()),
        ("timestamp", pa.string()),
        ("user_text", pa.string()),
        ("is_anon", pa.bool_()),
        ("is_bot", pa.bool_()),
        ("byte_size", pa.int64()),
        ("sha1", pa.string()),
        ("page_created", pa.string()),
        ("mwh_is_revert", pa.bool_()),
        ("mwh_is_reverted", pa.bool_()),
    ]
)


def bucket_files(bucket: int) -> list[Path]:
    """Every month's file for one bucket, in month order."""
    return sorted(OUT_DIR.glob(f"month=*/bucket={bucket:03d}.parquet"))


def _write_batch(columns: dict[str, list], writers: dict[int, pq.ParquetWriter], directory: Path) -> None:
    table = pa.table(columns, schema=SCHEMA)
    buckets = table["page_id"].to_numpy() % N_BUCKETS
    order = np.argsort(buckets, kind="stable")
    table, buckets = table.take(order), buckets[order]
    bounds = np.searchsorted(buckets, np.arange(N_BUCKETS + 1))
    for bucket in range(N_BUCKETS):
        lo, hi = bounds[bucket], bounds[bucket + 1]
        if hi > lo:
            if bucket not in writers:
                path = directory / f"bucket={bucket:03d}.parquet"
                writers[bucket] = pq.ParquetWriter(path, SCHEMA, compression="zstd")
            writers[bucket].write_table(table.slice(lo, hi - lo))


def extract_month(month: str) -> tuple[str, int, float]:
    """Returns (month, revisions written, seconds); -1 revisions if skipped."""
    final = OUT_DIR / f"month={month}"
    if final.exists() or not month_path(month).exists():
        return month, -1, 0.0
    start = time.monotonic()
    tmp = OUT_DIR / f"month={month}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    writers: dict[int, pq.ParquetWriter] = {}
    columns: dict[str, list] = {name: [] for name in SCHEMA.names}
    rows = 0
    for record in iter_revisions(month_path(month)):
        for name, values in columns.items():
            values.append(record[name])
        rows += 1
        if rows % BATCH_ROWS == 0:
            _write_batch(columns, writers, tmp)
            columns = {name: [] for name in SCHEMA.names}
    if columns["page_id"]:
        _write_batch(columns, writers, tmp)
    for writer in writers.values():
        writer.close()
    os.replace(tmp, final)
    return month, rows, time.monotonic() - start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    todo = [m for m in months() if not (OUT_DIR / f"month={m}").exists()]
    ready = [m for m in todo if month_path(m).exists()]
    print(f"{len(todo)} months to extract, {len(ready)} downloaded; {args.workers} workers")
    total = 0
    with Pool(args.workers) as pool:
        for month, rows, seconds in pool.imap_unordered(extract_month, ready):
            if rows >= 0:
                total += rows
                print(f"  {month}: {rows:,} mainspace revisions in {seconds:,.0f}s "
                      f"({time.monotonic() - start:,.0f}s elapsed)", flush=True)
    print(f"Extracted {total:,} revisions; {len(todo) - len(ready)} months still waiting for the download")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
