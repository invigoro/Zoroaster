"""Fetch English Wikipedia's recent mainspace edits into a Parquet file.

This is the daily input for the deployed pipeline (PLAN.md §6 step 9). The
records have the same fields as the history dumps'
(`src/ingest/recent_changes.py`). Recent changes keep 30 days.

Usage:
    python scripts/fetch_recent_changes.py --start 2026-09-29T00:00:00Z --end 2026-09-30T00:00:00Z --out FILE
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import requests

from scripts.build_enwiki_revisions import SCHEMA as DUMP_SCHEMA
from src.ingest.recent_changes import recent_changes
from src.parquet_io import RowGroupWriter

SCHEMA = pa.schema([f for f in DUMP_SCHEMA if not f.name.startswith("mwh_")])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", required=True, help="inclusive, e.g. 2026-09-29T00:00:00Z")
    parser.add_argument("--end", required=True, help="exclusive")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start_time, count = time.monotonic(), 0
    with RowGroupWriter(args.out, SCHEMA) as writer:
        for record in recent_changes(args.start, args.end, session=requests.Session()):
            writer.append(record)
            count += 1
            if count % 25_000 == 0:
                print(f"  {count:,} edits, up to {record['timestamp']} ({time.monotonic() - start_time:,.0f}s)", flush=True)
    print(f"{count:,} mainspace edits from {args.start} to {args.end} in {time.monotonic() - start_time:,.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
