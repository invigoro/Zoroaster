"""Fetch English Wikipedia's recent mainspace edits into Parquet files.

This is the daily input for the deployed pipeline (PLAN.md §6 step 9). The
records have the same fields as the history dumps'
(`src/ingest/recent_changes.py`). Recent changes keep exactly 30 days.

Two modes:
- `--start/--end/--out`: one file for an arbitrary window;
- `--days FIRST LAST --out-dir DIR`: one file per UTC day, oldest first, as
  `DIR/YYYY-MM-DD.parquet`, skipping days already fetched. Oldest first
  matters for backfills: those days expire soonest.

Usage:
    python scripts/fetch_recent_changes.py --start 2026-09-29T00:00:00Z --end 2026-09-30T00:00:00Z --out FILE
    python scripts/fetch_recent_changes.py --days 2026-09-01 2026-09-29 --out-dir data/processed/enwiki/live
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import requests

from scripts.build_enwiki_revisions import SCHEMA as DUMP_SCHEMA
from src.ingest.recent_changes import recent_changes
from src.parquet_io import RowGroupWriter

SCHEMA = pa.schema([f for f in DUMP_SCHEMA if not f.name.startswith("mwh_")])
LIVE_DIR = Path("data/processed/enwiki/live")


def fetch_window(start: str, end: str, out: Path, session: requests.Session) -> int:
    """Write the window's records to `out` (atomically); returns how many."""
    out.parent.mkdir(parents=True, exist_ok=True)
    start_time, count = time.monotonic(), 0
    with RowGroupWriter(out, SCHEMA) as writer:
        for record in recent_changes(start, end, session=session):
            writer.append(record)
            count += 1
            if count % 50_000 == 0:
                print(f"  {count:,} edits, up to {record['timestamp']} ({time.monotonic() - start_time:,.0f}s)", flush=True)
    print(f"{count:,} mainspace edits from {start} to {end} in {time.monotonic() - start_time:,.0f}s -> {out}", flush=True)
    return count


def day_files(first: date, last: date, out_dir: Path) -> list[tuple[date, Path]]:
    """(day, file) for each day from `first` to `last` that has no file yet, oldest first."""
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    return [(d, out_dir / f"{d.isoformat()}.parquet") for d in days if not (out_dir / f"{d.isoformat()}.parquet").exists()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", help="inclusive, e.g. 2026-09-29T00:00:00Z")
    parser.add_argument("--end", help="exclusive")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--days", nargs=2, metavar=("FIRST", "LAST"), type=date.fromisoformat)
    parser.add_argument("--out-dir", type=Path, default=LIVE_DIR)
    args = parser.parse_args(argv)
    session = requests.Session()
    if args.days:
        todo = day_files(*args.days, args.out_dir)
        print(f"{len(todo)} days to fetch", flush=True)
        for day, path in todo:
            fetch_window(f"{day.isoformat()}T00:00:00Z", f"{(day + timedelta(days=1)).isoformat()}T00:00:00Z", path, session)
        return 0
    if not (args.start and args.end and args.out):
        parser.error("give --start, --end and --out, or --days")
    fetch_window(args.start, args.end, args.out, session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
