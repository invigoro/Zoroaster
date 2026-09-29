"""Download the English Wikipedia MediaWiki history dump months Stage 1 needs.

Source: https://dumps.wikimedia.org/other/mediawiki_history/ (see PLAN.md
§6 step 5). One TSV.bz2 file per month of events; all months come from the
same snapshot, because records are rewritten retroactively between
snapshots (renames, reverts, moves).

The window runs from FIRST_MONTH (a year of feature lookback before the
training window) through the snapshot's end, about 22GB. Files already
downloaded are skipped; two download at a time, Wikimedia's usual limit
for dumps.

Usage:
    python scripts/download_enwiki_history.py
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_dump import download

SNAPSHOT = "2026-08"
FIRST_MONTH = "2023-06"
LAST_MONTH = "2026-09"  # the snapshot's final, partial month
RAW_DIR = Path("data/raw/enwiki_history")
BASE_URL = f"https://dumps.wikimedia.org/other/mediawiki_history/{SNAPSHOT}/enwiki/"
PARALLEL_DOWNLOADS = 2


def months(first: str = FIRST_MONTH, last: str = LAST_MONTH) -> list[str]:
    year, month = map(int, first.split("-"))
    out = []
    while f"{year:04d}-{month:02d}" <= last:
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def month_path(month: str) -> Path:
    return RAW_DIR / f"{SNAPSHOT}.enwiki.{month}.tsv.bz2"


def main() -> int:
    todo = [m for m in months() if not month_path(m).exists()]
    print(f"{len(months())} months {FIRST_MONTH}..{LAST_MONTH}; {len(todo)} to download")
    start = time.monotonic()

    def fetch(month: str) -> tuple[str, float]:
        t = time.monotonic()
        download(BASE_URL + month_path(month).name, month_path(month), progress=False)
        return month, time.monotonic() - t

    failures = 0
    with ThreadPoolExecutor(PARALLEL_DOWNLOADS) as pool:
        futures = [pool.submit(fetch, m) for m in todo]
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                month, seconds = future.result()
            except Exception as exc:  # keep going; a rerun retries what's missing
                failures += 1
                print(f"  FAILED: {exc}")
                continue
            size = month_path(month).stat().st_size
            print(
                f"  {done}/{len(todo)} {month}: {size / 1e9:.2f} GB in {seconds:,.0f}s "
                f"({size / 1e6 / seconds:.1f} MB/s); {time.monotonic() - start:,.0f}s elapsed",
                flush=True,
            )
    print(f"Done: {len(todo) - failures} downloaded, {failures} failed (rerun to retry)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
