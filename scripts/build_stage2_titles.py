"""Point-in-time page titles for the Stage 2 prompts.

The revision data keeps each page's title as of the dump snapshot
(2026-08), so a prompt would show a page renamed after an event under its
later name. The history dumps also record `page_title_historical`, the
title when each revision was made. This scans them for the revisions that
Stage 2 prompts name:
- each example's target revision: the page's title when edited;
- each shown bursting neighbor's last revision on the day before: its title
  when the prompt's signals were known.

Writes `data/processed/enwiki/stage2/titles.parquet` with columns kind
(`page` or `neighbor`), title (as of the snapshot), date (the example's
day, or the neighbor's), revision_id and title_then.

Usage:
    python scripts/build_stage2_titles.py [--workers N]
"""

from __future__ import annotations

import argparse
import bz2
import sys
import time
from datetime import timedelta
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.build_stage2_neighbor_changes import day_bounds
from scripts.build_stage2_targets import OUT_DIR
from scripts.download_enwiki_history import month_path
from scripts.fetch_stage2_diffs import EXAMPLES_DIR
from src.ingest.mediawiki_history import COLUMNS
from src.stage2.examples import MAX_NEIGHBORS_SHOWN

TITLES_PATH = OUT_DIR / "titles.parquet"
SCHEMA = pa.schema([("kind", pa.string()), ("title", pa.string()), ("date", pa.date32()),
                    ("revision_id", pa.int64()), ("title_then", pa.string())])
_ENTITY, _TYPE, _REVISION = (COLUMNS.index(c) for c in ("event_entity", "event_type", "revision_id"))
_TITLE, _TITLE_THEN = COLUMNS.index("page_title"), COLUMNS.index("page_title_historical")


def scan_titles(path: Path, wanted: set[int]) -> dict[int, tuple[str, str]]:
    """{revision id: (title, title then)} for the wanted revisions in one monthly dump."""
    found = {}
    with bz2.open(path, "rt", encoding="utf-8", errors="replace", newline="\n") as f:
        for line in f:
            fields = line.split("\t", _REVISION + 1)
            if fields[_ENTITY] != "revision" or fields[_TYPE] != "create" or not fields[_REVISION]:
                continue
            revision = int(fields[_REVISION])
            if revision in wanted:
                found[revision] = (fields[_TITLE], fields[_TITLE_THEN])
    return found


def _scan_month(job: tuple[str, set[int]]) -> tuple[str, dict, float]:
    month, wanted = job
    start = time.monotonic()
    return month, scan_titles(month_path(month), wanted), time.monotonic() - start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)
    start = time.monotonic()
    examples = pq.read_table(EXAMPLES_DIR, columns=["page_title", "date", "revision_id", "timestamp",
                                                    "bursting_neighbors"]).to_pylist()
    wanted = [{"kind": "page", "title": r["page_title"], "date": r["date"], "revision_id": r["revision_id"],
               "month": r["timestamp"][:7]} for r in examples]
    pairs = sorted({(t, r["date"] - timedelta(days=1)) for r in examples
                    for t in r["bursting_neighbors"][:MAX_NEIGHBORS_SHOWN]})
    wanted += [{"kind": "neighbor", "title": b["title"], "date": b["date"], "revision_id": b["end_revision"],
                "month": b["date"].strftime("%Y-%m")} for b in day_bounds([t for t, _ in pairs], [d for _, d in pairs])]
    by_month: dict[str, set[int]] = {}
    for w in wanted:
        by_month.setdefault(w["month"], set()).add(w["revision_id"])
    print(f"{len(examples):,} pages and {len(wanted) - len(examples):,} of {len(pairs):,} neighbor-days, "
          f"in {len(by_month)} monthly dumps ({time.monotonic() - start:,.0f}s)", flush=True)
    found: dict[int, tuple[str, str]] = {}
    with Pool(min(args.workers, len(by_month))) as pool:
        for month, titles, seconds in pool.imap_unordered(_scan_month, sorted(by_month.items())):
            found |= titles
            print(f"  {month}: {len(titles):,} of {len(by_month[month]):,} revisions ({seconds:,.0f}s)", flush=True)
    rows = [{k: w[k] for k in ("kind", "title", "date", "revision_id")}
            | {"title_then": (found.get(w["revision_id"]) or (None, None))[1] or None} for w in wanted]
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), TITLES_PATH)
    mismatched = sum(found[w["revision_id"]][0] != w["title"] for w in wanted if w["revision_id"] in found)
    print(f"Snapshot titles that disagree with the revision data: {mismatched:,}")
    for kind in ("page", "neighbor"):
        rs = [r for r in rows if r["kind"] == kind]
        renamed = [r for r in rs if r["title_then"] and r["title_then"] != r["title"]]
        print(f"{kind}s: {len(rs):,}, {sum(r['title_then'] is None for r in rs):,} not found, "
              f"{len(renamed):,} titled differently then")
        for r in renamed[:12]:
            print(f"    {r['date']}: {r['title_then']}  ->  {r['title']}")
    print(f"Wrote {TITLES_PATH} ({time.monotonic() - start:,.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
