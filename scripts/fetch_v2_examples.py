"""Fetch version 2's page states and derive its examples.

For each page-day in `data/processed/enwiki/v2/targets.parquet`
(`build_v2_targets.py`), this fetches the page at its three moments (50
revisions a request) and derives them with `src/forecast/changes.py`:
- **The prompt side**, from the page at the end of D-1:
  - its lead and section headings;
  - whether it's about a living person (`[[Category:Living people]]`), whose
    forecasts stay structured;
  - yesterday's change, from the end of D-2 to the end of D-1.
- **The target**, the day's change from the end of D-1 to the end of D: new
  prose, sections and kinds of change.

The page at the end of D-1 (capped) and the target's new spans are stored
too, so prompts and targets can be redesigned without refetching.

Writes `data/processed/enwiki/v2/examples/part-NNNNN.parquet`. A rerun
skips page-days already in a part.

Usage:
    python scripts/fetch_v2_examples.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from scripts.build_v2_targets import OUT_DIR, TARGETS_PATH
from src.forecast.changes import day_change, headings, lead
from src.stage2.fetch import fetch_contents

EXAMPLES_DIR = OUT_DIR / "examples"
TARGETS_PER_PART = 500
MAX_PAGE_CHARS = 300_000
MAX_SPAN_CHARS, MAX_SPANS_CHARS = 5_000, 30_000
LIVING = re.compile(r"\[\[\s*Category\s*:\s*Living[ _]people\s*[\]|]", re.I)
_LIST = pa.list_(pa.string())
DERIVED = [
    ("lead", pa.string()), ("heading_levels", pa.list_(pa.int8())), ("heading_titles", _LIST), ("living", pa.bool_()),
    ("yesterday_prose", pa.string()), ("yesterday_sections", _LIST), ("yesterday_kinds", _LIST),
    ("yesterday_known", pa.bool_()),
    ("prose", pa.string()), ("sections", _LIST), ("kinds", _LIST), ("inserted_chars", pa.int64()),
    ("removed_chars", pa.int64()), ("spans", _LIST), ("page_text", pa.string()),
]


def _capped(spans: list[str]) -> list[str]:
    kept, total = [], 0
    for span in spans:
        if total >= MAX_SPANS_CHARS:
            break
        kept.append(span[:MAX_SPAN_CHARS])
        total += len(kept[-1])
    return kept


def derive(target: dict, texts: dict[int, str | None]) -> dict | None:
    """The example for one page-day, or None if its page text at the end of D-1 or D is unavailable."""
    page, end = texts.get(target["prompt_id"]), texts.get(target["end_id"])
    if page is None or end is None:
        return None
    if target["start_id"] == target["prompt_id"]:
        yesterday, known = day_change(page, page), True  # no edits on D-1
    elif target["start_id"] is not None and texts.get(target["start_id"]) is not None:
        yesterday, known = day_change(texts[target["start_id"]], page, max_prose_chars=800), True
    else:
        yesterday, known = day_change(page, page), False  # no earlier state to diff against
    change = day_change(page, end)
    sections = headings(page)
    return target | {
        "lead": lead(page),
        "heading_levels": [level for level, _ in sections],
        "heading_titles": [title for _, title in sections],
        "living": bool(LIVING.search(page)),
        "yesterday_prose": yesterday["prose"], "yesterday_sections": yesterday["sections"],
        "yesterday_kinds": yesterday["kinds"], "yesterday_known": known,
        "prose": change["prose"], "sections": change["sections"], "kinds": change["kinds"],
        "inserted_chars": change["inserted_chars"], "removed_chars": change["removed_chars"],
        "spans": _capped(change["spans"]), "page_text": page[:MAX_PAGE_CHARS],
    }


def main() -> int:
    start = time.monotonic()
    targets = pq.read_table(TARGETS_PATH)
    schema = pa.schema(list(targets.schema) + [pa.field(n, t) for n, t in DERIVED])
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    parts = sorted(EXAMPLES_DIR.glob("part-*.parquet"))
    done = set()
    for part in parts:
        t = pq.read_table(part, columns=["page_id", "date"])
        done |= set(zip(t["page_id"].to_pylist(), t["date"].to_pylist()))
    todo = [t for t in targets.to_pylist() if (t["page_id"], t["date"]) not in done]
    print(f"{targets.num_rows:,} page-days, {len(done):,} already fetched, {len(todo):,} to go", flush=True)
    session = requests.Session()
    for n, begin in enumerate(range(0, len(todo), TARGETS_PER_PART), start=len(parts)):
        chunk = todo[begin : begin + TARGETS_PER_PART]
        ids = list(dict.fromkeys(i for t in chunk for i in (t["start_id"], t["prompt_id"], t["end_id"]) if i))
        texts = dict(fetch_contents(ids, session=session))
        rows = [row for row in (derive(t, texts) for t in chunk) if row is not None]
        tmp = EXAMPLES_DIR / f"part-{n:05d}.parquet.tmp"
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression="zstd")
        tmp.replace(EXAMPLES_DIR / f"part-{n:05d}.parquet")
        with_prose = sum(1 for r in rows if r["prose"])
        print(f"  part {n}: {len(rows)}/{len(chunk)} page-days, {with_prose} with new prose, "
              f"{sum(r['living'] for r in rows)} about living people ({time.monotonic() - start:,.0f}s)", flush=True)
    print(f"Done in {time.monotonic() - start:,.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
