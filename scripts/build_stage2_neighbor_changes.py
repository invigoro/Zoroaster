"""What changed on each Stage 2 example's bursting linked pages the day before.

Takes each example's top MAX_NEIGHBORS bursting neighbors, most editors
first (`build_stage2_targets.py`). For each (linked page Q, day D-1), this
takes the net change over that day: the word diff between Q's text at the
start of D-1 (the parent of its first revision that day) and at its end
(its last revision that day).
- It's point-in-time: both states are known by the end of D-1.
- Vandalism added and reverted within the day cancels out.
- A page created that day contributes its opening text, since it's diffed
  against nothing. That's typical of breaking news.

The snippet kept is the longest inserted span, whitespace-collapsed and
cut to SNIPPET_CHARS.

Writes `data/processed/enwiki/stage2/neighbor_changes/part-NNNNN.parquet`,
resumable like `fetch_stage2_diffs.py`.

Usage:
    python scripts/build_stage2_neighbor_changes.py
"""

from __future__ import annotations

import re
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import requests

from scripts.build_enwiki_labels import LABELS_DIR
from scripts.build_stage2_targets import OUT_DIR
from scripts.fetch_stage2_diffs import EXAMPLES_DIR
from src.stage2.diff import word_diff
from src.stage2.fetch import fetch_contents

CHANGES_DIR = OUT_DIR / "neighbor_changes"
MAX_NEIGHBORS = 3
SNIPPET_CHARS = 150
PAIRS_PER_PART = 500


def snippet(start: str | None, end: str) -> tuple[str | None, int]:
    """The day's longest inserted span (collapsed, cut short), and total inserted chars."""
    diff = word_diff(start, end)
    if not diff.inserted:
        return None, 0
    longest = re.sub(r"\s+", " ", max(diff.inserted, key=len)).strip()
    cut = longest if len(longest) <= SNIPPET_CHARS else longest[:SNIPPET_CHARS].rsplit(" ", 1)[0] + " …"
    return cut, sum(len(s) for s in diff.inserted)


def day_bounds(titles: list[str], days: list) -> list[dict]:
    """For each (title, day): the page's start-of-day and end-of-day revision ids."""
    labels = ds.dataset(LABELS_DIR, format="parquet")
    rows = labels.to_table(
        columns=["page_id", "page_title", "revision_id", "parent_id", "timestamp"],
        filter=pc.field("page_title").isin(sorted(set(titles))),
    )
    rows = rows.append_column("day", pc.cast(pc.strptime(pc.utf8_slice_codeunits(rows["timestamp"], 0, 10),
                                                        format="%Y-%m-%d", unit="s"), pa.date32()))
    wanted = set(zip(titles, days))
    by_pair: dict[tuple, list[tuple[int, int | None]]] = {}
    for title, day, revision, parent in zip(*(rows[c].to_pylist() for c in ("page_title", "day", "revision_id", "parent_id"))):
        if (title, day) in wanted:
            by_pair.setdefault((title, day), []).append((revision, parent))
    bounds = []
    for (title, day), revisions in by_pair.items():
        revisions.sort()
        bounds.append({"title": title, "date": day, "start_revision": revisions[0][1] or None,
                       "end_revision": revisions[-1][0]})
    return bounds


def main() -> int:
    start_time = time.monotonic()
    examples = pq.read_table(EXAMPLES_DIR, columns=["date", "bursting_neighbors"]).to_pylist()
    pairs = sorted({(t, r["date"] - timedelta(days=1)) for r in examples for t in r["bursting_neighbors"][:MAX_NEIGHBORS]})
    bounds = day_bounds([t for t, _ in pairs], [d for _, d in pairs])
    bounds.sort(key=lambda b: (b["date"], b["title"]))
    print(f"{len(pairs):,} (page, day) pairs; {len(bounds):,} found in the revision data "
          f"({time.monotonic() - start_time:,.0f}s)")
    CHANGES_DIR.mkdir(parents=True, exist_ok=True)
    first_part = len(list(CHANGES_DIR.glob("part-*.parquet")))
    session = requests.Session()
    for n, begin in enumerate(range(first_part * PAIRS_PER_PART, len(bounds), PAIRS_PER_PART), start=first_part):
        chunk = bounds[begin : begin + PAIRS_PER_PART]
        ids = [i for b in chunk for i in (b["start_revision"], b["end_revision"]) if i]
        texts = dict(fetch_contents(list(dict.fromkeys(ids)), session=session))
        rows = []
        for b in chunk:
            end = texts.get(b["end_revision"])
            start = texts.get(b["start_revision"]) if b["start_revision"] else None
            if end is None or (b["start_revision"] and start is None):
                continue
            text, inserted = snippet(start, end)
            rows.append(b | {"snippet": text, "inserted_chars": inserted, "created_that_day": not b["start_revision"]})
        tmp = CHANGES_DIR / f"part-{n:05d}.parquet.tmp"
        pq.write_table(pa.Table.from_pylist(rows), tmp)
        tmp.replace(CHANGES_DIR / f"part-{n:05d}.parquet")
        print(f"  part {n}: {len(rows)}/{len(chunk)} ({time.monotonic() - start_time:,.0f}s)", flush=True)
    total = pq.read_table(CHANGES_DIR)
    with_text = pc.sum(pc.invert(pc.is_null(total["snippet"]))).as_py()
    print(f"Done: {total.num_rows:,} pairs, {with_text:,} with a snippet, "
          f"{pc.sum(total['created_that_day']).as_py():,} pages created that day")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
