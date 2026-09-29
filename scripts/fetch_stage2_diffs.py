"""Fetch Stage 2 targets' text and compute their word-level diffs.

For each target revision from `build_stage2_targets.py`:
1. Fetch its wikitext and its parent's from the MediaWiki API
   (`src.stage2.fetch`: 50 revisions per request, sequential).
2. Compute the word-level diff (`src.stage2.diff`).
3. Keep only what training needs: the inserted and removed text, the
   section and the window of parent text around the change
   (`src.stage2.examples.edit_context`), and the changed line blocks
   themselves, before and after (capped), so targets can be re-derived
   without refetching. Full article text is never stored.

Targets that insert nothing (pure removals, whitespace-only changes), and
targets whose text can't be fetched, are dropped and counted.

Output goes to `data/processed/enwiki/stage2/examples/part-NNNNN.parquet`,
one part per chunk of targets, so an interrupted run resumes where it
stopped.

Usage:
    python scripts/fetch_stage2_diffs.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

from scripts.build_stage2_targets import OUT_DIR, TARGETS_PATH
from src.stage2.diff import word_diff
from src.stage2.examples import edit_context
from src.stage2.fetch import fetch_contents

EXAMPLES_DIR = OUT_DIR / "examples"
TARGETS_PER_PART = 500
MAX_ADDED_CHARS = 2000


def examples_from(targets: list[dict], texts: dict[int, str | None], dropped: Counter) -> list[dict]:
    rows = []
    for t in targets:
        text = texts.get(t["revision_id"])
        parent = texts.get(t["parent_id"]) if t["parent_id"] else None
        if text is None or (t["parent_id"] and parent is None):
            dropped["text unavailable"] += 1
            continue
        diff = word_diff(parent, text)
        if not diff.inserted:
            dropped["inserts nothing"] += 1
            continue
        section, context = edit_context(parent or "", diff.first_offset or 0)
        added = diff.added_text
        rows.append(t | {
            "section": section,
            "context": context,
            "added_text": added[:MAX_ADDED_CHARS],
            "removed_text": diff.removed_text[:MAX_ADDED_CHARS],
            "n_insertions": len(diff.inserted),
            "added_chars": len(added),
            "truncated": len(added) > MAX_ADDED_CHARS,
            "old_blocks": diff.old_blocks,
            "new_blocks": diff.new_blocks,
        })
    return rows


def main() -> int:
    start = time.monotonic()
    targets = pq.read_table(TARGETS_PATH).to_pylist()
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    done_parts = sorted(EXAMPLES_DIR.glob("part-*.parquet"))
    first_part = len(done_parts)
    todo = targets[first_part * TARGETS_PER_PART :]
    print(f"{len(targets):,} targets; {first_part} parts already written; {len(todo):,} to fetch")
    session = requests.Session()
    dropped: Counter[str] = Counter()
    kept = 0
    for n, begin in enumerate(range(0, len(todo), TARGETS_PER_PART), start=first_part):
        chunk = todo[begin : begin + TARGETS_PER_PART]
        ids = []
        for t in chunk:  # a target and its parent side by side, so they share a request
            ids += [t["revision_id"]] + ([t["parent_id"]] if t["parent_id"] else [])
        texts = dict(fetch_contents(list(dict.fromkeys(ids)), session=session))
        rows = examples_from(chunk, texts, dropped)
        kept += len(rows)
        tmp = EXAMPLES_DIR / f"part-{n:05d}.parquet.tmp"
        pq.write_table(pa.Table.from_pylist(rows), tmp)
        tmp.replace(EXAMPLES_DIR / f"part-{n:05d}.parquet")
        print(f"  part {n}: {len(rows)}/{len(chunk)} kept ({time.monotonic() - start:,.0f}s)", flush=True)
    total = pq.read_table(EXAMPLES_DIR)
    by_split = {v["values"]: v["counts"] for v in pc.value_counts(total["split"]).to_pylist()}
    print(f"Done: {total.num_rows:,} examples {by_split}; dropped this run: {dict(dropped)}; "
          f"median inserted chars {pc.approximate_median(total['added_chars']).as_py():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
