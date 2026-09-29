"""Fetch real diff text for the already-sampled test pages.

Wires up `src.ingest.fetch_diffs` (validated so far only with mocked HTTP)
against live Wikipedia, restricted to the small, deliberate sample chosen by
`build_test_features.py`. This is the first live-network step in the
pipeline and validates the diff-extraction step end-to-end before it's
pointed at the full English Wikipedia corpus.

Does not download anything new via `download_dump.py` — it reads the local
`data/processed/simplewiki_test_revert_labels.parquet` and
`simplewiki_test_sample_manifest.json` produced by the two `build_test_*`
scripts, then makes one (or two) HTTP calls per retained revision (neither
reverted nor itself a revert) belonging to a sampled page. Bot edits are
still included; see PLAN.md §7.

Usage:
    python scripts/build_test_diffs.py [max_revisions]

    `max_revisions` (optional) caps how many retained revisions are fetched,
    for a quick end-to-end validation pass instead of the full sample (the
    full 150-page sample is ~4,140 retained revisions, i.e. ~2+ hours at the
    polite 1 req/sec rate — pass no argument to run the full set).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from src.ingest.fetch_diffs import REQUEST_DELAY_SECONDS, fetch_revision_diff

LANG = "simple"  # this test corpus is Simple English Wikipedia, not "en"

LABELS_PATH = Path("data/processed/simplewiki_test_revert_labels.parquet")
MANIFEST_PATH = Path("data/processed/simplewiki_test_sample_manifest.json")
OUTPUT_PATH = Path("data/processed/simplewiki_test_diffs.parquet")

SCHEMA = pa.schema(
    [
        ("page_id", pa.int64()),
        ("page_title", pa.string()),
        ("revision_id", pa.int64()),
        ("parent_id", pa.int64()),
        ("timestamp", pa.string()),
        ("added_text", pa.string()),
        ("removed_text", pa.string()),
    ]
)


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    sampled_page_ids = {
        page_id for ids in manifest["strata"].values() for page_id in ids
    }
    print(f"{len(sampled_page_ids)} sampled pages from {MANIFEST_PATH}")

    table = pq.read_table(LABELS_PATH)
    rows = table.to_pylist()
    targets = [
        r
        for r in rows
        if r["page_id"] in sampled_page_ids and not r["is_reverted"] and not r["is_revert"]
    ]
    targets.sort(key=lambda r: (r["page_id"], r["revision_id"]))
    print(f"{len(targets):,} retained revisions to fetch across those pages")

    if len(sys.argv) > 1:
        max_revisions = int(sys.argv[1])
        targets = targets[:max_revisions]
        print(f"Capped to first {len(targets):,} revisions for this run")

    session = requests.Session()
    columns: dict[str, list] = {name: [] for name in SCHEMA.names}
    failures = 0

    for i, row in enumerate(targets):
        if i > 0:
            time.sleep(REQUEST_DELAY_SECONDS)
        try:
            diff = fetch_revision_diff(
                page_id=row["page_id"],
                revision_id=row["revision_id"],
                parent_id=row["parent_id"],
                lang=LANG,
                session=session,
            )
        except (ValueError, requests.RequestException) as exc:
            failures += 1
            print(f"  skip revision {row['revision_id']} ({row['page_title']}): {exc}")
            continue

        columns["page_id"].append(diff.page_id)
        columns["page_title"].append(row["page_title"])
        columns["revision_id"].append(diff.revision_id)
        columns["parent_id"].append(diff.parent_id)
        columns["timestamp"].append(row["timestamp"])
        columns["added_text"].append(diff.added_text)
        columns["removed_text"].append(diff.removed_text)

        if (i + 1) % 25 == 0 or (i + 1) == len(targets):
            print(f"  fetched {i + 1:,}/{len(targets):,} ({failures} failures so far)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns, schema=SCHEMA), OUTPUT_PATH, compression="zstd")
    print(
        f"Wrote {len(columns['page_id']):,} diffs ({failures} failures) -> {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
