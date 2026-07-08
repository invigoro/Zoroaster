"""End-to-end test run of the ingest pipeline on a small dump.

Downloads (if needed) the Simple English Wikipedia stub-meta-history dump,
streams it into mainspace revision records, runs identity-revert detection,
and writes the labeled revisions to a Parquet file for inspection.

This targets a small dump (~900MB compressed) so the pipeline can be
validated and iterated on quickly before pointing it at English Wikipedia
at scale.

Usage:
    python scripts/build_test_revert_labels.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.download_dump import download
from src.ingest.revert_detect import detect_reverts
from src.ingest.stub_stream import iter_mainspace_revisions

DUMP_URL = (
    "https://dumps.wikimedia.org/simplewiki/latest/"
    "simplewiki-latest-stub-meta-history.xml.gz"
)
RAW_PATH = Path("data/raw/simplewiki-latest-stub-meta-history.xml.gz")
OUTPUT_PATH = Path("data/processed/simplewiki_test_revert_labels.parquet")

SCHEMA = pa.schema(
    [
        ("page_id", pa.int64()),
        ("page_title", pa.string()),
        ("namespace", pa.int32()),
        ("revision_id", pa.int64()),
        ("parent_id", pa.int64()),
        ("timestamp", pa.string()),
        ("user_text", pa.string()),
        ("user_id", pa.int64()),
        ("is_anon", pa.bool_()),
        ("is_minor", pa.bool_()),
        ("comment", pa.string()),
        ("byte_size", pa.int64()),
        ("sha1", pa.string()),
        ("is_reverted", pa.bool_()),
    ]
)


def main() -> int:
    if not RAW_PATH.exists():
        print(f"Downloading {DUMP_URL} -> {RAW_PATH}")
        download(DUMP_URL, RAW_PATH)
    else:
        print(f"Using existing dump at {RAW_PATH}")

    print("Streaming + parsing revisions (mainspace only)...")
    start = time.monotonic()
    revisions = list(iter_mainspace_revisions(str(RAW_PATH)))
    elapsed = time.monotonic() - start
    page_count = len({r["page_id"] for r in revisions})
    print(
        f"Parsed {len(revisions):,} mainspace revisions across "
        f"{page_count:,} pages in {elapsed:,.1f}s"
    )

    print("Running identity-revert detection...")
    start = time.monotonic()
    labeled = list(detect_reverts(revisions))
    elapsed = time.monotonic() - start
    reverted_count = sum(1 for r in labeled if r["is_reverted"])
    print(
        f"Flagged {reverted_count:,} / {len(labeled):,} revisions as "
        f"reverted ({reverted_count / len(labeled):.2%}) in {elapsed:,.1f}s"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, list] = {name: [] for name in SCHEMA.names}
    for row in labeled:
        for name in columns:
            columns[name].append(row[name])
    table = pa.table(columns, schema=SCHEMA)
    pq.write_table(table, OUTPUT_PATH, compression="zstd")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
