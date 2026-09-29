"""End-to-end test run of the ingest pipeline on a small dump.

Downloads (if needed) the Simple English Wikipedia stub-meta-history dump,
streams it into mainspace revision records, runs identity-revert detection,
and writes the labeled revisions to a Parquet file for inspection.

This targets a small dump (~900MB compressed) so the pipeline can be
validated and iterated on quickly before pointing it at English Wikipedia
at scale. Everything streams one page at a time and is written in row
groups, so memory stays bounded by the largest single page history.

With `--from-parquet`, revision metadata is read from an existing labels
file instead of the dump (any old label columns are ignored), which re-runs
revert detection in a couple of minutes instead of re-parsing the dump.

Usage:
    python scripts/build_test_revert_labels.py
    python scripts/build_test_revert_labels.py --from-parquet <labels.parquet>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.download_dump import download
from src.ingest.revert_detect import detect_reverts
from src.ingest.stub_stream import iter_mainspace_revisions
from src.parquet_io import RowGroupWriter

DUMP_URL = (
    "https://dumps.wikimedia.org/simplewiki/latest/"
    "simplewiki-latest-stub-meta-history.xml.gz"
)
RAW_PATH = Path("data/raw/simplewiki-latest-stub-meta-history.xml.gz")
OUTPUT_PATH = Path("data/processed/simplewiki_test_revert_labels.parquet")

ROW_GROUP_SIZE = 250_000

REVISION_FIELDS = [
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
]
SCHEMA = pa.schema(
    REVISION_FIELDS
    + [
        ("is_reverted", pa.bool_()),
        ("reverted_by_revision_id", pa.int64()),
        ("is_revert", pa.bool_()),
    ]
)


def iter_parquet_revisions(path: Path) -> Iterator[dict]:
    columns = [name for name, _ in REVISION_FIELDS]
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=ROW_GROUP_SIZE):
        yield from batch.to_pylist()


def write_labels(revisions: Iterable[dict], output_path: Path) -> dict[str, int]:
    """Stream `revisions` through revert detection into `output_path`."""
    stats = {"revisions": 0, "pages": 0, "reverted": 0, "reverts": 0}
    last_page_id = None
    with RowGroupWriter(output_path, SCHEMA, ROW_GROUP_SIZE) as out:
        for row in detect_reverts(revisions):
            out.append(row)
            stats["revisions"] += 1
            stats["reverted"] += row["is_reverted"]
            stats["reverts"] += row["is_revert"]
            if row["page_id"] != last_page_id:
                stats["pages"] += 1
                last_page_id = row["page_id"]
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--from-parquet",
        type=Path,
        help="re-label revisions from an existing labels file instead of the dump",
    )
    args = parser.parse_args(argv)

    if args.from_parquet:
        print(f"Reading revision metadata from {args.from_parquet}")
        revisions = iter_parquet_revisions(args.from_parquet)
    else:
        if not RAW_PATH.exists():
            print(f"Downloading {DUMP_URL} -> {RAW_PATH}")
            download(DUMP_URL, RAW_PATH)
        else:
            print(f"Using existing dump at {RAW_PATH}")
        print("Streaming + parsing revisions (mainspace only)...")
        revisions = iter_mainspace_revisions(str(RAW_PATH))

    print("Running identity-revert detection...")
    start = time.monotonic()
    stats = write_labels(revisions, OUTPUT_PATH)
    elapsed = time.monotonic() - start
    total = stats["revisions"]
    print(
        f"Labeled {total:,} mainspace revisions across {stats['pages']:,} pages "
        f"in {elapsed:,.1f}s"
    )
    print(f"  reverted (undone within window): {stats['reverted']:,} ({stats['reverted'] / total:.2%})")
    print(f"  reverts (restore an earlier state): {stats['reverts']:,} ({stats['reverts'] / total:.2%})")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
