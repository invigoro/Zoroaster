"""Build the Simple Wikipedia test corpus's mainspace link graph.

Downloads (if needed) four SQL table dumps, about 154MB in all, from the
same dump run as the revision history (20260701; see PLAN.md §4). Writes
the unique mainspace links, with redirects resolved, to
`data/processed/simplewiki_test_links.parquet` as (source, target) page ids.
See `src.ingest.link_graph` for exactly what is kept.

Usage:
    python scripts/build_test_links.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.download_dump import download
from src.ingest.link_graph import build_link_graph

DUMP_DATE = "20260701"
TABLES = ("page", "redirect", "linktarget", "pagelinks")
RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/simplewiki_test_links.parquet")


def dump_path(table: str) -> Path:
    return RAW_DIR / f"simplewiki-{DUMP_DATE}-{table}.sql.gz"


def main() -> int:
    for table in TABLES:
        path = dump_path(table)
        if not path.exists():
            print(f"Downloading {path.name}")
            download(f"https://dumps.wikimedia.org/simplewiki/{DUMP_DATE}/{path.name}", path)

    start = time.monotonic()
    source, target = build_link_graph(*(dump_path(t) for t in TABLES))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"source": source, "target": target}), OUTPUT_PATH, compression="zstd")

    out_degree = np.bincount(source)
    in_degree = np.bincount(target)
    print(f"{len(source):,} unique mainspace links in {time.monotonic() - start:,.0f}s -> {OUTPUT_PATH}")
    for name, degree in (("out", out_degree), ("in", in_degree)):
        linked = degree[degree > 0]
        print(
            f"  {name}-links: {len(linked):,} pages have any; median {np.median(linked):.0f}, "
            f"p99 {np.percentile(linked, 99):.0f}, max {linked.max():,} (page {degree.argmax()})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
