"""Build the English Wikipedia mainspace link graph for the sampled pages.

Uses four SQL table dumps from the 20260901 run, about 11GB, which matches
the end of the MediaWiki history snapshot (2026-09-01). Only links with at
least one end in the Stage 1 page sample
(`build_enwiki_features.PAGE_SAMPLE_RATE`) are kept, since those are all the
link-neighbor features for sampled pages can use, and the full graph (1B+
links) wouldn't fit in memory. Writes (source, target) int32 page ids to
`data/processed/enwiki/links.parquet`. See `src.ingest.link_graph`.

Usage:
    python scripts/build_enwiki_links.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.build_enwiki_features import LINKS_PATH, PAGE_SAMPLE_RATE
from scripts.download_dump import download
from src.ingest.link_graph import build_link_graph
from src.ingest.sampling import hash_sample_mask

DUMP_DATE = "20260901"
TABLES = ("page", "redirect", "linktarget", "pagelinks")
RAW_DIR = Path("data/raw/enwiki_sql")
MAX_PAGE_ID = 200_000_000  # comfortably above English Wikipedia's (~82M in 2026)


def dump_path(table: str) -> Path:
    return RAW_DIR / f"enwiki-{DUMP_DATE}-{table}.sql.gz"


def main() -> int:
    for table in TABLES:
        path = dump_path(table)
        if not path.exists():
            print(f"Downloading {path.name}")
            download(f"https://dumps.wikimedia.org/enwiki/{DUMP_DATE}/{path.name}", path)
    start = time.monotonic()
    keep = hash_sample_mask(MAX_PAGE_ID, PAGE_SAMPLE_RATE)
    source, target = build_link_graph(*(dump_path(t) for t in TABLES), keep=keep)
    assert source.max(initial=0) < MAX_PAGE_ID and target.max(initial=0) < MAX_PAGE_ID
    pq.write_table(
        pa.table({"source": source.astype(np.int32), "target": target.astype(np.int32)}),
        LINKS_PATH,
        compression="zstd",
    )
    sampled_source, sampled_target = keep[source], keep[target]
    print(
        f"{len(source):,} mainspace links touching the {PAGE_SAMPLE_RATE:.0%} page sample "
        f"({sampled_source.sum():,} from a sampled page, {sampled_target.sum():,} to one) "
        f"in {time.monotonic() - start:,.0f}s -> {LINKS_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
