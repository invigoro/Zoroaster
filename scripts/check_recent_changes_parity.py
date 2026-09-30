"""Compare recent-changes records with the history dump's, edit by edit.

The deployed pipeline's daily input (`src/ingest/recent_changes.py`) has to
produce the same records as the dumps the models were trained on. The two
sources overlap only while recent changes (30 days) still reach back to the
dump's last hours. This compares one such window, fetched with
`fetch_recent_changes.py` into `data/processed/enwiki/daily/`:
- coverage both ways, by revision id;
- for revisions in both, agreement on every field the features use.

Usage:
    python scripts/check_recent_changes_parity.py [overlap.parquet]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from scripts.build_enwiki_revisions import OUT_DIR as REVISIONS_DIR

OVERLAP_PATH = Path("data/processed/enwiki/daily/rc_dump_overlap.parquet")
FIELDS = ("page_id", "page_title", "parent_id", "timestamp", "user_text", "is_anon", "is_bot", "byte_size", "sha1")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    rc = pq.read_table(argv[0] if argv else OVERLAP_PATH).to_pylist()
    start, end = min(r["timestamp"] for r in rc), max(r["timestamp"] for r in rc)
    dump = ds.dataset(REVISIONS_DIR, format="parquet", partitioning="hive").to_table(
        columns=list(FIELDS) + ["revision_id", "page_created"],
        filter=(pc.field("timestamp") >= start) & (pc.field("timestamp") <= end),
    ).to_pylist()
    by_id = {r["revision_id"]: r for r in dump}
    rc_ids = {r["revision_id"] for r in rc}
    both = [r for r in rc if r["revision_id"] in by_id]
    print(f"window {start} .. {end}: {len(rc):,} recent-changes records, {len(dump):,} dump records, {len(both):,} in both")
    print(f"  only in recent changes: {len(rc) - len(both):,}; only in the dump: {sum(i not in rc_ids for i in by_id):,}")
    disagree = Counter()
    examples: dict[str, tuple] = {}
    for r in both:
        d = by_id[r["revision_id"]]
        for f in FIELDS:
            if r[f] != d[f]:
                disagree[f] += 1
                examples.setdefault(f, (r["revision_id"], r[f], d[f]))
    for f in FIELDS:
        line = f"  {f:12} disagree on {disagree[f]:6,} ({disagree[f] / len(both):.2%})"
        if f in examples:
            line += f"   e.g. revision {examples[f][0]}: recent changes {examples[f][1]!r}, dump {examples[f][2]!r}"
        print(line)
    created = [r for r in both if r["page_created"]]
    print(f"  page creations: {len(created):,}; creation time agrees on "
          f"{sum(r['page_created'] == by_id[r['revision_id']]['page_created'] for r in created):,}")
    bots = Counter((r["is_bot"], by_id[r["revision_id"]]["is_bot"]) for r in both)
    print(f"  bot flag (recent changes, dump): {dict(bots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
