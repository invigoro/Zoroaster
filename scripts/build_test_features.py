"""Build activity (burst/co-burst) features and a stratified page sample
from the already-downloaded test revert-label dataset.

Does not download anything — it runs entirely against the local
`data/processed/simplewiki_test_revert_labels.parquet` produced by
`build_test_revert_labels.py`. This validates the feature/sampling logic on
a small corpus before it's pointed at a full-scale dataset later.

Usage:
    python scripts/build_test_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq

from src.common import is_bot_edit
from src.features.bursts import attach_co_burst, compute_co_burst_counts, compute_page_bursts
from src.ingest.sampling import sample_pages, stratify, summarize_pages

INPUT_PATH = Path("data/processed/simplewiki_test_revert_labels.parquet")
FEATURES_PATH = Path("data/processed/simplewiki_test_activity_features.parquet")
MANIFEST_PATH = Path("data/processed/simplewiki_test_sample_manifest.json")

PER_STRATUM_SAMPLE_SIZE = 25
SAMPLE_SEED = 1234

FEATURES_SCHEMA = pa.schema(
    [
        ("page_id", pa.int64()),
        ("date", pa.string()),
        ("edit_count", pa.int64()),
        ("baseline_mean", pa.float64()),
        ("baseline_std", pa.float64()),
        ("zscore", pa.float64()),
        ("is_burst", pa.bool_()),
        ("co_burst_count", pa.int64()),
    ]
)


def main() -> int:
    table = pq.read_table(INPUT_PATH)
    rows = table.to_pylist()
    retained = [r for r in rows if not r["is_reverted"]]
    human_retained = [r for r in retained if not is_bot_edit(r["user_text"])]
    print(
        f"{len(rows):,} total revisions, {len(retained):,} retained (non-reverted), "
        f"{len(human_retained):,} retained + non-bot (used for burst detection)"
    )

    print("Computing burst/co-burst features...")
    page_bursts = compute_page_bursts(human_retained)
    co_burst_totals = compute_co_burst_counts(page_bursts)
    enriched = attach_co_burst(page_bursts)

    pages_with_burst = {a["page_id"] for a in enriched if a["is_burst"]}
    print(
        f"{sum(1 for a in enriched if a['is_burst']):,} burst page-days "
        f"across {len(pages_with_burst):,} pages"
    )

    print("Top co-burst dates (most pages bursting simultaneously):")
    for date, count in sorted(co_burst_totals.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"  {date}: {count} pages")

    columns: dict[str, list] = {name: [] for name in FEATURES_SCHEMA.names}
    for row in enriched:
        for name in columns:
            columns[name].append(row[name])
    pq.write_table(
        pa.table(columns, schema=FEATURES_SCHEMA), FEATURES_PATH, compression="zstd"
    )
    print(f"Wrote {FEATURES_PATH}")

    summaries = summarize_pages(retained, pages_with_burst)
    strata = stratify(summaries)
    print("\nStratum sizes:")
    for key, page_ids in sorted(strata.items(), key=lambda kv: kv[0]):
        print(f"  {key}: {len(page_ids)} pages")

    sample = sample_pages(strata, per_stratum=PER_STRATUM_SAMPLE_SIZE, seed=SAMPLE_SEED)
    manifest = {
        "source": str(INPUT_PATH),
        "seed": SAMPLE_SEED,
        "per_stratum_sample_size": PER_STRATUM_SAMPLE_SIZE,
        "strata": sample,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    total_sampled = sum(len(v) for v in sample.values())
    print(f"\nSampled {total_sampled} pages across {len(sample)} strata -> {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
