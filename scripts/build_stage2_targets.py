"""Select Stage 2 examples: kept English Wikipedia edits, each with its Stage 1 signals.

Examples are random Stage 1 positive rows, so each comes with the
point-in-time features of its (page, day):
- train: panel positives in the Stage 1 train window (Jun-Dec 2024);
- validation: panel positives in the validation window (Mar-Sep 2025);
- test: evaluation-set positives, the 14 test days (Dec 2025-Jun 2026).
  These come after the candidate base models' training cutoffs, so the
  events behind them are unseen.

For each (page, day), the target is the page's first kept revision that day
(non-bot, not a revert, never reverted), and its parent is the diff base.
The triggers add the titles of linked pages that were bursting the day
before, most editors first.

Writes `data/processed/enwiki/stage2/targets.parquet`.

Usage:
    python scripts/build_stage2_targets.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from scripts import build_enwiki_features as enwiki
from scripts.build_enwiki_labels import LABELS_DIR
from src.stage1.splits import make_splits

OUT_DIR = Path("data/processed/enwiki/stage2")
TARGETS_PATH = OUT_DIR / "targets.parquet"
SIZES = {"train": 12_000, "validation": 1_000, "test": 3_000}
SEED = 1234
FEATURES = ("edits_1d", "edits_7d", "edits_30d", "is_burst_1d", "burst_z_1d")
MAX_NEIGHBORS = 8


def sample(table: pa.Table, n: int, rng: np.random.Generator) -> pa.Table:
    return table.take(np.sort(rng.choice(table.num_rows, size=min(n, table.num_rows), replace=False)))


def first_kept_revisions(selected: pa.Table) -> pa.Table:
    """The first kept revision of each selected (page_id, date)."""
    labels = ds.dataset(LABELS_DIR, format="parquet")
    kept = labels.to_table(
        columns=["page_id", "page_title", "revision_id", "parent_id", "timestamp"],
        filter=pc.field("page_id").isin(pc.unique(selected["page_id"]))
        & ~pc.field("is_bot") & ~pc.field("is_revert") & ~pc.field("is_reverted"),
    )
    kept = kept.append_column("date", pc.cast(pc.strptime(pc.utf8_slice_codeunits(kept["timestamp"], 0, 10),
                                                         format="%Y-%m-%d", unit="s"), pa.date32()))
    kept = kept.sort_by([("page_id", "ascending"), ("date", "ascending"), ("revision_id", "ascending")])
    first = kept.group_by(["page_id", "date"], use_threads=False).aggregate(
        [("revision_id", "first"), ("parent_id", "first"), ("timestamp", "first"), ("page_title", "first")]
    )
    # Aggregates come first and keys last, so rename by name, not position.
    return first.rename_columns([name.removesuffix("_first") for name in first.schema.names])


def bursting_neighbors(selected: pa.Table) -> list[list[str]]:
    """For each selected row, titles of linked pages bursting the day before."""
    pages = pc.cast(pc.unique(selected["page_id"]), pa.int32())  # the links file stores int32 ids
    links = pq.read_table(enwiki.LINKS_PATH)
    links = links.filter(pc.or_(pc.is_in(links["source"], pages), pc.is_in(links["target"], pages)))
    neighbors: dict[int, set[int]] = {}
    for s, t in zip(links["source"].to_pylist(), links["target"].to_pylist()):
        neighbors.setdefault(s, set()).add(t)
        neighbors.setdefault(t, set()).add(s)
    prev_days = pc.unique(pc.cast(pc.subtract(pc.cast(selected["date"], pa.int32()), 1), pa.date32()))
    bursts = pq.read_table(enwiki.BURST_DIR, columns=["page_id", "date", "editors", "is_burst"])
    bursts = bursts.filter(pc.and_(bursts["is_burst"], pc.is_in(bursts["date"], prev_days)))
    editors = {(p, d): e for p, d, e in zip(*(bursts[c].to_pylist() for c in ("page_id", "date", "editors")))}
    chosen: list[list[int]] = []
    for page, day in zip(selected["page_id"].to_pylist(), selected["date"].to_pylist()):
        prev = day - timedelta(days=1)
        hits = sorted(((editors[(q, prev)], q) for q in neighbors.get(page, ()) if (q, prev) in editors), reverse=True)
        chosen.append([q for _, q in hits[:MAX_NEIGHBORS]])
    ids = sorted({q for row in chosen for q in row})
    titles_table = ds.dataset(LABELS_DIR, format="parquet").to_table(
        columns=["page_id", "page_title"], filter=pc.field("page_id").isin(ids)
    )
    titles = dict(zip(titles_table["page_id"].to_pylist(), titles_table["page_title"].to_pylist()))
    return [[titles.get(q, str(q)) for q in row] for row in chosen]


def main() -> int:
    start = time.monotonic()
    meta = json.loads(enwiki.META_PATH.read_text())
    splits = make_splits(date.fromisoformat(meta["labels_final_through"]), **enwiki.SPLIT_DAYS)
    rng = np.random.default_rng(SEED)
    columns = ["page_id", "date", "y", *FEATURES]
    panel = pq.read_table(enwiki.PANEL_DIR, columns=columns)
    panel = panel.filter(panel["y"])
    evaluation = pq.read_table(enwiki.EVAL_DIR, columns=columns)
    evaluation = evaluation.filter(evaluation["y"])
    parts = []
    for split, rows in [
        ("train", panel.filter(splits.train.mask(panel["date"]))),
        ("validation", panel.filter(splits.validation.mask(panel["date"]))),
        ("test", evaluation),
    ]:
        chosen = sample(rows, SIZES[split], rng).drop_columns(["y"])
        parts.append(chosen.append_column("split", pa.array([split] * chosen.num_rows)))
        print(f"  {split}: {chosen.num_rows:,} of {rows.num_rows:,} positive rows")
    selected = pa.concat_tables(parts)

    revisions = first_kept_revisions(selected)
    selected = selected.join(revisions, keys=["page_id", "date"], join_type="inner")
    selected = selected.sort_by([("split", "ascending"), ("page_id", "ascending"), ("date", "ascending")])
    neighbors = bursting_neighbors(selected)
    selected = selected.append_column("bursting_neighbors", pa.array(neighbors, pa.list_(pa.string())))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(selected, TARGETS_PATH)
    with_neighbors = sum(1 for n in neighbors if n)
    print(f"{selected.num_rows:,} targets ({with_neighbors:,} with a bursting linked page the day before) "
          f"in {time.monotonic() - start:,.0f}s -> {TARGETS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
