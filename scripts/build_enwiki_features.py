"""Build English Wikipedia Stage 1 data: a sampled panel, a full-day
evaluation set, co-burst and site-wide daily totals.

Same features as the test corpus (`src.features.activity`; see
`build_test_features.py`), computed per page bucket in parallel from
`build_enwiki_labels.py`'s output. Differences, all forced by scale or by
the source:
- **Page sampling.** Panel and evaluation rows only for pages chosen by
  `hash_sampled(page_id, PAGE_SAMPLE_RATE)`. Co-burst and site-wide totals
  still use every page.
- **Shorter panel.** Panel rows only cover the train and validation windows.
  The evaluation set is every sampled page on every 13th test day.
- **Existence from the window.** A page exists from its first revision in
  the history window, which starts a year before the first prediction day.
  Pages with no revision in the window are outside the scored universe.
  Page age still comes from the page's creation date.
- **Source bot flags.** Bots are flagged by the source (user group or name),
  not by the name heuristic.

Windows come from `SPLIT_DAYS` (6 months each for train, validation and
test, with the usual 90-day embargoes; `src.stage1.splits`).

Outputs, in `data/processed/enwiki/`:
- `stage1_panel/part-NNN.parquet`
- `stage1_eval_days/part-NNN.parquet`
- `burst_days/part-NNN.parquet`: every burst page-day, for inspection or
  neighbor features later
- `co_burst.parquet`, `site_edits.parquet` and `stage1_panel.json`

Usage:
    python scripts/build_enwiki_features.py [--workers N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import date
from itertools import groupby
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts.build_enwiki_labels import EDITOR_DAYS_DIR, labels_path
from scripts.build_enwiki_revisions import N_BUCKETS, OUT_DIR as REVISIONS_DIR
from scripts.build_test_features import CO_BURST_SCHEMA, sample_days
from src.features.activity import MASS_EDITOR_PAGES_PER_DAY, PageActivity, day_ordinal
from src.features.bursts import CoBurstCounter
from src.ingest.revert_detect import TIME_WINDOW
from src.ingest.sampling import hash_sampled
from src.parquet_io import RowGroupWriter
from src.stage1.panel import PANEL_BASE_SCHEMA, PANEL_SCHEMA, add_co_burst, panel_row
from src.stage1.splits import EVAL_DAY_STRIDE, make_splits

OUT = Path("data/processed/enwiki")
PANEL_DIR = OUT / "stage1_panel"
EVAL_DIR = OUT / "stage1_eval_days"
BURST_DIR = OUT / "burst_days"
CO_BURST_PATH = OUT / "co_burst.parquet"
SITE_PATH = OUT / "site_edits.parquet"
META_PATH = OUT / "stage1_panel.json"
MASS_PATH = OUT / "mass_editor_days.parquet"

PAGE_SAMPLE_RATE = 0.2
NEGATIVE_SAMPLE_RATE = 0.005
SEED = 1234
SPLIT_DAYS = {"test_days": 182, "validation_days": 182, "train_days": 182}

BURST_SCHEMA = pa.schema(
    [
        ("page_id", pa.int64()),
        ("date", pa.date32()),
        ("editors", pa.int32()),
        ("is_burst", pa.bool_()),
        ("editors_excl_mass", pa.int32()),
        ("is_burst_excl_mass", pa.bool_()),
    ]
)
SITE_SCHEMA = pa.schema([("date", pa.date32()), ("edits", pa.int64())])

_worker: dict = {}  # per-process settings, set by _init_worker


def _init_worker(settings: dict) -> None:
    mass = pq.read_table(MASS_PATH)
    _worker.update(settings)
    _worker["mass"] = frozenset(zip(mass["user_text"].to_pylist(), mass["day"].to_pylist()))


def _base_path(directory: Path, bucket: int) -> Path:
    return directory / f"part-{bucket:03d}.base.parquet"


def build_bucket(bucket: int) -> dict:
    """Pass 1 for one bucket: panel/eval rows without co-burst, burst days, and daily totals."""
    s = _worker
    panel_start, panel_end = s["panel_start"], s["panel_end"]
    counts = {"pages": Counter(), "multi": Counter(), "excl_mass": Counter(), "site": Counter()}
    stats: Counter[str] = Counter()
    table = pq.read_table(labels_path(bucket))
    rows = (row for batch in table.to_batches(max_chunksize=200_000) for row in batch.to_pylist())
    with RowGroupWriter(_base_path(PANEL_DIR, bucket), PANEL_BASE_SCHEMA) as panel_out, RowGroupWriter(
        _base_path(EVAL_DIR, bucket), PANEL_BASE_SCHEMA
    ) as eval_out, RowGroupWriter(BURST_DIR / f"part-{bucket:03d}.parquet", BURST_SCHEMA) as burst_out:
        for page_id, page_rows in groupby(rows, key=lambda r: r["page_id"]):
            revisions = list(page_rows)
            created = revisions[0]["page_created"]
            page = PageActivity(revisions, s["mass"], day_ordinal(created) if created else None)
            stats["pages"] += 1
            for day, n in page.counts["edits"].items():
                counts["site"][day] += n
            for day in page.burst_days | page.burst_days_excl_mass:
                in_regular, in_excl = day in page.burst_days, day in page.burst_days_excl_mass
                if in_regular:
                    counts["pages"][day] += 1
                    counts["multi"][day] += page.editor_count(day) >= 2
                counts["excl_mass"][day] += in_excl
                burst_out.append({
                    "page_id": page_id, "date": date.fromordinal(day), "editors": page.editor_count(day),
                    "is_burst": in_regular, "editors_excl_mass": page.editor_count_excl_mass(day),
                    "is_burst_excl_mass": in_excl,
                })
            if not hash_sampled(page_id, PAGE_SAMPLE_RATE):
                continue
            stats["sampled_pages"] += 1
            first = max(page.first_day + 1, panel_start)
            kept = page.counts["kept_edits"]
            positives = [d for d in kept if first <= d <= panel_end]
            rng = random.Random(f"{SEED}:{page_id}")
            negatives = [d for d in sample_days(rng, first, panel_end, NEGATIVE_SAMPLE_RATE) if d not in kept]
            for day in sorted(positives + negatives):
                weight = 1.0 if day in kept else 1.0 / NEGATIVE_SAMPLE_RATE
                panel_out.append(panel_row(page, day, weight, s["last_final_day"]))
            stats["positives"] += len(positives)
            stats["negatives"] += len(negatives)
            stats["page_days"] += max(panel_end - first + 1, 0)
            for day in s["eval_days"]:
                if day > page.first_day:
                    eval_out.append(panel_row(page, day, 1.0, s["last_final_day"]))
                    stats["eval_rows"] += 1
                    stats["eval_positives"] += day in kept
    return {"bucket": bucket, "stats": stats, "counts": counts}


def finish_bucket(args: tuple[int, CoBurstCounter]) -> None:
    """Pass 2 for one bucket: join co-burst (other pages bursting on D-1) onto its rows."""
    bucket, co_burst = args
    for directory in (PANEL_DIR, EVAL_DIR):
        base = _base_path(directory, bucket)
        with RowGroupWriter(directory / f"part-{bucket:03d}.parquet", PANEL_SCHEMA) as out:
            for batch in pq.ParquetFile(base).iter_batches(batch_size=500_000):
                others = [
                    co_burst.others(d.toordinal() - 1, bursting, editors)
                    for d, bursting, editors in zip(
                        batch.column("date").to_pylist(),
                        batch.column("is_burst_1d").to_pylist(),
                        batch.column("editors_1d").to_pylist(),
                    )
                ]
                table = pa.Table.from_batches([batch])
                table = table.append_column(PANEL_SCHEMA.field("co_burst_1d"), pa.array([o[0] for o in others], pa.int32()))
                table = table.append_column(
                    PANEL_SCHEMA.field("co_burst_multi_editor_1d"), pa.array([o[1] for o in others], pa.int32())
                )
                out.write_table(table)
        base.unlink()


def write_mass_editor_days() -> int:
    """Sum per-bucket editor-day page counts (a page lives in one bucket) and keep the mass days."""
    counts = pq.read_table(EDITOR_DAYS_DIR).group_by(["user_text", "day"]).aggregate([("pages", "sum")])
    mass = counts.filter(pc.greater(counts["pages_sum"], MASS_EDITOR_PAGES_PER_DAY))
    pq.write_table(mass.select(["user_text", "day"]), MASS_PATH)
    covered = pc.sum(mass["pages_sum"]).as_py() or 0
    print(
        f"Mass editing (>{MASS_EDITOR_PAGES_PER_DAY} pages/day): {mass.num_rows:,} of {counts.num_rows:,} "
        f"editor-days, covering {covered:,} of {pc.sum(counts['pages_sum']).as_py():,} editor-page-days"
    )
    return mass.num_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    start = time.monotonic()
    for directory in (PANEL_DIR, EVAL_DIR, BURST_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    last_month = sorted(p for p in REVISIONS_DIR.glob("month=*") if not p.name.endswith(".tmp"))[-1]
    data_end = pc.max(pq.read_table(last_month, columns=["timestamp"])["timestamp"]).as_py()
    last_full_day = day_ordinal(data_end) - 1
    last_final_day = last_full_day - TIME_WINDOW.days
    splits = make_splits(date.fromordinal(last_final_day), **SPLIT_DAYS)
    eval_days = [d.toordinal() for d in splits.test.days(EVAL_DAY_STRIDE)]
    print(f"Data ends {data_end}; train {splits.train.start}..{splits.train.end}, validation "
          f"{splits.validation.start}..{splits.validation.end}, test {splits.test.start}..{splits.test.end} "
          f"({len(eval_days)} eval days)")
    write_mass_editor_days()

    settings = {
        "panel_start": splits.train.start.toordinal(),
        "panel_end": splits.validation.end.toordinal(),
        "last_final_day": last_final_day,
        "eval_days": eval_days,
    }
    totals: Counter[str] = Counter()
    co_burst, excl_mass, site = CoBurstCounter(), Counter(), Counter()
    with Pool(args.workers, initializer=_init_worker, initargs=(settings,)) as pool:
        for done, result in enumerate(pool.imap_unordered(build_bucket, range(N_BUCKETS)), start=1):
            totals.update(result["stats"])
            co_burst.pages.update(result["counts"]["pages"])
            co_burst.multi_editor.update(result["counts"]["multi"])
            excl_mass.update(result["counts"]["excl_mass"])
            site.update(result["counts"]["site"])
            if done % 16 == 0:
                print(f"  pass 1: {done}/{N_BUCKETS} buckets ({time.monotonic() - start:,.0f}s)", flush=True)
        pool.map(finish_bucket, [(b, co_burst) for b in range(N_BUCKETS)])

    days = sorted(set(co_burst.pages) | set(excl_mass))
    pq.write_table(
        pa.table(
            {
                "date": [date.fromordinal(d) for d in days],
                "pages_bursting": [co_burst.pages[d] for d in days],
                "pages_bursting_multi_editor": [co_burst.multi_editor[d] for d in days],
                "pages_bursting_excl_mass": [excl_mass[d] for d in days],
            },
            schema=CO_BURST_SCHEMA,
        ),
        CO_BURST_PATH,
    )
    site_days = sorted(site)
    pq.write_table(
        pa.table({"date": [date.fromordinal(d) for d in site_days], "edits": [site[d] for d in site_days]},
                 schema=SITE_SCHEMA),
        SITE_PATH,
    )
    META_PATH.write_text(json.dumps({
        "data_end": data_end,
        "last_panel_day": date.fromordinal(last_full_day).isoformat(),
        "labels_final_through": date.fromordinal(last_final_day).isoformat(),
        "split_days": SPLIT_DAYS,
        "page_sample_rate": PAGE_SAMPLE_RATE,
        "negative_sample_rate": NEGATIVE_SAMPLE_RATE,
        "seed": SEED,
        **{k: totals[k] for k in ("pages", "sampled_pages", "positives", "negatives", "page_days", "eval_rows", "eval_positives")},
    }, indent=2))

    print(f"{totals['pages']:,} pages ({totals['sampled_pages']:,} sampled); panel {totals['positives'] + totals['negatives']:,} "
          f"rows ({totals['positives']:,} positive, {totals['negatives']:,} sampled negative); eval {totals['eval_rows']:,} "
          f"rows ({totals['eval_positives']:,} positive) in {time.monotonic() - start:,.0f}s")
    for label, counter in [("", co_burst.pages), (" excluding mass editing", excl_mass)]:
        print(f"Top co-burst dates (pages bursting{label}):")
        for day, count in counter.most_common(8):
            print(f"  {date.fromordinal(day)}: {count:,} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
