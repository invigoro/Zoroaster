"""The daily job's core: features for day D from dump history plus live days.

This is the local version of PLAN.md §6 step 9. For a UTC day D, every page
with an edit in the LOOKBACK_DAYS before D is a candidate. Pages dormant
longer essentially never rank. For each candidate:
- its revisions come from the history dumps up to LIVE_START, then from the
  live days (`scripts/fetch_recent_changes.py`), up to the end of D-1;
- revert detection is re-run on that combined history, since a live edit
  can revert a dump-era one;
- then the training code's own `PageActivity` computes the features.

So the only difference from training is the input source (PLAN.md §5,
"Recent changes vs the history dumps"). One gap needs filling: recent
changes carry a creation date only for pages created that day. A page with
no dump revisions (none since HISTORY_START) and no creation in the live
days must predate the history, since its creation revision would otherwise
be there, so its creation day is taken as HISTORY_START, a lower bound on
its age. Without that, a page untouched for years looked brand new: 7.9% of
pages differed from the dump path on `page_age_days`
(`scripts/check_daily_parity.py`). Mass-editor days don't enter any
feature, only the excl-mass burst flags, so scoring doesn't need them.

Work is split by the dumps' page buckets (page_id % N_BUCKETS). Each bucket
writes its feature rows to Parquet.
"""

from __future__ import annotations

from datetime import date, timedelta
from itertools import groupby
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.features.activity import FEATURE_NAMES, MASS_EDITOR_PAGES_PER_DAY, PageActivity, day_ordinal
from src.ingest.revert_detect import detect_page_reverts
from src.parquet_io import RowGroupWriter
from src.stage1.panel import PANEL_BASE_SCHEMA

LIVE_START = date(2026, 9, 1)  # dump data before this day, recent changes from it on
HISTORY_START = date(2023, 6, 1)  # the history dumps' first month (download_enwiki_history.FIRST_MONTH)
LOOKBACK_DAYS = 30
BURST_MIN_EDITORS = 2  # the Stage 1 burst target (train_stage1.py --min-editors)
N_BUCKETS = 128
COLUMNS = ("page_id", "page_title", "revision_id", "parent_id", "timestamp", "user_text", "is_anon", "is_bot",
           "byte_size", "sha1", "page_created")
FEATURE_SCHEMA = pa.schema(
    [("page_id", pa.int64()), ("page_title", pa.string()), ("date", pa.date32())]
    + [PANEL_BASE_SCHEMA.field(name) for name in FEATURE_NAMES]
)


def _iso(day: date) -> str:
    return f"{day.isoformat()}T00:00:00Z"


def live_files(live_dir: Path, first: date, last: date) -> list[Path]:
    """The live day files from `first` to `last`; raises if one is missing, since a gap would skew features."""
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    missing = [d for d in days if not (live_dir / f"{d.isoformat()}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"live days missing: {', '.join(d.isoformat() for d in missing)}")
    return [live_dir / f"{d.isoformat()}.parquet" for d in days]


def split_by_bucket(files: Iterable[Path], out_dir: Path, n_buckets: int = N_BUCKETS) -> None:
    """Rewrite the live day files as one file per page bucket (`bucket=NNN.parquet`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.concat_tables((pq.read_table(f, columns=list(COLUMNS)) for f in files), promote_options="default")
    buckets = pa.array(table["page_id"].to_numpy() % n_buckets)
    for bucket in range(n_buckets):
        part = table.filter(pc.equal(buckets, bucket))
        pq.write_table(part, out_dir / f"bucket={bucket:03d}.parquet", compression="zstd")


def _created_day(revisions: list[dict]) -> int:
    created = next((r["page_created"] for r in revisions if r["page_created"]), None)
    return day_ordinal(created) if created else HISTORY_START.toordinal()


def _merged_history(dump_files: Iterable[Path], live: pa.Table, pages: pa.Array, dump_until: str) -> pa.Table:
    """`pages`' revisions: the dump's before `dump_until`, then `live`'s, sorted by page and revision."""
    live = live.filter(pc.is_in(live["page_id"], value_set=pages))
    dump_parts = []
    for path in dump_files if len(pages) else ():
        part = pq.read_table(path, columns=list(COLUMNS), filters=[("page_id", "in", pages.to_pylist())])
        if part.num_rows:
            dump_parts.append(part.filter(pc.less(part["timestamp"], dump_until)))
    return pa.concat_tables(dump_parts + [live], promote_options="default").sort_by(
        [("page_id", "ascending"), ("revision_id", "ascending")])


def bucket_features(day: date, dump_files: Iterable[Path], live_file: Path, out: Path,
                    dump_until: str | None = None) -> int:
    """Feature rows for day `day` for one bucket's candidate pages, written to `out`; returns the row count.

    `dump_files`: the bucket's history-dump revisions (any months). Rows from
    `dump_until` on (an ISO timestamp, default LIVE_START) are dropped in
    favor of the live data. `live_file`: the bucket's live revisions
    (`split_by_bucket`); rows from `day` on are dropped.
    """
    live = pq.read_table(live_file, columns=list(COLUMNS))
    live = live.filter(pc.less(live["timestamp"], _iso(day)))
    recent = live.filter(pc.greater_equal(live["timestamp"], _iso(day - timedelta(days=LOOKBACK_DAYS))))
    candidates = pa.array(pc.unique(recent["page_id"]).to_pylist(), pa.int64())
    table = _merged_history(dump_files, live, candidates, dump_until or _iso(LIVE_START))
    target = day.toordinal()
    count = 0
    with RowGroupWriter(out, FEATURE_SCHEMA) as writer:
        for page_rows in _pages(table):
            activity = PageActivity(detect_page_reverts(page_rows), frozenset(), _created_day(page_rows))
            writer.append({"page_id": activity.page_id, "page_title": page_rows[-1]["page_title"], "date": day}
                          | activity.features(target))
            count += 1
    return count


def live_mass_editors(day_file: Path, day: date) -> frozenset[tuple[str, int]]:
    """The day's mass editors, from its live records: non-bot editors of more
    than MASS_EDITOR_PAGES_PER_DAY distinct pages.

    Training counts only non-revert edits, which takes each page's revert
    detection. This counts reverts too, so it can only add editors near the
    threshold whose reverts tip them over; reverts never count as edits.
    """
    table = pq.read_table(day_file, columns=["user_text", "page_id", "is_bot"])
    table = table.filter(pc.and_(pc.invert(table["is_bot"]), pc.is_valid(table["user_text"])))
    pages = table.group_by("user_text").aggregate([("page_id", "count_distinct")])
    mass = pages.filter(pc.greater(pages["page_id_count_distinct"], MASS_EDITOR_PAGES_PER_DAY))
    return frozenset((user, day.toordinal()) for user in mass["user_text"].to_pylist())


def bucket_outcomes(day: date, dump_files: Iterable[Path], live_file: Path, pages: list[int],
                    mass: frozenset[tuple[str, int]]) -> list[dict]:
    """Whether each of `pages` (one bucket's) burst on `day`, by the Stage 1
    burst target: a burst with mass editors left out, by BURST_MIN_EDITORS+
    editors. It reads revisions through the end of `day`."""
    live = pq.read_table(live_file, columns=list(COLUMNS))
    live = live.filter(pc.less(live["timestamp"], _iso(day + timedelta(days=1))))
    table = _merged_history(dump_files, live, pa.array(pages, pa.int64()), _iso(LIVE_START))
    target, out = day.toordinal(), []
    for page_rows in _pages(table):
        activity = PageActivity(detect_page_reverts(page_rows), mass, _created_day(page_rows))
        editors = activity.editor_count_excl_mass(target)
        out.append({"page_id": activity.page_id, "edits": activity.counts["edits"][target], "editors": editors,
                    "burst": target in activity.burst_days_excl_mass and editors >= BURST_MIN_EDITORS})
    return out


def _pages(table: pa.Table) -> Iterator[list[dict]]:
    """One page's rows at a time from a table sorted by page, streaming through record batches."""
    rows = (row for batch in table.to_batches(max_chunksize=100_000) for row in batch.to_pylist())
    for _, page_rows in groupby(rows, key=lambda r: r["page_id"]):
        yield list(page_rows)
