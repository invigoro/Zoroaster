"""Version 3's development days: the site's own prophecies, as the prophet would read them.

For each day D from DEV_FIRST to DEV_LAST, the daily job saved two things:
- the whole wiki's top 100 (`predictions/D.json`);
- every candidate's features (`D.parquet`).

That's exactly what the prophet will read each night, so these days are the
development set (PLAN.md §6 step 11). They're also long after the training
data of both the burst model and the language models, so neither can know
what happened.

Nothing about D itself is used to choose the pages: they're the prophecy's top
TOP, edited on D or not. Version 2's examples were chosen among pages edited
on D, which would tell the prophet that something happened.

For each page-day, this records:
- its revision ids at the ends of D-2 and D-1, and its last never-reverted
  one on D. If nothing on D survived, D ends where D-1 did. Revisions come
  from the history dumps and the live days, with reverts detected on the
  merged history, as the daily job does (`src/deploy/daily.py`).
- its Stage 1 signals, from `D.parquet` (`D.json` lacks `edits_30d`, which
  version 2's prompt shows);
- its linked pages that burst on D-1 (`is_burst_1d` in `D.parquet`), most
  editors first.

Writes `data/processed/enwiki/v3/days.parquet`, which
`fetch_v2_examples.py --targets ... --out ...` turns into examples.

Usage:
    python scripts/build_v3_days.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts import build_enwiki_features as enwiki
from scripts.build_enwiki_revisions import N_BUCKETS, bucket_files
from scripts.build_v2_targets import day_bounds
from scripts.daily_predictions import PREDICTIONS_DIR
from scripts.fetch_recent_changes import LIVE_DIR
from src.deploy.daily import COLUMNS, LIVE_START, _merged_history, live_files
from src.ingest.revert_detect import detect_page_reverts
from src.stage2.examples import MAX_NEIGHBORS_SHOWN

V3_DIR = Path("data/processed/enwiki/v3")
DAYS_PATH = V3_DIR / "days.parquet"
DEV_FIRST, DEV_LAST = date(2026, 9, 18), date(2026, 10, 1)
TOP = 100  # all the daily job saves: world events are rarer than sports among the top pages (PLAN.md §2)
SIGNALS = ("edits_1d", "edits_7d", "edits_30d", "editors_1d", "burst_z_1d", "is_burst_1d", "page_age_days")


def bounds_or_unchanged(page_days: list[tuple[int, date]], revisions: dict[int, list]) -> list[dict]:
    """`day_bounds`, except that a page with nothing kept on D ends D as it began it."""
    out = day_bounds(page_days, revisions)
    for b in out:
        if b["end_id"] is None:
            b["end_id"] = b["prompt_id"]
    return out


def neighbors_bursting(pages: list[int], links: pa.Table, day_table: pa.Table, max_shown: int = MAX_NEIGHBORS_SHOWN
                       ) -> dict[int, list[str]]:
    """Each page's linked pages (either direction) that burst the day before, most editors first, as titles."""
    wanted = pa.array(pages, pa.int32())
    links = links.filter(pc.or_(pc.is_in(links["source"], wanted), pc.is_in(links["target"], wanted)))
    linked: dict[int, set[int]] = {}
    for s, t in zip(links["source"].to_pylist(), links["target"].to_pylist()):
        linked.setdefault(s, set()).add(t)
        linked.setdefault(t, set()).add(s)
    bursting = day_table.filter(day_table["is_burst_1d"])
    editors = dict(zip(bursting["page_id"].to_pylist(), bursting["editors_1d"].to_pylist()))
    titles = dict(zip(bursting["page_id"].to_pylist(), bursting["page_title"].to_pylist()))
    out = {}
    for page in pages:
        hits = sorted(((editors[q], -q) for q in linked.get(page, ()) if q in editors), reverse=True)
        out[page] = [titles[-q] for _, q in hits[:max_shown]]
    return out


def page_revisions(pages: list[int], last_day: date) -> dict[int, list[tuple[str, int, bool]]]:
    """Each page's (timestamp, revision id, is_reverted) through `last_day`, from the dumps and the live days."""
    wanted = pa.array(sorted(set(pages)), pa.int64())
    live = pa.concat_tables([pq.read_table(f, columns=list(COLUMNS)) for f in live_files(LIVE_DIR, LIVE_START, last_day)],
                            promote_options="default")
    by_bucket: dict[int, list[int]] = {}
    for page in wanted.to_pylist():
        by_bucket.setdefault(page % N_BUCKETS, []).append(page)
    out: dict[int, list] = {}
    for bucket, members in sorted(by_bucket.items()):
        table = _merged_history(bucket_files(bucket), live, pa.array(members, pa.int64()), f"{LIVE_START.isoformat()}T00:00:00Z")
        rows: dict[int, list[dict]] = {}
        for row in table.to_pylist():
            rows.setdefault(row["page_id"], []).append(row)
        for page, revs in rows.items():
            labeled = sorted(detect_page_reverts(revs), key=lambda r: (r["timestamp"], r["revision_id"]))
            out[page] = [(r["timestamp"], r["revision_id"], bool(r["is_reverted"])) for r in labeled]
    return out


def main() -> int:
    start = time.monotonic()
    days = [DEV_FIRST + timedelta(days=i) for i in range((DEV_LAST - DEV_FIRST).days + 1)]
    links = pq.read_table(enwiki.LINKS_PATH)
    rows = []
    for day in days:
        prophecy = json.loads((PREDICTIONS_DIR / f"{day.isoformat()}.json").read_text(encoding="utf-8"))
        top = prophecy["pages"][:TOP]
        day_table = pq.read_table(PREDICTIONS_DIR / f"{day.isoformat()}.parquet",
                                  columns=["page_id", "page_title", *dict.fromkeys(SIGNALS)])
        neighbors = neighbors_bursting([p["page_id"] for p in top], links, day_table)
        top_ids = pa.array([p["page_id"] for p in top], pa.int64())
        signals = {r["page_id"]: r for r in day_table.filter(pc.is_in(day_table["page_id"], top_ids)).to_pylist()}
        for p in top:
            rows.append({"page_id": p["page_id"], "date": day, "rank": p["rank"], "score": p["score"],
                         "page_title": p["title"].replace(" ", "_"), **{k: signals[p["page_id"]][k] for k in SIGNALS},
                         "bursting_neighbors": neighbors[p["page_id"]], "split": "dev"})
        print(f"  {day}: top {len(top)}, {sum(bool(neighbors[p['page_id']]) for p in top)} with a bursting linked page "
              f"({time.monotonic() - start:,.0f}s)", flush=True)
    revisions = page_revisions([r["page_id"] for r in rows], DEV_LAST)
    print(f"revisions for {len(revisions):,} pages ({time.monotonic() - start:,.0f}s)", flush=True)
    for row, b in zip(rows, bounds_or_unchanged([(r["page_id"], r["date"]) for r in rows], revisions)):
        row |= b
    kept = [r for r in rows if r["prompt_id"] is not None]
    V3_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(kept), DAYS_PATH)
    print(f"{len(kept):,} page-days ({sum(r['end_id'] != r['prompt_id'] for r in kept):,} changed on the day, "
          f"{len(rows) - len(kept)} dropped without a revision before the day) -> {DAYS_PATH} "
          f"({time.monotonic() - start:,.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
