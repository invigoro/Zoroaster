"""Fetch each cited page as it stood at the end of a prediction's due day, for grading (PLAN.md §6 step 11).

A prediction made for day D may come due up to a week later (`prophet.HORIZON`).
It's graded on its due day, against what its cited page gained between the
end of D-1, as the prophet saw it, and the end of the due day. For predictions
due on D itself, the examples already hold that change (`fetch_v2_examples.py`).

For each (page, D, due day) of a run's predictions due after D, up to LAST_DAY
(the last day whose Portal:Current events are fetched), this:
1. finds the page's last revision before the end of the due day (UTC);
2. fetches it, and derives its change from the page at the end of D-1
   (`src/forecast/changes.day_change`).

Writes `data/processed/enwiki/v3/due_pages/<run>.parquet`, with the columns
`grading_packs.py` reads from the examples: page_title, date (D), due,
prompt_id, end_id, sections, section_chars, kinds, prose and blocks.

Usage:
    python scripts/fetch_due_pages.py --prophecies data/processed/enwiki/v3/prophecies_run8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, time as day_time, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from scripts.build_v3_days import V3_DIR
from scripts.fetch_v2_examples import _capped
from scripts.prophesy import EXAMPLES_DIR
from src.forecast.changes import day_change
from src.mediawiki_api import API_URL, PAUSE_SECONDS, api_get
from src.stage2.fetch import fetch_contents

DUE_DIR = V3_DIR / "due_pages"
LAST_DAY = date(2026, 10, 1)
_LIST = pa.list_(pa.string())
SCHEMA = pa.schema([("page_title", pa.string()), ("date", pa.date32()), ("due", pa.date32()),
                    ("prompt_id", pa.int64()), ("end_id", pa.int64()), ("sections", _LIST),
                    ("section_chars", pa.list_(pa.int64())), ("kinds", _LIST), ("prose", pa.string()), ("blocks", _LIST)])


def day_end(day: date) -> datetime:
    """The last second of `day`, in UTC."""
    return datetime.combine(day, day_time(23, 59, 59), tzinfo=timezone.utc)


def revision_at(session: requests.Session, title: str, moment: datetime) -> int | None:
    """The id of `title`'s last revision at or before `moment`, or None if it had none."""
    params = {"action": "query", "prop": "revisions", "titles": title, "rvlimit": "1", "rvdir": "older",
              "rvstart": moment.strftime("%Y-%m-%dT%H:%M:%SZ"), "rvprop": "ids", "format": "json",
              "formatversion": "2", "maxlag": "5"}
    pages = api_get(session, API_URL.format(lang="en"), params, PAUSE_SECONDS).get("query", {}).get("pages", [])
    revisions = pages[0].get("revisions", []) if pages else []
    return revisions[0]["revid"] if revisions else None


def due_targets(prophecies: Path, last_day: date = LAST_DAY) -> list[tuple[str, date, date]]:
    """(page title, D, due day) for each of a run's predictions due after D and no later than `last_day`."""
    found = set()
    for path in sorted(prophecies.glob("????-??-??.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        made = date.fromisoformat(record["date"])
        for p in record["predictions"]:
            due = date.fromisoformat(p.get("due") or record["date"])
            if made < due <= last_day:
                found |= {(title, made, due) for title in p["evidence"]}
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prophecies", type=Path, required=True, help="a run's day records")
    args = parser.parse_args(argv)
    start = time.monotonic()
    targets = due_targets(args.prophecies)
    wanted = {(t.replace(" ", "_"), made) for t, made, _ in targets}
    rows = pq.read_table(EXAMPLES_DIR, columns=["page_title", "date", "prompt_id"]).to_pylist()
    before = {(r["page_title"], r["date"]): r["prompt_id"] for r in rows if (r["page_title"], r["date"]) in wanted}
    session = requests.Session()
    ends = {(t, due): revision_at(session, t, day_end(due)) for t, _, due in targets}
    # Both ends whole: the examples keep the page at the end of D-1 only up to 300,000 characters.
    ids = {i for i in [*ends.values(), *before.values()] if i}
    texts = dict(fetch_contents(sorted(ids), session=session))
    out = []
    for title, made, due in targets:
        prompt_id, end_id = before.get((title.replace(" ", "_"), made)), ends[(title, due)]
        if texts.get(prompt_id) is None or texts.get(end_id) is None:
            print(f"  skipped {title} ({made} -> {due}): no page text", flush=True)
            continue
        change = day_change(texts[prompt_id], texts[end_id])
        out.append({"page_title": title, "date": made, "due": due, "prompt_id": prompt_id, "end_id": end_id,
                    "sections": change["sections"], "section_chars": change["section_chars"], "kinds": change["kinds"],
                    "prose": change["prose"], "blocks": _capped(change["blocks"])})
    DUE_DIR.mkdir(parents=True, exist_ok=True)
    path = DUE_DIR / f"{args.prophecies.name}.parquet"
    pq.write_table(pa.Table.from_pylist(out, schema=SCHEMA), path, compression="zstd")
    print(f"{len(out)} of {len(targets)} page changes up to their due day -> {path} ({time.monotonic() - start:,.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
