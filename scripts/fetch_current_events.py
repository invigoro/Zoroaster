"""Fetch Wikipedia's Portal:Current events pages for a range of days: the judge's record of what happened, and
the prophet's evidence (PLAN.md §2, decided 2026-10-02).

For the judge, each page is fetched at its latest revision. Editors add a
day's events during the day and for a day or two after, so a page well past
its day is the fuller record. The fetch time and revision are saved with it.
Each day's news items (`src/prophecy/current_events.py`) are written to
`data/processed/enwiki/v3/current_events/D.json`.

With --known, for the prophet: for each day D, the pages of the KNOWN_DAYS
days before it, each as it stood at the end of D-1 (its last revision by
then), never later. Written to `current_events_known/D.json`.

Usage:
    python scripts/fetch_current_events.py --days 2026-09-18 2026-10-01 [--known]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from scripts.build_v3_days import V3_DIR
from src.mediawiki_api import API_URL, api_get
from src.prophecy.current_events import KNOWN_DAYS, items, known_at, page_title

OUT_DIR = V3_DIR / "current_events"
KNOWN_DIR = V3_DIR / "current_events_known"
REVISIONS_PER_REQUEST = 5  # newest first; more are fetched only if these all have their text hidden


def fetch(session: requests.Session, day: date, as_of: datetime | None = None) -> dict:
    """The day's page at its latest revision, or at its last one by `as_of`.

    Revisions whose text Wikipedia has hidden (revision-deleted) are passed over for the last one before them: on
    2026-10-06 the 10-05 page's last 22 revisions by the end of the day had their text hidden."""
    title = page_title(day)
    params = {"action": "query", "prop": "revisions", "rvprop": "content|ids|timestamp", "rvslots": "main",
              "titles": title, "rvlimit": str(REVISIONS_PER_REQUEST), "format": "json", "formatversion": "2"}
    if as_of:
        params |= {"rvdir": "older", "rvstart": as_of.strftime("%Y-%m-%dT%H:%M:%SZ")}
    while True:
        response = api_get(session, API_URL.format(lang="en"), params)
        page = response["query"]["pages"][0]
        shown = [r for r in page.get("revisions") or [] if "content" in r.get("slots", {}).get("main", {})]
        if shown or "continue" not in response:
            break
        params |= response["continue"]  # every revision so far hidden: look further back
    if page.get("missing") or not shown:  # no page, none yet by `as_of`, or none with its text shown
        return {"title": title, "date": day.isoformat(), "missing": True, "items": []}
    revision = shown[0]
    text = revision["slots"]["main"]["content"]
    return {"title": title, "date": day.isoformat(), "revision_id": revision["revid"], "edited": revision["timestamp"],
            "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"), "items": items(text)}


def known(session: requests.Session, day: date) -> dict:
    """What the prophet may read for `day`: the KNOWN_DAYS days' pages before it, as they stood at the end
    of the day before (`known_at`)."""
    moment = known_at(day)
    days = [fetch(session, day - timedelta(days=n), moment) for n in range(KNOWN_DAYS, 0, -1)]
    return {"date": day.isoformat(), "known_at": moment.isoformat(), "days": days}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=date.fromisoformat, nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--known", action="store_true", help="the prophet's evidence: each day's week before, as known")
    args = parser.parse_args(argv)
    out_dir = KNOWN_DIR if args.known else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    first, last = args.days
    for n in range((last - first).days + 1):
        day = first + timedelta(days=n)
        record = known(session, day) if args.known else fetch(session, day)
        (out_dir / f"{day.isoformat()}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
        found = sum(len(d["items"]) for d in record["days"]) if args.known else len(record["items"])
        print(f"  {day}: {found} items" + (" (no page)" if record.get("missing") else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
