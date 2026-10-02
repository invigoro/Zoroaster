"""Fetch Wikipedia's Portal:Current events pages for a range of days, as the judge's record of what happened.

Each page is fetched at its latest revision. Editors add a day's events during
the day and for a day or two after, so a page well past its day is the
fuller record. The fetch time and revision are saved with it. Each day's
news items (`src/prophecy/current_events.py`) are written to
`data/processed/enwiki/v3/current_events/D.json`.

Usage:
    python scripts/fetch_current_events.py --days 2026-09-18 2026-10-01
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
from src.prophecy.current_events import items, page_title

OUT_DIR = V3_DIR / "current_events"


def fetch(session: requests.Session, day: date) -> dict:
    title = page_title(day)
    data = api_get(session, API_URL.format(lang="en"), {
        "action": "query", "prop": "revisions", "rvprop": "content|ids|timestamp", "rvslots": "main",
        "titles": title, "format": "json", "formatversion": "2"})
    page = data["query"]["pages"][0]
    if page.get("missing"):
        return {"title": title, "date": day.isoformat(), "missing": True, "items": []}
    revision = page["revisions"][0]
    text = revision["slots"]["main"]["content"]
    return {"title": title, "date": day.isoformat(), "revision_id": revision["revid"], "edited": revision["timestamp"],
            "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"), "items": items(text)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=date.fromisoformat, nargs=2, required=True, metavar=("FIRST", "LAST"))
    args = parser.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    first, last = args.days
    for n in range((last - first).days + 1):
        day = first + timedelta(days=n)
        record = fetch(session, day)
        (OUT_DIR / f"{day.isoformat()}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  {day}: {len(record['items'])} items" + (" (no page)" if record.get("missing") else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
