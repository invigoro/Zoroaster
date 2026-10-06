"""The watchdog: did today's daily run go through? Run by GitHub Actions every night after it
(`.github/workflows/watchdog.yml`), so a failure reaches the user without their checking the site.

The daily job runs on the prophet's own machine at about 00:30 UTC (`run_daily.py`) and publishes the site, with
each step's outcome as `data/status.json`. This reads the published site, and fails, with what went wrong, if
- the status isn't today's: the job didn't run (the machine was off or asleep), or didn't publish;
- a step failed;
- the prophecy isn't today's.
GitHub emails the repository's owner when a scheduled workflow fails. The machine's log has the details:
`data/processed/enwiki/logs/daily/D.log`.

Standard library only, so the workflow needs nothing installed.

Usage:
    python scripts/check_daily.py [--site https://zoroaster.invigoro.me] [--day 2026-10-06]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone

SITE = "https://zoroaster.invigoro.me"


def problems(status: dict | None, prophecy: dict | None, day: date) -> list[str]:
    """What went wrong with `day`'s run, from the site's published status and prophecy; empty if nothing."""
    found = []
    if status is None:
        found.append("The site has no run status (data/status.json).")
    elif status.get("day") != day.isoformat():
        found.append(f"The site's last run is for {status.get('day')}, not {day}: the daily job didn't run, or "
                     "didn't publish. Is the machine on?")
    else:
        failed = [name for name, outcome in status.get("steps", {}).items() if outcome != "ok"]
        if failed:
            found.append(f"{len(failed)} step(s) of {day}'s run failed: {'; '.join(failed)}. "
                         f"See data/processed/enwiki/logs/daily/{day}.log.")
    if prophecy is None:
        found.append("The site has no prophecy (data/prophecy.json).")
    elif prophecy.get("date") != day.isoformat():
        found.append(f"The site's prophecy is for {prophecy.get('date')}, not {day}.")
    return found


def published(site: str, path: str) -> dict | None:
    """A JSON file from the published site, fresh, or None if it isn't there."""
    request = urllib.request.Request(f"{site}/{path}?checked={int(time.time())}",
                                     headers={"User-Agent": "Zoroaster watchdog", "Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", default=SITE)
    parser.add_argument("--day", type=date.fromisoformat, default=datetime.now(timezone.utc).date())
    args = parser.parse_args(argv)
    found = problems(published(args.site, "data/status.json"), published(args.site, "data/prophecy.json"), args.day)
    for problem in found:
        print(f"::error::{problem}")  # an annotation on the run, which GitHub's email shows
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
            lines = [f"- {p}" for p in found] or ["Every step went through, and today's prophecy is up."]
            summary.write("\n".join([f"## The daily run for {args.day}", "", *lines]) + "\n")
    if not found:
        print(f"{args.day}: every step went through, and the prophecy is up.")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
