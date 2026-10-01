"""The daily run: today's prophecy, yesterday's record, the site.

Run once a day shortly after 00:00 UTC; a Windows scheduled task does it,
windowless via pythonw (PLAN.md §6 step 9). For today's UTC date D:
1. Predict D (`daily_predictions.py`). That first fetches any live days up
   to D-1 that are missing, so a missed day catches up, within the 30 days
   recent changes keep.
2. Score D-1's prophecy, now that D-1 is over (`score_predictions.py`). If
   D-1 was never predicted (the machine was off), predict it first, as it
   would have been then, so the record has no gaps.
3. Build the site (`build_site.py`).
4. Delete the full candidate tables (`D.parquet`, ~50 MB a day) older than
   KEEP_DAYS; the JSON files stay.

Steps already done are skipped, and each step runs even if an earlier one
failed. The exit code is non-zero if any failed. Output goes to
`data/processed/enwiki/logs/daily/D.log`.

Usage:
    python scripts/run_daily.py [--day 2026-10-01]
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOG_DIR = Path("data/processed/enwiki/logs/daily")
KEEP_DAYS = 14

Step = tuple[str, Callable[[], None]]  # a step raises if it fails


def _ok(exit_code: int) -> None:
    if exit_code != 0:
        raise RuntimeError(f"exit code {exit_code}")


def plan(day: date, predictions: Path) -> list[Step]:
    """The steps still to do for `day`, in order."""
    from scripts import build_site, daily_predictions, score_predictions

    steps: list[Step] = []
    prev = day - timedelta(days=1)
    if not (predictions / f"{prev}.outcomes.json").exists():
        if not (predictions / f"{prev}.parquet").exists():
            steps.append((f"predict {prev} (missed)", lambda: _ok(daily_predictions.main(["--day", prev.isoformat()]))))
    if not (predictions / f"{day}.json").exists():
        steps.append((f"predict {day}", lambda: _ok(daily_predictions.main(["--day", day.isoformat()]))))
    if not (predictions / f"{prev}.outcomes.json").exists():
        steps.append((f"score {prev}", lambda: _ok(score_predictions.main(["--day", prev.isoformat()]))))
    steps.append(("build the site", lambda: _ok(build_site.main([]))))
    steps.append(("prune old candidate tables", lambda: print(f"removed {prune(predictions, day)} tables")))
    return steps


def prune(predictions: Path, day: date, keep_days: int = KEEP_DAYS) -> int:
    """Delete `YYYY-MM-DD.parquet` files more than `keep_days` before `day`; returns how many."""
    removed = 0
    for path in predictions.glob("????-??-??.parquet"):
        if date.fromisoformat(path.stem) < day - timedelta(days=keep_days):
            path.unlink()
            removed += 1
    return removed


def run(steps: list[Step], log: Callable[[str], object] = print) -> int:
    """Run every step, logging each outcome; 1 if any failed, else 0."""
    failed = 0
    for name, step in steps:
        start = time.monotonic()
        try:
            step()
            log(f"== {name}: ok ({time.monotonic() - start:,.0f}s)")
        except (Exception, SystemExit):
            failed += 1
            log(f"== {name}: FAILED ({time.monotonic() - start:,.0f}s)\n{traceback.format_exc()}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--day", type=date.fromisoformat, default=datetime.now(timezone.utc).date())
    args = parser.parse_args(argv)
    from scripts.daily_predictions import PREDICTIONS_DIR

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_DIR / f"{args.day.isoformat()}.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log_file  # pythonw has no console; everything goes to the day's log
    print(f"=== run_daily for {args.day}, started {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    status = run(plan(args.day, PREDICTIONS_DIR))
    print(f"=== finished {datetime.now(timezone.utc).isoformat(timespec='seconds')}, exit code {status}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
