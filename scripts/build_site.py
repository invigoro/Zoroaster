"""Assemble the static site: the page (`web/`) plus the latest prophecy and its record.

Copies `web/` into OUT, and:
- from `data/processed/enwiki/v3/daily/` (`prophesy_daily.py`), version 3's
  prophecy, which the home page shows: the newest `D.prophecy.json` as
  `data/prophecy.json`, every day's as `data/prophecies/D.json`, and the
  last ARCHIVE_DAYS together as `data/prophecies.json`, for the page's day
  selector. Each prediction graded so far (`public_grades.py`) carries its
  grade: the outcome, whether it was already known the night before or too
  vague to count, and its line if that passed the guardrails. Nothing else
  of the grade goes out;
- from `data/processed/enwiki/predictions/`, Stage 1's pages likeliest to
  burst, which the "How the prophet works" page shows: the newest `D.json`
  as `data/latest.json`, and as `data/D.json`; the newest `D.outcomes.json`
  as `data/latest.outcomes.json`, and dated;
- if it's been made, the public data behind that page's version 2 sections
  (`v2_report.py --site`), as `data/forecasts.json`;
- from `data/processed/enwiki/logs/daily/` (`run_daily.py`), the newest
  day's run so far, each step's name and outcome, as `data/status.json`,
  for the watchdog (`check_daily.py`).

The dated copies accumulate in OUT, so on the `gh-pages` branch they form
an archive of every prophecy and how it turned out. `--serve` previews the
site at http://localhost:8000 (the page fetches its JSON, which browsers
don't allow from a file:// page).

Usage:
    python scripts/build_site.py [--out DIR] [--serve]
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.daily_predictions import PREDICTIONS_DIR
from scripts.run_daily import LOG_DIR, STATUS_SUFFIX

WEB_DIR = Path("web")
BUILD_DIR = Path("data/processed/enwiki/site")
FORECASTS_PATH = Path("data/processed/enwiki/v2/qwen2.5-1.5b/site_forecasts.json")
PROPHECY_DIR = Path("data/processed/enwiki/v3/daily")  # version 3's daily records (`prophesy_daily.py`)
PUBLIC_SUFFIX = ".prophecy.json"  # the public part of a day's record: its published predictions alone
GRADES_PATH = Path("data/processed/enwiki/v3/grades_live/public.json")  # the live grades, as shown (`public_grades.py`)
GRADE_FIELDS = ("outcome", "already_known", "vague", "line")  # all the site shows of a grade
# The days a reader can look back on from the home page: long enough to see a week's predictions graded, since
# they're graded about once a week once due.
ARCHIVE_DAYS = 60


def newest(directory: Path, suffix: str) -> Path | None:
    """The file with the latest date name `YYYY-MM-DD{suffix}` in `directory`, if any."""
    files = sorted(p for p in directory.glob(f"????-??-??{suffix}") if p.name[:10].replace("-", "").isdigit())
    return files[-1] if files else None


def graded(prophecy: dict, grades: dict[tuple[str, str], dict]) -> dict:
    """A day's public prophecy, each prediction graded so far with its grade beside it. `grades`: by the day and
    the prediction's text as published, which is unique within a day."""
    def with_grade(p: dict) -> dict:
        grade = grades.get((prophecy["date"], p["text"]))
        return p | {"grade": {k: grade[k] for k in GRADE_FIELDS if k in grade}} if grade else p
    return prophecy | {"predictions": [with_grade(p) for p in prophecy["predictions"]]}


def build(out: Path, predictions: Path = PREDICTIONS_DIR, web: Path = WEB_DIR,
          forecasts: Path = FORECASTS_PATH, prophecies: Path = PROPHECY_DIR,
          statuses: Path = LOG_DIR, grades: Path = GRADES_PATH) -> dict[str, str | None]:
    """Write the site into `out`; returns which prophecy, record, forecasts, grades and status data it used."""
    out.mkdir(parents=True, exist_ok=True)
    for path in web.iterdir():
        if path.is_dir():  # e.g. web/img
            shutil.copytree(path, out / path.name, dirs_exist_ok=True)
        else:
            shutil.copy2(path, out / path.name)
    data = out / "data"
    data.mkdir(exist_ok=True)
    used: dict[str, str | None] = {}
    for suffix, latest in ((".json", "latest.json"), (".outcomes.json", "latest.outcomes.json")):
        source = newest(predictions, suffix)
        used[latest] = source.name if source else None
        if source:
            shutil.copy2(source, data / source.name)
            shutil.copy2(source, data / latest)
    used["forecasts.json"] = str(forecasts) if forecasts.exists() else None
    if forecasts.exists():
        shutil.copy2(forecasts, data / "forecasts.json")
    days = sorted(prophecies.glob(f"????-??-??{PUBLIC_SUFFIX}")) if prophecies.exists() else []
    used["prophecy.json"] = days[-1].name if days else None
    used["grades"] = str(grades) if grades.exists() else None
    by_text = {(g["date"], g["text"]): g for g in json.loads(grades.read_text(encoding="utf-8")).values()} \
        if grades.exists() else {}
    if days:
        shown = [graded(json.loads(path.read_text(encoding="utf-8")), by_text) for path in days]
        (data / "prophecy.json").write_text(json.dumps(shown[-1], indent=1, ensure_ascii=False), encoding="utf-8")
        archive = data / "prophecies"
        archive.mkdir(exist_ok=True)
        for path, day in zip(days, shown):
            (archive / path.name.replace(PUBLIC_SUFFIX, ".json")).write_text(
                json.dumps(day, indent=1, ensure_ascii=False), encoding="utf-8")
        recent = list(reversed(shown[-ARCHIVE_DAYS:]))
        (data / "prophecies.json").write_text(json.dumps(recent, ensure_ascii=False), encoding="utf-8")
    status = newest(statuses, STATUS_SUFFIX) if statuses.exists() else None
    used["status.json"] = status.name if status else None
    if status:
        shutil.copy2(status, data / "status.json")
    return used


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=BUILD_DIR)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args(argv)
    used = build(args.out)
    print(f"site in {args.out}: prophecy {used['prophecy.json']}; Stage 1's pages {used['latest.json']}, "
          f"record {used['latest.outcomes.json']}; grades {used['grades']}")
    if args.serve:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(args.out))
        print("previewing at http://localhost:8000 (Ctrl+C to stop)")
        http.server.ThreadingHTTPServer(("localhost", 8000), handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
