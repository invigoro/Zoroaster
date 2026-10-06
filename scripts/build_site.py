"""Assemble the static site: the page (`web/`) plus the latest prophecy and its record.

Copies `web/` into OUT, and:
- from `data/processed/enwiki/v3/daily/` (`prophesy_daily.py`), version 3's
  prophecy, which the home page shows: the newest `D.prophecy.json` as
  `data/prophecy.json`, every day's as `data/prophecies/D.json`, and the
  last ARCHIVE_DAYS together as `data/prophecies.json`, for the page's day
  selector;
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
ARCHIVE_DAYS = 14  # the days a reader can look back on from the home page


def newest(directory: Path, suffix: str) -> Path | None:
    """The file with the latest date name `YYYY-MM-DD{suffix}` in `directory`, if any."""
    files = sorted(p for p in directory.glob(f"????-??-??{suffix}") if p.name[:10].replace("-", "").isdigit())
    return files[-1] if files else None


def build(out: Path, predictions: Path = PREDICTIONS_DIR, web: Path = WEB_DIR,
          forecasts: Path = FORECASTS_PATH, prophecies: Path = PROPHECY_DIR,
          statuses: Path = LOG_DIR) -> dict[str, str | None]:
    """Write the site into `out`; returns which prophecy, record, forecasts and status data it used."""
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
    if days:
        shutil.copy2(days[-1], data / "prophecy.json")
        archive = data / "prophecies"
        archive.mkdir(exist_ok=True)
        for path in days:
            shutil.copy2(path, archive / path.name.replace(PUBLIC_SUFFIX, ".json"))
        recent = [json.loads(path.read_text(encoding="utf-8")) for path in reversed(days[-ARCHIVE_DAYS:])]
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
          f"record {used['latest.outcomes.json']}")
    if args.serve:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(args.out))
        print("previewing at http://localhost:8000 (Ctrl+C to stop)")
        http.server.ThreadingHTTPServer(("localhost", 8000), handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
