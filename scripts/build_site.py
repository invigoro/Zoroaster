"""Assemble the static site: the page (`web/`) plus the latest prophecy and its record.

Copies `web/` into OUT, and from `data/processed/enwiki/predictions/`:
- the newest `D.json` as `data/latest.json`, and as `data/D.json`;
- the newest `D.outcomes.json` as `data/latest.outcomes.json`, and dated;
- if it's been made, the public data behind the "How the prophet works"
  page (`v2_report.py --site`), as `data/forecasts.json`.

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
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.daily_predictions import PREDICTIONS_DIR

WEB_DIR = Path("web")
BUILD_DIR = Path("data/processed/enwiki/site")
FORECASTS_PATH = Path("data/processed/enwiki/v2/qwen2.5-1.5b/site_forecasts.json")


def newest(directory: Path, suffix: str) -> Path | None:
    """The file with the latest date name `YYYY-MM-DD{suffix}` in `directory`, if any."""
    files = sorted(p for p in directory.glob(f"????-??-??{suffix}") if p.name[:10].replace("-", "").isdigit())
    return files[-1] if files else None


def build(out: Path, predictions: Path = PREDICTIONS_DIR, web: Path = WEB_DIR,
          forecasts: Path = FORECASTS_PATH) -> dict[str, str | None]:
    """Write the site into `out`; returns which prophecy, record and forecasts data it used."""
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
    return used


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=BUILD_DIR)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args(argv)
    used = build(args.out)
    print(f"site in {args.out}: prophecy {used['latest.json']}, record {used['latest.outcomes.json']}")
    if args.serve:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(args.out))
        print("previewing at http://localhost:8000 (Ctrl+C to stop)")
        http.server.ThreadingHTTPServer(("localhost", 8000), handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
