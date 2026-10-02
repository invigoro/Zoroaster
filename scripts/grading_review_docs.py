"""Turn the grading packs and the draft grades into documents for the review page.

PLAN.md §6 step 11, phase 2. The review page is a private claude.ai artifact
with a database of three collections:
- `predictions`: one document per kept prediction, with the prophecy, the
  day it's due, its evidence before and after, diff links and the draft
  grade;
- `days`: one per day, with its Portal:Current events items. A prediction
  due after its day is graded on its due day's;
- `reviews`: written by the page when the user confirms or adjusts a grade.

This writes the first two as JSON files under a run's `review_docs/`, to seed
with the ArtifactData tool. Document ids are the pack keys with "#" made "-",
because "#" isn't allowed in a database path. A later run's ids start with its
name ("run7-2026-09-24-0"), and its documents carry its name and a label, so
the page can show the runs apart.

Usage:
    python scripts/grading_review_docs.py [--grades data/processed/enwiki/v3/grades_run7 --run run7 --label "..."]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.grading_packs import GRADES_DIR
from src.prophecy.current_events import page_title


def doc_id(key: str, run: str = "") -> str:
    return (f"{run}-" if run else "") + key.replace("#", "-")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grades", type=Path, default=GRADES_DIR, help="the run's grades folder")
    parser.add_argument("--run", default="", help="the run's name, for its document ids; none for the first run")
    parser.add_argument("--label", default="", help="what the page calls the run")
    args = parser.parse_args(argv)
    out = args.grades / "review_docs"
    packs = json.loads((args.grades / "packs.json").read_text(encoding="utf-8"))
    drafts = json.loads((args.grades / "drafts.json").read_text(encoding="utf-8"))
    missing = [k for k in packs if k not in drafts]
    if missing:
        raise SystemExit(f"{len(missing)} predictions have no draft grade yet, e.g. {missing[:3]}")
    for name in ("predictions", "days"):
        (out / name).mkdir(parents=True, exist_ok=True)
    days: dict[str, list[str]] = {}
    for key, p in packs.items():
        day, n = key.split("#")
        due = p.get("due") or day
        doc = {"date": day, "due": due, "n": int(n), "prediction": p["prediction"], "question": p["question"],
               "confidence": p["confidence"], "cited": p["cited"], "kept": p["kept"],
               "dropped_because": p["dropped_because"], "before": p["known_before"], "after": p["day_brought"],
               "diffs": p["diffs"], "draft": drafts[key]} | ({"run": args.run, "run_label": args.label} if args.run else {})
        (out / "predictions" / f"{doc_id(key, args.run)}.json").write_text(json.dumps(doc, ensure_ascii=False),
                                                                          encoding="utf-8")
        days[due] = p["current_events"]  # the due day's events (`grading_packs.py`), filed under that day
    for day, events in days.items():
        url = "https://en.wikipedia.org/wiki/" + page_title(date.fromisoformat(day)).replace(" ", "_")
        doc = {"date": day, "url": url, "events": events}
        (out / "days" / f"{day}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    sizes = [f.stat().st_size for f in (out / "predictions").glob("*.json")]
    print(f"{len(packs)} prediction documents (largest {max(sizes) / 1024:.0f} KB, {sum(sizes) / 1024:.0f} KB in all) "
          f"and {len(days)} day documents -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
