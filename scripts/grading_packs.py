"""Assemble the evidence for grading each prophecy (PLAN.md §6 step 11, phase 2).

The predictions graded are those kept, and those the quality checks (novelty,
grounding) dropped, so the hand grades can check those checks too
(`src/prophecy/judge.gradable`). Never one that a guardrail dropped, or a copy
or repeat. On the development days the grounding check alone dropped 49 of
81 predictions, which left too few kept ones to check the judge against.

For each prediction in `data/processed/enwiki/v3/prophecies/D.json`, the
pack holds:
- what was known by the end of D-1 (its cited pages' evidence blocks);
- what those pages gained from then to the end of its due day
  (`src/prophecy/judge.py`). For a prediction due on D, that's the
  examples' row; for one due later, `fetch_due_pages.py`'s;
- its due day's Portal:Current events items (`fetch_current_events.py`).

A prediction due after `fetch_due_pages.LAST_DAY` waits until its day's
evidence exists. With --published, only the predictions the selection
published are graded.

It writes `data/processed/enwiki/v3/grades/packs.json`, keyed "D#n" (the
prediction's index in that day's list). It also writes `packs.md`, the same
evidence organized by day, for drafting the hand grades.

Usage:
    python scripts/grading_packs.py [--prophecies data/processed/enwiki/v3/prophecies_run7 --grades data/processed/enwiki/v3/grades_run7] [--published]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq

from scripts.build_v3_days import V3_DIR
from scripts.fetch_current_events import OUT_DIR as EVENTS_DIR
from scripts.prophesy import EXAMPLES_DIR, OUT_DIR as PROPHECIES_DIR
from scripts.fetch_due_pages import DUE_DIR, LAST_DAY
from src.prophecy.judge import gradable, pack

GRADES_DIR = V3_DIR / "grades"
COLUMNS = ["page_title", "date", "prompt_id", "end_id", "sections", "section_chars", "kinds", "prose", "blocks"]


def known_before(record: dict) -> dict[str, str]:
    """Each page's evidence block, as the prophet saw it: blocks are separated by blank lines, in page order."""
    return dict(zip([p["title"] for p in record["pages"]], record["evidence"].split("\n\n")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prophecies", type=Path, default=PROPHECIES_DIR, help="a run's day records")
    parser.add_argument("--grades", type=Path, default=GRADES_DIR, help="where that run's packs go")
    parser.add_argument("--published", action="store_true", help="only the predictions the selection published")
    args = parser.parse_args(argv)
    rows = pq.read_table(EXAMPLES_DIR, columns=COLUMNS).to_pylist()
    by_day: dict[tuple[str, str], dict[str, dict]] = {}
    for r in rows:
        by_day.setdefault((r["date"].isoformat(),) * 2, {})[r["page_title"].replace("_", " ")] = r
    due_path = DUE_DIR / f"{args.prophecies.name}.parquet"  # predictions due after their day (`fetch_due_pages.py`)
    if due_path.exists():
        for r in pq.read_table(due_path).to_pylist():
            by_day.setdefault((r["date"].isoformat(), r["due"].isoformat()), {})[r["page_title"]] = r
    events: dict[str, list[str]] = {}
    packs, markdown, not_due = {}, ["# Prophecies to grade", ""], 0
    for path in sorted(args.prophecies.glob("????-??-??.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        day = record["date"]
        before = known_before(record)
        to_grade = [(n, p) for n, p in enumerate(record["predictions"])
                    if gradable(p) and (p.get("published") or not args.published)]
        markdown += [f"## {day}", ""]
        for n, prediction in to_grade:
            due = prediction.get("due") or day
            if due > LAST_DAY.isoformat():
                not_due += 1  # graded once its day's evidence exists
                continue
            if due not in events:
                events[due] = json.loads((EVENTS_DIR / f"{due}.json").read_text(encoding="utf-8"))["items"]
            key = f"{day}#{n}"
            packs[key] = pack(prediction, day, before, by_day.get((day, due), {}), events[due])
            p = packs[key]
            status = "kept" if p["kept"] else f"dropped: {'; '.join(p['dropped_because'])}"
            markdown += [f"### {key} [{status}; due {due}]: {p['prediction']}",
                         f"Question: {p['question']} (confidence: {p['confidence']})",
                         f"Current events on {due}:", *[f"- {e}" for e in events[due]], ""]
            for title in p["cited"]:
                markdown += [f"Before ({title}):", p["known_before"].get(title, "(not in the evidence)"), "",
                             f"On the day ({title}; diff: {p['diffs'].get(title)}):",
                             p["day_brought"].get(title, "(no change data)"), ""]
    args.grades.mkdir(parents=True, exist_ok=True)
    (args.grades / "packs.json").write_text(json.dumps(packs, indent=1, ensure_ascii=False), encoding="utf-8")
    (args.grades / "packs.md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"{len(packs)} predictions to grade ({sum(p['kept'] for p in packs.values())} kept) from "
          f"{len(list(args.prophecies.glob('????-??-??.json')))} days -> {args.grades / 'packs.json'} and packs.md; "
          f"{not_due} not due by {LAST_DAY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
