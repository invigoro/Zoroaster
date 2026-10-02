"""Assemble the evidence for grading each prophecy (PLAN.md §6 step 11, phase 2).

The predictions graded are those kept, and those the quality checks (novelty,
grounding) dropped, so the hand grades can check those checks too
(`src/prophecy/judge.gradable`). Never one that a guardrail dropped, or a copy
or repeat. On the development days the grounding check alone dropped 49 of
81 predictions, which left too few kept ones to check the judge against.

For each prediction in `data/processed/enwiki/v3/prophecies/D.json`, the
pack holds:
- what was known by the end of D-1 (its cited pages' evidence blocks);
- what D brought to those pages (`src/prophecy/judge.py`);
- D's Portal:Current events items (`fetch_current_events.py`).

It writes `data/processed/enwiki/v3/grades/packs.json`, keyed "D#n" (the
prediction's index in that day's list). It also writes `packs.md`, the same
evidence organized by day, for drafting the hand grades.

Usage:
    python scripts/grading_packs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq

from scripts.build_v3_days import V3_DIR
from scripts.fetch_current_events import OUT_DIR as EVENTS_DIR
from scripts.prophesy import EXAMPLES_DIR, OUT_DIR as PROPHECIES_DIR
from src.prophecy.judge import gradable, pack

GRADES_DIR = V3_DIR / "grades"
COLUMNS = ["page_title", "date", "prompt_id", "end_id", "sections", "section_chars", "kinds", "prose", "blocks"]


def known_before(record: dict) -> dict[str, str]:
    """Each page's evidence block, as the prophet saw it: blocks are separated by blank lines, in page order."""
    return dict(zip([p["title"] for p in record["pages"]], record["evidence"].split("\n\n")))


def main() -> int:
    rows = pq.read_table(EXAMPLES_DIR, columns=COLUMNS).to_pylist()
    by_day: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_day.setdefault(r["date"].isoformat(), {})[r["page_title"].replace("_", " ")] = r
    packs, markdown = {}, ["# Prophecies to grade", ""]
    for path in sorted(PROPHECIES_DIR.glob("????-??-??.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        day = record["date"]
        events = json.loads((EVENTS_DIR / f"{day}.json").read_text(encoding="utf-8"))["items"]
        before = known_before(record)
        to_grade = [(n, p) for n, p in enumerate(record["predictions"]) if gradable(p)]
        markdown += [f"## {day}", "", "Current events:", *[f"- {e}" for e in events], ""]
        for n, prediction in to_grade:
            key = f"{day}#{n}"
            packs[key] = pack(prediction, day, before, by_day.get(day, {}), events)
            p = packs[key]
            status = "kept" if p["kept"] else f"dropped: {'; '.join(p['dropped_because'])}"
            markdown += [f"### {key} [{status}]: {p['prediction']}",
                         f"Question: {p['question']} (confidence: {p['confidence']})", ""]
            for title in p["cited"]:
                markdown += [f"Before ({title}):", p["known_before"].get(title, "(not in the evidence)"), "",
                             f"On the day ({title}; diff: {p['diffs'].get(title)}):",
                             p["day_brought"].get(title, "(no change data)"), ""]
    GRADES_DIR.mkdir(parents=True, exist_ok=True)
    (GRADES_DIR / "packs.json").write_text(json.dumps(packs, indent=1, ensure_ascii=False), encoding="utf-8")
    (GRADES_DIR / "packs.md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"{len(packs)} predictions to grade ({sum(p['kept'] for p in packs.values())} kept) from "
          f"{len(list(PROPHECIES_DIR.glob('????-??-??.json')))} days -> {GRADES_DIR / 'packs.json'} and packs.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
