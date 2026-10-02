"""Turn the grading packs and the draft grades into documents for the review page.

PLAN.md §6 step 11, phase 2. The review page is a private claude.ai artifact
with a database of three collections:
- `predictions`: one document per kept prediction, with the prophecy, its
  evidence before and after, diff links and the draft grade;
- `days`: one per day, with its Portal:Current events items;
- `reviews`: written by the page when the user confirms or adjusts a grade.

This writes the first two as JSON files under `grades/review_docs/`, to seed
with the ArtifactData tool. Document ids are the pack keys with "#" made "-",
because "#" isn't allowed in a database path.

Usage:
    python scripts/grading_review_docs.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.grading_packs import GRADES_DIR
from src.prophecy.current_events import page_title

OUT = GRADES_DIR / "review_docs"


def doc_id(key: str) -> str:
    return key.replace("#", "-")


def main() -> int:
    packs = json.loads((GRADES_DIR / "packs.json").read_text(encoding="utf-8"))
    drafts = json.loads((GRADES_DIR / "drafts.json").read_text(encoding="utf-8"))
    missing = [k for k in packs if k not in drafts]
    if missing:
        raise SystemExit(f"{len(missing)} predictions have no draft grade yet, e.g. {missing[:3]}")
    for name in ("predictions", "days"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    days: dict[str, list[str]] = {}
    for key, p in packs.items():
        day, n = key.split("#")
        doc = {"date": day, "n": int(n), "prediction": p["prediction"], "question": p["question"],
               "confidence": p["confidence"], "cited": p["cited"], "kept": p["kept"],
               "dropped_because": p["dropped_because"], "before": p["known_before"], "after": p["day_brought"],
               "diffs": p["diffs"], "draft": drafts[key]}
        (OUT / "predictions" / f"{doc_id(key)}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        days[day] = p["current_events"]
    for day, events in days.items():
        url = "https://en.wikipedia.org/wiki/" + page_title(date.fromisoformat(day)).replace(" ", "_")
        doc = {"date": day, "url": url, "events": events}
        (OUT / "days" / f"{day}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    sizes = [f.stat().st_size for f in (OUT / "predictions").glob("*.json")]
    print(f"{len(packs)} prediction documents (largest {max(sizes) / 1024:.0f} KB, {sum(sizes) / 1024:.0f} KB in all) "
          f"and {len(days)} day documents -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
