"""Read the review page's saved reviews back into `confirmed.json`: the hand grades the judge is checked against.

PLAN.md §6 step 11, phase 2. The review page (`grading_review_docs.py`) saves a
`reviews/<id>` document for each prediction the user confirms or adjusts.
Claude downloads them with the ArtifactData tool (`list` with `out_dir`) into
`grades/reviews_download/reviews/`. This turns them into `confirmed.json` in a
run's grades folder, keyed like its packs ("D#n"), and says how many were
confirmed as drafted and how many adjusted.

A later run's documents carry its name before the date ("run7-2026-09-24-0").
The first graded run's have none.

Usage:
    python scripts/confirmed_grades.py [--run run7 --grades data/processed/enwiki/v3/grades_run7]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.grading_packs import GRADES_DIR

REVIEWS_DIR = GRADES_DIR / "reviews_download" / "reviews"
FIELDS = ("outcome", "already_known", "specificity", "grounded")
DATED_ID = re.compile(r"(?:(?P<run>[a-z]\w*)-)?(?P<day>\d{4}-\d{2}-\d{2})-(?P<n>\d+)")


def pack_key(doc_id: str, run: str = "") -> str | None:
    """The pack key for a review document of `run`: "2026-09-18-2" -> "2026-09-18#2", and with run "run7",
    "run7-2026-09-24-0" -> "2026-09-24#0". None for another run's document."""
    m = DATED_ID.fullmatch(doc_id)
    if not m or (m["run"] or "") != run:
        return None
    return f"{m['day']}#{m['n']}"


def confirmed(reviews: dict[str, dict], run: str = "") -> dict[str, dict]:
    """The grade in each of `run`'s reviews, keyed like its packs."""
    keyed = {key: review for key, review in ((pack_key(i, run), r) for i, r in reviews.items()) if key}
    return {key: {f: review["grade"][f] for f in FIELDS} for key, review in sorted(keyed.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reviews", type=Path, default=REVIEWS_DIR, help="the downloaded reviews")
    parser.add_argument("--run", default="", help="the run's name in its document ids; none for the first run")
    parser.add_argument("--grades", type=Path, default=GRADES_DIR, help="the run's grades folder, with packs.json")
    args = parser.parse_args(argv)
    reviews = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.reviews.glob("*.json"))}
    grades = confirmed(reviews, args.run)
    statuses = collections.Counter(r["status"] for i, r in reviews.items() if pack_key(i, args.run))
    packs = json.loads((args.grades / "packs.json").read_text(encoding="utf-8"))
    unknown = [k for k in grades if k not in packs]
    if unknown:
        raise SystemExit(f"{len(unknown)} reviews don't match any pack in {args.grades}, e.g. {unknown[:3]}")
    path = args.grades / "confirmed.json"
    path.write_text(json.dumps(grades, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(grades)} hand grades ({statuses['confirmed']} confirmed as drafted, {statuses['adjusted']} adjusted) "
          f"-> {path}; {len(packs) - len(grades)} of {len(packs)} predictions not yet reviewed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
