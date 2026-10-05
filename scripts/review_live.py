"""The live days' published prophecies, for review: what each said, which were rewritten or revised (with the
words they replaced), and which read as garbled or vague.

The user decided on 2026-10-05 to let the live prophecy run a few days, then "reassess the rate at which these get
garbled before deciding if further action is required". Every night's whole record is in
`data/processed/enwiki/v3/daily/D.json` (`prophesy_daily.py`); this reads them. The flags are a first pass: words the
rewrites have garbled with before ("airstrikes by major armed forces and an actor"), and teams or countries left
unnamed (`checks.PLACEHOLDER`). It's for a person to judge.

Writes `data/processed/enwiki/v3/daily_review.md` and prints a day-by-day count.

Usage:
    python scripts/review_live.py [--days 2026-10-05 2026-10-08]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_site import PROPHECY_DIR
from src.prophecy.checks import PLACEHOLDER

REVIEW_PATH = PROPHECY_DIR.parent / "daily_review.md"
GARBLED = re.compile(r"\b(?:an|another) actor\b|\ba group of actors\b|\b(?:in|near|to|from) a (?:region|location|place|"
                     r"area|town|city|country)\b|\ba prominent (?:individual|person|figure|entity)\b")


def flags(text: str) -> list[str]:
    """What in a published prediction may be garbled or vague, as quoted words."""
    return [m.group(0) for m in (*GARBLED.finditer(text), *PLACEHOLDER.finditer(text))]


def review(records: list[dict]) -> tuple[list[str], list[tuple]]:
    """The review's Markdown lines, and each day's counts: published, rewritten, revised, flagged."""
    lines, counts = ["# The live prophecy, for review", ""], []
    for r in records:
        published = [p for p in r["predictions"] if p.get("published")]
        flagged = [p for p in published if flags(p["text"])]
        counts.append((r["date"], len(published), sum(bool(p.get("rewritten_from")) for p in published),
                       sum(bool(p.get("revised_from")) for p in published), len(flagged)))
        lines += [f"## {r['date']}: {len(published)} published, {counts[-1][2]} rewritten, {counts[-1][3]} revised, "
                  f"{len(flagged)} flagged", ""]
        for n, p in enumerate(published, start=1):
            found = flags(p["text"])
            lines.append(f"{n}. [{p.get('topic')}, due {p.get('due')}] {p['text']}"
                         + (f" **May be garbled or vague: {'; '.join(found)}.**" if found else ""))
            if p.get("rewritten_from"):
                lines.append(f"   - Rewritten from: {p['rewritten_from']}")
            if p.get("revised_from"):
                lines.append(f"   - Revised from: {p['revised_from']} (its reports didn't give "
                             f"{', '.join(p.get('unsupported') or [])})")
        lines.append("")
    return lines, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=date.fromisoformat, nargs=2, metavar=("FIRST", "LAST"))
    args = parser.parse_args(argv)
    paths = sorted(p for p in PROPHECY_DIR.glob("????-??-??.json"))
    if args.days:
        paths = [p for p in paths if args.days[0].isoformat() <= p.stem <= args.days[1].isoformat()]
    lines, counts = review([json.loads(p.read_text(encoding="utf-8")) for p in paths])
    REVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")
    for day, published, rewritten, revised, flagged in counts:
        print(f"{day}: {published} published, {rewritten} rewritten, {revised} revised, {flagged} flagged")
    total = sum(c[1] for c in counts)
    print(f"{sum(c[4] for c in counts)} of {total} flagged -> {REVIEW_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
