"""Wikipedia's daily Portal:Current events pages: the record of major events the judge grades against, and
the prophet's evidence of what's happening in the world (PLAN.md §2, decided 2026-10-02).

PLAN.md §6 step 11. Each day's page lists news items under bold category
lines ('''Armed conflicts and attacks''', '''Sports'''). They're nested
bullets: a topic, its subtopics, then the news sentences.

`items` turns the wikitext into one plain line per news sentence, prefixed
with its category and topics:

    Sports › 2026 Asian Games › Japan wins the men's 3x3 basketball gold.

The pages are outside the mainspace scope (§2). The judge reads them at
their latest revision. The prophet reads, for day D, the KNOWN_DAYS pages
before D as they stood at the end of D-1 (`known_at`), since editors keep
adding a day's events for a day or two after it.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone

from src.stage2.wikitext import plain_text

KNOWN_DAYS = 7  # the week of reports before the day foretold

CATEGORY = re.compile(r"^'''([^']+)'''\s*$")
BULLET = re.compile(r"^(\*+)\s*(.*)$")
CITATION = re.compile(r"\[https?://[^\s\]]+[^\]]*\]")  # [https://... (Source)]
EMPHASIS = re.compile(r"'{2,}")
SEPARATOR = " › "


def page_title(day: date) -> str:
    return f"Portal:Current events/{day.year} {day:%B} {day.day}"


def known_at(day: date) -> datetime:
    """The last moment whose revisions the prophet may read for `day`: the end of the day before, in UTC."""
    return datetime.combine(day - timedelta(days=1), time(23, 59, 59), tzinfo=timezone.utc)


def _clean(text: str) -> str:
    return " ".join(EMPHASIS.sub("", plain_text(CITATION.sub("", text))).split())


def items(wikitext: str) -> list[str]:
    """One line per news sentence: its category and topics, then the sentence."""
    lines = [line.rstrip() for line in wikitext.splitlines()]
    bullets = [(i, len(m.group(1)), m.group(2)) for i, line in enumerate(lines) if (m := BULLET.match(line))]
    depth_after = {i: (bullets[k + 1][1] if k + 1 < len(bullets) else 0) for k, (i, _, _) in enumerate(bullets)}
    category, topics, out = "", [], []
    for i, line in enumerate(lines):
        heading = CATEGORY.match(line.strip())
        if heading:
            category, topics = _clean(heading.group(1)), []
            continue
        bullet = BULLET.match(line)
        if not bullet:
            continue
        depth, text = len(bullet.group(1)), _clean(bullet.group(2))
        topics = topics[: depth - 1]
        if depth_after[i] > depth:  # a topic: what follows is nested under it
            topics.append(text)
        elif text:
            out.append(SEPARATOR.join([*(p for p in [category] if p), *topics, text]))
    return out
