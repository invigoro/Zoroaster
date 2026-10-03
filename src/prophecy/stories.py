"""Stories from Portal:Current events, as the prophet's evidence (PLAN.md §2, decided 2026-10-02).

A story is a news item's first topic: "Armed conflicts and attacks › 2026 Iran war › 2026 Strait of Hormuz
crisis › …" belongs to "2026 Iran war". For day D the prophet reads each story reported on either of the two
days before (RECENT_DAYS), one at a time, with its reports from the week before (`current_events.KNOWN_DAYS`),
as they stood at the end of D-1 (`current_events.known_at`). Dates in them are marked relative to D, as on
pages:

    [S2] 2026 Iran war (conflict): 9 reports this week, the latest yesterday
    [3 days ago] …
    [yesterday] 2026 Strait of Hormuz crisis › An Indian wiper is killed during an attack on …

- Sport is left out: it comes from the pages (`selection.py`), and only when it settles a title.
- So are items without a topic: a one-off report, with no story to follow.
- Stories are ordered by how many reports they had in those two days, then over the week.
- In the prophet's records a story's title is STORY_PREFIX and its topic, apart from page titles.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta

from src.prophecy.current_events import SEPARATOR
from src.prophecy.evidence import _cut, mark_dates, relative

STORY_PREFIX = "Current events: "
REPORTS_SHOWN = 12  # the latest of a story's week, so a war's daily reports don't crowd out the rest
# A story is read if it was reported on either of the two days before. Editors add most of a day's items
# later: by the end of 2026-09-23, its page held 9 of the 28 it has now, and none on the Iran war. The
# page for the day before that was fuller by then (23 items for 2026-09-22).
RECENT_DAYS = 2
REPORT_CHARS = 400
# Each category's topic, as the selection labels predictions (`selection.topic`).
CATEGORY_TOPICS = {"Armed conflicts and attacks": "conflict", "Arts and culture": "culture",
                   "Business and economy": "economy", "Disasters and accidents": "disaster",
                   "Health and environment": "health", "International relations": "politics",
                   "Law and crime": "crime", "Politics and elections": "politics",
                   "Science and technology": "science", "Sports": "sport"}


def split(item: str) -> tuple[str, list[str], str]:
    """An item's category, topics and sentence."""
    parts = item.split(SEPARATOR)
    return (parts[0], parts[1:-1], parts[-1]) if len(parts) > 1 else ("", [], item)


def stories(known: dict, day: date) -> list[dict]:
    """The stories reported on the RECENT_DAYS days before `day`, best covered first, each with its week of
    reports (`known`: a `current_events_known/D.json` record). Only pages for days before `day` are read."""
    pages = [p for p in known["days"] if date.fromisoformat(p["date"]) < day]  # never the day foretold or after
    parsed = [(p["date"], *split(item)) for p in pages for item in p["items"]]
    parsed = [(d, c, t, s) for d, c, t, s in parsed if t and CATEGORY_TOPICS.get(c) != "sport"]
    firsts = {topics[0] for _, _, topics, _ in parsed}
    found: dict[str, dict] = {}
    for reported, category, topics, sentence in parsed:
        # The deepest topic the week also lists on its own: "Middle Eastern crisis › 2026 Iran war › …" is the
        # Iran war's, when another day lists "2026 Iran war" at the top.
        depth = next((n for n in range(len(topics) - 1, -1, -1) if topics[n] in firsts), 0)
        key = topics[depth]
        story = found.setdefault(key, {"title": STORY_PREFIX + key, "categories": Counter(), "reports": []})
        report = (reported, SEPARATOR.join([*topics[depth + 1:], sentence]))
        if report not in story["reports"]:
            story["reports"].append(report)
            story["categories"][category] += 1
    recent = {(day - timedelta(days=n)).isoformat() for n in range(1, RECENT_DAYS + 1)}
    out = []
    for topic, story in found.items():
        reports = sorted(story["reports"])
        latest = sum(d in recent for d, _ in reports)
        if not latest:
            continue  # nothing new lately
        category = story["categories"].most_common(1)[0][0]
        out.append({"title": story["title"], "story": topic, "category": category,
                    "topic": CATEGORY_TOPICS.get(category, "other"), "reports": reports, "recent": latest,
                    "week": len(reports), "latest": reports[-1][0]})
    return sorted(out, key=lambda s: (-s["recent"], -s["week"], s["story"]))


def story_block(number: int, story: dict, day: date) -> str:
    """One story's evidence: its latest reports, each with its day relative to `day`, dates in it marked."""
    latest = relative(*(date.fromisoformat(story["latest"]),) * 2, day)
    lines = [f"[S{number}] {story['story']} ({story['topic']}): {story['week']} "
             f"report{'s' if story['week'] != 1 else ''} this week, the latest {latest}"]
    for reported, text in story["reports"][-REPORTS_SHOWN:]:
        when = relative(*(date.fromisoformat(reported),) * 2, day)
        lines.append(f"[{when}] {_cut(mark_dates(text, day), REPORT_CHARS)}")
    return "\n".join(lines)


def reports_between(records: dict[str, list[str]], story: str, first: date, last: date) -> list[str]:
    """For grading: a story's reports from `first` to `last`, from the latest pages (`records`: each day's
    items), each with its date. A report counts wherever the story sits in its topics: a later page may file
    "Houthi–Saudi Arabian conflict" under "Middle Eastern crisis › Yemeni civil war", and matching the first
    topic alone left run 10's pack for it showing no reports, though the days held two."""
    out = []
    for n in range((last - first).days + 1):
        reported = (first + timedelta(days=n)).isoformat()
        for item in records.get(reported, []):
            category, topics, sentence = split(item)
            if story in topics:
                out.append(f"{reported}: {SEPARATOR.join([*topics[topics.index(story) + 1:], sentence])}")
    return out


def is_story(title: str) -> bool:
    return title.startswith(STORY_PREFIX)
