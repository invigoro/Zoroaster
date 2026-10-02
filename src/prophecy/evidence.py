"""What the prophet reads: the day's top pages, each as a short numbered block.

For each page, all as of the end of D-1:
- its rank in the burst prophecy, and yesterday's activity;
- its lead, cut short;
- yesterday's change: its main sections, its kinds, and some of its new text;
- its linked pages that burst yesterday;
- version 2's forecast of today's change (`forecast_v2.py`).

Milestone 1 (PLAN.md §6 step 11) leaves out every biography, living or not.
That covers pages in Category:Living people, and pages with a births or
deaths category. A recently dead person's page bursts for their death,
which is never predicted.

**Every date is marked relative to the day being foretold** (`mark_dates`),
for example "will be played on October 4 [in 4 days]". On the development
days 41 of 47 missed predictions named a result due after the day. In 27 of
those the lead gave the date, but the model didn't work out that it was
later. Now the code does that arithmetic.

**Lines of the page dated to the day itself** (`dated_lines`): schedule rows,
fixtures, an event's last day. Leads give an event's dates, but a day's own
matches are in its tables. On the development days 13 of 274 page-days'
evidence dated anything to the day, from the lead and yesterday's text
alone; 80 pages' text named the day's date somewhere.
"""

from __future__ import annotations

import re
from datetime import date

from src.forecast.metrics import main_sections

MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
_MONTH = "|".join(MONTHS)
_TO = r"\s*(?:–|—|-|to|until|and)\s*"


def _day(n: int) -> str:
    return rf"(?P<d{n}>\d{{1,2}})(?:st|nd|rd|th)?"


def _month(n: int) -> str:
    return rf"(?P<m{n}>{_MONTH})"


def _year(n: int) -> str:
    return rf"(?:,?\s+(?P<y{n}>\d{{4}}))?"


# The most specific first: a range across months, a range within one, then single dates.
DATE_PATTERNS = tuple(re.compile(p) for p in (
    rf"\b{_day(1)}\s+{_month(1)}{_year(1)}{_TO}{_day(2)}\s+{_month(2)}{_year(2)}\b",  # 24 September to 3 October 2026
    rf"\b{_month(1)}\s+{_day(1)}{_year(1)}{_TO}{_month(2)}\s+{_day(2)}{_year(2)}\b",  # September 27 – October 4, 2026
    rf"\b{_day(1)}{_TO}{_day(2)}\s+{_month(2)}{_year(2)}\b",  # 16–22 September 2026
    rf"\b{_month(1)}\s+{_day(1)}{_TO}{_day(2)}{_year(2)}\b",  # September 24–27, 2026
    rf"\b{_day(1)}\s+{_month(1)}{_year(1)}\b",  # 26 September 2026
    rf"\b{_month(1)}\s+{_day(1)}{_year(1)}\b",  # October 4; November 3, 2026
))

BIOGRAPHY = re.compile(
    r"\[\[\s*Category\s*:\s*(?:Living[ _]people|Possibly[ _]living[ _]people|\d{1,4}s?(?:[ _]BC)?[ _](?:births|deaths)"
    r"|Year[ _]of[ _](?:birth|death)[ _](?:missing|unknown))", re.IGNORECASE)

TOP = 20
LEAD_CHARS = 900  # the prophet reads one page at a time (prophet.py), so it has room for most of a lead
NEW_TEXT_CHARS = 600
NEIGHBORS_SHOWN = 6
DATED_LINES, DATED_LINE_CHARS = 4, 160

LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
REF = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.DOTALL)
DATE_TEMPLATE = re.compile(r"\{\{\s*(?:start date|end date|start date and age|end date and age|film date|dts)\s*\|"
                           r"(?:\s*[a-z]+\s*=[^|{}]*\|)*\s*(\d{4})\s*\|\s*(\d{1,2})\s*\|\s*(\d{1,2})[^{}]*\}\}",
                           re.IGNORECASE)
SHORT_TEMPLATE = re.compile(r"\{\{\s*[\w -]+\|\s*([^{}|=]{1,30}?)\s*\}\}")  # {{flag|Japan}}, {{fb|JPN}}
TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
TAG = re.compile(r"<[^>]+>")
DATE_FIELD = re.compile(r"\s*\|\s*date\s*=", re.IGNORECASE)  # a fixture box's date: its teams follow
FIELD_LINES = 5


def _cut(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " …"


def _title(title: str) -> str:
    return title.replace("_", " ")


def _nearest(month: str, day_of_month: str, year: str | None, near: date) -> date:
    """The date, in the year given, or else in the year that puts it nearest `near`. ValueError if it can't exist."""
    if year:
        return date(int(year), MONTHS.index(month) + 1, int(day_of_month))
    candidates = []
    for y in (near.year - 1, near.year, near.year + 1):
        try:
            candidates.append(date(y, MONTHS.index(month) + 1, int(day_of_month)))
        except ValueError:  # 29 February
            pass
    if not candidates:
        raise ValueError(f"no {day_of_month} {month}")
    return min(candidates, key=lambda d: abs((d - near).days))


def _span(m: re.Match, day: date) -> tuple[date, date] | None:
    """The first and last dates a match names (the same for a single date), or None if it can't exist."""
    g = m.groupdict()
    month1, month2 = g.get("m1") or g["m2"], g.get("m2") or g["m1"]
    try:
        last = _nearest(month2, g.get("d2") or g["d1"], g.get("y2") or g.get("y1"), day)
        # Nearest the last date, so "28 December to 3 January 2027" starts in 2026.
        first = _nearest(month1, g["d1"], g.get("y1"), last)
    except ValueError:
        return None
    return (first, last) if first <= last else None


def _ago(days: int) -> str:
    """A distance in time, in days up to two months, then months, then years."""
    if days <= 60:
        return f"{days} days"
    if days < 365:
        return f"about {round(days / 30.4)} months"
    years = round(days / 365.25)
    return "about a year" if years == 1 else f"about {years} years"


def relative(first: date, last: date, day: date) -> str:
    """When a date, or a span of dates, falls relative to `day`: "today", "in 4 days", "under way, ends tomorrow"."""
    if first == last:
        n = (first - day).days
        return {0: "today", 1: "tomorrow", -1: "yesterday"}.get(n) or (f"in {_ago(n)}" if n > 0 else f"{_ago(-n)} ago")
    if last < day:
        n = (day - last).days
        return "ended yesterday" if n == 1 else f"ended {_ago(n)} ago"
    if first > day:
        n = (first - day).days
        return "starts tomorrow" if n == 1 else f"starts in {_ago(n)}"
    n = (last - day).days
    ends = "ends today" if n == 0 else "ends tomorrow" if n == 1 else f"ends in {_ago(n)}"
    return f"starts today, {ends}" if first == day else f"under way, {ends}"


def _date_template(m: re.Match) -> str:
    year, month, day_of_month = m.groups()
    return f"{int(day_of_month)} {MONTHS[int(month) - 1]} {year}" if 1 <= int(month) <= 12 else ""


def clean_line(text: str) -> str:
    """A line of wikitext, readable. Unlike `stage2.wikitext.plain_text`, it keeps infobox fields
    ("| champion = …"), where results often go, and short template arguments ({{fb|JPN}} → JPN), and
    writes date templates out ({{Start date|2026|9|24}} → 24 September 2026). Table cells are joined with " · "."""
    text = DATE_TEMPLATE.sub(_date_template, REF.sub("", text))
    for _ in range(3):  # nested templates, innermost first
        text = TEMPLATE.sub("", SHORT_TEMPLATE.sub(r"\1", text))
    text = TAG.sub("", LINK.sub(r"\1", text)).replace("'''", "").replace("''", "")
    text = text.strip().lstrip("|!").replace("||", " · ").replace("!!", " · ")
    return " ".join(text.split())


def _is_the_day(text: str, day: date) -> bool:
    """Whether `text` names `day` itself as a date, or as the first or last day of a span."""
    return any(span and day in span for p in DATE_PATTERNS for span in (_span(m, day) for m in p.finditer(text)))


def dated_lines(page_text: str, day: date) -> list[str]:
    """The page's lines, as it stood the day before, that date something to `day` itself: schedule rows,
    fixtures, an event's first or last day. A fixture box's date field brings the lines after it, its teams."""
    lines, found = page_text.splitlines(), []
    for i, line in enumerate(lines):
        if str(day.day) not in line or not _is_the_day(clean_line(line), day):
            continue
        part = lines[i : i + FIELD_LINES] if DATE_FIELD.match(line) else [line]
        text = _cut(" · ".join(t for t in (clean_line(p).strip("{}| ") for p in part) if t), DATED_LINE_CHARS)
        if text and text not in found:
            found.append(text)
        if len(found) == DATED_LINES:
            break
    return found


def mark_dates(text: str, day: date) -> str:
    """`text` with each date it names followed by when that is relative to `day`, the day
    being foretold: "played on October 4 [in 4 days]"."""
    marks: list[tuple[int, int, str]] = []
    for pattern in DATE_PATTERNS:
        for m in pattern.finditer(text):
            if any(m.start() < end and start < m.end() for start, end, _ in marks):
                continue  # part of a longer date already marked
            span = _span(m, day)
            if span:
                marks.append((m.start(), m.end(), relative(*span, day)))
    out, last = [], 0
    for start, end, label in sorted(marks):
        out += [text[last:end], f" [{label}]"]
        last = end
    return "".join(out) + text[last:]


def is_biography(page_text: str) -> bool:
    """Whether a page's wikitext puts it in a category for people: living, births or deaths."""
    return bool(BIOGRAPHY.search(page_text))


def eligible(rows: list[dict], top: int = TOP, people: bool = False) -> list[dict]:
    """The day's pages the prophet may read, best ranked first: at most `top`,
    and, in milestone 1, no biographies (`is_biography` on `page_text`, or `living`)."""
    rows = sorted(rows, key=lambda r: r["rank"])
    return [r for r in rows if people or not (r["living"] or is_biography(r.get("page_text", "")))][:top]


def page_block(number: int, row: dict, forecast: dict | None) -> str:
    """One page's evidence, numbered for citation, with its dates marked relative to the page-day.
    Only what was known by the end of the day before: the page then, and yesterday's change."""
    day = row["date"]
    burst = ", a burst" if row["is_burst_1d"] else ""
    lines = [f"[{number}] {_title(row['page_title'])} (ranked {row['rank']} for bursting today)",
             f"Yesterday: {row['edits_1d']} edits by {row['editors_1d']} editors{burst}; "
             f"{row['edits_7d']} edits in the past week.",
             f"About: {_cut(mark_dates(row['lead'], day), LEAD_CHARS) or 'no lead text'}"]
    if not row["yesterday_known"] or not row["yesterday_sections"]:
        lines.append("Yesterday's changes: none" if row["yesterday_known"] else "Yesterday's changes: unknown")
    else:
        sections = "; ".join(main_sections(row["yesterday_sections"], row["yesterday_section_chars"]))
        change = f"Yesterday's changes: in {sections} ({', '.join(row['yesterday_kinds']) or 'no kinds detected'})"
        if row["yesterday_prose"]:
            change += f'. New text: "{_cut(mark_dates(row["yesterday_prose"], day), NEW_TEXT_CHARS)}"'
        lines.append(change)
    dated = dated_lines(row.get("page_text") or "", day)
    if dated:
        lines.append("Dated today on the page: " + " | ".join(mark_dates(d, day) for d in dated))
    if row["bursting_neighbors"]:
        lines.append("Linked pages that burst yesterday: "
                     + "; ".join(_title(t) for t in row["bursting_neighbors"][:NEIGHBORS_SHOWN]))
    if forecast:
        lines.append(f"The edit forecaster expects changes in: {'; '.join(forecast['sections']) or 'no sections'} "
                     f"({', '.join(forecast['kinds']) or 'no kinds'})")
    return "\n".join(lines)


def evidence_blocks(rows: list[dict], forecasts: dict[tuple[int, str], dict]) -> list[str]:
    """Every eligible page's block, numbered from 1 in rank order."""
    return [page_block(i, r, forecasts.get((r["page_id"], r["date"].isoformat()))) for i, r in enumerate(rows, start=1)]


def evidence_text(rows: list[dict], forecasts: dict[tuple[int, str], dict]) -> str:
    return "\n\n".join(evidence_blocks(rows, forecasts))
