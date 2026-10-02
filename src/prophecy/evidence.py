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
"""

from __future__ import annotations

import re

from src.forecast.metrics import main_sections

BIOGRAPHY = re.compile(
    r"\[\[\s*Category\s*:\s*(?:Living[ _]people|Possibly[ _]living[ _]people|\d{1,4}s?(?:[ _]BC)?[ _](?:births|deaths)"
    r"|Year[ _]of[ _](?:birth|death)[ _](?:missing|unknown))", re.IGNORECASE)

TOP = 20
LEAD_CHARS = 320
NEW_TEXT_CHARS = 320
NEIGHBORS_SHOWN = 6


def _cut(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " …"


def _title(title: str) -> str:
    return title.replace("_", " ")


def is_biography(page_text: str) -> bool:
    """Whether a page's wikitext puts it in a category for people: living, births or deaths."""
    return bool(BIOGRAPHY.search(page_text))


def eligible(rows: list[dict], top: int = TOP, people: bool = False) -> list[dict]:
    """The day's pages the prophet may read, best ranked first: at most `top`,
    and, in milestone 1, no biographies (`is_biography` on `page_text`, or `living`)."""
    rows = sorted(rows, key=lambda r: r["rank"])
    return [r for r in rows if people or not (r["living"] or is_biography(r.get("page_text", "")))][:top]


def page_block(number: int, row: dict, forecast: dict | None) -> str:
    """One page's evidence, numbered for citation."""
    burst = ", a burst" if row["is_burst_1d"] else ""
    lines = [f"[{number}] {_title(row['page_title'])} (ranked {row['rank']} for bursting today)",
             f"Yesterday: {row['edits_1d']} edits by {row['editors_1d']} editors{burst}; "
             f"{row['edits_7d']} edits in the past week.",
             f"About: {_cut(row['lead'], LEAD_CHARS) or 'no lead text'}"]
    if not row["yesterday_known"] or not row["yesterday_sections"]:
        lines.append("Yesterday's changes: none" if row["yesterday_known"] else "Yesterday's changes: unknown")
    else:
        sections = "; ".join(main_sections(row["yesterday_sections"], row["yesterday_section_chars"]))
        change = f"Yesterday's changes: in {sections} ({', '.join(row['yesterday_kinds']) or 'no kinds detected'})"
        if row["yesterday_prose"]:
            change += f'. New text: "{_cut(row["yesterday_prose"], NEW_TEXT_CHARS)}"'
        lines.append(change)
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
