"""Stage 2 examples: a prompt describing the page and the moment, and the
edit's inserted text as the target.

The prompt has the page title, the date, the section, and the parent
revision's text around the change, with EDIT_MARK where the insertion
goes. It can also carry *triggers*, the Stage 1 signals rendered as text:
the page's own recent activity and burst, plus the titles of linked pages
that were bursting the day before. Triggers are the only difference between
the two prompt variants that Stage 2 compares, so any gain from them is the
"something is happening" signal helping predict content.
"""

from __future__ import annotations

import re

EDIT_MARK = "⟦EDIT⟧"
MAX_NEIGHBORS_SHOWN = 8
_HEADING = re.compile(r"^(=+)\s*(.*?)\s*\1\s*$", re.MULTILINE)


def edit_context(old: str, offset: int, before: int = 600, after: int = 200) -> tuple[str, str]:
    """(section heading, text window) around char `offset` of `old`, with
    EDIT_MARK at the offset. The window starts and ends on word boundaries."""
    headings = list(_HEADING.finditer(old, 0, offset))
    section = headings[-1].group(2) if headings else "(lead)"
    start = max(0, offset - before)
    if start > 0:
        start = old.find(" ", start, offset) + 1 or start
    end = min(len(old), offset + after)
    if end < len(old):
        end = old.rfind(" ", offset, end) if old.rfind(" ", offset, end) > offset else end
    return section, old[start:offset] + EDIT_MARK + old[offset:end]


def trigger_text(
    features: dict,
    bursting_neighbors: list[str],
    max_neighbors: int = MAX_NEIGHBORS_SHOWN,
    changes: dict[str, str | None] | None = None,
    max_changes: int = 3,
    ranked_changes: list[tuple[str, str]] | None = None,
) -> str:
    """The Stage 1 signals for (page, day), as prompt lines.

    What changed on the bursting neighbors the day before comes one of two ways:
    - `changes` maps a neighbor's title to a snippet
      (`build_stage2_neighbor_changes.py`). Snippets are shown for the first
      `max_changes` neighbors that have one.
    - `ranked_changes` lists (neighbor, text) to show as given, already
      ranked and trimmed (`src.stage2.relevance`).
    """
    burst = "yes" if features["is_burst_1d"] else "no"
    neighbors = "; ".join(t.replace("_", " ") for t in bursting_neighbors[:max_neighbors]) or "none"
    text = (
        f"Edits to this page yesterday: {features['edits_1d']}; last 7 days: {features['edits_7d']}; "
        f"last 30 days: {features['edits_30d']}.\n"
        f"Page bursting yesterday: {burst} (z = {features['burst_z_1d']:.1f}).\n"
        f"Linked pages bursting yesterday: {neighbors}."
    )
    if ranked_changes is not None:
        shown = ranked_changes
    else:
        shown = [(t, changes[t]) for t in bursting_neighbors if changes and changes.get(t)][:max_changes]
    if shown:
        text += "\nWhat changed on them yesterday:" + "".join(
            f'\n- {t.replace("_", " ")}: "{s}"' for t, s in shown
        )
    return text


def build_prompt(title: str, date: str, section: str, context: str, triggers: str | None = None) -> str:
    lines = [f"Wikipedia page: {title.replace('_', ' ')}", f"Date: {date}"]
    if triggers:
        lines.append(triggers)
    lines += [
        f"Section: {section}",
        f"Text around the edit ({EDIT_MARK} marks where it goes):",
        context,
        "Inserted text:",
        "",
    ]
    return "\n".join(lines)
