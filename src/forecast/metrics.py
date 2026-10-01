"""Score version 2's structured forecasts against what a page actually did (PLAN.md §6 step 10).

A structured forecast names up to MAX_SECTIONS sections and a set of kinds
of change (`src.forecast.changes.KINDS`). Against the day's actual change
(`day_change`), each forecast gets:
- `section_precision`: the share of the sections it names that changed;
- `main_section_hit`: whether it names the section that changed most;
- `kinds_jaccard`: the overlap of its kinds with the actual ones.
`summarize` averages those and adds each kind's precision and recall.
"""

from __future__ import annotations

import math

from src.forecast.changes import KINDS

MAX_SECTIONS = 3


def main_sections(sections: list[str], section_chars: list[int], k: int = MAX_SECTIONS) -> list[str]:
    """The `k` sections with the most change, most first; ties keep page order."""
    order = sorted(range(len(sections)), key=lambda i: -section_chars[i])
    return [sections[i] for i in order[:k]]


def score(forecast: dict, actual: dict) -> dict:
    """One forecast ({"sections", "kinds"}) against one actual change."""
    named, changed = forecast["sections"][:MAX_SECTIONS], set(actual["sections"])
    main = main_sections(actual["sections"], actual["section_chars"], 1)
    said, did = set(forecast["kinds"]), set(actual["kinds"])
    return {
        "section_precision": len(set(named) & changed) / len(named) if named else math.nan,
        "main_section_hit": float(main[0] in named) if main else math.nan,
        "kinds_jaccard": len(said & did) / len(said | did) if said | did else 1.0,
        "kinds": {k: (k in said, k in did) for k in KINDS},
    }


def summarize(scores: list[dict]) -> dict:
    """Mean of each score (skipping undefined ones), and each kind's precision and recall."""
    out = {}
    for name in ("section_precision", "main_section_hit", "kinds_jaccard"):
        values = [s[name] for s in scores if not math.isnan(s[name])]
        out[name] = sum(values) / len(values) if values else math.nan
    for kind in KINDS:
        said = sum(s["kinds"][kind][0] for s in scores)
        did = sum(s["kinds"][kind][1] for s in scores)
        both = sum(s["kinds"][kind][0] and s["kinds"][kind][1] for s in scores)
        out[f"{kind} precision"] = both / said if said else math.nan
        out[f"{kind} recall"] = both / did if did else math.nan
    return out
