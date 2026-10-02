"""Topics never published about anyone (PLAN.md §6 steps 10 and 11).

The list lives here, in one place, so it can grow: health, death, crime,
legal trouble, personal life and the like.
- Published version 2 forecasts withhold section names that match it
  (`v2_report.py --site`).
- Version 3's prophet will be told to avoid it, and checked against it.

Matching is by word start, case-insensitive, and errs toward withholding.
"Court" also matches "Royal court", for example.
"""

from __future__ import annotations

import re

SENSITIVE_STEMS = (
    # health
    "health", "illness", "disease", "diagnos", "cancer", "medical", "hospital", "surgery", "injur", "disabilit",
    "mental", "dementia", "addiction", "rehab", "overdose", "pregnan",
    # death
    "death", "dead", "died", "dying", "funeral", "burial", "suicide", "murder", "killing", "assassinat",
    # crime and legal trouble
    "crime", "criminal", "arrest", "charge", "convict", "prison", "jail", "sentenc", "trial", "lawsuit", "legal",
    "litigation", "court", "indict", "allegation", "accus", "investigat", "scandal", "controvers", "abuse",
    "assault", "harass", "fraud", "misconduct",
    # personal life
    "personal", "private life", "family", "relationship", "marriage", "married", "divorce", "spouse", "children",
    "sexual", "religio",
)
_SENSITIVE = re.compile(r"\b(?:" + "|".join(re.escape(s) for s in SENSITIVE_STEMS) + ")", re.IGNORECASE)


def is_sensitive(text: str) -> bool:
    """Whether `text`, e.g. a section name, touches a topic on the list."""
    return bool(_SENSITIVE.search(text))


def sensitive_words(text: str) -> list[str]:
    """The words in `text` that match the list, in order, without repeats: why it was withheld."""
    return list(dict.fromkeys(m.group(0).lower() for m in _SENSITIVE.finditer(text)))
