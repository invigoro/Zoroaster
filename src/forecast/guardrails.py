"""Topics never published about any specific person (PLAN.md §6 steps 10 and 11).

The list lives here, in one place, so it can grow: health, death, crime,
legal trouble, personal life and the like.
- Published version 2 forecasts withhold section names that match it
  (`v2_report.py --site`).
- Version 3 may predict wars, disasters and crime in general terms ("a major
  drone attack will take place"; decided 2026-10-01). A prediction on one of
  these topics is dropped if it names a specific person or organization
  (`src/prophecy/checks.py`, which also asks the model about the topic, since
  this list misses words like "rob").

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
