"""Grading a prophecy against what its day actually brought (PLAN.md §6 step 11, phase 2).

The rubric is the same for the local judge and for the hand grades:
- **outcome:** whether the prediction had come true by the end of the day:
  "happened" (as stated), "partly", "did not happen", or "unknown" (the
  evidence doesn't say).
  - A prediction about something that comes only later, such as a final
    played after the day, counts as "did not happen".
  - One that had come true before the day counts as "happened", and
    already_known gives it no credit. The first wording said "on the day",
    which would have called Georgia's win the day before (2026-09-20) a
    miss.
- **already_known:** whether it was known by the end of the day before: the
  evidence from then reports or settles it, or it's about a match already
  played. An election held the day before may still be counting: Morocco
  voted on 2026-09-23, and its results came out on the 24th.
- **specificity:** 0 for vague ("news about X continues"), 1 for an outcome
  without detail ("X wins"), 2 for a named result, score or number
  ("X beats Y 3–1").
- **grounded:** whether the cited evidence supports making the prediction.
  It doesn't if the prediction:
  - is about something else;
  - picks a team, party or other participant the evidence never mentions;
  - or predicts for the day what the evidence says comes later.
  The first runs often picked teams from the model's own memory, which ends
  years before these days. The prophet's instructions forbid that (rule 5).

`credit` is 0 if already known, whatever the outcome. Otherwise it's the
outcome's points (happened 2, partly 1, did not happen 0) times
specificity, divided by 4. An unknown outcome that wasn't already known
isn't counted.

Each prediction's evidence (`pack`):
- **what was known by the end of the day before:** the cited pages' blocks,
  as the prophet saw them;
- **what the day brought:** each cited page's change over the day (its
  sections, kinds, new text, and changed lines as they read at the day's
  end), with a link to the day's diff;
- **that day's Portal:Current events items.**
"""

from __future__ import annotations

import json

from src.forecast.metrics import main_sections
from src.prophecy.checks import QUALITY_REASONS
from src.prophecy.evidence import clean_line
from src.prophecy.prophet import _objects

OUTCOMES = ("happened", "partly", "did not happen", "unknown")
POINTS = {"happened": 2, "partly": 1, "did not happen": 0}
NEW_TEXT_CHARS = 900
LINE_CHARS = 220
LINES_SHOWN = 8

JUDGE_SYSTEM = """You grade predictions against what actually happened. Use only the evidence given; your own knowledge ends years before these dates.

Grade each prediction on four things:
1. outcome: whether it had come true by the end of the day. "happened" if it had, as stated; "partly" if part of it had; "did not happen" if it hadn't, including when what it predicts comes only later (a final played after that day); "unknown" if the evidence doesn't say.
2. already_known: true if it was known by the end of the day before: the evidence from then already reported or settled it (a date, venue, line-up, schedule or result already known, or a match already played), false if not.
3. specificity: 0 if vague ("news about X will continue"), 1 if it names an outcome without detail ("X will win"), 2 if it names a precise result, score or number ("X will beat Y 3-1").
4. grounded: false if it is about something other than its cited evidence, picks a team, party or other participant the evidence never mentions, or predicts for that day something the evidence says comes later; otherwise true.

Answer with only a JSON object: {"outcome": "...", "already_known": true or false, "specificity": 0, 1 or 2, "grounded": true or false, "reason": "one or two sentences citing the evidence"}"""


def _cut(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " …"


def diff_url(row: dict) -> str | None:
    """The day's whole diff on Wikipedia, from the end of D-1 to the end of D."""
    if row["end_id"] == row["prompt_id"]:
        return None
    return f"https://en.wikipedia.org/w/index.php?diff={row['end_id']}&oldid={row['prompt_id']}"


def day_change_text(row: dict) -> str:
    """What a page gained over the day, for the judge."""
    if row["end_id"] == row["prompt_id"]:
        return "Not changed on the day."
    sections = "; ".join(main_sections(row["sections"], row["section_chars"], 5)) or "none"
    lines = [f"Sections changed: {sections} ({', '.join(row['kinds']) or 'no kinds detected'})"]
    if row["prose"]:
        lines.append(f'New text: "{_cut(row["prose"], NEW_TEXT_CHARS)}"')
    changed = [_cut(clean_line(b), LINE_CHARS) for b in row["blocks"]]
    changed = [c for c in changed if len(c) > 3][:LINES_SHOWN]
    if changed:
        lines.append("Changed lines, as they read at the end of the day: " + " | ".join(changed))
    return "\n".join(lines)


def gradable(prediction: dict) -> bool:
    """Whether to grade a screened prediction: kept, or dropped only by the quality checks (novelty,
    grounding). Their verdicts can then be checked against the hand grades' already_known and
    grounded. A prediction a guardrail dropped (a person, a sensitive topic), or a copy or repeat,
    is never graded."""
    return all(reason in QUALITY_REASONS for reason in prediction["dropped_because"])


def pack(prediction: dict, day: str, known_before: dict[str, str], rows_by_title: dict[str, dict],
         current_events: list[str]) -> dict:
    """Everything needed to grade one prediction: before, after, and the day's record of events."""
    cited = prediction["evidence"]
    return {
        "date": day, "prediction": prediction["text"], "question": prediction.get("question", ""),
        "confidence": prediction.get("confidence"), "cited": cited, "kept": prediction.get("kept", True),
        "dropped_because": prediction.get("dropped_because", []),
        "known_before": {t: known_before[t] for t in cited if t in known_before},
        "day_brought": {t: day_change_text(rows_by_title[t]) for t in cited if t in rows_by_title},
        "diffs": {t: diff_url(rows_by_title[t]) for t in cited if t in rows_by_title},
        "current_events": current_events,
    }


def judge_messages(p: dict) -> list[dict]:
    before = "\n\n".join(p["known_before"].values()) or "(no pages cited)"
    after = "\n\n".join(f"{t}:\n{text}" for t, text in p["day_brought"].items()) or "(no pages cited)"
    events = "\n".join(f"- {e}" for e in p["current_events"]) or "(none listed)"
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"The prediction, for {p['date']} (UTC): {p['prediction']}\n\n"
                                        f"What was known by the end of the day before:\n{before}\n\n"
                                        f"What the cited pages gained on {p['date']}:\n{after}\n\n"
                                        f"Wikipedia's list of that day's major events:\n{events}\n\n"
                                        "Grade the prediction. Answer with only the JSON object."}]


def parse_grade(answer: str) -> dict | None:
    """The judge's grade, checked against the rubric; None if it doesn't fit."""
    for item in _objects(answer):
        if not isinstance(item, dict):
            continue
        outcome, specificity = item.get("outcome"), item.get("specificity")
        if outcome not in OUTCOMES or specificity not in (0, 1, 2):
            continue
        return {"outcome": outcome, "already_known": bool(item.get("already_known")), "specificity": specificity,
                "grounded": bool(item.get("grounded")), "reason": str(item.get("reason", "")).strip()}
    return None


def credit(grade: dict) -> float | None:
    """0 to 1: the outcome's points times specificity, over 4. 0 if already known, whatever
    the outcome, since restating what's settled earns nothing; otherwise None if unknown."""
    if grade["already_known"]:
        return 0.0
    if grade["outcome"] == "unknown":
        return None
    return POINTS[grade["outcome"]] * grade["specificity"] / 4


def dumps(grade: dict) -> str:
    return json.dumps(grade, ensure_ascii=False)
