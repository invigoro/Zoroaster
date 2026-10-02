"""The prophet's instructions, and reading its predictions back (PLAN.md §6 step 11).

From the evidence for day D (`evidence.py`), the prophet writes N_PREDICTIONS
predictions about the real world on D. Each is one sentence that starts "I
predict that" and cites the evidence by number. The rules, in SYSTEM:
- real-world events on D, not Wikipedia;
- only the evidence, since the model's knowledge ends years earlier;
- each prediction follows from its own cited pages;
- specific and checkable, and about what will happen, not what already has;
- milestone 1: no people, living or dead;
- never health, death, crime, legal trouble or personal life.
The rules are repeated after the evidence, since a small model forgets
instructions that come before thousands of tokens. The checks in `checks.py`
don't rely on the prophet following them.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta

N_PREDICTIONS = 8
CONFIDENCES = ("low", "medium", "high")
PREFIX = "I predict that "

SYSTEM = """You are the prophet of Zoroaster, a project that foretells real-world events from what is happening on English Wikipedia.

You are given the Wikipedia pages that editors are most likely to be busy with on a coming day, and what was changed on each page the day before. From that evidence, foretell what will happen in the world on that day.

Rules:
1. Each prediction is one sentence that begins "I predict that", about a real-world event or development on {day} (UTC). Never predict anything about Wikipedia, its pages or its editors.
2. Use only the evidence. Your own knowledge ends years before these dates, so don't add facts the evidence doesn't give.
3. Each prediction must follow from the evidence of the pages it cites. Don't combine unrelated pages.
4. Be specific and checkable: name the event, place, team, organization, product or result. Avoid vague predictions such as "news about X will continue" or "X will be well received".
5. Predict what will happen next, not what the evidence says has already happened.
6. Don't name or describe any person, living or dead, even in passing. Write about events, places, teams, organizations, works and things.
7. Never predict anything about health, illness, death, crime, arrests, legal cases, scandals or anyone's personal life.
8. Cite the pages each prediction rests on, by their numbers.

A good prediction: "I predict that Japan will add at least five gold medals to its total at the 2026 Asian Games." It is specific, it can be checked tomorrow, and it follows from a page about Japan at those games.
A bad prediction: "I predict that the film will be well received." It is vague and can't be checked tomorrow.

Answer with only a JSON list of {n} objects, like this:
[{{"prediction": "I predict that ...", "evidence": [1, 4], "confidence": "medium"}}]
Confidence is "low", "medium" or "high"."""

REMINDER = ("Write {n} predictions for {day}. Each one begins \"I predict that\", follows from the pages it cites, "
            "names no person, and avoids health, death, crime, legal cases and personal life. Answer with only the JSON list.")


def _day_text(day: date) -> str:
    return f"{day:%A}, {day.day} {day:%B %Y}"


def messages(day: date, evidence: str, n: int = N_PREDICTIONS) -> list[dict]:
    """The chat for one day: the rules, then the evidence as of the end of the day before, then the rules in brief."""
    return [{"role": "system", "content": SYSTEM.format(day=_day_text(day), n=n)},
            {"role": "user", "content": f"Foretell {_day_text(day)}. Here is the evidence, as of the end of "
                                        f"{_day_text(day - timedelta(days=1))}, likeliest to burst first:\n\n{evidence}\n\n"
                                        + REMINDER.format(n=n, day=_day_text(day))}]


def _objects(answer: str) -> list[object]:
    """Every top-level {...} in `answer` that parses as JSON, so one broken
    object (an extra brace, say) doesn't lose the rest."""
    found, depth, start, quoted, escaped = [], 0, None, False, False
    for i, char in enumerate(answer):
        if quoted:
            quoted = not (char == '"' and not escaped)
            escaped = char == "\\" and not escaped
            continue
        if char == '"':
            quoted, escaped = True, False
        elif char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    found.append(json.loads(answer[start : i + 1]))
                except json.JSONDecodeError:
                    pass
    return found


def normalize(sentence: str) -> str:
    """The sentence starting "I predict that"; a leading "The"/"A"/"An" is lowercased after the prefix."""
    sentence = " ".join(sentence.split())
    if sentence.lower().startswith(PREFIX.lower()):
        return PREFIX + sentence[len(PREFIX):]
    if re.match(r"(The|A|An) ", sentence):
        sentence = sentence[0].lower() + sentence[1:]
    return PREFIX + sentence


def parse_predictions(answer: str, titles: list[str]) -> list[dict]:
    """The predictions in the prophet's answer, as [{"text", "evidence", "confidence"}],
    with cited numbers turned into page titles. Malformed entries are skipped."""
    out = []
    for item in _objects(answer):
        if not isinstance(item, dict) or not str(item.get("prediction", "")).strip():
            continue
        numbers = item.get("evidence") if isinstance(item.get("evidence"), list) else []
        cited = [titles[n - 1] for n in numbers if isinstance(n, int) and 1 <= n <= len(titles)]
        confidence = item.get("confidence")
        out.append({"text": normalize(str(item["prediction"])), "evidence": list(dict.fromkeys(cited)),
                    "confidence": confidence if confidence in CONFIDENCES else None})
    return out
