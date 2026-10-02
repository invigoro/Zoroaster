"""The prophet's instructions, and reading its predictions back (PLAN.md §6 step 11).

From the evidence for day D (`evidence.py`), the prophet writes up to
N_PREDICTIONS predictions about the real world on D.
- **Each prediction answers an unsettled question** about D ("who wins Game
  1?"), so it can be checked when D is over.
  - In the first runs, about half the predictions restated the evidence
    (venues, dates, line-ups). Naming the open question first is the
    structure against that, and `checks.py` drops what the evidence already
    settles.
- **The other rules, in SYSTEM:**
  - only the evidence, since the model's knowledge ends years earlier;
  - each prediction follows from its own cited pages;
  - it must be possible;
  - milestone 1: no people, living or dead;
  - never health, death, crime, legal trouble or personal life.
- The rules are repeated after the evidence, since a small model forgets
  instructions that come thousands of tokens before. The checks in
  `checks.py` don't rely on the prophet following them.
- **The instructions hold no real-world examples.** An earlier version gave
  two example predictions, written from a development day's evidence (a
  Wild Card matchup, a tropical storm). The model copied them onto other
  days, citing unrelated pages. That also leaked a development day into the
  instructions. So the instructions now describe good and bad predictions
  in general terms, and `checks.py` drops any copy of an example sentence.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta

N_PREDICTIONS = 8
CONFIDENCES = ("low", "medium", "high")
PREFIX = "I predict that "

SYSTEM = """You are the prophet of Zoroaster, a project that foretells real-world events from what is happening on English Wikipedia.

You are given the Wikipedia pages that editors are most likely to be busy with on a coming day, and what was changed on each page the day before. From that evidence, foretell what will happen in the world on {day} (UTC).

How to prophesy:
1. For each page, ask what is still unsettled on {day}: a game or match to be played, votes to be counted, a decision expected, a storm on the move, a release about to open. Skip pages where nothing is unsettled.
2. Predict how it will turn out: who wins, what the result or number is, what is decided. It must be settled by the end of {day}, so it can be checked then.
3. Never predict what the evidence already reports. A date, a venue, a line-up, a schedule or a result that is already known is not a prediction.
4. Make it possible: a single tournament has one gold medal winner, and an election has one result.
5. Use only the evidence. Your own knowledge ends years before these dates, so don't add facts the evidence doesn't give.
6. Each prediction must follow from the pages it cites. Don't combine unrelated pages.
7. Don't name or describe any person, living or dead, even in passing. Write about events, places, teams, organizations, works and things. In sports where individuals compete (darts, tennis, golf, boxing, athletics, motor racing), don't predict who wins: the winner is a person.
8. Never predict anything about health, illness, death, crime, arrests, legal cases, scandals or anyone's personal life.

Each prediction is a JSON object:
- "question": the unsettled question about {day} that it answers;
- "prediction": one sentence that begins "I predict that";
- "evidence": the numbers of the pages it rests on; a prediction is about the subject of the pages it cites;
- "confidence": "low", "medium" or "high".

Good predictions answer a question the day will settle: for a match played that day, who wins it; for votes counted that day, which party leads; for a storm, whether it strengthens or makes landfall; for a release that day, how it opens.
Bad predictions restate a date, venue, line-up, schedule or result the evidence already gives; say something certain; or concern something decided after that day.

Write up to {n} predictions, each about something different; fewer good ones are better than padding. Answer with only the JSON list."""

REMINDER = ("Foretell up to {n} things still unsettled on {day}, each answering its \"question\", following from the pages it "
            "cites, never restating the evidence, naming no person, and avoiding health, death, crime, legal cases and "
            "personal life. Answer with only the JSON list.")


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
    """The predictions in the prophet's answer, as [{"text", "question", "evidence", "confidence"}],
    with cited numbers turned into page titles. Malformed entries are skipped."""
    out = []
    for item in _objects(answer):
        if not isinstance(item, dict) or not str(item.get("prediction", "")).strip():
            continue
        numbers = item.get("evidence") if isinstance(item.get("evidence"), list) else []
        cited = [titles[n - 1] for n in numbers if isinstance(n, int) and 1 <= n <= len(titles)]
        confidence = item.get("confidence")
        out.append({"text": normalize(str(item["prediction"])), "question": " ".join(str(item.get("question", "")).split()),
                    "evidence": list(dict.fromkeys(cited)),
                    "confidence": confidence if confidence in CONFIDENCES else None})
    return out
