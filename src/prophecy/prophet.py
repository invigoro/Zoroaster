"""The prophet's instructions, and reading its answers back (PLAN.md §6 step 11).

The prophet reads the day's pages (`evidence.py`) one at a time, in two steps:
1. **The question** (QUESTION), only for pages whose evidence dates something
   to day D itself (`marked_today`: "[today]", "ends today", "starts today").
   The model says what happens that day, then the main question it settles,
   or "none".
   - On 16 development page-days whose answers were known, asking every page
     "is anything decided on D?" got "none" for all of them, finals
     included. Saying first what happens that day found the right ones, but
     also wrote questions for finals marked "[tomorrow]". Filtering in code
     by the date marks got 14 of 16 right.
   - A question about a contest one person wins is dropped
     (`checks.one_persons_contest`): milestone 1 names no person.
2. **The prediction** (PREDICTION), for each remaining question: one
   sentence beginning "I predict that", and a confidence.

Why one page at a time:
- **Timing.** In the development days' first runs, a single call over all 20
  pages wrote 68 gradable predictions, and 41 of them named a result due after
  the day. In 27 of those the evidence gave the date (a final "on October 4").
  Now the dates are marked relative to D (`evidence.mark_dates`), and the
  first step asks only about D.
- **Room.** With one page per call, the model can read most of the lead,
  where the single call had to cut each lead to 320 characters.

The rules, which `checks.py` enforces without relying on the prophet:
- Only the evidence: the model's own knowledge ends years before these days.
- Milestone 1 names no specific person, living or dead. General descriptions
  such as "an important politician" are fine (decided 2026-10-01).
- Wars, disasters and crime may be predicted in general terms, but never
  about a specific person or a named organization (decided 2026-10-01).
- Nothing about anyone's health or personal life.

**The instructions hold no real-world examples.** An earlier version gave two
example predictions, written from a development day's evidence (a Wild Card
matchup, a tropical storm). The model copied them onto other days, citing
unrelated pages, and `checks.py` drops any copy of an example sentence.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta

CONFIDENCES = ("low", "medium", "high")
PREFIX = "I predict that "
SENTENCE = re.compile(r"I predict that [^\n\"]+?(?:\.(?=\s|$)|(?=[\n\"]|$))", re.IGNORECASE)  # to its first full stop

SYSTEM = ("You are the prophet of Zoroaster, a project that foretells real-world events from what is happening on English "
          "Wikipedia. Your own knowledge ends years before these dates, so use only the evidence you are given.")

PAGE = """Here is what a Wikipedia page said by the end of {yesterday}. Each date in it is marked relative to {day}, the day you foretell: [today] means {day}.

{block}"""

QUESTION = PAGE + """

What does the evidence say happens on {day} itself? Look for dates marked [today], or "ends today": a match or final played, votes cast or counted, a result or decision announced.

Answer in two lines:
Today: what the evidence says happens on {day}, or nothing
Question: the main question {day} will settle about it (who wins, what the result is, what is decided), or none

The question can't be about something that ends on a later date."""
TODAY_MARK = re.compile(r"\[(?:today|[^\]]*ends today|starts today[^\]]*)\]")
NOTHING = ("nothing", "none", "no ", "there is nothing", "the evidence does not")

PREDICTION = PAGE + """

The question {day} will answer: {question}

Answer it with one prediction, using only this evidence.
- One sentence beginning "I predict that", about this page's subject.
- Be specific (who wins, the result, the number), but don't restate what the evidence already reports, and keep it possible.
- Name no specific person, living or dead, and don't point to one by a title or role. General descriptions, such as "an important politician", are fine.
- Wars, disasters and crime are fine in general terms, but never about a specific person or a named organization (a company, party, armed group, government body or team).
- Nothing about anyone's health or personal life.

Answer with only a JSON object: {{"prediction": "I predict that ...", "confidence": "low", "medium" or "high"}}"""


def _day_text(day: date) -> str:
    return f"{day:%A}, {day.day} {day:%B %Y}"


def _chat(template: str, day: date, block: str, **fields: str) -> list[dict]:
    text = template.format(day=_day_text(day), yesterday=_day_text(day - timedelta(days=1)), block=block, **fields)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def question_messages(day: date, block: str) -> list[dict]:
    """Step 1, for one page's evidence: what, if anything, D itself decides."""
    return _chat(QUESTION, day, block)


def prediction_messages(day: date, block: str, question: str) -> list[dict]:
    """Step 2, for a page with a question: the prediction."""
    return _chat(PREDICTION, day, block, question=question)


def marked_today(block: str) -> bool:
    """Whether a page's evidence dates anything to the day foretold: "[today]", "ends today" or "starts today"."""
    return bool(TODAY_MARK.search(block))


def parse_question(answer: str) -> str | None:
    """The question in step 1's answer: its "Question:" line, if that ends in "?". None if the "Today:"
    line says nothing happens: the model sometimes writes a question anyway."""
    for line in answer.splitlines():
        label, _, rest = line.partition(":")
        label, rest = label.strip().strip("*").lower(), " ".join(rest.split()).strip('"“”* ')
        if label == "today" and rest.lower().startswith(NOTHING):
            return None
        if label == "question":
            return rest if rest.endswith("?") else None
    return None


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


def parse_prediction(answer: str, title: str, question: str) -> dict | None:
    """Step 2's prediction, as {"text", "question", "evidence", "confidence"}; None if it holds none.
    A bare "I predict that ..." sentence counts when the JSON is missing."""
    for item in _objects(answer):
        if isinstance(item, dict) and str(item.get("prediction", "")).strip():
            confidence = item.get("confidence")
            return {"text": normalize(str(item["prediction"])), "question": question, "evidence": [title],
                    "confidence": confidence if confidence in CONFIDENCES else None}
    bare = SENTENCE.search(answer)
    return {"text": normalize(bare.group(0)), "question": question, "evidence": [title], "confidence": None} if bare else None
