"""The prophet's instructions, and reading its answers back (PLAN.md §6 step 11).

The prophet reads the day's pages (`evidence.py`) one at a time, in two steps:
1. **The question** (QUESTION), only for pages whose evidence dates something
   to day D or the HORIZON days after (`marked_within`: "[today]", "[in 3
   days]", "ends tomorrow"). The model says what's decided and on what date,
   then the main question it settles, or "none". The code reads the date and
   drops a question due outside the horizon. A horizon of 0 days asks only
   about D, as run 7 did (PLAN.md §3 item 26).
   - On 16 development page-days whose answers were known, asking every page
     "is anything decided on D?" got "none" for all of them, finals
     included. Saying first what happens that day found the right ones, but
     also wrote questions for finals marked "[tomorrow]". Filtering in code
     by the date marks got 14 of 16 right.
   - A prediction due after D is graded on its due date.
   - A question about a contest one person wins is dropped
     (`checks.one_persons_contest`): milestone 1 names no person.
2. **The prediction** (PREDICTION), for each remaining question: one
   sentence beginning "I predict that", and a confidence.
   - In the first per-page run, "a general description such as 'an
     important politician'" spilled over to teams ("an important team will
     win the gold medal"), and asking for "the result, the number" gave
     three Wild Card games the same score, 5–3. So the prediction names the
     teams, countries or parties, and gives a number only if the evidence
     gives a reason for one.

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

from src.prophecy.evidence import first_date

CONFIDENCES = ("low", "medium", "high")
PREFIX = "I predict that "
SENTENCE = re.compile(r"I predict that [^\n\"]+?(?:\.(?=\s|$)|(?=[\n\"]|$))", re.IGNORECASE)  # to its first full stop

SYSTEM = ("You are the prophet of Zoroaster, a project that foretells real-world events from what is happening on English "
          "Wikipedia. Your own knowledge ends years before these dates, so use only the evidence you are given.")

PAGE = """Here is what a Wikipedia page said by the end of {yesterday}. Each date in it is marked relative to {day}, the day you foretell: [today] means {day}.

{block}"""

HORIZON = 7  # days after D within which a prediction may come due (PLAN.md §6 step 11)

QUESTION = PAGE + """

What does the evidence say will be decided {when}? Look for dates marked [today], [tomorrow] or [in N days], or "ends today": a match or final, votes cast or counted, a result or decision announced.

Answer in three lines:
Event: what the evidence says is decided then, or nothing
Date: the day it's decided, as the evidence gives it (for example 4 October 2026)
Question: the main question that day will settle (who wins, what the result is, what is decided), or none

Leave out anything decided after {last}."""
# A date mark within the horizon: its offset in days is in one of the groups (today and tomorrow have none).
SOON_MARK = re.compile(r"\[(?:today|tomorrow|in (\d+) days)\]|\[starts (?:today|tomorrow|in (\d+) days)"
                       r"|ends (?:today|tomorrow|in (\d+) days)\]")
NOTHING = ("nothing", "none", "no ", "there is nothing", "the evidence does not")
# The prompt's own words, which the model sometimes repeats before its question.
ECHO = re.compile(r"the main question (?:that day|it) will settle is\s+", re.I)
QUESTION_WORD = re.compile(r"(?:(?:in|at|by|for|to|from) )?(?:who|whom|whose|what|which|when|where|whether|why|how|"
                           r"will|would|does|do|did|is|are|can)\b", re.I)

PREDICTION = PAGE + """

The question, settled on {due}: {question}

Answer it with one prediction, using only this evidence.
- One sentence beginning "I predict that", about this page's subject, naming it in full the first time (for example "the film Heart of the Beast", not "the film").
- Answer the question directly: who wins, what the result is, or what is decided. Name the teams, countries or parties involved.
- Give a score or number only if the evidence gives a reason for one. Don't restate what the evidence already reports, and keep it possible.
- Name no specific person, living or dead, and don't point to one by a title or role. A general description of a person, such as "an important politician", is fine.
- Wars, disasters and crime are fine in general terms, but name no organization in them: no company, party, armed group, government body or team. Countries and places are fine.
- Nothing about anyone's health or personal life.

Answer with only a JSON object: {{"prediction": "I predict that ...", "confidence": "low", "medium" or "high"}}"""


def _day_text(day: date) -> str:
    return f"{day:%A}, {day.day} {day:%B %Y}"


def _chat(template: str, day: date, block: str, **fields: str) -> list[dict]:
    text = template.format(day=_day_text(day), yesterday=_day_text(day - timedelta(days=1)), block=block, **fields)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def question_messages(day: date, block: str, horizon: int = HORIZON) -> list[dict]:
    """Step 1, for one page's evidence: what, if anything, is decided on D or within `horizon` days after."""
    last = day + timedelta(days=horizon)
    when = f"on {_day_text(day)} itself" if not horizon else f"between {_day_text(day)} and {_day_text(last)}"
    return _chat(QUESTION, day, block, when=when, last=_day_text(last))


def prediction_messages(day: date, block: str, question: str, due: date) -> list[dict]:
    """Step 2, for a page with a question: the prediction."""
    return _chat(PREDICTION, day, block, question=question, due=_day_text(due))


def _offset(mark: re.Match) -> int:
    """How many days after the foretold day a date mark falls: today 0, tomorrow 1, "in N days" N."""
    days = next((g for g in mark.groups() if g), None)
    return int(days) if days else 1 if "tomorrow" in mark.group(0) else 0


def marked_within(block: str, horizon: int = HORIZON) -> bool:
    """Whether a page's evidence dates anything to the day foretold or the `horizon` days after:
    "[today]", "[in 3 days]", "ends tomorrow", "starts in 2 days"."""
    return any(_offset(m) <= horizon for m in SOON_MARK.finditer(block))


def parse_question(answer: str, day: date, horizon: int = HORIZON) -> tuple[str, date] | None:
    """The question in step 1's answer, and the day it's due: its "Question:" and "Date:" lines. None if the
    "Event:" line says nothing is decided (the model sometimes writes a question anyway), if there's no
    question ("none", "None, as the release date is already set"), or if its date can't be read or falls
    outside the horizon. The date may be a bare mark ("[tomorrow]").

    The question needn't end in "?". Run 8's first attempt required one and lost 160 of 186 questions
    with good dates ("Who wins the gold medal"); a question word then gets one."""
    fields: dict[str, str] = {}
    for line in answer.splitlines():
        label, _, rest = line.partition(":")
        fields.setdefault(label.strip().strip("*").lower(), " ".join(rest.split()).strip('"“”* '))
    event, when = fields.get("event", ""), fields.get("date", "")
    question = ECHO.sub("", fields.get("question", "")).rstrip(". ")
    if event.lower().startswith(NOTHING) or not question or question.lower().startswith(NOTHING):
        return None
    due = {"today": day, "tomorrow": day + timedelta(days=1)}.get(when.lower().strip(". ")) or first_date(when, day)
    mark = SOON_MARK.search(when)
    if due is None and mark:
        due = day + timedelta(days=_offset(mark))
    if due is None or not day <= due <= day + timedelta(days=horizon):
        return None
    question = question[0].upper() + question[1:]
    return question + ("?" if QUESTION_WORD.match(question) and not question.endswith("?") else ""), due


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
