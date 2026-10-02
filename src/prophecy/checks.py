"""Checks on the prophet's predictions before any is kept (PLAN.md §6 step 11).

A prediction is dropped if any of these is true:
- **It touches a topic on the never-published list** (`src/forecast/guardrails.py`).
- **It names or points to a person (milestone 1).** Two separate questions
  are asked of the model, apart from the prophet's instructions:
  - PERSON_QUESTION: yes or no. Anything but a clear "no" counts as a person.
  - PEOPLE_QUESTION: list the people, or say "none". Anything but "none"
    counts as a person.
  In the first run the yes-or-no question alone kept "…with Luke Hodge
  carrying the premiership cup", so both questions are asked, and either
  one can drop a prediction.
  - The list sometimes names a team, country or party and then hedges
    ("Atlanta Braves; none"). So each listed name gets KIND_QUESTION, and
    only names classed as people count (`confirmed_people`). On 15 known
    names, asking for the kind of thing got all 15 right; yes-or-no
    phrasings got 13 and 14.
  - Words that always mean one person ("incumbent", "coach", "defending
    champion"; PERSON_ROLES) drop a prediction outright. The listing
    question missed "the defending champion will win the darts".
- **The evidence already settles it** (NOVELTY_QUESTION). The model sees only
  the cited pages' evidence, as of the end of the day before, and nothing
  from the day itself. This is a quality filter, so only a clear "yes" drops
  a prediction.
- **It isn't about its cited evidence** (GROUNDED_QUESTION), on the same
  evidence. Only a clear "no" drops it. One run predicted a baseball game
  from a fitness-race page.
- **It copies an example sentence from the instructions it was made with,
  or repeats an earlier prediction that day.** Both are word overlap, with
  no model involved.
  - A repeat is dropped only if what it repeats was kept. On 2026-09-18 the
    first of two identical predictions cited the wrong page; the second
    cited the right one and would otherwise have been dropped as the repeat.
"""

from __future__ import annotations

import re

from src.forecast.guardrails import sensitive_words

PERSON_ROLES = re.compile(
    r"\b(?:incumbents?|candidates?|coach(?:es)?|managers?|captains?|presidents?|prime ministers?|premiers?|ministers?|"
    r"chancellors?|governors?|mayors?|senators?|kings?|queens?|princes?|princess(?:es)?|emperors?|popes?|ceos?|"
    r"chair(?:man|woman|person)|founders?|singers?|actors?|actress(?:es)?|rappers?|players?|drivers?|riders?|"
    r"boxers?|golfers?|quarterbacks?|pitchers?|strikers?|goalkeepers?|authors?|directors?|defending champions?|"
    r"reigning champions?)\b", re.IGNORECASE)
NON_PERSON_KINDS = ("team", "country", "organization", "organisation", "place", "event", "work", "other")
WORD = re.compile(r"[a-z0-9]+")
FILLER = frozenset({"i", "predict", "that", "the", "a", "an", "will", "of", "in", "on", "at", "to", "and", "be"})
COPY_OVERLAP, REPEAT_OVERLAP = 0.6, 0.8
EXAMPLE = re.compile(r"I predict that [^\"]+")

PERSON_QUESTION = """Does the sentence below name a specific person, living or dead, or point to one, for example by a title or role such as "the president of France" or "the team's coach"? Teams, organizations, places, events and works don't count.

Sentence: {text}

Answer with one word: yes or no."""

PEOPLE_QUESTION = """List every person that the sentence below names, or points to by a title or role (such as "the president of France" or "the team's coach"). Teams, organizations, places, events and works aren't people.

Sentence: {text}

Answer with the people's names or roles, separated by semicolons, or with the single word: none."""

KIND_QUESTION = """What kind of thing is "{name}"? Answer with exactly one word: person, team, country, organization, place, event, work, or other. A title or role for one individual, such as "the coach" or "the defending champion", counts as person."""

NOVELTY_QUESTION = """Here is what was known by the end of the day before:

{evidence}

A prediction about the next day: {text}

Does the evidence above already report or settle what this prediction says, for example a date, venue, line-up, schedule or result that is already known? Answer with one word: yes or no."""


GROUNDED_QUESTION = """Here is the evidence a prediction cites, as known by the end of the day before:

{evidence}

The prediction, about the next day: {text}

Is the prediction about the event or subject in this evidence, with the evidence giving a reason to make it? Answer with one word: yes or no."""


def _words(text: str) -> set[str]:
    return set(WORD.findall(text.lower())) - FILLER


def overlap(a: str, b: str) -> float:
    """The share of content words two sentences share (Jaccard)."""
    x, y = _words(a), _words(b)
    return len(x & y) / len(x | y) if x | y else 1.0


def instruction_examples(instructions: str) -> list[str]:
    """The example predictions written into a prophet's instructions."""
    return EXAMPLE.findall(instructions)


def copies_an_example(text: str, examples: list[str]) -> bool:
    return any(overlap(text, e) >= COPY_OVERLAP for e in examples)


def repeats(text: str, earlier: list[str]) -> bool:
    """Whether `text` says nearly the same as one of `earlier`."""
    return any(overlap(text, e) >= REPEAT_OVERLAP for e in earlier)


def grounded_messages(text: str, cited_blocks: list[str]) -> list[dict]:
    evidence = "\n\n".join(cited_blocks) or "(no pages cited)"
    return [{"role": "user", "content": GROUNDED_QUESTION.format(evidence=evidence, text=text)}]


def not_grounded(answer: str) -> bool:
    """The grounding check's verdict: only a clear "no" counts."""
    return answer.strip().lower().startswith("no")


def person_messages(text: str) -> list[dict]:
    return [{"role": "user", "content": PERSON_QUESTION.format(text=text)}]


def people_messages(text: str) -> list[dict]:
    return [{"role": "user", "content": PEOPLE_QUESTION.format(text=text)}]


def novelty_messages(text: str, cited_blocks: list[str]) -> list[dict]:
    evidence = "\n\n".join(cited_blocks) or "(no pages cited)"
    return [{"role": "user", "content": NOVELTY_QUESTION.format(evidence=evidence, text=text)}]


def names_a_person(answer: str) -> bool:
    """The yes-or-no check's verdict: anything but a clear "no" counts as naming a person."""
    return not answer.strip().lower().startswith("no")


def kind_messages(name: str) -> list[dict]:
    return [{"role": "user", "content": KIND_QUESTION.format(name=name)}]


def person_roles(text: str) -> list[str]:
    """Words in `text` that always mean one person, without repeats."""
    return list(dict.fromkeys(m.group(0).lower() for m in PERSON_ROLES.finditer(text)))


NO_ANSWER = "(no answer)"  # an empty listing answer: an unknown person, never asked about


def listed_names(answer: str) -> list[str]:
    """The names in the listing check's answer, without "none"; [NO_ANSWER] if the answer was empty."""
    names = [n.strip() for n in answer.replace("\n", ";").split(";")]
    names = [n for n in names if n.strip(". ") and n.lower().strip(". ") != "none"]
    return names if answer.strip() else [NO_ANSWER]


def confirmed_people(names: list[str], answers: list[str]) -> list[str]:
    """The listed names that KIND_QUESTION didn't class as something other than a person. NO_ANSWER always counts."""
    return [n for n, a in zip(names, answers, strict=True)
            if n == NO_ANSWER or not a.strip().strip(".").lower().startswith(NON_PERSON_KINDS)]


def already_known(answer: str) -> bool:
    """The novelty check's verdict: only a clear "yes" counts."""
    return answer.strip().lower().startswith("yes")


def screen(predictions: list[dict], person_answers: list[str], people: list[list[str]], novelty_answers: list[str],
           grounded_answers: list[str], instructions: str) -> list[dict]:
    """Each prediction with `kept`, and the reasons it was dropped. `people`: each
    prediction's listed names that were confirmed as people (`confirmed_people`).
    A prediction that passes every other check is dropped if it repeats one kept earlier."""
    examples = instruction_examples(instructions)
    out, kept = [], []
    for prediction, person, named, novelty, grounded in zip(
            predictions, person_answers, people, novelty_answers, grounded_answers, strict=True):
        reasons = []
        if copies_an_example(prediction["text"], examples):
            reasons.append("copies an example from the instructions")
        words = sensitive_words(prediction["text"])
        if words:
            reasons.append(f"sensitive topic ({', '.join(words)})")
        roles = person_roles(prediction["text"])
        if named or roles:
            reasons.append(f"names or points to a person ({'; '.join([*named, *roles])})")
        elif names_a_person(person):
            reasons.append("names or points to a person")
        if already_known(novelty):
            reasons.append("the evidence already settles it")
        if not_grounded(grounded):
            reasons.append("not about its cited evidence")
        if not reasons and repeats(prediction["text"], kept):
            reasons.append("repeats an earlier prediction")
        if not reasons:
            kept.append(prediction["text"])
        out.append(prediction | {"kept": not reasons, "dropped_because": reasons, "person_check": person.strip(),
                                 "people_named": named, "novelty_check": novelty.strip(),
                                 "grounded_check": grounded.strip()})
    return out
