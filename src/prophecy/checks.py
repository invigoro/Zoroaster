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
"""

from __future__ import annotations

from src.forecast.guardrails import sensitive_words

PERSON_QUESTION = """Does the sentence below name a specific person, living or dead, or point to one, for example by a title or role such as "the president of France" or "the team's coach"? Teams, organizations, places, events and works don't count.

Sentence: {text}

Answer with one word: yes or no."""

PEOPLE_QUESTION = """List every person that the sentence below names, or points to by a title or role (such as "the president of France" or "the team's coach"). Teams, organizations, places, events and works aren't people.

Sentence: {text}

Answer with the people's names or roles, separated by semicolons, or with the single word: none."""


def person_messages(text: str) -> list[dict]:
    return [{"role": "user", "content": PERSON_QUESTION.format(text=text)}]


def people_messages(text: str) -> list[dict]:
    return [{"role": "user", "content": PEOPLE_QUESTION.format(text=text)}]


def names_a_person(answer: str) -> bool:
    """The yes-or-no check's verdict: anything but a clear "no" counts as naming a person."""
    return not answer.strip().lower().startswith("no")


def lists_people(answer: str) -> bool:
    """The listing check's verdict: anything but "none" counts as naming a person."""
    return answer.strip().strip(".").strip().lower() != "none"


def screen(predictions: list[dict], person_answers: list[str], people_answers: list[str]) -> list[dict]:
    """Each prediction with `kept`, and the reasons it was dropped."""
    out = []
    for prediction, person, people in zip(predictions, person_answers, people_answers, strict=True):
        reasons = []
        words = sensitive_words(prediction["text"])
        if words:
            reasons.append(f"sensitive topic ({', '.join(words)})")
        if names_a_person(person) or lists_people(people):
            reasons.append("names or points to a person")
        out.append(prediction | {"kept": not reasons, "dropped_because": reasons,
                                 "person_check": person.strip(), "people_check": people.strip()})
    return out
