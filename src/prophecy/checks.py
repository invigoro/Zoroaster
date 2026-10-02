"""Checks on the prophet's predictions before any is kept (PLAN.md §6 step 11).

A prediction is dropped if any of these is true:
- **It names or points to a specific person (milestone 1).** General
  descriptions such as "an important politician" don't count (decided
  2026-10-01). Two separate questions are asked of the model, apart from the
  prophet's instructions:
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
  - Words that always mean one person, in the singular after "the" or a
    possessive ("the incumbent", "the coach", "the defending champion";
    PERSON_ROLES), drop a prediction outright. The listing question missed
    "the defending champion will win the darts". After "a" or "an" they're
    general ("a former minister"), and plurals are groups.
  - Except where one begins an event's or club's name (NAME_NOUNS):
    "the 2026 Presidents Cup" was dropped on 2026-09-27, though the
    United States team won it that day.
- **Its question is about a contest one person wins** (`one_persons_contest`,
  checked on the prophet's question before it predicts): a race, a singles
  match, darts. "Red Bull will win the Azerbaijan Grand Prix" points to a
  driver. The page's title or the question's wording can show it
  (ONE_PERSON_EVENTS), and otherwise CONTEST_QUESTION must say teams,
  parties or neither. The question alone called a Grand Prix a contest
  between teams.
- **It has a specific organization doing, suffering or accused of something
  harmful**: a crime, an attack, a disaster, a scandal, a legal case (decided
  2026-10-01). Wars, disasters and crime are fine in general terms: "an
  important politician will rob a major bank" is kept, and "…will rob the Bank
  of America" isn't. Specific people are dropped already, above.
  - Its organizations are listed (ORGS_QUESTION) and each one's kind asked
    (KIND_QUESTION). Countries, places and events don't count.
  - HARM_QUESTIONS then ask, for each organization or team, whether the
    sentence has it doing harm, and whether it suffers harm or faces an
    accusation. Anything but a clear "no" to both drops the prediction. Teams
    winning finals and parties winning elections are kept.
  - A named army or armed group always drops it: a prediction naming one is
    about war, and the harm questions missed "the Israel Defense Forces will
    strike a hospital".
  - On known sentences, a general question ("is it about a war, a crime, a
    disaster…?") said "no" to robbery and to "found guilty", and the
    never-published list (`src/forecast/guardrails.py`) caught "team time
    trial". So the question is about each organization.
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

PERSON_ROLES = re.compile(  # singular only: plurals are groups
    r"\b(?:incumbent|candidate|coach|manager|captain|president|prime minister|premier|minister|chancellor|governor|"
    r"mayor|senator|king|queen|prince|princess|emperor|pope|ceo|chair(?:man|woman|person)|founder|singer|actor|"
    r"actress|rapper|player|driver|rider|boxer|golfer|quarterback|pitcher|striker|goalkeeper|author|director|"
    r"defending champion|reigning champion)\b", re.IGNORECASE)
NAME_NOUNS = re.compile(r"(?:['’]s?)?\s+(?:Cups?|Trophy|Championships?|League|Series|Park|Club|Stakes|Plate|Awards?|Prize"
                        r"|Bowl|Classic|Open|Games|Tour|Medal|Race)\b")
GENERAL = re.compile(r"\b(?:a|an|another|any|some|one|several|many|no)\s+(?:[\w'’-]+\s+){0,2}$", re.IGNORECASE)
NON_PERSON_KINDS = ("team", "country", "organization", "organisation", "place", "event", "work", "other")
NOT_ORG_KINDS = ("person", "country", "place", "event", "work", "other")
WORD = re.compile(r"[a-z0-9]+")
FILLER = frozenset({"i", "predict", "that", "the", "a", "an", "will", "of", "in", "on", "at", "to", "and", "be"})
COPY_OVERLAP, REPEAT_OVERLAP = 0.6, 0.8
EXAMPLE = re.compile(r"I predict that [^\"]+")
SETTLED, NOT_GROUNDED = "the evidence already settles it", "not about its cited evidence"
QUALITY_REASONS = (SETTLED, NOT_GROUNDED)  # the quality checks; the rest are guardrails, copies and repeats

PERSON_QUESTION = """Does the sentence below name a specific person, living or dead, or point to one by a title or role that fits one person, such as "the president of France" or "the team's coach"? General descriptions that fit many people, such as "an important politician" or "a famous singer", don't count. Teams, organizations, places, events and works don't count either.

Sentence: {text}

Answer with one word: yes or no."""

PEOPLE_QUESTION = """List every specific person that the sentence below names, or points to by a title or role that fits one person (such as "the president of France" or "the team's coach"). Don't list general descriptions that fit many people, such as "an important politician" or "protesters". Teams, organizations, places, events and works aren't people.

Sentence: {text}

Answer with the people's names or roles, separated by semicolons, or with the single word: none."""

ONE_PERSON_EVENTS = re.compile(r"Grand Prix|\b[Ss]ingles\b|\b[Dd]oubles\b|[Mm]arathon|\bOpen\b|[Dd]arts|Night Race|"
                               r"Formula (?:One|1|2|3|E)\b|NASCAR|IndyCar|MotoGP|[Bb]oxing|\bUFC\b|[Tt]ime trial|"
                               r"[Rr]oad race|[Ii]ndividual|Masters|Ballon d'Or|\b[Aa]wards?\b|\b[Pp]rizes?\b")
TEAM_CONTESTS = ("team", "part", "neither", "countr", "nation", "club", "side")

CONTEST_QUESTION = """Who competes for the result the question below asks about? It's from the Wikipedia page "{title}".

Question: {question}

Answer with exactly one word: teams (clubs, national teams, a country's team, sides), parties, individuals (one person wins, as in a race, a singles or doubles match, golf strokeplay, darts or boxing, even if the question asks for their country or team), or neither."""

HARM_QUESTIONS = (  # harm done, harm suffered, and the kind of organization: see harms()
    """In the sentence below, does "{org}" attack, strike, fire on, kill, rob, cheat or otherwise harm anyone or anything?

Sentence: {text}

Answer with one word: yes or no.""",
    """In the sentence below, does "{org}" suffer something harmful, such as an attack, a robbery, a disaster or an accident, or face an accusation, a fine, charges or a guilty verdict?

Sentence: {text}

Answer with one word: yes or no.""",
    """What kind of organization is "{org}"? Answer with exactly one word: military (an army, armed force or armed group), company, party, government, team, or other.""")
MILITARY = ("military", "army", "armed")

ORGS_QUESTION = """List every specific organization that the sentence below names or points to, such as a company, bank, political party, armed group, government body, army, team or club. Countries, cities, places, events, and general descriptions such as "a major bank", aren't specific organizations.

Sentence: {text}

Answer with the organizations' names, separated by semicolons, or with the single word: none."""

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
    """Words in `text` that mean one specific person, without repeats. Not where one begins a name
    ("President's Cup", "Drivers' Championship") or follows "a", "an" and the like ("a former minister")."""
    return list(dict.fromkeys(m.group(0).lower() for m in PERSON_ROLES.finditer(text)
                              if not NAME_NOUNS.match(text, m.end()) and not GENERAL.search(text[:m.start()])))


def contest_messages(title: str, question: str) -> list[dict]:
    return [{"role": "user", "content": CONTEST_QUESTION.format(title=title, question=question)}]


def one_persons_contest(title: str, question: str, answer: str) -> bool:
    """Whether a question for the prophet is about a contest one person wins: its page's title or its own
    wording says so (ONE_PERSON_EVENTS), or CONTEST_QUESTION's answer isn't clearly teams, parties or neither."""
    return (bool(ONE_PERSON_EVENTS.search(title) or ONE_PERSON_EVENTS.search(question))
            or not answer.strip().lower().startswith(TEAM_CONTESTS))


def orgs_messages(text: str) -> list[dict]:
    return [{"role": "user", "content": ORGS_QUESTION.format(text=text)}]


def harm_messages(text: str, org: str) -> list[list[dict]]:
    """The harm questions about one organization in a prediction: harm it does, harm it suffers, its kind."""
    return [[{"role": "user", "content": q.format(org=org, text=text)}] for q in HARM_QUESTIONS]


def harms(answers: list[str]) -> bool:
    """The harm questions' verdict on one organization: harm unless both harm questions get a clear "no".
    And a named army or armed group always counts. On known sentences the harm questions missed
    "the Israel Defense Forces will strike a hospital", and a prediction naming one is about war."""
    done, suffered, kind = (a.strip().lower() for a in answers)
    return not (done.startswith("no") and suffered.startswith("no")) or kind.startswith(MILITARY)


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


def confirmed_orgs(names: list[str], answers: list[str]) -> list[str]:
    """The listed names that KIND_QUESTION didn't class as a person, country, place, event, work or other:
    organizations and teams, and any it couldn't class. An empty listing answer names none."""
    return [n for n, a in zip(names, answers, strict=True)
            if n != NO_ANSWER and not a.strip().strip(".").lower().startswith(NOT_ORG_KINDS)]


def already_known(answer: str) -> bool:
    """The novelty check's verdict: only a clear "yes" counts."""
    return answer.strip().lower().startswith("yes")


def screen(predictions: list[dict], person_answers: list[str], people: list[list[str]], orgs: list[list[str]],
           harm_answers: list[list[str]], novelty_answers: list[str], grounded_answers: list[str],
           instructions: str) -> list[dict]:
    """Each prediction with `kept`, and the reasons it was dropped.
    - `people`: each prediction's listed names confirmed as people (`confirmed_people`);
    - `orgs`: its listed names confirmed as organizations (`confirmed_orgs`), and `harm_answers` the harm
      questions' answers for each of them.
    A prediction that passes every other check is dropped if it repeats one kept earlier."""
    examples = instruction_examples(instructions)
    out, kept = [], []
    for prediction, person, named, org, harm, novelty, grounded in zip(
            predictions, person_answers, people, orgs, harm_answers, novelty_answers, grounded_answers, strict=True):
        reasons = []
        if copies_an_example(prediction["text"], examples):
            reasons.append("copies an example from the instructions")
        roles = person_roles(prediction["text"])
        if named or roles:
            reasons.append(f"names or points to a person ({'; '.join([*named, *roles])})")
        elif names_a_person(person):
            reasons.append("names or points to a person")
        harmed = [o for o, a in zip(org, harm, strict=True) if harms(a)]
        if harmed:
            reasons.append(f"harm to or by a specific organization ({'; '.join(harmed)})")
        if already_known(novelty):
            reasons.append(SETTLED)
        if not_grounded(grounded):
            reasons.append(NOT_GROUNDED)
        if not reasons and repeats(prediction["text"], kept):
            reasons.append("repeats an earlier prediction")
        if not reasons:
            kept.append(prediction["text"])
        out.append(prediction | {"kept": not reasons, "dropped_because": reasons, "person_check": person.strip(),
                                 "people_named": named, "orgs_named": org,
                                 "harm_check": [[a.strip() for a in answers] for answers in harm],
                                 "novelty_check": novelty.strip(), "grounded_check": grounded.strip()})
    return out
