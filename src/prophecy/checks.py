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
- **No longer checked: whether it's about its cited evidence** (NOT_GROUNDED,
  still in older records). One single-call run predicted a baseball game
  from a fitness-race page. But the question dropped 49 of 81 predictions,
  agreeing with the hand grades 51 times in 68. In the first per-page run it
  dropped 4 of 9, all on topic ("the Padres will win Game 2" from the Wild
  Card Series). Each prediction is now written from its one page, which
  rules out the mix-up it guarded against.
- **It copies an example sentence from the instructions it was made with,
  or repeats an earlier prediction that day.** Both are word overlap, with
  no model involved.

A prediction dropped only by the guardrails (a specific person, harm to or
by a specific organization) is rewritten in general terms (GENERALIZE) and
checked again from scratch, as the user allowed (PLAN.md §2, 2026-10-02):
"Kevin Bacon will die tomorrow" becoming "A prominent actor will die
tomorrow". The rewrite is kept only if every check passes and it doesn't
repeat a prediction kept that day (`merge_rewrites`). The original stays in
the record, never published.
  - A repeat is dropped only if what it repeats was kept. On 2026-09-18 the
    first of two identical predictions cited the wrong page; the second
    cited the right one and would otherwise have been dropped as the repeat.
"""

from __future__ import annotations

import re

from src.prophecy.countries import names_a_country
from src.prophecy.evidence import MONTHS
from src.prophecy.stories import is_story

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
UNSUPPORTED = "adds details its evidence doesn't give"  # `details.py`, followed by the details
VAGUE = "doesn't name the teams or countries it means"
# The quality checks; the rest are guardrails, copies and repeats. A reason may follow with its details.
QUALITY_REASONS = (SETTLED, NOT_GROUNDED, UNSUPPORTED, VAGUE)
# A page's prediction may name its teams and countries, and should: the first live day's (2026-10-05) "an
# important team representing a major country will win the mixed team event" said neither which, nor at what.
# Not for a story's, where "an armed group" stands in for one the guardrails won't name.
_SOMEWHERE = ("European|Asian|African|American|Latin American|North American|South American|Western|Eastern|Arab|"
              "Nordic|Balkan|Gulf|Pacific|Caribbean")
_SIDES = "team|country|nation|club|side|squad"
PLACEHOLDER = re.compile(
    r"\b(?:an?|one)\s+(?:(?:important|prominent|major|leading|top|strong|big|certain|powerful|well-known|famous|"
    rf"dominant|unnamed|unspecified|other|{_SOMEWHERE})\s+)+(?:{_SIDES})s?\b"
    rf"|\banother\s+(?:{_SIDES})\b"
    # With the prompt asking for the teams, the judo prediction became "won by a team representing a country from Asia".
    rf"|\ba\s+(?:{_SIDES})\s+(?:from|representing|of)\s+an?\b"
    r"|\ban?\s+(?:country|nation)\s+(?:from|in)\s+(?:Asia|Europe|Africa|the Americas|North America|South America|"
    r"Latin America|Oceania|the Middle East|the Caribbean)\b")
# Armed groups that aren't a country's and that the news names often, which count as armed groups (`namable`)
# whatever the model says: the kind question once called the TPLF a party, and the organizations question listed
# none in "the Bab al-Mandeb Strait will be under Houthi control" (2026-10-05). Not every one.
KNOWN_ARMED = re.compile(
    r"(?<![\w-])(?:Houthis?|Ansar Allah|Hamas|Hezbollah|Hizbullah|Palestinian Islamic Jihad|Taliban|TPLF|"
    r"Tigray People's Liberation Front|Rapid Support Forces|Wagner Group|[Aa]l-Shabaab|Islamic State|ISIS|ISIL|"
    r"Daesh|Boko Haram|ISWAP|M23|Fano|PKK|Kurdistan Workers' Party|[Aa]l-Qaeda|AQAP|Hayat Tahrir al-Sham|"
    r"Polisario Front|FARC|Arakan Army|Kachin Independence Army|Viv Ansanm)(?![\w-])")


def known_armed_groups(text: str) -> list[str]:
    """The armed groups KNOWN_ARMED finds in a prediction, each once."""
    return list(dict.fromkeys(KNOWN_ARMED.findall(text)))

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
    """In the sentence below, does "{org}" attack, strike, fire on, kill, rob, cheat, hack, breach or otherwise harm anyone or anything?

Sentence: {text}

Answer with one word: yes or no.""",
    """In the sentence below, does "{org}" suffer something harmful, such as an attack, a robbery, a disaster or an accident, or face an accusation, a fine, charges or a guilty verdict?

Sentence: {text}

Answer with one word: yes or no.""",
    # Asked apart after the first stories from Portal:Current events: "the United States and Australia will
    # hold OpenAI responsible for the breach" got a "no" to the question above. Adding blame to that question
    # instead let "Manchester City will be found guilty" and "Boeing will face criminal charges" through. Of
    # three wordings, this one also caught the passive "OpenAI will be held responsible for…", 12 of 12.
    """In the sentence below, is "{org}", or will it be, blamed, held responsible, accused, investigated, sued, fined, charged or punished for something?

Sentence: {text}

Answer with one word: yes or no.""",
    """What kind of organization is "{org}"? Answer with exactly one word: state (a country's government, ministry, army, navy, air force or police, such as "Ukrainian forces" or the Israel Defense Forces), armed (an armed group, militia or rebel force that isn't a country's own, such as Hamas or the Rapid Support Forces), company, party, team, or other.""")
# A country's government or armed forces stand for the country, which the rules allow to do anything (the user,
# 2026-10-04: "it's fine to list countries or their overall government doing something (e.g. Ukraine's armed
# forces, or as we already have in the second one, Saudi Arabia's air force)"). Before that, any army counted
# as harm, as "the Israel Defense Forces will strike a hospital" did.
STATE_KINDS = ("state", "government")
# Armed groups that aren't a country's may be named too (the user, 2026-10-05: "I'm actually okay with the houthis
# being named ... military organizations can be named, especially if they're from a non-US country and especially
# if they're generally considered to be terrorists ... My main concern is with individuals"). Before that one always
# counted as harm, a prediction naming one being about war.
ARMED_KINDS = ("armed", "military", "army", "militia")

ORGS_QUESTION = """List every specific organization that the sentence below names or points to, such as a company, bank, newspaper or news website, political party, armed group, government body, army, team or club. Countries, cities, places, events, and general descriptions such as "a major bank", aren't specific organizations.

Sentence: {text}

Answer with the organizations' names, separated by semicolons, or with the single word: none."""

KIND_QUESTION = """What kind of thing is "{name}"? Answer with exactly one word: person, team, country, organization, place, event, work, or other. A title or role for one individual, such as "the coach" or "the defending champion", counts as person."""

NOVELTY_QUESTION = """Here is what was known by the end of the day before:

{evidence}

A prediction about the next day: {text}

Does the evidence above already report or settle what this prediction says, for example a date, venue, line-up, schedule or result that is already known? Answer with one word: yes or no."""



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


def is_state(name: str, kind: str) -> bool:
    """Whether an organization stands for its country: the model calls it a country's government or forces
    (STATE_KINDS), and its name says which country (`countries.names_a_country`). The model alone called the
    Rapid Support Forces, Sudan's paramilitary at war with its army, a country's forces, and the UN Security
    Council too."""
    return kind.strip().lower().startswith(STATE_KINDS) and names_a_country(name)


def namable(name: str, kind: str) -> bool:
    """Whether an organization may be named, harm or not, even in a war, a disaster or a crime: a country's
    government or armed forces (`is_state`), or an armed group (ARMED_KINDS). Not a company, a party, a court or
    any other organization."""
    return is_state(name, kind) or kind.strip().lower().startswith(ARMED_KINDS)


def harms(answers: list[str], name: str = "") -> bool:
    """The harm questions' verdict on one organization, `name`: harm unless every harm question gets a clear
    "no". A country's government or armed forces, and armed groups, never count (`namable`). A state body whose
    name names no country is an organization like any other ("the Federal Reserve will cut rates" is fine)."""
    *questions, kind = (a.strip().lower() for a in answers)
    if namable(name, kind):
        return False
    return not all(a.startswith("no") for a in questions)


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


def listed_orgs(names: list[str], answers: list[str]) -> list[str]:
    """The listed organizations that count in a war, a disaster or a crime (IN_SENSITIVE): all but those
    KIND_QUESTION classed as a person, country, place or event. Wider than `confirmed_orgs`: T24, a news
    website whose closure was a crime story, came back "other"."""
    return [n for n, a in zip(names, answers, strict=True)
            if n != NO_ANSWER and not a.strip().strip(".").lower().startswith(("person", "country", "place", "event"))]


def already_known(answer: str) -> bool:
    """The novelty check's verdict: only a clear "yes" counts."""
    return answer.strip().lower().startswith("yes")


SENSITIVE_TOPICS = ("conflict", "disaster", "crime")
# The user's rule (2026-10-01): wars, disasters and crimes are fine, "so long as they're not too specific to a
# person or an organization". The harm questions alone let "the TPLF continues to seize control of more towns"
# through in runs 11 and 12: no harm, said the model, and the TPLF a party, not an armed group. A country's
# government or armed forces stand for the country, which the rule allows ("an armed group will agree to
# negotiate with the Colombian government"), so an organization the harm questions class so (STATE_KINDS)
# doesn't count. Since 2026-10-05 neither do armed groups (`namable`): the user's concern is individuals.
IN_SENSITIVE = "names an organization in a war, a disaster or a crime"
GUARDRAIL_REASONS = ("names or points to a person", "harm to or by a specific organization", IN_SENSITIVE)

# Asked of a page's prediction that names an organization; a story's topic is its category's
# (`stories.CATEGORY_TOPICS`).
SENSITIVE_QUESTION = """What is the sentence below about? Answer with exactly one word: conflict (a war, an armed attack or fighting), disaster (a disaster or an accident), crime (a crime, a trial, an arrest or an investigation), or other.

Sentence: {text}"""


def sensitive_messages(text: str) -> list[dict]:
    return [{"role": "user", "content": SENSITIVE_QUESTION.format(text=text)}]


def sensitive_topic(answer: str) -> bool:
    """SENSITIVE_QUESTION's verdict: a war, a disaster or a crime."""
    return answer.strip().lower().startswith(SENSITIVE_TOPICS)

# Leaving out the details that single one out came after the first rewrite trial: "a prominent political leader
# will be convicted and removed from office by the Philippine Senate" names nobody, but fits one person. Keeping
# countries word for word, and descriptions true of whom they replace, came after run 10: "a prominent actor"
# was copied onto wrestlers, and "Saudi Arabia and the Houthis" became "a prominent actor from a country and an
# important politician from another country". Regions, and countries' governments and forces, came after the
# first live day (2026-10-04): "the TPLF continues to seize control of more towns in Afar and Amhara regions"
# became "a prominent armed group will continue to advance in a region", which the user found too vague to check.
GENERALIZE = """Rewrite the sentence below so that no reader could tell which specific person or organization it is about.
- Replace each specific person or organization{named} with a general description of what it really is, one that fits many: say what kind of person or group it is (an actor, a film producer, a wrestler, a politician, a bank, a technology company, an armed group), adding "prominent" or "major" only if that fits.
- Also leave out any detail that singles one out: a particular office or title, the institution or court that judges them, a vote count, a sentence's length.
- Countries, nationalities, regions, cities and places are not organizations: keep each one exactly as the sentence writes it, so a reader can still tell where it happens.
- A country's government or armed forces stand for the country: name them by it ("Ukraine's armed forces", "the Ethiopian government").
- Keep the rest as it is.

Sentence: {text}

Answer with only the rewritten sentence."""


def generalize_messages(text: str, names: list[str] | None = None) -> list[dict]:
    """The rewrite in general terms, for a prediction or, once published, the judge's reason for a grade.
    `names`: the people and organizations the checks dropped it for (`reason_names`), named to the model so it
    replaces those and keeps the rest. Told only to generalize, it also turned "Tel Aviv" into "a major city"
    and Israel into "a prominent actor" (2026-10-04)."""
    named = f" (here: {'; '.join(names)})" if names else ""
    return [{"role": "user", "content": GENERALIZE.format(text=text, named=named)}]


def armed_groups_only(prediction: dict) -> bool:
    """Whether the guardrails dropped a prediction only for organizations, no person, and either all of them armed
    groups that aren't a country's (`harms`) or in a war. Its rewrite isn't asked whether it still points to them
    (IDENTIFIES_QUESTION): the user asked for the place a group fights to be named (2026-10-04), and that often makes
    plain which group it is. Asked, the model dropped "a prominent armed group continues to seize control of more
    towns in Afar and Amhara regions", the TPLF's, which the kind question once called an armed group, once a party."""
    reasons = prediction["dropped_because"]
    if not reasons or any(r.startswith(GUARDRAIL_REASONS[0]) for r in reasons):  # a person
        return False
    names = reason_names(reasons)
    war = prediction.get("story_topic") == "conflict" or prediction.get("topic_check", "").lower().startswith("conflict")
    kinds = {o: (a[-1].strip().lower() if a else "")
             for o, a in zip(prediction.get("orgs_named", []), prediction.get("harm_check", []))}
    return bool(names) and (war or all(kinds.get(n, "").startswith(ARMED_KINDS) for n in names))


def reason_names(reasons: list[str]) -> list[str]:
    """The names in a prediction's guardrail reasons: "harm to or by a specific organization (A; B)" gives A
    and B. Only names, with a capital: the person check once listed "individuals"."""
    names = [n.strip() for r in reasons if r.startswith(GUARDRAIL_REASONS) and r.endswith(")") and "(" in r
             for n in r[r.rindex("(") + 1:-1].split(";")]
    return [n for n in dict.fromkeys(names) if n and n != n.lower()]


IDENTIFIABLE = "still points to the one it was about"
# Asked of a rewrite only, beside the original: vague words aren't enough if the situation still fits only
# the one it was about. "An important political leader will be convicted and removed from office by the
# Philippine Senate" names nobody, but there was one impeachment trial, and the model's own knowledge ends
# before it, so asked of the sentence alone it said no. The user's own example sets the line (PLAN.md §2):
# "a prominent actor will die tomorrow", rewritten from a sentence about one actor, is fine.
IDENTIFIES_QUESTION = """A sentence about a specific person or organization was rewritten so that a reader couldn't tell whom it means.

The original: {original}
The rewrite: {text}

Does the rewrite still clearly point to the same person or organization, so that a reader who follows the news would know whom it means? A description that fits many doesn't count: "a prominent actor will die tomorrow", rewritten from a sentence about one actor, doesn't point to that actor.

Answer with one word: yes or no."""


# The answer recorded for a rewrite not asked (`armed_groups_only`): a "no", which keeps it.
NOT_ASKED = "no (not asked: armed groups only, named by where they fight)"


def identifies_messages(text: str, original: str) -> list[dict]:
    return [{"role": "user", "content": IDENTIFIES_QUESTION.format(text=text, original=original)}]


STILL_NAMES = "the rewrite still names a specific organization"


def apply_identifies(rewrites: list[dict], answers: list[str]) -> list[dict]:
    """The rewrites, each kept one dropped if it still names any specific organization (`orgs_named`, from
    the checks), or unless IDENTIFIES_QUESTION got a clear "no" (`answers`, one per rewrite, empty for one
    already dropped).
    - In the first rewrite trial the question said "no" to every rewrite, even with the original beside it.
      What still pointed to the one meant was a named institution: "an important political leader will be
      convicted and removed from office by the Philippine Senate", "a major technology company will be held
      responsible for the infiltration of Australia's Medicare database". So a rewrite may name no
      organization at all, harm or not; only countries and places."""
    out = []
    for r, answer in zip(rewrites, answers, strict=True):
        # A country's government or armed forces may stay, named by its country ("Ethiopia's armed forces").
        kinds = [a[-1] if a else "" for a in r.get("harm_check") or []]
        named = [o for i, o in enumerate(r.get("orgs_named", [])) if not (i < len(kinds) and namable(o, kinds[i]))]
        if r["kept"] and named:
            r = r | {"kept": False, "dropped_because": [f"{STILL_NAMES} ({'; '.join(named)})"]}
        elif r["kept"] and not answer.strip().lower().startswith("no"):
            r = r | {"kept": False, "dropped_because": [IDENTIFIABLE]}
        out.append(r | {"identifies_check": answer.strip()})
    return out


def general_mention(name: str, text: str) -> bool:
    """Whether `text` mentions a listed organization only as a general description, after "a", "an" or the
    like: "an armed group", which the organizations question listed as one."""
    found = [m.start() for m in re.finditer(re.escape(name), text, re.IGNORECASE)]
    return bool(found) and all(GENERAL.search(text[:start]) for start in found)


def guarded_only(prediction: dict) -> bool:
    """Whether the guardrails alone dropped a prediction, so a rewrite in general terms may save it."""
    reasons = prediction["dropped_because"]
    return bool(reasons) and all(r.startswith(GUARDRAIL_REASONS) for r in reasons)


KEPT_NAME = "the rewrite keeps a name its original was dropped for"
KEPT_DETAIL = "the rewrite about a person keeps a detail that may single them out"
CAPITAL = re.compile(r"(?<![\w'’])[A-Z][\w'’-]*")
DATES_AND_SUCH = {"I", "The", "A", "An", *MONTHS, "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                  "Sunday"}


def kept_details(original: str, rewrite: str) -> list[str]:
    """The capitalized words a rewrite keeps from its original, but for countries, nationalities and dates: for a
    person, a named institution or event can single them out. In run 14 "the Second Impeachment of Vice President
    Sara Duterte will result in a 14-14 tie in the Philippine Senate" became "the Second Impeachment of a prominent
    politician will result in a 14-14 tie in the Philippine Senate", and the question whether it still points to
    her said no."""
    before = set(CAPITAL.findall(original))
    return list(dict.fromkeys(w for w in CAPITAL.findall(rewrite)
                              if w in before and w not in DATES_AND_SUCH and not names_a_country(w)))


def kept_names(reasons: list[str], rewrite: str) -> list[str]:
    """The names a prediction was dropped for, by the guardrails ("harm to or by a specific organization (A; B)"),
    that its rewrite still has, word for word. Run 13 published "the Sixth Circuit Court will uphold the stay on a
    prominent individual's execution": the checks of the rewrite didn't list the court, which they had of the
    original. Only names, with a capital (`reason_names`): a role such as "president" is checked again in the
    rewrite itself (`person_roles`)."""
    return [n for n in reason_names(reasons) if re.search(rf"(?<!\w){re.escape(n)}(?!\w)", rewrite)]


def merge_rewrites(screened: list[dict], rewritten: dict[int, dict]) -> list[dict]:
    """The day's predictions, each guarded one replaced by its rewrite (`rewritten`: index -> the rewrite as
    screened) if every check passed it, it keeps no name its original was dropped for (`kept_names`), and it
    repeats no prediction kept that day. Otherwise the original stays dropped, with the rewrite and why it
    failed."""
    kept = [p["text"] for p in screened if p["kept"]]
    out = []
    for i, p in enumerate(screened):
        r = rewritten.get(i)
        if r is None:
            out.append(p)
            continue
        still = kept_names(p["dropped_because"], r["text"])
        # For a person, stricter: no organization at all, a country's either, and no other detail kept but places.
        person = any(x.startswith(GUARDRAIL_REASONS[0]) for x in p["dropped_because"])
        detail = [*r.get("orgs_named", []), *kept_details(p["text"], r["text"])] if person else []
        if r["kept"] and not still and not detail and not repeats(r["text"], kept):
            kept.append(r["text"])
            out.append(r | {"rewritten_from": p["text"], "dropped_before_rewrite": p["dropped_because"]})
        else:
            why = (r["dropped_because"] or ([f"{KEPT_NAME} ({'; '.join(still)})"] if still else [])
                   or ([f"{KEPT_DETAIL} ({'; '.join(dict.fromkeys(detail))})"] if detail else [])
                   or ["repeats an earlier prediction"])
            out.append(p | {"rewrite": r["text"], "rewrite_dropped_because": why})
    return out


def screen(predictions: list[dict], person_answers: list[str], people: list[list[str]], orgs: list[list[str]],
           harm_answers: list[list[str]], novelty_answers: list[str], instructions: str,
           sensitive: list[bool] | None = None, listed: list[list[str]] | None = None) -> list[dict]:
    """Each prediction with `kept`, and the reasons it was dropped.
    - `people`: each prediction's listed names confirmed as people (`confirmed_people`);
    - `orgs`: its listed names confirmed as organizations (`confirmed_orgs`), and `harm_answers` the harm
      questions' answers for each of them;
    - `sensitive`: whether each is about a war, a disaster or a crime, where it may name no organization
      but a government (IN_SENSITIVE), of those `listed` (`listed_orgs`; by default, `orgs`).
    A prediction that passes every other check is dropped if it repeats one kept earlier."""
    examples = instruction_examples(instructions)
    sensitive = sensitive or [False] * len(predictions)
    listed = listed or orgs
    out, kept = [], []
    for prediction, person, named, org, harm, novelty, delicate, wide in zip(
            predictions, person_answers, people, orgs, harm_answers, novelty_answers, sensitive, listed, strict=True):
        reasons = []
        if copies_an_example(prediction["text"], examples):
            reasons.append("copies an example from the instructions")
        roles = person_roles(prediction["text"])
        if named or roles:
            reasons.append(f"names or points to a person ({'; '.join([*named, *roles])})")
        elif names_a_person(person):
            reasons.append("names or points to a person")
        harmed = [o for o, a in zip(org, harm, strict=True) if harms(a, o)]
        if harmed:
            reasons.append(f"harm to or by a specific organization ({'; '.join(harmed)})")
        kinds = {o: a[-1] for o, a in zip(org, harm, strict=True)}  # the harm questions' last
        in_sensitive = [o for o in dict.fromkeys([*org, *wide])
                        if delicate and o not in harmed and not namable(o, kinds.get(o, ""))]
        if in_sensitive:
            reasons.append(f"{IN_SENSITIVE} ({'; '.join(in_sensitive)})")
        if already_known(novelty):
            reasons.append(SETTLED)
        vague = PLACEHOLDER.search(prediction["text"])
        if vague and not all(is_story(t) for t in prediction.get("evidence") or [""]):
            reasons.append(f"{VAGUE} ({vague.group(0)})")
        if not reasons and repeats(prediction["text"], kept):
            reasons.append("repeats an earlier prediction")
        if not reasons:
            kept.append(prediction["text"])
        out.append(prediction | {"kept": not reasons, "dropped_because": reasons, "person_check": person.strip(),
                                 "people_named": named, "orgs_named": org, "sensitive": delicate,
                                 "harm_check": [[a.strip() for a in answers] for answers in harm],
                                 "novelty_check": novelty.strip()})
    return out
