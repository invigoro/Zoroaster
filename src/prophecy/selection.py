"""Which of a day's kept predictions are published: at most DAILY, world events first (PLAN.md §2).

The user (2026-10-02): "reduce the number of total predictions per day, maybe
to 10 or fewer, and prioritize non-sports ones, only taking the most
interesting sports predictions... I'm much more interested in world events."
- A prediction is sport if its page is (`is_sport_page`: the page's infobox
  or categories at the end of D-1 name a sport, a sports event or a
  tournament), or if the topic question says so. Otherwise it's a world
  event, labelled by the topic question (TOPIC_QUESTION).
  - The topic question alone wasn't enough: on run 8 it called 66 of 134
    kept predictions "other", most of them sport ("Who wins the gold
    medal?" on an Asian Games badminton page).
- World events come first, best ranked first: the rank in the burst prophecy
  of the page the prediction comes from. Then the stories of Portal:Current
  events (`stories.py`), best covered first, each labelled by its category.
- A sports prediction is published only if it settles a title, at most SPORTS
  a day, after the world events. It settles one if its page is a final
  (FINAL_PAGE: "2026 AFL Grand Final"), or if TITLE_QUESTION says so.
  - TITLE_QUESTION's first wording, without examples, said no to all 134
    on run 8, "Who wins the gold medal?" too. This one said yes to 36 of
    121 questions, nearly all of them finals, gold medals and titles.
"""

from __future__ import annotations

import re

DAILY, SPORTS = 10, 3

TOPIC_QUESTION = """What is the question below about? It's from the Wikipedia page "{title}".

Question: {question}

Answer with exactly one word: sport, politics, conflict, disaster, economy, science, health, culture, or other."""

TITLE_QUESTION = """Does this question decide a champion: who wins a final, a gold medal, a title or a championship?

Examples:
- "Who wins the gold medal?" yes
- "Which team wins the premiership?" (page: 2026 AFL Grand Final) yes
- "Who wins the final?" yes
- "Who wins the tournament?" yes
- "Who advances to the semi-finals?" no
- "Who wins each match?" no
- "Who wins Game 1?" no
- "Which teams advance to the Division Series?" no

The question: "{question}" (page: {title})

Answer with one word: yes or no."""

# Words that mark sport in an infobox's name ("international football competition", "Wrestling event") or a
# category ("Current sports events", "2026 in figure skating"). Not "shooting", which mass shootings share.
SPORT = re.compile(
    r"\bsports?\b(?! utility| cars?\b| bikes?\b)|\bsporting\b|football|basketball|baseball|softball|volleyball"
    r"|handball|hockey|cricket|rugby|tennis|badminton|golf|darts|snooker|chess|boxing|wrestling|judo|taekwondo"
    r"|fencing|athletics|swimming|rowing|cycling|motorsport|racing|racehorse|grand prix|formula one|nascar|skating"
    r"|skiing|curling|gymnastics|weightlifting|archery|triathlon|esports|olympic|paralympic|asian games"
    r"|commonwealth games|world cup|playoffs|postseason|championship|tournament", re.IGNORECASE)
# In an infobox's name only: "country at games", "Grand Prix race report". Video games' categories share
# them ("PlayStation 5 games"), and elections' short descriptions ("gubernatorial race").
SPORT_EVENT = re.compile(r"\bgames\b|\brace\b", re.IGNORECASE)
# About a work, not an event: "Sports films", "Racing video games". Unless it's a competition.
WORK = re.compile(r"\bfilms?\b|video games?|television|\balbums?\b|\bsongs?\b|\bnovels?\b|\bbooks?\b", re.IGNORECASE)
COMPETITION = re.compile(r"tournament|championship|esports", re.IGNORECASE)
INFOBOX = re.compile(r"\{\{\s*Infobox[ _]+([^|\n}<]+)", re.IGNORECASE)
SHORT_DESCRIPTION = re.compile(r"\{\{\s*Short description\s*\|([^}\n]+)", re.IGNORECASE)
CATEGORY = re.compile(r"\[\[\s*Category\s*:\s*([^|\]\n]+)", re.IGNORECASE)
# Templates only sports pages use, for the many without an infobox (a group stage, a qualifying group, a
# tennis draw): match boxes, league tables, the football flag icons ({{fb|ENG}}, {{fbu}}, {{fbaicon}}).
SPORT_TEMPLATE = re.compile(r"\{\{\s*(?:(?i:#invoke:\s*sports|(?:football|basketball|rugby|volleyball|handball|hockey)"
                            r" ?box|tennis events|single-innings cricket match|limited overs matches|test match)"
                            r"|fb[a-z]*\s*\|)")
# A page about a final itself, whatever its question asks ("Which team wins the match?").
FINAL_PAGE = re.compile(r"(?<!semi-)(?<!quarter-)\b(?:grand )?finals?\b|\bsuper bowl\b|\bworld series$", re.IGNORECASE)


def is_sport_page(page_text: str) -> bool:
    """Whether a page is about sport, as it stood at the end of D-1: its first infobox's name or its short
    description names a sport or a sports event, it uses a sports template, or a category names a sport.
    A work about sport, such as a film, is culture."""
    box, short = INFOBOX.search(page_text), SHORT_DESCRIPTION.search(page_text)
    for described, words in ((box, (SPORT, SPORT_EVENT)), (short, (SPORT,))):
        text = described.group(1) if described else ""
        if WORK.search(text) and not COMPETITION.search(text):  # "Video game tournament series" is sport
            return False
        if any(w.search(text) for w in words):
            return True
    return bool(SPORT_TEMPLATE.search(page_text)) or any(SPORT.search(c) and not WORK.search(c)
                                                         for c in CATEGORY.findall(page_text))


def topic_messages(title: str, question: str) -> list[dict]:
    return [{"role": "user", "content": TOPIC_QUESTION.format(title=title, question=question)}]


def title_messages(title: str, question: str) -> list[dict]:
    return [{"role": "user", "content": TITLE_QUESTION.format(title=title, question=question)}]


def topic(answer: str) -> str:
    """The topic word in TOPIC_QUESTION's answer, or a story's (`stories.CATEGORY_TOPICS`); "other" if it's
    none of them."""
    word = answer.strip().strip(".").lower().split(" ")[0] if answer.strip() else ""
    return word if word in ("sport", "politics", "conflict", "disaster", "economy", "science", "health", "culture",
                            "crime") else "sport" if word.startswith("sport") else "other"


def settles_a_title(answer: str) -> bool:
    """TITLE_QUESTION's verdict: only a clear "yes" lets a sports prediction through."""
    return answer.strip().lower().startswith("yes")


def select(predictions: list[dict], ranks: dict[str, int], topics: list[str], titles: list[str],
           sport_pages: set[str] = frozenset()) -> list[dict]:
    """Each prediction with `published`, and for a kept one not published, why. `topics` and `titles`
    are the two questions' answers for each prediction (empty for those the checks dropped); `ranks`
    gives each page's rank in the prophecy, and `sport_pages` the titles of its sports pages."""
    def rank(p: dict) -> int:
        return min((ranks.get(t, 10**6) for t in p["evidence"]), default=10**6)

    def label(p: dict, topic_answer: str) -> str:
        return "sport" if any(t in sport_pages for t in p["evidence"]) else topic(topic_answer)

    def title_decider(p: dict, title_answer: str) -> bool:
        return settles_a_title(title_answer) or any(FINAL_PAGE.search(t) for t in p["evidence"])

    labelled = [p | {"topic": label(p, a) if p["kept"] else None, "settles_a_title": p["kept"] and title_decider(p, t)}
                for p, a, t in zip(predictions, topics, titles, strict=True)]
    kept = sorted((i for i, p in enumerate(labelled) if p["kept"]), key=lambda i: rank(labelled[i]))
    world = [i for i in kept if labelled[i]["topic"] != "sport"]
    sport = [i for i in kept if labelled[i]["topic"] == "sport" and labelled[i]["settles_a_title"]]
    chosen = world[:DAILY]
    chosen += sport[: min(SPORTS, DAILY - len(chosen))]
    out = []
    for i, p in enumerate(labelled):
        why = None
        if p["kept"] and i not in chosen:
            why = ("a sports prediction that settles no title" if p["topic"] == "sport" and not p["settles_a_title"]
                   else f"over the day's limit: {DAILY}, at most {SPORTS} of them sport")
        out.append(p | {"published": i in chosen, "unpublished_because": why})
    return out
