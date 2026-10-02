"""Which of a day's kept predictions are published: at most DAILY, world events first (PLAN.md §2).

The user (2026-10-02): "reduce the number of total predictions per day, maybe
to 10 or fewer, and prioritize non-sports ones, only taking the most
interesting sports predictions... I'm much more interested in world events."
- Each kept prediction's question is classed by topic (TOPIC_QUESTION): sport,
  or a kind of world event.
- World events come first, best ranked first: the rank in the burst prophecy
  of the page the prediction comes from.
- A sports prediction is published only if its question settles a title
  (TITLE_QUESTION: a final, a gold medal match, a championship, or the game or
  series that decides one), at most SPORTS a day, after the world events.
"""

from __future__ import annotations

DAILY, SPORTS = 10, 3

TOPIC_QUESTION = """What is the question below about? It's from the Wikipedia page "{title}".

Question: {question}

Answer with exactly one word: sport, politics, conflict, disaster, economy, science, health, culture, or other."""

TITLE_QUESTION = """Does the question below settle who wins a title: a final, a gold medal match, a championship, or the game or series that decides one? It's from the Wikipedia page "{title}".

Question: {question}

Answer with one word: yes or no."""


def topic_messages(title: str, question: str) -> list[dict]:
    return [{"role": "user", "content": TOPIC_QUESTION.format(title=title, question=question)}]


def title_messages(title: str, question: str) -> list[dict]:
    return [{"role": "user", "content": TITLE_QUESTION.format(title=title, question=question)}]


def topic(answer: str) -> str:
    """The topic word in TOPIC_QUESTION's answer; "other" if it's none of them."""
    word = answer.strip().strip(".").lower().split(" ")[0] if answer.strip() else ""
    return word if word in ("sport", "politics", "conflict", "disaster", "economy", "science", "health", "culture") \
        else "sport" if word.startswith("sport") else "other"


def settles_a_title(answer: str) -> bool:
    """TITLE_QUESTION's verdict: only a clear "yes" lets a sports prediction through."""
    return answer.strip().lower().startswith("yes")


def select(predictions: list[dict], ranks: dict[str, int], topics: list[str], titles: list[str]) -> list[dict]:
    """Each prediction with `published`, and for a kept one not published, why. `topics` and `titles`
    are the two questions' answers for each prediction (empty for those the checks dropped); `ranks`
    gives each page's rank in the prophecy."""
    def rank(p: dict) -> int:
        return min((ranks.get(t, 10**6) for t in p["evidence"]), default=10**6)

    labelled = [p | {"topic": topic(a) if p["kept"] else None, "settles_a_title": p["kept"] and settles_a_title(t)}
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
