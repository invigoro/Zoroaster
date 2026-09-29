"""Rank the bursting neighbors' new sentences by relevance to this page.

The first snippets took each neighbor's longest new prose. They shared no
words with the edit's inserted text in 75% of cases (PLAN.md §5). This
ranks every new sentence on every shown neighbor instead, using only what
the prompt already has:
- a sentence that names this page comes first;
- the rest are ordered by the content words they share with the page's
  title, section and the text around the edit. Words are weighted by
  rarity (idf), and the sum is divided by the square root of the
  sentence's length, so long sentences don't win on size.

A sentence qualifies if it names the page or shares at least MIN_SHARED
content words. A content word is one found in under MAX_DF_SHARE of the
sentences the idf was fitted on, which drops words like "with" and "their".
Qualifying sentences are taken in order until BUDGET_CHARS.

It's point-in-time: the neighbor text is from the day before, and the page
text is what the edit started from.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from src.stage2.wikitext import prose

WORD = re.compile(r"[a-z][a-z'-]{3,}")
MIN_SENTENCE_CHARS = 25
MAX_SENTENCE_CHARS = 200
BUDGET_CHARS = 450  # the same as three 150-character snippets
MIN_SHARED = 2
MAX_DF_SHARE = 0.02
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def words(text: str) -> set[str]:
    """Lowercased words of 4+ letters, as in `scripts/analyze_stage2.py`."""
    return set(WORD.findall(text.lower()))


def sentences(spans: Iterable[str]) -> list[str]:
    """The prose sentences in a page's new text (`build_stage2_neighbor_changes.py` spans)."""
    out = []
    for span in spans:
        for piece in prose(span):
            out += [s for s in (x.strip() for x in _SENTENCE_END.split(piece)) if len(s) >= MIN_SENTENCE_CHARS]
    return out


class Idf:
    """Inverse document frequency over a collection of sentences."""

    def __init__(self, documents: Iterable[str]):
        self.df: Counter = Counter()
        self.n = 0
        for document in documents:
            self.df.update(words(document))
            self.n += 1

    def __call__(self, word: str) -> float:
        return math.log(1 + self.n / (1 + self.df[word]))

    def is_content(self, word: str) -> bool:
        return self.df[word] < MAX_DF_SHARE * self.n


def names_page(sentence: str, title: str) -> bool:
    """The sentence names the page: its title, or its title without a
    trailing disambiguator such as "(Marvel Cinematic Universe)"."""
    full = title.replace("_", " ")
    short = re.sub(r"\s*\([^)]*\)$", "", full)
    return any(len(t) >= 4 and re.search(rf"(?<!\w){re.escape(t)}(?!\w)", sentence) for t in {full, short})


def rank(title: str, section: str, context: str, neighbor_sentences: dict[str, list[str]],
         idf: Idf) -> list[tuple[str, str]]:
    """Qualifying (neighbor, sentence) pairs, most relevant first."""
    query = {w for w in words(" ".join((title.replace("_", " "), section, context))) if idf.is_content(w)}
    scored = []
    for neighbor, candidates in neighbor_sentences.items():
        for sentence in candidates:
            sentence_words = words(sentence)
            shared = sentence_words & query
            named = names_page(sentence, title)
            if named or len(shared) >= MIN_SHARED:
                score = sum(idf(w) for w in shared) / math.sqrt(len(sentence_words) + 1)
                scored.append((named, score, neighbor, sentence))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [(neighbor, sentence) for _, _, neighbor, sentence in scored]


def select(ranked: list[tuple[str, str]], budget: int = BUDGET_CHARS) -> list[tuple[str, str]]:
    """Take ranked sentences, each cut to MAX_SENTENCE_CHARS, while they fit
    in `budget`. Returns (neighbor, its sentences joined), in rank order."""
    chosen: dict[str, list[str]] = {}
    used = 0
    for neighbor, sentence in ranked:
        if len(sentence) > MAX_SENTENCE_CHARS:
            sentence = sentence[:MAX_SENTENCE_CHARS].rsplit(" ", 1)[0] + " …"
        if used + len(sentence) > budget:
            continue
        chosen.setdefault(neighbor, []).append(sentence)
        used += len(sentence)
    return [(neighbor, " ".join(s)) for neighbor, s in chosen.items()]
