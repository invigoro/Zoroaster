"""Rank version 2's forecasts by the model's own probabilities (PLAN.md §6 step 10).

Greedy decoding gives the model's single most likely header. That header
names few sections and never names the rare kinds of change. Ranking reads
the model's probabilities instead:
- **Sections:** each candidate is scored by the probability that the
  forecast names it first, which in training is the section that changed
  most. The candidates are the lead and every heading on the page at the
  end of D-1.
  - The top MAX_SECTIONS are named, since "yesterday again" names up to
    that many.
  - Only sections already on the page can be named.
- **Kinds:** headers list kinds in KINDS order, so a kinds line can only be
  one of 2^9 sets (`kinds_lines`).
  - Scoring every set gives each kind's exact probability of being listed,
    given the sections line.
  - A kind is forecast when its probability passes a threshold chosen on
    validation (`choose_thresholds`).
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from src.forecast.changes import KINDS, LEAD
from src.forecast.metrics import MAX_SECTIONS
from src.forecast.prompts import NONE

GRID = tuple(round(t, 2) for t in np.arange(0.05, 0.96, 0.05))


def section_candidates(example: dict) -> list[str]:
    """The sections a forecast can name: the lead, then the page's headings at the end of D-1."""
    return list(dict.fromkeys([LEAD, *example["heading_titles"]]))


def section_continuations(candidates: list[str]) -> list[str]:
    """After "Sections:", each candidate named first: then another section, or the line's end."""
    return [f" {s}{end}" for s in candidates for end in (";", "\n")]


def first_section_probs(logprobs: list[float], candidates: list[str]) -> list[float]:
    """Each candidate's probability of being named first, from `section_continuations`' log-probabilities."""
    return [math.exp(logprobs[2 * i]) + math.exp(logprobs[2 * i + 1]) for i in range(len(candidates))]


def rank_sections(candidates: list[str], probs: list[float], k: int = MAX_SECTIONS) -> list[str]:
    """The `k` most probable candidates, most probable first; ties keep page order."""
    order = sorted(range(len(candidates)), key=lambda i: -probs[i])
    return [candidates[i] for i in order[:k]]


def kinds_lines() -> list[tuple[str, ...]]:
    """Every set of kinds a header can list, in KINDS order, starting with the empty set."""
    return [combo for n in range(len(KINDS) + 1) for combo in combinations(KINDS, n)]


def kinds_continuation(kinds: tuple[str, ...]) -> str:
    """The text after "Kinds:" for a set of kinds, as `header` writes it."""
    return f" {', '.join(kinds) or NONE}\n"


def kind_marginals(lines: list[tuple[str, ...]], logprobs: list[float]) -> tuple[dict[str, float], float]:
    """Each kind's probability of being listed, normalized over the valid
    lines, and the total probability the valid lines hold."""
    probs = np.exp(np.array(logprobs, dtype=float))
    total = float(probs.sum())
    member = np.array([[k in line for k in KINDS] for line in lines], dtype=float)
    return dict(zip(KINDS, (probs @ member / total).tolist())), total


def build_trie(continuations: list[list[int]]) -> tuple[list[int], list[int], list[int], list[list[int]]]:
    """Merge token sequences into a trie, so shared prefixes are scored once.

    Returns each node's token, its parent (-1 under the root), its depth,
    and each sequence's path of nodes. Parents always come before their
    children."""
    tokens: list[int] = []
    parents: list[int] = []
    depths: list[int] = []
    paths: list[list[int]] = []
    index: dict[tuple[int, int], int] = {}
    for sequence in continuations:
        node, path = -1, []
        for token in sequence:
            if (node, token) not in index:
                index[(node, token)] = len(tokens)
                tokens.append(token)
                parents.append(node)
                depths.append(0 if node < 0 else depths[node] + 1)
            node = index[(node, token)]
            path.append(node)
        paths.append(path)
    return tokens, parents, depths, paths


def kinds_forecast(marginals: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    return [k for k in KINDS if marginals[k] >= thresholds[k]]


def _mean_jaccard(said: np.ndarray, did: np.ndarray) -> float:
    union = (said | did).sum(axis=1)
    both = (said & did).sum(axis=1)
    return float(np.where(union > 0, both / np.maximum(union, 1), 1.0).mean())  # nothing said or done counts as 1


def choose_thresholds(marginals: list[dict[str, float]], actual: list[list[str]], grid: tuple[float, ...] = GRID,
                      rounds: int = 3) -> dict[str, float]:
    """Per-kind thresholds that maximize the mean kinds Jaccard over the
    examples. Coordinate ascent from 0.5: a threshold only moves when that
    strictly improves the mean."""
    probs = np.array([[m[k] for k in KINDS] for m in marginals])
    did = np.array([[k in a for k in KINDS] for a in actual])
    thresholds = np.full(len(KINDS), 0.5)
    best = _mean_jaccard(probs >= thresholds, did)
    for _ in range(rounds):
        for j in range(len(KINDS)):
            for t in grid:
                trial = thresholds.copy()
                trial[j] = t
                score = _mean_jaccard(probs >= trial, did)
                if score > best + 1e-12:
                    best, thresholds = score, trial
    return dict(zip(KINDS, thresholds.tolist()))
