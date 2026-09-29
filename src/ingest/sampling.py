"""Stratified page sampling for the training corpus.

Popularity-based stratification (via pageviews dumps) is deferred until
we're ready to pull that data — for now, pages are stratified only on
signals already derivable from the revision history: edit frequency, bot
edit share, and historical burst activity (see `src.features.bursts`).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

import numpy as np

from src.common import is_bot_edit


def hash_sampled(page_id: int, rate: float) -> bool:
    """Deterministic page sample: Knuth multiplicative hashing of the page id
    into 32 bits. The high bits decide, so the sample is independent of
    `page_id % n` bucketing."""
    return (page_id * 2654435761) % 2**32 < rate * 2**32


def hash_sample_mask(size: int, rate: float) -> np.ndarray:
    """`hash_sampled` for every page id below `size`, as a boolean array."""
    ids = np.arange(size, dtype=np.int64)
    return ((ids * 2654435761) & 0xFFFFFFFF) < rate * 2**32


def _edit_frequency_bucket(edit_count: int) -> str:
    if edit_count < 5:
        return "low"
    if edit_count < 25:
        return "medium"
    return "high"


def summarize_pages(revisions: Iterable[dict], burst_page_ids: set[int]) -> dict[int, dict]:
    """Build one summary row per page_id from its retained revisions."""
    counts: dict[int, int] = defaultdict(int)
    bot_counts: dict[int, int] = defaultdict(int)
    anon_counts: dict[int, int] = defaultdict(int)
    titles: dict[int, str] = {}

    for revision in revisions:
        page_id = revision["page_id"]
        counts[page_id] += 1
        titles[page_id] = revision["page_title"]
        if is_bot_edit(revision["user_text"]):
            bot_counts[page_id] += 1
        if revision["is_anon"]:
            anon_counts[page_id] += 1

    return {
        page_id: {
            "page_id": page_id,
            "page_title": titles[page_id],
            "edit_count": total,
            "bot_edit_fraction": bot_counts[page_id] / total,
            "anon_edit_fraction": anon_counts[page_id] / total,
            "had_burst": page_id in burst_page_ids,
        }
        for page_id, total in counts.items()
    }


def stratify(summaries: dict[int, dict]) -> dict[tuple[str, bool], list[int]]:
    strata: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for page_id, summary in summaries.items():
        key = (_edit_frequency_bucket(summary["edit_count"]), summary["had_burst"])
        strata[key].append(page_id)
    return strata


def sample_pages(
    strata: dict[tuple[str, bool], list[int]], per_stratum: int, seed: int
) -> dict[str, list[int]]:
    """Randomly sample up to `per_stratum` page ids from each stratum."""
    rng = random.Random(seed)
    sample: dict[str, list[int]] = {}
    for key, page_ids in strata.items():
        chosen = (
            page_ids if len(page_ids) <= per_stratum else rng.sample(page_ids, per_stratum)
        )
        sample["|".join(str(part) for part in key)] = sorted(chosen)
    return sample
