"""Burst / co-burst activity features from retained (non-reverted) revisions.

These are the endogenous "something is happening" signals used in place of
an external news feed (see the project plan, deferred exogenous-enrichment
decision): a page's own edit-rate spike vs. its own history, and how many
*other* pages are spiking on the same day.

Callers should pass revisions that are both retained (non-reverted) and
non-bot: a coordinated vandalism wave or a routine bot maintenance run
(interwiki-link bots, AWB mass page creation, etc.) produces the same
statistical shape as a real-world-event-driven edit wave but isn't the
signal this feature is meant to capture. On the Simple Wikipedia test
corpus, the single largest co-burst day was in fact bot/AWB activity, not
an event, confirming this needs to be filtered upstream rather than
guarded against here.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Iterable, TypedDict

BURST_Z_THRESHOLD = 3.0
MIN_BURST_DAY_COUNT = 2


class DayActivity(TypedDict):
    page_id: int
    date: str
    edit_count: int
    baseline_mean: float
    baseline_std: float
    zscore: float
    is_burst: bool


def _date(timestamp: str) -> str:
    return timestamp[:10]


def daily_edit_counts(revisions: Iterable[dict]) -> dict[tuple[int, str], int]:
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for revision in revisions:
        counts[(revision["page_id"], _date(revision["timestamp"]))] += 1
    return counts


def compute_page_bursts(revisions: Iterable[dict]) -> list[DayActivity]:
    """Flag burst days per page using a whole-history baseline.

    This uses each page's full retained-edit history as its own baseline
    (mean/stddev of daily edit counts), which is a simplification of a
    proper trailing-window baseline — fine for validating the pipeline on a
    small test corpus, but worth revisiting (e.g. trailing N-day windows,
    excluding the day itself from its own baseline) before running at full
    scale.
    """
    counts = daily_edit_counts(revisions)
    by_page: dict[int, dict[str, int]] = defaultdict(dict)
    for (page_id, date), count in counts.items():
        by_page[page_id][date] = count

    results: list[DayActivity] = []
    for page_id, day_counts in by_page.items():
        values = list(day_counts.values())
        baseline_mean = mean(values)
        baseline_std = pstdev(values)
        for date, count in sorted(day_counts.items()):
            zscore = (count - baseline_mean) / baseline_std if baseline_std > 0 else 0.0
            is_burst = zscore >= BURST_Z_THRESHOLD and count >= MIN_BURST_DAY_COUNT
            results.append(
                {
                    "page_id": page_id,
                    "date": date,
                    "edit_count": count,
                    "baseline_mean": baseline_mean,
                    "baseline_std": baseline_std,
                    "zscore": zscore,
                    "is_burst": is_burst,
                }
            )
    return results


def compute_co_burst_counts(page_bursts: Iterable[DayActivity]) -> dict[str, int]:
    """Count how many distinct pages are bursting on each date."""
    counts: dict[str, int] = defaultdict(int)
    for activity in page_bursts:
        if activity["is_burst"]:
            counts[activity["date"]] += 1
    return counts


def attach_co_burst(page_bursts: list[DayActivity]) -> list[dict]:
    """Add `co_burst_count`: how many *other* pages are bursting the same day."""
    co_burst_by_date = compute_co_burst_counts(page_bursts)
    enriched = []
    for activity in page_bursts:
        co_burst = co_burst_by_date.get(activity["date"], 0)
        if activity["is_burst"]:
            co_burst -= 1
        enriched.append({**activity, "co_burst_count": co_burst})
    return enriched
