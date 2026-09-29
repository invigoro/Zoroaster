"""Causal burst scoring and cross-page co-burst counts.

A page is *bursting* on day d when its edit count that day is far above its
own recent baseline: the BASELINE_DAYS calendar days before d, zero-edit
days included, clipped to days since the page's first revision. Every input
is from day d or earlier, so the flag is known by the end of day d and is
safe as a feature for any later day. This replaces the original
whole-history baseline, which let future days leak into past z-scores.

The z-score's denominator is floored at MIN_BASELINE_STD. A quiet page's
baseline is flat (often all zeros), so without a floor its first few edits
after a lull, or a brand-new page's first day, would be infinitely
surprising. With the floor, a quiet page needs about 3 edits in a day to
burst, and a busy first day counts as a burst. Under the old
active-days-only population z-score, a page with fewer than 10 active days
could never burst (Samuelson's inequality caps z at sqrt(n - 1)).

"Co-burst" (how many *other* pages are bursting the same day) is the
endogenous stand-in for an external news signal (see the project plan).
Callers should score non-bot, non-revert edits: bot maintenance runs, AWB
mass edits and vandalism waves otherwise produce the same statistical
shape as a real-world-event-driven edit wave (see PLAN.md §5).
"""

from __future__ import annotations

from collections import Counter
from math import sqrt

BURST_Z_THRESHOLD = 3.0
MIN_BURST_DAY_COUNT = 2
BASELINE_DAYS = 90
MIN_BASELINE_STD = 1.0


def baseline_window(first_day: int, day: int) -> tuple[int, int]:
    """Inclusive (start, end) day ordinals of `day`'s baseline; empty (end <
    start) on the page's first day."""
    return max(first_day, day - BASELINE_DAYS), day - 1


def burst_zscore(count: int, baseline_sum: int, baseline_sum_sq: int, baseline_days: int) -> float:
    if baseline_days > 0:
        mean = baseline_sum / baseline_days
        std = sqrt(max(baseline_sum_sq / baseline_days - mean * mean, 0.0))
    else:
        mean = std = 0.0
    return (count - mean) / max(std, MIN_BASELINE_STD)


def is_burst(count: int, zscore: float) -> bool:
    return count >= MIN_BURST_DAY_COUNT and zscore >= BURST_Z_THRESHOLD


class CoBurstCounter:
    """Per-day counts of bursting pages, accumulated one page at a time.

    `multi_editor` counts only bursts with at least two distinct editors:
    on the Simple Wikipedia test corpus, top co-burst days were dominated by
    unrelated single-editor sessions that happened to share a calendar day,
    not by several editors reacting to one trigger (PLAN.md §5).
    """

    def __init__(self) -> None:
        self.pages: Counter[int] = Counter()
        self.multi_editor: Counter[int] = Counter()

    def add(self, day: int, editors: int) -> None:
        self.pages[day] += 1
        if editors >= 2:
            self.multi_editor[day] += 1

    def others(self, day: int, self_bursting: bool, self_editors: int) -> tuple[int, int]:
        """(pages, multi-editor pages) bursting on `day`, excluding the caller's own page."""
        return (
            self.pages[day] - self_bursting,
            self.multi_editor[day] - (self_bursting and self_editors >= 2),
        )
