"""Per-page daily activity and point-in-time features for Stage 1.

A page's labeled revisions (see `src.ingest.revert_detect`) are aggregated
into per-UTC-day counts on separate channels. Features for a prediction day
D read only days before D, i.e. what was knowable by the end of day D-1
(PLAN.md §2). The target for D is D's own `kept_edits`.

Channels:
- `edits`: non-bot, non-revert edits not yet reverted by the end of the day
  they were made. This is the input-side stand-in for "retained": the final
  `is_reverted` flag can't be used for inputs, because a revert can land up
  to 90 days later.
- `reverted_edits`: non-bot, non-revert edits reverted by the end of the
  same day (mostly vandalism).
- `reverts`: identity reverts by anyone, bots included. Knowable as soon as
  they're saved.
- `bot_edits`: bot edits that aren't reverts.
- `kept_edits`: non-bot, non-revert edits never reverted within the revert
  window. This depends on future revisions, so it's the *label* channel and
  must never feed an input feature.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import date
from itertools import accumulate
from typing import Iterable, Mapping

from src.common import is_bot_edit
from src.features.bursts import baseline_window, burst_zscore, is_burst

CHANNELS = ("edits", "reverted_edits", "reverts", "bot_edits", "kept_edits")

FEATURE_NAMES = (
    "page_age_days",
    "page_bytes",
    "days_since_last_edit",
    "edits_1d",
    "edits_7d",
    "edits_30d",
    "edits_365d",
    "editors_1d",
    "editors_7d",
    "reverted_edits_7d",
    "reverts_7d",
    "bot_edits_30d",
    "burst_z_1d",
    "is_burst_1d",
    "burst_days_30d",
    # The same weekday one and two weeks before D: weekly schedules (TV
    # episodes, fixtures, weekly shows) that the rolling windows smear out.
    "edits_lag7",
    "edits_lag14",
    "editors_lag7",
    "is_burst_lag7",
    "is_burst_lag14",
)


def day_ordinal(timestamp: str) -> int:
    """UTC day of a MediaWiki timestamp (`YYYY-MM-DDTHH:MM:SSZ`) as a date ordinal."""
    return date.fromisoformat(timestamp[:10]).toordinal()


# An editor making non-revert edits to more pages than this in one UTC day is
# running maintenance (AWB, category moves, template fixes) or mass page
# creation, not reacting to events one page at a time (PLAN.md §5). A round
# number chosen from the distribution, not tuned against the model.
MASS_EDITOR_PAGES_PER_DAY = 25


def is_bot(revision: dict) -> bool:
    """The revision's own bot flag when the source has one (the MediaWiki
    history dumps flag bots by user group or name), else the username
    heuristic in `src.common`."""
    flag = revision.get("is_bot")
    return is_bot_edit(revision["user_text"]) if flag is None else flag


def page_editor_days(revisions: Iterable[dict]) -> set[tuple[str, int]]:
    """(editor, day) pairs with a non-bot, non-revert edit to this page."""
    return {
        (r["user_text"], day_ordinal(r["timestamp"]))
        for r in revisions
        if r["user_text"] and not r["is_revert"] and not is_bot(r)
    }


def editor_day_page_counts(pages: Iterable[list[dict]]) -> Counter[tuple[str, int]]:
    """{(editor, day): distinct pages that non-bot editor made non-revert edits to that day}."""
    counts: Counter[tuple[str, int]] = Counter()
    for revisions in pages:
        counts.update(page_editor_days(revisions))
    return counts


def mass_editor_days(
    counts: Mapping[tuple[str, int], int], max_pages: int = MASS_EDITOR_PAGES_PER_DAY
) -> frozenset[tuple[str, int]]:
    """(editor, day) pairs above `max_pages`. Each is known by the end of its
    day, so filtering on them is point-in-time safe."""
    return frozenset(key for key, pages in counts.items() if pages > max_pages)


def _burst_days(counts: Mapping[int, int], first_day: int) -> set[int]:
    series = DailySeries(counts)
    squares = DailySeries({d: c * c for d, c in counts.items()})
    return {d for d, c in counts.items() if is_burst(c, _zscore(series, squares, first_day, d))}


def _zscore(series: DailySeries, squares: DailySeries, first_day: int, day: int) -> float:
    start, end = baseline_window(first_day, day)
    return burst_zscore(
        series.window_sum(day, day),
        series.window_sum(start, end),
        squares.window_sum(start, end),
        max(end - start + 1, 0),
    )


class DailySeries:
    """Sparse per-day counts (days as date ordinals) with O(log n) window sums."""

    def __init__(self, counts: Mapping[int, int]):
        self.days = sorted(counts)
        self._prefix = list(accumulate((counts[d] for d in self.days), initial=0))

    def window_sum(self, start: int, end: int) -> int:
        """Total over calendar days `start`..`end` inclusive (0 if empty)."""
        if end < start:
            return 0
        return self._prefix[bisect_right(self.days, end)] - self._prefix[bisect_left(self.days, start)]

    def last_day_before(self, day: int) -> int | None:
        i = bisect_left(self.days, day)
        return self.days[i - 1] if i > 0 else None


class PageActivity:
    """One page's daily activity channels, burst days and editors."""

    def __init__(
        self,
        revisions: list[dict],
        mass_editor_days: frozenset[tuple[str, int]] = frozenset(),
        created_day: int | None = None,
    ):
        """`revisions`: one page's revisions, labeled by `detect_page_reverts`.

        `mass_editor_days` (see `mass_editor_days()`) only affects
        `burst_days_excl_mass` and `editor_count_excl_mass`, a second burst
        definition that ignores those editors' `edits`. Every channel and
        feature is unchanged by it.

        `created_day` is the page's creation day, when the revisions don't
        reach back that far (the English Wikipedia history window). It only
        feeds `page_age_days`. Everything else still starts at the first
        revision in the data (`first_day`).
        """
        timestamps = {r["revision_id"]: r["timestamp"] for r in revisions}
        counts: dict[str, Counter[int]] = {name: Counter() for name in CHANNELS}
        editors: defaultdict[int, set[str]] = defaultdict(set)
        mass_edits: Counter[int] = Counter()
        mass_editors: defaultdict[int, set[str]] = defaultdict(set)

        for r in revisions:
            day = day_ordinal(r["timestamp"])
            if r["is_revert"]:
                counts["reverts"][day] += 1
            elif is_bot(r):
                counts["bot_edits"][day] += 1
            else:
                reverted_by = r["reverted_by_revision_id"]
                if reverted_by is not None and day_ordinal(timestamps[reverted_by]) <= day:
                    counts["reverted_edits"][day] += 1
                else:
                    counts["edits"][day] += 1
                    if r["user_text"]:
                        editors[day].add(r["user_text"])
                    if (r["user_text"], day) in mass_editor_days:
                        mass_edits[day] += 1
                        mass_editors[day].add(r["user_text"])
                if not r["is_reverted"]:
                    counts["kept_edits"][day] += 1

        self.page_id: int = revisions[0]["page_id"]
        self.first_day = min(day_ordinal(r["timestamp"]) for r in revisions)
        self.created_day = self.first_day if created_day is None else min(created_day, self.first_day)
        self.days = sorted(set().union(*counts.values()))  # days with any revision
        self.counts = counts
        self.channels = {name: DailySeries(c) for name, c in counts.items()}
        self.editors: dict[int, set[str]] = dict(editors)
        self._edit_squares = DailySeries({d: c * c for d, c in counts["edits"].items()})
        self.burst_days = {d for d, c in counts["edits"].items() if is_burst(c, self.burst_zscore(d))}
        self._bursts = DailySeries(dict.fromkeys(self.burst_days, 1))

        human = {d: c - mass_edits[d] for d, c in counts["edits"].items() if c > mass_edits[d]}
        self.burst_days_excl_mass = _burst_days(human, self.first_day) if mass_edits else self.burst_days
        self._mass_editors = dict(mass_editors)

        # The page's size at the end of each day with a revision: the last
        # revision that day with a known size, in revision order.
        size_on_day: dict[int, int] = {}
        for r in revisions:
            if r.get("byte_size") is not None:
                size_on_day[day_ordinal(r["timestamp"])] = r["byte_size"]
        self._size_days = sorted(size_on_day)
        self._sizes = [size_on_day[d] for d in self._size_days]

    def size_before(self, day: int) -> int | None:
        """The page's size in bytes at the end of `day - 1`, if known."""
        i = bisect_left(self._size_days, day)
        return self._sizes[i - 1] if i > 0 else None

    def burst_zscore(self, day: int) -> float:
        """z-score of `day`'s `edits` against its causal baseline."""
        return _zscore(self.channels["edits"], self._edit_squares, self.first_day, day)

    def editor_count(self, day: int) -> int:
        return len(self.editors.get(day, ()))

    def editor_count_excl_mass(self, day: int) -> int:
        return len(self.editors.get(day, set()) - self._mass_editors.get(day, set()))

    def features(self, day: int) -> dict:
        """Point-in-time features for prediction day `day` (reads days < `day` only)."""
        prev = day - 1
        edits = self.channels["edits"]
        last_edit = edits.last_day_before(day)
        return {
            "page_age_days": day - self.created_day,
            "page_bytes": self.size_before(day),
            "days_since_last_edit": day - last_edit if last_edit is not None else None,
            "edits_1d": edits.window_sum(prev, prev),
            "edits_7d": edits.window_sum(day - 7, prev),
            "edits_30d": edits.window_sum(day - 30, prev),
            "edits_365d": edits.window_sum(day - 365, prev),
            "editors_1d": self.editor_count(prev),
            "editors_7d": len(set().union(*(self.editors.get(d, ()) for d in range(day - 7, day)))),
            "reverted_edits_7d": self.channels["reverted_edits"].window_sum(day - 7, prev),
            "reverts_7d": self.channels["reverts"].window_sum(day - 7, prev),
            "bot_edits_30d": self.channels["bot_edits"].window_sum(day - 30, prev),
            "burst_z_1d": self.burst_zscore(prev),
            "is_burst_1d": prev in self.burst_days,
            "burst_days_30d": self._bursts.window_sum(day - 30, prev),
            "edits_lag7": edits.window_sum(day - 7, day - 7),
            "edits_lag14": edits.window_sum(day - 14, day - 14),
            "editors_lag7": self.editor_count(day - 7),
            "is_burst_lag7": day - 7 in self.burst_days,
            "is_burst_lag14": day - 14 in self.burst_days,
        }
