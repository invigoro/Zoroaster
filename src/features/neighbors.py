"""Link-neighbor burst features: were pages related to this one bursting?

The site-wide co-burst count is the same for every page on a given day, so
it can't say *which* pages an event will touch (PLAN.md §5). These features
make the cross-page signal page-specific. For page P and prediction day D,
they count P's link neighbors that were bursting:
- "in" neighbors are pages that link *to* P (a bursting page linking here);
- "out" neighbors are pages P links to.

Counts cover day D-1 and the 7 days before D. There's also a variant
counting only bursts with at least two distinct editors, and the fraction
of the page's neighbors that burst on D-1: hub pages like *United States*
would otherwise dominate the raw counts.

The bursts are point-in-time, since each flag is known by the end of its
day. The links are not: the graph is a single snapshot from the dump date.
A link added after D, possibly because of the very event being predicted,
still counts toward D's features. So results are optimistic, more so for
days long before the snapshot.

Everything is vectorized over numpy arrays. Days are days since the Unix
epoch, i.e. `date32` values cast to int.
"""

from __future__ import annotations

import numpy as np

DEGREE_FEATURES = ("in_links", "out_links")
NEIGHBOR_BURST_FEATURES = (
    "in_nbrs_bursting_1d",
    "out_nbrs_bursting_1d",
    "in_nbrs_bursting_7d",
    "out_nbrs_bursting_7d",
    "in_nbrs_multi_editor_bursting_1d",
    "out_nbrs_multi_editor_bursting_1d",
    "in_frac_bursting_1d",
    "out_frac_bursting_1d",
)
NEIGHBOR_FEATURES = DEGREE_FEATURES + NEIGHBOR_BURST_FEATURES

_DAY_BITS = 16  # days since the epoch fit in 16 bits until 2149


def _csr(rows: np.ndarray, cols: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(rows, kind="stable")
    indptr = np.zeros(size + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=size), out=indptr[1:])
    return indptr, cols[order]


def _expand(indptr: np.ndarray, indices: np.ndarray, pages: np.ndarray, days: np.ndarray):
    """(neighbor, day) for every CSR neighbor of each (page, day)."""
    starts = indptr[pages]
    lengths = indptr[pages + 1] - starts
    offsets = np.repeat(starts - np.cumsum(lengths) + lengths, lengths) + np.arange(lengths.sum())
    return indices[offsets], np.repeat(days, lengths)


class LinkGraph:
    """Page-id adjacency in both directions, in CSR form."""

    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = np.asarray(source, dtype=np.int64)
        target = np.asarray(target, dtype=np.int64)
        self.size = int(max(source.max(initial=0), target.max(initial=0))) + 1
        self.out_degree = np.bincount(source, minlength=self.size)
        self.in_degree = np.bincount(target, minlength=self.size)
        self.links_to = _csr(source, target, self.size)  # page -> pages it links to
        self.linked_from = _csr(target, source, self.size)  # page -> pages linking to it

    def degrees(self, pages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(in-degree, out-degree) per page; 0 for pages not in the graph."""
        known = pages < self.size
        clipped = np.where(known, pages, 0)
        return np.where(known, self.in_degree[clipped], 0), np.where(known, self.out_degree[clipped], 0)


class _DayCounts:
    """Sparse (page, day) -> count, with vectorized calendar-window sums."""

    def __init__(self, pages: np.ndarray, days: np.ndarray):
        keys = (pages.astype(np.int64) << _DAY_BITS) | days.astype(np.int64)
        self._keys, counts = np.unique(keys, return_counts=True)
        self._cum = np.concatenate([[0], np.cumsum(counts)])

    def window_sum(self, pages: np.ndarray, start_days: np.ndarray, end_days: np.ndarray) -> np.ndarray:
        """Total over days `start_days`..`end_days` inclusive, per row."""
        base = pages.astype(np.int64) << _DAY_BITS
        lo = np.searchsorted(self._keys, base | start_days, side="left")
        hi = np.searchsorted(self._keys, base | end_days, side="right")
        return self._cum[hi] - self._cum[lo]


class NeighborBursts:
    """Neighbor-burst features for any (page, prediction day) rows."""

    def __init__(self, graph: LinkGraph, burst_pages: np.ndarray, burst_days: np.ndarray, burst_editors: np.ndarray):
        """One entry per burst page-day: the page, the day, and its distinct editors."""
        self.graph = graph
        known = burst_pages < graph.size
        pages, days, multi = burst_pages[known], burst_days[known], burst_editors[known] >= 2
        # A bursting page counts toward the "in" neighbors of every page it
        # links to, and the "out" neighbors of every page linking to it.
        self._in = _DayCounts(*_expand(*graph.links_to, pages, days))
        self._out = _DayCounts(*_expand(*graph.linked_from, pages, days))
        self._in_multi = _DayCounts(*_expand(*graph.links_to, pages[multi], days[multi]))
        self._out_multi = _DayCounts(*_expand(*graph.linked_from, pages[multi], days[multi]))

    def features(self, pages: np.ndarray, days: np.ndarray, suffix: str = "") -> dict[str, np.ndarray]:
        """Features for prediction days `days`, reading bursts from earlier days
        only. `suffix` is appended to the burst features' names (not the
        degrees), to tell burst definitions apart."""
        pages = np.asarray(pages, dtype=np.int64)
        prev = np.asarray(days, dtype=np.int64) - 1
        week_start = prev - 6
        in_links, out_links = self.graph.degrees(pages)
        in_1d = self._in.window_sum(pages, prev, prev)
        out_1d = self._out.window_sum(pages, prev, prev)
        bursts = {
            "in_nbrs_bursting_1d": in_1d,
            "out_nbrs_bursting_1d": out_1d,
            "in_nbrs_bursting_7d": self._in.window_sum(pages, week_start, prev),
            "out_nbrs_bursting_7d": self._out.window_sum(pages, week_start, prev),
            "in_nbrs_multi_editor_bursting_1d": self._in_multi.window_sum(pages, prev, prev),
            "out_nbrs_multi_editor_bursting_1d": self._out_multi.window_sum(pages, prev, prev),
            "in_frac_bursting_1d": in_1d / np.maximum(in_links, 1),
            "out_frac_bursting_1d": out_1d / np.maximum(out_links, 1),
        }
        return {"in_links": in_links, "out_links": out_links} | {
            name + suffix: bursts[name] for name in NEIGHBOR_BURST_FEATURES
        }
