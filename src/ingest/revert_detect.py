"""Identity-revert detection over a page's revision history.

A revision is treated as reverted if a later revision restores the page to
a content state (identified by `sha1`) that already existed *before* it —
i.e. someone undid it. This is the standard identity-revert heuristic used
in prior Wikipedia edit-dynamics research, and it only needs the metadata
already present in the stub dumps (no article text required).

The lookahead window is capped at 15 revisions *or* 90 days, whichever is
tighter: a pure revision-count window undercounts reverts on slow-moving
pages (the "next 15 revisions" can span years), while a pure time window
overcounts on fast-moving pages.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable, Iterator

REVISION_WINDOW = 15
TIME_WINDOW = timedelta(days=90)


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def detect_reverts(revisions: Iterable[dict]) -> Iterator[dict]:
    """Yield each input revision dict with an added `is_reverted` bool.

    `revisions` need not be pre-sorted; they are grouped by `page_id` and
    sorted by `revision_id` internally. All revisions for a given page are
    buffered in memory to run the per-page window check, which is fine at
    the scale of a single test dump but would need a different approach
    (e.g. bounding by concurrently-open pages in a single sorted pass) at
    full English-Wikipedia scale.
    """
    by_page: dict[int, list[dict]] = defaultdict(list)
    for revision in revisions:
        by_page[revision["page_id"]].append(revision)

    for page_revisions in by_page.values():
        yield from _detect_reverts_for_page(page_revisions)


def _detect_reverts_for_page(page_revisions: list[dict]) -> Iterator[dict]:
    page_revisions.sort(key=lambda r: r["revision_id"])
    timestamps = [_parse_timestamp(r["timestamp"]) for r in page_revisions]

    # sha1 -> earliest index at which that content state was seen so far.
    earliest_index_for_sha1: dict[str, int] = {}

    for i, revision in enumerate(page_revisions):
        is_reverted = False
        deadline = timestamps[i] + TIME_WINDOW
        window_end = min(i + REVISION_WINDOW, len(page_revisions) - 1)

        for j in range(i + 1, window_end + 1):
            if timestamps[j] > deadline:
                break
            candidate_sha1 = page_revisions[j]["sha1"]
            if candidate_sha1 is None:
                continue
            prior_index = earliest_index_for_sha1.get(candidate_sha1)
            if prior_index is not None and prior_index < i:
                is_reverted = True
                break

        sha1 = revision["sha1"]
        if sha1 is not None and sha1 not in earliest_index_for_sha1:
            earliest_index_for_sha1[sha1] = i

        yield {**revision, "is_reverted": is_reverted}
