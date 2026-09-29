"""Identity-revert detection over a page's revision history.

A revision is an identity *revert* if its content (`sha1`) matches an
earlier revision of the page other than its immediate predecessor, i.e. it
restores an old content state. It undoes the revisions strictly between the
*most recent* earlier revision with that content and itself; those are
*reverted*. This is the standard identity-revert heuristic used in prior
Wikipedia edit-dynamics research, and it only needs the metadata already
present in the stub dumps (no article text required).

A revision only counts as reverted if the revert lands within 15 revisions
*or* 90 days of it, whichever is tighter: a pure revision-count window
undercounts reverts on slow-moving pages (the "next 15 revisions" can span
years), while a pure time window overcounts on fast-moving pages. The revert
itself is flagged whatever the window, since either way it adds no new
content. Neither reverts nor reverted revisions are training targets.

The two flags differ in *when* they're knowable. `is_revert` depends only on
earlier revisions, so it's known the moment the revision is saved.
`is_reverted` depends on revisions up to 90 days later, so point-in-time
features must not filter on it directly; `reverted_by_revision_id` says
which revision undid it, and therefore when that became known.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from itertools import groupby
from typing import Iterable, Iterator

REVISION_WINDOW = 15
TIME_WINDOW = timedelta(days=90)


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def detect_reverts(revisions: Iterable[dict]) -> Iterator[dict]:
    """Yield each input revision dict with `is_reverted`,
    `reverted_by_revision_id` and `is_revert` added.

    Streams one page at a time, so memory is bounded by the largest single
    page history. Requires each page's revisions to be contiguous in the
    input (order within a page doesn't matter), which holds for MediaWiki
    XML dumps and for the Parquet files this pipeline writes.
    """
    for _, page_revisions in groupby(revisions, key=lambda r: r["page_id"]):
        yield from detect_page_reverts(list(page_revisions))


def detect_page_reverts(page_revisions: list[dict]) -> list[dict]:
    """Label one page's revisions; returns them sorted by `revision_id`."""
    page_revisions = sorted(page_revisions, key=lambda r: r["revision_id"])
    timestamps = [_parse_timestamp(r["timestamp"]) for r in page_revisions]
    reverted_by: list[int | None] = [None] * len(page_revisions)
    is_revert = [False] * len(page_revisions)

    # sha1 -> most recent index at which that content state was seen so far.
    latest_index_for_sha1: dict[str, int] = {}

    for j, revision in enumerate(page_revisions):
        sha1 = revision["sha1"]
        if sha1 is None:
            continue
        restored = latest_index_for_sha1.get(sha1)
        latest_index_for_sha1[sha1] = j
        if restored is None or restored == j - 1:
            continue  # new content, or a null revision duplicating its parent
        is_revert[j] = True
        # Only the last REVISION_WINDOW revisions before j can be in window.
        for i in range(max(restored + 1, j - REVISION_WINDOW), j):
            if reverted_by[i] is None and timestamps[j] - timestamps[i] <= TIME_WINDOW:
                reverted_by[i] = revision["revision_id"]

    return [
        {
            **revision,
            "is_reverted": reverted_by[i] is not None,
            "reverted_by_revision_id": reverted_by[i],
            "is_revert": is_revert[i],
        }
        for i, revision in enumerate(page_revisions)
    ]
