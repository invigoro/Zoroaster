"""Stream-parse a MediaWiki stub-meta-history dump into revision records.

Stub dumps (`stub-meta-history.xml.gz`) contain full revision *metadata*
(id, timestamp, contributor, comment, byte size, sha1) for every revision of
every page, but no article text. That makes them cheap to process at scale
and exactly what's needed for identity-revert detection (see
`revert_detect.py`) before any full text is fetched.

Uses `mwxml`, which streams the (possibly compressed) dump page-by-page and
revision-by-revision without ever materializing the whole file in memory.
"""

from __future__ import annotations

from typing import Iterator, TypedDict

import mwxml
from mwtypes.files import reader

MAINSPACE = 0


class RevisionRecord(TypedDict):
    page_id: int
    page_title: str
    namespace: int
    revision_id: int
    parent_id: int | None
    timestamp: str
    user_text: str | None
    user_id: int | None
    is_anon: bool
    is_minor: bool
    comment: str | None
    byte_size: int | None
    sha1: str | None


def iter_mainspace_revisions(dump_path: str) -> Iterator[RevisionRecord]:
    """Yield one `RevisionRecord` per mainspace (namespace 0) revision.

    `dump_path` may point at a `.xml`, `.xml.gz`, `.xml.bz2`, or `.xml.7z`
    file — `mwtypes.files.reader` picks the right decompressor from the
    extension.
    """
    with reader(dump_path) as f:
        dump = mwxml.Dump.from_file(f)
        for page in dump:
            if page.namespace != MAINSPACE:
                continue
            for revision in page:
                yield _to_record(page, revision)


def _to_record(page, revision) -> RevisionRecord:
    user = revision.user
    return {
        "page_id": page.id,
        "page_title": page.title,
        "namespace": page.namespace,
        "revision_id": revision.id,
        "parent_id": revision.parent_id,
        "timestamp": str(revision.timestamp),
        "user_text": user.text if user is not None else None,
        "user_id": user.id if user is not None else None,
        "is_anon": user is not None and user.id is None,
        "is_minor": bool(revision.minor),
        "comment": revision.comment,
        "byte_size": revision.bytes,
        "sha1": revision.sha1,
    }
