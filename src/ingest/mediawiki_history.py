"""Stream mainspace revisions out of a MediaWiki history TSV dump.

Format reference:
https://wikitech.wikimedia.org/wiki/Data_Platform/Data_Lake/Edits/MediaWiki_history_dumps

- One bz2-compressed TSV per month of events, with no header row.
- 78 columns, listed in COLUMNS; checked against the 2026-08 snapshot,
  where every line has exactly 78 fields.
- Booleans are `true`/`false`, arrays are comma-separated, nulls are empty
  fields, and timestamps look like `2026-09-01 00:00:00.0`.
- Tabs and newlines inside text fields are backslash-escaped, so splitting a
  line on tabs is safe.

`iter_revisions` keeps revision-create events made while the page was in
mainspace. It drops revisions of pages that were later deleted, because the
Simple Wikipedia stub dump likewise holds only surviving pages. Records have
the fields the pipeline uses from `stub_stream.RevisionRecord`, plus:
- `is_bot`: the editor was flagged a bot at the time, by name or by user
  group. This replaces the name-only heuristic.
- `page_created`: the page's creation time, for page age. The history
  window doesn't reach back to most pages' first revision.
- `mwh_is_revert` / `mwh_is_reverted`: Wikimedia's own identity-revert
  flags, kept to cross-check `revert_detect`, which computes the flags that
  are actually used.
"""

from __future__ import annotations

import bz2
from pathlib import Path
from typing import Iterator

COLUMNS = (
    "wiki_db", "event_log_id", "event_entity", "event_type", "event_timestamp", "event_comment",
    "event_user_id", "event_user_central_id", "event_user_text_historical", "event_user_text",
    "event_user_blocks_historical", "event_user_blocks", "event_user_groups_historical", "event_user_groups",
    "event_user_is_bot_by_historical", "event_user_is_bot_by", "event_user_is_created_by_self",
    "event_user_is_created_by_system", "event_user_is_created_by_peer", "event_user_is_anonymous",
    "event_user_is_temporary", "event_user_is_permanent", "event_user_is_cross_wiki",
    "event_user_registration_timestamp", "event_user_creation_timestamp", "event_user_first_edit_timestamp",
    "event_user_revision_count", "event_user_seconds_since_previous_revision",
    "page_id", "page_title_historical", "page_title", "page_namespace_historical",
    "page_namespace_is_content_historical", "page_namespace", "page_namespace_is_content", "page_is_redirect",
    "page_is_deleted", "page_creation_timestamp", "page_first_edit_timestamp", "page_revision_count",
    "page_seconds_since_previous_revision",
    "user_id", "user_central_id", "user_text_historical", "user_text", "user_blocks_historical", "user_blocks",
    "user_groups_historical", "user_groups", "user_is_bot_by_historical", "user_is_bot_by",
    "user_is_created_by_self", "user_is_created_by_system", "user_is_created_by_peer", "user_is_anonymous",
    "user_is_temporary", "user_is_permanent", "user_registration_timestamp", "user_creation_timestamp",
    "user_first_edit_timestamp",
    "revision_id", "revision_parent_id", "revision_minor_edit", "revision_deleted_parts",
    "revision_deleted_parts_are_suppressed", "revision_text_bytes", "revision_text_bytes_diff",
    "revision_text_sha1", "revision_content_model", "revision_content_format",
    "revision_is_deleted_by_page_deletion", "revision_deleted_by_page_deletion_timestamp",
    "revision_is_identity_reverted", "revision_first_identity_reverting_revision_id",
    "revision_seconds_to_identity_revert", "revision_is_identity_revert", "revision_is_from_before_page_creation",
    "revision_tags",
)
_I = {name: i for i, name in enumerate(COLUMNS)}
_ENTITY, _TYPE, _NAMESPACE = _I["event_entity"], _I["event_type"], _I["page_namespace_historical"]
_DELETED = _I["revision_is_deleted_by_page_deletion"]


def _timestamp(value: str) -> str:
    """`2026-09-01 00:00:00.0` -> `2026-09-01T00:00:00Z`, the stub dumps' format."""
    return f"{value[:10]}T{value[11:19]}Z"


def _int(value: str) -> int | None:
    return int(value) if value else None


def parse_line(line: str) -> dict | None:
    """A mainspace revision record, or None for any other event."""
    f = line.rstrip("\n").split("\t")
    if len(f) != len(COLUMNS):
        raise ValueError(f"expected {len(COLUMNS)} fields, got {len(f)}: {line[:200]!r}")
    if f[_ENTITY] != "revision" or f[_TYPE] != "create" or f[_NAMESPACE] != "0" or f[_DELETED] == "true":
        return None
    created = f[_I["page_creation_timestamp"]]
    return {
        "page_id": int(f[_I["page_id"]]),
        "page_title": f[_I["page_title"]] or None,
        "revision_id": int(f[_I["revision_id"]]),
        "parent_id": _int(f[_I["revision_parent_id"]]),
        "timestamp": _timestamp(f[_I["event_timestamp"]]),
        "user_text": f[_I["event_user_text_historical"]] or None,
        # Logged-out editing moved from IP addresses to temporary accounts
        # (`~2026-47395-76`) in 2025; both count as anonymous.
        "is_anon": f[_I["event_user_is_anonymous"]] == "true" or f[_I["event_user_is_temporary"]] == "true",
        "is_bot": f[_I["event_user_is_bot_by_historical"]] != "",
        "byte_size": _int(f[_I["revision_text_bytes"]]),
        "sha1": f[_I["revision_text_sha1"]] or None,
        "page_created": _timestamp(created) if created else None,
        "mwh_is_revert": f[_I["revision_is_identity_revert"]] == "true",
        "mwh_is_reverted": f[_I["revision_is_identity_reverted"]] == "true",
    }


def iter_revisions(path: Path) -> Iterator[dict]:
    with bz2.open(path, "rt", encoding="utf-8", errors="replace", newline="\n") as f:
        for line in f:
            record = parse_line(line)
            if record is not None:
                yield record
