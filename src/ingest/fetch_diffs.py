"""Fetch revision text and compute the inserted/removed diff for a single
retained (non-reverted) revision, via the MediaWiki `action=query` API.

This is the targeted, per-revision fetch path from the project plan: once a
page/revision sample has been chosen (`sampling.py`), this is what pulls
actual content for just those revisions, rather than downloading the
multi-terabyte full-text dump.

Used by `scripts/build_test_diffs.py`, which validated it live on 300
revisions of the Simple Wikipedia sample. Building the training corpus this
way means one (or two) HTTP round trips per sampled revision against real
Wikipedia infrastructure, so before any large run it should batch revision
ids (up to 50 per request) and diff at word level; see PLAN.md §6.
"""

from __future__ import annotations

import difflib
import time
from dataclasses import dataclass

import requests

from src.common import USER_AGENT

API_URL = "https://{lang}.wikipedia.org/w/api.php"

# Be polite to shared infrastructure between our own requests; the
# MediaWiki action API has no fixed anonymous rate limit but this keeps a
# large batch fetch from hammering it.
REQUEST_DELAY_SECONDS = 1.0


@dataclass
class RevisionDiff:
    page_id: int
    revision_id: int
    parent_id: int | None
    added_text: str
    removed_text: str


def fetch_revision_wikitext(
    revision_id: int, lang: str = "en", session: requests.Session | None = None
) -> str | None:
    """Fetch the wikitext content of a single revision by id.

    Returns None if the revision has no retrievable content (e.g. deleted
    or suppressed).
    """
    session = session or requests.Session()
    response = session.get(
        API_URL.format(lang=lang),
        params={
            "action": "query",
            "prop": "revisions",
            "revids": revision_id,
            "rvprop": "content",
            "rvslots": "main",
            "format": "json",
            "formatversion": "2",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    pages = payload.get("query", {}).get("pages", [])
    if not pages or "revisions" not in pages[0]:
        return None
    return pages[0]["revisions"][0]["slots"]["main"]["content"]


def compute_diff(old_text: str | None, new_text: str) -> tuple[str, str]:
    """Return (added_text, removed_text) between two wikitext revisions,
    diffed line-by-line. `old_text=None` (page-creation edit) treats every
    line of `new_text` as added.
    """
    old_lines = (old_text or "").splitlines()
    new_lines = new_text.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)

    added: list[str] = []
    removed: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert"):
            added.extend(new_lines[j1:j2])
        if tag in ("replace", "delete"):
            removed.extend(old_lines[i1:i2])
    return "\n".join(added), "\n".join(removed)


def fetch_revision_diff(
    page_id: int,
    revision_id: int,
    parent_id: int | None,
    lang: str = "en",
    session: requests.Session | None = None,
) -> RevisionDiff:
    """Fetch a revision and its parent, and return the added/removed text
    between them. Raises ValueError if the target revision's content can't
    be retrieved.
    """
    session = session or requests.Session()
    new_text = fetch_revision_wikitext(revision_id, lang=lang, session=session)
    if new_text is None:
        raise ValueError(f"Revision {revision_id} has no retrievable content")

    old_text = None
    if parent_id:
        time.sleep(REQUEST_DELAY_SECONDS)
        old_text = fetch_revision_wikitext(parent_id, lang=lang, session=session)

    added_text, removed_text = compute_diff(old_text, new_text)
    return RevisionDiff(
        page_id=page_id,
        revision_id=revision_id,
        parent_id=parent_id,
        added_text=added_text,
        removed_text=removed_text,
    )
