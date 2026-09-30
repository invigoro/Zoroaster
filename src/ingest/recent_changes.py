"""Mainspace edits from the MediaWiki API's recent changes, as revision records.

This is the daily input path for the deployed pipeline (PLAN.md §6 step 9).
Recent changes keep 30 days, which covers the gap until the next monthly
history dump. Records match `mediawiki_history.parse_line`'s fields, so the
feature code runs on either source:
- titles use underscores, as in the dumps;
- `is_anon` covers IP editors and temporary accounts (`temp`);
- `is_bot` is the edit's bot flag. The dumps flag the editor's groups (or
  name) instead, so the two can differ;
- `page_created` is known only for pages created in the window (`new`).
  Otherwise it's None, for the caller to fill in from its page state;
- `sha1` is converted to the dumps' base-36 form (the API gives hex), so
  revert detection can match content across the two sources;
- a page creation's `parent_id` is 0, as in the dumps;
- the dumps' own revert flags (`mwh_is_revert`, `mwh_is_reverted`) have no
  counterpart and are left out. `revert_detect` computes the ones used.

Hidden users and hashes (revision deletion) come through as None, as in the
dumps.

Checked against the dumps on the 12 hours both cover
(`scripts/check_recent_changes_parity.py`, PLAN.md §5): every field agrees
on 99.8%+ of edits, apart from titles and user names changed since. The
dumps also count 0.8% more revisions: ones that page moves and protections
create, which recent changes list as log entries instead.
"""

from __future__ import annotations

import string
import time
from typing import Iterator

import requests

from src.mediawiki_api import API_URL, PAUSE_SECONDS, api_get

RC_PROPS = "user|userid|flags|timestamp|title|ids|sizes|sha1"
_BASE36 = string.digits + string.ascii_lowercase


def sha1_base36(hex_sha1: str) -> str:
    """The API's hex SHA-1 in MediaWiki's storage form: base 36, zero-padded to 31 digits."""
    n, digits = int(hex_sha1, 16), []
    while n:
        n, r = divmod(n, 36)
        digits.append(_BASE36[r])
    return "".join(reversed(digits)).rjust(31, "0")


def to_record(change: dict) -> dict | None:
    """A revision record for a mainspace edit or page creation; None for anything else."""
    if change.get("ns") != 0 or change.get("type") not in ("edit", "new"):
        return None
    created = change["type"] == "new"
    return {
        "page_id": change["pageid"],
        "page_title": change["title"].replace(" ", "_"),
        "revision_id": change["revid"],
        "parent_id": change.get("old_revid"),
        "timestamp": change["timestamp"],
        "user_text": None if change.get("userhidden") else change.get("user"),
        "is_anon": bool(change.get("anon") or change.get("temp")),
        "is_bot": bool(change.get("bot")),
        "byte_size": change.get("newlen"),
        "sha1": None if change.get("sha1hidden") or not change.get("sha1") else sha1_base36(change["sha1"]),
        "page_created": change["timestamp"] if created else None,
    }


def recent_changes(
    start: str,
    end: str,
    lang: str = "en",
    session: requests.Session | None = None,
    pause: float = PAUSE_SECONDS,
    api_url: str | None = None,
) -> Iterator[dict]:
    """Records for mainspace edits with start <= timestamp < end (ISO 8601,
    e.g. `2026-09-29T00:00:00Z`), oldest first."""
    session = session or requests.Session()
    url = api_url or API_URL.format(lang=lang)
    params = {
        "action": "query", "list": "recentchanges", "rcnamespace": "0", "rctype": "edit|new", "rcprop": RC_PROPS,
        "rcdir": "newer", "rcstart": start, "rcend": end, "rclimit": "max",
        "format": "json", "formatversion": "2", "maxlag": "5",
    }
    continuation: dict = {}
    while True:
        payload = api_get(session, url, {**params, **continuation}, pause)
        for change in payload.get("query", {}).get("recentchanges", []):
            record = to_record(change)
            if record is not None and start <= record["timestamp"] < end:  # the API's rcend is inclusive
                yield record
        if "continue" not in payload:
            return
        continuation = payload["continue"]
        time.sleep(pause)
