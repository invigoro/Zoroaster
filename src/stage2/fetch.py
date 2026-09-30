"""Fetch many revisions' wikitext from the MediaWiki action API, 50 at a time.

Requests go through `src.mediawiki_api.api_get`, which handles etiquette
and retries. When a batch's content exceeds the API's response size limit,
the API returns part of it plus a continuation token, which is followed.
"""

from __future__ import annotations

import time
from typing import Iterator, Sequence

import requests

from src.mediawiki_api import API_URL, PAUSE_SECONDS, api_get

BATCH_SIZE = 50  # the API's limit for `revids`


def fetch_contents(
    revision_ids: Sequence[int],
    lang: str = "en",
    session: requests.Session | None = None,
    pause: float = PAUSE_SECONDS,
    api_url: str | None = None,
) -> Iterator[tuple[int, str | None]]:
    """Yield (revision id, wikitext) for each id, in input order. The text is
    None if the revision is missing, deleted, or its text is hidden."""
    session = session or requests.Session()
    url = api_url or API_URL.format(lang=lang)
    for start in range(0, len(revision_ids), BATCH_SIZE):
        batch = list(revision_ids[start : start + BATCH_SIZE])
        params = {
            "action": "query", "prop": "revisions", "revids": "|".join(map(str, batch)),
            "rvprop": "ids|content", "rvslots": "main", "format": "json", "formatversion": "2", "maxlag": "5",
        }
        found: dict[int, str | None] = {}
        continuation: dict = {}
        while True:
            payload = api_get(session, url, {**params, **continuation}, pause)
            for page in payload.get("query", {}).get("pages", []):
                for revision in page.get("revisions", []):
                    found[revision["revid"]] = revision.get("slots", {}).get("main", {}).get("content")
            if "continue" not in payload:
                break
            continuation = payload["continue"]
            time.sleep(pause)
        for revision_id in batch:
            yield revision_id, found.get(revision_id)
        time.sleep(pause)
