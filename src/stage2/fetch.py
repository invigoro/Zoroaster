"""Fetch many revisions' wikitext from the MediaWiki action API, 50 at a time.

Etiquette for Wikimedia's servers:
- sequential requests with a pause between them;
- a descriptive User-Agent (`src.common`);
- `maxlag`, so the API can tell a bot to back off while replication lags.

Rate limits, server errors and maxlag responses are retried with backoff,
honoring Retry-After. When a batch's content exceeds the API's response size
limit, the API returns part of it plus a continuation token, which is
followed.
"""

from __future__ import annotations

import time
from typing import Iterator, Sequence

import requests

from src.common import USER_AGENT

API_URL = "https://{lang}.wikipedia.org/w/api.php"
BATCH_SIZE = 50  # the API's limit for `revids`
PAUSE_SECONDS = 1.0
MAX_RETRIES = 6
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _get(session: requests.Session, url: str, params: dict, pause: float) -> dict:
    for attempt in range(MAX_RETRIES):
        response = session.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
        maxlag = response.ok and response.json().get("error", {}).get("code") == "maxlag"
        if response.status_code in _RETRY_STATUS or maxlag:
            time.sleep(float(response.headers.get("Retry-After", pause * 5 * 2**attempt)))
            continue
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"API error: {payload['error']}")
        return payload
    raise RuntimeError(f"gave up after {MAX_RETRIES} attempts: {url} {params.get('revids', '')[:80]}")


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
            payload = _get(session, url, {**params, **continuation}, pause)
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
