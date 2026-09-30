"""A polite GET for the MediaWiki action API.

Etiquette for Wikimedia's servers:
- sequential requests, with a pause between them left to the caller;
- a descriptive User-Agent (`src.common`);
- `maxlag`, so the API can tell a bot to back off while replication lags.

Rate limits, server errors and maxlag responses are retried with backoff,
honoring Retry-After. So are dropped connections and timeouts: one ended a
September backfill after 5 minutes.
"""

from __future__ import annotations

import time

import requests

from src.common import USER_AGENT

API_URL = "https://{lang}.wikipedia.org/w/api.php"
PAUSE_SECONDS = 1.0
MAX_RETRIES = 6
_RETRY_STATUS = {429, 500, 502, 503, 504}


def api_get(session: requests.Session, url: str, params: dict, pause: float = PAUSE_SECONDS) -> dict:
    """The decoded JSON response, after retrying throttling and server errors."""
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(pause * 5 * 2**attempt)
            continue
        maxlag = response.ok and response.json().get("error", {}).get("code") == "maxlag"
        if response.status_code in _RETRY_STATUS or maxlag:
            time.sleep(float(response.headers.get("Retry-After", pause * 5 * 2**attempt)))
            continue
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"API error: {payload['error']}")
        return payload
    raise RuntimeError(f"gave up after {MAX_RETRIES} attempts: {url} {str(params)[:120]}")
