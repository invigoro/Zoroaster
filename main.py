"""Fetch a Wikipedia page and print its summary.

Uses Wikipedia's free REST API summary endpoint (no API key required):
    https://en.wikipedia.org/api/rest_v1/page/summary/{title}

Usage:
    python main.py "Zoroastrianism"
    python main.py "Albert Einstein" --lang de
"""

import argparse
import sys
import urllib.parse

import requests

# Wikimedia asks API clients to send a descriptive User-Agent identifying the
# project and a contact, so that abusive traffic can be traced. See:
# https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = "Zoroaster/0.1 (https://github.com/invigoro/Zoroaster)"

REST_SUMMARY_URL = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"


def get_summary(title: str, lang: str = "en") -> dict:
    """Return the summary payload for a Wikipedia page title.

    Raises requests.HTTPError if the page does not exist or the request fails.
    """
    # The title goes in the path, so percent-encode it (spaces, slashes, etc.).
    encoded_title = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = REST_SUMMARY_URL.format(lang=lang, title=encoded_title)

    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch a Wikipedia page and print its summary."
    )
    parser.add_argument(
        "title",
        nargs="?",
        default="Zoroastrianism",
        help="Wikipedia page title (default: %(default)s)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Wikipedia language edition, e.g. en, de, fr (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        data = get_summary(args.title, args.lang)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        if status == 404:
            print(f"No Wikipedia page found for {args.title!r}.", file=sys.stderr)
        else:
            print(f"Request failed (HTTP {status}): {exc}", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"Network error: {exc}", file=sys.stderr)
        return 1

    print(data.get("title", args.title))
    print("=" * len(data.get("title", args.title)))
    print(data.get("extract", "(no summary available)"))

    page_url = data.get("content_urls", {}).get("desktop", {}).get("page")
    if page_url:
        print(f"\nSource: {page_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
