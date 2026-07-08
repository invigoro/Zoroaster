"""Shared constants and helpers for interacting with Wikimedia services."""

# Wikimedia asks API clients to send a descriptive User-Agent identifying the
# project and a contact, so that abusive traffic can be traced. See:
# https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = "Zoroaster/0.1 (https://github.com/invigoro/Zoroaster)"

# Heuristic: WMF convention is bot accounts containing "bot" in the username.
# Not authoritative (misses bots that don't follow the convention, and could
# false-positive on a human username containing "bot") but requires no extra
# data source, unlike the `bot` user-group flag.
BOT_NAME_MARKERS = ("bot",)


def is_bot_edit(user_text: str | None) -> bool:
    if not user_text:
        return False
    return any(marker in user_text.lower() for marker in BOT_NAME_MARKERS)
