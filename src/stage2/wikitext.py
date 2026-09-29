"""Rough wikitext-to-text, for showing in a prompt what an edit said.

This drops what isn't prose: comments, references, templates (infoboxes,
citations), tables, file and category links, and HTML tags. It keeps link
labels, list items and headings as text.

A span cut out of a page can start or end inside markup. An unmatched
closer drops what came before it, and an unmatched opener drops the rest,
except that a cut-off link keeps its label.
"""

from __future__ import annotations

import html
import re

_COMMENT = re.compile(r"<!--.*?(?:-->|$)", re.S)
_REF = re.compile(r"<ref\b[^>]*?/>|<ref\b[^>]*>.*?(?:</ref>|$)", re.S | re.I)
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
_TABLE_ROW = re.compile(r"^[ \t]*[|!].*$", re.M)
_LINK = re.compile(r"\[\[([^\[\]|]*)(?:\|([^\[\]]*))?\]\]")
_EXTERNAL = re.compile(r"\[(?:https?:)?//[^\s\]]*\s?([^\]]*)\]")
_URL = re.compile(r"(?:https?:)?//\S+")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_MAGIC = re.compile(r"__[A-Z]+__")
_HEADING = re.compile(r"^[ \t]*=+[ \t]*(.*?)[ \t]*=+[ \t]*$", re.M)
_LIST_MARK = re.compile(r"^[ \t]*[*#:;]+[ \t]*", re.M)
_EMPTY_PARENS = re.compile(r"\(\s*[,;:]*\s*\)")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?)])")
_LEADING_JUNK = re.compile(r"^[\s|,;:.)\]}>+-]+")
_HIDDEN_NAMESPACES = ("file:", "image:", "category:")


def plain_text(markup: str) -> str:
    text = _COMMENT.sub("", markup)
    first_close, first_open = text.find("</ref>"), text.find("<ref")
    if first_close >= 0 and (first_open < 0 or first_close < first_open):
        text = text[first_close + len("</ref>") :]  # began inside a reference
    text = _REF.sub("", text)
    text = _trim_unbalanced(_remove_nested(text, _TEMPLATE), "{{", "}}")
    text = _TABLE_ROW.sub("", _trim_unbalanced(_TABLE.sub("", text), "{|", "|}"))
    text = _links(text)
    text = _URL.sub("", _EXTERNAL.sub(r"\1", text))
    text = _MAGIC.sub("", _TAG.sub("", text))
    text = re.sub(r"={2,}", " ", _LIST_MARK.sub("", _HEADING.sub(r"\1", text)))
    text = html.unescape(re.sub(r"'{2,}", "", text))
    text = re.sub(r"\s+", " ", text)
    text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", _EMPTY_PARENS.sub("", text))
    return _LEADING_JUNK.sub("", text).strip()


def prose(markup: str) -> list[str]:
    """The pieces of `markup`'s plain text that read as prose.

    Text is split where table or template separators (`|`) survived, as
    when a span falls entirely inside a citation. Pieces are kept if they
    are mostly letters and have no leftover `name=value` parameter.
    """
    pieces = (_LEADING_JUNK.sub("", p).strip() for p in plain_text(markup).split("|"))
    return [p for p in pieces if p and "=" not in p and sum(c.isalpha() for c in p) >= len(p) / 2]


def _remove_nested(text: str, pattern: re.Pattern) -> str:
    """Remove innermost matches until none are left."""
    while True:
        text, n = pattern.subn("", text)
        if not n:
            return text


def _trim_unbalanced(text: str, opener: str, closer: str) -> str:
    """With balanced pairs gone, any closer left comes before any opener:
    drop through the last closer and from the first opener."""
    if closer in text:
        text = text[text.rindex(closer) + len(closer) :]
    if opener in text:
        text = text[: text.index(opener)]
    return text


def _link_text(match: re.Match) -> str:
    target, label = match.group(1), match.group(2)
    if target.strip().lower().startswith(_HIDDEN_NAMESPACES):
        return ""
    return label or target.lstrip(":")


def _links(text: str) -> str:
    text = _remove_nested_links(text)
    if "]]" in text:  # began inside a link: keep its label
        head, tail = text.rsplit("]]", 1)
        text = head.rsplit("|", 1)[-1] + tail
    if "[[" in text:  # ends inside a link
        head, rest = text.split("[[", 1)
        hidden = rest.strip().lower().startswith(_HIDDEN_NAMESPACES) and "|" not in rest
        text = head + ("" if hidden else rest.rsplit("|", 1)[-1])
    return text


def _remove_nested_links(text: str) -> str:
    """Replace links innermost first, so a file caption's links go before the file link itself."""
    while True:
        text, n = _LINK.subn(_link_text, text)
        if not n:
            return text
