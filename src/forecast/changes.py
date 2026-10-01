"""What a page gained over a day: version 2's targets and inputs (PLAN.md §6 step 10).

Given a page's wikitext at the start and end of a day, `day_change` gives:
- `prose`: the day's new prose, in page order, cleaned of markup
  (`src.stage2.wikitext.prose`) and capped. Words inserted into an existing
  sentence come out as fragments; `blocks` (the changed paragraphs, as
  they read at the end) can give whole sentences instead;
- `sections`: the sections that changed, in page order, with
  `section_chars`, how much changed in each (new text, and removed text in
  the section it came from);
- `kinds`: what kinds of change it was (KINDS). "prose" means new text of
  at least MIN_SPAN_CHARS; smaller wording changes are "copyedits".
  "removals" means removed text of that size;
- sizes, for filtering.

Text already in the start text doesn't count as new. That covers a whole
span found there, and any whole line found there as a line, since a line
diff can report a moved paragraph as deleted and reinserted next to new
text (`new_spans`). Likewise, removed text still found at the end was
moved, not removed.

`lead` and `headings` describe the page at the start, for the prompt.
"""

from __future__ import annotations

import re

from src.stage2.diff import WordDiff, word_diff
from src.stage2.wikitext import plain_text, prose

MIN_SPAN_CHARS = 25  # shorter inserted spans can't hold a sentence; they're kept only for the kinds
KINDS = ("new section", "prose", "copyedits", "removals", "references", "table", "template fields", "categories",
         "links")
LEAD = "(lead)"
_HEADING = re.compile(r"^(={2,6})[ \t]*(.*?)[ \t]*\1[ \t]*$", re.M)
_CATEGORY = re.compile(r"\[\[\s*Category\s*:", re.I)
_LINK = re.compile(r"\[\[(?!\s*(?:Category|File|Image)\s*:)", re.I)
_TABLE_LINE = re.compile(r"^[ \t]*(?:\{\||\|-|\|\}|!)", re.M)
_FIELD_LINE = re.compile(r"^[ \t]*\|[ \t]*[\w ]+?[ \t]*=", re.M)


def headings(text: str) -> list[tuple[int, str]]:
    """(level, title) of each section heading, in order; level 2 is `== X ==`."""
    return [(len(m.group(1)), plain_text(m.group(2))) for m in _HEADING.finditer(text)]


def lead(text: str, max_chars: int = 1500) -> str:
    """The plain prose before the first heading, cut at a word boundary."""
    first = _HEADING.search(text)
    body = " ".join(prose(text[: first.start()] if first else text))
    if len(body) <= max_chars:
        return body
    return body[:max_chars].rsplit(" ", 1)[0] + " …"


def section_at(text: str, offset: int) -> str:
    """The title of the section `offset` falls in, or LEAD."""
    title = LEAD
    for m in _HEADING.finditer(text, 0, max(offset, 0)):
        title = plain_text(m.group(2))
    return title


def classify(span: str, old_headings: set[str]) -> set[str]:
    """The kinds of change an inserted span makes."""
    kinds = set()
    if any(title not in old_headings for _, title in headings(span)):
        kinds.add("new section")
    pieces = prose(_HEADING.sub("", span))
    if any(len(piece) >= MIN_SPAN_CHARS for piece in pieces):
        kinds.add("prose")
    elif pieces:
        kinds.add("copyedits")
    if "<ref" in span.lower():
        kinds.add("references")
    if _TABLE_LINE.search(span) or "||" in span:
        kinds.add("table")
    if _FIELD_LINE.search(span) or "{{" in span:
        kinds.add("template fields")
    if _CATEGORY.search(span):
        kinds.add("categories")
    if _LINK.search(span):
        kinds.add("links")
    return kinds


def span_sections(end: str, offset: int, span: str) -> list[str]:
    """The sections an inserted span's text goes into: where it starts
    (unless it opens with a heading), then each heading inside it."""
    inner = [title for _, title in headings(span)]
    return inner if _HEADING.match(span) else [section_at(end, offset)] + inner


def new_spans(start: str | None, end: str) -> tuple[list[str], int, int]:
    """The day's new inserted spans (moved text left out), and all inserted and removed chars."""
    return _new_spans(word_diff(start, end), start)


def _new_spans(diff: WordDiff, start: str | None) -> tuple[list[str], int, int]:
    old_lines = {line.strip() for line in start.split("\n")} if start else set()
    spans = []
    for s in diff.inserted:
        if start and s in start:
            continue
        new = "\n".join(line for line in s.split("\n") if line.strip() not in old_lines).strip()
        if new:
            spans.append(new)
    return spans, sum(len(s) for s in diff.inserted), sum(len(s) for s in diff.removed)


def day_change(start: str | None, end: str, max_prose_chars: int = 1200) -> dict:
    """What the page gained from `start` to `end` (see the module docstring)."""
    diff = word_diff(start, end)
    spans, inserted, removed = _new_spans(diff, start)
    old_headings = {title for _, title in headings(start or "")}
    section_chars: dict[str, int] = {}
    kinds: set[str] = set()
    pieces: list[str] = []
    for span in spans:
        kinds |= classify(span, old_headings)
        body = _HEADING.sub("", span)  # a new section's title goes in `sections`, not the prose
        pieces += [p for p in prose(body) if len(p) >= MIN_SPAN_CHARS] if len(body) >= MIN_SPAN_CHARS else []
        targets = span_sections(end, end.find(span), span)
        for section in targets:
            section_chars[section] = section_chars.get(section, 0) + len(span) // len(targets)
    end_lines = {line.strip() for line in end.split("\n")}
    for span in diff.removed:
        gone = "\n".join(line for line in span.split("\n") if line.strip() not in end_lines).strip()
        if len(gone) < MIN_SPAN_CHARS or span in end:
            continue  # moved, or a word swapped in a copyedit
        kinds.add("removals")
        section = section_at(start or "", (start or "").find(span))
        section_chars[section] = section_chars.get(section, 0) + len(gone)
    text = " ".join(pieces)
    if len(text) > max_prose_chars:
        text = text[:max_prose_chars].rsplit(" ", 1)[0] + " …"
    return {"prose": text, "sections": list(section_chars), "section_chars": list(section_chars.values()),
            "kinds": [k for k in KINDS if k in kinds], "inserted_chars": inserted, "removed_chars": removed,
            "spans": spans, "blocks": diff.new_blocks}
