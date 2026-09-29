"""Word-level diffs between two revisions' wikitext.

A wikitext paragraph is one line, so a line diff records a one-word fix as
the whole paragraph (3.6x too much text on the test corpus, PLAN.md §5).
Here a line diff only finds the changed blocks; a token diff inside each
block then finds exactly what was inserted and removed. Tokens are words,
single punctuation characters and whitespace runs, so the tokens of a text
join back to the text exactly.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

_TOKEN = re.compile(r"\s+|\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text)


@dataclass
class WordDiff:
    inserted: list[str] = field(default_factory=list)  # inserted spans, in order
    removed: list[str] = field(default_factory=list)  # removed spans, in order
    first_offset: int | None = None  # char offset in the old text of the first change

    @property
    def added_text(self) -> str:
        return "\n".join(self.inserted)

    @property
    def removed_text(self) -> str:
        return "\n".join(self.removed)


def word_diff(old: str | None, new: str) -> WordDiff:
    """Inserted and removed spans from `old` to `new` (`old=None`: page creation)."""
    old = old or ""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    line_starts = [0]
    for line in old_lines:
        line_starts.append(line_starts[-1] + len(line))
    result = WordDiff()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        old_block, new_block = "".join(old_lines[i1:i2]), "".join(new_lines[j1:j2])
        _token_diff(old_block, new_block, line_starts[i1], result)
    return result


def _token_diff(old_block: str, new_block: str, block_offset: int, result: WordDiff) -> None:
    a, b = tokenize(old_block), tokenize(new_block)
    offsets = [block_offset]
    for token in a:
        offsets.append(offsets[-1] + len(token))
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        inserted, removed = "".join(b[j1:j2]).strip(), "".join(a[i1:i2]).strip()
        if not inserted and not removed:
            continue  # whitespace-only change
        if result.first_offset is None:
            result.first_offset = offsets[i1]
        if inserted:
            result.inserted.append(inserted)
        if removed:
            result.removed.append(removed)
