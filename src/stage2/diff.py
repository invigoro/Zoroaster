"""Word-level diffs between two revisions' wikitext.

A wikitext paragraph is one line, so a line diff records a one-word fix as
the whole paragraph (3.6x too much text on the test corpus, PLAN.md §5).
Here a line diff only finds the changed blocks, and a word diff inside each
block then finds what changed.

Words are whitespace-delimited, as in `git diff --word-diff`, so a URL, a
wikilink or a number stays one token. Splitting on punctuation instead
turned a changed citation URL into dozens of fragments. Changes separated by
at most MERGE_GAP unchanged tokens (a word and its spaces) merge into one
span, and separate spans are joined with SPAN_SEPARATOR.

The changed blocks themselves (old and new, capped at MAX_BLOCK_CHARS) are
kept too, so targets can be re-derived later without refetching.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

_TOKEN = re.compile(r"\s+|\S+")
MERGE_GAP = 3  # unchanged tokens between two changes that still merge them
SPAN_SEPARATOR = " … "
MAX_BLOCK_CHARS = 4000


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text)


@dataclass
class WordDiff:
    inserted: list[str] = field(default_factory=list)  # inserted spans, in order
    removed: list[str] = field(default_factory=list)  # removed spans, in order
    first_offset: int | None = None  # char offset in the old text of the first change
    old_blocks: list[str] = field(default_factory=list)  # changed line blocks, before
    new_blocks: list[str] = field(default_factory=list)  # and after

    @property
    def added_text(self) -> str:
        return SPAN_SEPARATOR.join(self.inserted)

    @property
    def removed_text(self) -> str:
        return SPAN_SEPARATOR.join(self.removed)


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
        if _token_diff(old_block, new_block, line_starts[i1], result):
            result.old_blocks.append(old_block[:MAX_BLOCK_CHARS])
            result.new_blocks.append(new_block[:MAX_BLOCK_CHARS])
    return result


def _token_diff(old_block: str, new_block: str, block_offset: int, result: WordDiff) -> bool:
    """Add the block's changed spans to `result`; False if it only changed whitespace."""
    a, b = tokenize(old_block), tokenize(new_block)
    offsets = [block_offset]
    for token in a:
        offsets.append(offsets[-1] + len(token))
    groups: list[list[int]] = []  # [i1, i2, j1, j2] of merged non-equal runs
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if groups and i1 - groups[-1][1] <= MERGE_GAP and j1 - groups[-1][3] <= MERGE_GAP:
            groups[-1][1], groups[-1][3] = i2, j2
        else:
            groups.append([i1, i2, j1, j2])
    changed = False
    for i1, i2, j1, j2 in groups:
        inserted, removed = "".join(b[j1:j2]).strip(), "".join(a[i1:i2]).strip()
        if not inserted and not removed:
            continue  # whitespace-only change
        changed = True
        if result.first_offset is None:
            result.first_offset = offsets[i1]
        if inserted:
            result.inserted.append(inserted)
        if removed:
            result.removed.append(removed)
    return changed
