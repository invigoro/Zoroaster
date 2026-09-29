"""Stream rows out of a MediaWiki SQL table dump (`*.sql.gz`).

The dumps are mysqldump output: a `CREATE TABLE` statement that gives the
column order, then long `INSERT INTO ... VALUES (...),(...);` lines. Rows
are matched with one compiled regex per table, so parsing stays in C and
only one INSERT line is in memory at a time. Every row must parse, and the
rows must tile the line exactly; anything else raises, rather than letting
the regex resynchronize inside a string value and silently yield garbage.

Values come back as raw bytes: numbers as their digits, strings still
quoted and escaped exactly as in the dump (see `sql_str`). Titles are
compared across tables in this raw form. That's consistent, because every
table in a dump run comes from the same database and the same dump tool.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

_VALUE = rb"NULL|'(?:[^'\\]|\\.)*'|[-+0-9.eE]+"
_COLUMN = re.compile(rb"^\s+`([^`]+)`")
_VALUES_KEYWORD = b" VALUES"


_INSERT = b"\nINSERT INTO `"
_BLOCK_BYTES = 1 << 24


def _insert_bodies(f) -> Iterator[bytes]:
    """Each INSERT statement's rows as one `(..),(..),...;` bytes object.

    Dumps put a statement on one line (Simple Wikipedia) or one row per
    line after a bare `INSERT INTO ... VALUES` line (English Wikipedia).
    MySQL escapes newlines inside strings, so raw newlines only ever fall
    between rows (safe to drop) and `;\\n` only ever ends a statement.
    Works on 16MB blocks with C-level searches rather than line by line:
    on one-row-per-line dumps, per-line Python work dominated.
    """
    buffer, pos = b"\n", 0
    while True:
        start = buffer.find(_INSERT, pos)
        end = buffer.find(b";\n", start) if start >= 0 else -1
        if end < 0:
            block = f.read(_BLOCK_BYTES)
            if not block:
                return
            buffer, pos = buffer[pos if start < 0 else start :] + block, 0
            if start < 0:
                pos = max(len(buffer) - len(block) - len(_INSERT), 0)
            continue
        values = buffer.index(_VALUES_KEYWORD, start) + len(_VALUES_KEYWORD)
        yield buffer[values : end + 1].replace(b"\n", b"").strip()
        pos = end + 1


def read_columns(path: Path) -> list[str]:
    """Column names, in order, from the dump's CREATE TABLE statement."""
    columns: list[str] = []
    with gzip.open(path, "rb") as f:
        in_create = False
        for line in f:
            if line.startswith(b"CREATE TABLE"):
                in_create = True
            elif in_create:
                match = _COLUMN.match(line)
                if match:
                    columns.append(match.group(1).decode())
                elif line.startswith(b")"):
                    return columns
    raise ValueError(f"no CREATE TABLE statement in {path}")


def iter_rows(path: Path, columns: Sequence[str]) -> Iterator[tuple[bytes | None, ...]]:
    """Yield the requested `columns` of every row as raw bytes (None for NULL)."""
    all_columns = read_columns(path)
    index = [all_columns.index(c) for c in columns]
    row = re.compile(rb"\(" + b",".join([b"(" + _VALUE + b")"] * len(all_columns)) + rb"\)")
    with gzip.open(path, "rb") as f:
        for body in _insert_bodies(f):
            pos = 0
            for match in row.finditer(body):
                if match.start() != pos:
                    raise ValueError(f"unparsable row in {path} at byte {pos}: {body[pos:pos + 120]!r}")
                values = match.groups()
                yield tuple(None if values[i] == b"NULL" else values[i] for i in index)
                pos = match.end() + 1  # skip the ',' between rows
            if body[pos - 1 : pos] != b";":
                raise ValueError(f"unparsable row in {path} at byte {pos - 1}: {body[pos - 1:pos + 120]!r}")


_SEPARATORS = bytes.maketrans(b"(),", b"   ")


def iter_int_batches(path: Path, columns: Sequence[str]) -> Iterator[np.ndarray]:
    """Fast path for tables whose values are all integers (e.g. `pagelinks`).

    Yields one `(rows, len(columns))` int64 array per INSERT line, parsed by
    numpy, which is ~50x faster than `iter_rows` on English Wikipedia's
    ~1.7B-row `pagelinks`. Raises if a line holds anything else (a string
    or a NULL), or if the value count doesn't match rows x columns.
    """
    all_columns = read_columns(path)
    index = [all_columns.index(c) for c in columns]
    with gzip.open(path, "rb") as f:
        for body in _insert_bodies(f):
            body = body.rstrip(b";")
            if b"'" in body or b"N" in body:
                raise ValueError(f"{path}: not an all-integer table: {body[:120]!r}")
            values = np.fromstring(body.translate(_SEPARATORS), dtype=np.int64, sep=" ")
            rows = body.count(b"(")
            if values.size != rows * len(all_columns):
                raise ValueError(f"{path}: parsed {values.size} values for {rows} rows of {len(all_columns)}")
            yield values.reshape(rows, len(all_columns))[:, index]


def sql_str(value: bytes) -> bytes:
    """A quoted string value, unquoted but still escaped as in the dump."""
    return value[1:-1]
