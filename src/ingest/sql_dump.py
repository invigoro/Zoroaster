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

_VALUE = rb"NULL|'(?:[^'\\]|\\.)*'|[-+0-9.eE]+"
_COLUMN = re.compile(rb"^\s+`([^`]+)`")
_VALUES_KEYWORD = b" VALUES "


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
        for line in f:
            if not line.startswith(b"INSERT INTO"):
                continue
            pos = line.index(_VALUES_KEYWORD) + len(_VALUES_KEYWORD)
            for match in row.finditer(line, pos):
                if match.start() != pos:
                    raise ValueError(f"unparsable row in {path} at byte {pos}: {line[pos:pos + 120]!r}")
                values = match.groups()
                yield tuple(None if values[i] == b"NULL" else values[i] for i in index)
                pos = match.end() + 1  # skip the ',' between rows
            if line[pos - 1 : pos] != b";":
                raise ValueError(f"unparsable row in {path} at byte {pos - 1}: {line[pos - 1:pos + 120]!r}")


def sql_str(value: bytes) -> bytes:
    """A quoted string value, unquoted but still escaped as in the dump."""
    return value[1:-1]
