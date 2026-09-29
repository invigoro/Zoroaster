"""Streaming Parquet output shared by the build scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import pyarrow as pa
import pyarrow.parquet as pq


class RowGroupWriter:
    """Write dict rows to Parquet one row group at a time.

    Rows are buffered as columns and flushed every `row_group_size` rows, so
    memory stays bounded however many rows are written. Output goes to a
    temporary file that replaces `path` only on a clean exit from the `with`
    block, so a failed run never leaves a truncated file behind (and the
    input of a job may safely be its own output path).
    """

    def __init__(self, path: Path, schema: pa.Schema, row_group_size: int = 250_000):
        self.path = Path(path)
        self.schema = schema
        self.row_group_size = row_group_size
        self.rows_written = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp_path = self.path.with_name(self.path.name + ".tmp")
        self._writer = pq.ParquetWriter(self._tmp_path, schema, compression="zstd")
        self._columns: dict[str, list] = {name: [] for name in schema.names}
        self._buffered = 0

    def append(self, row: Mapping) -> None:
        for name, values in self._columns.items():
            values.append(row[name])
        self._buffered += 1
        if self._buffered >= self.row_group_size:
            self._flush()

    def write_table(self, table: pa.Table) -> None:
        self._flush()
        self._writer.write_table(table)
        self.rows_written += table.num_rows

    def _flush(self) -> None:
        if self._buffered:
            self._writer.write_table(pa.table(self._columns, schema=self.schema))
            self.rows_written += self._buffered
            self._columns = {name: [] for name in self.schema.names}
            self._buffered = 0

    def __enter__(self) -> RowGroupWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self._flush()
        self._writer.close()
        if exc_type is None:
            os.replace(self._tmp_path, self.path)
        else:
            self._tmp_path.unlink(missing_ok=True)
