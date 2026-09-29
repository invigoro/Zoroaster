"""Time-based train / validation / test windows for Stage 1.

Windows are separated by an embargo as long as the revert window. A label
("did the page get a kept edit on day D?") only becomes final once D is past
the 90-day revert window, so a model deployed on day T could only have been
trained on prediction days up to T - 90. Each window therefore ends at
least EMBARGO_DAYS before the next one starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pyarrow as pa
import pyarrow.compute as pc

from src.ingest.revert_detect import TIME_WINDOW

EMBARGO_DAYS = TIME_WINDOW.days
EVAL_DAY_STRIDE = 13  # coprime with 7, so eval days cycle evenly through weekdays


@dataclass(frozen=True)
class Window:
    start: date
    end: date  # inclusive

    def mask(self, dates: pa.Array | pa.ChunkedArray) -> pa.ChunkedArray:
        """Boolean mask of which `dates` (date32) fall inside the window."""
        start = pa.scalar(self.start, pa.date32())
        end = pa.scalar(self.end, pa.date32())
        return pc.and_(pc.greater_equal(dates, start), pc.less_equal(dates, end))

    def days(self, stride: int = 1) -> list[date]:
        n = (self.end - self.start).days + 1
        return [self.start + timedelta(days=i) for i in range(0, n, stride)]


@dataclass(frozen=True)
class Splits:
    train: Window
    validation: Window
    test: Window


def make_splits(
    labels_final_through: date,
    test_days: int = 365,
    validation_days: int = 365,
    train_days: int = 5 * 365,
    embargo_days: int = EMBARGO_DAYS,
) -> Splits:
    """Test is the latest `test_days` with final labels; validation and
    train precede it, each followed by an `embargo_days` gap."""
    test = Window(labels_final_through - timedelta(days=test_days - 1), labels_final_through)
    validation_end = test.start - timedelta(days=embargo_days + 1)
    validation = Window(validation_end - timedelta(days=validation_days - 1), validation_end)
    train_end = validation.start - timedelta(days=embargo_days + 1)
    train = Window(train_end - timedelta(days=train_days - 1), train_end)
    return Splits(train=train, validation=validation, test=test)
