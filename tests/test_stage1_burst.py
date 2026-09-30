import unittest
from datetime import date

import pyarrow as pa

from scripts.train_stage1 import with_burst_labels
from src.stage1.baselines import BASELINES


class BurstLabelTest(unittest.TestCase):
    def test_label_is_the_rows_own_page_and_day(self):
        rows = pa.table({
            "page_id": [7, 7, 7, 8, 80_000_000],
            "date": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 2), date(2026, 1, 2)],
            "y": [True, True, False, True, False],  # the old label, to be replaced
        })
        bursts = pa.table({"page_id": [7, 80_000_000], "date": [date(2026, 1, 2), date(2026, 1, 2)]})
        labeled = with_burst_labels(rows, bursts)
        # page 7 burst on Jan 2 only: not the day before or after, and not page 8 that day
        self.assertEqual(labeled["y"].to_pylist(), [False, True, False, False, True])
        self.assertEqual(labeled.schema.names, rows.schema.names)

    def test_persistence_baseline_ranks_yesterdays_bursts_first(self):
        table = pa.table({"burst_z_1d": [0.0, 5.2, 3.1, 5.2], "edits_1d": [9, 6, 4, 8]})
        scores = BASELINES["burst z yesterday"](table)
        self.assertEqual(sorted(range(4), key=lambda i: -scores[i]), [3, 1, 2, 0])  # ties broken by yesterday's edits


if __name__ == "__main__":
    unittest.main()
