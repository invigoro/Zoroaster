import unittest
from datetime import date

import numpy as np
import pyarrow as pa

from scripts.build_v2_targets import choose, day_bounds


class ChooseTest(unittest.TestCase):
    def test_top_per_day_then_random_from_the_rest(self):
        table = pa.table({"page_id": list(range(10)), "date": [date(2026, 1, 1)] * 5 + [date(2026, 1, 2)] * 5})
        scores = np.array([0.1, 0.9, 0.5, 0.8, 0.2, 0.3, 0.4, 0.95, 0.1, 0.6])
        chosen = choose(table, scores, top=2, extra=1, rng=np.random.default_rng(0))
        rows = list(zip(chosen["page_id"].to_pylist(), chosen["selection"].to_pylist()))
        self.assertEqual([p for p, kind in rows if kind == "top"], [1, 3, 7, 9])  # by score, each day
        randoms = [p for p, kind in rows if kind == "random"]
        self.assertEqual(len(randoms), 2)
        self.assertTrue(randoms[0] in (0, 2, 4) and randoms[1] in (5, 6, 8))  # from the rest of each day
        self.assertEqual(chosen["score"].to_pylist()[0], 0.9)


class DayBoundsTest(unittest.TestCase):
    def test_states_at_the_ends_of_d_minus_2_d_minus_1_and_d(self):
        revisions = {1: [
            ("2026-09-28T10:00:00Z", 11, False),
            ("2026-09-29T09:00:00Z", 12, False),
            ("2026-09-30T08:00:00Z", 13, False),
            ("2026-09-30T23:00:00Z", 14, True),  # reverted after midnight: not the end state
            ("2026-10-01T00:10:00Z", 15, False),
        ], 2: [("2026-09-30T05:00:00Z", 21, False)]}
        (one, two) = day_bounds([(1, date(2026, 9, 30)), (2, date(2026, 9, 30))], revisions)
        self.assertEqual(one, {"start_id": 11, "prompt_id": 12, "end_id": 13, "edits_on_day": 2})
        self.assertEqual(two, {"start_id": None, "prompt_id": None, "end_id": 21, "edits_on_day": 1})


if __name__ == "__main__":
    unittest.main()
