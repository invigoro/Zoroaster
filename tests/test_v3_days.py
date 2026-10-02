import unittest
from datetime import date

import pyarrow as pa

from scripts.build_v3_days import bounds_or_unchanged, neighbors_bursting


class BoundsTest(unittest.TestCase):
    def test_a_page_with_nothing_kept_on_the_day_ends_it_unchanged(self):
        revisions = {1: [("2026-09-28T10:00:00Z", 11, False), ("2026-09-29T09:00:00Z", 12, False),
                         ("2026-09-30T08:00:00Z", 13, True)],  # D's only edit was reverted
                     2: [("2026-09-28T10:00:00Z", 21, False), ("2026-09-30T05:00:00Z", 22, False)]}
        one, two = bounds_or_unchanged([(1, date(2026, 9, 30)), (2, date(2026, 9, 30))], revisions)
        self.assertEqual((one["start_id"], one["prompt_id"], one["end_id"]), (11, 12, 12))
        self.assertEqual((two["start_id"], two["prompt_id"], two["end_id"], two["edits_on_day"]), (21, 21, 22, 1))


class NeighborsTest(unittest.TestCase):
    def test_linked_pages_that_burst_the_day_before_most_editors_first(self):
        links = pa.table({"source": pa.array([1, 1, 4, 1], pa.int32()), "target": pa.array([2, 3, 1, 5], pa.int32())})
        day = pa.table({"page_id": [2, 3, 4, 5], "page_title": ["Two", "Three", "Four", "Five"],
                        "is_burst_1d": [True, False, True, True], "editors_1d": [3, 9, 7, 3]})
        self.assertEqual(neighbors_bursting([1, 6], links, day), {1: ["Four", "Two", "Five"], 6: []})
        self.assertEqual(neighbors_bursting([1], links, day, max_shown=1), {1: ["Four"]})  # links run both ways


if __name__ == "__main__":
    unittest.main()
