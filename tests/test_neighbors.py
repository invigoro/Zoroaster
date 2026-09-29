import unittest

import numpy as np

from src.features.neighbors import DEGREE_FEATURES, NEIGHBOR_BURST_FEATURES, LinkGraph, NeighborBursts

A, B, C, D, OUTSIDE = 1, 2, 3, 4, 50


class NeighborBurstsTest(unittest.TestCase):
    def setUp(self):
        # A -> B, C -> B, B -> D. A and C burst on day 10 (C with 2 editors), B on day 12.
        graph = LinkGraph(np.array([A, C, B]), np.array([B, B, D]))
        self.bursts = NeighborBursts(graph, np.array([A, C, B]), np.array([10, 10, 12]), np.array([1, 2, 1]))

    def rows(self, pages, days):
        return self.bursts.features(np.array(pages), np.array(days))

    def test_hand_computed(self):
        f = self.rows([B, B, D, A, OUTSIDE], [11, 10, 13, 13, 11])
        self.assertEqual(f["in_nbrs_bursting_1d"].tolist(), [2, 0, 1, 0, 0])  # B's day-10 bursts don't count on day 10
        self.assertEqual(f["in_nbrs_multi_editor_bursting_1d"].tolist(), [1, 0, 0, 0, 0])
        self.assertEqual(f["out_nbrs_bursting_1d"].tolist(), [0, 0, 0, 1, 0])  # A links to B, which burst on 12
        self.assertEqual(f["in_links"].tolist(), [2, 2, 1, 0, 0])
        self.assertEqual(f["out_links"].tolist(), [1, 1, 0, 1, 0])
        self.assertEqual(f["in_frac_bursting_1d"].tolist(), [1.0, 0.0, 1.0, 0.0, 0.0])

    def test_seven_day_window(self):
        f = self.rows([B, B, B], [11, 17, 18])  # windows 4..10, 10..16, 11..17
        self.assertEqual(f["in_nbrs_bursting_7d"].tolist(), [2, 2, 0])

    def test_suffix_applies_to_burst_features_only(self):
        f = self.bursts.features(np.array([B]), np.array([11]), suffix="_x")
        self.assertEqual(set(f), set(DEGREE_FEATURES) | {n + "_x" for n in NEIGHBOR_BURST_FEATURES})

    def test_features_ignore_bursts_on_or_after_the_prediction_day(self):
        rng = np.random.default_rng(0)
        source, target = rng.integers(0, 60, 400), rng.integers(0, 60, 400)
        keep = source != target
        graph = LinkGraph(source[keep], target[keep])
        pages, days, editors = rng.integers(0, 70, 300), rng.integers(100, 140, 300), rng.integers(1, 4, 300)
        full = NeighborBursts(graph, pages, days, editors)
        query_pages = rng.integers(0, 70, 200)
        for day in range(100, 142):
            past = days < day
            cut = NeighborBursts(graph, pages[past], days[past], editors[past])
            query_days = np.full(len(query_pages), day)
            a, b = full.features(query_pages, query_days), cut.features(query_pages, query_days)
            for name in a:
                np.testing.assert_array_equal(a[name], b[name], err_msg=f"{name} on day {day}")


if __name__ == "__main__":
    unittest.main()
