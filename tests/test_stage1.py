import math
import unittest
from datetime import date, datetime, timedelta

import numpy as np
import pyarrow as pa

from src.features.activity import PageActivity, day_ordinal
from src.features.bursts import CoBurstCounter
from src.ingest.revert_detect import detect_page_reverts
from src.stage1.baselines import BASELINES, lexicographic_score
from src.features.neighbors import LinkGraph, NeighborBursts
from src.stage1.features import (
    CATEGORICAL_FEATURES,
    EXCL_MASS_SUFFIX,
    FEATURE_SET_PARENTS,
    FEATURE_SETS,
    LINK_FEATURES,
    LINK_FEATURES_EXCL_MASS,
    add_context_columns,
    add_neighbor_columns,
    add_neighbor_columns_to,
    feature_matrix,
    site_edit_totals,
)
from src.stage1.metrics import average, day_metrics, mean_day_metrics, paired_difference, per_day_metrics
from src.stage1.panel import LABEL_COLUMNS, PANEL_BASE_SCHEMA, PANEL_SCHEMA, add_co_burst, panel_row
from src.stage1.splits import EMBARGO_DAYS, EVAL_DAY_STRIDE, make_splits


class SplitsTest(unittest.TestCase):
    def setUp(self):
        self.end = date(2026, 4, 2)
        self.splits = make_splits(self.end)

    def test_windows_are_ordered_with_embargo_gaps(self):
        s = self.splits
        self.assertEqual(s.test.end, self.end)
        self.assertEqual((s.test.end - s.test.start).days + 1, 365)
        # Every label in a window is final before the next window starts.
        self.assertEqual((s.test.start - s.validation.end).days - 1, EMBARGO_DAYS)
        self.assertEqual((s.validation.start - s.train.end).days - 1, EMBARGO_DAYS)
        self.assertLess(s.train.start, s.train.end)

    def test_mask(self):
        days = pa.array([self.splits.test.start - timedelta(days=1), self.splits.test.start, self.end], pa.date32())
        self.assertEqual(self.splits.test.mask(days).to_pylist(), [False, True, True])

    def test_eval_days_cycle_through_weekdays(self):
        days = self.splits.test.days(EVAL_DAY_STRIDE)
        self.assertEqual(len(days), 29)
        weekdays = [d.weekday() for d in days[:28]]
        self.assertEqual(sorted(weekdays), sorted(list(range(7)) * 4))


class MetricsTest(unittest.TestCase):
    def test_hand_computed_day(self):
        m = day_metrics(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 0, 1, 0]), np.random.default_rng(0))
        self.assertEqual(m["precision@100"], 2 / 100)  # missing ranks count as misses
        self.assertEqual(m["recall@1000"], 1.0)
        self.assertAlmostEqual(m["average_precision"], (1 / 1 + 2 / 3) / 2)

    def test_ties_are_broken_randomly_not_by_position(self):
        labels = np.array([1] + [0] * 99)
        aps = [day_metrics(np.zeros(100), labels, np.random.default_rng(s))["average_precision"] for s in range(200)]
        self.assertLess(np.mean(aps), 0.2)  # a positional tie-break would give AP = 1 every time

    def test_day_without_positives_is_skipped_for_recall_and_ap(self):
        scores = np.array([0.9, 0.1, 0.8, 0.2])
        labels = np.array([1, 0, 0, 0])
        days = np.array([1, 1, 2, 2])
        m = mean_day_metrics(scores, labels, days)
        self.assertEqual(m["average_precision"], 1.0)
        self.assertEqual(m["recall@1000"], 1.0)
        self.assertEqual(m["precision@100"], (1 / 100 + 0) / 2)
        self.assertFalse(math.isnan(m["precision@1000"]))
        self.assertEqual(m, average(per_day_metrics(scores, labels, days)))

    def test_paired_difference(self):
        nan = float("nan")
        baseline = [{"m": 0.1}, {"m": 0.2}, {"m": nan}]
        candidate = [{"m": 0.2}, {"m": 0.1}, {"m": 0.5}]
        d = paired_difference(baseline, candidate, "m")
        self.assertAlmostEqual(d["mean"], 0.0)
        self.assertAlmostEqual(d["se"], 0.1)  # std([0.1, -0.1], ddof=1) / sqrt(2)
        self.assertEqual((d["days_better"], d["days_worse"], d["days"]), (1, 1, 2))


class BaselinesTest(unittest.TestCase):
    def test_lexicographic_score_orders_by_keys_and_keeps_ties(self):
        scores = lexicographic_score(np.array([1, 2, 2, 0]), np.array([5, 1, 1, 9]))
        self.assertEqual(scores.tolist(), [1.0, 2.0, 2.0, 0.0])

    def test_recency_ranks_never_edited_pages_last(self):
        table = pa.table({"days_since_last_edit": pa.array([3, None, 1], pa.int32()),
                          "edits_365d": pa.array([5, 0, 1], pa.int32())})
        scores = BASELINES["most recent edit"](table)
        self.assertEqual(np.argsort(-scores).tolist(), [2, 0, 1])


class FeatureSetsTest(unittest.TestCase):
    def test_no_label_columns_in_any_feature_set(self):
        for name, columns in FEATURE_SETS.items():
            self.assertFalse(set(columns) & set(LABEL_COLUMNS), name)

    def test_each_feature_set_extends_its_parent(self):
        self.assertEqual(set(FEATURE_SET_PARENTS) | {"habits"}, set(FEATURE_SETS))
        for child, parent in FEATURE_SET_PARENTS.items():
            self.assertTrue(set(FEATURE_SETS[parent]) < set(FEATURE_SETS[child]), child)

    def test_feature_set_columns_are_known(self):
        known = set(PANEL_SCHEMA.names) | {"day_of_week", "site_edits_1d"} | set(LINK_FEATURES) | set(LINK_FEATURES_EXCL_MASS)
        for name, columns in FEATURE_SETS.items():
            self.assertTrue(set(columns) <= known, name)
            self.assertTrue(set(CATEGORICAL_FEATURES) <= set(columns), name)

    def test_add_neighbor_columns(self):
        graph = LinkGraph(np.array([1, 3]), np.array([2, 2]))  # 1 -> 2 and 3 -> 2
        bursts = NeighborBursts(graph, np.array([1, 3]), np.array([19723, 19723]), np.array([1, 2]))  # 2024-01-01
        table = pa.table({"page_id": pa.array([2, 2], pa.int64()),
                          "date": pa.array([date(2024, 1, 2), date(2024, 1, 3)], pa.date32())})
        table = add_neighbor_columns(table, bursts)
        table = add_neighbor_columns(table, bursts, EXCL_MASS_SUFFIX)
        self.assertEqual(table["in_nbrs_bursting_1d"].to_pylist(), [2, 0])
        self.assertEqual(table["in_nbrs_multi_editor_bursting_1d" + EXCL_MASS_SUFFIX].to_pylist(), [1, 0])
        self.assertEqual(table["in_links"].to_pylist(), [2, 2])
        self.assertEqual(table.schema.names.count("in_links"), 1)
        # Several tables at once give the same columns as one at a time.
        one, two = add_neighbor_columns_to([table.select(["page_id", "date"]).slice(0, 1),
                                            table.select(["page_id", "date"]).slice(1, 1)], bursts)
        self.assertEqual(one["in_nbrs_bursting_1d"].to_pylist() + two["in_nbrs_bursting_1d"].to_pylist(), [2, 0])

    def test_context_columns_use_the_previous_day(self):
        daily = pa.table(
            {"date": pa.array([date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 2)], pa.date32()), "edits": [3, 4, 9]}
        )
        table = pa.table({"date": pa.array([date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 1)], pa.date32())})
        out = add_context_columns(table, site_edit_totals(daily))
        self.assertEqual(out["site_edits_1d"].to_pylist(), [7, 9, 0])
        self.assertEqual(out["day_of_week"].to_pylist(), [1, 2, 0])  # 2024-01-01 was a Monday

    def test_feature_matrix(self):
        table = pa.table({"a": pa.array([1, None], pa.int32()), "b": [True, False]})
        m = feature_matrix(table, ["a", "b"])
        self.assertEqual(m.dtype, np.float32)
        self.assertTrue(np.isnan(m[1, 0]))
        self.assertEqual(m[:, 1].tolist(), [1.0, 0.0])


class PanelRowTest(unittest.TestCase):
    def test_row_matches_schema_features_and_label(self):
        t = datetime(2024, 3, 1, 12)
        revisions = [
            {"page_id": 7, "revision_id": i + 1, "timestamp": (t + timedelta(days=i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "sha1": f"s{i}", "user_text": "Alice"}
            for i in range(3)
        ]
        page = PageActivity(detect_page_reverts(revisions))
        day = day_ordinal(revisions[2]["timestamp"])
        counter = CoBurstCounter.from_counts([(day - 1, 5, 2)])
        row = add_co_burst(panel_row(page, day, 1.0, last_final_day=day), counter, day)
        self.assertEqual(set(row), set(PANEL_SCHEMA.names))
        self.assertEqual(set(PANEL_SCHEMA.names) - set(PANEL_BASE_SCHEMA.names), {"co_burst_1d", "co_burst_multi_editor_1d"})
        self.assertTrue(row["y"])
        self.assertEqual(row["kept_edits"], 1)
        self.assertEqual((row["co_burst_1d"], row["co_burst_multi_editor_1d"]), (5, 2))
        for name, value in page.features(day).items():
            self.assertEqual(row[name], value)


if __name__ == "__main__":
    unittest.main()
