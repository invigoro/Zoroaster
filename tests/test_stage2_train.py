import unittest
from types import SimpleNamespace

import torch

from scripts.train_stage2 import attach_changes, attach_relevant, collate, paired, point_in_time_titles, pooled, target_nll, unshortened

VOCAB = 50


class _Oracle(torch.nn.Module):
    """Puts a sharp logit on the true next token at every position, so
    correctly aligned labels score a near-zero loss and misaligned ones don't."""

    def forward(self, input_ids, attention_mask, logits_to_keep):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, VOCAB)
        logits[:, :-1].scatter_(2, input_ids[:, 1:].unsqueeze(-1), 30.0)
        return SimpleNamespace(logits=logits[:, -logits_to_keep:])


class CollateTest(unittest.TestCase):
    def test_left_padding_and_labels(self):
        ids, mask, labels, keep = collate([([5, 6, 7], [8, 9]), ([1], [2, 3, 4])], pad_id=0)
        self.assertEqual(ids.tolist(), [[5, 6, 7, 8, 9], [0, 0, 1, 2, 3, 4][1:]])
        self.assertEqual(mask.tolist(), [[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]])
        self.assertEqual(labels.tolist(), [[-100, -100, -100, 8, 9], [-100, -100, 2, 3, 4]])
        self.assertEqual(keep, 4)  # longest target + 1: its first token is predicted from the prompt's last

    def test_target_nll_is_aligned_with_the_kept_logits(self):
        pairs = [([5, 6, 7], [8, 9]), ([1], [2, 3, 4]), ([11, 12, 13, 14], [15])]
        ids, mask, labels, keep = collate(pairs, pad_id=0)
        nll, counts = target_nll(_Oracle(), ids, mask, labels, keep)
        self.assertEqual(counts.tolist(), [2, 3, 1])
        self.assertTrue(torch.all(nll / counts < 1e-6), nll)
        # Labels shifted one position earlier (still inside the kept window)
        # must score badly: the check can fail.
        shifted = torch.full_like(labels, -100)
        shifted[:, :-1] = labels[:, 1:]
        wrong, wrong_counts = target_nll(_Oracle(), ids, mask, shifted, keep + 1)
        self.assertEqual(wrong_counts.tolist(), [2, 3, 1])
        self.assertTrue(torch.all(wrong / wrong_counts > 1.0), wrong)


class PooledAndChangesTest(unittest.TestCase):
    def test_pooled_averages_each_example_over_runs(self):
        self.assertEqual(pooled([[(2.0, 2), (4.0, 1)], [(4.0, 2), (2.0, 1)]]), [(3.0, 2), (3.0, 1)])

    def test_changes_come_from_the_previous_day(self):
        import tempfile
        from datetime import date
        from pathlib import Path
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            pq.write_table(pa.table({"title": ["A", "A", "B"], "date": [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 1)],
                                     "snippet": ["day 1", "day 2", None]}), Path(tmp) / "part-00000.parquet")
            rows = [{"date": date(2025, 1, 2), "bursting_neighbors": ["A", "B", "C"]}]
            attach_changes(rows, Path(tmp))
        self.assertEqual(rows[0]["neighbor_changes"], {"A": "day 1", "B": None, "C": None})

    def test_point_in_time_titles(self):
        import tempfile
        from datetime import date
        from pathlib import Path
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "titles.parquet"
            pq.write_table(pa.table({
                "kind": ["page", "neighbor", "neighbor", "neighbor"],
                "title": ["P_now", "N_now", "N_now", "M"],
                "date": [date(2025, 1, 2), date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 1)],
                "revision_id": [7, 8, 9, 10],
                "title_then": ["P_then", "N_then", "N_later", "M"],
            }), path)
            rows = [{"revision_id": 7, "page_title": "P_now", "date": date(2025, 1, 2),
                     "bursting_neighbors": ["N_now", "M", "Q"], "neighbor_changes": {"N_now": "s", "M": None, "Q": None}}]
            renamed = point_in_time_titles(rows, path)
        self.assertEqual(renamed, {"pages": 1, "neighbors": 1})
        self.assertEqual(rows[0]["page_title"], "P_then")
        self.assertEqual(rows[0]["bursting_neighbors"], ["N_then", "M", "Q"])  # the day before's title, not the day's
        self.assertEqual(rows[0]["neighbor_changes"], {"N_then": "s", "M": None, "Q": None})
        self.assertEqual(rows[0]["snapshot_neighbors"], ["N_now", "M", "Q"])  # the change data's keys

    def test_relevant_changes_come_from_the_previous_day(self):
        import tempfile
        from datetime import date
        from pathlib import Path
        import pyarrow as pa
        import pyarrow.parquet as pq
        named = "Rachel Reeves was named chancellor in the new cabinet."
        with tempfile.TemporaryDirectory() as tmp:
            pq.write_table(pa.table({
                "title": ["Cabinet_now", "Cabinet_now"], "date": [date(2025, 1, 1), date(2025, 1, 2)],
                "spans": [[named], ["Rachel Reeves resigned from the cabinet after a scandal broke."]],  # the day itself
            }), Path(tmp) / "part-00000.parquet")
            row = {"split": "test", "date": date(2025, 1, 2), "page_title": "Rachel_Reeves", "section": "Career",
                   "context": "She is a politician.", "bursting_neighbors": ["Cabinet_then"],
                   "snapshot_neighbors": ["Cabinet_now"]}
            attach_relevant([row, row | {"split": "train"}], Path(tmp))
        self.assertEqual(row["relevant_changes"], [("Cabinet_then", named)])  # shown under the title then


class PairedTest(unittest.TestCase):
    def test_per_example_mean_token_difference(self):
        a = [(4.0, 2), (3.0, 1)]  # 2.0 and 3.0 per token
        b = [(2.0, 2), (3.0, 1)]  # 1.0 and 3.0
        d = paired(a, b)
        self.assertAlmostEqual(d["mean"], -0.5)
        self.assertEqual((d["share_improved"], d["examples"]), (0.5, 2))
        self.assertEqual(paired(a, b, [True, False])["mean"], -1.0)

    def test_unshortened_drops_examples_either_prompt_shortened(self):
        a, b = [False, True, False, False], [False, False, True, False]
        self.assertEqual(unshortened(None, a, b), [True, False, False, True])
        self.assertEqual(unshortened([False, True, True, True], a, b), [False, False, False, True])


if __name__ == "__main__":
    unittest.main()
