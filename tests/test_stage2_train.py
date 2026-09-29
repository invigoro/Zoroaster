import unittest
from types import SimpleNamespace

import torch

from scripts.train_stage2 import collate, paired, target_nll

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


class PairedTest(unittest.TestCase):
    def test_per_example_mean_token_difference(self):
        a = [(4.0, 2), (3.0, 1)]  # 2.0 and 3.0 per token
        b = [(2.0, 2), (3.0, 1)]  # 1.0 and 3.0
        d = paired(a, b)
        self.assertAlmostEqual(d["mean"], -0.5)
        self.assertEqual((d["share_improved"], d["examples"]), (0.5, 2))
        self.assertEqual(paired(a, b, [True, False])["mean"], -1.0)


if __name__ == "__main__":
    unittest.main()
