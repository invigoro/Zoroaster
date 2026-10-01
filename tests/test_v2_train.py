import math
import unittest
from datetime import date
from types import SimpleNamespace
from unittest import mock

import torch

from scripts import train_v2
from scripts.train_stage2 import collate, target_nll, token_nll
from scripts.train_v2 import encode, left_padded, length_batches, paired_metrics, split_nll, validity
from src.forecast.prompts import header, new_text

VOCAB = 50
EXAMPLE = {
    "page_title": "Jane_Roe", "date": date(2026, 3, 2), "lead": "Jane Roe is a British politician. " * 30,
    "heading_titles": ["Career", "Results"], "yesterday_known": True, "yesterday_sections": [],
    "yesterday_section_chars": [], "yesterday_kinds": [], "yesterday_prose": "",
    "edits_1d": 3, "edits_7d": 5, "edits_30d": 9, "is_burst_1d": True, "burst_z_1d": 4.2, "bursting_neighbors": [],
    "sections": ["Career"], "section_chars": [300], "kinds": ["prose"], "prose": "She was appointed chancellor.",
}


class _Chars:
    """A tokenizer with one token per character."""
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(c) for c in text])


class _Bigram(torch.nn.Module):
    """Logits that depend only on the current token, so each target token's
    NLL is the same wherever padding puts it."""

    def __init__(self):
        super().__init__()
        self.table = 3 * torch.randn(VOCAB, VOCAB, generator=torch.Generator().manual_seed(0))

    def forward(self, input_ids, attention_mask, logits_to_keep):
        return SimpleNamespace(logits=self.table[input_ids][:, -logits_to_keep:])


class EncodeTest(unittest.TestCase):
    def test_target_is_header_then_text_and_the_lead_shrinks_to_fit(self):
        prompt, target, head, shortened = encode(_Chars(), EXAMPLE, "page")
        self.assertEqual(target, [ord(c) for c in header(EXAMPLE) + new_text(EXAMPLE)] + [0])
        self.assertEqual(head, len(header(EXAMPLE)))
        self.assertFalse(shortened)
        with mock.patch.object(train_v2, "MAX_PROMPT_TOKENS", len(prompt) - 100):
            short, _, _, shortened = encode(_Chars(), EXAMPLE, "page")
        self.assertTrue(shortened and len(short) <= len(prompt) - 100)
        with mock.patch.object(train_v2, "TARGET_TOKENS", 10):
            _, target, head, _ = encode(_Chars(), EXAMPLE, "page")
        self.assertEqual((len(target), target[-1], head), (10, 0, 9))  # cut inside the header, then the end


class SplitTest(unittest.TestCase):
    def test_header_and_text_nll_add_up_and_match_the_header_alone(self):
        examples = [([5, 6, 7], [8, 9, 10, 11], 2), ([1], [2, 3, 4, 12, 13, 14], 3), ([11, 12, 13, 14], [15, 16], 1)]
        model = _Bigram()
        ids, mask, labels, keep = collate([(p, t) for p, t, _ in examples], pad_id=0)
        nll, _ = token_nll(model, ids, mask, labels, keep)
        parts = split_nll(nll, [len(t) for _, t, _ in examples], [h for _, _, h in examples])
        whole, _ = target_nll(model, ids, mask, labels, keep)
        headers, _ = target_nll(model, *collate([(p, t[:h]) for p, t, h in examples], pad_id=0))
        for (h_nll, h_count, t_nll, t_count), (_, t, h), total, alone in zip(parts, examples, whole, headers):
            self.assertEqual((h_count, t_count), (h, len(t) - h))
            self.assertAlmostEqual(h_nll, alone.item(), places=4)
            self.assertAlmostEqual(h_nll + t_nll, total.item(), places=4)
        self.assertEqual(len({round(p[0], 3) for p in parts}), 3)  # rows differ, so a misaligned split would show


class BatchingTest(unittest.TestCase):
    def test_left_padded(self):
        ids, mask = left_padded([[5, 6], [7]], pad_id=0)
        self.assertEqual((ids.tolist(), mask.tolist()), ([[5, 6], [0, 7]], [[1, 1], [0, 1]]))

    def test_length_batches_cap_rows_and_padded_tokens(self):
        lengths = [50, 10, 300, 20, 30, 100]
        batches = length_batches(lengths, max_rows=3, max_tokens=300)
        self.assertEqual(batches, [[1, 3, 4], [0, 5], [2]])  # at most 3 rows; adding 300 would pad [0, 5] to 900
        self.assertEqual(sorted(i for b in batches for i in b), list(range(6)))
        self.assertEqual(length_batches([500], 3, 300), [[0]])  # an over-long prompt still gets a batch


class ComparisonTest(unittest.TestCase):
    def test_paired_metrics_skip_undefined_examples(self):
        nan = math.nan
        a = [{"section_precision": 0.5, "main_section_hit": 0.0, "kinds_jaccard": 0.2},
             {"section_precision": nan, "main_section_hit": 1.0, "kinds_jaccard": 0.5},
             {"section_precision": 0.0, "main_section_hit": 0.0, "kinds_jaccard": 0.0}]
        b = [{"section_precision": 1.0, "main_section_hit": 1.0, "kinds_jaccard": 0.4},
             {"section_precision": 1.0, "main_section_hit": 1.0, "kinds_jaccard": 0.5},
             {"section_precision": 1.0, "main_section_hit": 1.0, "kinds_jaccard": 1.0}]
        d = paired_metrics(a, b, [True, True, False])
        self.assertEqual(d["section_precision"]["examples"], 1)  # the second is undefined for a, the third left out
        self.assertAlmostEqual(d["section_precision"]["mean"], 0.5)
        self.assertEqual((d["main_section_hit"]["mean"], d["main_section_hit"]["examples"]), (0.5, 2))
        self.assertAlmostEqual(d["kinds_jaccard"]["mean"], 0.1)


class ValidityTest(unittest.TestCase):
    def test_counts_invented_sections(self):
        rows = [{"heading_titles": ["Career"], "living": True, "sections": ["Career"], "section_chars": [5]},
                {"heading_titles": ["History"], "living": False, "sections": ["Aftermath"], "section_chars": [9]}]
        texts = ["Sections: Career; Arrest\nKinds: prose\nNew text:", "Kinds: prose"]
        predicted = [{"sections": ["Career", "Arrest"], "kinds": ["prose"]}, {"sections": ["(lead)"], "kinds": []}]
        check = validity(texts, predicted, rows)
        self.assertEqual((check["parsed"], check["sections_named"], check["kinds_named"]), (0.5, 1.5, 0.5))
        self.assertAlmostEqual(check["invented_share"], 1 / 3)  # Arrest; the lead always exists
        self.assertEqual(check["actual_new_share"], 0.5)  # Aftermath was new
        self.assertEqual(check["invented_for_living_people"], [("Arrest", 1)])


if __name__ == "__main__":
    unittest.main()
