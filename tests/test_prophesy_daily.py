import json
import unittest

from scripts.prophesy_daily import public


class PublicTest(unittest.TestCase):
    def test_only_the_published_predictions_go_out(self):
        record = {"date": "2026-10-04", "generated_at": "2026-10-04T00:41:00+00:00", "model": "Qwen/Qwen2.5-7B-Instruct",
                  "predictions": [
                      {"text": "I predict that Spain wins the final.", "due": "2026-10-05", "topic": "sport",
                       "confidence": "medium", "published": True, "evidence": ["2026 Final"]},
                      {"text": "I predict that an armed group seizes another town in Afar.", "due": "2026-10-11",
                       "topic": "conflict", "confidence": "low", "published": True,
                       "evidence": ["Current events: Tigray War"],
                       "rewritten_from": "I predict that the TPLF seizes another town in Afar."},
                      {"text": "I predict that it rains.", "due": "2026-10-04", "topic": "other", "confidence": "high",
                       "published": False, "kept": True}]}
        out = public(record)
        self.assertEqual([p["text"] for p in out["predictions"]],  # world events first, as the selection chose them
                         ["I predict that an armed group seizes another town in Afar.", "I predict that Spain wins the final."])
        self.assertEqual(out["predictions"][0], {"text": "I predict that an armed group seizes another town in Afar.",
                                                 "due": "2026-10-11", "topic": "conflict", "confidence": "low"})
        # A story's or a page's title can name a person, and a rewrite's original names whom it was about.
        for leak in ("TPLF", "Tigray", "2026 Final", "rains"):
            self.assertNotIn(leak, json.dumps(out))
        self.assertEqual((out["date"], out["model"]), ("2026-10-04", "Qwen/Qwen2.5-7B-Instruct"))
        self.assertNotIn("late", out)

    def test_a_prophecy_made_well_into_its_day_is_marked_late(self):
        # The 2026-10-06 prophecy crashed overnight and was made by hand that afternoon.
        record = {"date": "2026-10-06", "model": "Qwen/Qwen2.5-7B-Instruct", "predictions": []}
        self.assertTrue(public(record | {"generated_at": "2026-10-06T17:40:00+00:00"})["late"])
        self.assertNotIn("late", public(record | {"generated_at": "2026-10-06T00:46:20+00:00"}))
        self.assertNotIn("late", public(record | {"generated_at": "2026-10-05T23:59:00+00:00"}))


if __name__ == "__main__":
    unittest.main()
