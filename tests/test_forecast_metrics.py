import math
import unittest

from src.forecast.metrics import main_sections, score, summarize

ACTUAL = {"sections": ["(lead)", "Career", "Results"], "section_chars": [40, 300, 120], "kinds": ["prose", "table"]}


class MetricsTest(unittest.TestCase):
    def test_main_sections_by_size(self):
        self.assertEqual(main_sections(ACTUAL["sections"], ACTUAL["section_chars"], 2), ["Career", "Results"])
        self.assertEqual(main_sections(["A", "B"], [5, 5], 1), ["A"])  # ties keep page order

    def test_score_one_forecast(self):
        s = score({"sections": ["Career", "Legacy"], "kinds": ["prose", "references"]}, ACTUAL)
        self.assertEqual((s["section_precision"], s["main_section_hit"]), (0.5, 1.0))
        self.assertAlmostEqual(s["kinds_jaccard"], 1 / 3)  # prose shared; references and table not
        self.assertEqual(s["kinds"]["table"], (False, True))

    def test_empty_cases_and_summary(self):
        quiet = {"sections": [], "section_chars": [], "kinds": []}
        s = score({"sections": [], "kinds": []}, quiet)
        self.assertTrue(math.isnan(s["section_precision"]) and math.isnan(s["main_section_hit"]))
        self.assertEqual(s["kinds_jaccard"], 1.0)  # nothing said, nothing done
        a = score({"sections": ["Career"], "kinds": ["prose"]}, ACTUAL)
        b = score({"sections": ["Legacy"], "kinds": ["table"]}, ACTUAL)
        summary = summarize([a, b, s])
        self.assertEqual((summary["section_precision"], summary["main_section_hit"]), (0.5, 0.5))
        self.assertEqual((summary["prose precision"], summary["prose recall"]), (1.0, 0.5))
        self.assertTrue(math.isnan(summary["removals precision"]))


if __name__ == "__main__":
    unittest.main()
