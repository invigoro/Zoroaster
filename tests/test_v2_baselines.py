import unittest

from scripts.v2_baselines import common_kinds, forecasts


class BaselinesTest(unittest.TestCase):
    def test_yesterday_again_and_its_fallback(self):
        busy = {"yesterday_sections": ["(lead)", "Career"], "yesterday_section_chars": [10, 90],
                "yesterday_kinds": ["prose"]}
        f = forecasts(busy, ["links"])
        self.assertEqual(f["yesterday again"], {"sections": ["Career", "(lead)"], "kinds": ["prose"]})
        quiet = {"yesterday_sections": [], "yesterday_section_chars": [], "yesterday_kinds": []}
        self.assertEqual(forecasts(quiet, ["links"])["yesterday again"], {"sections": ["(lead)"], "kinds": ["links"]})

    def test_common_kinds_are_the_majority_ones(self):
        rows = [{"kinds": ["prose", "links"]}, {"kinds": ["links"]}, {"kinds": ["prose", "table"]}]
        self.assertEqual(common_kinds(rows), ["prose", "links"])


if __name__ == "__main__":
    unittest.main()
