import unittest

from scripts.public_grades import guardrail_reasons, public, to_check

GRADE = {"outcome": "happened", "already_known": False, "specificity": 1, "grounded": True}


class PublicGradesTest(unittest.TestCase):
    def test_only_the_guardrails_drop_a_line(self):
        screened = {"dropped_because": ["names or points to a person (the coach)", "the evidence already settles it",
                                        "doesn't name the teams or countries it means (an important team)"]}
        self.assertEqual(guardrail_reasons(screened), ["names or points to a person (the coach)"])
        self.assertEqual(guardrail_reasons({"dropped_because": ["the evidence already settles it"]}), [])

    def test_a_line_is_checked_once_as_it_reads(self):
        confirmed = {"a#0": GRADE | {"public": "Japan won."}, "b#1": GRADE | {"public": "It rained."}, "c#2": GRADE}
        earlier = {"a#0": {"check": {"line": "Japan won.", "dropped_because": []}},
                   "b#1": {"check": {"line": "It snowed.", "dropped_because": []}}}
        self.assertEqual(to_check(confirmed, earlier), {"b#1": "It rained."})  # edited since; c has no line

    def test_what_the_site_shows(self):
        confirmed = {"2026-10-05#3": GRADE | {"specificity": 0, "public": "Japan won."},
                     "2026-10-08#1": GRADE | {"outcome": "not possible", "already_known": True, "public": "The coach left."},
                     "2026-10-09#2": GRADE | {"public": "Rewritten since its check."}}
        texts = {k: f"I predict {k}." for k in confirmed}
        checks = {"2026-10-05#3": {"line": "Japan won.", "dropped_because": []},
                  "2026-10-08#1": {"line": "The coach left.", "dropped_because": ["names or points to a person (the coach)"]},
                  "2026-10-09#2": {"line": "Its first wording.", "dropped_because": []}}
        shown = public(confirmed, texts, checks)
        self.assertEqual({k: v["date"] for k, v in shown.items()},
                         {"2026-10-05#3": "2026-10-05", "2026-10-08#1": "2026-10-08", "2026-10-09#2": "2026-10-09"})
        self.assertEqual(shown["2026-10-05#3"]["line"], "Japan won.")
        self.assertTrue(shown["2026-10-05#3"]["vague"])
        # A line the guardrails drop isn't shown: the outcome stands alone until a new line is confirmed.
        self.assertNotIn("line", shown["2026-10-08#1"])
        self.assertEqual((shown["2026-10-08#1"]["outcome"], shown["2026-10-08#1"]["already_known"]), ("not possible", True))
        self.assertNotIn("line", shown["2026-10-09#2"])  # only a line checked as it now reads


if __name__ == "__main__":
    unittest.main()
