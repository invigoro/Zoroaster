import unittest

from src.prophecy.judge import credit, day_change_text, diff_url, judge_messages, pack, parse_grade

ROW = {"prompt_id": 10, "end_id": 12, "sections": ["(lead)", "Game 1"], "section_chars": [20, 900],
       "kinds": ["prose", "table"], "prose": "The Astros won Game 1, 5–3.",
       "blocks": ["| champion = [[Houston Astros]] (1)\n", "<ref>x</ref>"]}


class JudgeTest(unittest.TestCase):
    def test_what_the_day_brought(self):
        text = day_change_text(ROW)
        self.assertIn("Sections changed: Game 1; (lead) (prose, table)", text)
        self.assertIn('New text: "The Astros won Game 1, 5–3."', text)
        self.assertIn("champion = Houston Astros (1)", text)  # links cleaned; the empty ref line left out
        self.assertEqual(day_change_text(ROW | {"end_id": 10}), "Not changed on the day.")
        self.assertEqual(diff_url(ROW), "https://en.wikipedia.org/w/index.php?diff=12&oldid=10")
        self.assertIsNone(diff_url(ROW | {"end_id": 10}))

    def test_a_pack_holds_before_after_and_the_day_s_events(self):
        prediction = {"text": "I predict that the Astros win Game 1.", "question": "Who wins?", "evidence": ["ALWCS"]}
        p = pack(prediction, "2026-09-30", {"ALWCS": "[1] ALWCS ...", "Other": "[2] ..."}, {"ALWCS": ROW}, ["Sports › x"])
        self.assertEqual(list(p["known_before"]), ["ALWCS"])  # only the cited page
        chat = judge_messages(p)
        self.assertIn("The prediction, for 2026-09-30 (UTC): I predict that the Astros win Game 1.", chat[1]["content"])
        self.assertIn("- Sports › x", chat[1]["content"])

    def test_grades_parse_and_score(self):
        grade = parse_grade('Sure: {"outcome": "happened", "already_known": false, "specificity": 1, "grounded": true, '
                            '"reason": "The page says the Astros won."}')
        self.assertEqual(grade["outcome"], "happened")
        self.assertEqual(credit(grade), 0.5)
        self.assertEqual(credit(grade | {"specificity": 2}), 1.0)
        self.assertEqual(credit(grade | {"already_known": True}), 0.0)
        self.assertEqual(credit(grade | {"outcome": "partly"}), 0.25)
        self.assertIsNone(credit(grade | {"outcome": "unknown"}))
        self.assertIsNone(parse_grade('{"outcome": "maybe", "specificity": 1}'))  # not in the rubric
        self.assertIsNone(parse_grade("no json"))


class AgreementTest(unittest.TestCase):
    def test_field_agreement_and_credit_differences(self):
        from scripts.judge_prophecies import agreement

        judge = {"a": {"outcome": "happened", "already_known": False, "specificity": 2, "grounded": True},
                 "b": {"outcome": "did not happen", "already_known": False, "specificity": 1, "grounded": True},
                 "c": None}  # the judge's answer didn't parse
        hand = {"a": {"outcome": "happened", "already_known": False, "specificity": 1, "grounded": True},
                "b": {"outcome": "did not happen", "already_known": True, "specificity": 1, "grounded": False},
                "c": {"outcome": "unknown", "already_known": False, "specificity": 1, "grounded": True}}
        got = agreement(judge, hand)
        self.assertEqual(got["graded_by_both"], 2)
        self.assertEqual((got["outcome"], got["already_known"], got["specificity"]), (1.0, 0.5, 0.5))
        self.assertEqual(got["credit_mean_abs_diff"], 0.25)  # a: 1.0 vs 0.5; b: 0 vs 0
        self.assertEqual(got["mean_credit"], {"judge": 0.5, "hand": 0.25})


if __name__ == "__main__":
    unittest.main()
