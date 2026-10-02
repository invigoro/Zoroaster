import unittest

from src.prophecy.judge import credit, day_change_text, diff_url, gradable, judge_messages, pack, parse_grade

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
        # A change over five days, up to a prediction's due day, shows more lines: the final came 33rd of 64.
        long = ROW | {"blocks": [f"| SF-score{n} = {n}\n" for n in range(32)] + ["| final-team1 = JPN | final-score1 = 0\n"]}
        self.assertNotIn("final-team1", day_change_text(long))
        self.assertIn("final-team1 = JPN", day_change_text(long, days=5))
        p = pack({"text": "I predict …", "evidence": ["V"], "due": "2026-09-22"}, "2026-09-18", {}, {"V": long}, [])
        self.assertIn("final-team1 = JPN", p["day_brought"]["V"])
        self.assertEqual(diff_url(ROW), "https://en.wikipedia.org/w/index.php?diff=12&oldid=10")
        self.assertIsNone(diff_url(ROW | {"end_id": 10}))

    def test_a_pack_holds_before_after_and_the_day_s_events(self):
        prediction = {"text": "I predict that the Astros win Game 1.", "question": "Who wins?", "evidence": ["ALWCS"]}
        p = pack(prediction, "2026-09-30", {"ALWCS": "[1] ALWCS ...", "Other": "[2] ..."}, {"ALWCS": ROW}, ["Sports › x"])
        self.assertEqual(list(p["known_before"]), ["ALWCS"])  # only the cited page
        chat = judge_messages(p)
        self.assertIn("The prediction, for 2026-09-30 (UTC): I predict that the Astros win Game 1.", chat[1]["content"])
        self.assertIn("- Sports › x", chat[1]["content"])
        self.assertNotIn("kept", chat[1]["content"])  # the judge isn't told what the checks decided
        later = judge_messages(pack(prediction | {"due": "2026-10-04"}, "2026-09-30", {}, {}, []))[1]["content"]
        for phrase in ("made for 2026-09-30 and due on 2026-10-04", "from then to the end of 2026-10-04",
                       "the major events on 2026-10-04"):
            self.assertIn(phrase, later)

    def test_what_gets_graded(self):
        self.assertTrue(gradable({"dropped_because": []}))
        self.assertTrue(gradable({"dropped_because": ["not about its cited evidence", "the evidence already settles it"]}))
        for guarded in ("names or points to a person (Joey Logano)", "sensitive topic (trial)",
                        "copies an example from the instructions", "repeats an earlier prediction"):
            self.assertFalse(gradable({"dropped_because": ["not about its cited evidence", guarded]}), guarded)

    def test_grades_parse_and_score(self):
        grade = parse_grade('Sure: {"outcome": "happened", "already_known": false, "specificity": 1, "grounded": true, '
                            '"reason": "The page says the Astros won."}')
        self.assertEqual(grade["outcome"], "happened")
        self.assertEqual(credit(grade), 0.5)
        self.assertEqual(credit(grade | {"specificity": 2}), 1.0)
        self.assertEqual(credit(grade | {"already_known": True}), 0.0)
        self.assertEqual(credit(grade | {"outcome": "partly"}), 0.25)
        self.assertEqual(credit(grade | {"outcome": "not possible", "specificity": 2}), 0.0)  # scored, unlike unknown
        self.assertEqual(parse_grade('{"outcome": "not possible", "specificity": 1}')["outcome"], "not possible")
        self.assertIsNone(credit(grade | {"outcome": "unknown"}))
        # An election already held: settled before the day, even if the pack doesn't say who won.
        self.assertEqual(credit(grade | {"outcome": "unknown", "already_known": True}), 0.0)
        self.assertIsNone(parse_grade('{"outcome": "maybe", "specificity": 1}'))  # not in the rubric
        self.assertIsNone(parse_grade("no json"))


class ConfirmedGradesTest(unittest.TestCase):
    def test_reviews_are_keyed_like_their_run_s_packs(self):
        from scripts.confirmed_grades import confirmed, pack_key

        self.assertEqual(pack_key("2026-09-18-2"), "2026-09-18#2")
        self.assertEqual(pack_key("run7-2026-09-24-10", "run7"), "2026-09-24#10")
        self.assertIsNone(pack_key("run7-2026-09-24-0"))  # another run's
        self.assertIsNone(pack_key("2026-09-24-0", "run7"))
        grade = {"outcome": "happened", "already_known": False, "specificity": 1, "grounded": True}
        reviews = {"2026-09-24-0": {"status": "confirmed", "grade": grade, "note": ""},
                   "run7-2026-09-24-0": {"status": "adjusted", "grade": grade | {"outcome": "partly"}, "note": "x"}}
        self.assertEqual(confirmed(reviews), {"2026-09-24#0": grade})
        self.assertEqual(confirmed(reviews, "run7"), {"2026-09-24#0": grade | {"outcome": "partly"}})


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
