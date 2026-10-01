import unittest

from scripts.fetch_v2_examples import derive

D2 = "'''Jane Roe''' is a British politician.\n== Career ==\nShe was elected in 2010.\n[[Category:Living people]]\n"
D1 = D2.replace("in 2010.\n", "in 2010. She was re-elected in 2024 with a larger majority.\n")
D0 = D1.replace("majority.\n", "majority.\nIn 2026 she was appointed to the cabinet as chancellor.\n")
TARGET = {"page_id": 1, "start_id": 1, "prompt_id": 2, "end_id": 3}


class DeriveTest(unittest.TestCase):
    def test_prompt_side_and_target(self):
        row = derive(TARGET, {1: D2, 2: D1, 3: D0})
        self.assertEqual(row["lead"], "Jane Roe is a British politician.")
        self.assertEqual((row["heading_levels"], row["heading_titles"]), ([2], ["Career"]))
        self.assertTrue(row["living"])
        self.assertEqual(row["yesterday_prose"], "She was re-elected in 2024 with a larger majority.")
        self.assertEqual(row["prose"], "In 2026 she was appointed to the cabinet as chancellor.")
        self.assertEqual((row["sections"], row["kinds"]), (["Career"], ["prose"]))
        self.assertEqual(len(row["section_chars"]), 1)
        self.assertTrue(any("appointed to the cabinet" in b for b in row["blocks"]))
        self.assertEqual(row["page_text"], D1)  # the page as the forecaster saw it
        self.assertTrue(row["yesterday_known"])

    def test_quiet_yesterday_unknown_start_and_missing_text(self):
        quiet = derive(TARGET | {"start_id": 2}, {2: D1, 3: D0})
        self.assertEqual((quiet["yesterday_prose"], quiet["yesterday_known"]), ("", True))
        unknown = derive(TARGET | {"start_id": None}, {2: D1, 3: D0})
        self.assertEqual((unknown["yesterday_prose"], unknown["yesterday_known"]), ("", False))
        self.assertIsNone(derive(TARGET, {1: D2, 2: D1, 3: None}))  # the day's end state is hidden or gone
        self.assertFalse(derive(TARGET, {1: D2, 2: D2.replace("[[Category:Living people]]", ""), 3: D0})["living"])


if __name__ == "__main__":
    unittest.main()
