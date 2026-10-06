import unittest
from datetime import date

from scripts.check_daily import problems

DAY = date(2026, 10, 7)
PROPHECY = {"date": "2026-10-07", "predictions": []}


def status(day="2026-10-07", **steps):
    return {"day": day, "steps": steps or {"predict 2026-10-07": "ok", "prophesy 2026-10-07": "ok"}}


class ProblemsTest(unittest.TestCase):
    def test_a_good_night_has_none(self):
        self.assertEqual(problems(status(), PROPHECY, DAY), [])

    def test_a_failed_step_is_named(self):
        found = problems(status(**{"predict 2026-10-07": "ok", "prophesy 2026-10-07": "failed"}),
                         {"date": "2026-10-06"}, DAY)
        self.assertEqual(len(found), 2)  # as on 2026-10-06: the prophecy crashed, and yesterday's stayed up
        self.assertIn("prophesy 2026-10-07", found[0])
        self.assertIn("logs/daily/2026-10-07.log", found[0])
        self.assertIn("for 2026-10-06, not 2026-10-07", found[1])

    def test_a_night_that_never_ran_or_published(self):
        found = problems(status("2026-10-06"), {"date": "2026-10-06"}, DAY)
        self.assertIn("didn't run", found[0])
        self.assertEqual(len(problems(None, None, DAY)), 2)


if __name__ == "__main__":
    unittest.main()
