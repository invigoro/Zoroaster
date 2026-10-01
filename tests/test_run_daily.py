import tempfile
import unittest
from datetime import date
from pathlib import Path

from scripts.run_daily import plan, prune, run

DAY = date(2026, 10, 1)


def names(steps):
    return [name for name, _ in steps]


class PlanTest(unittest.TestCase):
    def test_normal_missed_and_done_days(self):
        with tempfile.TemporaryDirectory() as tmp:
            predictions = Path(tmp)
            # a missed day: nothing for yesterday
            self.assertEqual(names(plan(DAY, predictions)), [
                "predict 2026-09-30 (missed)", "predict 2026-10-01", "score 2026-09-30", "build the site",
                "publish the site", "prune old candidate tables"])
            # the normal day: yesterday was predicted, not yet scored
            (predictions / "2026-09-30.parquet").touch()
            (predictions / "2026-09-30.json").touch()
            self.assertEqual(names(plan(DAY, predictions)), [
                "predict 2026-10-01", "score 2026-09-30", "build the site", "publish the site", "prune old candidate tables"])
            # already done today
            (predictions / "2026-10-01.json").touch()
            (predictions / "2026-09-30.outcomes.json").touch()
            self.assertEqual(names(plan(DAY, predictions)), ["build the site", "publish the site", "prune old candidate tables"])


class RunTest(unittest.TestCase):
    def test_every_step_runs_and_a_failure_is_reported(self):
        ran, log = [], []

        def fail():
            ran.append("fail")
            raise SystemExit(2)  # what a script's argparse error raises

        steps = [("a", lambda: ran.append("a")), ("b", fail), ("c", lambda: ran.append("c"))]
        self.assertEqual(run(steps, log.append), 1)
        self.assertEqual(ran, ["a", "fail", "c"])
        self.assertTrue(log[1].startswith("== b: FAILED"))
        self.assertEqual(run([("a", lambda: None)], log.append), 0)


class PruneTest(unittest.TestCase):
    def test_only_old_candidate_tables_go(self):
        with tempfile.TemporaryDirectory() as tmp:
            predictions = Path(tmp)
            for name in ("2026-09-16.parquet", "2026-09-17.parquet", "2026-09-16.json", "2026-09-16.outcomes.json",
                         "backtest-2026-09-08-2026-09-29.json"):
                (predictions / name).touch()
            self.assertEqual(prune(predictions, DAY, keep_days=14), 1)  # Sep 16 is 15 days back
            left = sorted(p.name for p in predictions.iterdir())
        self.assertEqual(left, ["2026-09-16.json", "2026-09-16.outcomes.json", "2026-09-17.parquet",
                                "backtest-2026-09-08-2026-09-29.json"])


if __name__ == "__main__":
    unittest.main()
