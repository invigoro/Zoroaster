import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from scripts.run_daily import plan, prune, run, write_status

DAY = date(2026, 10, 1)


def names(steps):
    return [name for name, _ in steps]


class PlanTest(unittest.TestCase):
    def test_normal_missed_and_done_days(self):
        with tempfile.TemporaryDirectory() as tmp:
            predictions, prophecies = Path(tmp), Path(tmp) / "v3"
            prophecies.mkdir()
            # a missed day: nothing for yesterday
            self.assertEqual(names(plan(DAY, predictions, forecasts_run=predictions, prophecies=prophecies)), [
                "predict 2026-09-30 (missed)", "predict 2026-10-01", "score 2026-09-30", "prophesy 2026-10-01",
                "build the site", "publish the site", "prune old candidate tables"])
            # the normal day: yesterday was predicted, not yet scored
            (predictions / "2026-09-30.parquet").touch()
            (predictions / "2026-09-30.json").touch()
            self.assertEqual(names(plan(DAY, predictions, forecasts_run=predictions, prophecies=prophecies)), [
                "predict 2026-10-01", "score 2026-09-30", "prophesy 2026-10-01", "build the site", "publish the site",
                "prune old candidate tables"])
            # already done today, but for version 3's prophecy
            (predictions / "2026-10-01.json").touch()
            (predictions / "2026-09-30.outcomes.json").touch()
            self.assertEqual(names(plan(DAY, predictions, forecasts_run=predictions, prophecies=prophecies)), [
                "prophesy 2026-10-01", "build the site", "publish the site", "prune old candidate tables"])
            # and all of it: the prophecy's public part marks it done
            (prophecies / "2026-10-01.prophecy.json").touch()
            self.assertEqual(names(plan(DAY, predictions, forecasts_run=predictions, prophecies=prophecies)), [
                "build the site", "publish the site", "prune old candidate tables"])


class RunTest(unittest.TestCase):
    def test_forecasts_page_refreshed_when_ranked_forecasts_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            predictions, run_dir = Path(tmp), Path(tmp) / "run"
            run_dir.mkdir()
            (predictions / "2026-10-01.json").touch()
            (predictions / "2026-09-30.outcomes.json").touch()
            (predictions / "2026-10-01.prophecy.json").touch()
            self.assertNotIn("refresh the forecasts page", names(plan(DAY, predictions, forecasts_run=run_dir,
                                                                      prophecies=predictions)))
            (run_dir / "ranked.json").touch()
            self.assertEqual(names(plan(DAY, predictions, forecasts_run=run_dir, prophecies=predictions)), [
                "refresh the forecasts page", "build the site", "publish the site", "prune old candidate tables"])

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

    def test_each_step_s_outcome_is_noted_as_the_run_goes(self):
        noted = []
        steps = [("predict", lambda: None), ("prophesy", lambda: 1 / 0), ("build the site", lambda: None)]
        run(steps, lambda line: None, lambda outcomes: noted.append(dict(outcomes)))
        # The site is built with the steps before it noted, so a failed prophecy reaches the watchdog.
        self.assertEqual(noted[1], {"predict": "ok", "prophesy": "failed"})
        self.assertEqual(noted[-1], {"predict": "ok", "prophesy": "failed", "build the site": "ok"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "2026-10-06.status.json"
            write_status(path, date(2026, 10, 6), noted[-1])
            status = json.loads(path.read_text())
            self.assertEqual((status["day"], status["steps"]["prophesy"]), ("2026-10-06", "failed"))


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
