import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_site import build, newest


class BuildSiteTest(unittest.TestCase):
    def test_newest_prophecy_and_record_become_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            web, predictions, out = tmp / "web", tmp / "predictions", tmp / "out"
            web.mkdir(), predictions.mkdir()
            (web / "index.html").write_text("<html></html>")
            (web / "img").mkdir()
            (web / "img" / "banner.webp").write_bytes(b"image")
            for name, day in (("2026-09-06.json", "a"), ("2026-09-07.json", "b"), ("2026-09-06.outcomes.json", "c")):
                (predictions / name).write_text(json.dumps({"day": day}))
            (predictions / "2026-09-07.parquet").write_text("not copied")
            self.assertEqual(newest(predictions, ".json").name, "2026-09-07.json")  # not an .outcomes.json
            used = build(out, predictions, web, forecasts=tmp / "none.json", prophecies=tmp / "none",
                         statuses=tmp / "none", grades=tmp / "none.json")
            data = out / "data"
            self.assertEqual(used, {"latest.json": "2026-09-07.json", "latest.outcomes.json": "2026-09-06.outcomes.json",
                                    "forecasts.json": None, "prophecy.json": None, "grades": None, "status.json": None})
            self.assertFalse((data / "forecasts.json").exists())
            self.assertEqual(json.loads((data / "latest.json").read_text()), {"day": "b"})
            self.assertEqual(json.loads((data / "latest.outcomes.json").read_text()), {"day": "c"})
            self.assertTrue((out / "index.html").exists() and (data / "2026-09-07.json").exists())
            self.assertEqual((out / "img" / "banner.webp").read_bytes(), b"image")  # folders are copied too
            (tmp / "site_forecasts.json").write_text(json.dumps({"rows": []}))
            used = build(out, predictions, web, forecasts=tmp / "site_forecasts.json",  # again, over the last build
                         prophecies=tmp / "none", statuses=tmp / "none", grades=tmp / "none.json")
            self.assertEqual(json.loads((data / "forecasts.json").read_text()), {"rows": []})
            self.assertEqual(used["forecasts.json"], str(tmp / "site_forecasts.json"))
            self.assertFalse(list(out.rglob("*.parquet")))
            self.assertIsNone(newest(web, ".json"))

    def test_version_3_s_prophecy_and_the_days_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            web, v3, out = tmp / "web", tmp / "v3", tmp / "out"
            web.mkdir(), v3.mkdir()
            (web / "index.html").write_text("<html></html>")
            for day in ("2026-10-03", "2026-10-04"):
                (v3 / f"{day}.prophecy.json").write_text(json.dumps({"date": day, "predictions": []}))
                (v3 / f"{day}.json").write_text(json.dumps({"date": day, "secret": "the whole record"}))
            used = build(out, tmp / "none", web, forecasts=tmp / "none.json", prophecies=v3, statuses=tmp / "none",
                         grades=tmp / "none.json")
            data = out / "data"
            self.assertEqual(used["prophecy.json"], "2026-10-04.prophecy.json")
            self.assertEqual(json.loads((data / "prophecy.json").read_text())["date"], "2026-10-04")
            self.assertEqual([d["date"] for d in json.loads((data / "prophecies.json").read_text())],
                             ["2026-10-04", "2026-10-03"])  # newest first, for the day selector
            self.assertEqual(sorted(p.name for p in (data / "prophecies").iterdir()), ["2026-10-03.json", "2026-10-04.json"])
            # Only the public part goes out: a day's whole record names what it cites, titles that can name a person.
            self.assertNotIn("the whole record", "".join(p.read_text() for p in out.rglob("*.json")))

    def test_each_graded_prediction_carries_its_grade(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            web, v3, out = tmp / "web", tmp / "v3", tmp / "out"
            web.mkdir(), v3.mkdir()
            (web / "index.html").write_text("<html></html>")
            predictions = [{"text": "I predict that Japan wins.", "due": "2026-10-11"},
                           {"text": "I predict that Crew-12 lands.", "due": "2026-10-11"},
                           {"text": "I predict that it rains.", "due": "2026-10-12"}]
            (v3 / "2026-10-05.prophecy.json").write_text(json.dumps({"date": "2026-10-05", "predictions": predictions}))
            grades = {  # as public_grades.py writes them; a line the guardrails dropped has no "line"
                "2026-10-05#3": {"date": "2026-10-05", "text": "I predict that Japan wins.", "outcome": "happened",
                                 "already_known": False, "vague": True, "line": "Japan won.",
                                 "check": {"line": "Japan won.", "dropped_because": []}},
                "2026-10-05#5": {"date": "2026-10-05", "text": "I predict that Crew-12 lands.", "outcome": "not possible",
                                 "already_known": True, "vague": False,
                                 "check": {"line": "The commander said so.", "dropped_because": ["names or points to a person"]}},
                "2026-10-06#0": {"date": "2026-10-06", "text": "I predict that it rains.", "outcome": "did not happen",
                                 "already_known": False, "vague": False, "line": "Another day's."}}
            (tmp / "public.json").write_text(json.dumps(grades))
            build(out, tmp / "none", web, forecasts=tmp / "none.json", prophecies=v3, statuses=tmp / "none",
                  grades=tmp / "public.json")
            for name in ("prophecy.json", "prophecies/2026-10-05.json"):
                shown = json.loads((out / "data" / name).read_text())["predictions"]
                self.assertEqual(shown[0]["grade"], {"outcome": "happened", "already_known": False, "vague": True,
                                                     "line": "Japan won."})
                self.assertEqual(shown[1]["grade"], {"outcome": "not possible", "already_known": True, "vague": False})
                self.assertNotIn("grade", shown[2])  # the same words on another day aren't this one's
            self.assertEqual(json.loads((out / "data" / "prophecies.json").read_text())[0]["predictions"][0]["grade"]
                             ["outcome"], "happened")
            # Only what the site shows: never a dropped line, nor the checks.
            published = "".join(p.read_text() for p in out.rglob("*.json"))
            self.assertNotIn("The commander", published)
            self.assertNotIn("dropped_because", published)

    def test_the_newest_day_s_run_goes_out_for_the_watchdog(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            web, logs, out = tmp / "web", tmp / "logs", tmp / "out"
            web.mkdir(), logs.mkdir()
            (web / "index.html").write_text("<html></html>")
            for day in ("2026-10-05", "2026-10-06"):
                (logs / f"{day}.status.json").write_text(json.dumps({"day": day, "steps": {}}))
            (logs / "2026-10-06.log").write_text("Traceback (most recent call last): File C:/Users/someone/...")
            used = build(out, tmp / "none", web, forecasts=tmp / "none.json", prophecies=tmp / "none", statuses=logs,
                         grades=tmp / "none.json")
            self.assertEqual(used["status.json"], "2026-10-06.status.json")
            self.assertEqual(json.loads((out / "data" / "status.json").read_text())["day"], "2026-10-06")
            self.assertNotIn("Traceback", "".join(p.read_text() for p in out.rglob("*.*")))  # the logs stay here


if __name__ == "__main__":
    unittest.main()
