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
                         statuses=tmp / "none")
            data = out / "data"
            self.assertEqual(used, {"latest.json": "2026-09-07.json", "latest.outcomes.json": "2026-09-06.outcomes.json",
                                    "forecasts.json": None, "prophecy.json": None, "status.json": None})
            self.assertFalse((data / "forecasts.json").exists())
            self.assertEqual(json.loads((data / "latest.json").read_text()), {"day": "b"})
            self.assertEqual(json.loads((data / "latest.outcomes.json").read_text()), {"day": "c"})
            self.assertTrue((out / "index.html").exists() and (data / "2026-09-07.json").exists())
            self.assertEqual((out / "img" / "banner.webp").read_bytes(), b"image")  # folders are copied too
            (tmp / "site_forecasts.json").write_text(json.dumps({"rows": []}))
            used = build(out, predictions, web, forecasts=tmp / "site_forecasts.json",  # again, over the last build
                         prophecies=tmp / "none", statuses=tmp / "none")
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
            used = build(out, tmp / "none", web, forecasts=tmp / "none.json", prophecies=v3, statuses=tmp / "none")
            data = out / "data"
            self.assertEqual(used["prophecy.json"], "2026-10-04.prophecy.json")
            self.assertEqual(json.loads((data / "prophecy.json").read_text())["date"], "2026-10-04")
            self.assertEqual([d["date"] for d in json.loads((data / "prophecies.json").read_text())],
                             ["2026-10-04", "2026-10-03"])  # newest first, for the day selector
            self.assertEqual(sorted(p.name for p in (data / "prophecies").iterdir()), ["2026-10-03.json", "2026-10-04.json"])
            # Only the public part goes out: a day's whole record names what it cites, titles that can name a person.
            self.assertNotIn("the whole record", "".join(p.read_text() for p in out.rglob("*.json")))

    def test_the_newest_day_s_run_goes_out_for_the_watchdog(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            web, logs, out = tmp / "web", tmp / "logs", tmp / "out"
            web.mkdir(), logs.mkdir()
            (web / "index.html").write_text("<html></html>")
            for day in ("2026-10-05", "2026-10-06"):
                (logs / f"{day}.status.json").write_text(json.dumps({"day": day, "steps": {}}))
            (logs / "2026-10-06.log").write_text("Traceback (most recent call last): File C:/Users/someone/...")
            used = build(out, tmp / "none", web, forecasts=tmp / "none.json", prophecies=tmp / "none", statuses=logs)
            self.assertEqual(used["status.json"], "2026-10-06.status.json")
            self.assertEqual(json.loads((out / "data" / "status.json").read_text())["day"], "2026-10-06")
            self.assertNotIn("Traceback", "".join(p.read_text() for p in out.rglob("*.*")))  # the logs stay here


if __name__ == "__main__":
    unittest.main()
