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
            for name, day in (("2026-09-06.json", "a"), ("2026-09-07.json", "b"), ("2026-09-06.outcomes.json", "c")):
                (predictions / name).write_text(json.dumps({"day": day}))
            (predictions / "2026-09-07.parquet").write_text("not copied")
            self.assertEqual(newest(predictions, ".json").name, "2026-09-07.json")  # not an .outcomes.json
            used = build(out, predictions, web)
            data = out / "data"
            self.assertEqual(used, {"latest.json": "2026-09-07.json", "latest.outcomes.json": "2026-09-06.outcomes.json"})
            self.assertEqual(json.loads((data / "latest.json").read_text()), {"day": "b"})
            self.assertEqual(json.loads((data / "latest.outcomes.json").read_text()), {"day": "c"})
            self.assertTrue((out / "index.html").exists() and (data / "2026-09-07.json").exists())
            self.assertFalse(list(out.rglob("*.parquet")))
            self.assertIsNone(newest(web, ".json"))


if __name__ == "__main__":
    unittest.main()
