import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from scripts.fetch_due_pages import due_targets, to_fetch

D = date(2026, 10, 7)


class DueTargetsTest(unittest.TestCase):
    def test_the_pages_to_fetch_for_grading(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = {"date": "2026-10-07", "predictions": [
                {"evidence": ["Kosovo"], "due": "2026-10-07", "published": True},  # due on its day
                {"evidence": ["Judo"], "due": "2026-10-11", "published": True},
                {"evidence": ["Unpublished"], "due": "2026-10-08"},
                {"evidence": ["Current events: A story"], "due": "2026-10-08", "published": True},  # graded on events
                {"evidence": ["Later"], "due": "2026-10-14", "published": True}]}  # not due by the last day
            (Path(tmp) / "2026-10-07.json").write_text(json.dumps(record))
            every = due_targets(Path(tmp), date(2026, 10, 11))
            self.assertEqual(every, [("Judo", D, date(2026, 10, 11)), ("Kosovo", D, D),
                                     ("Unpublished", D, date(2026, 10, 8))])
            published = due_targets(Path(tmp), date(2026, 10, 11), published=True)
            self.assertEqual([t[0] for t in published], ["Judo", "Kosovo"])
        # A development day's change on D is in the examples already; a live day's isn't. Fetched rows aren't again.
        self.assertEqual(to_fetch(published, {("Kosovo", D)}, set()), [("Judo", D, date(2026, 10, 11))])
        self.assertEqual(to_fetch(published, set(), {("Judo", D, date(2026, 10, 11))}), [("Kosovo", D, D)])


if __name__ == "__main__":
    unittest.main()
