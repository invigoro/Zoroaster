import bz2
import tempfile
import unittest
from pathlib import Path

from scripts.build_stage2_neighbor_changes import SNIPPET_CHARS, snippet
from scripts.build_stage2_titles import scan_titles
from src.ingest.mediawiki_history import COLUMNS


def _line(**values: str) -> str:
    fields = {"wiki_db": "enwiki", "event_entity": "revision", "event_type": "create",
              "page_namespace_historical": "0"} | values
    return "\t".join(fields.get(c, "") for c in COLUMNS) + "\n"


class ScanTitlesTest(unittest.TestCase):
    def test_title_then_for_wanted_revisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "month.tsv.bz2"
            with bz2.open(path, "wt", encoding="utf-8") as f:
                f.write(_line(revision_id="1", page_title="Killing_of_X", page_title_historical="X_shooting"))
                f.write(_line(revision_id="2", page_title="B", page_title_historical="B"))
                f.write(_line(event_entity="page", event_type="move", page_title="C", page_title_historical="D"))
                f.write(_line(revision_id="3", page_title="E", page_title_historical="E"))
            found = scan_titles(path, {1, 3, 4})
        self.assertEqual(found, {1: ("Killing_of_X", "X_shooting"), 3: ("E", "E")})


class SnippetTest(unittest.TestCase):
    def test_new_prose_beats_longer_markup_and_moved_text(self):
        moved = "An older paragraph that was moved further down the page, long enough to be a span."
        start = f"Intro.\n{moved}\nOutro.\n"
        end = ("Intro.\nOutro.\n{{Infobox event|name=A long infobox|date=2024|location=Somewhere far away}}\n"
               "The rally was halted when shots were fired.<ref>{{cite news|title=Shots at rally|url=http://x.org}}</ref>\n"
               f"{moved}\n")
        text, inserted, spans = snippet(start, end)
        self.assertEqual(text, "The rally was halted when shots were fired.")
        self.assertGreater(inserted, len(text))
        self.assertFalse(any(moved in s for s in spans))

    def test_page_created_that_day_gives_its_lead(self):
        body = "{{Infobox event}}\n'''The attempt''' took place on July 13, 2024," + " and then more" * 20
        text, _, spans = snippet(None, body)
        self.assertTrue(text.startswith("The attempt took place on July 13, 2024, and then more"))
        self.assertTrue(text.endswith(" …") and len(text) <= SNIPPET_CHARS + 2)
        self.assertEqual(spans, [body])

    def test_markup_only_change_has_no_snippet(self):
        self.assertIsNone(snippet("A.\n", "A.\n{{cite web|title=Something long enough to count|url=http://x}}\n")[0])
        self.assertEqual(snippet("Same text.\n", "Same text.\n"), (None, 0, []))


if __name__ == "__main__":
    unittest.main()
