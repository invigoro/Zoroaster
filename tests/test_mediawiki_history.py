import bz2
import tempfile
import unittest
from pathlib import Path

from src.ingest.mediawiki_history import COLUMNS, iter_revisions, parse_line


def line(**values: str) -> str:
    """A 78-field TSV line with the given columns set and every other field empty."""
    fields = {"wiki_db": "enwiki", "event_entity": "revision", "event_type": "create",
              "event_timestamp": "2024-03-05 17:04:09.0", "page_namespace_historical": "0",
              "page_id": "42", "revision_id": "1000"} | values
    unknown = set(fields) - set(COLUMNS)
    assert not unknown, unknown
    return "\t".join(fields.get(c, "") for c in COLUMNS) + "\n"


class ParseLineTest(unittest.TestCase):
    def test_mainspace_revision(self):
        r = parse_line(line(
            page_title="Ariel_Pink", revision_parent_id="999", event_user_text_historical="Alice",
            event_user_is_bot_by_historical="", revision_text_bytes="91168", revision_text_sha1="abc123",
            page_creation_timestamp="2011-02-13 00:41:02.0", revision_is_identity_revert="true",
            revision_is_identity_reverted="false", event_user_is_temporary="false",
        ))
        self.assertEqual(r, {
            "page_id": 42, "page_title": "Ariel_Pink", "revision_id": 1000, "parent_id": 999,
            "timestamp": "2024-03-05T17:04:09Z", "user_text": "Alice", "is_anon": False, "is_bot": False,
            "byte_size": 91168, "sha1": "abc123", "page_created": "2011-02-13T00:41:02Z",
            "mwh_is_revert": True, "mwh_is_reverted": False,
        })

    def test_flags_and_empty_fields(self):
        bot = parse_line(line(event_user_is_bot_by_historical="name,group"))
        self.assertTrue(bot["is_bot"])
        self.assertEqual((bot["parent_id"], bot["byte_size"], bot["sha1"], bot["user_text"]), (None, None, None, None))
        self.assertTrue(parse_line(line(event_user_is_temporary="true"))["is_anon"])  # temporary accounts
        self.assertTrue(parse_line(line(event_user_is_anonymous="true"))["is_anon"])  # IP editors

    def test_other_events_are_skipped(self):
        self.assertIsNone(parse_line(line(event_entity="page")))
        self.assertIsNone(parse_line(line(event_type="delete")))
        self.assertIsNone(parse_line(line(page_namespace_historical="1")))
        self.assertIsNone(parse_line(line(revision_is_deleted_by_page_deletion="true")))

    def test_wrong_field_count_raises(self):
        with self.assertRaises(ValueError):
            parse_line("enwiki\trevision\tcreate\n")

    def test_iter_revisions_reads_bz2(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "month.tsv.bz2"
            with bz2.open(path, "wt", encoding="utf-8") as f:
                f.write(line(revision_id="1") + line(event_entity="user") + line(revision_id="2"))
            self.assertEqual([r["revision_id"] for r in iter_revisions(path)], [1, 2])


if __name__ == "__main__":
    unittest.main()
