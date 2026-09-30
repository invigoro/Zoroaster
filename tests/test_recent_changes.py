import http.server
import json
import threading
import unittest
from urllib.parse import parse_qs, urlparse

from scripts.fetch_recent_changes import day_files
from src.ingest.recent_changes import recent_changes, sha1_base36, to_record

EDIT = {"type": "edit", "ns": 0, "title": "2023–2024 Spanish protests", "pageid": 75263703, "revid": 1377674291,
        "old_revid": 1377674194, "rcid": 2074634203, "user": "~2026-48280-56", "temp": True, "bot": False, "new": False,
        "minor": False, "oldlen": 64093, "newlen": 63942, "timestamp": "2026-09-30T13:59:14Z", "tags": [],
        "sha1": "36a9dbbd8c0277542b72d027a87901cf7103272c"}


class Sha1Test(unittest.TestCase):
    def test_hex_to_the_dumps_base36(self):
        # revision 1372386579: the API's hex, and the history dump's value
        self.assertEqual(sha1_base36("36a9dbbd8c0277542b72d027a87901cf7103272c"), "6dvcwvvq9gzed4s09berqov8ayyek30")
        self.assertEqual(sha1_base36("0" * 39 + "1"), "0" * 30 + "1")  # zero-padded to 31 digits


class ToRecordTest(unittest.TestCase):
    def test_edit(self):
        self.assertEqual(to_record(EDIT), {
            "page_id": 75263703, "page_title": "2023–2024_Spanish_protests", "revision_id": 1377674291,
            "parent_id": 1377674194, "timestamp": "2026-09-30T13:59:14Z", "user_text": "~2026-48280-56",
            "is_anon": True, "is_bot": False, "byte_size": 63942, "sha1": "6dvcwvvq9gzed4s09berqov8ayyek30",
            "page_created": None,
        })

    def test_page_creation_hidden_fields_and_other_changes(self):
        new = to_record(EDIT | {"type": "new", "old_revid": 0, "oldlen": 0, "temp": False, "bot": True})
        self.assertEqual((new["parent_id"], new["page_created"], new["is_anon"], new["is_bot"]),
                         (0, "2026-09-30T13:59:14Z", False, True))
        hidden = to_record({k: v for k, v in EDIT.items() if k not in ("sha1", "user")} | {"sha1hidden": True, "userhidden": True})
        self.assertEqual((hidden["sha1"], hidden["user_text"]), (None, None))
        self.assertIsNone(to_record(EDIT | {"type": "log"}))
        self.assertIsNone(to_record(EDIT | {"ns": 1}))


class _FakeApi(http.server.BaseHTTPRequestHandler):
    """Two pages of recent changes joined by a continuation token. The last
    change is stamped exactly at the requested end, which the API includes."""

    calls: list[dict] = []

    def do_GET(self):
        params = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        type(self).calls.append(params)
        if "rccontinue" not in params:
            changes = [EDIT | {"revid": 1, "timestamp": "2026-09-29T00:00:00Z"},
                       EDIT | {"revid": 2, "type": "log", "timestamp": "2026-09-29T01:00:00Z"}]
            payload = {"continue": {"rccontinue": "20260929010000|2", "continue": "-||"}}
        else:
            changes = [EDIT | {"revid": 3, "timestamp": "2026-09-29T23:59:59Z"},
                       EDIT | {"revid": 4, "timestamp": params["rcend"]}]
            payload = {}
        body = json.dumps(payload | {"query": {"recentchanges": changes}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class RecentChangesTest(unittest.TestCase):
    def test_follows_continuation_and_keeps_the_window_half_open(self):
        server = http.server.HTTPServer(("127.0.0.1", 0), _FakeApi)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/w/api.php"
            records = list(recent_changes("2026-09-29T00:00:00Z", "2026-09-30T00:00:00Z", pause=0, api_url=url))
        finally:
            server.shutdown()
        self.assertEqual([r["revision_id"] for r in records], [1, 3])  # the log entry and the end-stamped edit are dropped
        first, second = _FakeApi.calls
        self.assertEqual((first["rcdir"], first["rcnamespace"], first["rcstart"]), ("newer", "0", "2026-09-29T00:00:00Z"))
        self.assertEqual(second["rccontinue"], "20260929010000|2")


class DayFilesTest(unittest.TestCase):
    def test_days_without_a_file_oldest_first(self):
        import tempfile
        from datetime import date
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "2026-09-02.parquet").touch()
            todo = day_files(date(2026, 9, 1), date(2026, 9, 4), Path(tmp))
        self.assertEqual([d.isoformat() for d, _ in todo], ["2026-09-01", "2026-09-03", "2026-09-04"])
        self.assertEqual(todo[0][1].name, "2026-09-01.parquet")


if __name__ == "__main__":
    unittest.main()
