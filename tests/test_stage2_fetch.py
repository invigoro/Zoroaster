import http.server
import json
import threading
import unittest
from urllib.parse import parse_qs, urlparse

from src.stage2.fetch import BATCH_SIZE, fetch_contents


class _FakeApi(http.server.BaseHTTPRequestHandler):
    """Canned MediaWiki API: text "text <id>" for every id except 13 (missing)
    and 14 (hidden). The first request gets a 429, and the first response to
    each batch is cut short with a continuation token."""

    calls: list[dict] = []
    throttled = False

    def do_GET(self):
        params = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        type(self).calls.append(params)
        if not type(self).throttled:
            type(self).throttled = True
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            return
        ids = [int(i) for i in params["revids"].split("|")]
        present = [i for i in ids if i != 13]
        half = present[: len(present) // 2] if "rvcontinue" not in params else present[len(present) // 2 :]
        revisions = [{"revid": i, "slots": {"main": {"texthidden": True} if i == 14 else {"content": f"text {i}"}}}
                     for i in half]
        payload = {"query": {"pages": [{"pageid": 1, "revisions": revisions}]}}
        if "rvcontinue" not in params:
            payload["continue"] = {"rvcontinue": "x", "continue": "||"}
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class FetchContentsTest(unittest.TestCase):
    def test_batches_continuation_retry_and_missing(self):
        _FakeApi.calls, _FakeApi.throttled = [], False
        server = http.server.HTTPServer(("127.0.0.1", 0), _FakeApi)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            ids = list(range(1, BATCH_SIZE + 11))  # two batches
            got = list(fetch_contents(ids, pause=0, api_url=f"http://127.0.0.1:{server.server_port}/w/api.php"))
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual([i for i, _ in got], ids)  # input order
        self.assertEqual(dict(got)[1], "text 1")
        self.assertIsNone(dict(got)[13])  # missing
        self.assertIsNone(dict(got)[14])  # hidden
        self.assertTrue(all(text == f"text {i}" for i, text in got if i not in (13, 14)))
        batches = [c["revids"].count("|") + 1 for c in _FakeApi.calls if "rvcontinue" not in c]
        self.assertEqual(batches, [BATCH_SIZE, BATCH_SIZE, 10])  # the 429'd request, its retry, batch 2
        self.assertTrue(all(c["maxlag"] == "5" for c in _FakeApi.calls))


if __name__ == "__main__":
    unittest.main()
