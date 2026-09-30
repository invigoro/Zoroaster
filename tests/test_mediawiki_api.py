import http.server
import json
import threading
import unittest

import requests

from src.mediawiki_api import api_get


class _Flaky(http.server.BaseHTTPRequestHandler):
    """Drops the first connection without a response, then answers."""

    calls = 0

    def do_GET(self):
        type(self).calls += 1
        if type(self).calls == 1:
            self.close_connection = True  # no status line: the client sees RemoteDisconnected
            return
        body = json.dumps({"query": {"ok": True}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class ApiGetTest(unittest.TestCase):
    def test_dropped_connection_is_retried(self):
        server = http.server.HTTPServer(("127.0.0.1", 0), _Flaky)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            payload = api_get(requests.Session(), f"http://127.0.0.1:{server.server_port}/w/api.php", {}, pause=0)
        finally:
            server.shutdown()
        self.assertEqual(payload, {"query": {"ok": True}})
        self.assertEqual(_Flaky.calls, 2)


if __name__ == "__main__":
    unittest.main()
