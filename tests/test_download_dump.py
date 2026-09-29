import http.server
import tempfile
import threading
import unittest
from pathlib import Path

from scripts.download_dump import download

PAYLOAD = b"x" * 5000


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        # "/short" claims more bytes than it sends, like a dropped connection.
        self.send_header("Content-Length", str(len(PAYLOAD) + (100 if self.path == "/short" else 0)))
        self.end_headers()
        self.wfile.write(PAYLOAD)

    def log_message(self, *args):
        pass


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def test_complete_download_is_renamed_into_place(self):
        dest = Path(self.tmp.name) / "file.bz2"
        download(f"{self.base}/ok", dest, progress=False)
        self.assertEqual(dest.read_bytes(), PAYLOAD)
        self.assertFalse(dest.with_name("file.bz2.part").exists())

    def test_truncated_download_never_looks_finished(self):
        dest = Path(self.tmp.name) / "file.bz2"
        with self.assertRaises(Exception):
            download(f"{self.base}/short", dest, progress=False)
        self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
