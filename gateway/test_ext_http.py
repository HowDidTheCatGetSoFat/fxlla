import json
import os
import socket
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ext_http  # noqa: E402

KEY = "test-key-not-real"
SEEN = {}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        SEEN["auth"] = self.headers.get("Authorization")
        if self.path == "/ok":
            self._send(200, '{"a":1}')
        elif self.path == "/401":
            self._send(401, "no")
        elif self.path == "/junk":
            self._send(200, "not json")
        elif self.path == "/slow":
            time.sleep(2)
            try:
                self._send(200, "{}")
            except OSError:
                pass


class ExtHttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        cls.srv.daemon_threads = True
        cls.base = "http://127.0.0.1:%d" % cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_ok(self):
        self.assertEqual(ext_http.get_json(self.base + "/ok"), {"a": 1})

    def test_http_error(self):
        with self.assertRaises(ext_http.ProbeError) as c:
            ext_http.get_json(self.base + "/401")
        self.assertEqual(c.exception.reason, "http 401")
        self.assertEqual(str(c.exception), "http 401")

    def test_bad_json(self):
        with self.assertRaises(ext_http.ProbeError) as c:
            ext_http.get_json(self.base + "/junk")
        self.assertEqual(c.exception.reason, "bad json")

    def test_timeout(self):
        t0 = time.time()
        with self.assertRaises(ext_http.ProbeError) as c:
            ext_http.get_json(self.base + "/slow", timeout=0.2)
        self.assertEqual(c.exception.reason, "timeout")
        self.assertLess(time.time() - t0, 1.0)

    def test_refused(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        with self.assertRaises(ext_http.Refused):
            ext_http.get_json("http://127.0.0.1:%d/x" % port)

    def test_headers_sent(self):
        ext_http.get_json(self.base + "/ok", {"Authorization": "Bearer " + KEY})
        self.assertEqual(SEEN["auth"], "Bearer " + KEY)

    def test_no_secret_in_error(self):
        hdr = {"Authorization": "Bearer " + KEY}
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        urls = [self.base + "/401", self.base + "/junk",
                self.base + "/slow", "http://127.0.0.1:%d/x" % port]
        for u in urls:
            try:
                ext_http.get_json(u, hdr, timeout=0.2)
            except Exception as e:
                self.assertNotIn(KEY, str(e))
                self.assertNotIn(u, str(e))
                self.assertNotIn("127.0.0.1", str(e))
            else:
                self.fail("no error for " + u)


if __name__ == "__main__":
    unittest.main()
