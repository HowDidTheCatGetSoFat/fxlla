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
        elif self.path == "/redir":
            self.send_response(302)
            self.send_header("Location", SEEN["other"] + "/target")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/target":
            SEEN["target_auth"] = self.headers.get("Authorization")
            self._send(200, "{}")
        elif self.path == "/nan":
            self._send(200, '{"a":NaN,"b":Infinity,"c":1e999,"d":2}')
        elif self.path == "/drip":
            self.send_response(200)
            self.send_header("Content-Length", "100000")
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.1)
            except OSError:
                pass
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

    def test_redirect_not_followed_and_no_key_leak(self):
        other = ThreadingHTTPServer(("127.0.0.1", 0), H)
        other.daemon_threads = True
        threading.Thread(target=other.serve_forever, daemon=True).start()
        SEEN["other"] = "http://127.0.0.1:%d" % other.server_address[1]
        SEEN.pop("target_auth", None)
        try:
            with self.assertRaises(ext_http.ProbeError) as c:
                ext_http.get_json(self.base + "/redir",
                                  {"Authorization": "Bearer " + KEY})
            self.assertEqual(c.exception.reason, "http 302")
            self.assertNotIn("target_auth", SEEN)
        finally:
            other.shutdown()
            other.server_close()

    def test_env_proxy_ignored(self):
        old = {k: os.environ.get(k) for k in ("http_proxy", "HTTP_PROXY", "no_proxy", "NO_PROXY")}
        try:
            os.environ["http_proxy"] = os.environ["HTTP_PROXY"] = "http://127.0.0.1:9"
            os.environ.pop("no_proxy", None)
            os.environ.pop("NO_PROXY", None)
            self.assertEqual(ext_http.get_json(self.base + "/ok"), {"a": 1})
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_non_finite_constants_become_none(self):
        r = ext_http.get_json(self.base + "/nan")
        self.assertIsNone(r["a"])
        self.assertIsNone(r["b"])
        self.assertIsNone(r["c"])
        self.assertEqual(r["d"], 2)

    def test_slow_drip_hits_total_deadline(self):
        t0 = time.time()
        with self.assertRaises(ext_http.ProbeError) as c:
            ext_http.get_json(self.base + "/drip", timeout=0.4)
        self.assertEqual(c.exception.reason, "timeout")
        self.assertLess(time.time() - t0, 1.2)

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
