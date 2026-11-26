"""End-to-end: a request through the gateway is proxied and recorded.

Stands up a fake backend that streams a small SSE completion, points a
pre-loaded gateway backend at it, drives one request through the real
gateway Handler, and asserts a passive sample lands in the stats file.
No real model is spawned.
"""
import importlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_STORE = tempfile.mkdtemp(prefix="fxlla-e2e-")
os.environ["FXLLA_STORE"] = _STORE
os.makedirs(os.path.join(_STORE, "models", "fake"), exist_ok=True)

gw = importlib.import_module("fxlla_gateway")

SSE = (
    b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n'
    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    b'data: [DONE]\n\n'
)


class _Backend(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(SSE)))
        self.end_headers()
        self.wfile.write(SSE)


class _DummyProc:
    pid = os.getpid()

    def poll(self):
        return None   # still running; ensure() checks before reusing it

    def terminate(self):
        pass


class TestGatewayE2E(unittest.TestCase):
    def setUp(self):
        self.backend = ThreadingHTTPServer(("127.0.0.1", 0), _Backend)
        self.bport = self.backend.server_address[1]
        self.bt = threading.Thread(target=self.backend.serve_forever, daemon=True)
        self.bt.start()

        # Pre-load a backend so MANAGER.ensure() returns instantly, no spawn.
        gw.MANAGER.backends["fake"] = gw.Backend(
            "fake", self.bport, _DummyProc(), 10, "fake", "gguf")

        self.gateway = ThreadingHTTPServer(("127.0.0.1", 0), gw.Handler)
        self.gport = self.gateway.server_address[1]
        self.gt = threading.Thread(target=self.gateway.serve_forever, daemon=True)
        self.gt.start()

        self.stats = os.path.join(_STORE, "stats.jsonl")
        os.environ["FXLLA_STATS_FILE"] = self.stats

    def tearDown(self):
        self.gateway.shutdown()
        self.backend.shutdown()
        gw.MANAGER.backends.pop("fake", None)
        os.environ.pop("FXLLA_STATS_FILE", None)

    def test_request_is_proxied_and_recorded(self):
        body = json.dumps({"model": "fake", "stream": True,
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % self.gport,
            data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            out = r.read()
        # the SSE content is proxied through unchanged
        self.assertIn(b"Hi", out)
        self.assertIn(b"[DONE]", out)

        # recording happens in the handler thread after the last chunk flushes,
        # so the client can see EOF slightly before the sample is written
        for _ in range(50):
            if os.path.exists(self.stats):
                break
            time.sleep(0.05)
        with open(self.stats, encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        self.assertEqual(len(lines), 1)
        s = json.loads(lines[0])
        self.assertEqual(s["model"], "fake")
        self.assertEqual(s["engine"], "gguf")
        self.assertEqual(s["source"], "gateway")
        self.assertGreater(s["ram_mb"], 0)   # real RSS of this test process
        self.assertIsNotNone(s["ttft_ms"])

    def test_admin_unload_frees_backends(self):
        req = urllib.request.Request(
            "http://127.0.0.1:%d/admin/unload" % self.gport,
            data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            info = json.loads(r.read())
        self.assertIn("fake", info["unloaded"])
        self.assertEqual(gw.MANAGER.status(), [])


if __name__ == "__main__":
    unittest.main()
