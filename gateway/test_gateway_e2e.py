"""End-to-end: a request through the gateway is proxied and recorded.

Stands up a fake backend that streams a small SSE completion, points a
pre-loaded gateway backend at it, drives one request through the real
gateway Handler, and asserts a passive sample lands in the stats file.
No real model is spawned.
"""
import http.client
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


USAGE_ONLY = b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
DELTA = b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'


class _Backend(BaseHTTPRequestHandler):
    last_body = None
    mode = "ok"              # "ok", "http500", "truncate", or "slow"
    release = None           # threading.Event the "slow" mode waits on

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).last_body = body
        mode = type(self).mode
        if mode == "http500":
            data = b'{"error":{"message":"boom"}}'
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        out = SSE
        if mode == "nofinish":
            out = (b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
                   b'data: [DONE]\n\n')
        if (body.get("stream_options") or {}).get("include_usage"):
            out = out.replace(b"data: [DONE]", USAGE_ONLY + b"data: [DONE]")
        type(self).sent = out
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        if mode != "slow":
            self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        if mode == "truncate":
            self.wfile.write(out[:len(out) // 2])
            self.wfile.flush()
            self.close_connection = True
            return
        if mode == "slow":
            self.close_connection = True
            # enough for the gateway's 1024-byte read to return
            self.wfile.write(DELTA * 40)
            self.wfile.flush()
            type(self).release.wait(10)
            try:
                for _ in range(300):
                    self.wfile.write(DELTA)
                    self.wfile.flush()
                    time.sleep(0.01)
            except OSError:
                pass
            return
        self.wfile.write(out)


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

        _Backend.mode = "ok"
        _Backend.last_body = None
        self._register("gguf")

        self.gateway = ThreadingHTTPServer(("127.0.0.1", 0), gw.Handler)
        self.gport = self.gateway.server_address[1]
        self.gt = threading.Thread(target=self.gateway.serve_forever, daemon=True)
        self.gt.start()

        self.stats = os.path.join(_STORE, "stats.jsonl")
        if os.path.exists(self.stats):
            os.remove(self.stats)
        os.environ["FXLLA_STATS_FILE"] = self.stats

    def _register(self, engine):
        # Pre-load a backend so MANAGER.ensure() returns instantly, no spawn.
        gw.MANAGER.backends["fake"] = gw.Backend(
            "fake", self.bport, _DummyProc(), 10, "fake", engine)

    def _post(self, **extra):
        body = {"model": "fake", "stream": True,
                "messages": [{"role": "user", "content": "hi"}]}
        body.update(extra)
        return urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % self.gport,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})

    def _last_row(self, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.exists(self.stats):
                with open(self.stats, encoding="utf-8") as f:
                    lines = [l for l in f.read().splitlines() if l.strip()]
                if lines:
                    return json.loads(lines[-1])
            time.sleep(0.05)
        self.fail("no row in stats file")

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

    def test_injects_usage_and_hides_it_from_a_client_that_did_not_ask(self):
        with urllib.request.urlopen(self._post(), timeout=10) as r:
            out = r.read()
        self.assertTrue(_Backend.last_body["stream_options"]["include_usage"])
        self.assertNotIn(b'"usage"', out)
        self.assertIn(b"[DONE]", out)
        row = self._last_row()
        self.assertEqual(row["output_tokens"], 2)
        self.assertEqual(row["prompt_tokens"], 5)
        self.assertIs(row["output_estimated"], False)
        self.assertEqual(row["finish"], "stop")
        self.assertIs(row["stream"], True)
        self.assertEqual(row["client_app"], "python")
        self.assertEqual(len(row["id"]), 8)

    def test_client_that_asked_gets_its_usage_untouched(self):
        req = self._post(stream_options={"include_usage": True})
        with urllib.request.urlopen(req, timeout=10) as r:
            out = r.read()
        self.assertEqual(out, _Backend.sent)
        self.assertIn(b'"usage"', out)
        self.assertEqual(self._last_row()["output_tokens"], 2)

    def test_mtplx_is_never_injected(self):
        self._register("mtplx")
        with urllib.request.urlopen(self._post(), timeout=10) as r:
            r.read()
        self.assertNotIn("stream_options", _Backend.last_body)
        row = self._last_row()
        self.assertIs(row["output_estimated"], True)
        self.assertEqual(row["output_tokens"], 2)

    def test_upstream_500_is_recorded_as_error(self):
        _Backend.mode = "http500"
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(self._post(), timeout=10)
        cm.exception.close()
        self.assertEqual(cm.exception.code, 500)
        row = self._last_row()
        self.assertEqual(row["finish"], "error")
        self.assertEqual(row["output_tokens"], 0)

    def test_truncated_stream_is_recorded_as_error(self):
        _Backend.mode = "truncate"
        conn = http.client.HTTPConnection("127.0.0.1", self.gport, timeout=10)
        conn.request("POST", "/v1/chat/completions", json.dumps(
            {"model": "fake", "stream": True,
             "messages": [{"role": "user", "content": "hi"}]}),
            {"Content-Type": "application/json"})
        resp = conn.getresponse()
        got = b""
        try:
            got = resp.read()
        except http.client.IncompleteRead as e:
            got = e.partial
        conn.close()
        self.assertIn(b"Hi", got)
        self.assertEqual(self._last_row()["finish"], "error")

    def test_client_disconnect_is_recorded_as_client_gone(self):
        _Backend.mode = "slow"
        _Backend.release = threading.Event()
        conn = http.client.HTTPConnection("127.0.0.1", self.gport, timeout=10)
        conn.request("POST", "/v1/chat/completions", json.dumps(
            {"model": "fake", "stream": True,
             "messages": [{"role": "user", "content": "hi"}]}),
            {"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read1(64)
        conn.close()
        _Backend.release.set()
        self.assertEqual(self._last_row(timeout=20)["finish"], "client_gone")

    def test_done_without_finish_reason_is_not_an_error(self):
        _Backend.mode = "nofinish"
        with urllib.request.urlopen(self._post(), timeout=10) as r:
            self.assertIn(b"[DONE]", r.read())
        self.assertIsNone(self._last_row()["finish"])

    def test_evicted_backend_goes_to_telemetry_only(self):
        real = gw.MANAGER.backend_meta
        calls = []

        def meta(alias):
            calls.append(alias)
            return real(alias) if len(calls) == 1 else None

        seen = []
        orig = gw.TELEMETRY.record
        gw.MANAGER.backend_meta = meta
        gw.TELEMETRY.record = lambda rec: (seen.append(dict(rec)), orig(rec))
        try:
            with urllib.request.urlopen(self._post(), timeout=10) as r:
                r.read()
            for _ in range(100):
                if seen:
                    break
                time.sleep(0.05)
        finally:
            del gw.MANAGER.backend_meta
            gw.TELEMETRY.record = orig
        self.assertEqual(len(seen), 1)
        self.assertIsNone(seen[0]["ram_mb"])
        self.assertEqual(seen[0]["model"], "fake")
        self.assertEqual(seen[0]["output_tokens"], 2)
        self.assertFalse(os.path.exists(self.stats))

    def test_old_fields_keep_their_meaning(self):
        with urllib.request.urlopen(self._post(), timeout=10) as r:
            r.read()
        row = self._last_row()
        self.assertEqual(row["source"], "gateway")
        self.assertIn("tps", row)
        self.assertIn("ttft_ms", row)
        self.assertEqual(row["model"], "fake")
        self.assertIsInstance(row["ts"], int)

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
