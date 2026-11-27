import datetime
import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ext_local  # noqa: E402
from ext_http import ProbeError, Refused  # noqa: E402

FX = Path(__file__).parent / "fixtures"
D27 = datetime.date(2026, 11, 27)
LINE = '[GIN] 2026/11/27 - 02:00:00 | 200 |   10ms | 127.0.0.1 | POST "/api/chat"\n'


def load(name):
    return json.loads((FX / name).read_text())


class GinTests(unittest.TestCase):
    def test_parse_gin_units(self):
        p = ext_local.parse_gin
        self.assertEqual(p('[GIN] 2026/11/27 - 01:24:17 | 200 |   5.74ms | 1.2.3.4 | POST "/api/chat"'),
                         (D27, 5.74, "/api/chat"))
        self.assertEqual(p('[GIN] 2026/11/27 - 01:24:17 | 200 | 850\u00b5s | 1.2.3.4 | POST "/api/chat"')[1], 0.85)
        self.assertEqual(p('[GIN] 2026/11/27 - 01:24:17 | 200 | 850us | 1.2.3.4 | POST "/api/chat"')[1], 0.85)
        self.assertEqual(p('[GIN] 2026/11/27 - 01:24:17 | 200 | 2.5s | 1.2.3.4 | POST "/api/chat"')[1], 2500.0)
        self.assertEqual(p('[GIN] 2026/11/27 - 01:24:17 | 200 | 1m2.5s | 1.2.3.4 | POST "/api/chat"')[1], 62500.0)
        self.assertEqual(p('[GIN] 2026/11/27 - 01:24:17 | 200 | 500ns | 1.2.3.4 | POST "/api/chat"')[1], 0.0005)
        self.assertEqual(p('[GIN] 2026/11/27 - 01:24:17 | 200 | 1ms | 1.2.3.4 | POST "/api/embed?a=b"')[2], "/api/embed")

    def test_parse_gin_non_gin(self):
        self.assertIsNone(ext_local.parse_gin("time=... level=INFO msg=x"))
        self.assertIsNone(ext_local.parse_gin(""))


class LogTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.path = Path(self.d.name) / "server.log"
        self.today = D27

    def log(self):
        return ext_local.OllamaLog(self.path, today=lambda: self.today)

    def test_log_counts_today_gen_only(self):
        self.path.write_bytes((FX / "ollama-server.log").read_bytes())
        s = self.log().today_stats()
        self.assertEqual(s, {"requests": 4, "median_ms": 1310.0})

    def test_first_read_is_bounded_and_drops_partial_line(self):
        old = ext_local.MAX_READ
        ext_local.MAX_READ = 4096
        try:
            self.path.write_bytes(b"x" * 100000 + b"\n" + LINE.encode() * 3)
            lg = self.log()
            s = lg.today_stats()
            self.assertEqual(s["requests"], 3)
            self.assertLessEqual(len(lg._pending), ext_local.MAX_PENDING)
        finally:
            ext_local.MAX_READ = old

    def test_incremental_read_is_chunked(self):
        old = ext_local.MAX_READ
        ext_local.MAX_READ = 200
        try:
            self.path.write_bytes(LINE.encode())
            lg = self.log()
            lg.today_stats()
            with open(self.path, "ab") as fh:
                fh.write(LINE.encode() * 9)
            for _ in range(20):
                s = lg.today_stats()
            self.assertEqual(s["requests"], 10)
        finally:
            ext_local.MAX_READ = old

    def test_newline_less_file_caps_pending(self):
        self.path.write_bytes(b"\x00" * 500000)
        lg = self.log()
        for _ in range(10):
            lg.today_stats()
        self.assertLessEqual(len(lg._pending), ext_local.MAX_PENDING)

    def test_log_incremental(self):
        self.path.write_bytes((FX / "ollama-server.log").read_bytes())
        lg = self.log()
        lg.today_stats()
        with open(self.path, "ab") as f:
            f.write(LINE.encode() * 2)
        s = lg.today_stats()
        self.assertEqual(s["requests"], 6)

    def test_log_rotation_resets(self):
        self.path.write_bytes((FX / "ollama-server.log").read_bytes())
        lg = self.log()
        lg.today_stats()
        self.path.write_text(LINE)
        self.assertEqual(lg.today_stats(), {"requests": 1, "median_ms": 10.0})

    def test_partial_line_waits(self):
        lg = self.log()
        self.path.write_text(LINE[:-1])
        self.assertEqual(lg.today_stats()["requests"], 0)
        with open(self.path, "a") as f:
            f.write("\n")
        self.assertEqual(lg.today_stats()["requests"], 1)

    def test_day_rollover(self):
        self.path.write_bytes((FX / "ollama-server.log").read_bytes())
        lg = self.log()
        self.assertEqual(lg.today_stats()["requests"], 4)
        self.today = datetime.date(2026, 11, 28)
        self.assertEqual(lg.today_stats(), {"requests": 0, "median_ms": None})
        self.today = datetime.date(2026, 11, 26)
        self.assertEqual(lg.today_stats()["requests"], 2)

    def test_missing_log(self):
        self.assertEqual(self.log().today_stats(), {"requests": 0, "median_ms": None})


class ParserTests(unittest.TestCase):
    def test_parse_ollama_loaded(self):
        r = ext_local.parse_ollama(load("ollama-ps-loaded_UNVERIFIED.json"), load("ollama-tags.json"),
                                   {"requests": 1, "median_ms": 2.0})
        m = r["loaded"][0]
        self.assertEqual(m["name"], "qwen3:8b")
        self.assertEqual(m["size_mb"], 6346)
        self.assertEqual(m["vram_mb"], 5086)
        self.assertEqual(m["context"], 4096)
        self.assertGreater(m["expires_in_s"], 0)
        self.assertEqual(r["installed"]["count"], 2)
        self.assertEqual(r["installed"]["size_gb"], 5.1)
        self.assertEqual(r["today"]["requests"], 1)

    def test_parse_ollama_empty(self):
        r = ext_local.parse_ollama({"models": []}, load("ollama-tags.json"), None)
        self.assertEqual(r["loaded"], [])
        self.assertEqual(r["installed"]["count"], 2)

    def test_lmstudio_loaded_only(self):
        r = ext_local.parse_lmstudio(load("lmstudio-v0-models.json"))
        self.assertEqual(r["loaded"], [{"name": "qwen3-4b", "context": 32768}])
        self.assertEqual(r["available"], 2)

    def test_lmstudio_port_default(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: [p.unlink() for p in d.iterdir()] and None or d.rmdir())
        self.assertEqual(ext_local.lmstudio_port(d / "missing.json"), 1234)
        bad = d / "bad.json"
        bad.write_text("{nope")
        self.assertEqual(ext_local.lmstudio_port(bad), 1234)
        good = d / "good.json"
        good.write_text('{"port": 4321}')
        self.assertEqual(ext_local.lmstudio_port(good), 4321)

    def test_parse_comfyui(self):
        r = ext_local.parse_comfyui(load("comfyui-system-stats_UNVERIFIED.json"),
                                    load("comfyui-queue_UNVERIFIED.json"))
        self.assertEqual(r["version"], "0.3.40")
        self.assertEqual(r["queue"], {"running": 1, "pending": 2})
        self.assertEqual(r["vram"], {"used_mb": 24576, "total_mb": 32768})
        self.assertEqual(r["ram"], {"used_mb": 24576, "total_mb": 32768})

    def test_wrong_shapes_never_raise(self):
        bad = [[], None, "x", 5, {"models": None}, {"models": "x"}, {"models": [None, 1, "s"]},
               {"data": [None, {"state": 3}]}, {"system": [], "devices": "x"},
               {"devices": [None, {"vram_total": "a", "vram_free": None}], "system": {"ram_total": "x"}},
               {"models": [{"size": "big", "expires_at": 5, "name": None}]}]
        for a in bad:
            for b in bad:
                self.assertIsInstance(ext_local.parse_ollama(a, b, None), dict)
                self.assertIsInstance(ext_local.parse_comfyui(a, b), dict)
            self.assertIsInstance(ext_local.parse_lmstudio(a), dict)


class NonFiniteTests(unittest.TestCase):
    def test_ollama_non_finite_never_raises(self):
        inf = float("inf")
        nan = float("nan")
        ps = {"models": [{"name": "a", "size": nan, "size_vram": inf,
                          "context_length": nan}]}
        tags = {"models": [{"name": "a", "size": inf}]}
        r = ext_local.parse_ollama(ps, tags, {"requests": 0, "median_ms": None})
        json.dumps(r, allow_nan=False)
        self.assertIsNone(r["loaded"][0]["size_mb"])
        self.assertIsNone(r["loaded"][0]["context"])

    def test_comfyui_non_finite_never_raises(self):
        nan = float("nan")
        stats = {"system": {"ram_total": nan, "ram_free": 1},
                 "devices": [{"vram_total": float("inf"), "vram_free": 1}]}
        r = ext_local.parse_comfyui(stats, {})
        json.dumps(r, allow_nan=False)


class ProbeTests(unittest.TestCase):
    def test_comfyui_fallback_url(self):
        calls = []

        def get(url, headers):
            calls.append(url)
            if url.startswith("http://h1"):
                raise Refused()
            if url.endswith("/system_stats"):
                return load("comfyui-system-stats_UNVERIFIED.json")
            return load("comfyui-queue_UNVERIFIED.json")

        r = ext_local.comfyui_probe(["http://h1:8000", "http://h2:8188"])(get)
        self.assertTrue(r["up"])
        self.assertEqual(r["url"], "http://h2:8188")
        self.assertEqual(r["queue"]["pending"], 2)

    def test_comfyui_probe_error_wins_over_refused(self):
        def get(url, headers):
            if "h1" in url:
                raise Refused()
            raise ProbeError("timeout")

        with self.assertRaises(ProbeError):
            ext_local.comfyui_probe(["http://h1", "http://h2"])(get)

    def test_probes_propagate_refused(self):
        def get(url, headers):
            raise Refused()

        for p in (ext_local.ollama_probe("http://x", None), ext_local.lmstudio_probe("http://x"),
                  ext_local.comfyui_probe(["http://x", "http://y"])):
            with self.assertRaises(Refused):
                p(get)

    def test_ollama_tags_failure_leaves_installed_none(self):
        def get(url, headers):
            if url.endswith("/api/ps"):
                return {"models": []}
            raise ProbeError("timeout")

        r = ext_local.ollama_probe("http://x", None)(get)
        self.assertTrue(r["up"])
        self.assertIsNone(r["installed"])

    def test_lmstudio_probe(self):
        r = ext_local.lmstudio_probe("http://x:1234")(lambda u, h: load("lmstudio-v0-models.json"))
        self.assertEqual(r["url"], "http://x:1234")
        self.assertEqual(len(r["loaded"]), 1)


if __name__ == "__main__":
    unittest.main()
