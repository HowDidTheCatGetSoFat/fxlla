import importlib
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
telemetry = importlib.import_module("telemetry")
Telemetry = telemetry.Telemetry
client_tag = telemetry.client_tag


class TestTelemetry(unittest.TestCase):
    def test_totals_and_ring(self):
        t = Telemetry(recent_max=2, clock=lambda: 1000.0)
        for n in range(3):
            t.record({"id": str(n), "ts": 1000, "tps": 40.0, "ttft_ms": 500,
                      "prompt_tokens": 100, "cached_tokens": 90, "output_tokens": 10,
                      "prompt_ms": 50, "finish": "stop", "client_app": "opencode"})
        s = t.snapshot()
        self.assertEqual([r["id"] for r in s["recent"]], ["2", "1"])
        self.assertEqual((s["totals"]["requests"], s["totals"]["output_tokens"]), (3, 30))
        self.assertEqual(s["day"]["cache_saved_tokens"], 270)
        self.assertEqual(s["day"]["clients"], {"opencode": 3})

    def test_errors_and_client_gone_are_counted(self):
        t = Telemetry(clock=lambda: 1000.0)
        t.record({"ts": 1000, "finish": "error"})
        t.record({"ts": 1000, "finish": "client_gone"})
        tot = t.snapshot()["totals"]
        self.assertEqual((tot["errors"], tot["client_gone"]), (1, 1))

    def test_series_drop_points_outside_the_window(self):
        now = [1000.0]
        t = Telemetry(window_s=300, clock=lambda: now[0])
        t.record({"ts": 1000, "tps": 10.0})
        now[0] = 1400.0
        t.record({"ts": 1400, "tps": 20.0})
        self.assertEqual([v for _, v in t.snapshot()["series"]["answer_tps"]], [20.0])

    def test_prompt_tps_point(self):
        t = Telemetry(clock=lambda: 1000.0)
        t.record({"ts": 1000, "prompt_tokens": 100, "cached_tokens": 20, "prompt_ms": 500})
        t.record({"ts": 1000, "prompt_tokens": 100, "cached_tokens": 100, "prompt_ms": 500})
        t.record({"ts": 1000, "prompt_tokens": 100, "prompt_ms": 0})
        self.assertEqual(t.snapshot()["series"]["prompt_tps"], [[1000, 160.0]])

    def test_ttft_p95(self):
        t = Telemetry(clock=lambda: 1000.0)
        for ms in range(1, 101):
            t.record({"ts": 1000, "ttft_ms": ms})
        self.assertEqual(t.snapshot()["ttft_p95_ms"], 95)

    def test_ttft_p95_none_without_points(self):
        t = Telemetry(clock=lambda: 1000.0)
        t.record({"ts": 1000})
        self.assertIsNone(t.snapshot()["ttft_p95_ms"])

    def test_day_rolls_over(self):
        now = [1795700000.0]
        t = Telemetry(clock=lambda: now[0])
        t.record({"ts": int(now[0]), "client_app": "curl", "prompt_tokens": 10, "cached_tokens": 5})
        now[0] += 86400
        t.record({"ts": int(now[0]), "client_app": "hermes"})
        self.assertEqual(t.snapshot()["day"]["clients"], {"hermes": 1})

    def test_concurrent_records_lose_nothing(self):
        t = Telemetry()
        ths = [threading.Thread(target=lambda: [t.record({"ts": 1, "output_tokens": 1})
                                                for _ in range(500)]) for _ in range(16)]
        for h in ths:
            h.start()
        for h in ths:
            h.join()
        self.assertEqual(t.snapshot()["totals"]["output_tokens"], 8000)

    def test_events_ring_newest_first(self):
        t = Telemetry(events_max=2, clock=lambda: 5.0)
        for k in ("load_started", "loaded", "evicted"):
            t.event(k, "tiny")
        self.assertEqual([e["kind"] for e in t.snapshot()["events"]], ["evicted", "loaded"])

    def test_new_id_is_8_hex_and_unique(self):
        t = Telemetry()
        ids = {t.new_id() for _ in range(200)}
        self.assertEqual(len(ids), 200)
        self.assertTrue(all(len(i) == 8 and int(i, 16) >= 0 for i in ids))

    def test_client_tag(self):
        self.assertEqual(client_tag("opencode/latest/2.4.1/cli"), "opencode")
        self.assertEqual(client_tag("Hermes-Agent/0.20"), "hermes")
        self.assertEqual(client_tag("curl/8.7.1"), "curl")
        self.assertEqual(client_tag("Python-urllib/3.15"), "python")
        self.assertEqual(client_tag(None), "other")


if __name__ == "__main__":
    unittest.main()
