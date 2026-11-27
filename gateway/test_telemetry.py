import importlib
import os
import subprocess
import time
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
telemetry = importlib.import_module("telemetry")
Telemetry = telemetry.Telemetry
client_tag = telemetry.client_tag
parse_ioreg = telemetry.parse_ioreg
parse_vm_stat = telemetry.parse_vm_stat
parse_swapusage = telemetry.parse_swapusage
MachineSampler = telemetry.MachineSampler


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

IOREG = ('"PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=86238248960,'
         '"Tiler Utilization %"=6,"Renderer Utilization %"=11,"Device Utilization %"=13,"In use system memory"=2423013376}')


class TestMachine(unittest.TestCase):
    def test_new_id_distinct_over_70000(self):
        t = Telemetry()
        ids = {t.new_id() for _ in range(70000)}
        self.assertEqual(len(ids), 70000)
        self.assertTrue(all(len(i) == 8 for i in ids))

    def test_parse_ioreg(self):
        self.assertEqual(parse_ioreg(IOREG), {"gpu_util_pct": 13, "gpu_inuse_bytes": 2423013376})

    def test_parse_swapusage(self):
        r = parse_swapusage("total = 2048.00M  used = 869.00M  free = 1179.00M  (encrypted)")
        self.assertEqual((r["swap_total_bytes"], r["swap_used_bytes"]), (2048 * 2**20, 869 * 2**20))

    def test_parse_vm_stat(self):
        text = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                "Pages free:                               100.\nPages inactive:                           50.\n"
                "Pages speculative:                        10.\n")
        self.assertEqual(parse_vm_stat(text, 16384, total_bytes=1000 * 16384)["ram_used_bytes"], 840 * 16384)

    def test_missing_commands_give_none_not_errors(self):
        def boom(*a, **k):
            raise FileNotFoundError("ioreg")
        s = MachineSampler(tempfile.gettempdir(), run=boom).sample()
        self.assertIsNone(s["gpu_util_pct"])
        self.assertIsNotNone(s["disk_free_bytes"])

    def test_a_hanging_command_is_cut_at_its_timeout(self):
        def hang(*a, **k):
            raise subprocess.TimeoutExpired(a[0], k.get("timeout"))
        self.assertIsNone(MachineSampler(tempfile.gettempdir(), run=hang).sample()["pressure"])

    def test_one_sample_is_bounded_by_a_shared_deadline(self):
        def slow(cmd, **k):
            time.sleep(k["timeout"])
            raise subprocess.TimeoutExpired(cmd, k["timeout"])
        t0 = time.monotonic()
        s = MachineSampler(tempfile.gettempdir(), run=slow).sample()
        self.assertLess(time.monotonic() - t0, 3)
        self.assertIsNone(s["pressure"])
        self.assertIsNone(s["swap_used_bytes"])
        self.assertIsNotNone(s["disk_free_bytes"])

    def test_sample_is_cached_for_ttl(self):
        calls = []

        def run(*a, **k):
            calls.append(a)
            raise FileNotFoundError()
        m = MachineSampler(tempfile.gettempdir(), ttl_s=60, run=run)
        m.sample()
        n = len(calls)
        m.sample()
        self.assertEqual(len(calls), n)

    def test_fake_run_fills_fields_and_pressure(self):
        def run(cmd, **k):
            out = {"ioreg": IOREG, "vm_stat": "Pages free: 1.\nPages inactive: 1.\nPages speculative: 1.\n",
                   "hw.memsize": "16384", "vm.swapusage": "total = 1.00M  used = 0.50M  free = 0.50M",
                   "kern.memorystatus_vm_pressure_level": "2"}
            key = cmd[0] if cmd[0] in out else cmd[-1]
            return subprocess.CompletedProcess(cmd, 0, stdout=out[key], stderr="")
        s = MachineSampler(tempfile.gettempdir(), run=run).sample()
        self.assertEqual(s["pressure"], "warn")
        self.assertEqual(s["gpu_util_pct"], 13)
        self.assertEqual(s["ram_total_bytes"], 16384)
        self.assertEqual(s["swap_total_bytes"], 2**20)


if __name__ == "__main__":
    unittest.main()
