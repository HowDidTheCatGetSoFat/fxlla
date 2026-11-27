import json
import os
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ext_http  # noqa: E402
import ext_strata  # noqa: E402
import externals  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class Counter:
    def __init__(self, result=None, exc=None):
        self.n = 0
        self.result = result
        self.exc = exc
        self.lock = threading.Lock()

    def __call__(self, get):
        with self.lock:
            self.n += 1
        if self.exc is not None:
            raise self.exc
        return dict(self.result or {"up": True, "url": "u"})


def make(probes, clock=None, **kw):
    return externals.Externals(probes, get=lambda u, h=None: {},
                               clock=clock or Clock(), **kw)


class ObserverTests(unittest.TestCase):
    def test_all_engines_present(self):
        ex = make({"a": ("ua", Counter()),
                   "b": ("ub", Counter(exc=ext_http.Refused()))})
        snap = ex.snapshot()
        self.assertEqual(set(snap), {"a", "b"})
        self.assertTrue(snap["a"]["up"])
        self.assertFalse(snap["b"]["up"])

    def test_refused_and_errors(self):
        ex = make({"r": ("ur", Counter(exc=ext_http.Refused())),
                   "h": ("uh", Counter(exc=ext_http.ProbeError("http 500"))),
                   "v": ("uv", Counter(exc=ValueError("x")))})
        snap = ex.snapshot()
        self.assertEqual(snap["r"], {"up": False, "url": "ur", "down_reason": None})
        self.assertEqual(snap["h"]["down_reason"], "http 500")
        self.assertEqual(snap["v"]["down_reason"], "error")

    def test_cache_ttl(self):
        clock = Clock()
        p = Counter()
        ex = make({"a": ("ua", p)}, clock)
        ex.snapshot()
        clock.t += 1.9
        ex.snapshot()
        self.assertEqual(p.n, 1)
        clock.t += 0.2
        ex.snapshot()
        self.assertEqual(p.n, 2)

    def test_down_retry(self):
        clock = Clock()
        p = Counter(exc=ext_http.Refused())
        ex = make({"a": ("ua", p)}, clock)
        ex.snapshot()
        clock.t += 5.0
        ex.snapshot()
        self.assertEqual(p.n, 1)
        clock.t += 5.1
        ex.snapshot()
        self.assertEqual(p.n, 2)

    def test_deadline_stale(self):
        clock = Clock()
        gate = threading.Event()
        state = {"block": False}

        def slow(get):
            if state["block"]:
                gate.wait(5)
            return {"up": True, "url": "ua", "v": 1}

        fast = Counter()
        ex = make({"slow": ("ua", slow), "fast": ("ub", fast)}, clock,
                  deadline_s=0.2)
        first = ex.snapshot()
        self.assertFalse(first["slow"]["stale"])
        state["block"] = True
        clock.t += 3.0
        t0 = time.monotonic()
        snap = ex.snapshot()
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertTrue(snap["slow"]["stale"])
        self.assertEqual(snap["slow"]["age_s"], 3.0)
        self.assertFalse(snap["fast"]["stale"])
        gate.set()

    def test_deadline_no_previous_is_timeout(self):
        gate = threading.Event()

        def slow(get):
            gate.wait(5)
            return {"up": True, "url": "ua"}

        ex = make({"slow": ("ua", slow)}, deadline_s=0.2)
        t0 = time.monotonic()
        snap = ex.snapshot()
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertEqual(snap["slow"],
                         {"up": False, "url": "ua", "down_reason": "timeout"})
        gate.set()

    def test_hung_probe_no_pileup(self):
        clock = Clock()
        gate = threading.Event()
        calls = {"n": 0}

        def hung(get):
            calls["n"] += 1
            gate.wait(5)
            return {"up": True, "url": "ua"}

        ex = make({"h": ("ua", hung)}, clock, deadline_s=0.1)
        ex.snapshot()
        for _ in range(5):
            clock.t += 3.0
            snap = ex.snapshot()
            self.assertEqual(snap["h"]["down_reason"], "timeout")
        self.assertEqual(calls["n"], 1)
        gate.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and ex._running:
            time.sleep(0.01)
        clock.t += 3.0
        snap = ex.snapshot()
        self.assertTrue(snap["h"]["up"])
        self.assertFalse(snap["h"]["stale"])

    def test_late_result_used_next(self):
        clock = Clock()
        gate = threading.Event()

        def late(get):
            gate.wait(5)
            return {"up": True, "url": "ua", "v": 7}

        ex = make({"l": ("ua", late)}, clock, deadline_s=0.1)
        self.assertEqual(ex.snapshot()["l"]["down_reason"], "timeout")
        gate.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and ex._running:
            time.sleep(0.01)
        clock.t += 3.0
        snap = ex.snapshot()
        self.assertTrue(snap["l"]["up"])
        self.assertFalse(snap["l"]["stale"])
        self.assertEqual(snap["l"]["v"], 7)

    def test_concurrent_callers_one_round(self):
        probes = {n: Counter() for n in ("a", "b", "c")}

        def slowish(c):
            def fn(get):
                time.sleep(0.05)
                return c(get)
            return fn

        ex = make({n: ("u" + n, slowish(c)) for n, c in probes.items()})
        results = []

        def call():
            results.append(ex.snapshot())

        threads = [threading.Thread(target=call) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(results), 8)
        for c in probes.values():
            self.assertEqual(c.n, 1)

    def test_default_urls_env(self):
        env = {"FXLLA_STRATA_URL": "http://s:1/", "FXLLA_OLLAMA_URL": "http://o:2/",
               "FXLLA_LMSTUDIO_URL": "http://l:3/", "FXLLA_COMFYUI_URL": "http://c:4/"}
        with mock.patch.dict(os.environ, env, clear=False):
            ex = externals.default_externals()
        self.assertEqual({k: v[0] for k, v in ex.probes.items()},
                         {"strata": "http://s:1", "ollama": "http://o:2",
                          "lmstudio": "http://l:3", "comfyui": "http://c:4"})
        clean = {k: v for k, v in os.environ.items()
                 if not k.startswith("FXLLA_") or k == "FXLLA_STORE"}
        with mock.patch.dict(os.environ, clean, clear=True), \
                mock.patch.object(externals.ext_local, "lmstudio_port",
                                  return_value=4321):
            ex = externals.default_externals()
        urls = {k: v[0] for k, v in ex.probes.items()}
        self.assertEqual(urls["strata"], "http://127.0.0.1:8096")
        self.assertEqual(urls["ollama"], "http://127.0.0.1:11434")
        self.assertEqual(urls["lmstudio"], "http://127.0.0.1:4321")
        self.assertEqual(urls["comfyui"], "http://127.0.0.1:8000")
        seen = []

        def fake_get(url, headers=None):
            seen.append(url)
            raise ext_http.Refused()

        with self.assertRaises(ext_http.Refused):
            ex.probes["comfyui"][1](fake_get)
        self.assertEqual(seen, ["http://127.0.0.1:8000/system_stats",
                                "http://127.0.0.1:8188/system_stats"])

    def test_empty_externals(self):
        e = externals.empty_externals()
        self.assertEqual(set(e), {"strata", "ollama", "lmstudio", "comfyui"})
        for v in e.values():
            self.assertEqual(v, {"up": False, "url": None, "down_reason": "error"})

    def test_key_never_in_snapshot(self):
        key = "test-key-not-real"

        def get(url, headers=None):
            if url.endswith("/health"):
                return {"status": "ok"}
            raise ext_http.ProbeError("http 401")

        probe = ext_strata.StrataProbe("http://127.0.0.1:1",
                                       key_source=lambda: [key])
        ex = externals.Externals({"strata": ("http://127.0.0.1:1", probe)},
                                 get=get, clock=Clock())
        snap = ex.snapshot()
        self.assertNotIn(key, json.dumps(snap))


if __name__ == "__main__":
    unittest.main()
