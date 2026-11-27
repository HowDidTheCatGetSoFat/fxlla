import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ext_http  # noqa: E402
import ext_strata  # noqa: E402

FIX = Path(__file__).parent / "fixtures"
KEY = "test-key-not-real"


def load(name):
    return json.loads((FIX / name).read_text())


HEALTH = {"model": "qwen3.8-flash-next-iq3_s", "max_context": 262144,
          "loaded": True}


def fake_get(keys_ok, telemetry=None, metrics=None, calls=None):
    """keys_ok: set of accepted keys."""
    def get(url, headers=None):
        if calls is not None:
            calls.append((url, headers))
        if url.endswith("/health"):
            return HEALTH
        auth = (headers or {}).get("Authorization", "")
        if auth.replace("Bearer ", "") not in keys_ok:
            raise ext_http.ProbeError("http 401")
        if url.endswith("/v1/telemetry"):
            return telemetry if telemetry is not None else load("strata-telemetry.json")
        if url.endswith("/metrics"):
            return metrics if metrics is not None else load("strata-metrics-live.json")
        raise ext_http.ProbeError("http 404")
    return get


class FindKeys(unittest.TestCase):
    def test_find_keys_order(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name, body, mt in (("a.json", {"api_key": "ka"}, 100),
                                   ("b.json", {"api_key": "kb"}, 300),
                                   ("c.json", {"api_key": "ka"}, 200),
                                   ("d.json", {"other": 1}, 400)):
                (d / name).write_text(json.dumps(body))
                os.utime(d / name, (mt, mt))
            (d / "bad.json").write_text("{nope")
            os.utime(d / "bad.json", (500, 500))
            keys = ext_strata.find_keys({"STRATA_API_KEY": "env"}, d)
            self.assertEqual(keys, ["env", "kb", "ka"])

    def test_missing_dir(self):
        self.assertEqual(ext_strata.find_keys({}, Path("/nonexistent-dir-x")), [])


class Rows(unittest.TestCase):
    def test_row_maps_fields(self):
        t = load("strata-telemetry.json")
        rec = t["recent"][0]
        r = ext_strata.row(rec, "m", t["instance"]["boot_time"])
        self.assertEqual(r["engine"], "strata")
        self.assertEqual(r["source"], "strata")
        self.assertEqual(r["cached_tokens"], rec["reused"])
        self.assertEqual(r["ttft_ms"], round(rec["first_engine_token_s"] * 1000))
        self.assertEqual(r["duration_ms"], int(rec["duration_s"] * 1000))
        self.assertEqual(r["ts"], int(rec["time"]))
        self.assertEqual(r["tps"], round(rec["output_tokens"] / (rec["decode_ms"] / 1000), 1))
        self.assertIsNone(r["reasoning_tokens"])
        self.assertEqual(set(r), set(ext_strata.ROW_KEYS))

    def test_row_edge_cases(self):
        base = {"time": 1000.0, "output_tokens": 10, "decode_ms": 0}
        self.assertIsNone(ext_strata.row(base, "m", 0)["tps"])
        self.assertIsNone(ext_strata.row({"time": 1000.0, "output_tokens": 10}, "m", 0)["tps"])
        self.assertIsNone(ext_strata.row(dict(base, time=5.0), "m", 10))
        self.assertIsNone(ext_strata.row("x", "m", 0))
        self.assertIsNone(ext_strata.row({"output_tokens": 1}, "m", 0))
        r = ext_strata.row({"time": 1.0, "output_tokens": "9", "decode_ms": 5}, "m", None)
        self.assertIsNone(r["output_tokens"])
        self.assertIsNone(r["tps"])


class Parse(unittest.TestCase):
    def test_parse_fixture(self):
        t = load("strata-telemetry.json")
        e = ext_strata.parse(HEALTH, t, None, "ok")
        self.assertEqual(e["model"], "qwen3.8-flash-next-iq3_s")
        self.assertEqual(e["context"], 262144)
        self.assertEqual(e["ram"]["total_mb"], 137438953472 // 1048576)
        self.assertEqual(e["totals"]["requests"], 5174)
        self.assertIsNone(e["live"])
        boot = t["instance"]["boot_time"]
        n = sum(1 for r in t["recent"] if r["time"] >= boot)
        self.assertEqual(len(e["recent"]), min(20, n))
        ts = [r["ts"] for r in e["recent"]]
        self.assertEqual(ts, sorted(ts, reverse=True))

    def test_parse_health_only(self):
        e = ext_strata.parse(HEALTH, None, None, "missing")
        self.assertEqual(e["model"], "qwen3.8-flash-next-iq3_s")
        self.assertEqual(e["context"], 262144)
        self.assertTrue(e["loaded"])
        self.assertEqual(e["auth"], "missing")
        self.assertEqual(e["recent"], [])

    def test_parse_live(self):
        e = ext_strata.parse(HEALTH, load("strata-telemetry.json"),
                             load("strata-metrics-live.json"), "ok")
        self.assertEqual(e["live"]["generated"], 50)

    def test_wrong_shapes_never_raise(self):
        junk = [None, [], "x", 5, {}, [None], {"recent": "x"},
                {"recent": [None, 1, "a", {"time": "z"}]},
                {"model": None, "work": None, "ram": [], "totals": "x",
                 "rates": 3, "instance": [], "recent": {"a": 1}},
                {"model": {"id": 5, "context": "big"}, "ram": {"used": "x"}},
                {"live": "x"}, {"live": {"state": None}}]
        for a in junk:
            for b in junk:
                for c in junk:
                    self.assertIsInstance(ext_strata.parse(a, b, c, "ok"), dict)


class NonFinite(unittest.TestCase):
    INF = float("inf")
    NAN = float("nan")

    def test_num_rejects_non_finite_and_absurd(self):
        for v in (self.INF, -self.INF, self.NAN, 10 ** 400):
            self.assertIsNone(ext_strata._num(v))
        self.assertEqual(ext_strata._num(5), 5)

    def test_bad_row_does_not_wipe_entry(self):
        tel = {"model": {"id": "m", "context": self.NAN},
               "instance": {"state": "ready"},
               "ram": {"used": self.INF, "total": 1048576},
               "work": {"active": self.INF},
               "recent": [{"time": self.NAN, "decode_ms": 10 ** 400},
                          {"time": 100, "output_tokens": 5, "decode_ms": 1000}]}
        r = ext_strata.parse({"model": "h"}, tel, None, "ok")
        self.assertEqual(r["model"], "m")
        self.assertEqual(r["state"], "ready")
        self.assertIsNone(r["context"])
        self.assertIsNone(r["ram"]["used_mb"])
        self.assertEqual(r["ram"]["total_mb"], 1)
        self.assertIsNone(r["work"]["active"])
        self.assertEqual(len(r["recent"]), 1)
        json.dumps(r, allow_nan=False)

    def test_exception_in_one_row_skips_only_that_row(self):
        orig = ext_strata.row
        calls = []

        def boom(rec, model, boot):
            calls.append(rec)
            if rec == "bad":
                raise ValueError("x")
            return orig(rec, model, boot)
        ext_strata.row = boom
        try:
            r = ext_strata.parse(None, {"model": {"id": "m"},
                                        "recent": ["bad", {"time": 5}]},
                                 None, "ok")
        finally:
            ext_strata.row = orig
        self.assertEqual(r["model"], "m")
        self.assertEqual(len(r["recent"]), 1)


class Probe(unittest.TestCase):
    def test_probe_key_flow(self):
        calls = []
        p = ext_strata.StrataProbe("http://x", lambda: ["A", KEY])
        get = fake_get({KEY}, calls=calls)
        e = p(get)
        self.assertEqual(e["auth"], "ok")
        self.assertTrue(e["up"])
        calls.clear()
        e = p(get)
        tel = [h for u, h in calls if u.endswith("/v1/telemetry")]
        self.assertEqual(tel, [{"Authorization": "Bearer " + KEY}])
        self.assertEqual(e["auth"], "ok")

    def test_key_rotation_rereads_disk(self):
        pool = ["old"]
        p = ext_strata.StrataProbe("http://x", lambda: list(pool))
        p(fake_get({"old"}))
        pool[:] = [KEY]
        e = p(fake_get({KEY}))
        self.assertEqual(e["auth"], "ok")
        self.assertEqual(e["totals"]["requests"], 5174)

    def test_probe_no_keys(self):
        e = ext_strata.StrataProbe("http://x", lambda: [])(fake_get(set()))
        self.assertEqual(e["auth"], "missing")
        self.assertEqual(e["model"], "qwen3.8-flash-next-iq3_s")
        e = ext_strata.StrataProbe("http://x", lambda: ["a", "b"])(fake_get(set()))
        self.assertEqual(e["auth"], "rejected")

        def refuse(url, headers=None):
            raise ext_http.Refused()
        with self.assertRaises(ext_http.Refused):
            ext_strata.StrataProbe("http://x", lambda: [KEY])(refuse)

    def test_metrics_only_when_active(self):
        t = load("strata-telemetry.json")
        calls = []
        ext_strata.StrataProbe("http://x", lambda: [KEY])(fake_get({KEY}, t, calls=calls))
        self.assertFalse(any(u.endswith("/metrics") for u, _ in calls))
        t2 = json.loads(json.dumps(t))
        t2["work"]["active"] = 1
        calls.clear()
        e = ext_strata.StrataProbe("http://x", lambda: [KEY])(fake_get({KEY}, t2, calls=calls))
        self.assertTrue(any(u.endswith("/metrics") for u, _ in calls))
        self.assertEqual(e["live"]["generated"], 50)

    def test_other_errors_keep_auth(self):
        def get(url, headers=None):
            if url.endswith("/health"):
                return HEALTH
            raise ext_http.ProbeError("timeout")
        e = ext_strata.StrataProbe("http://x", lambda: [KEY])(get)
        self.assertEqual(e["auth"], "ok")
        self.assertEqual(e["recent"], [])

    def test_key_never_in_entry(self):
        t = load("strata-telemetry.json")
        t2 = json.loads(json.dumps(t))
        t2["work"]["active"] = 1
        cases = [
            ext_strata.StrataProbe("http://x", lambda: [KEY])(fake_get({KEY})),
            ext_strata.StrataProbe("http://x", lambda: [KEY])(fake_get({KEY}, t2)),
            ext_strata.StrataProbe("http://x", lambda: [KEY])(fake_get(set())),
            ext_strata.StrataProbe("http://x", lambda: [])(fake_get(set())),
        ]
        for e in cases:
            self.assertNotIn(KEY, json.dumps(e))


if __name__ == "__main__":
    unittest.main()
