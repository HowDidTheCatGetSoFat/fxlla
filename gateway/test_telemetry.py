import importlib
import json
import os
import shutil
import sqlite3
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


tools_reader = importlib.import_module("tools_reader")
ToolsReader = tools_reader.ToolsReader

# Local noon on a fixed date, so "today" is the same in every time zone.
NOW = time.mktime((2027, 1, 15, 12, 0, 0, 0, 0, -1))


class Clock:
    def __init__(self, t=NOW):
        self.t = t

    def __call__(self):
        return self.t


class ReaderBase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="fxlla-reader-")
        self.store = os.path.join(self.root, "store")
        self.state = os.path.join(self.root, "state")
        os.makedirs(self.store)
        os.makedirs(self.state)
        self.old_env = os.environ.get("FXLLA_STATE_DIR")
        self.old_out = os.environ.pop("FXLLA_MEDIA_OUT", None)
        os.environ["FXLLA_STATE_DIR"] = self.state
        self.clock = Clock()
        self.alive = set()
        self.reader = ToolsReader(self.store, clock=self.clock,
                                  alive=lambda pid: pid in self.alive)

    def tearDown(self):
        if self.old_env is None:
            os.environ.pop("FXLLA_STATE_DIR", None)
        else:
            os.environ["FXLLA_STATE_DIR"] = self.old_env
        if self.old_out is not None:
            os.environ["FXLLA_MEDIA_OUT"] = self.old_out
        shutil.rmtree(self.root, ignore_errors=True)

    def write_stream(self, name, recs, raw=""):
        with open(os.path.join(self.state, name + ".jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.write(raw)

    def write_job(self, rec):
        d = os.path.join(self.store, "media", "jobs")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, rec["id"] + ".json"), "w") as f:
            json.dump(rec, f)


class TestKbReader(ReaderBase):
    def test_missing_everything_is_empty(self):
        k = self.reader.kb()
        self.assertEqual(k["recent"], [])
        self.assertEqual(k["bases"], [])
        self.assertIsNone(k["db_bytes"])
        self.assertEqual(k["today"], {"searches": 0, "adds": 0,
                                      "avg_search_ms": None, "cold_starts": 0})

    def test_today_aggregates_and_recent_cap(self):
        recs = [{"ts": NOW - 10 - i, "kind": "search", "kb": "a", "hits": 2,
                 "duration_ms": 100 + 100 * (i % 2), "embed_cold": i == 0,
                 "ok": True} for i in range(60)]
        recs.append({"ts": NOW - 5, "kind": "add", "kb": "a", "source": "x.md",
                     "chunks_added": 3, "duration_ms": 9, "ok": True})
        recs.append({"ts": NOW - 3 * 86400, "kind": "search", "kb": "a",
                     "duration_ms": 9999, "embed_cold": True, "ok": True})
        recs.sort(key=lambda r: r["ts"])   # the log is append order
        self.write_stream("kb", recs)
        k = self.reader.kb()
        self.assertEqual(len(k["recent"]), 50)
        self.assertEqual(k["recent"][0]["kind"], "add")   # newest first
        self.assertEqual(k["today"]["searches"], 60)
        self.assertEqual(k["today"]["adds"], 1)
        self.assertEqual(k["today"]["cold_starts"], 1)
        self.assertEqual(k["today"]["avg_search_ms"], 150)

    def test_corrupt_log_degrades(self):
        self.write_stream("kb", [{"ts": NOW, "kind": "search", "kb": "a",
                                  "duration_ms": 10, "ok": True}],
                          raw="{not json\n\x00\xff garbage")
        k = self.reader.kb()
        self.assertEqual(k["today"]["searches"], 1)

    def test_non_dict_and_odd_values_do_not_raise(self):
        self.write_stream("kb", [{"ts": "x", "kind": "search",
                                  "duration_ms": "slow"}, {"kind": 5}])
        k = self.reader.kb()
        self.assertEqual(k["today"]["searches"], 0)

    def make_db(self):
        os.makedirs(os.path.join(self.store, "kb"))
        path = os.path.join(self.store, "kb", "kb.db")
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE chunks (kb TEXT, source TEXT, idx INTEGER,"
                    " text TEXT, emb BLOB)")
        con.executemany("INSERT INTO chunks VALUES (?,?,?,?,?)", [
            ("a", "f1", 0, "t", b""), ("a", "f1", 1, "t", b""),
            ("a", "f2", 0, "t", b""), ("b", "g", 0, "t", b"")])
        con.commit()
        con.close()
        return path

    def test_bases_from_readonly_db(self):
        path = self.make_db()
        k = self.reader.kb()
        self.assertEqual(sorted(k["bases"], key=lambda b: b["kb"]), [
            {"kb": "a", "sources": 2, "chunks": 3},
            {"kb": "b", "sources": 1, "chunks": 1}])
        self.assertEqual(k["db_bytes"], os.path.getsize(path))

    def test_missing_db_is_never_created(self):
        self.reader.kb()
        self.assertFalse(os.path.exists(os.path.join(self.store, "kb")))

    def test_corrupt_db_degrades(self):
        os.makedirs(os.path.join(self.store, "kb"))
        with open(os.path.join(self.store, "kb", "kb.db"), "wb") as f:
            f.write(b"this is not sqlite" * 50)
        k = self.reader.kb()
        self.assertEqual(k["bases"], [])

    def test_db_without_chunks_table_degrades(self):
        os.makedirs(os.path.join(self.store, "kb"))
        sqlite3.connect(os.path.join(self.store, "kb", "kb.db")).close()
        self.assertEqual(self.reader.kb()["bases"], [])

    def test_caching(self):
        self.write_stream("kb", [{"ts": NOW, "kind": "add", "kb": "a",
                                  "ok": True}])
        self.assertEqual(self.reader.kb()["today"]["adds"], 1)
        self.write_stream("kb", [{"ts": NOW, "kind": "add", "kb": "a",
                                  "ok": True}] * 2)
        self.assertEqual(self.reader.kb()["today"]["adds"], 1)   # cached
        self.clock.t += 3
        self.assertEqual(self.reader.kb()["today"]["adds"], 2)


class TestGraphReader(ReaderBase):
    def test_missing_everything_is_empty(self):
        g = self.reader.graph()
        self.assertEqual(g["recent"], [])
        self.assertEqual(g["today"], {"queries": 0, "indexes": 0,
                                      "avg_query_ms": None})
        self.assertEqual(g["stats"], {"files": None, "defs": None,
                                      "refs": None, "indexed_at": None})

    def test_stats_from_newest_successful_index_and_kuzu_mtime(self):
        os.makedirs(os.path.join(self.store, "graph"))
        kz = os.path.join(self.store, "graph", "graph.kuzu")
        with open(kz, "wb") as f:
            f.write(b"x")
        os.utime(kz, (1234567, 1234567))
        self.write_stream("graph", [
            {"ts": NOW - 50, "kind": "index", "repo": "r", "files": 1,
             "defs": 2, "refs": 3, "ok": True},
            {"ts": NOW - 40, "kind": "index", "repo": "r", "files": 10,
             "defs": 20, "refs": 30, "ok": True},
            {"ts": NOW - 30, "kind": "index", "repo": "r", "files": None,
             "defs": None, "refs": None, "ok": False, "error": "boom"},
            {"ts": NOW - 20, "kind": "query", "query": "def", "results": 1,
             "duration_ms": 40, "ok": True},
            {"ts": NOW - 10, "kind": "query", "query": "refs", "results": 0,
             "duration_ms": 60, "ok": True}])
        g = self.reader.graph()
        self.assertEqual(g["stats"], {"files": 10, "defs": 20, "refs": 30,
                                      "indexed_at": 1234567})
        self.assertEqual(g["today"], {"queries": 2, "indexes": 3,
                                      "avg_query_ms": 50})
        self.assertEqual(g["recent"][0]["query"], "refs")

    def test_never_imports_kuzu(self):
        self.write_stream("graph", [])
        self.reader.graph()
        self.assertNotIn("kuzu", sys.modules)

    def test_corrupt_log_degrades(self):
        self.write_stream("graph", [], raw="}}}{{{\n")
        g = self.reader.graph()
        self.assertEqual(g["recent"], [])
        self.assertIsNone(g["stats"]["files"])


def job(i, **kw):
    rec = {"id": "%d-abcdef" % (1700000000 + i), "kind": "image", "job": True,
           "media": "image", "status": "done", "created": NOW - 1000 + i,
           "started": NOW - 900 + i, "finished": NOW - 800 + i,
           "model": "m", "steps": 4, "width": 512, "height": 512,
           "peak_rss_mb": 100, "progress": None, "caller": "cli",
           "pid": 111, "argv": ["gen", "--prompt", "P"], "summary": "S",
           "output": "/out/x.png", "log": "/out/x.log", "error": None}
    rec.update(kw)
    return rec


class TestMediaReader(ReaderBase):
    def test_missing_dir_is_empty(self):
        m = self.reader.media()
        self.assertEqual(m["running"], [])
        self.assertEqual(m["queued"], [])
        self.assertEqual(m["recent"], [])
        self.assertEqual(m["today"], {"jobs": 0, "done": 0, "failed": 0,
                                      "gpu_s": 0})

    def test_running_eta_and_queued(self):
        self.alive.add(222)
        self.write_job(job(1, status="running", started=NOW - 100,
                           finished=None, pid=222,
                           progress={"step": 2, "total": 8, "at": NOW - 1}))
        self.write_job(job(2, status="running", started=NOW - 10,
                           finished=None, pid=222, progress=None))
        self.write_job(job(3, status="queued", started=None, finished=None,
                           pid=None))
        m = self.reader.media()
        by = {r["id"]: r for r in m["running"]}
        a = by["1700000001-abcdef"]
        self.assertEqual(a["elapsed_s"], 100)
        self.assertEqual(a["eta_s"], 300)    # 100 / 2 * 6
        self.assertIsNone(by["1700000002-abcdef"]["eta_s"])
        self.assertEqual([q["id"] for q in m["queued"]], ["1700000003-abcdef"])

    def test_dead_worker_is_died_in_recent_and_file_untouched(self):
        rec = job(1, status="running", finished=None, pid=999)
        self.write_job(rec)
        path = os.path.join(self.store, "media", "jobs", rec["id"] + ".json")
        with open(path) as f:
            before = f.read()
        mtime = os.path.getmtime(path)
        m = self.reader.media()
        self.assertEqual(m["running"], [])
        self.assertEqual([r["status"] for r in m["recent"]], ["died"])
        with open(path) as f:
            self.assertEqual(f.read(), before)
        self.assertEqual(os.path.getmtime(path), mtime)

    def test_today_and_recent_cap(self):
        for i in range(25):
            self.write_job(job(i, status="done" if i % 5 else "failed",
                               started=NOW - 100, finished=NOW - 50))
        old = job(90, finished=NOW - 5 * 86400, created=NOW - 5 * 86400,
                  started=NOW - 5 * 86400 - 10)
        self.write_job(old)
        m = self.reader.media()
        self.assertEqual(len(m["recent"]), 20)
        self.assertEqual(m["today"]["jobs"], 25)
        self.assertEqual(m["today"]["failed"], 5)
        self.assertEqual(m["today"]["done"], 20)
        self.assertEqual(m["today"]["gpu_s"], 25 * 50)

    def test_allowlist_only(self):
        self.write_job(job(1, error="line one\nline two /Users/x/secret"))
        item = self.reader.media()["recent"][0]
        self.assertEqual(set(item) - {"elapsed_s", "eta_s"}, set(
            tools_reader.MEDIA_FIELDS))
        for banned in ("argv", "summary", "output", "log", "pid"):
            self.assertNotIn(banned, item)
        self.assertEqual(item["error"], "line one")

    def test_error_is_capped_and_home_scrubbed(self):
        home = os.path.expanduser("~")
        self.write_job(job(1, error=(home + "/a ") + "x" * 500))
        err = self.reader.media()["recent"][0]["error"]
        self.assertLessEqual(len(err), 200)
        self.assertNotIn(home, err)

    def test_corrupt_and_odd_records_are_skipped(self):
        d = os.path.join(self.store, "media", "jobs")
        os.makedirs(d)
        with open(os.path.join(d, "bad.json"), "w") as f:
            f.write("{nope")
        with open(os.path.join(d, "list.json"), "w") as f:
            f.write("[1,2]")
        with open(os.path.join(d, "x.log"), "w") as f:
            f.write("log")
        self.write_job(job(1, created="yesterday", finished="never",
                           progress={"step": "a", "total": 0}))
        m = self.reader.media()
        self.assertEqual(len(m["recent"]), 1)

    def test_media_out_override(self):
        out = os.path.join(self.root, "elsewhere")
        os.makedirs(os.path.join(out, "jobs"))
        rec = job(1)
        with open(os.path.join(out, "jobs", rec["id"] + ".json"), "w") as f:
            json.dump(rec, f)
        os.environ["FXLLA_MEDIA_OUT"] = out
        try:
            self.assertEqual(len(self.reader.media()["recent"]), 1)
        finally:
            os.environ.pop("FXLLA_MEDIA_OUT")

    def test_listing_is_cached(self):
        self.write_job(job(1))
        self.assertEqual(len(self.reader.media()["recent"]), 1)
        self.write_job(job(2))
        self.assertEqual(len(self.reader.media()["recent"]), 1)
        self.clock.t += 3
        self.assertEqual(len(self.reader.media()["recent"]), 2)


class TestIsolation(ReaderBase):
    def test_a_failing_section_is_empty_not_raised(self):
        def boom(*a, **k):
            raise RuntimeError("x")
        self.reader._kb_events = boom
        self.reader._jobs = boom
        self.assertEqual(self.reader.kb()["recent"], [])
        self.assertEqual(self.reader.media()["recent"], [])
        snap = self.reader.snapshot()
        self.assertEqual(set(snap), {"kb", "graph", "media"})


if __name__ == "__main__":
    unittest.main()
