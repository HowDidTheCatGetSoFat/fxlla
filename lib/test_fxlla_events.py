import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fxlla_events as ev  # noqa: E402

WORKER = (
    "import sys; sys.path.insert(0, sys.argv[1]); import fxlla_events as ev\n"
    "w = int(sys.argv[2])\n"
    "for i in range(int(sys.argv[3])):\n"
    "    ev.append('kb', {'w': w, 'i': i, 'pad': 'x' * 300})\n"
)


class Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        p = mock.patch.dict(os.environ, {"FXLLA_STATE_DIR": self.td.name})
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop("FXLLA_CALLER", None)

    def path(self, stream="kb"):
        return os.path.join(self.td.name, stream + ".jsonl")


class TestStateDir(Base):
    def test_override(self):
        self.assertEqual(ev.state_dir(), self.td.name)

    def test_xdg(self):
        with mock.patch.dict(os.environ, {"XDG_STATE_HOME": "/x/state"}):
            os.environ.pop("FXLLA_STATE_DIR")
            self.assertEqual(ev.state_dir(), "/x/state/fxlla")

    def test_home_default(self):
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("FXLLA_STATE_DIR")
            os.environ.pop("XDG_STATE_HOME", None)
            self.assertEqual(
                ev.state_dir(),
                os.path.join(os.path.expanduser("~"), ".local", "state", "fxlla"),
            )

    def test_isolation(self):
        ev.append("kb", {"kind": "search"})
        self.assertTrue(os.path.exists(self.path()))


class TestAppend(Base):
    def test_adds_ts_and_caller(self):
        ev.append("kb", {"kind": "search"})
        rec = ev.read_tail("kb")[0]
        self.assertIsInstance(rec["ts"], float)
        self.assertEqual(rec["caller"], "cli")

    def test_caller_from_env(self):
        with mock.patch.dict(os.environ, {"FXLLA_CALLER": "mcp"}):
            ev.append("kb", {"kind": "search"})
        self.assertEqual(ev.read_tail("kb")[0]["caller"], "mcp")

    def test_existing_ts_and_caller_kept(self):
        ev.append("kb", {"ts": 5.0, "caller": "gw"})
        rec = ev.read_tail("kb")[0]
        self.assertEqual((rec["ts"], rec["caller"]), (5.0, "gw"))

    def test_mode_0600(self):
        ev.append("kb", {"a": 1})
        self.assertEqual(stat.S_IMODE(os.stat(self.path()).st_mode), 0o600)

    def test_creates_missing_dir(self):
        sub = os.path.join(self.td.name, "a", "b")
        with mock.patch.dict(os.environ, {"FXLLA_STATE_DIR": sub}):
            ev.append("kb", {"a": 1})
            self.assertEqual(len(ev.read_tail("kb")), 1)

    def test_unwritable_never_raises(self):
        blocker = os.path.join(self.td.name, "file")
        open(blocker, "w").close()
        with mock.patch.dict(os.environ, {"FXLLA_STATE_DIR": blocker + "/sub"}):
            with mock.patch("sys.stderr"):
                ev.append("kb", {"a": 1})
            self.assertEqual(ev.read_tail("kb"), [])

    def test_unserializable_never_raises(self):
        with mock.patch("sys.stderr"):
            ev.append("kb", {"a": object()})

    def test_concurrent_processes_no_interleave(self):
        procs = [
            subprocess.Popen([sys.executable, "-c", WORKER, HERE, str(w), "100"])
            for w in range(6)
        ]
        for p in procs:
            self.assertEqual(p.wait(), 0)
        with open(self.path(), "rb") as f:
            lines = f.read().split(b"\n")
        self.assertEqual(lines[-1], b"")
        recs = [json.loads(x) for x in lines[:-1]]
        self.assertEqual(len(recs), 600)
        seen = {(r["w"], r["i"]) for r in recs}
        self.assertEqual(len(seen), 600)


class TestRotation(Base):
    def test_rotates_past_threshold(self):
        with mock.patch.object(ev, "MAX_BYTES", 500):
            for i in range(40):
                ev.append("kb", {"i": i, "pad": "y" * 50})
        self.assertTrue(os.path.exists(self.path() + ".1"))
        self.assertLess(os.path.getsize(self.path()), 500 + 200)
        recs = ev.read_tail("kb")
        self.assertEqual(recs[0]["i"], 39)
        self.assertLess(len(recs), 40)

    def test_default_threshold_is_5mb(self):
        self.assertEqual(ev.MAX_BYTES, 5 * 1024 * 1024)

    def test_older_backup_replaced(self):
        with open(self.path() + ".1", "w") as f:
            f.write("old\n")
        with mock.patch.object(ev, "MAX_BYTES", 100):
            for i in range(10):
                ev.append("kb", {"i": i, "pad": "z" * 50})
        with open(self.path() + ".1") as f:
            self.assertNotIn("old", f.read())


class TestReadTail(Base):
    def test_missing_file(self):
        self.assertEqual(ev.read_tail("kb"), [])

    def test_newest_first(self):
        for i in range(5):
            ev.append("kb", {"i": i})
        self.assertEqual([r["i"] for r in ev.read_tail("kb")], [4, 3, 2, 1, 0])

    def test_max_lines(self):
        for i in range(10):
            ev.append("kb", {"i": i})
        self.assertEqual([r["i"] for r in ev.read_tail("kb", 3)], [9, 8, 7])

    def test_bad_lines_skipped(self):
        ev.append("kb", {"i": 0})
        with open(self.path(), "a") as f:
            f.write("not json\n[1,2]\n\n")
        ev.append("kb", {"i": 1})
        self.assertEqual([r["i"] for r in ev.read_tail("kb")], [1, 0])

    def test_torn_first_line_in_tail_skipped(self):
        for i in range(2000):
            ev.append("kb", {"i": i, "pad": "p" * 100})
        recs = ev.read_tail("kb", 5000)
        self.assertEqual(recs[0]["i"], 1999)
        self.assertGreater(len(recs), 100)
        ids = [r["i"] for r in recs]
        self.assertEqual(ids, list(range(1999, 1999 - len(ids), -1)))

    def test_ignores_rotated_file(self):
        with open(self.path() + ".1", "w") as f:
            f.write(json.dumps({"i": "old"}) + "\n")
        self.assertEqual(ev.read_tail("kb"), [])


if __name__ == "__main__":
    unittest.main()
