import contextlib
import http.server
import importlib
import io
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
core = importlib.import_module("core")
mcp_server = importlib.import_module("rag_mcp")

try:
    import sqlite_vec  # noqa: F401
    _HAVE_VEC = True
except Exception:
    _HAVE_VEC = False


class TestCore(unittest.TestCase):
    def test_chunk_overlap(self):
        self.assertEqual(len(core.chunk_text("x" * 2000, size=800, overlap=120)), 3)

    def test_chunk_empty(self):
        self.assertEqual(core.chunk_text("   "), [])

    def test_cosine(self):
        self.assertAlmostEqual(core._cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(core._cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(core._cosine([0, 0], [1, 1]), 0.0)

    def test_pack_roundtrip(self):
        v = [0.1, 0.2, 0.3]
        self.assertEqual([round(x, 4) for x in core._unpack(core._pack(v))], v)

    def test_index_enabled(self):
        for on in ("1", "true", "YES", "On"):
            os.environ["FXLLA_KB_INDEX"] = on
            self.assertTrue(core._index_enabled())
        for off in ("", "0", "false", "no"):
            os.environ["FXLLA_KB_INDEX"] = off
            self.assertFalse(core._index_enabled())
        os.environ.pop("FXLLA_KB_INDEX", None)

    def test_vec_table_quoted(self):
        self.assertEqual(core._vec_table("my.kb-1"), '"vec_my.kb-1"')

    def test_load_vec_disabled_returns_none(self):
        # With the index off, _load_vec never touches the extension.
        os.environ.pop("FXLLA_KB_INDEX", None)
        con = sqlite3.connect(":memory:")
        self.assertIsNone(core._load_vec(con))


@unittest.skipUnless(_HAVE_VEC, "sqlite-vec not installed (run under uv run --with sqlite-vec)")
class TestVecIndex(unittest.TestCase):
    # Builds a tiny store directly (no embedder) and checks the KNN index
    # returns the same ordering as the brute-force cosine scan.
    def _con(self):
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE chunks (kb TEXT, source TEXT, idx INTEGER, text TEXT, emb BLOB)")
        vecs = {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0], "c": [0.9, 0.1, 0.0]}
        for i, (name, v) in enumerate(vecs.items()):
            con.execute("INSERT INTO chunks VALUES (?,?,?,?,?)",
                        ("k", name, i, name, core._pack(v)))
        con.commit()
        return con

    def test_indexed_matches_bruteforce(self):
        os.environ["FXLLA_KB_INDEX"] = "1"
        try:
            con = self._con()
            vec = core._load_vec(con)
            self.assertIsNotNone(vec)
            q = [1.0, 0.0, 0.0]
            top = core._search_indexed(con, vec, "k", q, 2)
            self.assertEqual([r[1] for r in top], ["a", "c"])
            self.assertAlmostEqual(top[0][0], 1.0, places=4)
        finally:
            os.environ.pop("FXLLA_KB_INDEX", None)

    def test_rebuild_on_stale(self):
        os.environ["FXLLA_KB_INDEX"] = "1"
        try:
            con = self._con()
            vec = core._load_vec(con)
            core._ensure_vec_index(con, vec, "k")
            self.assertEqual(con.execute('SELECT COUNT(*) FROM "vec_k"').fetchone()[0], 3)
            con.execute("INSERT INTO chunks VALUES (?,?,?,?,?)",
                        ("k", "d", 3, "d", core._pack([0.0, 0.0, 1.0])))
            con.commit()
            core._ensure_vec_index(con, vec, "k")
            self.assertEqual(con.execute('SELECT COUNT(*) FROM "vec_k"').fetchone()[0], 4)
        finally:
            os.environ.pop("FXLLA_KB_INDEX", None)


class _Health(http.server.BaseHTTPRequestHandler):
    """Answers /health like llama-server does, and /props when told which model.

    `model` None stands for a server that will not identify itself: another
    embedding server, or a build too old to answer /props.
    """
    model = None

    def do_GET(self):
        if self.path.startswith("/props") and self.model:
            body = json.dumps({"model_path": self.model}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200 if self.path.startswith("/health") else 404)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *_a):
        pass


@contextlib.contextmanager
def _fake_server(model=None):
    """A live health endpoint on a free port, standing in for llama-server."""
    handler = type("_H", (_Health,), {"model": model})
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    saved = core.EMBED_PORT
    core.EMBED_PORT = srv.server_address[1]
    try:
        yield core.EMBED_PORT
    finally:
        core.EMBED_PORT = saved
        srv.shutdown()
        srv.server_close()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestEmbedderLifecycle(unittest.TestCase):
    def test_a_slow_poll_interval_would_dominate_startup(self):
        # The regression this guards: the server was ready in ~0.17s but the poll
        # interval was one second, so the first probe missed it and every query
        # slept through a server that was already up. Asserting the constant is
        # not enough - it does not prove the loop uses it. This times the real
        # loop against a server that starts listening shortly after the spawn.
        port = _free_port()
        saved_port, saved_model = core.EMBED_PORT, core._embed_model
        saved_popen = core.subprocess.Popen
        # KB_DIR too: with FXLLA_STORE unset it is the relative path "kb", so
        # taking the startup lock littered a kb/embed.lock into whatever
        # directory the suite was run from.
        saved_kb = core.KB_DIR
        core.KB_DIR = tempfile.mkdtemp()
        core.EMBED_PORT = port
        core._embed_model = lambda: "/dev/null"

        class _Alive:
            def poll(self):
                return None
            def terminate(self):
                pass
            def wait(self, timeout=None):
                return 0

        srv = {}

        def _fake_popen(*_a, **_k):
            def serve():
                time.sleep(0.15)  # what a real model load costs, roughly
                srv["s"] = http.server.HTTPServer(("127.0.0.1", port), _Health)
                srv["s"].serve_forever()
            threading.Thread(target=serve, daemon=True).start()
            return _Alive()

        core.subprocess.Popen = _fake_popen
        try:
            begin = time.monotonic()
            with core.Embedder():
                elapsed = time.monotonic() - begin
            # Ready at 0.15s. A one-second interval cannot come in under 0.6s.
            self.assertLess(elapsed, 0.6, "startup is dominated by the poll interval")
        finally:
            core.EMBED_PORT, core._embed_model = saved_port, saved_model
            core.subprocess.Popen = saved_popen
            core.KB_DIR = saved_kb
            if "s" in srv:
                srv["s"].shutdown()
                srv["s"].server_close()

    def test_starting_a_server_takes_an_exclusive_lock(self):
        # Asserts the mechanism, not cross-process behaviour: a unit test cannot
        # race two interpreters. Without the lock, ten concurrent searches each
        # started a server and up to four in ten failed after losing the bind;
        # that was verified by running fifty real searches, not here.
        port = _free_port()
        saved = (core.EMBED_PORT, core._embed_model, core.subprocess.Popen,
                 core.fcntl.flock, core.KB_DIR)
        core.EMBED_PORT = port
        core._embed_model = lambda: "/dev/null"
        core.KB_DIR = tempfile.mkdtemp()
        modes = []
        core.fcntl.flock = lambda fh, mode: modes.append(mode)
        srv = {}

        class _Alive:
            def poll(self):
                return None
            def terminate(self):
                pass
            def wait(self, timeout=None):
                return 0

        def _fake_popen(*_a, **_k):
            def serve():
                time.sleep(0.05)
                srv["s"] = http.server.HTTPServer(("127.0.0.1", port), _Health)
                srv["s"].serve_forever()
            threading.Thread(target=serve, daemon=True).start()
            return _Alive()

        core.subprocess.Popen = _fake_popen
        try:
            with core.Embedder():
                pass
            self.assertIn(core.fcntl.LOCK_EX, modes, "spawning was not serialized")
        finally:
            (core.EMBED_PORT, core._embed_model, core.subprocess.Popen,
             core.fcntl.flock, core.KB_DIR) = saved
            if "s" in srv:
                srv["s"].shutdown()
                srv["s"].server_close()

    def test_the_server_is_started_without_forcing_a_pooling_mode(self):
        # Every embedding GGUF declares its own pooling_type and llama.cpp
        # honours it: nomic MEAN, bge CLS, qwen3-embedding LAST. Forcing mean was
        # right for nomic and silently wrong for the rest, and a badly pooled
        # vector is still a vector, so nothing downstream would have complained.
        # Measured before removing it: omitting the flag reproduces --pooling
        # mean bit for bit on nomic, while --pooling cls visibly differs.
        port = _free_port()
        saved = (core.EMBED_PORT, core._embed_model, core.subprocess.Popen,
                 core.KB_DIR)
        core.EMBED_PORT = port
        core._embed_model = lambda: "/dev/null"
        core.KB_DIR = tempfile.mkdtemp()
        seen = {}
        srv = {}

        class _Alive:
            def poll(self):
                return None
            def terminate(self):
                pass
            def wait(self, timeout=None):
                return 0

        def _fake_popen(argv, *_a, **_k):
            seen["argv"] = argv

            def serve():
                time.sleep(0.05)
                srv["s"] = http.server.HTTPServer(("127.0.0.1", port), _Health)
                srv["s"].serve_forever()
            threading.Thread(target=serve, daemon=True).start()
            return _Alive()

        core.subprocess.Popen = _fake_popen
        try:
            with core.Embedder():
                pass
            self.assertIn("llama-server", seen["argv"][0])
            self.assertNotIn("--pooling", seen["argv"])
        finally:
            (core.EMBED_PORT, core._embed_model, core.subprocess.Popen,
             core.KB_DIR) = saved
            if "s" in srv:
                srv["s"].shutdown()
                srv["s"].server_close()

    def test_ready_deadline_stays_generous(self):
        # A large embedding model on an external disk can take a while to load.
        # An earlier revision quietly halved this while fixing the interval.
        self.assertGreaterEqual(core._READY_TIMEOUT, 120.0)

    def test_health_timeout_tolerates_a_loaded_machine(self):
        # Too short a timeout reports a working-but-slow server as absent, which
        # then fails the query outright.
        self.assertGreaterEqual(core._HEALTH_TIMEOUT, 3.0)

    def test_healthy_is_false_when_nothing_listens(self):
        saved = core.EMBED_PORT
        core.EMBED_PORT = _free_port()
        try:
            self.assertFalse(core._server_healthy(timeout=0.5))
        finally:
            core.EMBED_PORT = saved

    def test_reuses_a_live_server_without_spawning(self):
        with _fake_server():
            saved = core.subprocess.Popen

            def _no(*_a, **_k):
                raise AssertionError("spawned a server while one was listening")

            core.subprocess.Popen = _no
            try:
                with core.Embedder() as emb:
                    self.assertIsNone(emb.proc)  # None means borrowed
            finally:
                core.subprocess.Popen = saved

    def test_borrowed_server_is_left_running(self):
        with _fake_server() as port:
            with core.Embedder():
                pass
            self.assertTrue(core._server_healthy(timeout=1.0), "killed a borrowed server")
            self.assertEqual(port, core.EMBED_PORT)

    def test_a_server_we_started_is_stopped(self):
        class _Spy:
            def __init__(self):
                self.terminated = False
            def terminate(self):
                self.terminated = True
            def wait(self, timeout=None):
                return 0

        emb = core.Embedder.__new__(core.Embedder)
        emb.proc, emb._lock = _Spy(), None
        emb.__exit__(None, None, None)
        self.assertTrue(emb.proc.terminated)

    def test_a_rejected_request_exits_with_a_message_not_a_traceback(self):
        # A server answering /health but rejecting /v1/embeddings is the shape of
        # a chat server on the embedding port. The MCP layer forwards stderr to a
        # model verbatim, so a traceback there is a 2 KB tool result.
        with _fake_server():  # _Health 404s on POST
            emb = core.Embedder.__new__(core.Embedder)
            emb.proc, emb._lock = None, None
            with self.assertRaises(SystemExit) as ctx:
                emb.embed(["x"])
            message = str(ctx.exception)
            # The hint only the dedicated HTTPError branch produces. Without it
            # the rejection falls through to the connection-reset retry, which
            # both wastes a round trip and reports the wrong cause.
            self.assertIn("--embeddings", message)
            self.assertLess(len(message), 400)


class TestDimensionGuard(unittest.TestCase):
    # A server for a different model returns a different width, and _cosine zips
    # to the shorter vector: no error, just meaningless scores, and on add a base
    # permanently holding two widths.
    def _con(self, dim):
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE chunks (kb TEXT, source TEXT, idx INTEGER, "
                    "text TEXT, emb BLOB)")
        con.execute("INSERT INTO chunks VALUES ('k','s',0,'t',?)",
                    (core._pack([0.1] * dim),))
        return con

    def test_reports_the_stored_width(self):
        self.assertEqual(core._kb_dim(self._con(768), "k"), 768)

    def test_none_for_an_empty_base(self):
        self.assertIsNone(core._kb_dim(self._con(768), "other"))

    def test_a_mismatch_refuses(self):
        with self.assertRaises(SystemExit) as ctx:
            core._require_dim(self._con(768), "k", [0.1] * 384)
        self.assertIn("384", str(ctx.exception))
        self.assertIn("768", str(ctx.exception))

    def test_a_match_passes(self):
        core._require_dim(self._con(768), "k", [0.1] * 768)

    def test_an_empty_base_accepts_any_width(self):
        core._require_dim(self._con(768), "fresh", [0.1] * 384)


class TestModelSelection(unittest.TestCase):
    # An alias names both the catalog entry and the directory `fxlla pull` made,
    # so selecting a model is selecting a directory.
    @contextlib.contextmanager
    def _selected(self, alias, store, explicit=True):
        saved = (core.EMBED_ALIAS, core._EMBED_ENV, core.STORE)
        core.EMBED_ALIAS, core._EMBED_ENV, core.STORE = (
            alias, alias if explicit else "", store)
        try:
            yield
        finally:
            core.EMBED_ALIAS, core._EMBED_ENV, core.STORE = saved

    def _dir(self, alias):
        root = tempfile.mkdtemp()
        directory = os.path.join(root, "models", alias)
        os.makedirs(directory)
        return root, directory

    def test_the_default_alias_keeps_the_original_directory(self):
        root, _d = self._dir("embed")
        with self._selected("embed", root, explicit=False):
            self.assertEqual(core._embed_dir(), os.path.join(root, "models", "embed"))

    def test_an_alias_selects_its_own_directory(self):
        root, _d = self._dir("embed-qwen3")
        with self._selected("embed-qwen3", root):
            self.assertTrue(core._embed_dir().endswith("models/embed-qwen3"))

    def test_an_alias_cannot_walk_out_of_the_store(self):
        # It is interpolated into a path, so a traversal would read weights from
        # anywhere on disk.
        for bad in ("../../etc", "..", ".", "a/b", "", "with space", "x;rm"):
            with self._selected(bad, "/tmp"):
                with self.assertRaises(SystemExit, msg=f"accepted {bad!r}"):
                    core._embed_dir()

    def test_entry_decides_when_a_directory_holds_two_quants(self):
        # Re-pulling a different quant leaves both files behind; the glob would
        # take whichever sorts first, which is not the one that was asked for.
        root, directory = self._dir("embed")
        for name in ("aaa-q4_k_m.gguf", "zzz-q8_0.gguf"):
            open(os.path.join(directory, name), "w").close()
        with open(os.path.join(directory, ".entry"), "w") as fh:
            fh.write("zzz-q8_0.gguf\n")
        with self._selected("embed", root):
            self.assertEqual(os.path.basename(core._embed_model()), "zzz-q8_0.gguf")

    def test_a_missing_entry_falls_back_to_the_glob(self):
        root, directory = self._dir("embed")
        open(os.path.join(directory, "only.gguf"), "w").close()
        with self._selected("embed", root):
            self.assertEqual(os.path.basename(core._embed_model()), "only.gguf")

    def test_a_stale_entry_falls_back_to_the_glob(self):
        # `.entry` survives a file being deleted by hand.
        root, directory = self._dir("embed")
        open(os.path.join(directory, "real.gguf"), "w").close()
        with open(os.path.join(directory, ".entry"), "w") as fh:
            fh.write("deleted.gguf")
        with self._selected("embed", root):
            self.assertEqual(os.path.basename(core._embed_model()), "real.gguf")

    def test_entry_cannot_point_outside_its_directory(self):
        # The traversal has to land on a file that really exists, or the check
        # passes for the wrong reason: a path to nowhere falls back to the glob
        # whether or not anything is guarding it.
        root, directory = self._dir("embed")
        open(os.path.join(directory, "real.gguf"), "w").close()
        open(os.path.join(root, "outside.gguf"), "w").close()
        with open(os.path.join(directory, ".entry"), "w") as fh:
            fh.write("../../outside.gguf")
        with self._selected("embed", root):
            chosen = core._embed_model()
        self.assertEqual(os.path.dirname(chosen), directory)
        self.assertEqual(os.path.basename(chosen), "real.gguf")

    def test_none_when_nothing_is_installed(self):
        root, _d = self._dir("embed")
        with self._selected("embed", root):
            self.assertIsNone(core._embed_model())


class TestServerIdentity(unittest.TestCase):
    # A server outlives the command that started it, so switching models leaves
    # the previous one listening. Verified against the real thing before this
    # existed: with no model installed at all, add and search both succeeded
    # against a borrowed server. nomic and embeddinggemma are both 768-dim and
    # qwen3-embedding and bge-large both 1024, so the width guard cannot see it.
    @contextlib.contextmanager
    def _running(self, path, explicit=""):
        # Both globals come from one variable at import, so a test that moves one
        # without the other is testing a state that cannot occur.
        saved = (core._server_model, core._EMBED_ENV, core.EMBED_ALIAS)
        core._server_model = lambda *_a, **_k: path
        core._EMBED_ENV = explicit
        core.EMBED_ALIAS = explicit or "embed"
        try:
            yield
        finally:
            core._server_model, core._EMBED_ENV, core.EMBED_ALIAS = saved

    def test_a_different_model_refuses(self):
        with self._running("/store/models/embed/nomic.gguf"):
            with self.assertRaises(SystemExit) as ctx:
                core._require_server_model("/store/models/embed-small/bge.gguf")
        message = str(ctx.exception)
        self.assertIn("nomic.gguf", message)
        self.assertIn("bge.gguf", message)
        self.assertIn("fxlla kb stop", message)  # the way out, not just the fault

    def test_the_same_model_passes(self):
        with self._running("/store/models/embed/nomic.gguf"):
            core._require_server_model("/store/models/embed/nomic.gguf")

    def test_the_same_model_reached_by_a_symlink_passes(self):
        # A store on an external disk is routinely reached through one.
        root = tempfile.mkdtemp()
        real = os.path.join(root, "real.gguf")
        open(real, "w").close()
        link = os.path.join(root, "link.gguf")
        os.symlink(real, link)
        with self._running(link):
            core._require_server_model(real)

    def test_a_server_that_will_not_identify_itself_is_allowed(self):
        # Someone pointing FXLLA_EMBED_PORT at their own embedding server is a
        # legitimate setup: unverifiable is not the same as wrong.
        with self._running(None):
            with contextlib.redirect_stderr(io.StringIO()) as err:
                core._require_server_model("/store/models/embed/nomic.gguf")
        self.assertEqual(err.getvalue(), "")

    def test_it_says_so_when_a_model_was_actually_chosen(self):
        with self._running(None, explicit="embed-qwen3"):
            with contextlib.redirect_stderr(io.StringIO()) as err:
                core._require_server_model("/store/models/embed-qwen3/q.gguf")
        self.assertIn("embed-qwen3", err.getvalue())

    def test_the_check_is_actually_wired_into_the_reuse_path(self):
        # Testing _require_server_model on its own proves nothing about whether
        # anyone calls it, and the reuse path is the only place it matters: this
        # goes through Embedder against a live server that names its model.
        saved = core._embed_model
        core._embed_model = lambda: "/store/models/embed-small/bge.gguf"
        try:
            with _fake_server(model="/store/models/embed/nomic.gguf"):
                with self.assertRaises(SystemExit) as ctx:
                    with core.Embedder():
                        pass
            self.assertIn("nomic.gguf", str(ctx.exception))
        finally:
            core._embed_model = saved

    def test_an_identified_server_with_nothing_installed_locally_is_allowed(self):
        # The pre-existing arrangement: no local weights, a server someone else
        # runs. Nothing was chosen, so there is nothing to contradict.
        with self._running("/somewhere/else.gguf"):
            core._require_server_model(None)


class TestReindex(unittest.TestCase):
    # Switching models makes every stored vector unusable. The chunk text is in
    # the table already, so re-embedding never re-reads sources that may have
    # moved or changed since.
    def _store(self, rows, dim=768):
        root = tempfile.mkdtemp()
        saved = (core.KB_DIR, core.DB_PATH)
        core.KB_DIR = os.path.join(root, "kb")
        os.makedirs(core.KB_DIR)
        core.DB_PATH = os.path.join(core.KB_DIR, "kb.db")
        con = core._db()
        con.executemany("INSERT INTO chunks VALUES (?,?,?,?,?)",
                        [("k", "s.md", i, f"chunk {i}", core._pack([0.1] * dim))
                         for i in range(rows)])
        con.commit()
        con.close()
        return saved

    @staticmethod
    def _fake(dim, fail_after=None):
        class _E:
            calls = 0

            def __enter__(self_):
                return self_

            def __exit__(self_, *_a):
                return False

            def embed(self_, texts):
                _E.calls += 1
                if fail_after is not None and _E.calls > fail_after:
                    raise RuntimeError("server died")
                return [[0.25] * dim for _ in texts]
        return _E

    @contextlib.contextmanager
    def _patched(self, embedder):
        # Its per-batch progress is worth having on a base that takes minutes,
        # and worth swallowing here.
        saved = core.Embedder
        core.Embedder = embedder
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                yield
        finally:
            core.Embedder = saved

    def _widths(self):
        con = sqlite3.connect(core.DB_PATH)
        try:
            return [n // 4 for (n,) in con.execute(
                "SELECT length(emb) FROM chunks WHERE kb='k'")]
        finally:
            con.close()

    def test_every_vector_is_rewritten_at_the_new_width(self):
        saved = self._store(10)
        try:
            with self._patched(self._fake(384)):
                core.cmd_reindex(core.argparse.Namespace(name="k"))
            self.assertEqual(self._widths(), [384] * 10)
        finally:
            core.KB_DIR, core.DB_PATH = saved

    def test_the_text_is_left_alone(self):
        saved = self._store(4)
        try:
            with self._patched(self._fake(384)):
                core.cmd_reindex(core.argparse.Namespace(name="k"))
            con = sqlite3.connect(core.DB_PATH)
            texts = [t for (t,) in con.execute(
                "SELECT text FROM chunks WHERE kb='k' ORDER BY idx")]
            con.close()
            self.assertEqual(texts, [f"chunk {i}" for i in range(4)])
        finally:
            core.KB_DIR, core.DB_PATH = saved

    def test_it_batches_instead_of_sending_one_huge_request(self):
        saved = self._store(core._REINDEX_BATCH * 2 + 1)
        try:
            fake = self._fake(384)
            with self._patched(fake):
                core.cmd_reindex(core.argparse.Namespace(name="k"))
            self.assertEqual(fake.calls, 3)
        finally:
            core.KB_DIR, core.DB_PATH = saved

    def test_an_interrupted_run_leaves_the_base_untouched(self):
        # Half a base holds two widths at once and _kb_dim reads whichever row
        # comes first, so a partial write is a store that lies about itself.
        # This is the assertion behind the single-transaction comment.
        saved = self._store(core._REINDEX_BATCH * 2)
        try:
            with self._patched(self._fake(384, fail_after=1)):
                with self.assertRaises(RuntimeError):
                    core.cmd_reindex(core.argparse.Namespace(name="k"))
            self.assertEqual(set(self._widths()), {768}, "a partial reindex landed")
        finally:
            core.KB_DIR, core.DB_PATH = saved

    def test_an_empty_base_is_refused(self):
        saved = self._store(0)
        try:
            with self.assertRaises(SystemExit):
                core.cmd_reindex(core.argparse.Namespace(name="absent"))
        finally:
            core.KB_DIR, core.DB_PATH = saved


class TestEvalSet(unittest.TestCase):
    # The golden set is hand written, so the cheap structural mistakes are the
    # likely ones: a path typo, an expectation naming a file nobody indexes.
    @classmethod
    def setUpClass(cls):
        with open(core.EVAL_SET, encoding="utf-8") as fh:
            cls.spec = json.load(fh)

    def test_every_corpus_file_exists(self):
        for rel in self.spec["corpus"]:
            self.assertTrue(os.path.isfile(os.path.join(core.REPO_ROOT, rel)), rel)

    def test_every_expected_file_is_in_the_corpus(self):
        corpus = set(self.spec["corpus"])
        for item in self.spec["queries"]:
            for rel in item["expect"]:
                self.assertIn(rel, corpus, f"{rel} is expected but never indexed")

    def test_queries_are_distinct(self):
        queries = [item["q"] for item in self.spec["queries"]]
        self.assertEqual(len(queries), len(set(queries)))

    def test_every_query_expects_something(self):
        for item in self.spec["queries"]:
            self.assertTrue(item["expect"], item["q"])


class _FakeEmbedder:
    """Stands in for Embedder: no server, a fixed 3-wide vector per text."""
    started = False

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        pass

    def embed(self, texts):
        return [[1.0, 0.0, 0.5] for _ in texts]


class _FakeColdEmbedder(_FakeEmbedder):
    started = True


class TestKbEvents(unittest.TestCase):
    QUERY = "zebra-unique-query-text"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.state = os.path.join(self.tmp, "state")
        self.saved = (core.KB_DIR, core.DB_PATH, core.Embedder,
                      os.environ.get("FXLLA_STATE_DIR"),
                      os.environ.pop("FXLLA_CALLER", None))
        core.KB_DIR = os.path.join(self.tmp, "kb")
        core.DB_PATH = os.path.join(core.KB_DIR, "kb.db")
        core.Embedder = _FakeEmbedder
        os.environ["FXLLA_STATE_DIR"] = self.state
        self.doc = os.path.join(self.tmp, "notes.txt")
        with open(self.doc, "w") as f:
            f.write("alpha beta gamma " * 20)

    def tearDown(self):
        core.KB_DIR, core.DB_PATH, core.Embedder = self.saved[:3]
        for key, val in (("FXLLA_STATE_DIR", self.saved[3]),
                         ("FXLLA_CALLER", self.saved[4])):
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def _main(self, *argv):
        out = io.StringIO()
        saved = sys.argv
        sys.argv = ["core.py"] + list(argv)
        code = None
        try:
            with contextlib.redirect_stdout(out):
                try:
                    core.main()
                except SystemExit as e:
                    code = e.code
        finally:
            sys.argv = saved
        return out.getvalue(), code

    def _events(self):
        path = os.path.join(self.state, "kb.jsonl")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_add_search_rm_each_write_one_record(self):
        self._main("add", "k1", self.doc)
        self._main("search", "k1", self.QUERY, "--json")
        self._main("rm", "k1")
        add, search, rm = self._events()
        self.assertEqual((add["kind"], add["kb"], add["source"], add["ok"]),
                         ("add", "k1", "notes.txt", True))
        self.assertGreater(add["chunks_added"], 0)
        self.assertEqual((search["kind"], search["kb"], search["ok"]),
                         ("search", "k1", True))
        self.assertGreater(search["hits"], 0)
        self.assertIsNotNone(search["top_score"])
        self.assertFalse(search["embed_cold"])
        self.assertIsNone(search["error"])
        self.assertEqual((rm["kind"], rm["kb"], rm["source"], rm["ok"]),
                         ("rm", "k1", None, True))
        for rec in (add, search, rm):
            self.assertIsInstance(rec["duration_ms"], int)
            self.assertEqual(rec["caller"], "cli")
            self.assertIn("ts", rec)

    def test_embed_cold_reflects_a_server_this_process_started(self):
        self._main("add", "k1", self.doc)
        core.Embedder = _FakeColdEmbedder
        self._main("search", "k1", self.QUERY, "--json")
        self.assertTrue(self._events()[-1]["embed_cold"])

    def test_embedder_started_is_false_until_it_spawns_a_server(self):
        self.assertIs(self.saved[2].started, False)

    def test_failure_is_recorded_and_exit_code_kept(self):
        _out, code = self._main("search", "empty", self.QUERY, "--json")
        self.assertIn("empty", str(code))
        rec = self._events()[-1]
        self.assertFalse(rec["ok"])
        self.assertEqual((rec["kind"], rec["hits"]), ("search", 0))
        self.assertIsNone(rec["top_score"])
        self.assertTrue(rec["error"].startswith("knowledge base"))
        self.assertLessEqual(len(rec["error"]), 200)

    def test_error_is_first_line_capped_and_home_free(self):
        home = os.path.expanduser("~")
        saved = core.cmd_rm

        def boom(_a):
            sys.exit("first %s/x\n" % home + "second line" + "y" * 400)
        core.cmd_rm = boom
        try:
            self._main("rm", "k1")
        finally:
            core.cmd_rm = saved
        err = self._events()[-1]["error"]
        self.assertNotIn("second", err)
        self.assertNotIn(home, err)
        self.assertLessEqual(len(err), 200)

    def test_no_query_text_in_the_log(self):
        self._main("add", "k1", self.doc)
        self._main("search", "k1", self.QUERY, "--json")
        self._main("search", "nokb", self.QUERY)
        with open(os.path.join(self.state, "kb.jsonl")) as f:
            raw = f.read()
        self.assertNotIn(self.QUERY, raw)
        self.assertNotIn("alpha beta", raw)
        self.assertNotIn(self.tmp, raw)

    def test_printed_output_is_unchanged_by_telemetry(self):
        self._main("add", "k1", self.doc)
        on, code_on = self._main("search", "k1", self.QUERY, "--json")
        # A state dir that is a regular file makes every append fail.
        blocker = os.path.join(self.tmp, "blocker")
        open(blocker, "w").close()
        os.environ["FXLLA_STATE_DIR"] = blocker
        off, code_off = self._main("search", "k1", self.QUERY, "--json")
        self.assertEqual(on, off)
        self.assertEqual(code_on, code_off)
        self.assertTrue(json.loads(on))

    def test_mcp_search_is_attributed_to_mcp(self):
        # Real subprocess path: an empty base exits before any embedder starts.
        env = dict(os.environ, FXLLA_STORE=self.tmp)
        env.pop("FXLLA_CALLER", None)
        req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": "rag_search",
                                     "arguments": {"kb": "none", "query": "q"}}})
        proc = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(core.__file__), "rag_mcp.py")],
            input=req + "\n", capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("error:", proc.stdout)
        self.assertEqual(self._events()[-1]["caller"], "mcp")


class TestMCP(unittest.TestCase):
    def test_initialize(self):
        r = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertEqual(r["result"]["serverInfo"]["name"], "fxlla-rag")

    def test_tools_list(self):
        r = mcp_server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.assertEqual([t["name"] for t in r["result"]["tools"]], ["rag_search"])

    def test_notification_no_response(self):
        self.assertIsNone(mcp_server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))


if __name__ == "__main__":
    unittest.main()
