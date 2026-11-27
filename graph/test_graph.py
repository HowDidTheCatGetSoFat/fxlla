import ast
import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
codegraph = importlib.import_module("codegraph")
mcp_server = importlib.import_module("graph_mcp")

try:
    import kuzu  # noqa: F401
    _HAVE_KUZU = True
except Exception:
    _HAVE_KUZU = False

try:
    import tree_sitter_language_pack  # noqa: F401
    _HAVE_TS = True
except Exception:
    _HAVE_TS = False

SNIPPET = """
class A:
    def m(self):
        helper()

def helper():
    pass

def top():
    A().m()
"""


class TestVisitor(unittest.TestCase):
    def _visit(self):
        v = codegraph._Visitor("x.py")
        v.visit(ast.parse(SNIPPET))
        return v

    def test_defs(self):
        v = self._visit()
        kinds = {name: kind for name, _qual, kind, _f, _l in v.defs}
        self.assertEqual(set(kinds), {"A", "m", "helper", "top"})
        self.assertEqual(kinds["A"], "class")
        self.assertEqual(kinds["m"], "method")
        self.assertEqual(kinds["helper"], "function")

    def test_qualname_and_caller(self):
        v = self._visit()
        quals = {name: qual for name, qual, _k, _f, _l in v.defs}
        self.assertEqual(quals["m"], "A.m")
        callers = {name: caller for name, _f, _l, caller in v.refs}
        self.assertEqual(callers["helper"], "A.m")
        self.assertEqual(callers["m"], "top")


def _run(cmd, **kw):
    # Call a cmd_* function with json output and return the parsed result.
    ns = types.SimpleNamespace(json=True, **kw)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmd(ns)
    return json.loads(buf.getvalue() or "[]")


@unittest.skipUnless(_HAVE_KUZU, "kuzu not installed (run under uv run --with kuzu)")
class TestGraphKuzu(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.py = os.path.join(self.tmp, "sample.py")
        with open(self.py, "w") as f:
            f.write(SNIPPET)
        # Redirect the module store to a fresh graph and reset the cached conn.
        codegraph.GRAPH_DIR = self.tmp
        codegraph.DB_PATH = os.path.join(self.tmp, "graph.kuzu")
        codegraph._CONN = None
        with contextlib.redirect_stdout(io.StringIO()):
            codegraph.cmd_index(types.SimpleNamespace(paths=[self.py]))

    def tearDown(self):
        codegraph._CONN = None

    def test_def(self):
        rows = _run(codegraph.cmd_def, name="helper")
        self.assertEqual([r["qualname"] for r in rows], ["helper"])
        self.assertEqual(rows[0]["kind"], "function")

    def test_callers(self):
        rows = _run(codegraph.cmd_callers, name="helper")
        self.assertEqual([r["caller"] for r in rows], ["A.m"])

    def test_impact_transitive_with_depth(self):
        rows = _run(codegraph.cmd_impact, name="helper", depth=5)
        by_caller = {r["caller"]: r["depth"] for r in rows}
        self.assertEqual(by_caller, {"A.m": 1, "top": 2})

    def test_unused(self):
        names = {r["qualname"] for r in _run(codegraph.cmd_unused)}
        self.assertIn("top", names)       # nothing references top
        self.assertNotIn("helper", names)  # A.m references helper

    def test_reindex_is_idempotent(self):
        with contextlib.redirect_stdout(io.StringIO()):
            codegraph.cmd_index(types.SimpleNamespace(paths=[self.py]))
        self.assertEqual(len(_run(codegraph.cmd_def, name="helper")), 1)


@unittest.skipUnless(_HAVE_TS, "tree-sitter-language-pack not installed")
class TestTsExtract(unittest.TestCase):
    # Pure extraction (no graph store): tree-sitter produces the same shape as
    # the ast visitor for non-Python languages.
    def _extract(self, name, body):
        tsextract = importlib.import_module("tsextract")
        d = tempfile.mkdtemp()
        p = os.path.join(d, name)
        with open(p, "w") as f:
            f.write(body)
        lang = tsextract.LANG_BY_EXT[os.path.splitext(name)[1]]
        return tsextract.extract(p, lang)

    def test_javascript(self):
        defs, refs = self._extract("a.js",
                                   "class A { m() { helper(); } }\n"
                                   "function helper() {}\n")
        kinds = {nm: k for nm, _q, k, _f, _l in defs}
        self.assertEqual(kinds.get("A"), "class")
        self.assertEqual(kinds.get("m"), "method")
        self.assertEqual(kinds.get("helper"), "function")
        callers = {nm: c for nm, _f, _l, c in refs}
        self.assertEqual(callers.get("helper"), "A.m")

    def test_go(self):
        defs, refs = self._extract("b.go",
                                   "package main\nfunc helper() {}\n"
                                   "func top() { helper() }\n")
        self.assertIn("helper", {nm for nm, *_ in defs})
        self.assertEqual({nm: c for nm, _f, _l, c in refs}.get("helper"), "top")


@unittest.skipUnless(_HAVE_KUZU and _HAVE_TS, "kuzu and tree-sitter required")
class TestGraphMultiLang(unittest.TestCase):
    # End to end: a JS file and a Python file share the graph and resolve across
    # languages by name.
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        with open(os.path.join(self.tmp, "a.js"), "w") as f:
            f.write("function helper() {}\nfunction top() { helper(); }\n")
        with open(os.path.join(self.tmp, "b.py"), "w") as f:
            f.write("def helper():\n    pass\n\ndef run():\n    helper()\n")
        codegraph.GRAPH_DIR = self.tmp
        codegraph.DB_PATH = os.path.join(self.tmp, "graph.kuzu")
        codegraph._CONN = None
        with contextlib.redirect_stdout(io.StringIO()):
            codegraph.cmd_index(types.SimpleNamespace(paths=[self.tmp]))

    def tearDown(self):
        codegraph._CONN = None

    def test_def_spans_languages(self):
        files = {os.path.splitext(r["file"])[1] for r in _run(codegraph.cmd_def, name="helper")}
        self.assertEqual(files, {".js", ".py"})

    def test_callers_across_languages(self):
        callers = {r["caller"] for r in _run(codegraph.cmd_callers, name="helper")}
        self.assertEqual(callers, {"top", "run"})

    def test_same_line_redefinition_does_not_crash(self):
        # Two defs sharing file::line::qualname must not violate the Def primary
        # key (they are deduped), so indexing does not abort.
        with open(os.path.join(self.tmp, "dup.js"), "w") as f:
            f.write("function dup(){} function dup(){}\n")
        with contextlib.redirect_stdout(io.StringIO()):
            codegraph.cmd_index(types.SimpleNamespace(paths=[self.tmp]))
        self.assertEqual(len(_run(codegraph.cmd_def, name="dup")), 1)


class _EventsBase(unittest.TestCase):
    QUERY = "zebra_unique_symbol"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.state = os.path.join(self.tmp, "state")
        self._saved_env = {k: os.environ.get(k)
                           for k in ("FXLLA_STATE_DIR", "FXLLA_CALLER")}
        os.environ.pop("FXLLA_CALLER", None)
        os.environ["FXLLA_STATE_DIR"] = self.state

    def tearDown(self):
        for key, val in self._saved_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def _main(self, *argv):
        out = io.StringIO()
        saved = sys.argv
        sys.argv = ["codegraph.py"] + list(argv)
        code = None
        try:
            with contextlib.redirect_stdout(out):
                try:
                    codegraph.main()
                except SystemExit as e:
                    code = e.code
        finally:
            sys.argv = saved
        return out.getvalue(), code

    def _events(self):
        path = os.path.join(self.state, "graph.jsonl")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]


class TestGraphEventsStubbed(_EventsBase):
    # The commands are replaced, so these run without kuzu.
    def setUp(self):
        super().setUp()
        self._saved = {n: getattr(codegraph, n) for n in
                       ("cmd_def", "cmd_refs", "cmd_callers", "cmd_impact",
                        "cmd_unused", "cmd_index", "cmd_stats")}

        def fake_query(args):
            print(json.dumps([{"x": 1}, {"x": 2}, {"x": 3}]))
            return 3
        for n in ("cmd_def", "cmd_refs", "cmd_callers", "cmd_impact", "cmd_unused"):
            setattr(codegraph, n, fake_query)

    def tearDown(self):
        for n, fn in self._saved.items():
            setattr(codegraph, n, fn)
        super().tearDown()

    def test_each_query_kind_is_recorded(self):
        self._main("def", self.QUERY, "--json")
        self._main("refs", self.QUERY, "--json")
        self._main("callers", self.QUERY, "--json")
        self._main("impact", self.QUERY, "--json")
        self._main("unused", "--json")
        recs = self._events()
        self.assertEqual([r["query"] for r in recs],
                         ["def", "refs", "callers", "impact", "unused"])
        for r in recs:
            self.assertEqual((r["kind"], r["results"], r["ok"], r["error"],
                              r["caller"]), ("query", 3, True, None, "cli"))
            self.assertIsInstance(r["duration_ms"], int)

    def test_no_query_text_in_the_log(self):
        self._main("def", self.QUERY, "--json")
        self._main("impact", self.QUERY, "--json")
        with open(os.path.join(self.state, "graph.jsonl")) as f:
            self.assertNotIn(self.QUERY, f.read())

    def test_a_failing_query_is_recorded_and_exits_the_same(self):
        def boom(_a):
            sys.exit("kuzu is required.\nsecond line")
        codegraph.cmd_refs = boom
        _out, code = self._main("refs", self.QUERY)
        self.assertEqual(code, "kuzu is required.\nsecond line")
        rec = self._events()[-1]
        self.assertEqual((rec["query"], rec["ok"], rec["results"]),
                         ("refs", False, None))
        self.assertEqual(rec["error"], "kuzu is required.")

    def test_index_record_has_counts_and_repo_basename_only(self):
        def fake_index(args):
            print("indexed 4 files: 9 defs, 12 refs")
            return (4, 9, 12)
        codegraph.cmd_index = fake_index
        out, _ = self._main("index", os.path.join(self.tmp, "myrepo"))
        self.assertEqual(out, "indexed 4 files: 9 defs, 12 refs\n")
        rec = self._events()[-1]
        self.assertEqual((rec["kind"], rec["repo"], rec["files"], rec["defs"],
                          rec["refs"], rec["ok"]),
                         ("index", "myrepo", 4, 9, 12, True))
        with open(os.path.join(self.state, "graph.jsonl")) as f:
            self.assertNotIn(self.tmp, f.read())

    def test_index_of_dot_logs_the_resolved_directory_name(self):
        codegraph.cmd_index = lambda a: (1, 1, 1)
        repo = os.path.join(self.tmp, "dotrepo")
        os.makedirs(repo)
        cwd = os.getcwd()
        os.chdir(repo)
        try:
            self._main("index", ".")
            self._main("index", "..")
        finally:
            os.chdir(cwd)
        dot, dotdot = self._events()
        self.assertEqual(dot["repo"], os.path.basename(os.path.realpath(repo)))
        self.assertEqual(dotdot["repo"],
                         os.path.basename(os.path.realpath(self.tmp)))

    def test_index_of_root_logs_null_repo(self):
        codegraph.cmd_index = lambda a: (1, 1, 1)
        self._main("index", "/")
        self.assertIsNone(self._events()[-1]["repo"])

    def test_failed_index_has_null_counts(self):
        def boom(_a):
            sys.exit("no indexable source files found in the given paths")
        codegraph.cmd_index = boom
        self._main("index", self.tmp)
        rec = self._events()[-1]
        self.assertFalse(rec["ok"])
        self.assertEqual((rec["files"], rec["defs"], rec["refs"]),
                         (None, None, None))

    def test_other_commands_write_nothing(self):
        codegraph.cmd_stats = lambda a: print("files: 0")
        self._main("stats")
        self.assertEqual(self._events(), [])

    def test_output_is_unchanged_when_the_log_cannot_be_written(self):
        on = self._main("def", self.QUERY, "--json")
        blocker = os.path.join(self.tmp, "blocker")
        open(blocker, "w").close()
        os.environ["FXLLA_STATE_DIR"] = blocker
        off = self._main("def", self.QUERY, "--json")
        self.assertEqual(on, off)

    def test_mcp_server_marks_its_subprocesses_as_mcp(self):
        # A real subprocess: with or without kuzu the query ends in a record.
        env = dict(os.environ, FXLLA_STORE=self.tmp)
        env.pop("FXLLA_CALLER", None)
        req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": "list_unused", "arguments": {}}})
        proc = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(codegraph.__file__),
                                          "graph_mcp.py")],
            input=req + "\n", capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self._events()[-1]["caller"], "mcp")


@unittest.skipUnless(_HAVE_KUZU, "kuzu not installed (run under uv run --with kuzu)")
class TestGraphEventsKuzu(_EventsBase):
    def setUp(self):
        super().setUp()
        self.py = os.path.join(self.tmp, "sample.py")
        with open(self.py, "w") as f:
            f.write(SNIPPET)
        self._saved_paths = (codegraph.GRAPH_DIR, codegraph.DB_PATH)
        codegraph.GRAPH_DIR = os.path.join(self.tmp, "graph")
        codegraph.DB_PATH = os.path.join(codegraph.GRAPH_DIR, "graph.kuzu")
        codegraph._CONN = None

    def tearDown(self):
        codegraph._CONN = None
        codegraph.GRAPH_DIR, codegraph.DB_PATH = self._saved_paths
        super().tearDown()

    def test_real_index_and_queries(self):
        self._main("index", self.tmp)
        out_def, _ = self._main("def", "helper", "--json")
        self._main("impact", "helper", "--depth", "5", "--json")
        index, qdef, impact = self._events()
        self.assertEqual((index["kind"], index["repo"], index["files"],
                          index["defs"], index["ok"]),
                         ("index", self.tmp.split(os.sep)[-1], 1, 4, True))
        self.assertGreater(index["refs"], 0)
        self.assertEqual((qdef["query"], qdef["results"]), ("def", 1))
        self.assertEqual((impact["query"], impact["results"]), ("impact", 2))
        self.assertEqual(len(json.loads(out_def)), 1)

    def test_real_index_of_dot_logs_the_directory_name(self):
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            self._main("index", ".")
        finally:
            os.chdir(cwd)
        rec = self._events()[-1]
        self.assertEqual(rec["repo"], os.path.basename(os.path.realpath(self.tmp)))
        self.assertEqual((rec["files"], rec["ok"]), (1, True))

    def test_json_output_identical_with_and_without_the_log(self):
        self._main("index", self.py)
        on = self._main("callers", "helper", "--json")
        blocker = os.path.join(self.tmp, "blocker")
        open(blocker, "w").close()
        os.environ["FXLLA_STATE_DIR"] = blocker
        off = self._main("callers", "helper", "--json")
        self.assertEqual(on, off)
        self.assertEqual(json.loads(on[0])[0]["caller"], "A.m")

    def test_no_query_text_in_the_log(self):
        self._main("index", self.py)
        self._main("def", self.QUERY, "--json")
        with open(os.path.join(self.state, "graph.jsonl")) as f:
            self.assertNotIn(self.QUERY, f.read())


class TestMCP(unittest.TestCase):
    def test_initialize(self):
        r = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertEqual(r["result"]["serverInfo"]["name"], "fxlla-graph")

    def test_tools_list(self):
        r = mcp_server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.assertEqual(
            {t["name"] for t in r["result"]["tools"]},
            {"find_definition", "find_references", "find_callers",
             "find_impact", "list_unused"})


if __name__ == "__main__":
    unittest.main()
