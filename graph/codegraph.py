#!/usr/bin/env python3
"""fxlla code graph: index code symbols and references, query the graph.

Python is parsed with the standard library `ast`; other languages (JavaScript,
TypeScript, Go, Rust, Java, C/C++, Ruby, ...) with tree-sitter (see tsextract).
Both produce the same shape and are stored in an embedded KuzuDB graph under
<store>/graph. Definitions (functions, classes, methods) and references (calls,
with the enclosing scope) become nodes; a CALLS relationship between definitions
is derived by name so transitive queries like change-impact are a single Cypher
variable-length path.

KuzuDB and tree-sitter are not in the standard library. This module is normally
run under a python that has them (e.g. `uv run --with kuzu --with tree-sitter
--with tree-sitter-language-pack`); `fxlla graph` handles that. Those imports are
deferred so the ast extraction and the MCP layer can be imported without them.

Usage (normally driven via `fxlla graph`):
  codegraph.py index <path...>   index files or directories (Python via ast,
                                 other languages via tree-sitter)
  codegraph.py def <name>        where a symbol is defined
  codegraph.py refs <name>       where a symbol is referenced
  codegraph.py callers <name>    which functions call a symbol
  codegraph.py impact <name>     transitive callers (blast radius)
  codegraph.py unused            definitions never referenced by name
  codegraph.py stats             graph totals
  codegraph.py ls                indexed files
  codegraph.py rm                clear the graph
"""
import argparse
import ast
import json
import os
import sys
import time

sys.path.append(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
try:
    import fxlla_events
except Exception:  # telemetry is optional: never block the tool on it
    fxlla_events = None

import tsextract  # tree-sitter extraction for non-Python files (lazy heavy deps)

STORE = os.environ.get("FXLLA_STORE", "")
GRAPH_DIR = os.path.join(STORE, "graph")
DB_PATH = os.path.join(GRAPH_DIR, "graph.kuzu")

_CONN = None  # (database, connection); the database must outlive the connection


def _conn():
    global _CONN
    if _CONN is not None:
        return _CONN[1]
    try:
        import kuzu
    except Exception:
        sys.exit("kuzu is required. Run `fxlla graph` (it uses uv run --with kuzu) "
                 "or set FXLLA_GRAPH_PYTHON to an interpreter that has kuzu.")
    os.makedirs(GRAPH_DIR, exist_ok=True)
    db = kuzu.Database(DB_PATH)
    con = kuzu.Connection(db)
    con.execute(
        "CREATE NODE TABLE IF NOT EXISTS Def"
        "(id STRING, name STRING, qualname STRING, kind STRING, file STRING, "
        "line INT64, PRIMARY KEY(id))")
    con.execute(
        "CREATE NODE TABLE IF NOT EXISTS Ref"
        "(id STRING, name STRING, file STRING, line INT64, caller STRING, "
        "PRIMARY KEY(id))")
    con.execute("CREATE REL TABLE IF NOT EXISTS CALLS(FROM Def TO Def)")
    _CONN = (db, con)
    return con


def _all(res):
    out = []
    while res.has_next():
        out.append(res.get_next())
    return out


class _Visitor(ast.NodeVisitor):
    def __init__(self, file):
        self.file = file
        self.names = []   # scope stack of names
        self.kinds = []   # parallel stack of kinds
        self.defs = []    # (name, qualname, kind, file, line)
        self.refs = []    # (name, file, line, caller)

    def _visit_def(self, node, kind):
        qual = ".".join(self.names + [node.name])
        self.defs.append((node.name, qual, kind, self.file, node.lineno))
        self.names.append(node.name)
        self.kinds.append(kind)
        self.generic_visit(node)
        self.names.pop()
        self.kinds.pop()

    def visit_FunctionDef(self, node):
        parent_class = bool(self.kinds) and self.kinds[-1] == "class"
        self._visit_def(node, "method" if parent_class else "function")

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._visit_def(node, "class")

    def visit_Call(self, node):
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name:
            self.refs.append((name, self.file, node.lineno, ".".join(self.names)))
        self.generic_visit(node)


def _indexable(name):
    # Python (ast) plus every extension tree-sitter extraction supports.
    return name.endswith(".py") or os.path.splitext(name)[1].lower() in tsextract.LANG_BY_EXT


# Dependency and build directories: not the user's code, and huge, so pruning
# them keeps the graph relevant and the index fast (important now that JS/TS,
# Go, Rust, etc. are indexed, e.g. node_modules and target).
_SKIP_DIRS = {".git", "__pycache__", "node_modules", "vendor", "target",
              "dist", "build", ".venv", "venv", ".tox", ".mypy_cache"}


def _gather(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, names in os.walk(p):
                dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
                for n in names:
                    if _indexable(n):
                        files.append(os.path.join(root, n))
        elif os.path.isfile(p) and _indexable(p):
            files.append(p)
    return files


# Extracts (defs, refs) from one file: Python via the stdlib ast visitor, other
# languages via tree-sitter. Returns None when the file cannot be parsed or is
# unsupported, so cmd_index skips it. Matches the (name, qualname, kind, file,
# line) / (name, file, line, caller) shapes.
def _extract_file(f):
    if f.endswith(".py"):
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                tree = ast.parse(fh.read(), filename=f)
        except (SyntaxError, ValueError):
            return None
        v = _Visitor(f)
        v.visit(tree)
        return v.defs, v.refs
    lang = tsextract.LANG_BY_EXT.get(os.path.splitext(f)[1].lower())
    if not lang:
        return None
    try:
        return tsextract.extract(f, lang)
    except Exception:
        return None


# Drops and rebuilds the CALLS edges from the current Def/Ref rows. An edge goes
# from the definition that encloses a call (matched by caller qualname within the
# same file) to every definition sharing the called name (name-approximate, since
# Python calls are not statically resolved). MERGE keeps a single edge per pair.
def _rebuild_calls(con):
    con.execute("MATCH ()-[c:CALLS]->() DELETE c")
    con.execute(
        "MATCH (c:Def), (e:Def), (r:Ref) "
        "WHERE r.caller = c.qualname AND r.file = c.file AND r.name = e.name "
        "MERGE (c)-[:CALLS]->(e)")


def cmd_index(args):
    files = _gather(args.paths)
    if not files:
        sys.exit("no indexable source files found in the given paths")
    con = _conn()
    total_defs = total_refs = indexed = 0
    for f in files:
        extracted = _extract_file(f)
        if extracted is None:
            continue
        file_defs, file_refs = extracted
        indexed += 1
        con.execute("MATCH (d:Def) WHERE d.file = $f DETACH DELETE d", {"f": f})
        con.execute("MATCH (r:Ref) WHERE r.file = $f DETACH DELETE r", {"f": f})
        # Dedup by id: two definitions can share file::line::qualname (e.g. a
        # same-line redefinition in a tree-sitter language), and Def.id is the
        # Kuzu primary key, so a duplicate would abort the whole index run.
        defs = list({f"{fl}::{ln}::{q}": {"id": f"{fl}::{ln}::{q}", "name": nm,
                                          "q": q, "k": k, "f": fl, "l": ln}
                     for (nm, q, k, fl, ln) in file_defs}.values())
        refs = [{"id": f"{fl}::{ln}::{i}", "name": nm, "f": fl, "l": ln, "c": c}
                for i, (nm, fl, ln, c) in enumerate(file_refs)]
        if defs:
            con.execute(
                "UNWIND $rows AS r CREATE (:Def {id:r.id, name:r.name, "
                "qualname:r.q, kind:r.k, file:r.f, line:r.l})", {"rows": defs})
        if refs:
            con.execute(
                "UNWIND $rows AS r CREATE (:Ref {id:r.id, name:r.name, "
                "file:r.f, line:r.l, caller:r.c})", {"rows": refs})
        total_defs += len(defs)
        total_refs += len(refs)
    _rebuild_calls(con)
    print(f"indexed {indexed} files: {total_defs} defs, {total_refs} refs")
    return indexed, total_defs, total_refs


def cmd_unused(args):
    # Definitions never referenced by name: dead-code candidates. Approximate,
    # so this excludes dunders and will still list entry points and test
    # functions called reflectively.
    con = _conn()
    rows = _all(con.execute(
        "MATCH (d:Def) WHERE NOT d.name STARTS WITH '__' "
        "AND NOT EXISTS { MATCH (r:Ref) WHERE r.name = d.name } "
        "RETURN d.qualname, d.kind, d.file, d.line ORDER BY d.file, d.line"))
    return _emit(args,
                 [{"qualname": q, "kind": k, "file": f, "line": ln} for q, k, f, ln in rows],
                 lambda r: f"{r['kind']:8} {r['qualname']}  {r['file']}:{r['line']}")


def cmd_stats(args):
    con = _conn()
    files = _all(con.execute("MATCH (d:Def) RETURN count(DISTINCT d.file)"))[0][0]
    defs = _all(con.execute("MATCH (d:Def) RETURN count(*)"))[0][0]
    refs = _all(con.execute("MATCH (r:Ref) RETURN count(*)"))[0][0]
    by_kind = _all(con.execute(
        "MATCH (d:Def) RETURN d.kind, count(*) ORDER BY d.kind"))
    top = _all(con.execute(
        "MATCH (r:Ref) RETURN r.name, count(*) AS c ORDER BY c DESC, r.name LIMIT 10"))
    if getattr(args, "json", False):
        print(json.dumps({
            "files": files, "defs": defs, "refs": refs,
            "by_kind": {k: c for k, c in by_kind},
            "top_referenced": [{"name": n, "count": c} for n, c in top]}))
        return
    print(f"files: {files}  defs: {defs}  refs: {refs}")
    print("by kind: " + ", ".join(f"{k} {c}" for k, c in by_kind))
    print("most referenced:")
    for n, c in top:
        print(f"  {c:4}  {n}")


def cmd_ls(_args):
    con = _conn()
    rows = _all(con.execute(
        "MATCH (d:Def) RETURN d.file, count(*) ORDER BY d.file"))
    if not rows:
        print("(graph is empty)")
        return
    for f, n in rows:
        print(f"{f}\t{n} defs")


def cmd_rm(_args):
    con = _conn()
    con.execute("MATCH (n) DETACH DELETE n")
    print("graph cleared")


def _emit(args, items, fmt):
    if getattr(args, "json", False):
        print(json.dumps(items))
        return len(items)
    if not items:
        print("(none)")
        return 0
    for it in items:
        print(fmt(it))
    return len(items)


def cmd_def(args):
    con = _conn()
    rows = _all(con.execute(
        "MATCH (d:Def) WHERE d.name = $n OR d.qualname = $n "
        "RETURN d.qualname, d.kind, d.file, d.line ORDER BY d.file, d.line",
        {"n": args.name}))
    return _emit(args,
                 [{"qualname": q, "kind": k, "file": f, "line": ln} for q, k, f, ln in rows],
                 lambda r: f"{r['kind']:8} {r['qualname']}  {r['file']}:{r['line']}")


def cmd_refs(args):
    con = _conn()
    rows = _all(con.execute(
        "MATCH (r:Ref) WHERE r.name = $n "
        "RETURN r.file, r.line, r.caller ORDER BY r.file, r.line", {"n": args.name}))
    return _emit(args,
                 [{"file": f, "line": ln, "caller": c} for f, ln, c in rows],
                 lambda r: f"{r['file']}:{r['line']}" + (f"  in {r['caller']}" if r['caller'] else ""))


def cmd_callers(args):
    con = _conn()
    rows = _all(con.execute(
        "MATCH (r:Ref) WHERE r.name = $n AND r.caller <> '' "
        "RETURN DISTINCT r.caller, r.file ORDER BY r.caller", {"n": args.name}))
    return _emit(args,
                 [{"caller": c, "file": f} for c, f in rows],
                 lambda r: f"{r['caller']}  ({r['file']})")


def cmd_impact(args):
    # Transitive callers (the blast radius of changing a symbol) as a single
    # Cypher variable-length path over the derived CALLS graph. Name-approximate,
    # so the depth is capped. Reports each caller definition once at its shortest
    # distance from the target.
    depth = max(1, min(int(getattr(args, "depth", None) or 5), 50))
    con = _conn()
    rows = _all(con.execute(
        f"MATCH p = (a:Def)-[:CALLS*1..{depth}]->(t:Def) WHERE t.name = $n "
        "RETURN a.qualname AS caller, min(length(p)) AS depth "
        "ORDER BY depth, caller", {"n": args.name}))
    return _emit(args,
                 [{"depth": d, "caller": c} for c, d in rows],
                 lambda r: f"{'  ' * r['depth']}{r['caller']} (depth {r['depth']})")


# Telemetry: one record per query or index run in <state dir>/graph.jsonl, written
# after the command finishes. It carries counts and timings only: never the symbol
# asked about, and for an index only the basename of the first path.
def _record(kind, fields, begin, error):
    try:
        if fxlla_events is None:
            return
        rec = {"kind": kind}
        rec.update(fields)
        rec["duration_ms"] = int((time.monotonic() - begin) * 1000)
        rec["ok"] = error is None
        rec["error"] = error
        fxlla_events.append("graph", rec)
    except Exception:
        pass


def _error_line(code):
    text = str(code).strip().splitlines()
    first = text[0] if text else ""
    home = os.path.expanduser("~")
    if home and home != "/":
        first = first.replace(home, "~")
    return first[:200]


def _run_recorded(run, args):
    """Run a query or index command and record it. Output and exit unchanged."""
    is_index = args.cmd == "index"
    fields = ({"repo": os.path.basename(os.path.normpath(args.paths[0])),
               "files": None, "defs": None, "refs": None}
              if is_index else {"query": args.cmd, "results": None})
    begin = time.monotonic()
    error = None
    try:
        got = run(args)
        if is_index and got:
            fields.update(files=got[0], defs=got[1], refs=got[2])
        elif not is_index:
            fields["results"] = got
    except SystemExit as e:
        if e.code not in (None, 0):
            error = _error_line(e.code) or "exit %s" % (e.code,)
        raise
    except BaseException as e:
        error = _error_line("%s: %s" % (type(e).__name__, e)) or type(e).__name__
        raise
    finally:
        _record("index" if is_index else "query", fields, begin, error)


def main():
    p = argparse.ArgumentParser(prog="fxlla-graph")
    sub = p.add_subparsers(dest="cmd", required=True)
    ix = sub.add_parser("index")
    ix.add_argument("paths", nargs="+")
    for name in ("def", "refs", "callers"):
        sp = sub.add_parser(name)
        sp.add_argument("name")
        sp.add_argument("-j", "--json", action="store_true")
    im = sub.add_parser("impact")
    im.add_argument("name")
    im.add_argument("--depth", type=int, default=5)
    im.add_argument("-j", "--json", action="store_true")
    un = sub.add_parser("unused")
    un.add_argument("-j", "--json", action="store_true")
    st = sub.add_parser("stats")
    st.add_argument("-j", "--json", action="store_true")
    sub.add_parser("ls")
    sub.add_parser("rm")
    args = p.parse_args()
    run = {
        "index": cmd_index, "ls": cmd_ls, "rm": cmd_rm,
        "def": cmd_def, "refs": cmd_refs, "callers": cmd_callers,
        "impact": cmd_impact, "unused": cmd_unused, "stats": cmd_stats,
    }[args.cmd]
    if args.cmd in ("index", "def", "refs", "callers", "impact", "unused"):
        _run_recorded(run, args)
    else:
        run(args)


if __name__ == "__main__":
    if not STORE or not os.path.isdir(STORE):
        sys.exit("FXLLA_STORE is not set or does not exist")
    main()
