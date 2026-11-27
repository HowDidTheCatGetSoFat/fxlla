#!/usr/bin/env python3
"""fxlla RAG: index files into a local knowledge base and search them.

Storage is a SQLite database under <store>/kb. Embeddings come from a local
llama.cpp embedding server running the catalog model selected by
FXLLA_EMBED_MODEL (default: the 'embed' alias). Standard library only,
plus llama-server for embeddings.

Search defaults to a brute-force cosine scan. Set FXLLA_KB_INDEX=1 and run this
module under a python that can load extensions (e.g. `uv run --with sqlite-vec`)
to use a sqlite-vec KNN index instead; the index is rebuilt from the chunks table
on demand and the code silently falls back to the scan when unavailable.

Usage (normally driven via `fxlla kb`):
  core.py add <kb> <path...>    index files or directories into a knowledge base
  core.py search <kb> <query>   top-k chunks for a query
  core.py ls                    list knowledge bases
  core.py rm <kb>               delete a knowledge base
  core.py reindex <kb>          re-embed every chunk with the current embedding model
  core.py eval                  score retrieval quality on the repo's own docs
"""
import argparse
import contextlib
import fcntl
import glob
import hashlib
import json
import math
import os
import re
import sqlite3
import http.client
import struct
import subprocess
import sys
import time
import urllib.request

sys.path.append(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
try:
    import fxlla_events
except Exception:  # telemetry is optional: never block the tool on it
    fxlla_events = None

STORE = os.environ.get("FXLLA_STORE", "")
KB_DIR = os.path.join(STORE, "kb")
DB_PATH = os.path.join(KB_DIR, "kb.db")
EMBED_PORT = int(os.environ.get("FXLLA_EMBED_PORT", "8090"))

# Which catalog alias supplies the embeddings. `fxlla pull <alias>` names the
# directory after the alias, so choosing a model is naming one. Unset means the
# original `embed` alias, which is what every base built so far was indexed with.
_EMBED_ENV = os.environ.get("FXLLA_EMBED_MODEL", "").strip()
EMBED_ALIAS = _EMBED_ENV or "embed"


def _index_enabled():
    return os.environ.get("FXLLA_KB_INDEX", "").lower() in ("1", "true", "yes", "on")


def _embed_dir():
    # The alias becomes a path component, so it may not walk out of the store.
    if EMBED_ALIAS in (".", "..") or not re.match(r"^[A-Za-z0-9._-]+$", EMBED_ALIAS):
        sys.exit("invalid FXLLA_EMBED_MODEL '%s' (use letters, digits, . _ -)"
                 % EMBED_ALIAS)
    return os.path.join(STORE, "models", EMBED_ALIAS)


def _embed_model():
    """Weights of the selected model, or None when none are installed.

    `fxlla pull` records the file it fetched in `.entry`. Trusting that keeps the
    choice deterministic when a directory ends up holding two quants of the same
    model, where globbing would silently take whichever sorts first.
    """
    directory = _embed_dir()
    try:
        with open(os.path.join(directory, ".entry"), encoding="utf-8") as fh:
            entry = os.path.basename(fh.read().strip())
        if entry:
            path = os.path.join(directory, entry)
            if os.path.isfile(path):
                return path
    except OSError:
        pass
    ggufs = sorted(glob.glob(os.path.join(directory, "*.gguf")))
    return ggufs[0] if ggufs else None


# The poll interval is what decides the cost of a query: the server answers in
# well under a second, so at a one-second interval every search slept through a
# server that was already up. The deadline is generous instead, because a larger
# model on an external disk can genuinely take a while to load.
_READY_TIMEOUT = 180.0
_POLL_INTERVAL = 0.02
# A loaded machine can be slow to answer /health. Too short a timeout reports a
# working server as absent, which then fails the query outright.
_HEALTH_TIMEOUT = 5.0


def _server_healthy(timeout=_HEALTH_TIMEOUT):
    try:
        urllib.request.urlopen(
            f"http://127.0.0.1:{EMBED_PORT}/health", timeout=timeout).read()
        return True
    except Exception:
        return False


def _server_model(timeout=_HEALTH_TIMEOUT):
    """Weights the running server reports holding, or None when it will not say.

    llama.cpp answers /props with the path it loaded. Another embedding server,
    or an older build, leaves us unable to tell - which is not the same as a
    mismatch, so the caller treats the two differently.
    """
    try:
        props = json.load(urllib.request.urlopen(
            f"http://127.0.0.1:{EMBED_PORT}/props", timeout=timeout))
    except Exception:
        return None
    if not isinstance(props, dict):
        return None
    # model_path only. model_alias looks like a usable fallback and is a trap:
    # llama-server's --alias replaces it with a name of the operator's choosing,
    # which no path can ever equal, so the one case it would cover - a server
    # fxlla did not start - is the case most likely to be aliased. Treating a
    # missing model_path as unverifiable keeps that server working.
    return props.get("model_path") or None


# A server outlives the command that started it, so switching models leaves the
# previous one listening. Reuse used to adopt whatever answered: with no model
# installed at all, add and search both succeeded against a borrowed server. That
# was harmless while one model was possible and is not any more - nomic and
# embeddinggemma are both 768-dim, qwen3-embedding and bge-large both 1024, so
# _require_dim cannot see the difference and the store is silently poisoned.
def _require_server_model(wanted):
    running = _server_model()
    if running is not None and wanted is not None:
        if os.path.realpath(running) != os.path.realpath(wanted):
            sys.exit(
                "the embedding server on port %d holds %s, but this command is "
                "configured for %s. Vectors from two models are not comparable "
                "and, at equal width, nothing downstream can detect the mix. "
                "Stop it with 'fxlla kb stop', or point FXLLA_EMBED_PORT at the "
                "right server." % (EMBED_PORT, running, wanted))
        return
    # Unverifiable. Pointing FXLLA_EMBED_PORT at a server fxlla did not start is
    # a legitimate setup, so this proceeds - but only says so out loud once the
    # user is actually choosing between models, where being wrong costs a store.
    if _EMBED_ENV and running is None:
        print("warning: the embedding server on port %d does not report which "
              "model it loaded, so '%s' could not be confirmed."
              % (EMBED_PORT, EMBED_ALIAS), file=sys.stderr)


# Runs a local llama.cpp embedding server for the lifetime of the context, or
# reuses one that is already listening.
class Embedder:
    # True when this instance launched the embedding server (a cold start),
    # false when it reused one that was already listening.
    started = False

    def __enter__(self):
        # A server someone else started is not ours to stop. Reusing it is also
        # what makes repeated queries cost milliseconds: leave one running and
        # searches skip startup entirely.
        self.proc = None
        self._lock = None
        wanted = _embed_model()
        # Naming a model that is not installed is a mistake worth reporting on its
        # own: otherwise a leftover server answers and the search quietly runs on
        # the previous model instead of the one that was asked for.
        if wanted is None and _EMBED_ENV:
            sys.exit("FXLLA_EMBED_MODEL selects '%s' but no weights are installed "
                     "at %s. Run: fxlla pull %s"
                     % (EMBED_ALIAS, _embed_dir(), EMBED_ALIAS))
        if _server_healthy():
            _require_server_model(wanted)
            return self
        # Serialize starting one. Without this, concurrent searches each spawn a
        # server, all but one fail to bind, and the losers have to guess whether
        # the port is held by a peer or by something unrelated. Holding the lock
        # makes that unambiguous: nobody else is starting one, so if ours dies the
        # failure is real.
        os.makedirs(KB_DIR, exist_ok=True)
        self._lock = open(os.path.join(KB_DIR, "embed.lock"), "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX)
            if _server_healthy():  # started while we waited for the lock
                _require_server_model(wanted)
                return self
            model = wanted if wanted is not None else _embed_model()
            if not model:
                sys.exit("embedding model not found. Run: fxlla pull embed --quant Q5_K_M")
            # No --pooling: every embedding GGUF declares its own pooling_type and
            # llama.cpp honours it. Forcing mean was right for nomic and wrong for
            # the rest - bge wants CLS, qwen3-embedding wants last-token - which
            # degrades them silently, since a badly pooled vector is still a
            # vector. Verified: omitting the flag reproduces --pooling mean
            # bit-for-bit on nomic, while --pooling cls visibly differs.
            self.proc = subprocess.Popen(
                ["llama-server", "--embeddings", "-m", model, "--host", "127.0.0.1",
                 "--port", str(EMBED_PORT)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.started = True
            deadline = time.time() + _READY_TIMEOUT
            while time.time() < deadline:
                if _server_healthy():
                    return self
                if self.proc.poll() is not None:
                    self.proc = None
                    sys.exit("the embedding server exited on start. Port %d may be "
                             "held by something else, or the model may be unreadable."
                             % EMBED_PORT)
                time.sleep(_POLL_INTERVAL)
            self.__exit__(None, None, None)
            sys.exit("embedding server did not become ready")
        finally:
            self._release_lock()

    def _release_lock(self):
        if self._lock is not None:
            try:
                fcntl.flock(self._lock, fcntl.LOCK_UN)
                self._lock.close()
            except Exception:
                pass
            self._lock = None

    def _post(self, texts):
        body = json.dumps({"input": texts}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{EMBED_PORT}/v1/embeddings", data=body,
            headers={"Content-Type": "application/json"})
        data = json.load(urllib.request.urlopen(req, timeout=300))
        return [d["embedding"] for d in data["data"]]

    def embed(self, texts):
        try:
            return self._post(texts)
        # HTTPError subclasses OSError, so it has to be caught first or the retry
        # below swallows it - and retrying a rejected request is pointless anyway.
        except urllib.error.HTTPError as exc:
            sys.exit("the server on port %d rejected an embedding request (%s). Is "
                     "it an embedding server, started with --embeddings?"
                     % (EMBED_PORT, exc))
        except (http.client.RemoteDisconnected, ConnectionError, OSError) as first:
            # A borrowed server can be stopped by its owner while our request is
            # in flight. Re-acquire - starting one if needed - and try once more,
            # rather than dying with a traceback halfway through an index.
            self.__enter__()
            try:
                return self._post(texts)
            except Exception:
                sys.exit("the embedding server on port %d stopped responding (%s)"
                         % (EMBED_PORT, first))

    def __exit__(self, *_a):
        self._release_lock()
        if self.proc is None:
            return  # someone else's server: leave it running
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


def _db():
    os.makedirs(KB_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "CREATE TABLE IF NOT EXISTS chunks "
        "(kb TEXT, source TEXT, idx INTEGER, text TEXT, emb BLOB)"
    )
    con.execute("CREATE INDEX IF NOT EXISTS chunks_kb ON chunks(kb)")
    return con


# Vector width already stored for a knowledge base, or None when it is empty.
def _kb_dim(con, kb):
    row = con.execute(
        "SELECT length(emb) FROM chunks WHERE kb=? LIMIT 1", (kb,)).fetchone()
    return row[0] // 4 if row else None


# Embeddings from a different model are not comparable with the stored ones, and
# _cosine silently zips to the shorter vector, so a mismatch does not fail - it
# quietly returns nonsense and, on add, mixes widths into one base permanently.
# The reachable cause is a server on EMBED_PORT belonging to another model.
def _require_dim(con, kb, vector):
    stored = _kb_dim(con, kb)
    if stored is not None and len(vector) != stored:
        sys.exit(
            "the embedding server returned %d dimensions but '%s' holds %d. That "
            "server is a different model than the one this base was built with. "
            "Either point FXLLA_EMBED_MODEL back at the model it was built with, "
            "or re-embed it with the current one: fxlla kb reindex %s"
            % (len(vector), kb, stored, kb))


def _valid_kb(name):
    if not re.match(r"^[A-Za-z0-9._-]+$", name or ""):
        sys.exit("invalid knowledge base name (use letters, digits, . _ -)")
    return name


def chunk_text(text, size=800, overlap=120):
    text = text.strip()
    out = []
    i = 0
    while i < len(text):
        piece = text[i:i + size]
        if piece.strip():
            out.append(piece)
        i += max(size - overlap, 1)
    return out


# Telemetry: one record per kb command in <state dir>/kb.jsonl, written after the
# command finishes. Only counts, timings and the base name are recorded, never
# query text, chunk text or paths. Commands fill in what they know with _note().
_EVENT = {}


def _note(**fields):
    _EVENT.update(fields)


def _error_line(code):
    text = str(code).strip().splitlines()
    first = text[0] if text else ""
    home = os.path.expanduser("~")
    if home and home != "/":
        first = first.replace(home, "~")
    return first[:200]


@contextlib.contextmanager
def _recorded(kind, defaults):
    """Time a command and append its kb event. Never changes what it does."""
    _EVENT.clear()
    _EVENT.update(defaults)
    begin = time.monotonic()
    error = None
    try:
        yield
    except SystemExit as e:
        if e.code not in (None, 0):
            error = _error_line(e.code) or "exit %s" % (e.code,)
        raise
    except BaseException as e:
        error = _error_line("%s: %s" % (type(e).__name__, e)) or type(e).__name__
        raise
    finally:
        try:
            if fxlla_events is not None:
                rec = {"kind": kind}
                rec.update(_EVENT)
                rec["duration_ms"] = int((time.monotonic() - begin) * 1000)
                rec["ok"] = error is None
                rec["error"] = error
                fxlla_events.append("kb", rec)
        except Exception:
            pass


def cmd_ls(_args):
    con = _db()
    rows = con.execute(
        "SELECT kb, COUNT(DISTINCT source), COUNT(*) FROM chunks GROUP BY kb"
    ).fetchall()
    if not rows:
        print("(no knowledge bases)")
        return
    for kb, sources, chunks in rows:
        print(f"{kb}\t{sources} sources\t{chunks} chunks")


def cmd_rm(args):
    _valid_kb(args.name)
    con = _db()
    n = con.execute("DELETE FROM chunks WHERE kb=?", (args.name,)).rowcount
    # Dropping a vec0 virtual table needs the extension loaded on this connection.
    # Load it best-effort; if it is unavailable the stale index is left behind and
    # a later search rebuilds it once the extension is present.
    _load_vec_raw(con)
    try:
        con.execute("DROP TABLE IF EXISTS %s" % _vec_table(args.name))
    except sqlite3.OperationalError:
        pass
    con.commit()
    print(f"removed {n} chunks from '{args.name}'")


TEXT_EXTS = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".swift", ".sh", ".json",
    ".yaml", ".yml", ".toml", ".html", ".css", ".go", ".rs", ".java",
    ".c", ".h", ".cpp", ".hpp",
}


def _pack(vec):
    return struct.pack(f"{len(vec)}f", *vec)


def _gather(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, names in os.walk(p):
                if (os.sep + ".git" + os.sep) in (root + os.sep) or root.endswith(os.sep + ".git"):
                    continue
                for n in names:
                    if os.path.splitext(n)[1].lower() in TEXT_EXTS:
                        files.append(os.path.join(root, n))
        elif os.path.isfile(p):
            files.append(p)
    return files


def cmd_add(args):
    _valid_kb(args.name)
    files = _gather(args.paths)
    if not files:
        sys.exit("no text files found in the given paths")
    con = _db()
    total = 0
    with Embedder() as emb:
        for f in files:
            try:
                with open(f, encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            except Exception:
                continue
            chunks = chunk_text(text)
            if not chunks:
                continue
            # embed first: on failure the existing chunks are left intact
            vecs = emb.embed(chunks)
            _require_dim(con, args.name, vecs[0])
            con.execute("DELETE FROM chunks WHERE kb=? AND source=?", (args.name, f))
            con.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?)",
                [(args.name, f, i, c, _pack(v)) for i, (c, v) in enumerate(zip(chunks, vecs))],
            )
            con.commit()
            total += len(chunks)
            _note(chunks_added=total)
            print(f"  {f}: {len(chunks)} chunks")
    print(f"indexed {len(files)} files, {total} chunks into '{args.name}'")


# Chunk text is stored alongside the vector, so switching models re-embeds what
# is already in the table instead of re-reading sources that may have moved or
# changed since. Batched to keep one request from carrying a whole base.
_REINDEX_BATCH = 64


def cmd_reindex(args):
    _valid_kb(args.name)
    con = _db()
    rows = con.execute(
        "SELECT rowid, text FROM chunks WHERE kb=? ORDER BY rowid", (args.name,)
    ).fetchall()
    if not rows:
        sys.exit(f"knowledge base '{args.name}' is empty")
    before = _kb_dim(con, args.name)
    done = 0
    # One transaction: a base half re-embedded holds two widths at once, and
    # _kb_dim reads whichever row comes first, so an interrupted run would leave
    # a store that lies about itself. Committing once makes a crash a no-op.
    with Embedder() as emb:
        for i in range(0, len(rows), _REINDEX_BATCH):
            batch = rows[i:i + _REINDEX_BATCH]
            vecs = emb.embed([text for _rid, text in batch])
            con.executemany(
                "UPDATE chunks SET emb=? WHERE rowid=?",
                [(_pack(v), rid) for (rid, _t), v in zip(batch, vecs)])
            done += len(batch)
            print(f"  {done}/{len(rows)} chunks", flush=True)
    # The vec0 table bakes its width in at creation, so a width change makes the
    # existing index unusable. Dropping it inside the same transaction leaves the
    # next search to rebuild it from the new vectors.
    _load_vec_raw(con)
    try:
        con.execute("DROP TABLE IF EXISTS %s" % _vec_table(args.name))
    except sqlite3.OperationalError:
        pass
    con.commit()
    after = _kb_dim(con, args.name)
    change = f"{before} -> {after} dimensions" if before != after else f"{after} dimensions"
    print(f"reindexed {len(rows)} chunks in '{args.name}' with {EMBED_ALIAS} ({change})")


def _unpack(blob):
    return struct.unpack(f"{len(blob) // 4}f", blob)


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# Loads the sqlite-vec extension into a connection regardless of FXLLA_KB_INDEX.
# Returns the module on success, else None. Used by maintenance paths (rm) that
# must touch a vec0 table even when the index is not the active search backend.
def _load_vec_raw(con):
    try:
        import sqlite_vec
    except Exception:
        return None
    try:
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
    except Exception:
        return None
    return sqlite_vec


# Loads sqlite-vec only when the index is opted in (FXLLA_KB_INDEX) and a python
# that permits loading extensions is in use (e.g. `uv run --with sqlite-vec`).
# Returns None otherwise, so the caller falls back to the brute-force scan.
def _load_vec(con):
    if not _index_enabled():
        return None
    return _load_vec_raw(con)


def _vec_table(kb):
    # kb is validated against [A-Za-z0-9._-]+, so double-quoting is safe.
    return '"vec_%s"' % kb.replace('"', '')


# Rebuilds a per-kb vec0 index from the chunks table when it is missing or stale
# (row count differs). The chunks table stays the source of truth; the index only
# stores each chunk's rowid and embedding, so a rebuild never re-embeds anything.
def _ensure_vec_index(con, vec, kb):
    rows = con.execute("SELECT rowid, emb FROM chunks WHERE kb=?", (kb,)).fetchall()
    if not rows:
        return 0
    table = _vec_table(kb)
    dim = len(rows[0][1]) // 4
    try:
        have = con.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0]
    except sqlite3.OperationalError:
        have = None
    if have == len(rows):
        return dim
    con.execute("DROP TABLE IF EXISTS %s" % table)
    con.execute(
        "CREATE VIRTUAL TABLE %s USING vec0(embedding float[%d] distance_metric=cosine)"
        % (table, dim)
    )
    con.executemany(
        "INSERT INTO %s(rowid, embedding) VALUES (?, ?)" % table,
        [(rid, vec.serialize_float32(list(_unpack(blob)))) for rid, blob in rows],
    )
    con.commit()
    return dim


# KNN search through the vec0 index. Cosine distance is 1 - cosine similarity, so
# the reported score matches the brute-force _cosine path.
def _search_indexed(con, vec, kb, qv, k):
    _ensure_vec_index(con, vec, kb)
    table = _vec_table(kb)
    hits = con.execute(
        "SELECT rowid, distance FROM %s WHERE embedding MATCH ? AND k = ? "
        "ORDER BY distance" % table,
        (vec.serialize_float32(list(qv)), k),
    ).fetchall()
    out = []
    for rowid, distance in hits:
        row = con.execute(
            "SELECT source, idx, text FROM chunks WHERE rowid=?", (rowid,)
        ).fetchone()
        if row:
            out.append((1.0 - distance, row[0], row[1], row[2]))
    return out


# Ranked hits for one query vector. `vec` is the loaded sqlite-vec module or
# None; the caller loads it once rather than per query, which matters when a
# whole eval run goes through here.
def _top_k(con, kb, qv, k, vec):
    if vec is not None:
        return _search_indexed(con, vec, kb, qv, k)
    rows = con.execute(
        "SELECT source, idx, text, emb FROM chunks WHERE kb=?", (kb,)).fetchall()
    scored = [
        (_cosine(qv, _unpack(blob)), source, idx, text)
        for source, idx, text, blob in rows
    ]
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:k]


def cmd_search(args):
    _valid_kb(args.name)
    con = _db()
    n = con.execute("SELECT COUNT(*) FROM chunks WHERE kb=?", (args.name,)).fetchone()[0]
    if not n:
        sys.exit(f"knowledge base '{args.name}' is empty")
    with Embedder() as emb:
        _note(embed_cold=bool(emb.started))
        qv = emb.embed([args.query])[0]
    _require_dim(con, args.name, qv)
    top = _top_k(con, args.name, qv, args.k, _load_vec(con))
    _note(hits=len(top), top_score=round(top[0][0], 4) if top else None)
    if args.json:
        print(json.dumps([
            {"score": round(s, 4), "source": src, "chunk": idx, "text": txt}
            for s, src, idx, txt in top
        ]))
        return
    for score, source, idx, text in top:
        print(f"[{score:.3f}] {source}#{idx}")
        print("  " + " ".join(text.split())[:200])


# Retrieval quality, so switching embedding models is a measurement rather than a
# guess. The corpus is the repository's own documentation: real prose about
# distinct topics, nothing invented, nothing fetched.
EVAL_SET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Its own base, so a run never disturbs a real one. Dropped again at the end.
EVAL_KB = "_eval"


def _eval_index(con, emb, files):
    con.execute("DELETE FROM chunks WHERE kb=?", (EVAL_KB,))
    _load_vec_raw(con)
    try:
        con.execute("DROP TABLE IF EXISTS %s" % _vec_table(EVAL_KB))
    except sqlite3.OperationalError:
        pass
    total = 0
    for path in files:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            chunks = chunk_text(fh.read())
        if not chunks:
            continue
        vecs = emb.embed(chunks)
        con.executemany(
            "INSERT INTO chunks VALUES (?,?,?,?,?)",
            [(EVAL_KB, path, i, c, _pack(v))
             for i, (c, v) in enumerate(zip(chunks, vecs))])
        total += len(chunks)
    con.commit()
    return total


def _eval_drop(con):
    con.execute("DELETE FROM chunks WHERE kb=?", (EVAL_KB,))
    _load_vec_raw(con)
    try:
        con.execute("DROP TABLE IF EXISTS %s" % _vec_table(EVAL_KB))
    except sqlite3.OperationalError:
        pass
    con.commit()


def _corpus_fingerprint(files):
    """Short digest of the corpus, so two runs can be told apart.

    The corpus is live documentation: editing it moves every score. Reporting the
    fingerprint turns "are these two numbers comparable" from a judgement call
    into a string comparison.
    """
    digest = hashlib.sha256()
    for path in files:
        with open(path, "rb") as fh:
            digest.update(fh.read())
    return digest.hexdigest()[:12]


def cmd_eval(args):
    with open(EVAL_SET, encoding="utf-8") as fh:
        spec = json.load(fh)
    files = [os.path.join(REPO_ROOT, rel) for rel in spec["corpus"]]
    missing = [rel for rel, path in zip(spec["corpus"], files) if not os.path.isfile(path)]
    if missing:
        sys.exit("the eval corpus is missing %s. It is this repository's own "
                 "documentation, so run this from a checkout." % ", ".join(missing))
    queries = spec["queries"]

    con = _db()
    model = _embed_model()
    started = time.monotonic()
    with Embedder() as emb:
        chunks = _eval_index(con, emb, files)
        indexed = time.monotonic() - started
        vec = _load_vec(con)
        results, latencies = [], []
        for item in queries:
            begin = time.monotonic()
            qv = emb.embed([item["q"]])[0]
            hits = _top_k(con, EVAL_KB, qv, args.k, vec)
            latencies.append(time.monotonic() - begin)
            ranked = [os.path.relpath(src, REPO_ROOT) for _s, src, _i, _t in hits]
            rank = next((i for i, src in enumerate(ranked, 1)
                         if src in item["expect"]), None)
            results.append({"query": item["q"], "expect": item["expect"],
                            "rank": rank, "top": ranked[:3]})
    dim = _kb_dim(con, EVAL_KB)
    if not args.keep:
        _eval_drop(con)

    n = len(results)
    at1 = sum(1 for r in results if r["rank"] == 1)
    atk = sum(1 for r in results if r["rank"] is not None)
    mrr = sum(1.0 / r["rank"] for r in results if r["rank"]) / n if n else 0.0
    latencies.sort()
    median = latencies[len(latencies) // 2] if latencies else 0.0
    summary = {
        "model": EMBED_ALIAS,
        "weights": os.path.basename(model) if model else None,
        "dimensions": dim,
        "corpus": _corpus_fingerprint(files),
        "files": len(files), "chunks": chunks,
        "index_seconds": round(indexed, 2),
        "queries": n, "k": args.k,
        "recall_at_1": round(at1 / n, 3) if n else 0.0,
        "recall_at_k": round(atk / n, 3) if n else 0.0,
        "mrr": round(mrr, 3),
        "median_query_seconds": round(median, 3),
    }
    if args.json:
        print(json.dumps({"summary": summary, "results": results}, indent=2))
        return

    print(f"model:    {EMBED_ALIAS} ({summary['weights']}, {dim} dimensions)")
    print(f"corpus:   {len(files)} files, {chunks} chunks, "
          f"fingerprint {summary['corpus']}, indexed in {indexed:.1f}s")
    print(f"queries:  {n} (k={args.k})")
    print(f"recall@1: {at1}/{n} ({at1 / n:.0%})" if n else "recall@1: n/a")
    print(f"recall@{args.k}: {atk}/{n} ({atk / n:.0%})" if n else "")
    print(f"MRR:      {mrr:.3f}")
    print(f"latency:  {median * 1000:.0f} ms median per query")
    misses = [r for r in results if r["rank"] is None]
    if misses:
        print(f"\n{len(misses)} miss(es):")
        for r in misses:
            print(f"  {r['query']}")
            print(f"    wanted {' or '.join(r['expect'])}, got {', '.join(r['top'])}")
    print("\nThese numbers rank models against each other on one corpus. The "
          "corpus is the repository's own docs, so editing them moves every "
          "score: compare two runs only when the fingerprint matches.")


# The basename of what was added; with several paths there is no single source.
def _add_source(args):
    paths = getattr(args, "paths", None) or []
    return (os.path.basename(os.path.realpath(paths[0])) or None) if len(paths) == 1 else None


def main():
    p = argparse.ArgumentParser(prog="fxlla-rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ls")
    rm = sub.add_parser("rm")
    rm.add_argument("name")
    ad = sub.add_parser("add")
    ad.add_argument("name")
    ad.add_argument("paths", nargs="+")
    se = sub.add_parser("search")
    se.add_argument("name")
    se.add_argument("query")
    se.add_argument("-k", type=int, default=5)
    se.add_argument("-j", "--json", action="store_true")
    ri = sub.add_parser("reindex")
    ri.add_argument("name")
    ev = sub.add_parser("eval")
    ev.add_argument("-k", type=int, default=5)
    ev.add_argument("-j", "--json", action="store_true")
    ev.add_argument("--keep", action="store_true",
                    help="leave the eval knowledge base in place for inspection")
    args = p.parse_args()
    run = {"ls": cmd_ls, "rm": cmd_rm, "add": cmd_add, "search": cmd_search,
           "reindex": cmd_reindex, "eval": cmd_eval}[args.cmd]
    base = {"kb": args.name} if hasattr(args, "name") else {}
    defaults = {
        "search": dict(base, hits=0, top_score=None, embed_cold=False),
        "add": dict(base, source=_add_source(args), chunks_added=0),
        "rm": dict(base, source=None),
    }.get(args.cmd)
    if defaults is None:
        run(args)
        return
    with _recorded(args.cmd, defaults):
        run(args)


if __name__ == "__main__":
    if not STORE or not os.path.isdir(STORE):
        sys.exit("FXLLA_STORE is not set or does not exist")
    main()
