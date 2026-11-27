"""Read-only views of kb, graph and media activity for /monitor.

Sources:
- kb.jsonl and graph.jsonl in the fxlla state dir, through lib/fxlla_events.
- <store>/kb/kb.db, opened read-only; never created.
- <store>/graph/graph.kuzu, only its mtime (kuzu is never opened).
- <store>/media/jobs/*.json (or $FXLLA_MEDIA_OUT/jobs), never written.

Every section degrades to empty or None on any failure.
"""
import glob
import json
import os
import re
import sqlite3
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "lib"))
import fxlla_events  # noqa: E402

EVENT_TTL_S = 2.0
DB_TTL_S = 10.0
JOBS_TTL_S = 2.0
TAIL_MAX = 5000
RECENT_EVENTS = 50
RECENT_JOBS = 20
ERROR_MAX = 200
KB_BUSY_S = 0.2

# A URL keeps its scheme and host; the path, query and credentials go.
_URL_RE = re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.-]*://)(?:[^\s'\"/@]*@)?([^\s'\"/?#]*)[^\s'\"]*")
# A path token: absolute, home-relative or relative, with at least one slash.
# A segment may hold up to three space-separated words ("My Drive"), so a
# macOS path with spaces is cut whole. Only its basename is ever shown.
_SEG = r"[^/\s'\":]+(?: [^/\s'\":]+){0,2}"
_PATH_RE = re.compile(
    r"(?<![\w.~/:@+-])(?:~[\w.-]*|\.{1,2}|[\w.@+-]+)?/(?:(?:%s)?/)*(?:%s)?"
    % (_SEG, _SEG))

# The only fields a media item may expose. argv, summary, output and log hold
# the prompt and home-directory paths and are never copied.
MEDIA_FIELDS = ("id", "job", "media", "model", "steps", "width", "height",
                "status", "created", "started", "finished", "peak_rss_mb",
                "progress", "caller", "error")

_KB_FIELDS = ("ts", "kind", "kb", "source", "hits", "top_score", "chunks_added",
              "duration_ms", "embed_cold", "ok", "error", "caller")
_GRAPH_FIELDS = ("ts", "kind", "query", "repo", "results", "files", "defs",
                 "refs", "duration_ms", "ok", "error", "caller")
_FINISHED = ("done", "failed", "cancelled", "died")


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, OverflowError, ValueError, TypeError):
        return False
    return True


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def _scalar(v):
    return v if v is None or isinstance(v, (bool, int, float, str)) else None


def _basename(match):
    tok = match.group(0)
    name = os.path.basename(tok.rstrip("/"))
    return name if name else tok


def _clean_error(v):
    if not isinstance(v, str) or not v:
        return None
    line = v.strip().splitlines()[0] if v.strip() else ""
    line = _URL_RE.sub(lambda m: m.group(1) + m.group(2), line)
    line = _PATH_RE.sub(_basename, line)
    return line[:ERROR_MAX]


def _pick(rec, fields):
    out = {}
    for k in fields:
        v = rec.get(k)
        out[k] = _clean_error(v) if k == "error" else _scalar(v)
    return out


def _avg_ms(recs):
    vals = [_num(r.get("duration_ms")) for r in recs]
    vals = [v for v in vals if v is not None]
    return int(round(sum(vals) / len(vals))) if vals else None


def _start_of_day(now):
    t = time.localtime(now)
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))


def _today(recs, now):
    start = _start_of_day(now)
    return [r for r in recs if (_num(r.get("ts")) or 0) >= start]


def empty_snapshot():
    return {"kb": {"recent": [],
                   "today": {"searches": 0, "adds": 0, "avg_search_ms": None,
                             "cold_starts": 0},
                   "bases": [], "db_bytes": None},
            "graph": {"recent": [],
                      "today": {"queries": 0, "indexes": 0,
                                "avg_query_ms": None},
                      "stats": {"files": None, "defs": None, "refs": None,
                                "indexed_at": None}},
            "media": {"running": [], "queued": [], "recent": [],
                      "today": {"jobs": 0, "done": 0, "failed": 0,
                                "gpu_s": 0}}}


class ToolsReader:
    def __init__(self, store, clock=time.time, alive=_pid_alive):
        self.store = store
        self.clock = clock
        self.alive = alive
        self._cache = {}
        self._kb_good = ([], None)

    def _cached(self, key, ttl, fn):
        now = self.clock()
        hit = self._cache.get(key)
        if hit is not None and 0 <= now - hit[0] < ttl:
            return hit[1]
        val = fn()
        self._cache[key] = (now, val)
        return val

    # kb and graph event streams

    def _kb_events(self):
        return self._cached("kb", EVENT_TTL_S,
                            lambda: fxlla_events.read_tail("kb", TAIL_MAX))

    def _graph_events(self):
        return self._cached("graph", EVENT_TTL_S,
                            lambda: fxlla_events.read_tail("graph", TAIL_MAX))

    def _kb_db(self):
        """(bases, db_bytes, stale). A failed read is cached for the same TTL
        as a good one, and the last good value is served meanwhile."""
        now = self.clock()
        hit = self._cache.get("kbdb")
        if hit is not None and 0 <= now - hit[0] < DB_TTL_S:
            return hit[1]
        try:
            bases, size = self._read_kb_db()
            val = (bases, size, False)
            self._kb_good = (bases, size)
        except Exception:
            bases, size = self._kb_good
            val = (bases, size, True)
        self._cache["kbdb"] = (now, val)
        return val

    def _read_kb_db(self):
        path = os.path.join(self.store, "kb", "kb.db")
        if not os.path.isfile(path):
            return [], None
        size = os.path.getsize(path)
        con = sqlite3.connect("file:%s?mode=ro" % _uri_path(path), uri=True,
                              timeout=KB_BUSY_S)
        try:
            con.execute("PRAGMA busy_timeout=%d" % int(KB_BUSY_S * 1000))
            rows = con.execute(
                "SELECT kb, COUNT(DISTINCT source), COUNT(*) FROM chunks "
                "GROUP BY kb ORDER BY kb").fetchall()
        finally:
            con.close()
        return [{"kb": r[0], "sources": r[1], "chunks": r[2]}
                for r in rows], size

    def kb(self):
        out = {"recent": [],
               "today": {"searches": 0, "adds": 0, "avg_search_ms": None,
                         "cold_starts": 0},
               "bases": [], "db_bytes": None}
        try:
            events = self._kb_events()
            out["recent"] = [_pick(r, _KB_FIELDS)
                             for r in events[:RECENT_EVENTS]]
            today = _today(events, self.clock())
            searches = [r for r in today if r.get("kind") == "search"]
            out["today"] = {
                "searches": len(searches),
                "adds": sum(1 for r in today if r.get("kind") == "add"),
                "avg_search_ms": _avg_ms(searches),
                "cold_starts": sum(1 for r in searches
                                   if r.get("embed_cold") is True)}
        except Exception:
            pass
        try:
            out["bases"], out["db_bytes"], stale = self._kb_db()
            if stale:
                out["stale"] = True
        except Exception:
            pass
        return out

    def graph(self):
        out = {"recent": [],
               "today": {"queries": 0, "indexes": 0, "avg_query_ms": None},
               "stats": {"files": None, "defs": None, "refs": None,
                         "indexed_at": None}}
        try:
            events = self._graph_events()
            out["recent"] = [_pick(r, _GRAPH_FIELDS)
                             for r in events[:RECENT_EVENTS]]
            today = _today(events, self.clock())
            queries = [r for r in today if r.get("kind") == "query"]
            out["today"] = {
                "queries": len(queries),
                "indexes": sum(1 for r in today if r.get("kind") == "index"),
                "avg_query_ms": _avg_ms(queries)}
            for r in events:
                if r.get("kind") == "index" and r.get("ok") is True:
                    out["stats"].update({k: _num(r.get(k))
                                         for k in ("files", "defs", "refs")})
                    break
        except Exception:
            pass
        try:
            mt = os.stat(os.path.join(self.store, "graph",
                                      "graph.kuzu")).st_mtime
            out["stats"]["indexed_at"] = int(mt)
        except Exception:
            pass
        return out

    # media

    def _jobs_dir(self):
        base = os.environ.get("FXLLA_MEDIA_OUT") or os.path.join(
            self.store, "media")
        return os.path.join(base, "jobs")

    def _jobs(self):
        return self._cached("jobs", JOBS_TTL_S, self._read_jobs)

    def _read_jobs(self):
        recs = []
        for path in glob.glob(os.path.join(self._jobs_dir(), "*.json")):
            try:
                with open(path, encoding="utf-8") as fh:
                    rec = json.load(fh)
            except Exception:
                continue
            if isinstance(rec, dict) and isinstance(rec.get("id"), str):
                recs.append(rec)
        return recs

    def _status(self, rec):
        st = rec.get("status")
        pid = rec.get("pid")
        if st in ("queued", "running") and isinstance(pid, int) \
                and not isinstance(pid, bool) and pid > 0 \
                and not self.alive(pid):
            return "died"
        return st if isinstance(st, str) else None

    def _item(self, rec, status, now):
        item = _pick(rec, MEDIA_FIELDS)
        item["status"] = status
        for k in ("created", "started", "finished", "peak_rss_mb", "steps",
                  "width", "height"):
            item[k] = _num(rec.get(k))
        if isinstance(item["model"], str) and item["model"][:1] in "/~":
            item["model"] = os.path.basename(item["model"].rstrip("/"))
        item["progress"] = _progress(rec.get("progress"))
        return item

    def media(self):
        out = {"running": [], "queued": [], "recent": [],
               "today": {"jobs": 0, "done": 0, "failed": 0, "gpu_s": 0}}
        try:
            now = self.clock()
            start = _start_of_day(now)
            finished = []
            for rec in self._jobs():
                try:
                    status = self._status(rec)
                    item = self._item(rec, status, now)
                    if status == "running":
                        t0 = item["started"] or item["created"]
                        item["elapsed_s"] = int(now - t0) if t0 else None
                        item["eta_s"] = _eta(item["elapsed_s"],
                                             item["progress"])
                        out["running"].append(item)
                    elif status == "queued":
                        out["queued"].append(item)
                    elif status in _FINISHED:
                        finished.append(item)
                except Exception:
                    continue
            out["running"].sort(key=lambda i: i["created"] or 0)
            out["queued"].sort(key=lambda i: i["created"] or 0)
            finished.sort(key=lambda i: i["finished"] or i["created"] or 0,
                          reverse=True)
            out["recent"] = finished[:RECENT_JOBS]
            today = [i for i in finished
                     if (i["finished"] or i["created"] or 0) >= start]
            gpu = 0.0
            for i in today:
                if i["started"] and i["finished"] and i["finished"] > i["started"]:
                    gpu += i["finished"] - i["started"]
            out["today"] = {
                "jobs": len(today),
                "done": sum(1 for i in today if i["status"] == "done"),
                "failed": sum(1 for i in today
                              if i["status"] in ("failed", "died")),
                "gpu_s": int(round(gpu))}
        except Exception:
            pass
        return out

    def snapshot(self):
        return {"kb": self.kb(), "graph": self.graph(), "media": self.media()}


def _progress(p):
    if not isinstance(p, dict):
        return None
    step, total = _num(p.get("step")), _num(p.get("total"))
    if step is None or total is None:
        return None
    return {"step": step, "total": total, "at": _num(p.get("at"))}


def _eta(elapsed, progress):
    if elapsed is None or not progress:
        return None
    step, total = progress["step"], progress["total"]
    if step <= 0 or total < step:
        return None
    return int(round(elapsed / step * (total - step)))


def _uri_path(path):
    from urllib.parse import quote
    return quote(os.path.abspath(path))
