"""Shared append-only JSONL event streams for fxlla (kb, graph).

Standard library only. Writers never raise; readers skip bad lines.
"""
import json
import os
import sys
import time

MAX_BYTES = 5 * 1024 * 1024
_TAIL_CHUNK = 64 * 1024


def state_dir():
    d = os.environ.get("FXLLA_STATE_DIR")
    if d:
        return d
    x = os.environ.get("XDG_STATE_HOME")
    if x:
        return os.path.join(x, "fxlla")
    return os.path.join(os.path.expanduser("~"), ".local", "state", "fxlla")


def _path(stream):
    return os.path.join(state_dir(), stream + ".jsonl")


def append(stream, record):
    """Append one JSON line to <state_dir>/<stream>.jsonl. Never raises."""
    try:
        rec = dict(record)
        rec.setdefault("ts", time.time())
        rec.setdefault("caller", os.environ.get("FXLLA_CALLER", "cli"))
        data = (json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8")
        d = state_dir()
        os.makedirs(d, mode=0o700, exist_ok=True)
        path = os.path.join(d, stream + ".jsonl")
        try:
            if os.path.getsize(path) > MAX_BYTES:
                os.replace(path, path + ".1")
        except OSError:
            pass
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
    except Exception as e:
        try:
            sys.stderr.write("fxlla_events: append failed: %s\n" % type(e).__name__)
        except Exception:
            pass


def read_tail(stream, max_lines=200):
    """Return up to max_lines newest parseable records, newest first."""
    try:
        with open(_path(stream), "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            buf = b""
            # Need max_lines complete lines, plus one newline before the first.
            while pos > 0 and buf.count(b"\n") <= max_lines + 1:
                step = min(_TAIL_CHUNK, pos)
                pos -= step
                f.seek(pos)
                buf = f.read(step) + buf
    except OSError:
        return []
    lines = buf.split(b"\n")
    if pos > 0:
        lines = lines[1:]  # first line may be cut by the seek
    out = []
    for raw in reversed(lines):
        if len(out) >= max_lines:
            break
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out
