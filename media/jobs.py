#!/usr/bin/env python3
"""Background jobs for fxlla media generation.

Heavy renders (video especially) can run for minutes, which is too long for an
MCP tool call to block on. Submitting a job returns an id immediately; the caller
polls for status and the output path.

No daemon and no dependencies: submitting writes a JSON record under
<media out>/jobs and spawns a detached worker (this module's `run`) that owns the
record for the rest of the job's life. The worker takes an exclusive flock before
starting, so media jobs run one at a time - they share unified memory with the
gateway's models and two concurrent renders would thrash or OOM the machine.
A job waiting on that lock stays 'queued'.

Usage (normally driven via `fxlla media`):
  jobs.py run <id>     execute a submitted job (the detached worker)
"""
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

STORE = os.environ.get("FXLLA_STORE", "")
OUT_DIR = os.environ.get("FXLLA_MEDIA_OUT") or os.path.join(STORE, "media")
JOBS_DIR = os.path.join(OUT_DIR, "jobs")
GENERATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate.py")

# Statuses a job can still leave on its own.
ACTIVE = ("queued", "running")
_ID_RE = re.compile(r"^[0-9]{10,}-[0-9a-f]{6}$")


def _dir():
    os.makedirs(JOBS_DIR, exist_ok=True)
    return JOBS_DIR


def valid_id(job_id):
    # Job ids reach this module from MCP tool arguments, so validate the shape
    # instead of trusting it in a path join.
    return bool(_ID_RE.match(job_id or ""))


def _path(job_id):
    return os.path.join(_dir(), job_id + ".json")


def log_path(job_id):
    return os.path.join(_dir(), job_id + ".log")


def new_id():
    return "%d-%s" % (time.time(), uuid.uuid4().hex[:6])


def _read(job_id):
    try:
        with open(_path(job_id), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


_TMP_COUNTER = [0]


def _write(rec):
    # Atomic: readers (fxlla media jobs) never see a half-written record. Each
    # write has its own tmp name (pid plus a counter), so two writers on one
    # record never share a half-written file.
    path = _path(rec["id"])
    _TMP_COUNTER[0] += 1
    tmp = "%s.%d.%d.tmp" % (path, os.getpid(), _TMP_COUNTER[0])
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rec, fh)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def lock_path(job_id):
    return _path(job_id) + ".lock"


# What a finished run reports about itself. A record already 'cancelled' keeps
# these as they are: the cancel is the person's decision, the exit is its echo.
_OUTCOME_FIELDS = ("status", "finished", "output", "error")


def _update(job_id, _keep_cancelled=False, **fields):
    """Re-read then write under a per-record lock, so concurrent updates (this
    module's status changes, generate.py's progress) do not clobber each other.
    With _keep_cancelled, a record already cancelled gets only the telemetry
    fields."""
    if not os.path.exists(_path(job_id)):
        return None
    with open(lock_path(job_id), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rec = _read(job_id)
        if rec is None:
            return None
        if _keep_cancelled and rec.get("status") == "cancelled":
            fields = {k: v for k, v in fields.items() if k not in _OUTCOME_FIELDS}
        rec.update(fields)
        _write(rec)
        return rec


def _alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reap(rec):
    """Mark a job failed when its worker vanished (crash, reboot, kill -9)."""
    if rec and rec.get("status") in ACTIVE and rec.get("pid") and not _alive(rec["pid"]):
        rec = _update(rec["id"], status="failed", finished=time.time(),
                      error="worker process died") or rec
    return rec


def submit(kind, argv, summary=""):
    """Record a job and spawn its detached worker. Returns the job record."""
    job_id = new_id()
    rec = {"id": job_id, "kind": kind, "status": "queued", "argv": list(argv),
           "summary": summary, "output": None, "error": None, "pid": None,
           "created": time.time(), "started": None, "finished": None,
           "log": log_path(job_id)}
    rec.update(_telemetry_fields(kind, True))
    _write(rec)
    with open(log_path(job_id), "ab") as log:
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "run", job_id],
            stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            start_new_session=True, env=os.environ)
    # start_new_session makes the worker a process-group leader, so cancelling it
    # can signal the whole group (worker plus the generator it spawns).
    if (_read(job_id) or {}).get("status") == "queued":
        rec = _update(job_id, pid=proc.pid) or rec
    return rec


def _telemetry_fields(media, is_job):
    """The fields the monitor reads. generate.py fills the None ones in."""
    return {"job": is_job, "media": media, "model": None, "steps": None,
            "width": None, "height": None, "peak_rss_mb": None,
            "progress": None, "caller": os.environ.get("FXLLA_CALLER", "cli")}


def update(job_id, **fields):
    """_update that never raises (telemetry must not fail a render) and never
    replaces a cancelled status."""
    try:
        return _update(job_id, _keep_cancelled=True, **fields)
    except Exception:
        return None


def begin_sync(media, fields, argv=(), summary=""):
    """Record a run that has no worker (no --async). Returns its id, or None
    when the record cannot be written, which must not stop the render."""
    try:
        now = time.time()
        job_id = new_id()
        rec = {"id": job_id, "kind": media, "status": "running",
               "argv": list(argv), "summary": summary, "output": None,
               "error": None, "pid": os.getpid(), "created": now,
               "started": now, "finished": None, "log": None}
        rec.update(_telemetry_fields(media, False))
        rec.update(fields)
        _write(rec)
        return job_id
    except Exception:
        return None


def get(job_id):
    if not valid_id(job_id):
        return None
    return _reap(_read(job_id))


def wait(job_id, timeout_s=600, poll_s=1.0):
    """Block until a job leaves the active states, or the timeout.

    Returns the record either way; the caller reads `status`. Without this an
    agent has no way to await a render and busy-polls instead: one issued 47
    `media jobs` calls waiting on a single video, which is both wasteful and
    exactly what a runaway loop looks like from outside."""
    import time as _time
    deadline = _time.monotonic() + max(0, timeout_s)
    while True:
        rec = get(job_id)
        if rec is None or rec.get("status") not in ACTIVE:
            return rec
        if _time.monotonic() >= deadline:
            return rec
        _time.sleep(poll_s)


def listing():
    jobs = []
    for name in sorted(os.listdir(_dir())):
        if not name.endswith(".json"):
            continue
        rec = _reap(_read(name[:-len(".json")]))
        if rec:
            jobs.append(rec)
    jobs.sort(key=lambda r: r.get("created") or 0, reverse=True)
    return jobs


def prune():
    """Drop finished job records (and their logs). Active jobs are kept."""
    removed = 0
    for rec in listing():
        if rec["status"] in ACTIVE:
            continue
        for path in (_path(rec["id"]), log_path(rec["id"]),
                     lock_path(rec["id"])):
            try:
                os.remove(path)
            except OSError:
                pass
        removed += 1
    return removed


def cancel(job_id):
    rec = get(job_id)
    if rec is None or rec["status"] not in ACTIVE:
        return rec
    # Record the decision before signalling, so the exit that follows finds it.
    out = _update(job_id, status="cancelled", finished=time.time())
    pid = rec.get("pid")
    if pid:
        try:
            if rec.get("job") is False:
                # A sync run is not a session leader: its process group is the
                # caller's shell pipeline or MCP server. Signal only the pid;
                # generate.py turns that SIGTERM into an exit that also stops
                # its backend child.
                os.kill(pid, signal.SIGTERM)
            else:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    return out


def run(job_id):
    """The detached worker: serialize on the lock, then run the generator."""
    if not valid_id(job_id) or _read(job_id) is None:
        sys.exit("unknown job: %s" % job_id)
    with open(os.path.join(_dir(), "lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rec = _read(job_id)
        # Cancelled while it sat in the queue: nothing left to do.
        if rec is None or rec["status"] not in ACTIVE:
            return
        _update(job_id, status="running", started=time.time(), pid=os.getpid())
        # Truncated at job start: a log is one run's output, not a history.
        # The generator's stderr goes straight into the file, and a pump copies
        # its stdout in as it arrives, so the log is live and survives a cancel
        # or a kill -9 of the worker.
        log = log_path(job_id)
        env = dict(os.environ, FXLLA_JOB_ID=job_id)
        out_lines = []
        try:
            open(log, "wb").close()
            with open(log, "ab", buffering=0) as errfile:
                proc = subprocess.Popen(
                    [sys.executable, GENERATE] + list(rec["argv"]),
                    stdout=subprocess.PIPE, stderr=errfile,
                    stdin=subprocess.DEVNULL, env=env)
                try:
                    out_lines = _pump_stdout(proc.stdout, log)
                finally:
                    proc.stdout.close()
                proc.wait()
        # A failure is announced too: eight minutes of waiting for nothing is
        # exactly what someone needs told, and silence reads as still running.
        except Exception as exc:  # the generator could not even start
            _append_log(log, "generator could not start: %s\n" % exc)
            notify(_update(job_id, status="failed", finished=time.time(),
                           error=str(exc)))
            return
        if proc.returncode != 0:
            notify(_update(job_id, status="failed", finished=time.time(),
                           error=(_log_text(log).strip() or "generation failed")[-800:]))
            return
        lines = [ln.strip() for ln in out_lines if ln.strip()]
        if not lines:
            notify(_update(job_id, status="failed", finished=time.time(),
                           error="generator returned no output path"))
            return
        # What the backend said on a run that worked. Only failures ever read
        # stderr, so mflux's own "--steps is ignored" reached nobody. The
        # generator relays the backend's raw output in a job, so only lines it
        # marked itself (WARNING_MARK) count: a backend line that happens to
        # start with "warning: " must not change this field.
        warned = []
        for ln in _log_text(log).splitlines():
            if ln.startswith(WARNING_MARK):
                warned.append(ln[len(WARNING_MARK):])
        notify(_update(job_id, status="done", finished=time.time(),
                       output=lines[-1],
                       **({"warnings": warned} if warned else {})))


# Prefix of the warning lines generate.py prints in a job (see _report_warnings).
WARNING_MARK = "fxlla-warning: "


def _append_log(path, text):
    try:
        with open(path, "ab", buffering=0) as fh:
            fh.write(text.encode("utf-8", "replace"))
    except OSError:
        pass


def _log_text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _pump_stdout(stream, log):
    """Copy the generator's stdout into the log line by line as it arrives and
    return the lines. O_APPEND keeps each write whole beside stderr's."""
    lines = []
    with open(log, "ab", buffering=0) as fh:
        for raw in iter(stream.readline, b""):
            fh.write(raw)
            lines.append(raw.decode("utf-8", "replace").rstrip("\n"))
    return lines


def _as(text):
    """A Python string as an AppleScript string literal.

    json.dumps gets the quote and backslash escaping right, which is the point
    - a prompt carries both, and one interpolated raw would end the literal and
    turn the rest of somebody's prompt into script. ensure_ascii=False because
    AppleScript has no \\uXXXX escape and would print it verbatim."""
    return json.dumps(text, ensure_ascii=False)


def notify(rec):
    """Tell the desktop the render landed.

    A tool call is request/response and an agent turn ends when the model stops
    calling tools, so nothing in MCP can reopen a finished turn to announce
    this - the model can only answer when asked. That leaves the person
    watching, and a render that takes eight minutes is one they have stopped
    watching. Best effort and silent: a headless or cron run has no desktop and
    that is not a failure of the render."""
    if not rec or sys.platform != "darwin":
        return
    if os.environ.get("FXLLA_MEDIA_NOTIFY", "1") in ("0", "false", ""):
        return
    took = ""
    if rec.get("started") and rec.get("finished"):
        took = " in %ds" % round(rec["finished"] - rec["started"])
    failed = rec.get("status") == "failed"
    title = "fxlla: %s %s%s" % (rec.get("kind", "render"),
                                "failed" if failed else "ready", took)
    body = (rec.get("summary") or rec["id"]) if failed else (
        os.path.basename(rec.get("output") or "") or rec["id"])
    _post(title, body[:120], rec.get("output") if not failed else None)


def _post(title, body, output):
    """Show it, and make clicking it land somewhere sensible.

    `osascript` notifications belong to Script Editor, so clicking one opened
    Script Editor - an app the user never invoked and which has nothing to do
    with the render. terminal-notifier can attach the finished file to the
    click; without it, attributing the notification to Finder at least means a
    click opens a file browser rather than a script IDE."""
    try:
        if shutil.which("terminal-notifier"):
            cmd = ["terminal-notifier", "-title", title, "-message", body,
                   "-sender", "com.apple.finder", "-group", "fxlla-media"]
            if output and os.path.exists(output):
                # Reveal the file itself: the one thing a click could usefully do.
                cmd += ["-execute", "open -R %s" % shlex.quote(output)]
            subprocess.run(cmd, capture_output=True, timeout=10, check=False)
            return
        subprocess.run(
            ["osascript", "-e",
             "tell application \"Finder\" to display notification %s with title %s"
             % (_as(body), _as(title))],
            capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        pass


def _fmt_age(rec):
    end = rec.get("finished") or time.time()
    start = rec.get("created") or end
    return "%ds" % int(end - start)


def describe(rec):
    """One line for `fxlla media jobs`."""
    tail = rec.get("output") or ""
    if not tail and rec.get("error"):
        # Errors are captured stderr, usually a traceback: the last line carries
        # the actual message, so show that rather than the top of the trace.
        lines = [ln.strip() for ln in rec["error"].splitlines() if ln.strip()]
        tail = lines[-1] if lines else ""
    if len(tail) > 60:
        tail = tail[:57] + "..."
    return "%s  %-8s %-9s %-6s %s" % (rec["id"], rec["kind"], rec["status"],
                                      _fmt_age(rec), tail or rec.get("summary", ""))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "run":
        sys.exit("usage: jobs.py run <id>")
    run(sys.argv[2])
