#!/usr/bin/env python3
"""Measure chat models as fxlla serves them.

Quality is scored by mechanical checks only - executing generated code against
asserts, comparing structured tool calls, validating output shape - never by
another model judging prose. Speed comes from streamed probes and from the
quality requests themselves. The unit under measurement is the deployment:
weights + quantization + engine + server, because that is what the user runs.

Two runs are comparable exactly when the printed task fingerprint AND harness
version match; the fingerprint covers the RENDERED task set (prompts, generated
filler, tool schemas, check specs), so editing anything a score depends on ends
comparability visibly. Results are written only under the state dir, which this
harness never reads back: recorded results can never re-enter the measurement.

Invoked by `fxlla eval` (bin/fxlla cmd_eval), which owns catalog, cache, RAM
and coexistence policy and pipes one "alias|dest|engine|size_mb|role" line per
model on stdin. Standard library only.
"""
import argparse
import ast
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "gateway"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # gateway/metrics.py: StreamMetrics, append_sample
import sandbox

# Bump when extraction, checking, probing, or sandbox semantics change: the
# rendered-set fingerprint cannot see checker logic, so this is the other half
# of the comparability rule.
# v2: extract_code takes the last fence that COMPILES, not the last fence -
#     measured on qwen3-coder, whose replies end in fenced example output.
# v3: extract_code concatenates the compiling fences that BIND names, in
#     order. Both single-fence rules were measured wrong (qwen3-coder closes
#     with fenced example OUTPUT; redteam-4bit closes with a fenced USAGE
#     example that compiles, so last-compiling extracted the call without the
#     definition), and plain concatenation broke on a third measured style:
#     explanatory fragments that compile but are not self-contained, quoted
#     while discussing the bug.
# v4: the streamed probe reads a LINE at a time instead of an 8 KB block.
#     read(8192) blocks until it has that many bytes or the response ends, so
#     every answer shorter than 8 KB - which is most of them - was timed as if
#     it arrived in one piece: TTFT became the whole-response time and the
#     decode window collapsed to microseconds, which is where a reported
#     1,865,386 tok/s came from. Measured on the same small model before and
#     after: ttft 110 ms -> 52 ms, and the rate spread across probes went from
#     667..892 to 609..611. Speed numbers from v3 and earlier are not
#     comparable with these.
# v5: the decode clock started at the first VISIBLE token while the token count
#     came from the server's completion_tokens, which counts the thinking too.
#     Reasoning models therefore had every generated token divided by only the
#     time the answer took. No two servers name that field the same way -
#     llama.cpp streams `reasoning_content`, mlx_lm streams `reasoning` - and
#     metrics.py read neither, so this hit both engines and by different
#     amounts, which means it did not even preserve the ranking. Measured
#     against the servers' own clocks on three models: reported 403.1 / 267.9 /
#     170.2 tok/s where the truth was 98.3 / 113.8 / 53.9 - and the top two
#     swap. Speed numbers from v4 and earlier are not comparable with these.
EVAL_HARNESS_VERSION = 5

EVAL_PORT = int(os.environ.get("FXLLA_EVAL_PORT", "8097"))
FXLLA_BIN = os.environ.get("FXLLA_BIN", os.path.join(REPO_ROOT, "bin", "fxlla"))
EVAL_DIR = os.environ.get("FXLLA_EVAL_DIR", "")
TASKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.json")

READY_POLL_S = 0.1   # the 1s-poll regression must not recur; a test times this
DIM_ORDER = ("instructions", "tools", "candor", "code", "context")  # cheap first

# No task is given less than this, whatever it declares. A reasoning model
# spends tokens thinking BEFORE it answers, and a budget sized for the answer
# alone is spent before the answer starts: the model is scored on an empty
# reply and the number reads as incapacity. Measured here on a Qwen3.5 pair -
# every dimension budgeted at 256 or 512 collapsed, `code` at 2048 did not, and
# the two models' scores differed mostly by how much each one happened to think.
# A budget that small is not a hard test, it is an unrealistic one: nothing
# calling these models in practice caps them there.
#
# Raising it changes what is measured, so it changes the fingerprint too - the
# floor is applied at render, which is the form that gets hashed. Scores from
# before it are not comparable, and the run header says so on its own.
ANSWER_FLOOR = int(os.environ.get("FXLLA_EVAL_ANSWER_FLOOR", "2048"))
PROBES = [
    "Alpha check: describe what a hash table is in about four sentences.",
    "Brief answer: explain the difference between a thread and a process.",
    "Consider a queue and a stack: when would you pick each? Short answer.",
    "Describe what happens when a DNS name is resolved, briefly.",
]
PROBE_MAX_TOKENS = 256
MIN_TOKENS_FOR_TPS = 64


# --------------------------------------------------------------------------
# Task rendering and fingerprint

_WORDS = (
    "system record window process table branch signal beacon output filter "
    "module vector garden harbor timber meadow copper anchor lantern orchard "
    "quarry saddle marble willow canyon hollow ledger tunnel barrel spindle "
    "kettle summit valley thicket prairie cellar attic gable rafter mortar "
    "gravel pebble stream inlet delta ridge basin plateau mesa dune tundra "
    "fjord atoll lagoon reef strait channel current tide swell breaker"
).split()


def _lcg(seed):
    """Deterministic filler source. Hand-rolled so the fingerprint owns it:
    no dependence on the random module's implementation details."""
    state = seed & 0x7FFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state


def gen_filler(seed, n_words, plants):
    """n_words of pseudo-prose with plant sentences inserted at depth
    fractions. Plants carry serial-shaped tokens (digits and capitals) that
    the lowercase filler can never produce by accident."""
    rng = _lcg(seed)
    words = []
    for i in range(n_words):
        words.append(_WORDS[next(rng) % len(_WORDS)])
        if i % 13 == 12:
            words[-1] += "."
    positions = sorted(((p["depth"], p["text"]) for p in plants), reverse=True)
    for depth, text in positions:
        at = min(int(n_words * depth), len(words))
        words.insert(at, text)
    return " ".join(words)


def load_tasks(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def render_tasks(spec, dims=None, quick=False):
    """The executable task list: exact prompts (filler generated), messages,
    tools, max_tokens, check specs. This rendered form - not the file bytes -
    is what gets fingerprinted, so the filler generator is inside the hash."""
    rendered = []
    seen_dims = set()
    for task in spec["tasks"]:
        if dims and task["dim"] not in dims:
            continue
        if quick and task["dim"] in seen_dims:
            continue
        seen_dims.add(task["dim"])
        out = {
            "id": task["id"],
            "dim": task["dim"],
            "max_tokens": max(task["max_tokens"], ANSWER_FLOOR),
            "check": task["check"],
            "stream": task["dim"] in ("code", "context"),
        }
        if "context_gen" in task:
            gen = task["context_gen"]
            out["prompt"] = (gen_filler(gen["seed"], gen["words"], gen["plants"])
                             + task["prompt_suffix"])
        elif "messages" in task:
            out["messages"] = task["messages"]
        else:
            out["prompt"] = task["prompt"]
        if "tools" in task:
            out["tools"] = task["tools"]
        rendered.append(out)
    return rendered


def fingerprint(rendered):
    """12-hex prefix over the canonical rendered set. String equality of this
    plus EVAL_HARNESS_VERSION is the entire comparability rule."""
    blob = json.dumps(rendered, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------
# Extraction

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
_TEXT_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def strip_think(content):
    return _THINK_RE.sub("", content or "").strip()


def _binds_names(tree):
    """True when any top-level statement durably binds a name: an import, a
    def, a class, or an assignment. Distinguishes code that IS part of a
    program from quoted fragments and bare demo calls."""
    return any(isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef,
                                 ast.Assign, ast.AnnAssign, ast.AugAssign))
               for node in tree.body)


def extract_code(content):
    """The compiling fences that BIND names, concatenated in reply order.

    Three reply styles measured on real models shaped this rule, each of
    which broke a simpler one. qwen3-coder closes with fenced EXAMPLE OUTPUT
    (not Python): plain last-fence extracted the demo as code. redteam-4bit
    closes with a fenced USAGE example that compiles: last-compiling-fence
    extracted `print(topo_sort(...))` without the definition and scored six
    correct solutions as NameError. The same model also QUOTES compiling but
    non-self-contained fragments while explaining a bug (`if a[mid] < x:` on
    its own): plain concatenation executed the fragment and crashed the
    import. Keeping the fences that bind names takes the imports, defs,
    classes and assignments in order - a later iteration of a function
    shadows an earlier one, Python's own semantics - and drops output text,
    bare demo calls, and quoted fragments. Fallbacks preserve failure
    visibility: no binding fence means all compiling fences, nothing
    compiling means the last fence so the syntax error surfaces, no fence at
    all means the whole content."""
    text = strip_think(content)
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        return text
    compiling, binding = [], []
    for block in blocks:
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        compiling.append(block.strip())
        if _binds_names(tree):
            binding.append(block.strip())
    if binding:
        return "\n\n".join(binding)
    if compiling:
        return "\n\n".join(compiling)
    return blocks[-1].strip()


def _parse_call_obj(obj):
    name = obj.get("name")
    args = obj.get("arguments", obj.get("parameters"))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not name or not isinstance(args, dict):
        return None
    return {"name": name, "args": args}


def extract_tool_call(message):
    """(call, channel): channel "structured" for message.tool_calls - the only
    thing opencode and Claude Code consume - or "text" when a parseable call
    sits in the content channel. Text-channel calls fail the headline score but
    fill a separate recoverable column: they measure a serving gap, and the
    remedy (swap engine) differs from a model that cannot call at all."""
    calls = message.get("tool_calls") or []
    if calls:
        fn = calls[0].get("function", {})
        parsed = _parse_call_obj(
            {"name": fn.get("name"), "arguments": fn.get("arguments")})
        if parsed:
            return parsed, "structured"
    text = strip_think(message.get("content") or "")
    m = _TEXT_CALL_RE.search(text)
    candidate = m.group(1) if m else None
    if candidate is None:
        m2 = re.search(r"\{.*\}", text, re.S)
        candidate = m2.group(0) if m2 else None
    if candidate:
        try:
            parsed = _parse_call_obj(json.loads(candidate))
            if parsed:
                return parsed, "text"
        except ValueError:
            pass
    return None, None


def extract_json(content):
    """Forgiving JSON extraction for shape checks: fences and prose around the
    object are tolerated here because fence discipline has its own task. The
    OUTERMOST structure wins: trying {..} first would return the inner object
    of a one-object array like [{...}]."""
    text = strip_think(content)
    pairs = [("{", "}"), ("[", "]")]
    brace = text.find("{")
    bracket = text.find("[")
    if bracket != -1 and (brace == -1 or bracket < brace):
        pairs.reverse()
    for open_ch, close_ch in pairs:
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                continue
    return None


# --------------------------------------------------------------------------
# Checkers. Each returns {"pass": bool, "detail": str, ...extras}.

def _shape_ok(spec, data):
    """Typed structural comparison. Spec grammar: {"eq": value} exact;
    {"type": "object", "keys": {...}, "extra_ok": bool}; {"type": "array",
    "len": n, "items": spec}; {"type": "string"|"number"|"integer"|"boolean"}."""
    if "eq" in spec:
        return (True, "") if data == spec["eq"] else (False, "expected %r, got %r" % (spec["eq"], data))
    kind = spec.get("type")
    if kind == "object":
        if not isinstance(data, dict):
            return False, "expected an object, got %s" % type(data).__name__
        for key, sub in spec.get("keys", {}).items():
            if key not in data:
                return False, "missing key %r" % key
            ok, why = _shape_ok(sub, data[key])
            if not ok:
                return False, "%s: %s" % (key, why)
        if not spec.get("extra_ok", False):
            extra = set(data) - set(spec.get("keys", {}))
            if extra:
                return False, "unexpected keys %s" % sorted(extra)
        return True, ""
    if kind == "array":
        if not isinstance(data, list):
            return False, "expected an array, got %s" % type(data).__name__
        if "len" in spec and len(data) != spec["len"]:
            return False, "expected %d items, got %d" % (spec["len"], len(data))
        if "items" in spec:
            for i, item in enumerate(data):
                ok, why = _shape_ok(spec["items"], item)
                if not ok:
                    return False, "[%d]: %s" % (i, why)
        return True, ""
    checks = {"string": str, "boolean": bool, "integer": int, "number": (int, float)}
    if kind in checks:
        # bool subclasses int, so without this a JSON true satisfies both
        # numeric kinds.
        if kind in ("integer", "number") and isinstance(data, bool):
            return False, "expected %s, got a boolean" % kind
        if isinstance(data, checks[kind]):
            return True, ""
        return False, "expected %s, got %s" % (kind, type(data).__name__)
    return False, "unknown shape spec %r" % spec


def _args_match(expected, got):
    """Exact keys; each expected value is either a literal or {"any_of": [...]}
    for the few prompts where two spellings are both correct. Undeclared extra
    arguments fail: an agent stack passes them straight to the tool."""
    if set(got) != set(expected):
        return False, "args keys %s, expected %s" % (sorted(got), sorted(expected))
    for key, want in expected.items():
        have = got[key]
        if isinstance(want, dict) and "any_of" in want:
            if have not in want["any_of"]:
                return False, "%s=%r not in %r" % (key, have, want["any_of"])
        elif have != want:
            return False, "%s=%r, expected %r" % (key, have, want)
    return True, ""


def _contains_all(text, groups):
    """groups: list of entries; a string entry must appear, a list entry means
    any one of its alternatives must appear. Case-insensitive."""
    low = text.lower()
    for entry in groups:
        alts = entry if isinstance(entry, list) else [entry]
        if not any(a.lower() in low for a in alts):
            return False, "missing %r" % (entry,)
    return True, ""


def check_task(task, message, run_code=sandbox.run_code):
    check = task["check"]
    kind = check["type"]
    content = message.get("content") or ""

    if kind == "run_tests":
        code = extract_code(content)
        if not code.strip():
            return {"pass": False, "detail": "no code in the reply"}
        result = run_code(code, check["tests"])
        if result["verdict"] == "error":
            return {"pass": False, "detail": "harness error: " + result["output"][:200],
                    "harness_error": True, "check_s": result["seconds"]}
        # --keep-failed means failed: a kept directory for a passing solution
        # would be orphaned - never named, never reclaimed.
        if result.get("taskdir") and result["verdict"] == "pass":
            shutil.rmtree(result["taskdir"], ignore_errors=True)
            result["taskdir"] = None
        tail = result["output"].strip().splitlines()[-1][:120] if result["output"].strip() else ""
        detail = "" if result["verdict"] == "pass" else "%s: %s" % (result["verdict"], tail)
        if result.get("taskdir"):
            detail += " [kept: %s]" % result["taskdir"]
        # Sandbox execution is harness time, not model time: recorded per task
        # so wall_s can be read honestly.
        return {"pass": result["verdict"] == "pass", "detail": detail,
                "check_s": result["seconds"]}

    if kind == "tool_call":
        call, channel = extract_tool_call(message)
        if call is None:
            return {"pass": False, "detail": "no tool call found", "channel": None}
        if call["name"] != check["name"]:
            return {"pass": False, "channel": channel,
                    "detail": "called %s, expected %s" % (call["name"], check["name"])}
        ok, why = _args_match(check["args"], call["args"])
        if not ok:
            return {"pass": False, "channel": channel, "detail": why}
        if channel != "structured":
            return {"pass": False, "channel": channel, "recoverable": True,
                    "detail": "correct call, but in the text channel: an agent stack never sees it"}
        return {"pass": True, "channel": channel, "detail": ""}

    if kind == "no_tool_call":
        call, channel = extract_tool_call(message)
        if call is not None:
            return {"pass": False, "channel": channel,
                    "detail": "called %s when no call was wanted" % call["name"]}
        text = strip_think(content)
        if not text:
            return {"pass": False, "detail": "empty reply"}
        ok, why = _contains_all(text, check.get("must_contain", []))
        return {"pass": ok, "detail": why}

    if kind == "json_shape":
        data = extract_json(content)
        if data is None:
            return {"pass": False, "detail": "no parseable JSON in the reply"}
        ok, why = _shape_ok(check["shape"], data)
        return {"pass": ok, "detail": why}

    if kind == "exact":
        got = strip_think(content)
        want = check["value"]
        return {"pass": got == want,
                "detail": "" if got == want else "got %r" % got[:80]}

    if kind == "contains":
        ok, why = _contains_all(strip_think(content), check["values"])
        return {"pass": ok, "detail": why}

    if kind == "absent":
        low = strip_think(content).lower()
        if not low:
            return {"pass": False, "detail": "empty reply"}
        hits = [w for w in check["words"] if w.lower() in low]
        return {"pass": not hits,
                "detail": "used forbidden %s" % hits if hits else ""}

    if kind == "line_rules":
        lines = [l for l in strip_think(content).splitlines() if l.strip()]
        if len(lines) != check["count"]:
            return {"pass": False, "detail": "%d lines, expected %d" % (len(lines), check["count"])}
        rx = re.compile(check["each_matches"])
        bad = [l for l in lines if not rx.fullmatch(l.strip())]
        return {"pass": not bad,
                "detail": "line %r does not match" % bad[0][:40] if bad else ""}

    if kind == "word_count":
        words = strip_think(content).split()
        if not words:
            return {"pass": False, "detail": "empty reply"}
        return {"pass": len(words) <= check["max"],
                "detail": "" if len(words) <= check["max"] else "%d words, cap %d" % (len(words), check["max"])}

    return {"pass": False, "detail": "unknown check type %r" % kind, "harness_error": True}


# --------------------------------------------------------------------------
# HTTP

def _post(port, body, timeout):
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def _body(model_id, task_messages, max_tokens, tools=None, stream=False):
    body = {"model": model_id, "messages": task_messages,
            "temperature": 0, "top_p": 1, "seed": 0,
            "max_tokens": max_tokens, "stream": stream}
    if tools:
        body["tools"] = tools
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def request_streamed(port, model_id, task_messages, max_tokens, tools=None, timeout=600):
    """Returns (message, sm, usage, finish_reason, wall_s). sm is a fed
    StreamMetrics; usage tokens are exact when the server emits the trailing
    usage chunk, else the delta count stands in (marked ~ downstream)."""
    body = _body(model_id, task_messages, max_tokens, tools, stream=True)
    start = time.monotonic()
    sm = metrics.StreamMetrics(start)
    content, finish, usage = [], None, None
    tool_calls = {}
    try:
        resp = _post(port, body, timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            # Some servers reject stream_options; losing exact usage is
            # recorded (the ~ marker), not fatal.
            body.pop("stream_options", None)
            resp = _post(port, body, timeout)
        else:
            raise
    with resp:
        buf = b""
        while True:
            # A LINE at a time, not a fixed block. `read(8192)` blocks until it
            # has that many bytes or the response ends, so anything shorter
            # than 8 KB - which is most probe answers - arrived in one piece as
            # far as the clock could tell: the first-token stamp landed next to
            # the last and the decode window collapsed to microseconds. That
            # made TTFT the whole-response time and the rate a division by
            # almost zero. readline returns as soon as one is available, which
            # is the granularity SSE actually has.
            chunk = resp.readline()
            if not chunk:
                break
            sm.feed(chunk)
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if not data or data == b"[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = tool_calls.setdefault(
                            idx, {"function": {"name": "", "arguments": ""}})
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
    wall = time.monotonic() - start
    message = {"content": "".join(content)}
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return message, sm, usage, finish, wall


def request_plain(port, model_id, task_messages, max_tokens, tools=None, timeout=600):
    body = _body(model_id, task_messages, max_tokens, tools, stream=False)
    start = time.monotonic()
    with _post(port, body, timeout) as resp:
        obj = json.loads(resp.read())
    wall = time.monotonic() - start
    choice = (obj.get("choices") or [{}])[0]
    return (choice.get("message") or {}, obj.get("usage"),
            choice.get("finish_reason"), wall)


# --------------------------------------------------------------------------
# Server lifecycle

def port_in_use(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _ready_url(engine, port):
    # llama-server answers HTTP - including /v1/models - while weights are
    # still loading; its /health flips only when it can serve. cmd_on polls
    # /health for the same reason. mlx_lm.server has no /health.
    path = "/health" if engine == "gguf" else "/v1/models"
    return "http://127.0.0.1:%d%s" % (port, path)


def wait_ready(proc, engine, port, timeout_s):
    """Poll until the server can serve, at READY_POLL_S. Returns seconds to
    ready, or raises RuntimeError naming the cause. Death is checked between
    polls so a bad model fails in milliseconds, not after the full timeout."""
    url = _ready_url(engine, port)
    begin = time.monotonic()
    deadline = begin + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("the server exited on start (code %s)" % proc.returncode)
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return time.monotonic() - begin
        except Exception:
            time.sleep(READY_POLL_S)
    raise RuntimeError("not ready within %ds" % timeout_s)


def spawn_backend(alias, port, log_path):
    log = open(log_path, "ab")
    try:
        return subprocess.Popen(
            [FXLLA_BIN, "_backend", alias, str(port)],
            stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            start_new_session=True)
    finally:
        log.close()  # the child holds its own copy of the fd


def teardown(proc, port):
    """SIGTERM the group, escalate, then wait for the port to actually close:
    Metal can hold a dying process long enough for the next spawn to lose the
    bind. A port that never frees is an error naming the pid, not a hang."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            # A process wedged in the Metal driver can sit uninterruptible even
            # after SIGKILL; an unbounded wait here would hang the sweep, which
            # is exactly what the port-drain loop below exists to report.
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not port_in_use(port):
            return
        time.sleep(0.2)
    raise RuntimeError("port %d still held after teardown (pid was %d)" % (port, proc.pid))


class RssSampler:
    """Peak RSS of the server, sampled every 2s. Process RSS on unified memory
    is an approximation and is labeled as one."""
    def __init__(self, pid):
        self.pid = pid
        self.peak_mb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["ps", "-o", "rss=", "-p", str(self.pid)],
                                     capture_output=True, text=True, timeout=5)
                kb = int(out.stdout.strip() or 0)
                self.peak_mb = max(self.peak_mb, kb // 1024)
            except Exception:
                pass
            self._stop.wait(2)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_a):
        self._stop.set()
        self._thread.join(timeout=3)


# --------------------------------------------------------------------------
# Provenance

def server_provenance(engine, port):
    binary = {"gguf": "llama-server", "omlx": "omlx", "mtplx": "mtplx"}.get(engine, "mlx_lm.server")
    path = shutil.which(binary)
    version = None
    if engine == "gguf":
        try:
            props = json.load(urllib.request.urlopen(
                "http://127.0.0.1:%d/props" % port, timeout=5))
            version = props.get("build_info")
        except Exception:
            pass
    return {"binary": path, "version": version}


def weights_provenance(dest):
    out = {"source": None, "entry": None}
    for key in ("source", "entry"):
        try:
            with open(os.path.join(dest, "." + key), encoding="utf-8") as fh:
                out[key] = fh.read().strip()
        except OSError:
            pass
    return out


def _sysctl(name):
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def machine_provenance():
    power = None
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                             text=True, timeout=5)
        power = "ac" if "AC Power" in out.stdout else "battery"
    except Exception:
        pass
    mem = _sysctl("hw.memsize")
    return {
        "model": _sysctl("hw.model"),
        "chip": _sysctl("machdep.cpu.brand_string"),
        "mem_gb": int(mem) // (1 << 30) if mem and mem.isdigit() else None,
        "os": platform.mac_ver()[0] or platform.platform(),
        "power": power,   # a hot or battery-fed laptop throttles later models
    }


# --------------------------------------------------------------------------
# Per-model run

def model_request_id(engine, alias, dest):
    # gguf: llama-server is started with --alias, so the alias is the id.
    # omlx: serves the model under its directory basename (== alias) and
    #   validates the field, so send the alias - a path would 404.
    # mtplx: single-model server, accepts any field; send the alias for tidiness.
    # mlx: mlx_lm.server takes the model DIRECTORY path. /v1/models is never
    # consulted: mlx_lm enumerates the whole HF cache there.
    return alias if engine in ("gguf", "omlx", "mtplx") else dest


def ready_timeout_s(size_mb):
    # 200 MB/s is a conservative external-disk read rate; the 132 GB model
    # needs ~11 minutes and a flat cap would kill it.
    return max(180, size_mb // 200)


def eval_model(plan, rendered, args, run_dir):
    """One model, cold spawn to teardown. Returns the result record."""
    alias, dest, engine, size_mb, role = plan
    port = EVAL_PORT
    if port_in_use(port):
        raise RuntimeError(
            "something is already listening on the eval port %d; refusing to "
            "adopt it (it could be any model). Free the port or set "
            "FXLLA_EVAL_PORT." % port)
    record = {
        "alias": alias, "engine": engine, "role": role, "size_mb": size_mb,
        "harness_version": EVAL_HARNESS_VERSION,
        "partial": False, "failed_to_load": False,
    }
    log_path = os.path.join(run_dir, "%s.server.log" % alias)
    proc = spawn_backend(alias, port, log_path)
    try:
        try:
            record["load_s"] = round(
                wait_ready(proc, engine, port, ready_timeout_s(size_mb)), 2)
        except RuntimeError as exc:
            record["failed_to_load"] = True
            record["error"] = str(exc)
            _print_log_tail(log_path)
            return record
        model_id = model_request_id(engine, alias, dest)
        record["server"] = server_provenance(engine, port)
        record["weights"] = weights_provenance(dest)

        tps_samples, ttft_samples = [], []
        usage_exact = True
        wall_begin = time.monotonic()
        total_completion_tokens = 0

        with RssSampler(proc.pid) as rss:
            # Probe 0 is the first-request penalty (for mlx, the entire weight
            # load; for gguf, warmup) and is reported separately, never in the
            # medians. Its socket timeout scales with size like the readiness
            # wait does: a flat cap that the load math itself says is too small
            # would kill the 132 GB model here.
            try:
                for i, prompt in enumerate(PROBES if not args.quick else PROBES[:2]):
                    message, sm, usage, _fin, wall = request_streamed(
                        port, model_id, [{"role": "user", "content": prompt}],
                        PROBE_MAX_TOKENS,
                        timeout=(ready_timeout_s(size_mb) + 120) if i == 0 else 600)
                    ttft_ms, tps, tokens = sm.result(sm.start + wall)
                    if usage is None:
                        usage_exact = False
                    total_completion_tokens += tokens or 0
                    if i == 0:
                        record["first_request_s"] = round(wall, 2)
                        record["first_request_ttft_ms"] = ttft_ms
                    else:
                        if ttft_ms is not None:
                            ttft_samples.append(ttft_ms)
                        if tps is not None and (tokens or 0) >= MIN_TOKENS_FOR_TPS:
                            tps_samples.append(tps)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                # A server that dies or stalls during the probes has no task
                # results to salvage; record the cause and keep the sweep going.
                record["failed_to_load"] = True
                record["error"] = "loaded, but died during the probes: %s" % exc
                _print_log_tail(log_path)
                return record

            state = {"usage_exact": usage_exact, "tokens": total_completion_tokens}

            def run_tasks_once():
                results = []
                for task in rendered:
                    task_messages = task.get("messages") or [
                        {"role": "user", "content": task["prompt"]}]
                    entry = {"id": task["id"], "dim": task["dim"]}
                    begin = time.monotonic()
                    try:
                        if task["stream"]:
                            message, sm, usage, finish, wall = request_streamed(
                                port, model_id, task_messages,
                                task["max_tokens"], task.get("tools"))
                            ttft_ms, tps, tokens = sm.result(sm.start + wall)
                            if task["dim"] == "context":
                                entry["ttft_ms"] = ttft_ms
                                if usage and usage.get("prompt_tokens"):
                                    entry["prompt_tokens"] = usage["prompt_tokens"]
                                    if ttft_ms:
                                        entry["prefill_tps"] = round(
                                            usage["prompt_tokens"] / (ttft_ms / 1000.0))
                            if tps is not None and (tokens or 0) >= MIN_TOKENS_FOR_TPS:
                                tps_samples.append(tps)
                            if usage is None:
                                state["usage_exact"] = False
                            state["tokens"] += tokens or 0
                        else:
                            message, usage, finish, wall = request_plain(
                                port, model_id, task_messages,
                                task["max_tokens"], task.get("tools"))
                            if usage and usage.get("completion_tokens"):
                                state["tokens"] += usage["completion_tokens"]
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        entry.update({"pass": False, "detail": "request failed: %s" % exc,
                                      "harness_error": True,
                                      "seconds": round(time.monotonic() - begin, 1)})
                        results.append(entry)
                        continue
                    verdict = check_task(task, message,
                                         run_code=lambda code, tests: sandbox.run_code(
                                             code, tests, keep_dir=args.keep_failed))
                    entry.update(verdict)
                    if not entry.get("pass"):
                        # Enough of the raw reply to diagnose a failure from the
                        # run record alone, without re-running the model.
                        entry["reply_head"] = strip_think(
                            message.get("content") or "")[:300]
                    if finish == "length":
                        entry["truncated"] = True
                    entry["seconds"] = round(time.monotonic() - begin, 1)
                    results.append(entry)
                return results

            results = []
            interrupted = False
            try:
                # Pass 1 is the headline, always: two people running the same
                # command must report the same shape. Extra passes only measure
                # this model's own run-to-run noise, so every headline number -
                # speed medians, tokens, wall - is frozen at the pass-1
                # boundary below before any repeat runs.
                results = run_tasks_once()
                pass1_tps = len(tps_samples)
                pass1_tokens = state["tokens"]
                pass1_wall = time.monotonic() - wall_begin
                if args.repeats > 1:
                    flips = set()
                    baseline = {r["id"]: r for r in results}
                    for _ in range(args.repeats - 1):
                        for r in run_tasks_once():
                            base = baseline.get(r["id"], {})
                            # A transient request or sandbox error is harness
                            # noise, not model noise: it must not land in the
                            # list the README calls the model's noise floor.
                            if r.get("harness_error") or base.get("harness_error"):
                                continue
                            if r.get("pass") != base.get("pass"):
                                flips.add(r["id"])
                    record["repeats"] = args.repeats
                    # The flipped ids ARE this model's measured noise floor.
                    record["unstable_tasks"] = sorted(flips)
                del tps_samples[pass1_tps:]
                state["tokens"] = pass1_tokens
            except KeyboardInterrupt:
                interrupted = True
                pass1_wall = time.monotonic() - wall_begin
            record["partial"] = interrupted
            record["rss_peak_mb"] = rss.peak_mb

        record["results"] = results
        record["tokens_spent"] = state["tokens"]
        record["usage_exact"] = state["usage_exact"]
        record["wall_s"] = round(pass1_wall, 1)
        record["ttft_ms"] = _summ(ttft_samples, 0)
        record["tok_s"] = _summ(tps_samples, 1)
        ctx_entries = [r for r in results if r["dim"] == "context" and r.get("ttft_ms")]
        if ctx_entries:
            record["ttft_context_ms"] = {
                r["id"]: r["ttft_ms"] for r in ctx_entries}
            prefills = [r["prefill_tps"] for r in ctx_entries if r.get("prefill_tps")]
            if prefills:
                record["prefill_tok_s"] = int(statistics.median(prefills))
        return record
    finally:
        # A teardown failure must never destroy the results of a model that
        # just spent minutes of GPU time: the record is already built, so the
        # stuck port is reported and the NEXT model's pre-spawn check is what
        # refuses to proceed against it.
        try:
            teardown(proc, port)
        except RuntimeError as exc:
            record["teardown_error"] = str(exc)
            sys.stderr.write("warning: %s\n" % exc)


def _summ(samples, digits):
    if not samples:
        return None
    return {"median": round(statistics.median(samples), digits),
            "min": round(min(samples), digits),
            "max": round(max(samples), digits),
            "n": len(samples)}


def _print_log_tail(log_path, n=25):
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            tail = fh.readlines()[-n:]
        sys.stderr.write("".join(tail))
    except OSError:
        pass


# --------------------------------------------------------------------------
# Report

def dim_counts(results):
    """(measured, passed, harness_errors) per dimension. A request that died
    in transit or a sandbox that could not run says nothing about the model,
    so harness errors leave the denominator instead of counting as failures -
    and are reported, never silently dropped."""
    out = {}
    for r in results:
        measured, passed, herr = out.get(r["dim"], (0, 0, 0))
        if r.get("harness_error"):
            herr += 1
        else:
            measured += 1
            passed += 1 if r.get("pass") else 0
        out[r["dim"]] = (measured, passed, herr)
    return out


def render_report(records, fp, quick, tasks_per_dim):
    lines = []
    add = lines.append
    add("fxlla eval  |  tasks %s  harness v%d%s" % (
        fp, EVAL_HARNESS_VERSION, "  [QUICK: pipeline check, not a ranking]" if quick else ""))
    add("")
    header = "%-18s %8s %9s %10s %14s %9s" % (
        "model", "load_s", "first_s", "ttft_ms", "tok/s", "rss_gb")
    header += "  " + "  ".join("%-12s" % d for d in DIM_ORDER)
    header += "  %8s %7s" % ("tokens", "wall_s")
    add(header)
    for rec in records:
        if rec.get("failed_to_load"):
            add("%-18s FAILED TO LOAD: %s" % (rec["alias"], rec.get("error", "")))
            continue
        counts = dim_counts(rec.get("results", []))
        # The marker slot always occupies one column so rows stay aligned with
        # the header whether or not it fires.
        marker = " " if rec.get("usage_exact") else "~"
        row = "%-18s %8s %9s %10s %13s%s %9s" % (
            rec["alias"], rec.get("load_s", "-"),
            rec.get("first_request_s", "-"),
            (rec.get("ttft_ms") or {}).get("median", "-"),
            (rec.get("tok_s") or {}).get("median", "-"), marker,
            round(rec.get("rss_peak_mb", 0) / 1024.0, 1) or "-")
        herr_total = 0
        for dim in DIM_ORDER:
            measured, passed, herr = counts.get(dim, (0, 0, 0))
            herr_total += herr
            cell = "%d/%d" % (passed, measured) if measured else "-"
            if herr:
                cell += "!"
            row += "  %-12s" % cell
        row += "  %8d %7s" % (rec.get("tokens_spent", 0), rec.get("wall_s", "-"))
        add(row)
        if herr_total:
            add("%-18s   ! %d task(s) hit harness errors and are excluded from "
                "the denominators above" % ("", herr_total))
        if rec.get("partial"):
            add("%-18s   PARTIAL RUN: interrupted, counts above are incomplete" % "")
    add("")
    for rec in records:
        fails = [r for r in rec.get("results", []) if not r.get("pass")]
        if not fails:
            continue
        add("%s failures:" % rec["alias"])
        for r in fails:
            extra = ""
            if r.get("recoverable"):
                extra = " [recoverable from text channel]"
            if r.get("truncated"):
                extra += " [hit max_tokens]"
            if r.get("harness_error"):
                extra += " [HARNESS ERROR - not a model failure]"
            add("  %-22s %s%s" % (r["id"], r.get("detail", ""), extra))
        tool_results = [r for r in rec.get("results", []) if r["dim"] == "tools"]
        if tool_results and all(r.get("channel") != "structured" for r in tool_results):
            add("  note: no response carried a structured tool_calls field; this is "
                "a serving-layer gap (engine/template), not model inability.")
        add("")
    add("Reading these numbers:")
    for dim in DIM_ORDER:
        n = tasks_per_dim.get(dim, 0)
        if n:
            add("  - %s: %d tasks, so one task is %.0f points; a 1-2 task gap is noise."
                % (dim, n, 100.0 / n))
    add("  - the cold cost of a model is load_s PLUS first_s: llama-server loads")
    add("    weights before answering /health, so its load_s is the real load and")
    add("    first_s adds warmup; mlx_lm.server starts instantly and loads on the")
    add("    FIRST REQUEST, so its load_s is only process startup and the weight")
    add("    load is inside first_s. Compare cold costs as the sum, never load_s")
    add("    alone across engines.")
    add("  - tok/s gaps inside two models' min..max spreads are noise; ttft below")
    add("    ~50 ms of difference is noise; load_s and first_s are single samples.")
    add("  - ~ marks token rates from delta counts (server sent no usage): compare")
    add("    ~ numbers only with other ~ numbers.")
    add("  - comparable runs require the same tasks fingerprint AND harness version;")
    add("    scores are per-machine and rank local models against each other only.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main

def read_plan(stream):
    plans = []
    for line in stream:
        line = line.strip()
        if not line:
            continue
        alias, dest, engine, size_mb, role = line.split("|")
        plans.append((alias, dest, engine, int(size_mb), role))
    return plans


def main():
    ap = argparse.ArgumentParser(prog="fxlla-eval")
    ap.add_argument("--dim", default="")
    ap.add_argument("--tasks", default=TASKS_PATH)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--keep-failed", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    dims = [d.strip() for d in args.dim.split(",") if d.strip()] or None
    if dims:
        bad = set(dims) - set(DIM_ORDER)
        if bad:
            sys.exit("unknown dimensions: %s (have: %s)" % (sorted(bad), ", ".join(DIM_ORDER)))
    spec = load_tasks(args.tasks)
    rendered = render_tasks(spec, dims=dims, quick=args.quick)
    rendered.sort(key=lambda t: DIM_ORDER.index(t["dim"]))
    fp = fingerprint(rendered)
    tasks_per_dim = {}
    for t in rendered:
        tasks_per_dim[t["dim"]] = tasks_per_dim.get(t["dim"], 0) + 1

    plans = read_plan(sys.stdin)
    if args.list:
        print("tasks %s  harness v%d  |  %d tasks: %s" % (
            fp, EVAL_HARNESS_VERSION, len(rendered),
            ", ".join("%s %d" % (d, n) for d, n in sorted(tasks_per_dim.items()))))
        for alias, _dest, engine, size_mb, role in plans:
            print("  %-18s %-5s %-8s %d MB" % (alias, engine, role, size_mb))
        return
    if not plans:
        sys.exit("no models to eval (nothing cached with role dev/agentic/max; "
                 "name one explicitly, e.g.: fxlla eval qwen3-coder)")

    run_dir = os.path.join(EVAL_DIR, "run-%d" % int(time.time()))
    os.makedirs(run_dir, exist_ok=True)
    machine = machine_provenance()
    records = []
    try:
        for plan in plans:
            alias = plan[0]
            sys.stderr.write("== %s (%s, %d MB)\n" % (alias, plan[2], plan[3]))
            try:
                rec = eval_model(plan, rendered, args, run_dir)
            except KeyboardInterrupt:
                break
            except RuntimeError as exc:
                rec = {"alias": alias, "failed_to_load": True, "error": str(exc),
                       "harness_version": EVAL_HARNESS_VERSION}
            rec["fingerprint"] = fp
            rec["machine"] = machine
            records.append(rec)
            with open(os.path.join(run_dir, "%s.json" % alias), "w",
                      encoding="utf-8") as fh:
                json.dump(rec, fh, indent=1)
            if rec.get("partial"):
                break  # interrupted mid-model: keep what we have, stop the sweep
            if not rec.get("failed_to_load"):
                # The qualifiers travel with the numbers: a tok/s median is
                # unreadable without usage_exact, and load_s without first_s
                # is the mlx lie the table itself refuses to tell.
                summary = {"ts": int(time.time()), "alias": alias,
                           "engine": rec.get("engine"),
                           "fingerprint": fp, "harness_version": EVAL_HARNESS_VERSION,
                           "machine_model": machine.get("model"),
                           "dims": {d: "%d/%d" % (p, m) for d, (m, p, _h) in
                                    dim_counts(rec.get("results", [])).items()},
                           "tok_s": (rec.get("tok_s") or {}).get("median"),
                           "usage_exact": rec.get("usage_exact"),
                           "load_s": rec.get("load_s"),
                           "first_request_s": rec.get("first_request_s")}
                metrics.append_sample(os.path.join(EVAL_DIR, "history.jsonl"), summary)
    finally:
        if records:
            if args.json:
                print(json.dumps({"fingerprint": fp,
                                  "harness_version": EVAL_HARNESS_VERSION,
                                  "machine": machine, "records": records}, indent=1))
            else:
                print(render_report(records, fp, args.quick, tasks_per_dim))
            sys.stderr.write("full detail: %s\n" % run_dir)


if __name__ == "__main__":
    main()
