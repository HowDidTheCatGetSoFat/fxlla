#!/usr/bin/env python3
"""fxlla multi-model gateway.

One OpenAI-compatible endpoint that fronts many models. It aggregates the
downloaded chat models in /v1/models (embedding models in the same store are
excluded - they cannot chat) and, on each request, routes to the backend for
the requested model, loading it on demand and evicting the least-recently-used
backend when a load would exceed the RAM budget.

Standard library only. Backends are launched through `fxlla _backend <alias>
<port>` so the launch logic stays in the CLI (the single source of truth).

Config via environment:
  FXLLA_HOST, FXLLA_PORT        gateway bind address (default 127.0.0.1:8080)
  FXLLA_STORE                   model store (models under <store>/models)
  FXLLA_BACKEND_PORT_BASE       first internal backend port (default 8100)
  FXLLA_GATEWAY_BUDGET_MB       resident RAM budget (default: ~GPU reservable)
  FXLLA_BIN                     path to the fxlla CLI (default: fxlla on PATH)
  FXLLA_CTX                     CEILING on the context window served to a gguf
                                model (default 32768). Each one gets the window
                                it was trained for, read from its own header,
                                capped by this - so a 7B trained to 128k and a
                                27B trained to 262k stop sharing one number.
                                Raise it to let the big ones use their full
                                window; the cost is RAM, since llama-server
                                allocates the KV cache up front. Reported per
                                model as "context" in /v1/models, from the same
                                reader that starts the backend.
  FXLLA_ROPE_STRETCH            1 to serve the window a model ADVERTISES rather
                                than the one it was trained for (default 0).
                                Only differs where rope scaling is baked in:
                                262144 against 1048576 on a YaRN factor of 4.
                                The stretch is what makes the larger window
                                reachable, so it stays on there - but it costs
                                attention quality, and the KV cache is
                                allocated for the whole window up front. Raise
                                FXLLA_CTX to match or the ceiling wins and you
                                get neither.
  FXLLA_KEEP_WARM               idle minutes before a resident backend is
                                unloaded (default 10, 0 = never). The same
                                variable and units the single-model server
                                uses, which used to be the only one honouring
                                it - here the memory was freed only when the
                                next load would not fit.
  FXLLA_STATS_FILE              passive metrics time-series (default: the CLI's
                                stats.jsonl under the state dir)
  FXLLA_VISION_ROUTING          0 to stop reading images for models that cannot
                                (default on). Off, an image sent to a text model
                                fails in the backend as it used to. An image is
                                forwarded untouched only to a model the catalog
                                gives role 'vision' AND that has a projector on
                                disk; a model carrying an inherited, undeclared
                                vision tower gets a description like any other.
  FXLLA_VISION_MODEL            which model reads them (default: the first
                                catalog alias with role 'vision' that has a
                                multimodal projector on disk). Naming one here
                                is itself a declaration, so it need only have
                                the projector.
  FXLLA_VISION_MAX_IMAGES       images one request may carry (default 4); each
                                is read separately, so a batch holds the
                                connection for as long as all of them take
"""

import collections
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ggufmeta  # noqa: E402  (local module, added to sys.path above)
import metrics  # noqa: E402
import telemetry  # noqa: E402

# Poll interval while waiting for a backend to answer. Small enough that a fast
# model load is not rounded up to the next whole second.
READY_POLL_INTERVAL = 0.05
HOST = os.environ.get("FXLLA_HOST", "127.0.0.1")
PORT = int(os.environ.get("FXLLA_PORT", "8080"))
STORE = os.environ.get("FXLLA_STORE", "")
MODELS_DIR = os.path.join(STORE, "models")
PORT_BASE = int(os.environ.get("FXLLA_BACKEND_PORT_BASE", "8100"))
FXLLA_BIN = os.environ.get("FXLLA_BIN", "fxlla")
# Where each backend's output is kept, beside gateway.log. `fxlla serve` passes
# its state dir; run by hand, the gateway falls back to the temp dir.
STATE_DIR = os.environ.get("FXLLA_STATE_DIR") or tempfile.gettempdir()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def log(msg):
    sys.stderr.write("[gateway] %s\n" % msg)
    sys.stderr.flush()


def backend_log_path(alias):
    return os.path.join(STATE_DIR, "backend-%s.log" % alias)


def last_log_line(path, limit=300):
    """The last non-empty line of a backend log, without color codes, or ''."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            tail = f.read().decode("utf-8", "replace").replace("\0", "")
    except (OSError, TypeError):
        return ""
    lines = [ln.strip() for ln in _ANSI.sub("", tail).splitlines() if ln.strip()]
    return lines[-1][:limit] if lines else ""


def _is_loopback(addr):
    """True for IPv4/IPv6 loopback, including IPv4-mapped IPv6."""
    return addr == "::1" or addr.startswith("127.") or addr.startswith("::ffff:127.")


def dir_size_mb(path):
    try:
        out = subprocess.check_output(["du", "-sk", path], stderr=subprocess.DEVNULL)
        return int(out.split()[0]) // 1024
    except Exception:
        return 0


def default_budget_mb():
    env = os.environ.get("FXLLA_GATEWAY_BUDGET_MB")
    if env:
        return int(env)
    try:
        total_mb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) // (1024 * 1024)
    except Exception:
        total_mb = 65536
    try:
        cur = int(subprocess.check_output(["sysctl", "-n", "iogpu.wired_limit_mb"]))
    except Exception:
        cur = 0
    eff = cur if cur > 0 else total_mb * 3 // 4
    return max(eff - 4096, 4096)  # leave a margin for the gateway itself


BUDGET_MB = default_budget_mb()


CATALOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "config", "models.conf")


def _role_aliases(role):
    """Catalog aliases with this role, or None when the catalog cannot be read.

    None and the empty set are different answers - "nothing declares this
    role" versus "there was no way to find out" - and collapsing them made a
    catalog that existed but could not be opened look like a catalog that
    declared nothing, which denies every model instead of failing open."""
    out = set()
    try:
        with open(CATALOG, encoding="utf-8") as fh:
            for line in fh:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 5 and not parts[0].startswith("#") and parts[3] == role:
                    out.add(parts[0])
    except OSError:
        return None
    return out


# Sending an image to a text model used to be a crash: mlx_lm.server raises
# "Only 'text' content type is supported" on any non-text part, and a GGUF
# backend without a projector drops it silently. Rather than make every client
# discover which of its models can see, the gateway reads the image with one
# that can and hands the chosen model a description. One request in, one answer
# out, two models used - the capability lives behind the endpoint instead of in
# whatever is calling it.
VISION_ROUTING = os.environ.get("FXLLA_VISION_ROUTING", "1") not in ("0", "false", "")


def _has_projector(alias):
    """True when a multimodal projector sits next to this model's weights.

    This answers whether llama-server CAN be handed an image: bin/fxlla passes
    --mmproj when it finds one, and without it the image goes nowhere.

    Unanchored and case-folded, matching what bin/fxlla actually looks for.
    Anchored at the front it agreed with bin/fxlla only for publishers who put
    the word first, and disagreed silently for the rest - this said "cannot
    see" about a model bin/fxlla would have handed the projector, had bin/fxlla
    found it either. Two wrongs that agreed are still two wrongs.

    The case folding is here for the same reason and must stay in step: a glob
    is case-sensitive, bin/fxlla's own lookup is not, and this answering "no"
    about a projector the launcher WILL pass is how a model that can see gets
    told it cannot."""
    try:
        names = os.listdir(os.path.join(MODELS_DIR, alias))
    except OSError:
        return False
    return any(n.lower().endswith(".gguf") and "mmproj" in n.lower() for n in names)


def _can_see(alias):
    """True when this model is trusted to read an image itself.

    Two separate questions had been collapsed into one. The projector on disk
    says an image CAN reach the model; the catalog role says its vision was
    meant to be used. Those came apart the first time a model shipped a vision
    tower it had inherited and never tuned - a Qwen3.5 derivative whose own
    author writes that text-only training did not evaluate image understanding.
    Reading the file's existence as a statement about quality was inferring a
    claim nobody had made, so both now have to agree, and the safe answer wins
    by default: an undeclared model gets a description from one chosen for the
    job rather than being trusted with its own untested eyes. Declaring it is a
    deliberate act, and role 'vision' in the catalog is where it is made."""
    declared = _role_aliases("vision")
    if declared is None:
        # Unreadable, not merely absent: a moved catalog and an unreadable one
        # are the same situation from here. There are no declarations to read
        # AND no reader to be found, so demanding a declaration would turn a
        # working vision model into a 502. Fail open to the projector, which
        # is the only evidence left.
        return _has_projector(alias)
    return alias in declared and _has_projector(alias)


def _vision_alias():
    """The model that will do the reading, or None."""
    preferred = os.environ.get("FXLLA_VISION_MODEL")
    if preferred:
        # Validated, not trusted: an override naming a text model would send
        # it an image and surface as a vision failure blaming the wrong thing.
        # Only the projector is required - naming a model here IS the
        # declaration, and it should not have to be in the catalog to be used.
        if not _has_projector(preferred):
            raise RuntimeError(
                "FXLLA_VISION_MODEL is set to %r, which has no multimodal "
                "projector on disk and cannot read an image" % preferred)
        return preferred
    for alias in sorted(_role_aliases("vision") or ()):
        if _has_projector(alias):
            return alias
    return None


def _image_parts(body):
    """Every (message, index) holding an image, in order."""
    found = []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return found
    for message in messages:
        # Every level is checked: this runs on EVERY request, including ones
        # with no image, so anything raising here turns a request that used to
        # be forwarded into a 502 blamed on vision.
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for i, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "image_url":
                found.append((message, i))
    return found


def _asked(body):
    """The text the user sent alongside the image, for relevance.

    Passed to the reader as CONTEXT, never as a claim to check: asked whether
    an expected string was present, a vision model confirmed it and missed a
    whole block of invented text; asked to enumerate, it reported the invention
    at once. So the question it receives always says report, never verify."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""
    # Latest turn first: an older question in the same conversation would
    # steer the reader at the wrong thing.
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()[:400]
        if not isinstance(content, list):
            continue
        texts = [p["text"] for p in content
                 if isinstance(p, dict) and p.get("type") == "text"
                 and isinstance(p.get("text"), str)]
        if texts:
            return " ".join(texts).strip()[:400]
    return ""


# Descriptions already produced, keyed by the image bytes. An OpenAI client
# resends the whole conversation every turn, so without this a picture sent
# once is re-read on every subsequent turn - paying its cost again and, worse,
# describing it differently each time, so the answering model sees the same
# image change its mind. Bounded because the entries are long-lived by design.
_SEEN = {}
_SEEN_ORDER = []
_SEEN_MAX = 64
_SEEN_LOCK = threading.Lock()

# Serial reads with an independent timeout each, so one request could hold a
# connection for images x 600 s. Four covers comparing a couple of renders,
# which is what a caller actually does; more than that is a client bug or a
# way to occupy the gateway indefinitely.
MAX_IMAGES = int(os.environ.get("FXLLA_VISION_MAX_IMAGES", "4"))


def _cache_key(part):
    url = ((part or {}).get("image_url") or {}).get("url")
    if not isinstance(url, str):
        return None
    return hashlib.sha256(url.encode("utf-8", "replace")).hexdigest()


def _read_image(part, asked):
    """One image, as text, from the local vision model."""
    alias = _vision_alias()
    if not alias:
        raise RuntimeError(
            "this request carries an image and no catalog model can read one. "
            "Add a model with role 'vision' (see config/models.conf) and pull it")
    try:
        port, model_field = MANAGER.ensure(alias)
    except KeyError:
        raise RuntimeError(
            "the vision model %r is not downloaded. Pull it: fxlla pull %s"
            % (alias, alias))
    question = ("Describe this image for someone who cannot see it. List what "
                "is actually present and quote any text or lettering exactly. "
                "Report only what is there - do not judge whether anything is "
                "correct or expected.")
    if asked:
        question += ("\n\nFor relevance, they were asked: %r. Let that guide "
                     "what you cover, but still report what is present rather "
                     "than confirming anything." % asked)
    body = {"model": model_field, "max_tokens": 700, "messages": [
        {"role": "user", "content": [{"type": "text", "text": question},
                                     dict(part)]}]}
    # Straight to the backend port, not back through this server: a request
    # that re-entered here would translate its own translation.
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        answer = json.loads(r.read())
    choices = answer.get("choices") or []
    text = (choices[0].get("message", {}).get("content") or "").strip() if choices else ""
    if not text:
        raise RuntimeError("the vision model returned nothing for this image")
    return alias, text


def _read_cached(part, asked):
    """The description for this image, computed once per conversation."""
    key = _cache_key(part)
    if key is not None:
        with _SEEN_LOCK:
            hit = _SEEN.get(key)
        if hit:
            return hit
    result = _read_image(part, asked)
    if key is not None:
        with _SEEN_LOCK:
            if key not in _SEEN:
                _SEEN[key] = result
                _SEEN_ORDER.append(key)
                while len(_SEEN_ORDER) > _SEEN_MAX:
                    _SEEN.pop(_SEEN_ORDER.pop(0), None)
    return result


def add_vision(body, alias):
    """Replace images the chosen model cannot read with a description of them.

    Returns the alias that did the reading, or None when nothing was done -
    no image, routing disabled, or the chosen model is trusted to see for
    itself, in which case the image is passed through untouched because a
    description is strictly lossier than the thing itself.

    The order matters. Whether there is an image at all is answered in memory,
    so a request without one - nearly all of them - never touches the disk."""
    if not VISION_ROUTING:
        return None
    parts = _image_parts(body)
    if not parts:
        return None
    # Checked before the cap: the cap exists because each image costs a serial
    # read, and a model reading them itself pays no such price.
    if _can_see(alias):
        return None
    if len(parts) > MAX_IMAGES:
        raise RuntimeError(
            "this request carries %d images and the limit is %d: each is read "
            "separately, so a large batch holds the connection open for as long "
            "as all of them take. Send fewer, or raise "
            "FXLLA_VISION_MAX_IMAGES." % (len(parts), MAX_IMAGES))
    reader = None
    for message, index in parts:
        reader, text = _read_cached(message["content"][index], _asked(body))
        # Marked as a description on purpose: the model downstream must not
        # answer as though it had seen the image itself.
        message["content"][index] = {
            "type": "text",
            "text": "[an image was attached; %s read it and reports:]\n%s"
                    % (reader, text)}
    return reader


def _embed_identities():
    """(aliases, repos) of catalog entries with role embed. Both are needed:
    a pull by alias names the directory after the alias, a pull by org/repo
    names it after the repo, and .source records which repo either came from."""
    aliases, repos = set(), set()
    try:
        with open(CATALOG, encoding="utf-8") as fh:
            for line in fh:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 5 and not parts[0].startswith("#") and parts[3] == "embed":
                    aliases.add(parts[0])
                    repos.add(parts[1])
    except OSError:
        pass
    return aliases, repos


# What llama-server is started with (-c) by cmd_backend: for gguf the SERVED
# window, whatever the weights could do. mlx_lm.server serves the model's own
# window, which config.json declares.
SERVED_GGUF_CTX = int(os.environ.get("FXLLA_CTX", "32768"))

# Opt in to the window a model ADVERTISES rather than the one it was trained
# for. Only meaningful where rope scaling is baked in - it is the difference
# between 262144 and 1048576 on a model shipped with YaRN factor 4 - and it is
# off by default because the stretch costs attention quality across a window
# almost nobody fills, and the KV cache is allocated for all of it up front.
ROPE_STRETCH = os.environ.get("FXLLA_ROPE_STRETCH", "0") not in ("", "0", "false")


def model_context(alias):
    """The context window a request to this model actually gets, or None when
    it cannot be determined. This feeds opencode's per-model limit: without a
    declared limit its context meter and auto-compaction run on a made-up
    number."""
    d = os.path.join(MODELS_DIR, alias)
    if engine_for(alias) == "gguf":
        # The same reader bin/fxlla starts the backend with, so this reports
        # what is actually served rather than a number that merely used to be
        # passed. Falls back to the cap when the header cannot be read, which
        # is what the backend falls back to as well.
        served, _rope, _mtp = ggufmeta.serve_plan(d, SERVED_GGUF_CTX, ROPE_STRETCH)
        return served or SERVED_GGUF_CTX
    try:
        with open(os.path.join(d, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
        # A multimodal model nests its text settings: Gemma 4 keeps
        # max_position_embeddings under text_config, beside vision_config and
        # audio_config, and declares nothing at the top level. Reading only the
        # top level reported no window for it at all - and per the docstring
        # above, "no window" is exactly when opencode's meter and its
        # auto-compaction start running on a number nobody supplied.
        value = cfg.get("max_position_embeddings")
        if not value:
            nested = cfg.get("text_config")
            if isinstance(nested, dict):
                value = nested.get("max_position_embeddings")
        return int(value) if value else None
    except (OSError, ValueError, TypeError):
        return None


# Characters per token, used only to notice a request that cannot possibly
# fit. Deliberately generous: English prose runs about 4 and code lower, so
# dividing by 3.5 OVER-estimates the token count for prose and lands near the
# truth for code. The rule this feeds needs both its names - the failure it
# catches and the case it must not.
#
# Catches: a conversation that has grown past the window. Measured, that fails
# in the worst possible way - the backend simply never answers. 180 seconds
# with no error, no rejection, nothing. An opencode session here reached about
# 728k tokens against a 262k window and read as a hung chat rather than a full
# one, and `/compact` could not rescue it because compacting sends the whole
# conversation, which is precisely the thing that does not fit.
#
# Must not catch: a request merely near the limit. Hence a comparison against
# the window itself rather than some fraction of it, and an estimate that errs
# high only for prose. A wrongly refused request is visible and costs seconds;
# a silent stall costs a session.
_CHARS_PER_TOKEN = 3.5


def _text_chars(value):
    """Characters of prompt text in one content value, images excluded.

    An image is NOT its base64 length. A vision model spends a bounded number
    of tokens on a picture - hundreds, not the hundred thousand characters the
    encoding takes - and a model that cannot see has the image replaced by a
    short description before this ever runs. Counting the data: URL made an
    ordinary phone photo look like a 114k-token prompt, which is a refusal
    aimed at exactly the wrong request. How many images may ride along is
    already bounded elsewhere (FXLLA_VISION_MAX_IMAGES)."""
    if isinstance(value, str):
        return 0 if value.startswith("data:") else len(value)
    if isinstance(value, dict):
        return sum(_text_chars(v) for v in value.values())
    if isinstance(value, list):
        return sum(_text_chars(v) for v in value)
    return 0


def _prompt_chars(body):
    """Characters the model will actually be asked to read, or None.

    Counts message text and any tool schemas, since those ride along in the
    prompt. Returns None when there is nothing recognizable to measure - that
    means "no opinion", and the backend still gets the request."""
    total = 0
    seen = False
    messages = body.get("messages")
    if isinstance(messages, list):
        seen = True
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            total += _text_chars(msg.get("content"))
            for key in ("reasoning_content", "reasoning", "name"):
                piece = msg.get(key)
                if isinstance(piece, str):
                    total += len(piece)
            calls = msg.get("tool_calls")
            if isinstance(calls, list):
                total += len(json.dumps(calls))
    # /v1/completions carries its input at the top level instead, and the same
    # proxy serves it. Measuring only `messages` left that path unguarded.
    prompt = body.get("prompt")
    if isinstance(prompt, (str, list)):
        seen = True
        total += _text_chars(prompt)
    if not seen:
        return None
    tools = body.get("tools")
    if tools:
        total += len(json.dumps(tools))
    return total


def oversized_prompt(body, alias):
    """A message explaining why this request cannot fit, or None.

    Returning a message rather than a bool because the number is the whole
    point: "too big" sends someone to restart things, "about 728k against
    262k" sends them to start a new session, which is the only cure."""
    window = model_context(alias)
    if not window:
        return None
    chars = _prompt_chars(body)
    if not chars:
        return None
    estimate = int(chars / _CHARS_PER_TOKEN)
    reserve = body.get("max_tokens")
    if not isinstance(reserve, int) or reserve < 0:
        reserve = 0
    if estimate + reserve <= window:
        return None
    return (
        "this conversation is about %d tokens and '%s' has a %d token window, "
        "so it cannot be processed - the backend would accept it and never "
        "answer. Start a new session; compacting will not help, because "
        "compacting sends the whole conversation. (Token count is estimated "
        "from %d characters at %.1f characters per token.)"
        % (estimate + reserve, alias, window, chars, _CHARS_PER_TOKEN))


# How many times the same tool call may return the same answer before this is
# called a loop. 0 disables the check.
#
# The number that motivated it: a local model ran one command 240 times over
# 8.5 hours, each attempt blocking for its full 120 s timeout. The command
# started a server in the foreground, so it could never exit and the result was
# byte-identical every time. That is the whole mechanism - a model with no new
# information retrying is not a malfunction, it is the only thing it can do.
#
# 8 is chosen to sit well above deliberate repetition. Re-running a test suite
# three or four times while editing is ordinary; eight identical results in a
# row is not a workflow.
# Parsed defensively, unlike its neighbours, for a specific reason: the refusal
# this feeds tells the reader to "set FXLLA_LOOP_LIMIT=0 to disable", which
# invites FXLLA_LOOP_LIMIT=off. A bare int() there turns a wrong guess about
# one check into a gateway that will not import at all.
def _loop_limit():
    raw = os.environ.get("FXLLA_LOOP_LIMIT", "8")
    try:
        return int(raw)
    except (TypeError, ValueError):
        print("[gateway] FXLLA_LOOP_LIMIT=%r is not a whole number; "
              "using 8 (0 disables the check)" % raw, flush=True)
        return 8


LOOP_LIMIT = _loop_limit()


def _exchange_signatures(messages):
    """One signature per tool call/result pair, in conversation order.

    The signature deliberately includes the RESULT, not just the call. Same
    call with a CHANGING result is progress - polling a build, watching a file
    grow - and must never be mistaken for a loop. Same call with the same
    result is the model learning nothing, which is the thing worth stopping.

    Pairs are matched by ORDER, not by tool_call_id, and that is not a
    simplification. Keyed by id, a client that reuses ids - a fixed string, or
    a counter restarting each turn, which small local-model harnesses really do
    - had every historical call rewritten to the newest result for that id. A
    build polled twenty times with twenty different answers then looked like
    twenty identical ones and got refused: precisely the case this must let
    through. Order also means the id is never used as a dict key, so an id that
    is a list or a dict cannot raise on the way in.
    """
    pending, signatures = [], []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        calls = msg.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function")
                if not isinstance(fn, dict):
                    continue
                pending.append((str(fn.get("name")), str(fn.get("arguments"))))
        # A result answers the oldest call still waiting. Several calls issued
        # in one assistant turn are answered in the order they were made.
        if msg.get("role") == "tool" and pending:
            name, arguments = pending.pop(0)
            signatures.append(
                (name, arguments, _text_chars_repr(msg.get("content"))))
    return signatures


def _text_chars_repr(content):
    if isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True, default=str)


# The conversation-based check above cannot see past a compaction: opencode
# replaces the tool history with a summary, so the run restarts at zero and a
# loop that has been going for hours reads as its first attempt. That is not
# hypothetical - the 240-attempt session contained two compactions; they landed
# outside the run by luck.
#
# So remember the newest exchange of each request HERE, where compaction cannot
# reach. The window is wall-clock rather than a count, because the thing worth
# bounding is time burned, and a loop that pauses for an hour between attempts
# is not the failure this is about.
_LOOP_MEMORY = []                 # [(signature, monotonic seconds)]
_LOOP_MEMORY_LOCK = threading.Lock()
_LOOP_MEMORY_MAX = 512
_LOOP_WINDOW_S = 1800.0


def _remember_exchange(signature, now=None):
    """Record this request's newest exchange; return how many of the last ones
    in the window carry the same signature.

    Counted over a run at the tail rather than over the window as a whole, so
    the same command reached again after doing other work starts a fresh run
    instead of inheriting an old one.
    """
    stamp = time.monotonic() if now is None else now
    with _LOOP_MEMORY_LOCK:
        _LOOP_MEMORY.append((signature, stamp))
        if len(_LOOP_MEMORY) > _LOOP_MEMORY_MAX:
            del _LOOP_MEMORY[:len(_LOOP_MEMORY) - _LOOP_MEMORY_MAX]
        run = 0
        for sig, when in reversed(_LOOP_MEMORY):
            if sig != signature or stamp - when > _LOOP_WINDOW_S:
                break
            run += 1
        return run


def _reset_loop_memory():
    with _LOOP_MEMORY_LOCK:
        del _LOOP_MEMORY[:]


def looping_tool_calls(body):
    """A message naming the repetition, or None.

    Checked at the gateway rather than in any one client, because the loop is a
    property of the conversation and every client sends the conversation here.
    """
    if LOOP_LIMIT <= 0:
        return None
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    signatures = _exchange_signatures(messages)
    if not signatures:
        return None
    last = signatures[-1]

    in_conversation = 0
    for sig in reversed(signatures):
        if sig != last:
            break
        in_conversation += 1

    # Remembered here as well, so the count survives a compaction that empties
    # the conversation of its history. Whichever run is longer decides.
    remembered = _remember_exchange(last)
    run = max(in_conversation, remembered)
    if run < LOOP_LIMIT:
        return None

    name, arguments, _outcome = last
    shown = arguments if len(arguments) <= 200 else arguments[:200] + "..."
    where = ("this conversation has" if in_conversation >= remembered
             else "this model has, across compactions,")
    return (
        "%s called '%s' %d times in a row with the same arguments AND the same "
        "result, so the model is not learning anything new and will keep going. "
        "Look at that call rather than retrying: %s "
        "(set FXLLA_LOOP_LIMIT=0 to disable this check)"
        % (where, name, run, shown))


# A tool call that reached the text channel instead of the tool channel.
#
# mlx_lm ships a parser for this family, and it anchors the closing tag to the
# end of the string:  re.compile(r"<function=(.*?)</function>$", re.DOTALL).
# qwen3-coder sometimes closes a Llama-shaped call with a Hermes-shaped
# `</tool_call>`, so `</function>` is no longer last, the anchor fails, and the
# whole call falls through as prose. Measured at temperature 0: the prompt
# "Run the shell command: echo hi" parses 3/3, and "Use the bash tool to run:
# echo hi" parses 0/3 - deterministic, and about the wording, not luck.
#
# The client never sees the text channel, so from opencode's side the model
# simply did not call anything. evals/README.md already names this class and
# says the remedy is a serving-layer fix; this is that fix.
_TEXT_CALL_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_TEXT_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)


def _coerce_argument(value, name, schema):
    """Text is all the channel carries, so give the declared type back."""
    text = value.strip()
    spec = (schema or {}).get(name) or {}
    kind = str(spec.get("type", "string")).lower()
    if text.lower() == "null":
        return None
    try:
        if kind.startswith(("int", "long")):
            return int(text)
        if kind.startswith(("num", "float")):
            return float(text)
        if kind.startswith("bool"):
            return text.lower() in ("true", "1", "yes")
        if kind in ("object", "array"):
            return json.loads(text)
    except (TypeError, ValueError):
        return text
    return text


def text_tool_calls(content, tools):
    """Tool calls hiding in a text reply, in OpenAI shape, or None.

    Only attempted when the request declared tools and the name matches one of
    them: a reply that merely discusses `<function=...>` is prose, and turning
    prose into a call would be worse than missing one.
    """
    if not content or not tools or "<function=" not in content:
        return None
    schemas = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            schemas[fn["name"]] = (fn.get("parameters") or {}).get("properties") or {}
    if not schemas:
        return None
    calls = []
    for match in _TEXT_CALL_RE.finditer(content):
        name = match.group(1).strip()
        if name not in schemas:
            continue        # not a tool this request offered
        args = {}
        for pname, pvalue in _TEXT_PARAM_RE.findall(match.group(2)):
            args[pname.strip()] = _coerce_argument(pvalue, pname.strip(), schemas[name])
        calls.append({
            "id": "call_%s" % hashlib.sha1(
                ("%s%s%d" % (name, json.dumps(args, sort_keys=True), len(calls))
                 ).encode()).hexdigest()[:16],
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    return calls or None


class _StreamRescue:
    """Watches an SSE reply for a tool call that landed in the text channel.

    Passes everything through untouched except `data: [DONE]`, which it holds
    so a recovered call can be appended before the client stops reading. It
    therefore costs a normal reply nothing: no buffering, no delay, and if
    nothing needs repairing the held line is emitted exactly as it arrived.
    """

    def __init__(self, tools):
        self.tools = tools
        self.text = []
        self.had_tool_calls = False
        self.done_line = None
        self.model = ""

    def consume(self, data):
        """(bytes to forward, bytes still incomplete)."""
        out = bytearray()
        while b"\n" in data:
            line, data = data.split(b"\n", 1)
            stripped = line.strip()
            if stripped == b"data: [DONE]":
                self.done_line = line + b"\n"
                continue                      # held until finish()
            self._inspect(stripped)
            out += line + b"\n"
        return bytes(out), data

    def _inspect(self, line):
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if not payload:
            return
        try:
            obj = json.loads(payload)
        except ValueError:
            return
        if isinstance(obj.get("model"), str):
            self.model = obj["model"]
        for choice in obj.get("choices") or []:
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            if delta.get("tool_calls"):
                self.had_tool_calls = True
            piece = delta.get("content")
            if isinstance(piece, str):
                self.text.append(piece)

    def finish(self, leftover):
        out = bytearray(leftover)
        calls = None
        if not self.had_tool_calls:
            calls = text_tool_calls("".join(self.text), self.tools)
        if calls:
            chunk = {"object": "chat.completion.chunk", "model": self.model,
                     "choices": [{"index": 0, "finish_reason": "tool_calls",
                                  "delta": {"tool_calls": [
                                      dict(c, index=i) for i, c in enumerate(calls)]}}]}
            out += b"data: " + json.dumps(chunk).encode() + b"\n\n"
            log("recovered %d tool call(s) from the text channel of a stream; "
                "the backend's parser did not match them" % len(calls))
        out += self.done_line or b"data: [DONE]\n\n"
        return bytes(out)


def downloaded_models():
    """Map alias -> {size_mb} for CHAT models with a completion marker.

    Embedding models live in the same store but cannot chat: serving one here
    would spawn llama-server without --embeddings on a BERT, and registering
    one in an editor is exactly how a non-chat model ends up as somebody's
    local chat model. They are fxlla kb's business, not the gateway's."""
    out = {}
    if not os.path.isdir(MODELS_DIR):
        return out
    embed_aliases, embed_repos = _embed_identities()
    for name in sorted(os.listdir(MODELS_DIR)):
        d = os.path.join(MODELS_DIR, name)
        if not (os.path.isdir(d) and os.path.exists(os.path.join(d, ".source"))):
            continue
        if name in embed_aliases:
            continue
        try:
            with open(os.path.join(d, ".source"), encoding="utf-8") as fh:
                if fh.read().strip() in embed_repos:
                    continue
        except OSError:
            pass
        out[name] = {"size_mb": dir_size_mb(d)}
    return out


class Backend:
    def __init__(self, alias, port, proc, size_mb, model_field, engine, context=None):
        self.alias = alias
        self.port = port
        self.proc = proc
        self.size_mb = size_mb
        self.model_field = model_field  # value to send in the proxied 'model' field
        self.engine = engine            # mlx | gguf | omlx | mtplx, resolved at load
        self.last_used = time.monotonic()
        # Requests currently being proxied to this backend. last_used is
        # stamped when one is dispatched and never again, so a generation that
        # runs longer than the keep-warm window looks idle while it is still
        # streaming - and the reaper killed it mid-answer. A count, not a
        # refreshed timestamp: it does not depend on anything ticking.
        self.inflight = 0
        # What this backend is working on right now, for /health. Without it a
        # slow first turn and a hung one look identical from outside.
        self.started = 0.0
        self.prompt_tokens = None
        # Output tokens seen so far on the in-flight request, so /health can
        # say "generating, 1900 tokens" instead of leaving it as "reading".
        self.produced = 0
        # Context window, read once at registration (it can touch a GGUF header).
        self.context = context
        # Who asked for the in-flight request, for the monitor.
        self.client_addr = None
        self.client_app = None
        self.max_tokens = None
        self.cached_tokens = None
        # (monotonic, produced) per progress() call, for a live delta rate.
        self.progress_pts = collections.deque(maxlen=64)


def engine_for(alias):
    """Engine marker for a model: 'gguf', 'omlx', 'mtplx', or 'mlx' (default)."""
    try:
        with open(os.path.join(MODELS_DIR, alias, ".engine")) as f:
            return f.read().strip() or "mlx"
    except Exception:
        return "mlx"


def model_field_from(alias, engine):
    """The 'model' value a backend expects given its engine: the path for MLX,
    the alias for GGUF (llama-server --alias) and OMLX. Never trust the backend's
    enumerated id, since mlx_lm.server lists the whole HF cache in /v1/models.

    OMLX must get the bare alias, not the path: mlx_lm.server ignores the field
    entirely (single model), so sending it a path is harmless, but omlx serves
    the model under its directory basename (== the alias) and actively validates
    the field - a path would 404 every request. MTPLX is a single-model server
    that accepts any model field, so the alias is the tidy choice there too."""
    return alias if engine in ("gguf", "omlx", "mtplx") else os.path.join(MODELS_DIR, alias)


def model_field_for(alias):
    """model_field_from with the engine resolved from disk."""
    return model_field_from(alias, engine_for(alias))


def rss_mb(pid):
    """Resident memory of a process in MB via ps, or 0 if unavailable."""
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)],
                                      stderr=subprocess.DEVNULL)
        return int(out.strip()) // 1024
    except Exception:
        return 0


class Manager:
    def __init__(self):
        self.backends = {}          # alias -> Backend
        self.loading = {}           # alias -> (Event, port) for in-flight loads
        self.lock = threading.Lock()
        self.epoch = 0              # bumped by unload_all to cancel in-flight loads
        # Threads inside ensure() blocked on a load (the loader counts too), and
        # when the current queue started. Read under the lock.
        self.waiting = 0
        self.oldest_wait_started = None

    def _alloc_port(self):
        used = {b.port for b in self.backends.values()}
        used |= {p for (_ev, p) in self.loading.values()}
        p = PORT_BASE
        while p in used:
            p += 1
        return p

    def _resident_mb(self):
        return sum(b.size_mb for b in self.backends.values())

    # Waits for a freshly spawned backend to answer. The poll interval decides the
    # floor on a model switch: a small model answers in about a second, and at a
    # one-second interval the first probe missed it and the loop reported ready at
    # two - a full second of sleep on every load. Watching the process as well
    # turns a backend that dies on start into an immediate failure instead of
    # burning the whole timeout.
    def _wait_ready(self, port, timeout=180, proc=None):
        url = "http://127.0.0.1:%d/v1/models" % port
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    r.read()
                    return True
            except Exception:
                pass
            if proc is not None and proc.poll() is not None:
                return False  # it exited instead of listening
            time.sleep(READY_POLL_INTERVAL)
        return False

    def _evict_one(self):
        if not self.backends:
            return
        victim = min(self.backends.values(), key=lambda b: b.last_used)
        log("evicting %s (LRU, %d MB)" % (victim.alias, victim.size_mb))
        try:
            victim.proc.terminate()
            try:
                victim.proc.wait(timeout=10)
            except Exception:
                victim.proc.kill()
        except Exception:
            pass
        del self.backends[victim.alias]
        TELEMETRY.event("evicted", victim.alias, size_mb=victim.size_mb)

    def ensure(self, alias):
        """Return (port, model_field) for alias, loading and evicting as needed.

        The lock is held only for the fast registry operations. The slow model
        load runs outside the lock, so requests to already-loaded models never
        block behind another model's startup. Concurrent requests for the same
        not-yet-loaded model wait on a per-model Event instead of the lock."""
        port = size_mb = load_epoch = None
        with self.lock:
            b = self.backends.get(alias)
            # A backend that exited on its own (a crash, an OOM kill, someone
            # stopping it) stayed registered, and every request after it got
            # "Connection refused" until an /admin/unload. Drop it and load
            # again. poll() is non-blocking and reaps the exit status.
            if b is not None and b.proc is not None and b.proc.poll() is not None:
                log("%s exited on its own (code %s); reloading"
                    % (alias, b.proc.returncode))
                TELEMETRY.event("exited", alias, code=b.proc.returncode)
                del self.backends[alias]
                b = None
            if b is not None:
                b.last_used = time.monotonic()
                return b.port, b.model_field
            entry = self.loading.get(alias)
            if entry is not None:
                ev = entry[0]
                loader = False
            else:
                models = downloaded_models()
                if alias not in models:
                    raise KeyError(alias)
                size_mb = models[alias]["size_mb"]
                while self.backends and self._resident_mb() + size_mb > BUDGET_MB:
                    self._evict_one()
                port = self._alloc_port()
                ev = threading.Event()
                self.loading[alias] = (ev, port)
                loader = True
                load_epoch = self.epoch
            # Counted for the loader and every waiter alike; undone in the
            # finally below, so a raise cannot leave the queue stuck.
            if self.waiting == 0:
                self.oldest_wait_started = time.monotonic()
            self.waiting += 1
        try:
            return self._finish_ensure(alias, loader, ev, port, size_mb, load_epoch)
        finally:
            with self.lock:
                self.waiting -= 1
                if self.waiting <= 0:
                    self.waiting = 0
                    self.oldest_wait_started = None

    def _finish_ensure(self, alias, loader, ev, port, size_mb, load_epoch):
        """The slow half of ensure(), run outside the lock (see ensure)."""
        if not loader:
            # another thread is loading this model; wait for it, do not hold a lock
            ev.wait(timeout=200)
            with self.lock:
                b = self.backends.get(alias)
                if b is None:
                    raise RuntimeError("model '%s' failed to load" % alias)
                b.last_used = time.monotonic()
                return b.port, b.model_field

        # loader path: spawn and wait OUTSIDE the lock
        proc = None
        ready = False
        stale = False
        model_field = None
        log_path = backend_log_path(alias)
        load_t0 = time.monotonic()
        TELEMETRY.event("load_started", alias, size_mb=size_mb)
        try:
            log("loading %s on :%d (%d MB)" % (alias, port, size_mb))
            # The backend's own output is the only place a failed start is
            # explained, so it goes to a log per model, truncated on each load.
            # It went to /dev/null, and `exec: mlx_lm.server: not found` reached
            # the client as "did not become ready" with nothing to go on.
            # Unlinked first: a previous process of this alias may still be
            # dying with the old file open at its own offset, and truncating
            # that same inode would let its next write leave NUL bytes.
            try:
                try:
                    os.unlink(log_path)
                except FileNotFoundError:
                    pass
                out = open(log_path, "wb")
            except OSError:
                out, log_path = subprocess.DEVNULL, None
            try:
                proc = subprocess.Popen([FXLLA_BIN, "_backend", alias, str(port)],
                                        stdout=out, stderr=subprocess.STDOUT)
            finally:
                if out is not subprocess.DEVNULL:
                    out.close()   # the child holds its own copy
            ready = self._wait_ready(port, proc=proc)
        finally:
            # Read before the lock: a GGUF header read is not registry work.
            ctx = None
            if ready:
                try:
                    ctx = model_context(alias)
                except Exception:
                    ctx = None
            with self.lock:
                self.loading.pop(alias, None)
                # If unload_all ran while this model was loading, do not register
                # it: the gateway already reported the memory freed.
                stale = ready and self.epoch != load_epoch
                if ready and not stale:
                    engine = engine_for(alias)
                    model_field = model_field_from(alias, engine)
                    self.backends[alias] = Backend(
                        alias, port, proc, size_mb, model_field, engine, ctx)
                ev.set()
        if not ready or stale:
            # Read before terminate(): a backend still loading at the timeout
            # would otherwise report exiting with -15, the signal we just sent.
            code = proc.poll() if proc is not None else None
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            if stale:
                TELEMETRY.event("load_failed", alias, reason="unloaded during load")
                raise RuntimeError("model '%s' was unloaded during load" % alias)
            why = ("exited with code %d" % code if code is not None
                   else "did not answer within the load timeout")
            generic = why   # the telemetry event carries no log line or path
            last = last_log_line(log_path)
            if last:
                why += ": " + last
            if log_path:
                why += " (log: %s)" % log_path
            log("failed to load %s: %s" % (alias, why))
            TELEMETRY.event("load_failed", alias, reason=generic)
            raise RuntimeError("backend for %s did not become ready: %s" % (alias, why))
        TELEMETRY.event("loaded", alias, seconds=round(time.monotonic() - load_t0, 1),
                        size_mb=size_mb)
        return port, model_field

    def backend_meta(self, alias):
        """(pid, engine) for a resident backend, or None if it is not loaded."""
        with self.lock:
            b = self.backends.get(alias)
            if b is None or b.proc is None:
                return None
            return b.proc.pid, b.engine

    def status(self):
        with self.lock:
            now = time.monotonic()
            out = []
            for b in self.backends.values():
                item = {"alias": b.alias, "port": b.port, "size_mb": b.size_mb,
                        "idle_s": int(now - b.last_used),
                        "inflight": b.inflight,
                        "context": b.context,
                        "keep_warm_left_s": None}
                if not b.inflight and KEEP_WARM_S > 0:
                    item["keep_warm_left_s"] = max(
                        0, int(KEEP_WARM_S - (now - b.last_used)))
                if b.cached_tokens is not None:
                    item["cached_tokens"] = b.cached_tokens
                if b.inflight:
                    item["client_addr"] = b.client_addr
                    item["client_app"] = b.client_app
                    item["max_tokens"] = b.max_tokens
                if b.inflight and b.started:
                    item["busy_s"] = int(now - b.started)
                    if b.prompt_tokens:
                        item["prompt_tokens"] = b.prompt_tokens
                    # Reading the prompt and writing the answer are different
                    # phases and cost differently, so say which one is running.
                    # Once a token has come back the model is generating, and
                    # reporting "context to read" then is what made a normal
                    # long answer look like a stuck prefill.
                    produced = getattr(b, "produced", 0)
                    if produced > 0:
                        item["phase"] = "generating"
                        item["output_tokens"] = produced
                        item["live_tps"] = self._live_tps(b, now)
                    else:
                        item["phase"] = "reading prompt"
                out.append(item)
            return out

    @staticmethod
    def _live_tps(b, now):
        """Deltas per second over the last 2 s of progress() calls. None for
        omlx (about 14 tokens per delta, so a delta rate would be false) and
        when fewer than 2 points fall in the window."""
        if b.engine == "omlx":
            return None
        pts = [p for p in tuple(b.progress_pts) if p[0] >= now - 2.0]
        if len(pts) < 2 or pts[-1][0] <= pts[0][0]:
            return None
        return round((pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0]), 1)

    def progress(self, alias, produced):
        """Update the live output-token count for the in-flight request.

        Best-effort and lock-free on purpose: it runs per streamed chunk and a
        stale read from status() is harmless - the number only informs a human
        watching it climb."""
        b = self.backends.get(alias)
        if b is not None:
            b.produced = produced
            b.progress_pts.append((time.monotonic(), produced))

    def begin(self, alias, prompt_tokens=None, client_addr=None, client_app=None,
              max_tokens=None):
        """Mark a request as in flight against this backend.

        The prompt size and start time are kept so /health can say what the
        gateway is DOING, not only what it is holding. A long first turn is
        indistinguishable from a hang from the outside: a 169k-token
        conversation spends about 80 seconds in prefill before a single token
        appears, and nothing anywhere said so - it was cancelled three times in
        a row for looking dead."""
        with self.lock:
            b = self.backends.get(alias)
            if b is not None:
                b.inflight += 1
                b.last_used = time.monotonic()
                b.started = time.monotonic()
                b.prompt_tokens = prompt_tokens
                b.produced = 0
                b.client_addr = client_addr
                b.client_app = client_app
                b.max_tokens = max_tokens
                b.cached_tokens = None
                b.progress_pts.clear()
            return b

    def end(self, alias, backend=None):
        """Release it, and restamp: idleness starts when the answer finishes,
        not when it was asked for.

        Pass the Backend that begin() returned. By alias alone, a request on a
        backend that died and was reloaded meanwhile released the NEW one, and
        the reaper could then stop it while its own request still streamed."""
        with self.lock:
            b = self.backends.get(alias) if backend is None else backend
            if b is not None:
                b.inflight = max(0, b.inflight - 1)
                b.last_used = time.monotonic()

    def reap_idle(self, idle_s):
        """Unload backends untouched for longer than idle_s. Returns aliases.

        The gateway used to free memory only under pressure - a model was
        evicted when the NEXT load would not fit, and otherwise sat there
        forever. FXLLA_KEEP_WARM said "auto-stop after N idle minutes" and was
        read by `fxlla on` alone, so the multi-model path, which is the one
        most people run, quietly ignored it while /health reported an idle
        counter that nothing acted on. Two 27B models held 45 GB between them
        after a quarter of an hour of silence here.

        Decided under the lock, terminated outside it, the way unload_all
        does: a slow exit must not block the request that is arriving.
        """
        if idle_s <= 0:
            return []
        now = time.monotonic()
        with self.lock:
            # inflight, not just the clock: a long generation is stamped when
            # it is dispatched and never again, so it reads as idle while it
            # is still streaming. Killing it there cuts the answer off mid
            # sentence for a client that is doing nothing wrong.
            victims = [b for b in self.backends.values()
                       if not b.inflight and now - b.last_used >= idle_s]
            for b in victims:
                del self.backends[b.alias]
        for b in victims:
            try:
                b.proc.terminate()
                try:
                    b.proc.wait(timeout=10)
                except Exception:
                    b.proc.kill()
            except Exception:
                pass
            log("unloaded %s (idle %ds, %d MB)"
                % (b.alias, int(now - b.last_used), b.size_mb))
            TELEMETRY.event("idle_unloaded", b.alias, idle_s=int(now - b.last_used),
                            size_mb=b.size_mb)
        return [b.alias for b in victims]

    def unload_all(self):
        """Terminate every resident backend and clear the registry, freeing
        their memory. The gateway keeps serving and reloads a model on the next
        request. Returns the aliases that were unloaded, only after the
        processes have actually exited so their memory is released."""
        with self.lock:
            self.epoch += 1          # cancel any in-flight load (see ensure)
            victims = list(self.backends.values())
            freed = list(self.backends.keys())
            self.backends.clear()
        # terminate and wait outside the lock so a slow exit does not block
        # other requests; wait so the memory is gone before we return.
        for b in victims:
            try:
                b.proc.terminate()
            except Exception:
                pass
        for b in victims:
            try:
                b.proc.wait(timeout=10)
            except Exception:
                try:
                    b.proc.kill()
                except Exception:
                    pass
            TELEMETRY.event("admin_unloaded", b.alias, size_mb=b.size_mb)
        return freed

    def shutdown(self):
        self.unload_all()


MANAGER = Manager()
TELEMETRY = telemetry.Telemetry()

# Engines whose servers honor stream_options.include_usage. mtplx is left out:
# it reports usage on its own terms and must not be sent options it may reject.
# gguf is left out too: llama-server puts `timings` on the finish chunk, and
# with include_usage it moves to the usage-only chunk the stripper removes, so
# the client would lose it. Its counts come from `timings` instead.
_USAGE_INJECT_ENGINES = {"mlx", "omlx"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass  # quiet; the gateway logs what it needs

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            models = downloaded_models()
            self._json(200, {"object": "list", "data": [
                {"id": a, "object": "model", "owned_by": "fxlla",
                 "size_mb": m["size_mb"], "context": model_context(a)}
                for a, m in models.items()]})
        elif self.path.rstrip("/") in ("/health", "/v1/health"):
            self._json(200, {"status": "ok", "resident": MANAGER.status(),
                             "budget_mb": BUDGET_MB})
        elif self.path.rstrip("/") in ("/monitor", "/v1/monitor"):
            self._json(200, monitor_snapshot(self.server.server_address[1]))
        else:
            self._json(404, {"error": {"message": "not found: %s" % self.path}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""

        # Admin: free resident models so a heavy local job (media generation)
        # has headroom in unified memory. The gateway reloads on demand after.
        # Loopback only, so a non-local bind (FXLLA_HOST=0.0.0.0) cannot let a
        # remote host unload models and deny inference.
        if self.path.rstrip("/") == "/admin/unload":
            if not _is_loopback(self.client_address[0]):
                self._json(403, {"error": {"message": "admin endpoints are loopback-only"}})
                return
            freed = MANAGER.unload_all()
            if freed:
                log("unloaded on request: %s" % ", ".join(freed))
            self._json(200, {"unloaded": freed})
            return

        try:
            body = json.loads(raw or b"{}")
        except Exception:
            self._json(400, {"error": {"message": "invalid JSON body"}})
            return

        alias = body.get("model")
        if not alias:
            self._json(400, {"error": {"message": "missing 'model' field"}})
            return

        # An image the chosen model cannot read is translated before anything
        # else, so a failure here is reported as itself rather than surfacing
        # as the backend's "Only 'text' content type is supported".
        try:
            reader = add_vision(body, alias)
        except Exception as e:
            self._json(502, {"error": {
                "message": "could not read the image in this request: %s" % e,
                "type": "vision_failed"}})
            return

        # AFTER add_vision, and that order is the whole correctness of it: this
        # measures the body that will actually be sent. Measured before it, a
        # single ordinary photo attached to a text-only model was refused as a
        # 114k-token overflow, because the base64 was counted as prompt text -
        # while add_vision was about to replace that image with a 116-character
        # description that fits anything. A rejection rule has to name the
        # legitimate case it must not catch, and that was it.
        #
        # Still before MANAGER.ensure, so a request that genuinely cannot fit
        # does not first cost 16 GB of weights coming off disk.
        # Before the size check, because a looping conversation eventually
        # becomes an oversized one and the loop is the cause. Told about its
        # size, someone starts a new session and loops again; told about the
        # repetition, they look at the call.
        # Wrapped like its neighbours. A guard is not worth a request: if this
        # cannot read a conversation it has no opinion about it, and the
        # backend still gets its chance.
        try:
            loop = looping_tool_calls(body)
        except Exception:  # noqa: BLE001
            loop = None
        if loop:
            self._json(400, {"error": {"message": loop, "type": "tool_loop"}})
            return

        overflow = oversized_prompt(body, alias)
        if overflow:
            self._json(400, {"error": {"message": overflow,
                                       "type": "context_overflow"}})
            return

        try:
            port, model_field = MANAGER.ensure(alias)
        except KeyError:
            self._json(404, {"error": {
                "message": "model '%s' is not downloaded. Pull it: fxlla pull %s" % (alias, alias),
                "type": "model_not_found"}})
            return
        except Exception as e:
            self._json(503, {"error": {"message": "could not load '%s': %s" % (alias, e)}})
            return

        # Held for the whole request, not just the dispatch: the reaper reads
        # this, and a generation that outlives the keep-warm window was being
        # terminated while it streamed.
        chars = _prompt_chars(body) or 0
        max_tokens = body.get("max_tokens")
        if max_tokens is None:
            max_tokens = body.get("max_completion_tokens")
        held = MANAGER.begin(alias, int(chars / _CHARS_PER_TOKEN) if chars else None,
                             client_addr=self.client_address[0],
                             client_app=telemetry.client_tag(self.headers.get("User-Agent")),
                             max_tokens=max_tokens)
        self._held = held
        try:
            self._proxy(alias, port, model_field, body, reader)
        finally:
            MANAGER.end(alias, held)

    def _proxy(self, alias, port, model_field, body, reader):
        # send the model value each backend expects (path for MLX, alias for GGUF)
        if reader:
            log("read the image with %s, answering with %s" % (reader, alias))
        body["model"] = model_field
        measure = metrics.is_completion_path(self.path)
        # Ask the engine for exact usage when the client did not. Decided from
        # the resident backend's own engine, read once before dispatch. The
        # extra usage-only event is stripped again before the client sees it.
        injected = False
        if measure and body.get("stream"):
            try:
                opts = body.get("stream_options")
                asked = isinstance(opts, dict) and opts.get("include_usage")
                meta = MANAGER.backend_meta(alias)
                if (not asked and meta is not None
                        and meta[1] in _USAGE_INJECT_ENGINES):
                    base = opts if isinstance(opts, dict) else {}
                    body["stream_options"] = {**base, "include_usage": True}
                    injected = True
            except Exception as e:  # noqa: BLE001
                log("metrics: %s" % e)
        payload = json.dumps(body).encode()
        upstream = "http://127.0.0.1:%d%s" % (port, self.path)
        req = urllib.request.Request(upstream, data=payload,
                                     headers={"Content-Type": "application/json"})
        start = time.monotonic()
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            if reader:
                self.send_header("X-Fxlla-Vision", reader)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            if measure:
                self._record(alias, start, None, None, body, "error")
            return
        except Exception as e:
            self._json(502, {"error": {"message": "backend error: %s" % e}})
            if measure:
                self._record(alias, start, None, None, body, "error")
            return

        # stream the response back (works for both plain JSON and SSE)
        self.send_response(resp.status)
        ctype = resp.headers.get("Content-Type", "application/json")
        self.send_header("Content-Type", ctype)
        # Say that two models were used. Without this, a description that goes
        # wrong looks like the answering model being wrong, and the debugging
        # goes to the wrong place.
        if reader:
            self.send_header("X-Fxlla-Vision", reader)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        streaming = "event-stream" in ctype
        sm = metrics.StreamMetrics(start) if (measure and streaming) else None
        stripper = metrics.UsageChunkStripper() if (injected and streaming) else None
        # Buffered whenever the reply might carry a tool call the backend's own
        # parser dropped into the text channel, since repairing it means
        # rewriting the body. Metrics want the buffer anyway.
        rescue = bool(body.get("tools"))
        buf = bytearray() if (measure or rescue) and not streaming else None
        # Streaming is repaired without buffering the answer: the text flows
        # through as it arrives and only `data: [DONE]` is held back, so a
        # recovered call can be appended before the client stops reading.
        # Buffering the whole stream would have worked too and would have cost
        # every reply its incremental output to fix the few that need it.
        seen = _StreamRescue(body.get("tools")) if (rescue and streaming) else None
        pending = b""
        finish = None   # "error" or "client_gone" when the exchange broke
        got = 0
        cut = False
        done = False
        tail8 = b""
        try:
            while True:
                try:
                    chunk = resp.read1(1024)
                except Exception:
                    finish = "error"   # the upstream died mid-reply
                    break
                if not chunk:
                    break
                got += len(chunk)
                if not done:
                    # a marker split across two reads is still found
                    done = b"data: [DONE]" in tail8 + chunk
                    tail8 = (tail8 + chunk)[-16:]
                if sm is not None:
                    try:
                        sm.feed(chunk)
                        # Live output count so `fxlla status` can show progress
                        # while the answer is still streaming. Cheap: the delta
                        # count is already maintained by StreamMetrics.
                        MANAGER.progress(alias, sm.deltas)
                    except Exception:
                        sm = None  # never let metrics break the proxy
                if buf is not None and len(buf) < 4 * 1024 * 1024:
                    buf.extend(chunk)
                if stripper is not None:
                    held = b""
                    try:
                        held = stripper.held()
                        chunk = stripper.consume(chunk)
                    except Exception as e:  # noqa: BLE001
                        # Fail open: the client gets the bytes as they came,
                        # with whatever the stripper had held back.
                        log("metrics: %s" % e)
                        stripper = None
                        chunk = held + chunk
                if seen is not None:
                    pending += chunk
                    out, pending = seen.consume(pending)
                    chunk = out
                if not chunk:
                    continue
                if not (rescue and not streaming):
                    try:
                        self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    except OSError:
                        finish = "client_gone"
                        break
            if finish is None:
                # http.client reports a short body as a clean EOF when read in
                # pieces, so a stream cut off upstream is told apart here. This
                # only classifies the record; the client still gets a normal end.
                try:
                    want = int(resp.headers.get("Content-Length", ""))
                except ValueError:
                    want = None
                if want is not None and got < want:
                    cut = True
                elif sm is not None and sm.finish is None and not done:
                    cut = True   # ended with neither [DONE] nor a finish_reason
            if finish is None:
                try:
                    if stripper is not None:
                        try:
                            tail = stripper.finish()
                        except Exception as e:  # noqa: BLE001
                            log("metrics: %s" % e)
                            try:
                                tail = stripper.held()
                            except Exception:  # noqa: BLE001
                                tail = b""
                        if seen is not None:
                            pending += tail
                        elif tail:
                            self.wfile.write(b"%X\r\n%s\r\n" % (len(tail), tail))
                    if rescue and not streaming:
                        out = self._rescue_tool_calls(bytes(buf), body.get("tools"))
                        self.wfile.write(b"%X\r\n%s\r\n" % (len(out), out))
                        self.wfile.flush()
                    elif seen is not None:
                        tail = seen.finish(pending)
                        if tail:
                            self.wfile.write(b"%X\r\n%s\r\n" % (len(tail), tail))
                            self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                except OSError:
                    finish = "client_gone"
        except Exception:
            finish = finish or "error"
        if finish is None and cut:
            finish = "error"
        elif finish is not None:
            # No terminating chunk was sent, so the connection cannot be reused.
            self.close_connection = True
            try:
                resp.close()
            except Exception:
                pass
        if measure:
            self._record(alias, start, sm, bytes(buf) if buf is not None else None,
                         body, finish)

    def _rescue_tool_calls(self, raw, tools):
        """The response body, with a text-channel tool call promoted if there
        is one. Returns the bytes unchanged on anything unexpected: a repair
        that mangles a good reply is worse than one that misses a bad one.
        """
        try:
            doc = json.loads(raw)
            choice = doc["choices"][0]
            message = choice["message"]
            if message.get("tool_calls"):
                return raw
            calls = text_tool_calls(message.get("content"), tools)
            if not calls:
                return raw
            message["tool_calls"] = calls
            # The tags were the call, not prose; leaving them would show the
            # user raw syntax AND feed them back as context next turn.
            message["content"] = None
            choice["finish_reason"] = "tool_calls"
            log("recovered %d tool call(s) from the text channel; the backend's "
                "parser did not match them" % len(calls))
            return json.dumps(doc).encode()
        except Exception:
            return raw

    def _record(self, alias, start, sm, body_bytes, body, finish=None):
        """Record one completed request: always in the in-memory telemetry,
        and in stats.jsonl when the backend is still resident (which is what
        keeps ram_mb real).

        Best-effort: any failure here is logged and swallowed so telemetry never
        affects the proxied response."""
        try:
            end = time.monotonic()
            rep = metrics.engine_report(None, None)
            estimated = False
            ttft_ms = tps = None
            tokens = 0
            reply_finish = None
            if sm is not None:
                ttft_ms, tps, tokens = sm.result(end)
                rep = sm.report()
                reply_finish = sm.finish
                if rep["output_tokens"] is None and tokens:
                    rep["output_tokens"] = tokens
                    estimated = True
            elif body_bytes is not None:
                # Non-streamed: no first-token signal, so tps is over the whole
                # wall time (includes prompt processing); an approximation.
                tokens = metrics.usage_from_json(body_bytes)
                tps = round(tokens / (end - start), 1) if tokens and end > start else None
                try:
                    doc = json.loads(body_bytes)
                    rep = metrics.engine_report(doc.get("usage"), doc.get("timings"))
                    reply_finish = (doc.get("choices") or [{}])[0].get("finish_reason")
                except Exception:  # noqa: BLE001
                    pass
            if rep["output_tokens"] is None and sm is None and finish == "error":
                rep["output_tokens"] = 0
            elif rep["output_tokens"] is None and sm is not None:
                rep["output_tokens"] = 0
            fin = finish or reply_finish
            held = getattr(self, "_held", None)
            if held is not None and rep.get("cached_tokens") is not None:
                held.cached_tokens = rep["cached_tokens"]
            ua = self.headers.get("User-Agent")
            extra = dict(rep)
            extra.update({
                "id": TELEMETRY.new_id(),
                "output_estimated": estimated,
                "duration_ms": int((end - start) * 1000),
                "finish": fin,
                "stream": bool(body.get("stream")),
                "client_addr": self.client_address[0],
                "client_app": telemetry.client_tag(ua),
            })
            meta = MANAGER.backend_meta(alias)
            if meta is None:
                # evicted between response and record: no RSS to report
                TELEMETRY.record(metrics.build_sample(
                    time.time(), alias, None, None, ttft_ms, tps, **extra))
                return
            pid, engine = meta
            sample = metrics.build_sample(
                time.time(), alias, engine, rss_mb(pid), ttft_ms, tps, **extra)
            metrics.append_sample(metrics.stats_file(), sample)
            TELEMETRY.record(sample)
        except Exception as e:
            log("metrics: %s" % e)


def _term(signum, frame):
    raise KeyboardInterrupt()


# Idle minutes before a resident backend is unloaded. Same name and same units
# as the single-model server's watchdog, because a user who set it once should
# not have to discover that it governed only one of the two ways to run this.
def _keep_warm_s():
    """Idle seconds before a backend is unloaded, or 0 for never.

    Parsed defensively because it is read at import: a typo like "10m" used to
    raise at module scope and take the whole gateway down before it bound a
    port, while the single-model watchdog treats the same bad value as an
    ignorable nuisance and keeps serving. A malformed setting should cost the
    feature, not the process."""
    raw = (os.environ.get("FXLLA_KEEP_WARM") or "10").strip()
    try:
        return max(0, int(raw)) * 60
    except ValueError:
        log("FXLLA_KEEP_WARM=%r is not a whole number of minutes; "
            "keeping backends resident" % raw)
        return 0


KEEP_WARM_S = _keep_warm_s()


def _git_version():
    try:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(["git", "-C", repo, "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return "unknown"
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


BOOT_TS = int(time.time())
VERSION = _git_version()
MACHINE = telemetry.MachineSampler(MODELS_DIR)


def monitor_snapshot(port=None):
    """Everything /monitor serves. Manager fields are read under its lock
    (status() takes the lock itself, so it is called first); the machine
    sample runs outside it because it shells out."""
    backends = MANAGER.status()
    with MANAGER.lock:
        resident_mb = MANAGER._resident_mb()
        waiting = MANAGER.waiting
        started = MANAGER.oldest_wait_started
        oldest = int(time.monotonic() - started) if started is not None else None
    snap = TELEMETRY.snapshot()
    machine = MACHINE.sample()
    return {
        "schema_version": 1,
        "instance": {"pid": os.getpid(), "host": HOST, "port": port or PORT,
                     "boot_ts": BOOT_TS, "budget_mb": BUDGET_MB,
                     "keep_warm_min": KEEP_WARM_S // 60, "version": VERSION},
        "backends": backends,
        "resident_mb": resident_mb,
        "work": {"waiting": waiting, "oldest_wait_s": oldest,
                 "client_gone_total": snap["totals"].get("client_gone", 0)},
        "totals": snap["totals"],
        "series": snap["series"],
        "ttft_p95_ms": snap["ttft_p95_ms"],
        "day": snap["day"],
        "recent": snap["recent"],
        "events": snap["events"],
        "machine": machine,
    }

# Checked often enough that "10 minutes" is not really 15, cheap enough to
# ignore: it walks a dict of at most a handful of entries.
_REAP_INTERVAL_S = 30


def _reaper():
    while True:
        time.sleep(_REAP_INTERVAL_S)
        try:
            MANAGER.reap_idle(KEEP_WARM_S)
        except Exception as exc:      # never let the reaper kill the gateway
            log("reaper: %s" % exc)


def main():
    signal.signal(signal.SIGTERM, _term)
    if not STORE or not os.path.isdir(MODELS_DIR):
        log("FXLLA_STORE is unset or has no models dir: %r (start via 'fxlla serve')" % STORE)
        sys.exit(1)
    log("store=%s budget=%d MB backends from :%d" % (STORE, BUDGET_MB, PORT_BASE))
    log("models: %s" % (", ".join(downloaded_models().keys()) or "(none)"))
    if KEEP_WARM_S > 0:
        log("keep-warm: unloading a backend after %d idle min" % (KEEP_WARM_S // 60))
        threading.Thread(target=_reaper, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    log("listening on http://%s:%d/v1" % (HOST, PORT))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        MANAGER.shutdown()


if __name__ == "__main__":
    main()
