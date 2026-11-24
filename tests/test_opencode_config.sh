#!/usr/bin/env bash
#
# lib/opencode_config.py, the reader/writer behind every wire-opencode command:
# the v1 to v2 translation (mirrors opencode's migrate.ts and normalize.ts),
# reading a file that is not strict JSON, and writing it back safely. The file
# holds API keys and the user's own settings, so a bad read must never turn into
# a write that replaces them.
#
# Run: bash tests/test_opencode_config.sh
# Re-exec under bash if started by another shell (BASH_SOURCE below is bash-only).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -u

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FXLLA="$ROOT/bin/fxlla"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PYTHONPATH="$ROOT/lib" TMP="$TMP" python3 - <<'PY'
import json, os, stat, sys
import opencode_config as oc

TMP = os.environ["TMP"]
fails = 0


def check(name, cond, got=None):
    global fails
    if cond:
        print("ok   - %s" % name)
    else:
        fails += 1
        print("FAIL - %s%s" % (name, "" if got is None else " (got %r)" % (got,)))


def write(name, text):
    path = os.path.join(TMP, name)
    with open(path, "w") as f:
        f.write(text)
    return path


# --- mcp.servers: the v2 container, or a user's v1 server named "servers" ---
# normalize.ts reads mcp.servers as the container unless its value is a v1
# entry (type local or remote); then it is a v1 server called "servers".
user = {"type": "local", "command": ["x"], "enabled": True}
d = oc.migrate({"mcp": {"servers": dict(user),
                        "fxlla-rag": {"type": "local", "command": ["r"], "enabled": False}}})
srv = d["mcp"]["servers"]
check("a v1 server named 'servers' moves into the container", srv.get("servers", {}).get("command") == ["x"], srv)
check("the fxlla v1 entry joins it in the container", srv.get("fxlla-rag", {}).get("disabled") is True, srv)
check("the user's server keeps its meaning (enabled -> not disabled)", srv.get("servers", {}).get("disabled") is False, srv)
d = {"mcp": {"servers": dict(user)}}
oc.set_server(d, "fxlla-graph", ["g"], {})
check("set_server over a v1 'servers' entry keeps it and adds ours",
      sorted(d["mcp"]["servers"]) == ["fxlla-graph", "servers"], d["mcp"])
d = oc.migrate({"mcp": {"servers": {"other": {"type": "local", "command": ["o"]}}}})
check("a real v2 container is left as it is", list(d["mcp"]["servers"]) == ["other"], d["mcp"])

# --- model and provider fields survive (migrate.ts migrateModel / StandardProvider) ---
v1 = {"provider": {"local": {
    "npm": "@ai-sdk/openai-compatible", "name": "L", "api": "http://api",
    "options": {"baseURL": "http://u", "apiKey": "k", "setCacheKey": True,
                "headers": {"X": "1", "bad": 2}, "body": {"b": 1}},
    "models": {"alias": {
        "id": "real-id", "name": "N", "family": "f", "tool_call": True,
        "modalities": {"input": ["text", "image"], "output": ["text"]},
        "interleaved": {"field": "reasoning_content"},
        "options": {"chat_template_kwargs": {"a": 1}, "reasoningEffort": "low"},
        "variants": {"hi": {"reasoningEffort": "high", "extra": 1}},
        "cost": {"input": 1, "output": 2, "cache_read": 0.5,
                 "context_over_200k": {"input": 3, "output": 4}},
        "status": "deprecated", "headers": {"H": "v"},
        "limit": {"context": 1000, "output": 10.9}}}}}}
p = oc.migrate(v1)["providers"]["local"]
m = p["models"]["alias"]
check("provider setCacheKey becomes compatibility.supportsPromptCacheKey",
      p.get("compatibility") == {"supportsPromptCacheKey": True}, p.get("compatibility"))
check("setCacheKey is not left in settings", "setCacheKey" not in p["settings"], p["settings"])
check("provider headers keep only string values", p.get("headers") == {"X": "1"}, p.get("headers"))
check("provider body is lifted out", p.get("body") == {"b": 1}, p.get("body"))
check("a model's id becomes modelID", m.get("modelID") == "real-id", m)
check("tool_call and modalities become capabilities",
      m.get("capabilities") == {"tools": True, "input": ["text", "image"], "output": ["text"]}, m.get("capabilities"))
check("interleaved becomes compatibility.reasoningField",
      m.get("compatibility") == {"reasoningField": "reasoning_content"}, m.get("compatibility"))
check("openai-compatible model options split into settings and body",
      m.get("settings") == {"reasoningEffort": "low"} and m.get("body") == {"chat_template_kwargs": {"a": 1}}, m)
check("variants become a list of id plus split options",
      m.get("variants") == [{"id": "hi", "settings": {"reasoningEffort": "high"}, "body": {"extra": 1}}], m.get("variants"))
check("cost becomes a base entry and a 200k context tier",
      m.get("cost") == [{"input": 1, "output": 2, "cache": {"read": 0.5}},
                        {"input": 3, "output": 4, "tier": {"type": "context", "size": 200000}}], m.get("cost"))
check("a deprecated model is disabled", m.get("disabled") is True, m)
check("model headers and family and name are kept",
      m.get("headers") == {"H": "v"} and m.get("family") == "f" and m.get("name") == "N", m)
check("limits are truncated to integers", m.get("limit") == {"context": 1000, "output": 10}, m.get("limit"))
mm = oc.migrate({"provider": {"local": {"npm": "other", "models": {"m": {
    "options": {"thinking": {"blockBinding": False}, "keep": 1}}}}}})["providers"]["local"]["models"]["m"]
check("a thinking blockBinding opt-out moves to compatibility",
      mm.get("compatibility") == {"supportsThinkingBlockBinding": False}
      and mm.get("settings") == {"keep": 1}, mm)

# --- odd v1 shapes are skipped, never raise ---
odd = [
    {"provider": {"local": {"options": [1]}}},
    {"provider": {"local": {"npm": 3}}},
    {"provider": {"local": {"models": {"m": {"limit": {"context": "x"}}}}}},
    {"provider": {"local": {"models": {"m": {"limit": {"context": float("inf"), "output": float("nan")}}}}}},
    {"provider": {"local": {"models": {"m": {"options": "x", "variants": [1], "cost": 5,
                                             "modalities": 3, "interleaved": 7, "provider": "p"}}}}},
]
raised = []
for case in odd:
    try:
        oc.migrate(case)
    except Exception as e:
        raised.append(e.__class__.__name__)
check("odd v1 shapes do not raise", not raised, raised)
lim = oc.migrate({"provider": {"local": {"models": {"m": {"limit": {
    "context": float("inf"), "input": "x", "output": 5}}}}}})["providers"]["local"]["models"]["m"]["limit"]
check("a bad limit value is dropped and the good one kept", lim == {"output": 5}, lim)
check("a non-string npm sets no package", "package" not in oc.migrate(
    {"provider": {"local": {"npm": 3}}})["providers"]["local"])

# --- reading a file that is not strict JSON ---
path = write("c.json", '{\n  // a note\n  "keep": "a // not a comment",\n'
                       '  /* block */ "list": [1, 2,],\n  "n": {"x": 1,},\n}\n')
d = oc.load(path)
check("JSONC comments and trailing commas are read",
      d == {"keep": "a // not a comment", "list": [1, 2], "n": {"x": 1}}, d)
check("a missing file reads as empty", oc.load(os.path.join(TMP, "nope.json")) == {})
check("an empty file reads as empty", oc.load(write("e.json", "  \n")) == {})
broken = write("b.json", '{"a": [1, 2')
try:
    oc.load(broken)
    check("a broken file raises ConfigError", False)
except oc.ConfigError as e:
    check("a broken file raises ConfigError", broken in str(e), str(e))
try:
    oc.open_config(broken)
    check("open_config exits on a broken file", False)
except SystemExit as e:
    check("open_config exits on a broken file, naming it", broken in str(e), str(e))

# --- writing ---
out = os.path.join(TMP, "w", "opencode.json")
os.makedirs(os.path.dirname(out))
oc.save(out, {"a": 1})
check("save writes a 0600 file", stat.S_IMODE(os.stat(out).st_mode) == 0o600, oct(os.stat(out).st_mode))
os.chmod(out, 0o644)
oc.save(out, {"a": 2})
check("save tightens a looser mode to 0600", stat.S_IMODE(os.stat(out).st_mode) == 0o600)
os.chmod(out, 0o400)
oc.save(out, {"a": 3})
check("save keeps an existing stricter mode", stat.S_IMODE(os.stat(out).st_mode) == 0o400)
check("save leaves no temp file behind", os.listdir(os.path.dirname(out)) == ["opencode.json"], os.listdir(os.path.dirname(out)))
check("the saved content is the data", json.load(open(out)) == {"a": 3})
try:
    oc.save(out, {"a": {1, 2}})
except TypeError:
    pass
check("a failed save leaves the old file and no temp file",
      json.load(open(out)) == {"a": 3} and os.listdir(os.path.dirname(out)) == ["opencode.json"])

sys.exit(1 if fails else 0)
PY
rc=$?

# --- through the CLI: a JSONC file keeps its keys, a broken one is not touched ---
J="$TMP/jsonc"; mkdir -p "$J/opencode"
printf '{\n  // my notes\n  "instructions": ["/keep/me.md"],\n}\n' > "$J/opencode/opencode.json"
XDG_CONFIG_HOME="$J" FXLLA_STORE="$TMP/store" bash "$FXLLA" wire-opencode >/dev/null 2>&1
if python3 -c "import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d['instructions']==['/keep/me.md'] and 'local' in d['providers'] else 1)" "$J/opencode/opencode.json"; then
  echo "ok   - wire-opencode on a JSONC file keeps its other keys"
else
  echo "FAIL - wire-opencode on a JSONC file keeps its other keys"; rc=1
fi
B="$TMP/broken"; mkdir -p "$B/opencode"
printf '{"instructions": [' > "$B/opencode/opencode.json"
cp "$B/opencode/opencode.json" "$B/before"
XDG_CONFIG_HOME="$B" FXLLA_STORE="$TMP/store" bash "$FXLLA" wire-opencode >/dev/null 2>"$B/err"; st=$?
if [ "$st" -ne 0 ]; then echo "ok   - wire-opencode on a broken file exits non-zero"
else echo "FAIL - wire-opencode on a broken file exits non-zero"; rc=1; fi
if cmp -s "$B/before" "$B/opencode/opencode.json"; then echo "ok   - and leaves the file byte-identical"
else echo "FAIL - and leaves the file byte-identical"; rc=1; fi
if grep -q "opencode.json" "$B/err"; then echo "ok   - and names the file in the error"
else echo "FAIL - and names the file in the error"; rc=1; fi

if [ "$rc" -ne 0 ]; then printf '\ntest(s) failed\n'; exit 1; fi
printf '\nall opencode_config tests passed\n'
