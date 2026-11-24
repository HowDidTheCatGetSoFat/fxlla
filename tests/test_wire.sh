#!/usr/bin/env bash
#
# RAG MCP wiring: `kb wire-opencode` registers a command whose interpreter tracks
# FXLLA_KB_INDEX, so an index-enabled setup runs the MCP server (and its core.py
# subprocess) under a sqlite-vec-capable python.
#
# Run: bash tests/test_wire.sh
# Re-exec under bash if started by another shell (BASH_SOURCE below is bash-only).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FXLLA="$ROOT/bin/fxlla"

fails=0
pass() { printf 'ok   - %s\n' "$1"; }
fail() { printf 'FAIL - %s\n' "$1"; fails=$((fails + 1)); }

CFG="$(mktemp -d)"
trap 'rm -rf "$CFG"' EXIT

oc="$CFG/opencode/opencode.json"

# reads mcp.servers.fxlla-rag.<field> (the v2 layout) as a compact JSON string
field() { python3 -c "import json,sys;print(json.dumps(json.load(open('$oc'))['mcp']['servers']['fxlla-rag']['$1']))"; }

# --- index OFF: plain python3, FXLLA_KB_INDEX empty in the environment ---
XDG_CONFIG_HOME="$CFG" FXLLA_STORE=/tmp bash "$FXLLA" kb wire-opencode >/dev/null
cmd_off="$(field command)"
env_off="$(field environment)"
case "$cmd_off" in
  *'"python3"'*) pass "index off: command uses python3" ;;
  *) fail "index off: command uses python3 (got $cmd_off)" ;;
esac
case "$env_off" in
  *'"FXLLA_KB_INDEX": ""'*) pass "index off: FXLLA_KB_INDEX empty in env" ;;
  *) fail "index off: FXLLA_KB_INDEX empty in env (got $env_off)" ;;
esac

# --- index ON: uv run --with sqlite-vec, FXLLA_KB_INDEX forwarded ---
XDG_CONFIG_HOME="$CFG" FXLLA_STORE=/tmp FXLLA_KB_INDEX=1 bash "$FXLLA" kb wire-opencode >/dev/null
cmd_on="$(field command)"
env_on="$(field environment)"
case "$cmd_on" in
  *'"uv"'*'"sqlite-vec"'*) pass "index on: command uses uv run --with sqlite-vec" ;;
  *) fail "index on: command uses uv (got $cmd_on)" ;;
esac
case "$env_on" in
  *'"FXLLA_KB_INDEX": "1"'*) pass "index on: FXLLA_KB_INDEX=1 in env" ;;
  *) fail "index on: FXLLA_KB_INDEX=1 in env (got $env_on)" ;;
esac

# --- the provider carries a dummy apiKey ---
# The local servers validate no key, but opencode PROMPTS for one when the
# provider omits it - measured: selecting the provider stopped dead at an api
# key dialog. Any non-empty string satisfies both sides.
XDG_CONFIG_HOME="$CFG" FXLLA_STORE=/tmp bash "$FXLLA" wire-opencode >/dev/null
prov="$(python3 -c "import json;print(json.dumps(json.load(open('$oc'))['providers']['local']['settings']))")"
case "$prov" in
  *'"apiKey"'*) pass "provider carries an apiKey so opencode does not prompt" ;;
  *) fail "provider carries an apiKey so opencode does not prompt (got $prov)" ;;
esac

# --- opencode v2 layout, and in-place migration of the v1 entries fxlla owns ---
# opencode v2 reads the v1 layout but flags it deprecated: provider -> providers,
# options -> settings, npm -> package, mcp.<name> -> mcp.servers.<name> and
# enabled -> disabled. fxlla writes v2, and when the file still holds its own v1
# entries it moves them rather than leave one server in two formats. Entries
# fxlla does not own stay exactly as the user wrote them.
M="$(mktemp -d)"; moc="$M/opencode/opencode.json"; mkdir -p "$M/opencode"
cat > "$moc" <<'JSON'
{
  "instructions": ["/keep/me.md"],
  "mcp": {
    "other": {"type": "local", "command": ["x"], "enabled": false},
    "fxlla-rag": {"type": "local", "command": ["old"], "environment": {"A": "1"}, "enabled": true},
    "fxlla-graph": {"type": "local", "command": ["old"], "enabled": false, "timeout": 7000}
  },
  "provider": {
    "other": {"npm": "x", "options": {"baseURL": "http://x"}},
    "local": {
      "npm": "@ai-sdk/openai-compatible", "name": "Old",
      "options": {"baseURL": "http://old", "apiKey": "my-own-key"},
      "models": {"m1": {"name": "M1", "limit": {"context": 8192, "output": 2048}}}
    }
  }
}
JSON
oc_get() { python3 -c "import json;d=json.load(open('$moc'));print(json.dumps($1))" 2>/dev/null || echo ERR; }
# expect <description> <python expression over d> <expected JSON>
expect() {
  if [ "$(oc_get "$2")" = "$3" ]; then pass "$1"; else fail "$1 (got $(oc_get "$2"))"; fi
}
mig() { XDG_CONFIG_HOME="$M" FXLLA_STORE=/tmp bash "$FXLLA" "$@" >/dev/null; }
mig wire-opencode
expect "migrated mcp servers live under mcp.servers" '"servers" in d["mcp"] and sorted(d["mcp"]["servers"])' '["fxlla-graph", "fxlla-rag"]'
expect "no fxlla server is left in the v1 mcp layout" '[k for k in d["mcp"] if k.startswith("fxlla-")]' '[]'
expect "server uses disabled, not enabled" 'd["mcp"]["servers"]["fxlla-rag"]["disabled"]' 'false'
expect "v2 server carries no 'enabled' key" '"enabled" in d["mcp"]["servers"]["fxlla-graph"]' 'false'
expect "a foreign mcp entry is left untouched" 'd["mcp"]["other"]' '{"type": "local", "command": ["x"], "enabled": false}'
expect "provider uses package, not npm" 'd["providers"]["local"]["package"]' '"aisdk:@ai-sdk/openai-compatible"'
expect "provider 'local' left the v1 key" '"local" in d.get("provider", {})' 'false'
expect "a foreign v1 provider is left untouched" 'd["provider"]["other"]["npm"]' '"x"'
expect "the user's apiKey survives the migration" 'd["providers"]["local"]["settings"]["apiKey"]' '"my-own-key"'
expect "baseURL is refreshed from the current gateway" 'd["providers"]["local"]["settings"]["baseURL"] != "http://old"' 'true'
expect "a v1 mcp timeout becomes the v2 per-phase timeout" 'd["mcp"]["servers"]["fxlla-graph"]["timeout"]' '{"startup": 7000, "catalog": 7000, "execution": 7000}'
# the model map and its limits move with the provider (no gateway here, so
# wire-opencode leaves the models alone and only migrates them)
expect "provider models and limits survive the migration" 'd["providers"]["local"]["models"]["m1"]' '{"name": "M1", "limit": {"context": 8192, "output": 2048}}'
expect "unrelated top-level keys are untouched" 'd["instructions"]' '["/keep/me.md"]'
mig wire-opencode --all
expect "wire-opencode --all registers all three servers under mcp.servers" 'sorted(d["mcp"]["servers"])' '["fxlla-graph", "fxlla-media", "fxlla-rag"]'
expect "--all leaves no fxlla server in the v1 layout" '[k for k in d["mcp"] if k.startswith("fxlla-")]' '[]'
expect "skills install keeps the user's own instructions" '"/keep/me.md" in d["instructions"]' 'true'
snap="$(cat "$moc")"
mig wire-opencode --all
if [ "$(cat "$moc")" = "$snap" ]; then pass "a second run changes nothing"; else fail "a second run changes nothing"; fi
rm -rf "$M"

# --- the chosen embedding model is baked in too ---
# opencode launches rag_mcp.py directly rather than through `fxlla kb mcp`, so
# this registration is the only thing telling it which model to embed with. Left
# out, rag_search embeds with the default against a base reindexed onto another,
# which either errors on width or, between two models of equal width, does not.
case "$env_off" in
  *'"FXLLA_EMBED_MODEL": ""'*) pass "no model chosen: FXLLA_EMBED_MODEL empty in env" ;;
  *) fail "no model chosen: FXLLA_EMBED_MODEL empty in env (got $env_off)" ;;
esac
XDG_CONFIG_HOME="$CFG" FXLLA_STORE=/tmp FXLLA_EMBED_MODEL=embed-qwen3 \
  bash "$FXLLA" kb wire-opencode >/dev/null
env_model="$(field environment)"
case "$env_model" in
  *'"FXLLA_EMBED_MODEL": "embed-qwen3"'*) pass "chosen model is forwarded to the MCP" ;;
  *) fail "chosen model is forwarded to the MCP (got $env_model)" ;;
esac

# the MCP path always points at rag_mcp.py
case "$cmd_on" in
  *rag_mcp.py*) pass "command targets rag_mcp.py" ;;
  *) fail "command targets rag_mcp.py (got $cmd_on)" ;;
esac

# --- kb stop is reachable ------------------------------------------------
# Reuse means a server outlives the command that started it, so there has to be a
# way to stop it. This also catches the subcommand not being wired at all: an
# unreachable branch falls through to core.py, which rejects the argument.
out="$(FXLLA_STORE=/tmp XDG_CONFIG_HOME="$CFG" bash "$FXLLA" kb stop 2>&1 || true)"
if grep -qi "no embedding server" <<< "$out"; then pass "kb stop reports an idle port"
else fail "kb stop reports an idle port (got: $(tail -1 <<< "$out"))"; fi
if ! grep -qi "invalid choice" <<< "$out"; then pass "kb stop is wired, not passed to core.py"
else fail "kb stop is wired, not passed to core.py"; fi

out="$(FXLLA_STORE=/tmp XDG_CONFIG_HOME="$CFG" bash "$FXLLA" __complete kb)"
if grep -qx stop <<< "$out"; then pass "completions list 'stop'"
else fail "completions list 'stop'"; fi

# --- fxlla do forwards the variables that choose the planner --------------
# config.env assigns with `: "${VAR:=default}"`, which sets without exporting,
# so a name missing from the export lists never reaches the subprocess. That is
# exactly how FXLLA_AGENT_MODEL - documented as the way to pick the planner -
# was being dropped, leaving every run on the hardcoded fallback silently.
for var in FXLLA_AGENT_MODEL FXLLA_DEFAULT_MODEL FXLLA_AGENT_MAX_STEPS FXLLA_AGENT_MAX_SECONDS; do
  if grep -q "$var" <<< "$(sed -n '/^FXLLA_AGENT_ENV=/,/"$/p' "$FXLLA")"; then
    pass "fxlla do forwards $var"
  else
    fail "fxlla do forwards $var"
  fi
done

# and cmd_do actually exports that list, not just declares it
# shellcheck disable=SC2016  # the literal text is what is being searched for
if grep -qF 'export $FXLLA_AGENT_ENV' "$FXLLA"; then pass "cmd_do exports the agent env"
else fail "cmd_do exports the agent env"; fi

# 'do' quoted: unquoted it reads as the loop keyword to shellcheck (SC1010).
# --- the gateway is handed the knobs it reads -------------------------------
# `fxlla serve` passes an explicit list, and config.env assigns with
# `: "${VAR:=default}"` - which sets without exporting. A name missing from
# that list means the user's value never arrives and the gateway silently
# falls back to its own default, which is how FXLLA_AGENT_MODEL was lost.
launch_block="$(grep -B 14 'nohup python3 .*fxlla_gateway.py' "$FXLLA")"
for var in FXLLA_CTX FXLLA_KEEP_WARM FXLLA_ROPE_STRETCH FXLLA_STATS_FILE FXLLA_STORE; do
  if grep -qE "(^|[[:space:]])$var=" <<< "$launch_block"; then
    pass "serve forwards $var to the gateway"
  else
    fail "serve forwards $var to the gateway"
  fi
done

out="$(FXLLA_STORE=/tmp XDG_CONFIG_HOME="$CFG" bash "$FXLLA" 'do' --help 2>&1 || true)"
if grep -qi "max-steps" <<< "$out"; then pass "do is wired to the loop"
else fail "do is wired to the loop (got: $(tail -1 <<< "$out"))"; fi

# --- the stats probe counts generated tokens, not visible ones -------------
# _probe() is Python embedded in bin/fxlla, so no Python suite reaches it, and
# it is a SECOND implementation of what gateway/metrics.py does. It once
# counted only `content` deltas over a 48-token budget - and a reasoning model
# spends all 48 thinking, emits no visible content, and made the probe report
# a flat `ttft=0 tps=0` into stats.jsonl, which the menu bar then charts.
# Measured on gemma-4-26b-qat: 0 before, 124.5-124.8 tok/s after.
#
# Asserted as the ABSENCE of the content-only test rather than a count of the
# right keys: a count breaks the day a fourth server spelling is added.
if grep -qE '^\s*if d\.get\("content"\):\s*$' "$FXLLA"; then
  fail "_probe is back to counting only visible content (reasoning models report 0)"
else
  pass "_probe does not test for visible content alone"
fi
for key in reasoning_content reasoning; do
  if grep -q "\"$key\"" "$FXLLA"; then
    pass "_probe knows the '$key' spelling"
  else
    fail "_probe does not know the '$key' spelling, so one engine reports 0"
  fi
done

# --- serve identifies the responder, not just "something answered" ---------
# A 200 on /health means something is listening, not that it is ours.
# llama-server answers /health too, so a stale single-model server holding the
# port made the gateway die on bind while `fxlla serve` read the squatter's
# reply and announced "Gateway up", listing the intruder's one model.
# Reproduced with a foreign listener on 8080 before the fix.
#
# budget_mb is a field only the gateway puts in /health, so requiring it in the
# readiness check is what tells the two apart. Asserted as the absence of the
# bare check: naming the exact good line breaks on any harmless rewrite.
# shellcheck disable=SC2016  # greps bin/fxlla for literal source; $ must not expand
if grep -qE 'curl -sf "http://\$FXLLA_HOST:\$FXLLA_PORT/health" >/dev/null' "$FXLLA"; then
  fail "serve treats any /health 200 as its own gateway"
else
  pass "serve does not treat a bare /health 200 as its own gateway"
fi
if grep -q 'budget_mb' "$FXLLA"; then
  pass "serve identifies the responder by a field only the gateway emits"
else
  fail "serve has no way to tell its gateway from another listener"
fi

# --- every setting the gateway reads is one `fxlla serve` forwards -----------
# config.env is sourced into plain shell variables and never exported, and the
# launch line passes an explicit allowlist. A setting the gateway reads that
# nobody forwards is therefore ignored in silence - the knob exists, the docs
# describe it, and setting it does nothing. Four were in that state:
# FXLLA_LOOP_LIMIT and the three vision knobs. Measured: FXLLA_LOOP_LIMIT=3 in
# config.env left the gateway still refusing at 8.
#
# Derived from the two files rather than from a list kept by hand, because a
# hand-kept list is the thing that drifted in the first place.
GW="$ROOT/gateway/fxlla_gateway.py"
reads="$(grep -ohE 'os\.environ\.get\("FXLLA_[A-Z_]+"' "$GW" \
         | sed 's/.*"\(FXLLA_[A-Z_]*\)".*/\1/' | sort -u)"
# Comments are stripped first. Without that, the comment ABOVE the forwarding
# loop - which names the very variables it is explaining - satisfied this check
# on its own, and removing a real forward left the test green.
forwards="$(sed -n '/^cmd_serve()/,/^}/p' "$FXLLA" | sed 's/#.*//' \
            | grep -ohE 'FXLLA_[A-Z_]+' | sort -u)"
missing=""
while IFS= read -r var; do
  [ -n "$var" ] || continue
  printf '%s\n' "$forwards" | grep -qx "$var" || missing="$missing $var"
done <<EOF
$reads
EOF
if [ -z "$missing" ]; then
  pass "every FXLLA_* the gateway reads is forwarded by 'fxlla serve'"
else
  fail "the gateway reads these but 'fxlla serve' never forwards them, so setting them in config.env does nothing:$missing"
fi

if [ "$fails" -ne 0 ]; then printf '\n%d test(s) failed\n' "$fails"; exit 1; fi
printf '\nall wire tests passed\n'
