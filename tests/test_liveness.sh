#!/usr/bin/env bash
#
# "Is the server up?" is answered from a pid file, and a pid file is a claim.
# Pids get reused, so `kill -0` alone says yes for whatever process holds that
# number now - and a stale file naming a live stranger makes `fxlla on` and
# `fxlla serve` refuse to start, pointing at a server that is not there.
#
# Run: bash tests/test_liveness.sh
set -uo pipefail

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

fails=0
ran=0
pass() { ran=$((ran + 1)); printf 'ok   - %s\n' "$1"; }
fail() { ran=$((ran + 1)); printf 'FAIL - %s\n' "$1"; fails=$((fails + 1)); }

# shellcheck disable=SC2034  # read by lib/core.sh when it is sourced below
REPO_ROOT="$ROOT"
# No `|| true`: under `set -u` a failed source ends the file before its first
# check, and silence reads exactly like success.
# shellcheck disable=SC1091
. "$ROOT/lib/core.sh"

TMP="$(mktemp -d)"
cleanup() { [ -n "${LIVE:-}" ] && kill "$LIVE" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

# A process we know the command line of, to stand in for both "ours" and
# "someone else's".
sleep 300 &
LIVE=$!
# Drop it from the job table so killing it at exit does not print a
# "Terminated" line into the middle of the test output.
disown "$LIVE" 2>/dev/null || true

if _pid_is "$LIVE" 'sleep 300'; then
  pass "a live pid whose command matches is alive"
else
  fail "a live pid whose command matches was reported dead"
fi

# The whole point: the pid is live, but it is not our process.
if _pid_is "$LIVE" 'fxlla_gateway\.py'; then
  fail "a live pid running something else was reported as ours (pid reuse gets through)"
else
  pass "a live pid running something else is not ours"
fi

DEAD_HOST=$(sh -c 'echo $$')   # exits immediately; its pid is free again
sleep 1
if _pid_is "$DEAD_HOST" 'sh'; then
  fail "a pid that has exited was reported alive"
else
  pass "a pid that has exited is not alive"
fi

if _pid_is "" 'anything'; then
  fail "an empty pid was reported alive"
else
  pass "an empty pid is not alive"
fi

# --- through the real accessors -----------------------------------------
GATEWAY_PID="$TMP/gateway.pid"
PID_FILE="$TMP/server.pid"

printf '%s\n' "$LIVE" > "$GATEWAY_PID"
if gateway_alive; then
  fail "gateway_alive believed a pid file naming an unrelated live process"
else
  pass "gateway_alive rejects a pid file naming an unrelated live process"
fi

printf '%s\n' "$LIVE" > "$PID_FILE"
if server_alive; then
  fail "server_alive believed a pid file naming an unrelated live process"
else
  pass "server_alive rejects a pid file naming an unrelated live process"
fi

rm -f "$GATEWAY_PID"
if gateway_alive; then
  fail "gateway_alive reported alive with no pid file at all"
else
  pass "gateway_alive is false with no pid file"
fi

# server_alive must still say yes for a real engine process, or `fxlla off` and
# the keep-warm watchdog would stop seeing a server they are meant to manage.
# All three engine command lines have to be accepted - gguf models run
# llama-server, MLX ones mlx_lm.server, OMLX ones `omlx serve` - so check the
# pattern (the same one server_alive uses) against a realistic command for each.
_engine_pat='llama-server|mlx_lm\.server|omlx-server|omlx serve|mtplx\.server\.openai'
while IFS='|' read -r label cmd; do
  [ -n "$label" ] || continue
  if printf '%s\n' "$cmd" | grep -qE -- "$_engine_pat"; then
    pass "the engine pattern accepts $label"
  else
    fail "the engine pattern rejects $label, so a running server would look stopped"
  fi
done <<'EOF'
llama-server|/usr/local/bin/llama-server --model /x --port 8080
mlx_lm.server|/opt/venv/bin/python -m mlx_lm.server --model /x --port 8080
omlx serve (argv)|/opt/omlx/.venv/bin/omlx serve --model-dir /x --port 8080
omlx-server (setproctitle)|omlx-server
mtplx (server.openai)|/opt/mtplx/.venv/bin/python -m mtplx.server.openai --model /x --port 8080
EOF

# The anchors are `omlx-server` and `omlx serve`, not a bare `omlx`, precisely so
# a path that merely contains the word - the console script installs under an
# `omlx` directory - does not read as a running server.
if printf '%s\n' "/Users/x/omlx-src/bin/python -m http.server 9000" | grep -qE -- "$_engine_pat"; then
  fail "a process under an omlx path but not running 'omlx serve' false-matched"
else
  pass "a process under an omlx path but not running 'omlx serve' is not matched"
fi

printf '\n%s\n' "-----"
EXPECTED=13
if [ "$ran" -ne "$EXPECTED" ]; then
  printf 'FAIL - ran %d assertions, expected %d (the file did not finish)\n' "$ran" "$EXPECTED"
  exit 1
fi
if [ "$fails" -eq 0 ]; then printf 'all %d liveness tests passed\n' "$ran"; else printf '%d test(s) failed\n' "$fails"; exit 1; fi
