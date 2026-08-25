#!/usr/bin/env bash
#
# Pull argument handling that does not need the network: downloader validation
# and the usage message.
#
# Run: bash tests/test_pull.sh
#
# Several checks grep bin/fxlla for literal shell source, so the `$` in those
# patterns must NOT expand - single quotes are the point, not an oversight.
# shellcheck disable=SC2016
set -euo pipefail

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FXLLA="$ROOT/bin/fxlla"

fails=0
pass() { printf 'ok   - %s\n' "$1"; }
fail() { printf 'FAIL - %s\n' "$1"; fails=$((fails + 1)); }

# an unknown downloader is rejected before any network access
if FXLLA_STORE=/tmp bash "$FXLLA" pull tiny --downloader nope >/dev/null 2>&1; then
  fail "unknown downloader rejected"
else
  pass "unknown downloader rejected"
fi

# a bare pull prints usage and fails
if FXLLA_STORE=/tmp bash "$FXLLA" pull >/dev/null 2>&1; then
  fail "bare pull errors"
else
  pass "bare pull errors"
fi

# --- a projector is not a quant to choose between ---------------------------
# google/gemma-4-12B-it-qat-q4_0-gguf ships one build and one projector. Both
# end in .gguf, so counting every .gguf made a repo with nothing to choose look
# ambiguous: the pull refused, and listed `mmproj-...` as one of the builds to
# pick from - a file that is not a model and cannot be served as one.
#
# This is a SOURCE-level check, and it is weaker than running the thing: the
# file list comes from the network, so the real path cannot run offline. It
# pins the shape of the fix - the count and the offered list both read a
# projector-filtered variable - which is exactly what a careless edit would
# undo. The behaviour itself is verified by pulling the model.
if grep -q 'builds="\$(echo "\$ggufs" | grep -iv .mmproj' "$FXLLA"; then
  pass "the quant list is filtered of projectors"
else
  fail "the quant list is filtered of projectors"
fi
if grep -q 'elif \[ "\$(echo "\$builds" | grep -c \.)" -eq 1 \]' "$FXLLA"; then
  pass "ambiguity is counted over builds, not over every .gguf"
else
  fail "ambiguity is counted over builds, not over every .gguf"
fi
# Stated as an absence, not a count. The first version of this asserted
# "exactly 2 listings read $builds" and broke the moment a third, correct
# listing was added - a test that fails when the code gets better is worse than
# no test. What must hold is that NO listing offers the unfiltered set.
unfiltered="$(grep -c 'echo "\$ggufs" | while IFS= read -r l' "$FXLLA" || true)"
if [ "${unfiltered:-0}" -eq 0 ]; then
  pass "no 'Available' listing offers the unfiltered .gguf set"
else
  fail "no 'Available' listing offers the unfiltered .gguf set (found $unfiltered)"
fi

# A --quant that matches only a projector selects no model. Non-empty was being
# read as usable: it downloaded the projector, wrote a blank entry marker, and
# printed "Done" plus how to start a directory with nothing in it to start.
if grep -q 'matches only a projector, which is not a model' "$FXLLA"; then
  pass "a quant matching only a projector is refused"
else
  fail "a quant matching only a projector is refused"
fi

# --- the .engine marker records the engine that was resolved ----------------
# The non-gguf branch used to write a hardcoded "mlx", so a `--omlx` pull (or a
# catalog row with engine=omlx) landed on disk marked mlx, and every reader then
# launched mlx_lm.server on an omlx directory. The fix echoes $engine, which is
# non-empty by that point. Source-level, like the projector checks above: the
# behaviour is verified by a real pull, this pins the shape a careless edit undoes.
if grep -q 'echo "\$engine" > "\$dest/.engine"' "$FXLLA"; then
  pass "the .engine marker is written from the resolved engine"
else
  fail "the .engine marker is written from the resolved engine"
fi
if grep -q 'echo "mlx" > "\$dest/.engine"' "$FXLLA"; then
  fail "the .engine marker still hardcodes mlx (a --omlx pull would be mislabelled)"
else
  pass "the .engine marker no longer hardcodes mlx"
fi
# --omlx is a real engine flag, parsed the way --gguf/--mlx are.
if grep -q -- '--omlx)         engine=omlx' "$FXLLA"; then
  pass "pull accepts --omlx"
else
  fail "pull accepts --omlx"
fi
if grep -q -- '--mtplx)        engine=mtplx' "$FXLLA"; then
  pass "pull accepts --mtplx"
else
  fail "pull accepts --mtplx"
fi

# --- subfolder repo spec parsing (_repo_subdir) --------------------------------
# A repo spec may carry a subfolder (owner/name/mlx-8bit) so a pull can take only
# that subtree of a repo that also ships a huge full-precision copy at its root.
# shellcheck disable=SC2034  # REPO_ROOT is read by lib/core.sh when sourced below
REPO_ROOT="$ROOT"
# shellcheck disable=SC1090
set +e; . "$ROOT/lib/core.sh"; set -e
_rs() { _repo_subdir "$1" | tr '\t' '|'; }
if [ "$(_rs owner/name)" = "owner/name|" ]; then
  pass "a plain owner/name has no subdir"
else
  fail "a plain owner/name has no subdir (got '$(_rs owner/name)')"
fi
if [ "$(_rs owner/name/mlx-8bit)" = "owner/name|mlx-8bit" ]; then
  pass "a single subfolder is split off from the repo"
else
  fail "a single subfolder is split off (got '$(_rs owner/name/mlx-8bit)')"
fi
if [ "$(_rs owner/name/a/b)" = "owner/name|a/b" ]; then
  pass "a nested subfolder is kept whole"
else
  fail "a nested subfolder is kept whole (got '$(_rs owner/name/a/b)')"
fi
# The pull path must flatten the subtree and record the full spec in .source.
if grep -q 'mv "\$dest/\$subdir/"\* "\$dest/"' "$FXLLA"; then
  pass "a subfolder pull flattens the subtree into the model dir"
else
  fail "a subfolder pull flattens the subtree into the model dir"
fi
# Behavioural, repo-independent: the same awk the pull uses selects only the
# subtree from a synthetic listing (size<TAB>path rows).
_synth="$(printf '10\tconfig.json\n20\tmlx-8bit/config.json\n30\tmlx-8bit/w.safetensors\n40\tmlx-4bit/config.json\n')"
_sel="$(printf '%s\n' "$_synth" | awk -F'\t' -v s="mlx-8bit/" 'index($2,s)==1' | awk -F'\t' '{print $2}' | paste -sd, -)"
if [ "$_sel" = "mlx-8bit/config.json,mlx-8bit/w.safetensors" ]; then
  pass "the subdir filter selects only the subtree"
else
  fail "the subdir filter selects only the subtree (got '$_sel')"
fi
# And the flatten lifts that subtree up to the model-dir root.
_t="$(mktemp -d)"; mkdir -p "$_t/mlx-8bit"; : > "$_t/mlx-8bit/config.json"; : > "$_t/mlx-8bit/w.safetensors"
mv "$_t/mlx-8bit/"* "$_t/"; rm -rf "$_t/mlx-8bit"
if [ -f "$_t/config.json" ] && [ -f "$_t/w.safetensors" ] && [ ! -d "$_t/mlx-8bit" ]; then
  pass "the flatten lifts the subtree into the model dir"
else
  fail "the flatten lifts the subtree into the model dir"
fi
rm -rf "$_t"

# --- every aria2 invocation carries the stall guard -------------------------
# aria2 defaults --lowest-speed-limit to 0, which means it never abandons a
# connection that has stopped delivering - and a socket that stays OPEN while
# moving no bytes is not a timeout either, so nothing else rescues it. Counted
# rather than grepped once, because the failure this catches is a THIRD call
# site added later without the guard.
calls="$(grep -c '^\s*aria2c ' "$FXLLA" || true)"
guarded="$(grep -A 8 '^\s*aria2c ' "$FXLLA" | grep -c 'ARIA_STALL_GUARD' || true)"
if [ "$calls" -gt 0 ] && [ "$calls" -eq "$guarded" ]; then
  pass "all $calls aria2c invocations carry the stall guard"
else
  fail "all aria2c invocations carry the stall guard ($guarded of $calls)"
fi

# a dead connection is caught by the timeout, which IS retryable
if grep -qE 'ARIA_STALL_GUARD=.*--timeout=[0-9]' "$ROOT/lib/core.sh"; then
  pass "a silent connection times out"
else
  fail "a silent connection times out"
fi

# and NOT by a speed floor. aria2 treats --lowest-speed-limit as a terminal
# abort that --max-tries does not cover, so a 22 GB transfer averaging 14 MiB/s
# was killed outright by one dip to 96 KB/s. Pinned so it does not come back.
# Matched on the assignment, not the file: the comment above it names the flag
# in order to explain why it is gone.
if grep -E '^ARIA_STALL_GUARD=' "$ROOT/lib/core.sh" | grep -q 'lowest-speed-limit'; then
  fail "no speed floor: it aborts a healthy download and does not retry"
else
  pass "no speed floor: it aborts a healthy download and does not retry"
fi

# retries are unlimited: these are model weights, and giving up partway means
# starting a 22 GB transfer again from the beginning.
if grep -qE 'ARIA_STALL_GUARD=.*--max-tries=0' "$ROOT/lib/core.sh"; then
  pass "a dropped connection is retried indefinitely"
else
  fail "a dropped connection is retried indefinitely"
fi

if [ "$fails" -ne 0 ]; then printf '\n%d test(s) failed\n' "$fails"; exit 1; fi
printf '\nall pull tests passed\n'
