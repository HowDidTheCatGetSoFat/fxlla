#!/usr/bin/env bash
#
# The PATH a backend is launched with. A process started from a GUI app or by
# launchd inherits PATH=/usr/bin:/bin:/usr/sbin:/sbin, not the user's shell
# PATH. The menu bar app starts the gateway that way, and every backend it
# spawned died on `exec mlx_lm.server` (installed in ~/.local/bin) or
# `exec llama-server` (Homebrew) with exit 127. These run `fxlla _backend` the
# way the gateway does, with that inherited environment and nothing else.
#
# Run: bash tests/test_path.sh
# Re-exec under bash if started by another shell (BASH_SOURCE below is bash-only).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -u

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FXLLA="$ROOT/bin/fxlla"

fails=0
pass() { printf 'ok   - %s\n' "$1"; }
fail() { printf 'FAIL - %s\n' "$1"; fails=$((fails + 1)); }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

GUI_PATH=/usr/bin:/bin:/usr/sbin:/sbin
H="$TMP/home"            # a home with ~/.local/bin, where uv installs mlx_lm.server
BARE="$TMP/bare"         # a home without it
STORE="$TMP/store"
mkdir -p "$H/.local/bin" "$BARE" "$STORE/models/plain" "$TMP/mine"

# Each stub says which copy ran and the PATH it was started with.
stub() { # <path> <label>
  # shellcheck disable=SC2016  # $PATH is for the stub to expand when it runs
  printf '#!/bin/sh\necho "%s"\necho "PATH=$PATH"\n' "$2" > "$1"
  chmod +x "$1"
}
stub "$H/.local/bin/mlx_lm.server" home-stub
stub "$TMP/mine/mlx_lm.server" caller-stub

# env -i, so nothing from this shell leaks in: the backend sees only what a GUI
# launch would hand it. /bin/bash exists on macOS and Linux runners alike.
backend() { # <home> <PATH>
  env -i HOME="$1" PATH="$2" FXLLA_STORE="$STORE" /bin/bash "$FXLLA" _backend plain 9999 2>&1
}
path_of() { printf '%s\n' "$1" | sed -n 's/^PATH=//p'; }
count_in() { # <needle> <colon list> ; how many entries equal the needle
  printf '%s\n' "$2" | tr ':' '\n' | grep -cxF -- "$1" || true
}

# --- the GUI case: the engine in ~/.local/bin is found -----------------------
out="$(backend "$H" "$GUI_PATH")"
case "$out" in
  home-stub*) pass "a GUI-inherited PATH still finds the engine in ~/.local/bin" ;;
  *) fail "a GUI-inherited PATH still finds the engine in ~/.local/bin (got: $out)" ;;
esac
p="$(path_of "$out")"
case "$p" in
  "$GUI_PATH":*) pass "the inherited entries keep their place at the front" ;;
  *) fail "the inherited entries keep their place at the front (got: $p)" ;;
esac

# --- appended, not prepended: a copy the caller put on PATH still wins -------
# The legitimate case a fill-in must not override: someone pointing PATH at
# their own build of the engine.
out="$(backend "$H" "$TMP/mine:$GUI_PATH")"
case "$out" in
  caller-stub*) pass "an engine already on the caller's PATH wins over the filled-in one" ;;
  *) fail "an engine already on the caller's PATH wins over the filled-in one (got: $out)" ;;
esac

# --- a directory already present is not added twice --------------------------
out="$(backend "$H" "$H/.local/bin:$GUI_PATH")"
p="$(path_of "$out")"
n="$(count_in "$H/.local/bin" "$p")"
if [ "$n" = 1 ]; then pass "a directory already on PATH appears once"
else fail "a directory already on PATH appears once (seen $n times in: $p)"; fi
case "$p" in
  "$H/.local/bin:$GUI_PATH"*) pass "and the caller's order is unchanged" ;;
  *) fail "and the caller's order is unchanged (got: $p)" ;;
esac

# --- a directory that does not exist is not added -----------------------------
# shellcheck disable=SC2016  # expanded by the child shell, after core.sh ran
p="$(env -i HOME="$BARE" PATH="$GUI_PATH" REPO_ROOT="$ROOT" /bin/bash -c \
      '. "$REPO_ROOT/lib/core.sh"; printf "%s" "$PATH"' 2>&1)"
n="$(count_in "$BARE/.local/bin" "$p")"
if [ "$n" = 0 ]; then pass "a missing ~/.local/bin is not added"
else fail "a missing ~/.local/bin is not added (got: $p)"; fi

# --- an empty PATH does not gain a leading ':' (the current directory) --------
# shellcheck disable=SC2016  # expanded by the child shell, after core.sh ran
# core.sh runs mkdir, which an empty PATH cannot find, so stub it out here.
p="$(env -i HOME="$H" PATH="" REPO_ROOT="$ROOT" /bin/bash -c \
      'mkdir() { :; }; . "$REPO_ROOT/lib/core.sh"; printf "%s" "$PATH"' 2>/dev/null)"
case "$p" in
  :*|*::*|*:) fail "an empty PATH gets no empty element (got: $p)" ;;
  "$H/.local/bin"*) pass "an empty PATH gets no empty element" ;;
  *) fail "an empty PATH gets no empty element (got: $p)" ;;
esac

# --- doctor still judges the shell's PATH, not the filled-in one -------------
# Its "~/.local/bin on PATH" line is advice about the user's shell rc. Read
# after the fill-in it would always say ok, and the advice would be gone.
doc="$(env -i HOME="$H" PATH="$GUI_PATH" FXLLA_STORE="$STORE" /bin/bash "$FXLLA" doctor 2>&1 || true)"
case "$doc" in
  *"$H/.local/bin not on PATH"*) pass "doctor still warns when the caller's PATH lacks ~/.local/bin" ;;
  *) fail "doctor still warns when the caller's PATH lacks ~/.local/bin (got: $(printf '%s' "$doc" | grep -F '.local/bin'))" ;;
esac
doc="$(env -i HOME="$H" PATH="$H/.local/bin:$GUI_PATH" FXLLA_STORE="$STORE" /bin/bash "$FXLLA" doctor 2>&1 || true)"
case "$doc" in
  *"$H/.local/bin on PATH"*) pass "and reports ok when it has it" ;;
  *) fail "and reports ok when it has it (got: $(printf '%s' "$doc" | grep -F '.local/bin'))" ;;
esac

if [ "$fails" -ne 0 ]; then printf '\n%d test(s) failed\n' "$fails"; exit 1; fi
printf '\nall path tests passed\n'
