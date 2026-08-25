# Shared helpers and configuration loading. Sourced by bin/fxlla.
# shellcheck shell=bash
# shellcheck disable=SC2034  # several vars are exported for use by bin/fxlla after sourcing

set -euo pipefail

# --- output ---------------------------------------------------------------
_c()   { printf '\033[%sm' "$1"; }
info() { printf '%s%s%s\n' "$(_c '0;36')" "$*" "$(_c 0)"; }
ok()   { printf '%s%s%s\n' "$(_c '0;32')" "$*" "$(_c 0)"; }
warn() { printf '%s%s%s\n' "$(_c '0;33')" "$*" "$(_c 0)" >&2; }
die()  { printf '%s%s%s\n' "$(_c '0;31')" "$*" "$(_c 0)" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing '$1'. Run: fxlla setup"; }
# True for the usual opt-in values; empty, 0, false, no, off are all false.
is_true() { case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in 1|true|yes|on) return 0;; *) return 1;; esac; }

# --- configuration --------------------------------------------------------
# Precedence: already-exported vars > ~/.config/fxlla/config.env > defaults.
# config.env uses plain assignments, which would otherwise clobber a value the
# user exported in the shell. To honour the precedence, snapshot the config
# vars already exported, source the file, then re-apply the snapshot so the
# environment wins. (This runs under bash: bin/fxlla re-execs if not.)
_user_cfg="${XDG_CONFIG_HOME:-$HOME/.config}/fxlla/config.env"
if [ -f "$_user_cfg" ]; then
  _saved_env="$(export -p | grep -E '^declare -x (FXLLA_[A-Za-z0-9_]*|HF_TOKEN)=' || true)"
  # shellcheck source=/dev/null
  . "$_user_cfg"
  [ -n "$_saved_env" ] && eval "$_saved_env"
  unset _saved_env
fi

# Portable default: weights are large, so an external disk is the common
# choice, but hard-coding one author's volume made `fxlla setup` die on every
# other machine. Set FXLLA_STORE in config.env to point at a disk.
: "${FXLLA_STORE:=${XDG_DATA_HOME:-$HOME/.local/share}/fxlla/store}"
: "${FXLLA_RATE_MBIT:=25}"
: "${FXLLA_HOST:=127.0.0.1}"
: "${FXLLA_PORT:=8080}"
: "${FXLLA_DEFAULT_MODEL:=qwen3-coder}"
: "${FXLLA_SERVER_ARGS:=}"
: "${FXLLA_MTPLX_ARGS:=}"      # extra flags for `mtplx serve` only (strict argparse)
: "${FXLLA_KEEP_WARM:=10}"     # idle minutes before auto-stop (0 = never)
# CEILING on the context window, not the window itself: each gguf model is
# served what its own header says it was trained for, capped by this. This is
# the value that actually wins - config.env is optional and its example is not
# what anyone runs by default - so leaving it at the old 8192 capped a 27B
# trained to 262k right back down to 8192, silently undoing the feature.
: "${FXLLA_CTX:=32768}"        # ceiling for llama-server (gguf), per-model below it
: "${FXLLA_NGL:=999}"          # layers offloaded to GPU for llama-server (gguf)
: "${FXLLA_MEDIA_MODEL:=z-image-turbo}"  # default image model (fxlla media models)
: "${FXLLA_MEDIA_HF_HOME:=}"   # HF cache holding diffusion weights (empty = HF default)
: "${FXLLA_MEDIA_OUT:=}"       # media output dir (empty = <FXLLA_STORE>/media)
: "${FXLLA_VIDEO_BIN:=ltx-2-mlx}"  # path to the ltx-2-mlx binary (fxlla media video)
# FXLLA_VOICE_PYTHON is deliberately NOT defaulted here. media/generate.py
# resolves it (explicit value, then the uv tool venv `fxlla setup --media`
# installs, then python3), and a default assigned here would be forwarded into
# the MCP registration as an explicit choice - beating the venv that actually
# has mlx-audio, with a python3 that does not.
: "${FXLLA_VOICE_MODEL:=YUGOROU/Chatterbox-Multilingual-MLX-4bit}"  # TTS model
: "${FXLLA_VOICE_REF:=}"       # reference voice wav (required for voice; sets timbre)
: "${FXLLA_VOICE_LANG:=en}"    # default speech language code
: "${FXLLA_MEDIA_KEEP_MODELS:=}"  # set to 1 to keep gateway models during media jobs
: "${FXLLA_MEDIA_SKIP_QUALITY:=}"  # set to 1 to accept output that fails the content checks
: "${FXLLA_ASSUME_YES:=}"      # set to 1 to authorize large downloads without asking
: "${FXLLA_CONFIRM_ABOVE_GB:=5}"  # confirm before transferring more than this many GB
: "${FXLLA_CIVITAI_TOKEN:=}"   # Civitai API token for downloading from civitai.com
: "${FXLLA_DOWNLOADER:=aria2}"  # default pull transfer: aria2 (bandwidth-capped) or hf
: "${FXLLA_KB_INDEX:=}"        # set to 1 for the sqlite-vec KNN index (kb search)
: "${FXLLA_KB_PYTHON:=}"       # override interpreter for rag/core.py (see fxlla kb)
: "${FXLLA_EMBED_PORT:=8090}"  # port for the local llama.cpp embedding server
# Eval backends: clear of 8080 (server/gateway), 8090 (embeddings) and the
# gateway's 8100+ backend range.
: "${FXLLA_EVAL_PORT:=8097}"
: "${FXLLA_GRAPH_PYTHON:=}"    # override interpreter for the code graph (needs kuzu)

MODELS_DIR="$FXLLA_STORE/models"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/fxlla"
PID_FILE="$STATE_DIR/server.pid"
WATCH_PID="$STATE_DIR/watch.pid"
ACT_FILE="$STATE_DIR/activity"
LOG_FILE="$STATE_DIR/server.log"
CURRENT_FILE="$STATE_DIR/current"
STATS_FILE="$STATE_DIR/stats.jsonl"
GATEWAY_PID="$STATE_DIR/gateway.pid"
GATEWAY_LOG="$STATE_DIR/gateway.log"
CATALOG="$REPO_ROOT/config/models.conf"
MEDIA_CATALOG="$REPO_ROOT/config/media.conf"
SELF="$REPO_ROOT/bin/fxlla"
BASE_URL="http://$FXLLA_HOST:$FXLLA_PORT/v1"

mkdir -p "$STATE_DIR"

# bytes/s for aria2c from the configured megabits
rate_bytes() { echo $(( FXLLA_RATE_MBIT * 1000000 / 8 )); }

# Survive a connection that dies mid-transfer, without inventing a reason to
# kill one that is merely having a bad minute.
#
# --timeout is the mechanism that matters: a socket delivering NOTHING for this
# long errors, and that error is retryable, so aria2 reconnects and resumes.
#
# There is deliberately no --lowest-speed-limit here, and that is a correction
# rather than an omission. It was set to 128K on the theory that anything slower
# was a hang. A 22 GB pull running at 14 MiB/s average dipped to 96 KB/s for a
# moment and aria2 killed it outright - that abort is terminal, NOT covered by
# --max-tries, so the download simply ended. The rule was added for a stall that
# was never confirmed (a progress reading misinterpreted; the transfer was at the
# full 200 Mbps all along) and the only thing it ever caught was a healthy
# download. A rejection rule needs both names: the failure it catches and the
# legitimate case it must not. This one only ever had the second.
#
# Unlimited retries on purpose. These are model weights: giving up after five
# tries chooses "start the 22 GB again" on the user's behalf.
ARIA_STALL_GUARD="--timeout=60 --connect-timeout=30 --max-tries=0 --retry-wait=5"

# Consent for a large download.
#
# The load-bearing case is a transfer nobody asked for: an agent, a script, an MCP
# call, or the weights a media render fetches on first use. Those get refused with
# instructions, because there is no one to ask. A human at a terminal is offered
# the size and can decline, which is a courtesy - they already typed the command.
# Callers that obtained consent some other way (the app shows a size dialog before
# it calls pull) pass --yes and skip all of this.
#
# Call this BEFORE creating directories or writing marker files, so a refusal
# really does leave nothing behind.
require_download_consent() {
  local gb="$1" what="$2" reply
  is_true "${FXLLA_ASSUME_YES:-}" && return 0
  # Fail closed on a size we cannot read: better to ask about a transfer that
  # turns out to be small than to skip the question on a 60 GB one.
  case "$gb" in
    ''|*[!0-9.]*|*.*.*) gb="" ;;
  esac
  if [ -n "$gb" ] \
     && awk -v g="$gb" -v t="${FXLLA_CONFIRM_ABOVE_GB:-5}" 'BEGIN{exit !(g+0 <= t+0)}'; then
    return 0
  fi
  local shown="${gb:-an unknown number of}"
  # Two separate questions. Is a human driving? That is stdin being a terminal;
  # a script, an agent, or an MCP call has it redirected and must get the refusal
  # immediately rather than wait on a prompt nobody will answer. And where does
  # the question go? /dev/tty, not stdout, which is often redirected - a prompt
  # written there is invisible while the read blocks. /dev/tty can also exist and
  # still fail to open, so probe it by opening it.
  if [ -t 0 ] && (exec </dev/tty >/dev/tty) 2>/dev/null; then
    # Prompt on the terminal itself, not on stdout: stdout is often redirected,
    # and a prompt written there is invisible while the read blocks forever.
    # Ignoring SIGTTIN makes the read fail instead of stopping a background job.
    printf '%sDownload %s GB for %s? [y/N] %s' \
      "$(_c '0;33')" "$shown" "$what" "$(_c 0)" > /dev/tty 2>/dev/null || true
    trap '' TTIN 2>/dev/null || true
    reply=""
    read -r -t 60 reply < /dev/tty 2>/dev/null || reply=""
    trap - TTIN 2>/dev/null || true
    case "$(printf '%s' "$reply" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')" in
      y|yes) return 0 ;;
      "")    die "no answer, so nothing was downloaded" ;;
      *)     die "cancelled" ;;
    esac
  fi
  die "this would transfer about ${shown} GB for ${what}, and nothing here asked a human first.
  Nothing was downloaded. Present the size to the user, and once they agree re-run
  with --yes. FXLLA_CONFIRM_ABOVE_GB raises the threshold if you want fewer stops."
}

# is the store mounted / present?
require_store() {
  [ -d "$FXLLA_STORE" ] || die "Store '$FXLLA_STORE' not found. Is the disk mounted? Set FXLLA_STORE in ~/.config/fxlla/config.env"
  mkdir -p "$MODELS_DIR"
}

# Strip leading and trailing whitespace. This was `xargs` everywhere, which
# trims but is not a trimmer: xargs also PARSES, so one apostrophe in a catalog
# note ("Google's own build") is an unterminated quote and it exits non-zero.
# That killed `fxlla models` at the offending row - every model below it simply
# was not listed, with the only sign a stray "xargs: unterminated quote" after
# the table. Two rows had been invisible for weeks. A trimmer must not have an
# opinion about the text it trims.
trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

# read a catalog field for a given alias:  _catalog_field <alias> <n>
_catalog_field() {
  local q="$1" n="$2" alias line
  while IFS= read -r line; do
    case "$line" in \#*|'') continue;; esac
    alias="$(trim "$(echo "$line" | cut -d'|' -f1)")"
    [ "$alias" = "$q" ] || continue
    trim "$(echo "$line" | cut -d'|' -f"$n")"
    printf '\n'
    return 0
  done < "$CATALOG"
  return 1
}

# alias -> HF repo (or passthrough when it already looks like org/repo)
# read a field for a given alias from the media weight catalog (config/media.conf)
_media_field() {
  local q="$1" n="$2" alias line
  [ -f "$MEDIA_CATALOG" ] || return 1
  while IFS= read -r line; do
    case "$line" in \#*|'') continue;; esac
    alias="$(trim "$(echo "$line" | cut -d'|' -f1)")"
    [ "$alias" = "$q" ] || continue
    trim "$(echo "$line" | cut -d'|' -f"$n")"
    printf '\n'
    return 0
  done < "$MEDIA_CATALOG"
  return 1
}

# The Hugging Face caches the media toolchains read, one per line.
#
# FXLLA_MEDIA_HF_HOME may name SEVERAL, separated by ':' the way PATH is,
# because weights outgrow a disk long before anyone plans for it: one volume
# here holds 1.0 TB of them and is 98% full, so the next 32 GB model has to
# land somewhere else without the ones already fetched turning into "missing".
#
# Each entry is a cache ROOT - the directory that CONTAINS `hub/`, not `hub/`
# itself. Empty means the HF default. media/weights.py applies the same rule,
# so a pull and a render always agree on where weights live.
#
# The FIRST root is where new downloads go; every root is searched when asking
# whether something is already here.
media_hf_roots() {
  local raw="${FXLLA_MEDIA_HF_HOME:-}" parts=() p
  if [ -z "$raw" ]; then printf '%s\n' "$HOME/.cache/huggingface"; return 0; fi
  # IFS=':' only, so a path with spaces in it survives - and there is one:
  # "/Volumes/verga - Data/...". Splitting on whitespace would shred it.
  local IFS_SAVE="$IFS"; IFS=':'
  read -r -a parts <<< "$raw"
  IFS="$IFS_SAVE"
  for p in ${parts[@]+"${parts[@]}"}; do
    [ -n "$p" ] || continue
    printf '%s\n' "$p"
  done
}

# Where a new download goes: the first root named.
media_hf_write_root() {
  local r first=""
  while IFS= read -r r; do
    [ -n "$r" ] || continue
    [ -z "$first" ] && first="$r"
  done <<EOF
$(media_hf_roots)
EOF
  [ -n "$first" ] || first="$HOME/.cache/huggingface"
  printf '%s' "$first"
}

# Kept for callers that just want something to print.
media_hf_home() { printf '%s' "${FXLLA_MEDIA_HF_HOME:-}"; }

# Is a repo present in that cache? A repo directory can exist with only metadata
# (an interrupted fetch, or a listing that never downloaded), so look for actual
# weight-sized content rather than trusting the directory. This also keeps the
# check working across cache layouts: plain downloads fill blobs/, xet-backed ones
# add trees/, and both keep the real bytes under the repo directory.
# -print -quit stops at the first hit without a pipe into head, which under
# `set -o pipefail` would SIGPIPE find and report a false negative.
# Which root holds a repo, or nothing. Prints the root and returns 0 on a hit.
# This is the question a render has to answer too: HF_HOME takes ONE path, so
# knowing "it is cached somewhere" is not enough - the toolchain has to be
# pointed at the root that actually has it.
media_repo_root() {
  local repo="$1" root dir hit
  while IFS= read -r root; do
    [ -n "$root" ] || continue
    # models--<org>--<name>: slashes become double dashes, existing dashes are
    # kept (verified against a real cache, e.g.
    # models--black-forest-labs--FLUX.1-dev).
    dir="$root/hub/models--$(printf '%s' "$repo" | sed 's|/|--|g')"
    [ -d "$dir" ] || continue
    hit="$(find "$dir" -type f -size +1M -print -quit 2>/dev/null || true)"
    [ -n "$hit" ] || continue
    printf '%s' "$root"
    return 0
  done <<EOF
$(media_hf_roots)
EOF
  return 1
}

# Is a repo present in any of the caches? A repo directory can exist with only
# metadata (an interrupted fetch, or a listing that never downloaded), so
# media_repo_root looks for actual weight-sized content rather than trusting
# the directory. That also keeps this working across cache layouts: plain
# downloads fill blobs/, xet-backed ones add trees/, and both keep the real
# bytes under the repo directory.
media_repo_cached() { media_repo_root "$1" >/dev/null; }

# Split a repo spec into repo<TAB>subdir. A Hugging Face repo is exactly
# owner/name; any extra path segments are a directory WITHIN the repo. Several
# publishers ship more than one MLX quant in one repo under mlx-8bit/, mlx-4bit/
# and put a full-precision copy at the root, so pulling the whole thing would
# fetch hundreds of GB - the subdir lets `fxlla pull` take only that subtree.
# subdir is empty for a plain owner/name.
_repo_subdir() {
  local r="$1" sub=""
  case "$r" in
    */*/*) sub="${r#*/*/}"; r="${r%/"$sub"}" ;;
  esac
  printf '%s\t%s\n' "$r" "$sub"
}

resolve_repo() {
  local q="$1"; [ -z "$q" ] && return 1
  local repo; repo="$(_catalog_field "$q" 2 || true)"
  if [ -n "$repo" ]; then echo "$repo"; return 0; fi
  # A trailing slash comes free with a repo path pasted out of a URL. Drop it
  # here, because this value is what gets written to .source and pasted into an
  # HF API path, and both want the repo without it.
  case "$q" in */*) _strip_slashes "$q"; printf '\n'; return 0;; esac
  return 1
}

# alias -> engine (mlx|gguf|omlx|mtplx). Defaults to mlx.
resolve_engine() {
  local e; e="$(_catalog_field "$1" 5 2>/dev/null || true)"
  echo "${e:-}"
}

# Normalize an org/repo the way both sides of a comparison must see it: trailing
# slashes off (a repo path pasted from a URL carries one, and `basename` would
# then quietly hand back a different folder name than the alias), and case
# folded, because HF resolves repo names that way. LC_ALL=C keeps the fold to
# ASCII regardless of the caller's locale.
# Quote a string so it survives being pasted into a shell. Advice printed with
# a bare %s is only correct while no path holds a space, and FXLLA_STORE is a
# user-set path - an external volume or a home directory with a space in the
# account name is enough to turn a printed `mv A B` into a four-argument move
# that does something else entirely.
shq() {
  local s="$1" q="'"
  s="${s//$q/$q\\$q$q}"
  printf '%s%s%s' "$q" "$s" "$q"
}

_strip_slashes() {
  local s="$1"
  while [ "${s%/}" != "$s" ]; do s="${s%/}"; done
  printf '%s' "$s"
}

_norm_repo() {
  # Case is folded only for COMPARING. The repo string that reaches an HF URL
  # keeps its own case, which that API is picky about.
  _strip_slashes "$1" | LC_ALL=C tr '[:upper:]' '[:lower:]'
}

# HF repo -> catalog alias. Empty when the repo is not in the catalog.
_catalog_alias_for_repo() {
  local q line repo
  # No catalog, no answer - and say nothing on stderr about it. Without this
  # guard the `done < "$CATALOG"` below prints a raw bash redirection error on
  # every call, once per model in `fxlla doctor`. _media_field guards its
  # catalog the same way.
  [ -f "$CATALOG" ] || return 1
  q="$(_norm_repo "$1")"
  [ -n "$q" ] || return 1
  while IFS= read -r line; do
    case "$line" in \#*|'') continue;; esac
    repo="$(trim "$(echo "$line" | cut -d'|' -f2)")"
    [ "$(_norm_repo "$repo")" = "$q" ] || continue
    trim "$(echo "$line" | cut -d'|' -f1)"
    printf '\n'
    return 0
  done < "$CATALOG"
  return 1
}

# Local folder name for an alias/repo. A repo the catalog knows resolves to its
# alias, so `pull qwen3.5-9b` and `pull mradermacher/Qwen3.5-9B-GGUF` name one
# directory instead of downloading the same weights twice under two names - and
# so the alias in the catalog can still find what the repo spelling fetched.
local_name() {
  local a
  case "$1" in
    */*) ;;
    *)   printf '%s\n' "$1"; return 0 ;;
  esac
  a="$(_catalog_alias_for_repo "$1" || true)"
  if [ -n "$a" ]; then printf '%s\n' "$a"; else basename "$1"; fi
}

# Model directories holding a catalog model under a name the catalog cannot
# find. Pulling by org/repo used to name the directory after the repo, so the
# same weights had two identities and the alias reported them missing. The
# directory name is derived; .source is what was actually fetched, so ask it.
# Prints "dir<TAB>alias" per stray directory.
stray_model_dirs() {
  local d name src alias
  [ -d "$MODELS_DIR" ] || return 0
  for d in "$MODELS_DIR"/*/; do
    [ -d "$d" ] || continue
    [ -f "$d/.source" ] || continue
    name="$(basename "$d")"
    src="$(trim "$(cat "$d/.source" 2>/dev/null || true)")"
    [ -n "$src" ] || continue
    alias="$(_catalog_alias_for_repo "$src" || true)"
    [ -n "$alias" ] || continue
    [ "$alias" = "$name" ] && continue
    printf '%s\t%s\n' "$name" "$alias"
  done
}

# list files in an HF repo: prints "size<TAB>path" per file
_hf_list() {
  HF_REPO="$1" HF_TOKEN="${HF_TOKEN:-}" python3 - <<'PY'
import os, json, sys, urllib.request
repo = os.environ["HF_REPO"]; tok = os.environ.get("HF_TOKEN", "")
url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
while url:
    req = urllib.request.Request(url)
    if tok: req.add_header("Authorization", f"Bearer {tok}")
    try:
        r = urllib.request.urlopen(req)
    except Exception as e:
        sys.stderr.write(str(e) + "\n"); sys.exit(2)
    for e in json.load(r):
        if e.get("type") != "file": continue
        p = e["path"]
        if p == ".gitattributes" or p.startswith("."): continue
        print(f'{e.get("size",0)}\t{p}')
    link = r.headers.get("Link", ""); nxt = ""
    for part in link.split(","):
        if 'rel="next"' in part: nxt = part[part.find("<")+1:part.find(">")]
    url = nxt
PY
}

# A pid file is a claim, not a fact. Pids get reused, and `kill -0` answers for
# whoever holds the number now - pid 1 answers it all day, and this state
# directory has held a pid file for a week pointing at a process that was long
# gone. So confirm the process is still the one we started: a stale file that
# happens to name a live pid otherwise makes `fxlla on` and `fxlla serve` refuse
# to start, pointing at a gateway that is not there.
# -ww asks for the untruncated command line; the gateway's marker is at the end
# of a long interpreter path, so a truncated one would never match.
_pid_is() {
  local p="$1" pat="$2"
  [ -n "$p" ] || return 1
  kill -0 "$p" 2>/dev/null || return 1
  ps -ww -o command= -p "$p" 2>/dev/null | grep -qE -- "$pat"
}

server_pid()   { [ -f "$PID_FILE" ] && cat "$PID_FILE" 2>/dev/null || true; }
# Any of the four engines can be behind this pid: gguf models run llama-server,
# MLX ones mlx_lm.server, OMLX ones `omlx serve`, MTPLX ones a python running
# `mtplx.server.openai`. omlx calls setproctitle, so it shows as `omlx-server`
# (verified against a live one) rather than its argv - but the title is only set
# if setproctitle is present, so match the argv `omlx serve` too. MTPLX's console
# script execs into `python -m mtplx.server.openai` (verified), so match that
# module path. None of these substrings appears in a mere directory path, which
# is the false match to avoid.
server_alive() { _pid_is "$(server_pid)" 'llama-server|mlx_lm\.server|omlx-server|omlx serve|mtplx\.server\.openai'; }

gateway_pid()   { [ -f "$GATEWAY_PID" ] && cat "$GATEWAY_PID" 2>/dev/null || true; }
gateway_alive() { _pid_is "$(gateway_pid)" 'fxlla_gateway\.py'; }
