# External Engines in /monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `GET /monitor` gains an `externals` section describing Strata, Ollama, LM Studio and ComfyUI, probed on demand over HTTP to 127.0.0.1.

**Architecture:** One small HTTP helper module (`ext_http.py`), two parser/probe modules (`ext_strata.py`, `ext_local.py`) that turn raw engine JSON into entries and are tested from fixtures, and an observer (`externals.py`) that runs the probes in parallel under a shared deadline, caches, serves stale values and backs off from down engines. The gateway calls the observer from `monitor_snapshot`.

**Tech Stack:** Python 3 stdlib only (`urllib.request`, `json`, `concurrent.futures`, `threading`, `re`, `statistics`). Tests with `unittest`.

**Spec:** `docs/superpowers/specs/2026-11-27-external-engines-design.md`. Recon with raw responses: `/Users/fx/Documents/claude-work/telemetry-t3/` (`strata-recon.md`, `engines-recon.md`, `raw_v1_telemetry.json`, `raw_metrics.json`).

## Global Constraints

- Stdlib only. Gateway modules are flat files in `gateway/`, imported by name (`import tools_reader`), as the gateway puts its own dir on `sys.path`.
- English and ASCII only in every tracked file. No em dashes, arrows or emojis.
- Never run `lms`, `ollama`, `comfy` or `strata`. Only HTTP GETs to URLs given by config. Tests never touch a real engine or a real port: they use fake `get` functions or a local `http.server` on port 0.
- The Strata key is sent only as `Authorization: Bearer <key>`, only to the Strata base URL. It never appears in an entry, a `down_reason`, an exception message or a log line. Tests and fixtures use the fake key `test-key-not-real`.
- `down_reason` vocabulary is closed: `None` (connection refused), `"timeout"`, `"http <code>"`, `"bad json"`, `"error"`.
- Shared deadline 1.0 s per snapshot, cache TTL 2.0 s, down-engine retry 10.0 s, per-GET timeout 0.5 s.
- `/monitor` `schema_version` stays 1.
- Never `git add -A`; add named files. No AI attribution in commit messages.
- Strata request rows use exactly fxlla's row keys: `ts, model, engine, ram_mb, ttft_ms, tps, source, prompt_tokens, cached_tokens, output_tokens, reasoning_tokens, prompt_ms, decode_ms, duration_ms, finish, client_app`.

## Review Focus

1. An engine that accepts the TCP connection and never answers: `/monitor` still returns in about 1 s, the entry is stale or `down_reason: "timeout"`, and repeated snapshots do not pile up threads for that engine (Task 4 test `test_hung_probe_no_pileup`).
2. Strata relaunched with a new key while the gateway runs: the remembered key gets a 401, keys are re-read from disk, and the new one works without a gateway restart (Task 2 test `test_key_rotation_rereads_disk`).
3. Ollama's log rotated, truncated, or crossing midnight, or ending in a partial line: counts reset or continue correctly and nothing raises (Task 3 tests `test_log_rotation_resets`, `test_partial_line_waits`, `test_day_rollover`).
4. An engine returning JSON of the wrong shape (a list, nulls, strings for numbers): the entry has nulls, and no exception escapes a parser (Tasks 2 and 3 tests `test_wrong_shapes_never_raise`).
5. Strata `recent` rows with `decode_ms` 0 or missing, rows older than `instance.boot_time`, or missing fields: `tps` is None and old rows are dropped (Task 2 test `test_row_edge_cases`).

---

### Task 1: HTTP helper

**Files:**
- Create: `gateway/ext_http.py`
- Test: `gateway/test_ext_http.py`

**Interfaces:**
- Produces:
  - `class Refused(Exception)`: connection refused or no listener. Carries no message.
  - `class ProbeError(Exception)`: `.reason` is one of `"timeout"`, `"http <code>"`, `"bad json"`, `"error"`; `str(e) == e.reason`.
  - `get_json(url: str, headers: dict | None = None, timeout: float = 0.5) -> object`: GET, return parsed JSON. Maps `ConnectionRefusedError` (also inside `URLError.reason`) to `Refused`, `socket.timeout`/`TimeoutError` to `ProbeError("timeout")`, `HTTPError` to `ProbeError(f"http {code}")`, undecodable body to `ProbeError("bad json")`, anything else to `ProbeError("error")`. Reads at most 4 MB. Never includes the URL or headers in the exception.
  - `GetFn = Callable[[str, dict | None], object]`: the signature every probe receives; `get_json` partially applied with the timeout satisfies it.

- [ ] **Step 1: Write failing tests** using a `http.server.ThreadingHTTPServer` on `127.0.0.1:0` in a thread, with handlers for `/ok` (200 `{"a":1}`), `/401` (401), `/junk` (200 `not json`), `/slow` (sleeps 2 s). Tests: `test_ok` returns `{"a": 1}`; `test_http_error` raises `ProbeError` with `reason == "http 401"`; `test_bad_json` reason `"bad json"`; `test_timeout` with `timeout=0.2` reason `"timeout"` and returns in under 1 s; `test_refused` against a port bound then closed raises `Refused`; `test_headers_sent` the handler records `Authorization` and the test sees `Bearer test-key-not-real`; `test_no_secret_in_error` the `str()` of every raised exception contains neither the URL nor the key.
- [ ] **Step 2: Run** `cd gateway && python3 -m unittest test_ext_http -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `gateway/ext_http.py` as specified.
- [ ] **Step 4: Run** the same command. Expected: all PASS.
- [ ] **Step 5: Commit** `git add gateway/ext_http.py gateway/test_ext_http.py && git commit -m "feat(gateway): HTTP helper for external engine probes"`.

### Task 2: Strata probe

**Files:**
- Create: `gateway/ext_strata.py`, `gateway/fixtures/strata-telemetry.json`, `gateway/fixtures/strata-metrics-live.json`
- Test: `gateway/test_ext_strata.py`

**Interfaces:**
- Consumes: `ext_http.Refused`, `ext_http.ProbeError`, a `get(url, headers) -> object` function.
- Produces:
  - `find_keys(env: Mapping = os.environ, launch_dir: Path = Path.home() / ".local/state/strata/launch") -> list[str]`: `STRATA_API_KEY` first if set and non-empty, then the string `api_key` of each `*.json` in `launch_dir`, newest mtime first; unreadable, invalid or keyless files skipped; duplicates removed, order kept.
  - `row(rec: object, model: str | None, boot_time: float | None) -> dict | None`: the mapping in the spec section "Strata rows". Returns None when `rec` is not a dict, has no numeric `time`, or `time < boot_time`. `tps` rounded to 1 decimal, None unless `decode_ms > 0` and `output_tokens` is a number. Non-numeric fields become None.
  - `parse(health: object, telemetry: object | None, metrics: object | None, auth: str) -> dict`: returns the Strata entry body per the spec's Output shape, keys `auth, model, context, loaded, state, work{active,queued}, live, rates{answer_tok_s,prefill_tok_s}, ram{used_mb,total_mb}, totals{requests,prompt_tokens,output_tokens,since}, recent`. With `telemetry` None, fills `model`, `context` (`max_context`), `loaded` from `health` and everything else None or `[]`. `live` is None unless `metrics["live"]["state"]` is not `"idle"`; then it copies `phase, prompt_read, prompt_total, generated, max_tokens, elapsed_s, tok_s`. `recent` newest first, at most 20. Bytes to MB with `// (1024*1024)`. Never raises.
  - `class StrataProbe(base_url: str, key_source: Callable[[], list[str]] = find_keys)` with `__call__(get) -> dict`: GET `/health` without key (exceptions propagate to the observer: the engine is down). Then try the remembered key, else each key from `key_source()`, on `/v1/telemetry`; a `ProbeError` with reason `"http 401"` moves to the next key and clears the remembered one; first success is remembered. If no key works, `auth` is `"missing"` (no keys) or `"rejected"` (all 401). Fetch `/metrics` only when the telemetry has `work.active > 0`. Other `ProbeError`s on telemetry or metrics leave those inputs None and keep `auth` truthful (`"ok"` if a key worked). Returns `{"up": True, "url": base_url, **parse(...)}`.

- [ ] **Step 1: Make fixtures.** Copy `/Users/fx/Documents/claude-work/telemetry-t3/raw_v1_telemetry.json` to `gateway/fixtures/strata-telemetry.json`, replacing `instance.id`, `instance.launch_attestation`, every `recent[].id` and `activity.events[].request_id` with fixed fake hex strings. Write `strata-metrics-live.json` as `{"live": {"state": "generating", "phase": "decode", "prompt_read": 1000, "prompt_total": 1000, "generated": 50, "max_tokens": 700, "elapsed_s": 2.5, "tok_s": 40.0}}`. Grep both for any 64-char or `sk-` token before committing.
- [ ] **Step 2: Write failing tests:**
  - `test_find_keys_order`: env key first, then files newest first, dedup, a bad-JSON file and a file without `api_key` skipped (temp dir, `os.utime` to set mtimes).
  - `test_row_maps_fields`: fixture's first recent row gives `engine == source == "strata"`, `cached_tokens == reused`, `ttft_ms == round(first_engine_token_s*1000)`, `duration_ms == int(duration_s*1000)`, `ts == int(time)`, `tps == round(output_tokens/(decode_ms/1000), 1)`, `reasoning_tokens is None`, and the key set equals the Global Constraints list.
  - `test_row_edge_cases`: `decode_ms` 0 and missing give `tps None`; `time < boot_time` gives None; a non-dict gives None; a string `output_tokens` gives None for it.
  - `test_parse_fixture`: `model == "qwen3.8-flash-next-iq3_s"`, `context == 262144`, `ram.total_mb == 137438953472 // 1048576`, `totals.requests == 5174`, `live is None`, `recent` length `min(20, rows newer than boot)`.
  - `test_parse_health_only`: telemetry None gives model and context from health, `auth` as passed, `recent == []`.
  - `test_parse_live`: metrics fixture gives `live.generated == 50`.
  - `test_wrong_shapes_never_raise`: `parse` with lists, None, strings and nested nulls in every position returns a dict.
  - `test_probe_key_flow` with a fake `get`: key A gives 401, key B works; second call uses B first (count calls); `auth == "ok"`.
  - `test_key_rotation_rereads_disk`: remembered key now 401s, `key_source` returns a new key, probe succeeds with it in the same call.
  - `test_probe_no_keys` gives `auth == "missing"`, all-401 gives `"rejected"`, and health refused propagates `Refused`.
  - `test_metrics_only_when_active`: `/metrics` requested only if `work.active > 0`.
  - `test_key_never_in_entry`: `"test-key-not-real" not in json.dumps(entry)` for every case above.
- [ ] **Step 3: Run** `cd gateway && python3 -m unittest test_ext_strata -v`. Expected: FAIL.
- [ ] **Step 4: Implement** `gateway/ext_strata.py`.
- [ ] **Step 5: Run** again. Expected: PASS.
- [ ] **Step 6: Commit** the module, test and both fixtures: `feat(gateway): Strata probe for /monitor externals`.

### Task 3: Ollama, LM Studio and ComfyUI probes

**Files:**
- Create: `gateway/ext_local.py`, `gateway/fixtures/ollama-tags.json`, `gateway/fixtures/ollama-ps-loaded_UNVERIFIED.json`, `gateway/fixtures/ollama-server.log`, `gateway/fixtures/lmstudio-v0-models.json`, `gateway/fixtures/comfyui-system-stats_UNVERIFIED.json`, `gateway/fixtures/comfyui-queue_UNVERIFIED.json`
- Test: `gateway/test_ext_local.py`

**Interfaces:**
- Consumes: `ext_http.Refused`, `ext_http.ProbeError`, a `get(url, headers) -> object` function.
- Produces:
  - `GEN_PATHS = ("/api/chat", "/api/generate", "/api/embed", "/api/embeddings", "/v1/chat/completions", "/v1/completions", "/v1/embeddings")`.
  - `parse_gin(line: str) -> tuple[datetime.date, float, str] | None`: for `[GIN] 2026/11/27 - 01:24:17 | 200 |   5.74ms | 1.2.3.4 | POST "/api/chat"` returns `(date(2026,11,27), 5.74, "/api/chat")`. Latency units `ns`, `µs` or `us`, `ms`, `s`, and Go durations like `1m2.5s`, all to ms. The path is stripped of a query string. Non-GIN lines give None.
  - `class OllamaLog(path: Path, today: Callable[[], datetime.date] = datetime.date.today)` with `.today_stats() -> {"requests": int, "median_ms": float | None}`: reads from the last offset; file size smaller than the offset means rotation, so restart at 0 and clear; a trailing line without `\n` is held until completed; a new date clears the counts; only lines with today's date and a path in `GEN_PATHS` count; median of their latencies rounded to 1 decimal; at most 10000 latencies kept. Missing file gives `{"requests": 0, "median_ms": None}`. Never raises.
  - `parse_ollama(ps: object, tags: object, stats: dict | None) -> dict`: `loaded: [{name, size_mb, vram_mb, context, expires_in_s}]` (`expires_in_s` from RFC3339 `expires_at` minus now, int, None if unparseable), `installed: {count, size_gb}` (1 decimal, GiB), `today: stats`.
  - `lmstudio_port(config_path: Path = Path.home() / ".lmstudio/.internal/http-server-config.json") -> int`: its integer `port`, default 1234 on any problem.
  - `parse_lmstudio(models: object) -> dict`: `loaded: [{name: id, context: max_context_length}]` for entries with `state == "loaded"`; also `available: int` count of all entries.
  - `parse_comfyui(stats: object, queue: object) -> dict`: `version` (`system.comfyui_version`), `queue: {running: len(queue_running), pending: len(queue_pending)}`, `vram: {used_mb, total_mb}` summed over `devices` (`vram_total - vram_free`, bytes to MB), `ram: {used_mb, total_mb}` from `system.ram_total`/`ram_free`.
  - Probe factories, each returning a callable `get -> dict` that lets `Refused`/`ProbeError` from its first GET propagate and returns `{"up": True, "url": ..., **parse_x(...)}`:
    - `ollama_probe(base_url: str, log: OllamaLog | None)`: GET `/api/ps` then `/api/tags`; a failure on `/api/tags` alone leaves `installed` None.
    - `lmstudio_probe(base_url: str)`: GET `/api/v0/models`.
    - `comfyui_probe(base_urls: list[str])`: tries each URL's `/system_stats` in order; the first that answers is reported as `url` and gives `/queue`; if all are refused, raise `Refused`; if one gave a `ProbeError` and none answered, raise that.

- [ ] **Step 1: Make fixtures.** `ollama-tags.json` from a live `curl -s -m2 http://127.0.0.1:11434/api/tags` (model names are fine to keep; drop `digest`). `ollama-ps-loaded_UNVERIFIED.json` from Ollama's API docs shape with one model. `ollama-server.log`: about 20 lines mixing `GET "/api/tags"` lines, `POST "/api/chat"` and `POST "/v1/chat/completions"` lines across two dates, units `ms`, `µs`, `s` and `1m2.5s`, a non-GIN line, IPs as `127.0.0.1`. `lmstudio-v0-models.json` with one loaded LLM and one not-loaded embedding model, shaped per `engines-recon.md`. ComfyUI fixtures from field names in `/Users/fx/Documents/github/ComfyUI/server.py` (`/system_stats`, `/queue`).
- [ ] **Step 2: Write failing tests:**
  - `test_parse_gin_units`: each unit in the log fixture converts correctly (`1m2.5s` gives 62500.0, `850µs` gives 0.85).
  - `test_log_counts_today_gen_only`: with `today` fixed to the fixture's later date, counts only that date's generation lines; listing calls excluded; median correct.
  - `test_log_incremental`: append two lines after a read; the next read counts only the new ones on top.
  - `test_log_rotation_resets`: truncate and write a shorter file; counts come only from the new content.
  - `test_partial_line_waits`: a line written without `\n` is not counted until its newline arrives.
  - `test_day_rollover`: changing `today` clears counts.
  - `test_missing_log`: returns zeros, no raise.
  - `test_parse_ollama_loaded`: `size_mb`, `vram_mb`, `context` from the UNVERIFIED fixture; `expires_in_s` positive for a future time.
  - `test_parse_ollama_empty`: `{"models": []}` gives `loaded == []`, `installed.count` from tags.
  - `test_lmstudio_loaded_only` and `test_lmstudio_port_default` (missing and malformed config give 1234).
  - `test_parse_comfyui`: queue lengths and VRAM math.
  - `test_comfyui_fallback_url`: first URL refused, second answers, `url` is the second.
  - `test_wrong_shapes_never_raise`: every `parse_*` with lists, None, strings and null fields returns a dict.
  - `test_probes_propagate_refused`: a fake `get` raising `Refused` makes each probe raise `Refused`.
- [ ] **Step 3: Run** `cd gateway && python3 -m unittest test_ext_local -v`. Expected: FAIL.
- [ ] **Step 4: Implement** `gateway/ext_local.py`.
- [ ] **Step 5: Run** again. Expected: PASS.
- [ ] **Step 6: Commit** the module, test and fixtures: `feat(gateway): Ollama, LM Studio and ComfyUI probes for /monitor`.

### Task 4: Observer and /monitor wiring

**Files:**
- Create: `gateway/externals.py`
- Modify: `gateway/fxlla_gateway.py` (module globals next to `TOOLS = tools_reader.ToolsReader(STORE)` near line 1814; `monitor_snapshot` near line 1817)
- Modify: `.github/workflows/ci.yml` line 74 and `AGENTS.md` Testing list: append `gateway.test_ext_http gateway.test_ext_strata gateway.test_ext_local gateway.test_externals`
- Modify: `CHANGELOG.md` (Unreleased, Added: one entry)
- Test: `gateway/test_externals.py`; extend `gateway/test_gateway_e2e.py` with one assertion

**Interfaces:**
- Consumes: `ext_http.Refused`, `ext_http.ProbeError`, `ext_http.get_json`; `ext_strata.StrataProbe`; `ext_local.ollama_probe`, `ext_local.lmstudio_probe`, `ext_local.comfyui_probe`, `ext_local.OllamaLog`, `ext_local.lmstudio_port`.
- Produces:
  - `class Externals(probes: dict[str, tuple[str, Callable]], get: Callable = None, deadline_s: float = 1.0, ttl_s: float = 2.0, down_retry_s: float = 10.0, clock: Callable[[], float] = time.monotonic)`: `probes` maps engine name to `(url, probe_fn)`; `get` defaults to `ext_http.get_json` with timeout 0.5.
  - `Externals.snapshot() -> dict[str, dict]`: one entry per engine, always every key of `probes`. Rules:
    - Within `ttl_s` of the last snapshot, return it again (one lock; concurrent callers wait for the round in progress instead of starting another).
    - An engine whose last result was down and less than `down_retry_s` old is not probed; its last entry is reused.
    - An engine whose previous probe is still running is not resubmitted.
    - Probes run on a persistent `ThreadPoolExecutor(max_workers=len(probes))`; the snapshot waits until all finish or `deadline_s` passes.
    - Success gives the probe's entry plus `stale: False, age_s: 0.0`. `Refused` gives `{"up": False, "url": url, "down_reason": None}`; `ProbeError` gives `down_reason: e.reason`; any other exception `down_reason: "error"`.
    - Missing the deadline: the last good entry with `stale: True` and `age_s` (1 decimal) since it was produced; with none, `{"up": False, "url": url, "down_reason": "timeout"}`. A late result that arrives afterwards is stored for the next snapshot.
  - `default_externals() -> Externals`: builds the four probes. URLs from `FXLLA_STRATA_URL` (default `http://127.0.0.1:8096`), `FXLLA_OLLAMA_URL` (`http://127.0.0.1:11434`), `FXLLA_LMSTUDIO_URL` (default `http://127.0.0.1:<lmstudio_port()>`), `FXLLA_COMFYUI_URL` (if set, the only URL; else `http://127.0.0.1:8000` then `http://127.0.0.1:8188`; the reported `url` for the entry before any answer is the first). Ollama log `~/.ollama/logs/server.log`. Trailing slashes stripped.
  - `empty_externals() -> dict`: the four names, each `{"up": False, "url": None, "down_reason": "error"}`.
  - Gateway: `EXTERNALS = externals.default_externals()` next to `TOOLS`; in `monitor_snapshot`, `"externals": EXTERNALS.snapshot()` wrapped in `try/except Exception` falling back to `externals.empty_externals()`, exactly like `TOOLS`.

- [ ] **Step 1: Write failing tests in `gateway/test_externals.py`** with fake probe functions and a fake clock:
  - `test_all_engines_present`: snapshot keys equal the probe names, including down ones.
  - `test_refused_and_errors`: `Refused` gives `down_reason None`; `ProbeError("http 500")` gives `"http 500"`; `ValueError` gives `"error"`.
  - `test_cache_ttl`: two snapshots within 2.0 s call each probe once; at 2.1 s again.
  - `test_down_retry`: a refused engine is not re-probed before 10.0 s, and is at 10.1 s.
  - `test_deadline_stale`: a probe that blocks on an `Event` past the deadline (use `deadline_s=0.2`) returns the previous good entry with `stale True` and `age_s` set; with no previous entry, `down_reason "timeout"`; snapshot wall time under 0.5 s.
  - `test_hung_probe_no_pileup`: while one probe is blocked, five more snapshots (clock advanced past TTL each time) never call it again; after releasing it, its result appears in the next snapshot.
  - `test_late_result_used_next`: a probe finishing after the deadline is served (not stale) on the following snapshot.
  - `test_concurrent_callers_one_round`: 8 threads calling `snapshot` at once trigger one probe call per engine.
  - `test_default_urls_env`: env overrides and the ComfyUI two-URL default (patch `os.environ` and `lmstudio_port`).
  - `test_key_never_in_snapshot`: a `StrataProbe` with `key_source` returning `["test-key-not-real"]` and a fake `get` raising `ProbeError("http 401")` for telemetry; `json.dumps(snapshot)` does not contain the key.
- [ ] **Step 2: Extend `gateway/test_gateway_e2e.py`**: in the existing `/monitor` test, assert `"externals" in body` and that its keys are `{"strata", "ollama", "lmstudio", "comfyui"}`. `EXTERNALS` is built at gateway import, so the test replaces the gateway module's `EXTERNALS` with an `Externals` whose four probes target a closed local port (and `OllamaLog` on a missing file), and restores it afterwards. It must never talk to a real engine.
- [ ] **Step 3: Run** `cd gateway && python3 -m unittest test_externals -v`. Expected: FAIL.
- [ ] **Step 4: Implement** `gateway/externals.py` and the gateway wiring.
- [ ] **Step 5: Run** the full CI list from `.github/workflows/ci.yml` line 74 with the four new modules appended. Expected: all PASS (the known order dependence between `gateway.test_gateway_e2e` and `gateway.test_gateway` is pre-existing; run in the CI order).
- [ ] **Step 6: Commit** `gateway/externals.py gateway/test_externals.py gateway/fxlla_gateway.py gateway/test_gateway_e2e.py .github/workflows/ci.yml AGENTS.md CHANGELOG.md`: `feat(gateway): external engines in /monitor`.

### Task 5: Live verification (controller)

- [ ] Restart the gateway from main and read `curl -s http://127.0.0.1:8080/monitor | python3 -m json.tool` `externals`: Strata `up`, `auth: "ok"`, model and totals matching `/v1/telemetry`; Ollama `up` with `installed.count` equal to `/api/tags`; LM Studio and ComfyUI `down` with `down_reason: null`.
- [ ] Time `/monitor` with all four probed: under 1.2 s cold, under 0.1 s cached.
- [ ] `grep` the gateway log and the `/monitor` body for the Strata key read into a shell variable (never printed): zero matches.
- [ ] Ollama loaded shape and LM Studio loaded state: verify only if the user agrees to load a model; otherwise the UNVERIFIED fixtures stay marked so.
