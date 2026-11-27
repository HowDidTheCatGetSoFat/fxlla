# Gateway Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The fxlla gateway records everything a live monitor needs: a full
record per request, live state, totals, machine stats and events. It serves
all of it from `GET /monitor`.

**Architecture:**
- **Parsing.** Engine usage and timings are parsed by pure functions in
  `gateway/metrics.py`. That module already owns stream parsing.
- **State.** In-memory state (rings, totals, rate series, daily aggregates,
  machine sampling) lives in a new standard-library module,
  `gateway/telemetry.py`, owned by one `Telemetry` object.
- **Wiring.** `gateway/fxlla_gateway.py` wires both into the proxy path and
  the Manager, and adds the endpoint.

**Tech Stack:** Python 3 standard library only, and `unittest`.

**Spec:** `docs/superpowers/specs/2026-11-26-gateway-telemetry-design.md`.
Read it in full. The sections "Verified on 2026-11-26" and "Added after the
monitor mockup" override the earlier sections where they differ.

## Global Constraints

- **Language and dependencies.**
  - Standard library only in `gateway/`.
  - English and ASCII only in every tracked file, fixtures included.
  - No em dashes, en dashes, arrows, or emojis (AGENTS.md).
- **Telemetry never changes a response.**
  - Every capture path is wrapped so any exception is logged with
    `log("metrics: ...")` and swallowed.
  - Bytes sent to the client are identical to today's, except for the
    usage-only chunk the gateway itself caused.
- **No invented numbers.** A field the engine did not report is `None`
  (JSON null). The only estimate allowed is the delta count, stored as
  `output_tokens` with `output_estimated: true`.
- **`stats.jsonl` is additive only.**
  - Keep `ts`, `model`, `engine`, `ram_mb`, `ttft_ms`, `tps`, and
    `source: "gateway"` with today's computation.
  - The 5000-line cap stays.
- **Never record prompt or answer text, headers other than User-Agent, or
  keys.**
- **The injection allowlist is exactly `{"mlx", "gguf", "omlx"}`.** Never
  inject for `mtplx`.
- **Tests must pass on the Linux CI runner.** Nothing in a test may need
  `ioreg`, `vm_stat`, `sysctl`, or a real engine. Machine parsers are tested
  on captured text.
- **Commits.** Use `git add` with explicit paths, never `-A`. No AI
  attribution lines.
- **Suites.**
  - `bash tests/run.sh`
  - `FXLLA_STORE=/tmp python3 -m unittest rag.test_rag graph.test_graph gateway.test_gateway gateway.test_metrics gateway.test_gateway_e2e media.test_media evals.test_evals agent.test_loop gateway.test_ggufmeta gateway.test_telemetry`
  - `media.test_media TestQualityAudio.test_unparseable_format_gets_no_verdict`
    fails on Python 3.15 before this work. It is out of scope; do not count
    it.

## Review Focus

- **The client already asks for usage.** opencode sends
  `stream_options.include_usage: true`. Its stream must reach it
  byte-identical, with the usage chunk intact. That test is in Task 5.
- **An SSE event split across reads.** An event can be split across the
  1024-byte upstream reads, or end in `\r\n\r\n`. The usage stripper must
  still remove exactly that one event and pass everything else untouched.
  That test is in Task 2.
- **The upstream dies mid-stream.** A truncated stream, a stream with no
  `[DONE]`, or an HTTP 500 is still recorded with `finish: "error"`. No
  exception reaches the client path. That test is in Task 5.
- **Many concurrent requests.** Totals and rings updated from 16 threads at
  once lose no counts. That test is in Task 3.
- **A machine command that is missing or hangs.** On Linux, or when `ioreg`
  hangs, the machine fields are `None` within the 2-second budget, and
  `/monitor` still answers. Those tests are in Tasks 4 and 7.

---

### Task 1: Parse what each engine reports

**Files:**
- Modify: `gateway/metrics.py`
- Test: `gateway/test_metrics.py`

**Interfaces:**
- Produces:
  - `engine_report(usage: dict | None, timings: dict | None) -> dict`.
    It returns exactly these keys:
    - `prompt_tokens`, `cached_tokens`, `output_tokens`
    - `reasoning_tokens`
    - `prompt_ms`, `decode_ms`
    - `draft_tokens`, `draft_accepted`
  - Every value is an int or None. Millisecond values are rounded to int.

- [ ] **Step 1: Write the failing tests** in a new class `TestEngineReport`.
  The values come from real responses of the installed engines:

```python
def test_llama_server_stream(self):
    r = metrics.engine_report(
        {"completion_tokens": 200, "prompt_tokens": 12, "total_tokens": 212,
         "prompt_tokens_details": {"cached_tokens": 0}},
        {"cache_n": 0, "prompt_n": 12, "prompt_ms": 45.795,
         "predicted_n": 200, "predicted_ms": 1860.769})
    self.assertEqual((r["prompt_tokens"], r["cached_tokens"], r["output_tokens"]), (12, 0, 200))
    self.assertEqual((r["prompt_ms"], r["decode_ms"]), (46, 1861))

def test_llama_server_cache_counts_from_timings(self):
    r = metrics.engine_report(
        {"completion_tokens": 161, "prompt_tokens": 12,
         "prompt_tokens_details": {"cached_tokens": 8}},
        {"cache_n": 8, "prompt_n": 4, "prompt_ms": 16.361,
         "predicted_n": 161, "predicted_ms": 1482.787})
    self.assertEqual((r["prompt_tokens"], r["cached_tokens"]), (12, 8))

def test_mlx_has_tokens_but_no_engine_timings(self):
    r = metrics.engine_report(
        {"prompt_tokens": 31, "completion_tokens": 2, "total_tokens": 33,
         "prompt_tokens_details": {"cached_tokens": 30}}, None)
    self.assertEqual((r["prompt_tokens"], r["cached_tokens"], r["output_tokens"]), (31, 30, 2))
    self.assertIsNone(r["prompt_ms"]); self.assertIsNone(r["decode_ms"])

def test_omlx_timings_live_inside_usage_in_seconds(self):
    r = metrics.engine_report(
        {"prompt_tokens": 22, "completion_tokens": 300,
         "prompt_tokens_details": {"cached_tokens": 0},
         "completion_tokens_details": {"reasoning_tokens": 300},
         "prompt_eval_duration": 4.05, "generation_duration": 7.03}, None)
    self.assertEqual((r["prompt_ms"], r["decode_ms"], r["reasoning_tokens"]), (4050, 7030, 300))

def test_mtplx_reports_draft_acceptance(self):
    r = metrics.engine_report(
        {"prompt_tokens": 22, "completion_tokens": 108,
         "completion_tokens_details": {"reasoning_tokens": 24}},
        {"prompt_n": 22, "predicted_n": 108, "prompt_ms": 631.235,
         "predicted_ms": 2664.547, "draft_n": 84, "draft_n_accepted": 79})
    self.assertEqual((r["draft_tokens"], r["draft_accepted"]), (84, 79))
    self.assertEqual((r["prompt_ms"], r["decode_ms"]), (631, 2665))

def test_nothing_reported_is_all_none(self):
    self.assertTrue(all(v is None for v in metrics.engine_report(None, None).values()))

def test_garbage_types_are_none_not_an_exception(self):
    r = metrics.engine_report({"prompt_tokens": "12", "completion_tokens": None}, {"prompt_ms": "x"})
    self.assertIsNone(r["prompt_tokens"]); self.assertIsNone(r["prompt_ms"])
```

- [ ] **Step 2: Run the tests and confirm they fail.**
  Run: `python3 -m unittest gateway.test_metrics.TestEngineReport -v`
  Expected: FAIL, because `engine_report` is not defined.

- [ ] **Step 3: Implement `engine_report` in `gateway/metrics.py`.**
  - **Precedence.**
    - For the token counts, `usage` wins.
    - `cached_tokens` comes from `usage.prompt_tokens_details.cached_tokens`,
      and falls back to `timings.cache_n`.
    - `prompt_ms` and `decode_ms` come from `timings.prompt_ms` and
      `timings.predicted_ms`. When those are absent, they come from
      `usage.prompt_eval_duration` and `usage.generation_duration`, times
      1000.
  - **Value types.** Accept int, or float where a duration is expected.
    Anything else becomes None. bool is not a number here.

- [ ] **Step 4: Run the tests and confirm they pass.**
  Same command. Expected: OK.

- [ ] **Step 5: Commit.**
  `git add gateway/metrics.py gateway/test_metrics.py && git commit -m "feat(gateway): parse token, cache and timing reports from each engine"`

---

### Task 2: Keep what the stream says, and strip only the usage chunk we caused

**Files:**
- Modify: `gateway/metrics.py`
- Create:
  - `gateway/fixtures/omlx-stream.sse`
  - `gateway/fixtures/mtplx-stream.sse`
- Test: `gateway/test_metrics.py`

**Interfaces:**
- Consumes: `engine_report` from Task 1.
- Produces:
  - `StreamMetrics` gains attributes, filled in by `_line`. The existing
    attributes and `result()` are unchanged.
    - `usage: dict | None` (the last `usage` object seen)
    - `timings: dict | None` (the last `timings` object seen)
    - `finish: str | None` (the last non-empty `choices[i].finish_reason`)
    - `report() -> dict`, equal to `engine_report(self.usage, self.timings)`
  - `class UsageChunkStripper` with two methods.
    - `consume(data: bytes) -> bytes`
    - `finish() -> bytes`
    - It removes every SSE event whose JSON has `choices == []` and a
      `usage` key. All other bytes pass through unchanged and in order.
      The caller constructs it only when the gateway injected the option.

- [ ] **Step 1: Create the fixtures.**
  - Source:
    - `~/Documents/claude-work/telemetry-verify/omlx-coder-stream.sse`
    - `~/Documents/claude-work/telemetry-verify/mtplx-qwen38-stream.sse`
  - Keep the first 3 `data:` events, then the last 3: finish, usage, and
    `[DONE]`.
  - Replace any non-ASCII text inside `content`, `reasoning_content`, or
    `reasoning` with `x`.
  - Check: `LC_ALL=C grep -c '[^[:print:][:space:]]' gateway/fixtures/*.sse`
    prints 0 for each file.

- [ ] **Step 2: Write the failing tests.**

```python
def test_stream_keeps_usage_timings_and_finish_mtplx(self):
    sm = metrics.StreamMetrics(0.0)
    sm.feed(open(FIX + "/mtplx-stream.sse", "rb").read())
    self.assertEqual(sm.finish, "stop")
    self.assertEqual(sm.report()["draft_accepted"], 79)

def test_stream_keeps_omlx_usage(self):
    sm = metrics.StreamMetrics(0.0)
    sm.feed(open(FIX + "/omlx-stream.sse", "rb").read())
    self.assertEqual(sm.report()["output_tokens"], 300)
    self.assertEqual(sm.finish, "length")

USAGE_ONLY = b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
BODY = (b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')

def test_stripper_removes_only_the_usage_only_event(self):
    s = metrics.UsageChunkStripper()
    out = s.consume(BODY + USAGE_ONLY + b"data: [DONE]\n\n") + s.finish()
    self.assertEqual(out, BODY + b"data: [DONE]\n\n")

def test_stripper_survives_any_split_point(self):
    whole = BODY + USAGE_ONLY + b"data: [DONE]\n\n"
    for cut in range(1, len(whole)):
        s = metrics.UsageChunkStripper()
        out = s.consume(whole[:cut]) + s.consume(whole[cut:]) + s.finish()
        self.assertEqual(out, BODY + b"data: [DONE]\n\n", cut)

def test_stripper_handles_crlf_events(self):
    s = metrics.UsageChunkStripper()
    crlf = USAGE_ONLY.replace(b"\n\n", b"\r\n\r\n")
    self.assertEqual(s.consume(BODY + crlf) + s.finish(), BODY)

def test_stripper_forwards_usage_that_rides_with_choices(self):
    ev = b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":2}}\n\n'
    s = metrics.UsageChunkStripper()
    self.assertEqual(s.consume(ev) + s.finish(), ev)

def test_stripper_passes_non_json_and_comments_through(self):
    raw = b": keepalive\n\ndata: not json\n\n"
    s = metrics.UsageChunkStripper()
    self.assertEqual(s.consume(raw) + s.finish(), raw)
```

- [ ] **Step 3: Run the tests and confirm they fail.**
  Run: `python3 -m unittest gateway.test_metrics -v`
  Expected: the new tests FAIL, and the existing tests still pass.

- [ ] **Step 4: Implement.**
  - **`StreamMetrics._line`.** Additionally store `obj["usage"]` and
    `obj["timings"]` when they are dicts, and the finish_reason.
  - **`UsageChunkStripper`.**
    - Buffer the bytes.
    - Emit each complete event (ending `\n\n` or `\r\n\r\n`) unless it
      matches.
    - `finish()` returns any trailing partial event unchanged.
    - A complete event that is not matched is emitted at once. Only an
      incomplete tail is held, so streaming stays incremental.

- [ ] **Step 5: Run the tests and confirm they pass.**
  Same command. Expected: OK.

- [ ] **Step 6: Commit.**
  `git add gateway/metrics.py gateway/test_metrics.py gateway/fixtures/omlx-stream.sse gateway/fixtures/mtplx-stream.sse && git commit -m "feat(gateway): keep usage, timings and finish from streams; strip an injected usage chunk"`

---

### Task 3: The telemetry store

**Files:**
- Create:
  - `gateway/telemetry.py`
  - `gateway/test_telemetry.py`

**Interfaces:**
- Produces `class Telemetry`, one per gateway and thread-safe through one
  internal `threading.Lock`:
  - **Constructor.**
    `__init__(self, recent_max: int = 200, events_max: int = 200, window_s: int = 300, clock=time.time)`
  - **`new_id() -> str`.** An 8-hex-character id, unique within the run.
  - **`record(rec: dict) -> None`.**
    - Appends to the recent ring.
    - Updates the totals: `requests`, `prompt_tokens`, `cached_tokens`,
      `output_tokens`, `errors` (finish `error`), and `client_gone`.
      Counts treat None as 0.
    - Adds rate points for the sparkline series:
      - `answer_tps` from `rec["tps"]`
      - `prompt_tps` from `prompt_tokens - cached_tokens` over `prompt_ms`,
        only when both are present and `prompt_ms > 0`
      - `ttft_ms`
    - Updates the day aggregates for the local date of `rec["ts"]`:
      `cache_saved_tokens` and the per-`client_app` counts. They reset
      when the date changes.
  - **`event(kind: str, alias: str | None, **fields) -> None`.**
    - Appends `{"ts", "kind", "alias", **fields}`.
    - `kind` is one of: `load_started`, `loaded`, `load_failed`, `evicted`,
      `idle_unloaded`, `admin_unloaded`, `exited`.
  - **`snapshot() -> dict`.** It has these keys:
    - `totals`: the counters above plus `since`, which is the boot epoch
    - `recent`: newest first
    - `events`: newest first
    - `series`: `answer_tps`, `prompt_tps`, and `ttft_ms`, each a list of
      `[ts, value]` within `window_s`
    - `ttft_p95_ms`: an int or None
    - `day`: `{"date", "cache_saved_tokens", "clients": {app: count}}`
  - **`client_tag(user_agent: str | None) -> str`**, a module-level
    function. It returns `opencode`, `hermes`, `curl`, `python`, or
    `other`, by case-insensitive substring: `opencode`, `hermes`, `curl/`,
    `python`.

- [ ] **Step 1: Write the failing tests** in `TestTelemetry`. They use an
  injected fake clock.

```python
def test_totals_and_ring(self):
    t = Telemetry(recent_max=2, clock=lambda: 1000.0)
    for n in range(3):
        t.record({"id": str(n), "ts": 1000, "tps": 40.0, "ttft_ms": 500,
                  "prompt_tokens": 100, "cached_tokens": 90, "output_tokens": 10,
                  "prompt_ms": 50, "finish": "stop", "client_app": "opencode"})
    s = t.snapshot()
    self.assertEqual([r["id"] for r in s["recent"]], ["2", "1"])
    self.assertEqual((s["totals"]["requests"], s["totals"]["output_tokens"]), (3, 30))
    self.assertEqual(s["day"]["cache_saved_tokens"], 270)
    self.assertEqual(s["day"]["clients"], {"opencode": 3})

def test_errors_and_client_gone_are_counted(self):
    t = Telemetry(clock=lambda: 1000.0)
    t.record({"ts": 1000, "finish": "error"}); t.record({"ts": 1000, "finish": "client_gone"})
    self.assertEqual((t.snapshot()["totals"]["errors"], t.snapshot()["totals"]["client_gone"]), (1, 1))

def test_series_drop_points_outside_the_window(self):
    now = [1000.0]
    t = Telemetry(window_s=300, clock=lambda: now[0])
    t.record({"ts": 1000, "tps": 10.0}); now[0] = 1400.0
    t.record({"ts": 1400, "tps": 20.0})
    self.assertEqual([v for _, v in t.snapshot()["series"]["answer_tps"]], [20.0])

def test_ttft_p95(self):
    t = Telemetry(clock=lambda: 1000.0)
    for ms in range(1, 101):
        t.record({"ts": 1000, "ttft_ms": ms})
    self.assertEqual(t.snapshot()["ttft_p95_ms"], 95)

def test_day_rolls_over(self):
    now = [1795700000.0]
    t = Telemetry(clock=lambda: now[0])
    t.record({"ts": int(now[0]), "client_app": "curl", "prompt_tokens": 10, "cached_tokens": 5})
    now[0] += 86400
    t.record({"ts": int(now[0]), "client_app": "hermes"})
    self.assertEqual(t.snapshot()["day"]["clients"], {"hermes": 1})

def test_concurrent_records_lose_nothing(self):
    t = Telemetry()
    ths = [threading.Thread(target=lambda: [t.record({"ts": 1, "output_tokens": 1}) for _ in range(500)]) for _ in range(16)]
    [h.start() for h in ths]; [h.join() for h in ths]
    self.assertEqual(t.snapshot()["totals"]["output_tokens"], 8000)

def test_events_ring_newest_first(self):
    t = Telemetry(events_max=2, clock=lambda: 5.0)
    for k in ("load_started", "loaded", "evicted"):
        t.event(k, "tiny")
    self.assertEqual([e["kind"] for e in t.snapshot()["events"]], ["evicted", "loaded"])

def test_client_tag(self):
    self.assertEqual(client_tag("opencode/latest/2.4.1/cli"), "opencode")
    self.assertEqual(client_tag("curl/8.7.1"), "curl")
    self.assertEqual(client_tag("Python-urllib/3.15"), "python")
    self.assertEqual(client_tag(None), "other")
```

- [ ] **Step 2: Run the tests and confirm they fail.**
  Run: `python3 -m unittest gateway.test_telemetry -v`
  Expected: an ImportError, because there is no `gateway/telemetry.py`.

- [ ] **Step 3: Implement `gateway/telemetry.py`.**
  - Use `collections.deque(maxlen=...)` for the rings and for each series.
  - Prune by time in `snapshot()` and in `record()`.
  - p95 is nearest-rank over the TTFT points in the window.
  - The test file adds the gateway directory to `sys.path` the same way
    `gateway/test_metrics.py` does.

- [ ] **Step 4: Run the tests and confirm they pass.**
  Same command. Expected: OK.

- [ ] **Step 5: Commit.**
  `git add gateway/telemetry.py gateway/test_telemetry.py && git commit -m "feat(gateway): an in-memory telemetry store for requests, events and rates"`

---

### Task 4: Machine stats

**Files:**
- Modify:
  - `gateway/telemetry.py`
  - `gateway/test_telemetry.py`

**Interfaces:**
- Produces:
  - **Pure parsers.**
    - `parse_ioreg(text: str) -> dict`, with keys `gpu_util_pct` and
      `gpu_inuse_bytes`.
    - `parse_vm_stat(text: str, page_size: int, total_bytes: int) -> dict`,
      with key `ram_used_bytes`. Used is `total_bytes` minus free,
      inactive, and speculative pages.
    - `parse_swapusage(text: str) -> dict`, with keys `swap_used_bytes`
      and `swap_total_bytes`.
  - **`class MachineSampler(store_dir: str, ttl_s: float = 1.0, run=subprocess.run)`.**
    - `sample() -> dict` returns these keys:
      - `ram_total_bytes`, `ram_used_bytes`
      - `swap_used_bytes`, `swap_total_bytes`
      - `pressure`: `normal`, `warn`, `critical`, or None, from
        `sysctl -n kern.memorystatus_vm_pressure_level`, where 1, 2 and 4
        map to normal, warn and critical
      - `gpu_util_pct`, `gpu_inuse_bytes`
      - `disk_free_bytes` and `disk_total_bytes`, from `os.statvfs(store_dir)`
    - The result is cached for `ttl_s`.
    - Each command runs with `timeout=2`. Any failure, absence, or timeout
      makes that command's fields None.

- [ ] **Step 1: Write the failing tests.** They use captured text from this
  Mac, and a fake `run`.

```python
IOREG = ('"PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=86238248960,'
         '"Tiler Utilization %"=6,"Renderer Utilization %"=11,"Device Utilization %"=13,"In use system memory"=2423013376}')

def test_parse_ioreg(self):
    self.assertEqual(parse_ioreg(IOREG), {"gpu_util_pct": 13, "gpu_inuse_bytes": 2423013376})

def test_parse_swapusage(self):
    r = parse_swapusage("total = 2048.00M  used = 869.00M  free = 1179.00M  (encrypted)")
    self.assertEqual((r["swap_total_bytes"], r["swap_used_bytes"]), (2048 * 2**20, 869 * 2**20))

def test_parse_vm_stat(self):
    text = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free:                               100.\nPages inactive:                           50.\n"
            "Pages speculative:                        10.\n")
    self.assertEqual(parse_vm_stat(text, 16384, total_bytes=1000 * 16384)["ram_used_bytes"], 840 * 16384)

def test_missing_commands_give_none_not_errors(self):
    def boom(*a, **k): raise FileNotFoundError("ioreg")
    s = MachineSampler(tempfile.gettempdir(), run=boom).sample()
    self.assertIsNone(s["gpu_util_pct"]); self.assertIsNotNone(s["disk_free_bytes"])

def test_a_hanging_command_is_cut_at_its_timeout(self):
    def hang(*a, **k): raise subprocess.TimeoutExpired(a[0], k.get("timeout"))
    self.assertIsNone(MachineSampler(tempfile.gettempdir(), run=hang).sample()["pressure"])

def test_sample_is_cached_for_ttl(self):
    calls = []
    def run(*a, **k): calls.append(a); raise FileNotFoundError()
    m = MachineSampler(tempfile.gettempdir(), ttl_s=60, run=run)
    m.sample(); n = len(calls); m.sample()
    self.assertEqual(len(calls), n)
```

- [ ] **Step 2: Run the tests and confirm they fail.**
  Run: `python3 -m unittest gateway.test_telemetry -v`
  Expected: the new tests FAIL with an ImportError for the names.

- [ ] **Step 3: Implement.**
  - **Commands.**
    - `ioreg -r -c IOAccelerator -d 1 -w0`
    - `vm_stat`
    - `sysctl -n hw.memsize`
    - `sysctl -n vm.swapusage`
    - `sysctl -n kern.memorystatus_vm_pressure_level`
  - **Parsing.** Use regex matches. A missing key gives None.

- [ ] **Step 4: Run the tests and confirm they pass.**
  Same command. Expected: OK.
  Also run one live check, for this Mac only:
  `python3 -c "import sys; sys.path.insert(0,'gateway'); import telemetry; print(telemetry.MachineSampler('/Volumes/1TB-WD750-1/llm').sample())"`
  It prints a dict with no None for RAM, GPU, swap, or disk.

- [ ] **Step 5: Commit.**
  `git add gateway/telemetry.py gateway/test_telemetry.py && git commit -m "feat(gateway): sample RAM, swap, pressure, GPU and disk without sudo"`

---

### Task 5: Record every request in full

**Files:**
- Modify:
  - `gateway/fxlla_gateway.py`: `Handler.do_POST`, `_proxy`, and `_record`
  - `gateway/metrics.py`: `build_sample`
- Test: `gateway/test_gateway_e2e.py`

**Interfaces:**
- Consumes:
  - `engine_report`, `StreamMetrics.{usage,timings,finish,report}` and
    `UsageChunkStripper` from Tasks 1 and 2.
  - `Telemetry.record`, `Telemetry.new_id` and `client_tag` from Task 3.
- Produces:
  - A module global `TELEMETRY = telemetry.Telemetry()` in
    `fxlla_gateway.py`.
  - `build_sample(ts, model, engine, ram_mb, ttft_ms, tps, **extra)`. It
    merges `extra` after the existing keys, and the existing keys cannot be
    overridden.
- **The full record, written to `stats.jsonl` and passed to
  `TELEMETRY.record`.** It holds the existing keys plus:
  - `id`, `prompt_tokens`, `cached_tokens`, `output_tokens`,
    `output_estimated`
  - `reasoning_tokens`, `prompt_ms`, `decode_ms`, `duration_ms`
  - `draft_tokens`, `draft_accepted`
  - `finish` (`stop`, `length`, `tool_calls`, `client_gone`, or `error`)
  - `stream`, `client_addr`, `client_app`

- [ ] **Step 1: Write the failing e2e tests.**
  - **Extend the fake `_Backend`.**
    - It records the last request body in `_Backend.last_body`.
    - When `stream_options.include_usage` is set, it appends `USAGE_ONLY`
      (from Task 2) before `[DONE]`.
    - A class attribute `mode` selects one of three behaviors: `"ok"`,
      `"http500"`, or `"truncate"`. `"truncate"` sends half of the SSE,
      then closes.
  - **Make the engine selectable.** The existing setUp registers the fake
    backend as `"gguf"`. Add a helper `_register(engine)` so a test can
    register it as `"mtplx"`.
  - **Tests.**
    - `test_injects_usage_and_hides_it_from_a_client_that_did_not_ask`:
      - The upstream body has `stream_options.include_usage` True.
      - The client stream contains no `"usage"`.
      - The last `stats.jsonl` row has `output_tokens == 2`,
        `output_estimated is False`, `finish == "stop"`, `stream is True`,
        and `client_app == "python"` (urllib's User-Agent).
    - `test_client_that_asked_gets_its_usage_untouched`:
      - The client sends `stream_options.include_usage` True.
      - The bytes it receives equal exactly what the fake backend sent.
    - `test_mtplx_is_never_injected`:
      - The fake backend is registered as `mtplx`.
      - `"stream_options" not in _Backend.last_body`.
      - The row has `output_estimated` True, because the fake sends no
        usage unasked.
    - `test_upstream_500_is_recorded_as_error`:
      - With mode http500, the client gets 500.
      - The row has `finish == "error"` and `output_tokens == 0`.
    - `test_truncated_stream_is_recorded_as_error`:
      - The row has `finish == "error"`.
      - The client received the bytes that did arrive.
    - `test_client_disconnect_is_recorded_as_client_gone`:
      - The client reads the first chunk, then closes its socket.
      - The fake backend then sends 200 more delta events, so the
        gateway's write fails.
      - The row has `finish == "client_gone"`.
    - `test_old_fields_keep_their_meaning`:
      - The row has `source == "gateway"`.
      - It has `tps` and `ttft_ms` keys, and `model == "fake"`.
  - **Drive it.** Every test reads the newest row of the temp
    `FXLLA_STATS_FILE`, which the existing setUp already sets.

- [ ] **Step 2: Run the tests and confirm they fail.**
  Run: `FXLLA_STORE=/tmp python3 -m unittest gateway.test_gateway_e2e -v`
  Expected: the new tests FAIL, and the existing e2e tests still pass.

- [ ] **Step 3: Implement in `_proxy` and `_record`.**
  - **Injection.** Inject when all three hold:
    - the path is a completion path (`metrics.is_completion_path`)
    - `body.get("stream")` is truthy
    - the client did not set `stream_options.include_usage`

    Also require `engine_for(alias) in {"mlx", "gguf", "omlx"}`. When
    injecting, set `body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}`
    before serializing, and wrap the write path in a `UsageChunkStripper`.
  - **Split the broad `except` in the read and write loop.**
    - An exception from `resp.read` sets `finish = "error"`.
    - BrokenPipeError, ConnectionResetError, or OSError from
      `self.wfile.write` or `flush` sets `finish = "client_gone"` and stops
      reading.
    - The bytes written on the success path are unchanged.
  - **Upstream HTTPError and the 502 path.** Both call `_record` with
    `finish = "error"` after replying.
  - **`_record`.**
    - Build the record from `sm.report()` or, for non-streamed replies,
      from `engine_report` on the parsed body's `usage` and `timings`.
    - `output_estimated` is True only when there is no usage and the delta
      count is used.
    - `duration_ms` is the gateway clock from dispatch to the last byte.
    - `finish` is the classified value, or else `sm.finish` or the body's
      `choices[0].finish_reason`.
    - It always calls `TELEMETRY.record(rec)`.
    - It appends to `stats.jsonl` only when `MANAGER.backend_meta(alias)`
      is not None. That check is what keeps `ram_mb` real.
    - Remove the old `if not tokens: return`, so errors are recorded.
  - **`client_addr`** is `self.client_address[0]`. **`client_app`** is
    `client_tag(self.headers.get("User-Agent"))`.

- [ ] **Step 4: Run the tests and confirm they pass.**
  Run the full Python suite command from Global Constraints. Expected: OK,
  apart from the known media failure.

- [ ] **Step 5: Commit.**
  `git add gateway/fxlla_gateway.py gateway/metrics.py gateway/test_gateway_e2e.py && git commit -m "feat(gateway): record every request with tokens, cache, timings, finish and client"`

---

### Task 6: Live state and events from the Manager

**Files:**
- Modify: `gateway/fxlla_gateway.py`: `Manager`, `Backend`, `do_POST`, and
  `_proxy`
- Test: `gateway/test_gateway.py`

**Interfaces:**
- Consumes: `TELEMETRY.event` from Task 3.
- Produces:
  - **`Manager.status()` items gain:**
    - `context`, from `model_context(alias)` and cached on the Backend at
      registration
    - `keep_warm_left_s`, when idle and `KEEP_WARM_S > 0`; otherwise None
    - `client_addr`, `client_app` and `max_tokens`, for the in-flight
      request
    - `cached_tokens`, once known
    - `live_tps`: output deltas per second over the last 2 s of
      `progress()` calls while generating, rounded to 0.1.
      - It is None for `omlx`, because omlx packs about 14 tokens per
        delta, so a delta rate there would be a false number.
      - It is also None when fewer than 2 progress points fall in the
        window.
  - **`Manager.begin(alias, prompt_tokens=None, client_addr=None, client_app=None, max_tokens=None)`.**
    Keyword arguments only, defaulted. It still returns the Backend.
  - **New Manager counters, read under the lock:**
    - `waiting: int`: threads inside `ensure` that are blocked on a load
    - `oldest_wait_started: float | None`
  - **Events, through `TELEMETRY.event`:**
    - `ensure` emits `load_started`, `loaded` (with `seconds` and
      `size_mb`), `load_failed` (with `reason`, the existing `why`
      string), and `exited` (with `code`, from the dead-backend branch).
    - `_evict_one` emits `evicted`.
    - `reap_idle` emits `idle_unloaded`.
    - `unload_all` emits `admin_unloaded`, one per alias.
- **Choosing `max_tokens`.** Take `body.get("max_tokens")`, then
  `body.get("max_completion_tokens")`, else None.

- [ ] **Step 1: Write the failing tests** in `TestLiveState`. Reuse the
  `TestBackendStartFailure` setUp pattern for the failing launcher, and use
  `_FakeProc` and `_ExitedProc`.
  - `test_load_failure_emits_load_started_then_load_failed`
    - Assert the event kinds, newest first: `["load_failed", "load_started"]`.
    - Assert that `reason` contains `"exited with code 127"`.
  - `test_dead_backend_emits_exited_with_its_code`
    - `_ExitedProc` returns code -9. Assert an `exited` event with
      `code == -9`.
  - `test_reap_idle_emits_idle_unloaded` and
    `test_unload_all_emits_admin_unloaded`.
  - `test_status_shows_client_and_max_tokens_while_in_flight`
    - Call `begin("busy", 10, client_addr="10.0.0.5", client_app="opencode", max_tokens=4096)`.
    - The status item has those three values and `inflight == 1`.
  - `test_live_tps_from_progress_and_none_for_omlx`
    - Register an mlx backend.
    - Call `progress` 21 times with produced 0..20 over an injected 1-second
      span: patch `gw.time.monotonic`.
    - `live_tps` is 20.0.
    - The same sequence on an `omlx` backend gives `live_tps is None`.
  - `test_keep_warm_left_counts_down`
    - With `KEEP_WARM_S` patched to 3600 and `last_used` set 600 s ago,
      `keep_warm_left_s` is between 2990 and 3000.
  - `test_waiting_counts_threads_blocked_on_a_load`
    - Hold a load open with a launcher that sleeps, as in the existing
      timeout test.
    - Start 3 threads calling `ensure` for the same alias.
    - While they block, `m.waiting == 3`; the loader counts too.
    - After the load fails, `m.waiting == 0`.
  - For isolation, every test patches `gw.TELEMETRY` with a fresh
    `Telemetry()`.

- [ ] **Step 2: Run the tests and confirm they fail.**
  Run: `FXLLA_STORE=/tmp python3 -m unittest gateway.test_gateway.TestLiveState -v`
  Expected: FAIL.

- [ ] **Step 3: Implement.**
  - **`waiting`.** Increment it under the lock on entry to the slow path,
    for both the loader and the waiters, and decrement it in a `finally`.
    No new lock.
  - **`do_POST`.** Pass `client_addr`, `client_app` and `max_tokens` to
    `MANAGER.begin`.
  - **`_proxy`.** Set `held.cached_tokens` when the report arrives. Each
    task's code is allowed to update the Backend held for that request,
    and only that one.

- [ ] **Step 4: Run the tests and confirm they pass.**
  Run the full Python suite. Expected: OK, apart from the known media
  failure.

- [ ] **Step 5: Commit.**
  `git add gateway/fxlla_gateway.py gateway/test_gateway.py && git commit -m "feat(gateway): live per-request state, a load queue count and structured events"`

---

### Task 7: `GET /monitor`, and the docs

**Files:**
- Modify:
  - `gateway/fxlla_gateway.py`: `do_GET`, plus module globals `BOOT_TS`
    and `MACHINE`
  - `gateway/README.md`
  - `CHANGELOG.md`
- Test: `gateway/test_gateway_e2e.py`

**Interfaces:**
- Consumes: `TELEMETRY.snapshot()` from Task 3,
  `MachineSampler(MODELS_DIR).sample()` from Task 4, and the Manager items
  and counters from Task 6.
- Produces: `GET /monitor` and `GET /v1/monitor`. They return a JSON
  object with exactly these top-level keys:
  - `schema_version`: 1
  - `instance`: `pid`, `host`, `port`, `boot_ts`, `budget_mb`,
    `keep_warm_min`, and `version`. `version` is
    `git -C <repo> describe --always --dirty`, computed once at import;
    on failure it is `"unknown"`.
  - `backends`: `MANAGER.status()`
  - `resident_mb`
  - `work`: `{"waiting", "oldest_wait_s", "client_gone_total"}`
  - `totals`, `series`, `ttft_p95_ms`, `day`, `recent`, and `events`, from
    the telemetry snapshot
  - `machine`: `MACHINE.sample()`
- **`/health` is byte-for-byte unchanged in shape.**

- [ ] **Step 1: Write the failing tests.**
  - `test_monitor_has_every_section`
    - Make one streamed request, then `GET /monitor`.
    - Assert the top-level keys exactly.
    - Assert that `recent[0]["id"]` matches the id in `stats.jsonl`, and
      `instance["port"] == self.gport`.
  - `test_health_shape_is_unchanged`
    - The keys are exactly `{"status", "resident", "budget_mb"}`.
  - `test_monitor_answers_when_machine_commands_fail`
    - Patch `gw.MACHINE` with a `MachineSampler` whose `run` raises.
    - Assert 200 and `machine["gpu_util_pct"] is None`.
  - `test_monitor_needs_no_auth_and_never_contains_text`
    - Send a request whose prompt contains `"SECRET-PROMPT-TEXT"`.
    - Assert that the string is absent from the `/monitor` body.

- [ ] **Step 2: Run the tests and confirm they fail.**
  Run: `FXLLA_STORE=/tmp python3 -m unittest gateway.test_gateway_e2e -v`
  Expected: the new tests FAIL with 404.

- [ ] **Step 3: Implement the route in `do_GET`.**
  - Read the Manager fields under `MANAGER.lock`.
  - Call `MACHINE.sample()` outside the lock.

- [ ] **Step 4: Update the docs.**
  - **`gateway/README.md`** gets a section, "Telemetry and /monitor":
    - the record fields, with one line each
    - the injection allowlist and why mtplx is excluded
    - that `/monitor` is unauthenticated on a LAN bind and what it exposes
      (model names, token counts, client addresses, never text)
  - **`CHANGELOG.md`** gets an `### Added` entry under `[Unreleased]`, in
    the file's existing prose style.

- [ ] **Step 5: Run everything and confirm it passes.**
  - Run `bash tests/run.sh`. Expected: `all N suites passed` and
    `shellcheck clean`.
  - Run the full Python suite. Expected: OK, apart from the known media
    failure.
  - Check ASCII:
    `git diff main --name-only | xargs -I{} sh -c 'LC_ALL=C grep -Hn "[^[:print:][:space:]]" "{}"'`
    prints nothing.

- [ ] **Step 6: Run the end-to-end check on this Mac.** Strata keeps
  running; do not stop it.
  1. Restart the gateway: `fxlla unserve; fxlla serve`.
  2. Send `curl` streamed requests to `tiny` (mlx) and to
     `Qwen3.5-4B-GGUF` (gguf), without `stream_options`.
  3. Confirm that neither client stream contains `"usage"`.
  4. `curl -s localhost:8080/monitor | python3 -m json.tool` shows both
     requests in `recent`, with non-null `prompt_tokens` and
     `output_tokens`. The gguf row has non-null `prompt_ms`. `machine` has
     no None values.
  5. Run `opencode run -m local/tiny "Say OK"`. It answers `OK`, and its
     row has `client_app == "opencode"`.
  6. Unload: `curl -s -XPOST localhost:8080/admin/unload`.

- [ ] **Step 7: Commit.**
  `git add gateway/fxlla_gateway.py gateway/test_gateway_e2e.py gateway/README.md CHANGELOG.md && git commit -m "feat(gateway): serve a /monitor snapshot of requests, state, totals, machine and events"`

---

## After the last task

Before pushing to main, run an adversarial review of the whole branch diff
(repo convention). Fix what it confirms, then push.
