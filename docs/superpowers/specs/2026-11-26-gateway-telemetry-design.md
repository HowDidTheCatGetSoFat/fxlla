# Gateway telemetry: capture what a live monitor needs

Date: 2026-11-26. Status: design, awaiting approval.

## Intent

The user wants a live terminal monitor for fxlla in the style of Strata's
`strata monitor`. That screen shows the model and its state, current work, live
answer and prompt-reading speed, time to first token, GPU and memory, totals
since boot, and a table of recent requests with input, output, and reused
tokens, duration, and finish reason.

The user asked that fxlla first capture everything it is missing, and that the
monitor come second, as its own piece of work. This document covers only the
capture. The monitor will read what this produces.

The gateway is the place to capture. It already sees every request, every
backend load and eviction, and the streamed bytes of every answer.

## Goals

1. **A full record per finished request.** Each completed request is
   appended to the existing stats time-series (`stats.jsonl`, written by
   `gateway/metrics.py`) with these fields:
   - `id`: short, unique per gateway run.
   - `ts`, `model`, `engine`, `ram_mb`, `ttft_ms`, `tps`: existing, and keep
     their current meaning.
   - `prompt_tokens`, `cached_tokens`, `output_tokens`. `cached_tokens` is the
     prompt tokens served from the engine's prompt or prefix cache.
   - `prompt_ms`, `decode_ms`, `duration_ms`. These are the time spent reading
     the prompt, the time spent generating, and the wall time from dispatch to
     the last byte.
   - `finish`: one of `stop`, `length`, `tool_calls`, `client_gone`, `error`.
   - `stream`: true or false.
   - `client`: the remote address, plus a short app tag from the User-Agent
     (opencode, hermes, curl, other). This is new relative to Strata. It
     matters here because two machines share one gateway.
2. **Live state.** For each resident backend, what the gateway already tracks
   (phase, prompt tokens, output so far, busy seconds), plus:
   - The current answer rate in tok/s, over a short sliding window of streamed
     deltas.
   - The queue: requests waiting on a model that is still loading, and the age
     of the oldest one.
   - The count of client disconnects mid-answer since boot.
3. **Totals since gateway start.** Requests, prompt tokens, cached tokens,
   output tokens, errors, and the boot time.
4. **Machine.**
   - Unified memory, total and used.
   - GPU utilization percent and GPU in-use memory, read without sudo from
     `ioreg` (IOAccelerator `PerformanceStatistics`).
   - Each backend's RSS, which exists today.
5. **Structured events.**
   - Load started, loaded (with seconds and MB), load failed (with reason),
     evicted (LRU), unloaded (idle), unloaded (admin), and backend exited on
     its own.
   - They go to a bounded in-memory ring. `gateway.log` keeps its text lines
     as today.
6. **One read endpoint.** `GET /monitor` returns a JSON snapshot with the
   instance, backends, live work, totals, machine, recent requests (the last
   N from memory), and events. The future monitor, `fxlla status`, and the
   menu bar app can all read it.

## Non-goals

- The monitor UI itself. That is the next piece of work.
- Prometheus or OpenTelemetry export.
- Per-user accounting or auth.
- Counting tokens with a tokenizer in the gateway.
- Changing what the single-model server (`fxlla on`) records.

## Anti-goals

These would make it a failure even if it works:

- **Telemetry changing a response.** Any capture error must be swallowed, as
  `_record` does today. A client must never see a byte it would not have seen
  without telemetry. That includes the usage chunk the gateway may request on
  the client's behalf (see Approach).
- **Invented numbers.** A field the engine did not report is `null`, not an
  estimate presented as a measurement. The existing rule in
  `StreamMetrics.result` (an implausible tps becomes None) is the model to
  follow. If an estimate is ever stored, it gets a separate field name.
- **Breaking existing readers.**
  - `stats.jsonl` keeps every current field with its current meaning. New
    fields are only added.
  - Readers are `fxlla stats` (`bin/fxlla`) and the menu bar app
    (`app/Sources/fxllaMenuBar/StatsSample.swift`). Swift's Decodable ignores
    unknown keys, and that has to be confirmed by a test or build.
  - The 5000-line cap stays.
- **Slowing the proxy path.** No per-chunk work beyond what StreamMetrics
  already does. Machine stats are sampled lazily, when `/monitor` is read, and
  cached for about 1 s. They are not sampled per request.
- **Leaking secrets.** Never record prompt or answer text, headers other than
  the User-Agent, or API keys.

## Constraints

- Standard library only in the gateway and `metrics.py`, as today.
- English and ASCII only, per AGENTS.md.
- The CLI stays the source of truth. The gateway exposes data, and the
  CLI/app read it.
- Four engines sit behind the gateway, and they report differently:
  - llama-server (gguf) returns a `timings` object (`prompt_n`, `prompt_ms`,
    `predicted_n`, `predicted_ms`, `cache_n`) and `usage`.
  - mlx_lm.server, omlx, and mtplx return OpenAI `usage`. In a stream they do
    so only when `stream_options.include_usage` is set. omlx packs about 11
    tokens per SSE delta, so delta counting undercounts it about 10x.
  - What each engine actually returns must be verified against the installed
    versions, listed below. Do not assume it from documentation.
- Installed engines as of 2026-11-25:
  - llama.cpp 1.1.0-dev (b12765)
  - mlx-lm 0.32.3.dev22
  - mlx 0.33.0.dev
  - omlx 0.8.3
  - mtplx 2.13.3
- Tests: `bash tests/run.sh`, and the Python suite list from
  `.github/workflows/ci.yml`. The `tiny` model is used for an end-to-end check.

## Approaches

- **Capture in the gateway (chosen).**
  - Extend the existing proxy and metrics path.
  - Ask streaming engines for usage.
  - Parse whichever of `usage` or `timings` arrives.
  - Keep live state and totals in the Manager.
  - Expose `/monitor`.
  - This is the only place that sees all four engines and every client.
- **Read each engine's own metrics endpoint (rejected).** llama-server has
  `/metrics` and `/slots`, while the MLX-family servers expose different
  things or nothing. The result would be uneven per engine, and it would
  still miss client identity and disconnects.
- **Tokenize in the gateway (rejected).** It is heavy, it needs a tokenizer
  per model, and it duplicates numbers the engines already report.

## Approach details decided

- **The usage chunk.** For a streamed completion where the client did not set
  `stream_options.include_usage`, the gateway sets it upstream. It then drops
  the resulting usage-only chunk before it reaches the client. The rule is
  that a chunk with empty `choices` and a `usage` key is dropped only when
  the client did not ask for it. If the client asked, it passes through
  untouched. This rule is narrowed, and restricted to an engine
  allowlist, under "Verified on 2026-11-26" below.
- **Source precedence.**
  - Engine `timings` comes first.
  - Then `usage`, including `prompt_tokens_details.cached_tokens` when
    present.
  - Then gateway clocks for `duration_ms` and `ttft_ms`.
  - `output_tokens` falls back to the delta count only if no usage arrives,
    and it is then marked by a separate boolean field `output_estimated`.
- **`finish`.**
  - Taken from the last `finish_reason` seen.
  - `client_gone` when writing to the client fails (the broad `except` around
    the write loop in `_proxy` is where this is detected today).
  - `error` for an upstream HTTP error or a 502.
  - Failed requests are recorded too, with zero output. Today they are
    skipped (`if not tokens: return`).
- **`/monitor` is unauthenticated, as `/health` is.** The gateway binds
  0.0.0.0 here, so anyone on the LAN can read model names, token counts, and
  client IPs. That is acceptable for this LAN and is called out as a known
  exposure.

## Calls made after the builder check

These are the designer's calls, settled from the code and from the
anti-goals:

- **`tps` and `ttft_ms` keep their gateway-clock computation**, as in
  `StreamMetrics.result` today. Engine `timings` and `usage` feed only the new
  fields (`prompt_ms`, `decode_ms`, `prompt_tokens`, `cached_tokens`,
  `output_tokens`). The plotted series in `fxlla stats` and the menu bar do
  not shift. One exception: when usage arrives, `tps` already uses its
  `completion_tokens` (existing behavior), and that stays.
- **Failed and zero-output requests.** They are written to `stats.jsonl` only
  when a backend was resident, so `ram_mb` is a real RSS. Readers tolerate a
  null `tps` and `ttft_ms`: the Swift charts `compactMap` them, and
  `_stats_fmt` prints "-". However, `ram_mb` is plotted unconditionally, and a
  0 would draw a false dip. A request that failed before any backend existed
  goes only to the in-memory ring, the totals, and an event.
- **An engine that rejects `stream_options`.** Verify each engine first. If
  one rejects it, do not inject for that engine; keep a per-engine allowlist
  in code. Its streamed rows then carry `output_estimated: true`. Never retry
  silently, and never let injection break a client.
- **The `instance` section of `/monitor`** holds:
  - gateway pid, bind host and port, and boot time
  - `budget_mb` and the keep-warm minutes
  - the fxlla version (`git describe` of the repo, or "unknown")
- **The queue** counts request threads blocked in `Manager.ensure` waiting on
  a load. That covers the loader thread itself and the non-loader threads
  that wait on the per-model Event. Use a counter maintained under the
  existing lock. No new locking scheme.
- **`client_gone`.** Split the read loop's broad `except` so that an upstream
  read failure is told apart from a client write failure (BrokenPipeError,
  ConnectionResetError). The bytes sent to the client must be identical to
  today; only the classification changes. Cover it with tests for both
  failure sides.

## Verified on 2026-11-26 (installed versions)

- **ioreg.** `ioreg -r -c IOAccelerator` gives `PerformanceStatistics`
  without sudo:
  - `Device Utilization %`, `Renderer Utilization %`, `Tiler Utilization %`
  - `In use system memory`
  - `Alloc system memory`
- **mlx_lm.server (0.32.3.dev22).** It accepts `stream_options.include_usage`
  and sends a final chunk with `choices: []` and a `usage` key, before
  `[DONE]`. `usage` carries `prompt_tokens`, `completion_tokens`, and
  `prompt_tokens_details.cached_tokens`, in streamed and non-streamed
  replies alike. It sends no `timings`.
- **llama-server (b12765).** It accepts `include_usage` the same way. Its
  final chunk carries both `usage` and `timings` (`cache_n`, `prompt_n`,
  `prompt_ms`, `predicted_n`, `predicted_ms`), and so does a non-streamed
  reply. Note that `usage.prompt_tokens` counts cached tokens too:
  `prompt_tokens = prompt_n + cache_n`.
- **omlx (0.8.3, tested with `omlx-coder`).**
  - It accepts `include_usage` and sends a usage-only final chunk
    (`choices: []`). Without the option it sends no usage at all.
  - `usage` also carries engine timings, in seconds:
    - `prompt_eval_duration` and `generation_duration`
    - `time_to_first_token`, `total_time`, and `model_load_duration`
    - `completion_tokens_details.reasoning_tokens`
  - It still packs many tokens per SSE delta: 300 tokens arrived in 21
    deltas, about 14 per delta. Delta counting undercounts omlx by about
    14x, so injection matters most here.
- **mtplx (2.13.3, tested with `mtplx-qwen38`).**
  - It sends `usage` even when not asked. Its final chunk is NOT
    usage-only: it carries `choices: [{delta: {}, finish_reason}]` together
    with `usage` and `timings`.
  - `timings` uses llama-server names (`prompt_n`, `prompt_ms`,
    `predicted_n`, `predicted_ms`), plus `draft_n` and `draft_n_accepted`
    from MTP speculation.
- **opencode (2.4.1) sends `stream_options.include_usage: true` itself.**
  This was captured from a real request through a provider configured with
  `aisdk:@ai-sdk/openai-compatible`, with User-Agent
  `opencode/latest/2.4.1/cli`. So for opencode the gateway injects nothing
  and only parses. Other clients (curl, Hermes) are unknown and go through
  the injection rule.

Consequences, corrections to the rules above:
- **Engine timing fields.**
  - `prompt_ms` and `decode_ms` come from engine-reported timings only:
    - llama-server and mtplx: `timings.prompt_ms` and
      `timings.predicted_ms`
    - omlx: `usage.prompt_eval_duration` and `usage.generation_duration`,
      times 1000
  - They are null for mlx_lm.server, which reports none. The gateway's own
    clocks appear as `ttft_ms` and `duration_ms` and are not copied into
    the engine fields.
- **Injection allowlist.** Inject for mlx, gguf, and omlx. Never inject
  for mtplx, which already sends usage unasked; its stream reaches the
  client exactly as it does today.
- **The drop rule is narrowed.**
  - The gateway removes only what its own injection caused. If it injected
    and a chunk has `choices: []` plus `usage`, that chunk is dropped.
  - If an injected engine ever puts `usage` on a chunk that also has
    choices, the chunk is forwarded unchanged rather than rewritten.
  - Anything an engine sends unasked is never stripped.
- **New optional fields per request**, null when the engine does not report
  them:
  - `reasoning_tokens`, from `completion_tokens_details`
  - `draft_tokens` and `draft_accepted`, from mtplx `timings`. These give
    the MTP acceptance rate, which the monitor can show.

## Added after the monitor mockup was approved

The approved home screen is described in
`2026-11-26-monitor-look.md`. It needs these live fields from the gateway:

- **Per in-flight request:**
  - the client (address and app tag)
  - the requested `max_tokens`, so output shows as "612/4096"
  - the cached prompt tokens once known
- **Per resident backend:**
  - the context window it was started with, from `ggufmeta.serve_plan` or
    the engine's own config; the monitor shows "12k/262k"
  - the seconds left before the keep-warm reaper unloads it
- **In `/monitor`:**
  - `budget_mb` and the resident total
  - a short in-memory series of the last 5 minutes of answer tok/s,
    prompt-reading tok/s, and TTFT, for the sparklines
  - TTFT p95 over that window
  - prompt tokens saved by cache, and the share of requests per client app,
    over the current day
  - These are computed in the gateway from the in-memory ring. They are
    not written to `stats.jsonl`.

## Left to the builder

- The exact JSON shape of `/monitor`, beyond the sections listed in Goal 6.
- The sizes of the in-memory rings (recent requests, events), as long as they
  are bounded.
- How client app tags are matched from the User-Agent.
- Whether `/health` gains any of the new live fields. It must stay backward
  compatible.
- How to organize the code between `fxlla_gateway.py` and `metrics.py`.
- The per-engine verification method. Recorded fixtures of real responses
  from each installed engine are preferred as test data.
