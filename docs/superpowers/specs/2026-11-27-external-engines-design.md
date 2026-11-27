# External engines in /monitor

Date: 2026-11-27. Status: design, for approval. Piece 3 of 4. Piece 1 is
`2026-11-26-gateway-telemetry-design.md`, piece 2 is
`2026-11-27-tools-telemetry-design.md`. The screen it feeds is
`2026-11-26-monitor-look.md`: the panel Other engines, tab 6 Externos, and
Strata rows in Recent requests.

## Intent

fxlla is not the only inference engine on this Mac. Strata serves a large
model on port 8096 next to the gateway, Ollama answers on 11434, and LM
Studio and ComfyUI start now and then. The monitor must show, on one
screen, what each of them holds in memory and what it is doing, because
they compete with fxlla for the same RAM.

## Approach (decided by the user)

The gateway observes the engines and adds an `externals` section to the
same `GET /monitor` JSON. The monitor, the menu bar app and the LattePanda
then read one source.

Rejected: the monitor polls the engines itself. It would keep Strata's key
off the network, but the app and the LattePanda would see nothing.

Cost the user accepted: `/monitor` has no auth and listens on the LAN, so a
summary of data that Strata guards with a key becomes visible on the LAN.
The summary is limited to what fxlla already publishes about itself (see
Privacy).

## Goals

- One `externals` object in `/monitor`, one entry per engine: `strata`,
  `ollama`, `lmstudio`, `comfyui`.
- Each entry says `up` or `down`, and when up, its loaded model(s), memory,
  and current work.
- Strata requests appear with the same fields as fxlla's `recent` rows, so
  the monitor can merge them into Recent requests.
- `/monitor` stays fast when every engine is down.

## Non-goals

- No proxying, no load or unload of external models, no actions.
- No token counts for Ollama, LM Studio or ComfyUI: they do not publish
  them outside the response (Ollama's log has only method, path, status and
  latency).
- No ComfyUI step progress: it exists only on the `/ws` websocket.
- No persistence. External history lives in the engines; the gateway keeps
  nothing across restarts.

## Anti-goals

- **Waking an engine.** `lms ps` starts LM Studio's service and server
  (observed in recon). The observer only makes HTTP requests to ports that
  already listen. It never runs `lms`, `ollama`, `comfy` or `strata`.
- **Leaking a key.** The Strata key is read from disk, sent only to
  127.0.0.1, and never logged, returned, or put in an error string.
- **Slowing /monitor.** A hung engine must not hold up the response.

## How each engine is read

All probes are HTTP GETs to 127.0.0.1 with a short timeout, run in
parallel. A refused connection means `down` with no reason. A timeout or a
bad response means `down` with a short reason (`timeout`, `http 401`,
`bad json`).

**Strata** (port 8096, `FXLLA_STRATA_URL` overrides the base URL)
- `GET /health` (no key) gives up/down, `model`, `max_context`, `loaded`.
- `GET /v1/telemetry` (key) gives `model{id,loaded,context}`,
  `work{active,queued}`, `rates{answer_tok_s,prefill_tok_s}`,
  `ram{used,total}`, `totals{requests,prompt_tokens,output_tokens,since}`,
  `recent[]` and `activity.events[]`.
- `GET /metrics` (key) gives the `live` block: `state`, `phase`,
  `prompt_read`, `prompt_total`, `generated`, `max_tokens`, `elapsed_s`,
  `tok_s`. This is the only source of a live in-flight row.
- **The key.** Sent as `Authorization: Bearer <key>`. Taken from
  `STRATA_API_KEY` if set; otherwise from the `api_key` field of the JSON
  files in `~/.local/state/strata/launch/` (mode 0600, owned by the user).
  More than one launch file can exist, so the observer tries each key and
  remembers the one that worked until it gets a 401. It uses the full
  `api_key`, not Strata's read-only `telemetry_read_key`, because that one
  does not open `/metrics`, the only source of the live row. With no working key,
  the entry still shows the `/health` data and `auth: "missing"`.

**Ollama** (port 11434, `FXLLA_OLLAMA_URL`)
- `GET /api/ps`: loaded models with `name`, `size`, `size_vram`,
  `expires_at`, `context_length`. The loaded shape comes from Ollama's docs;
  recon only saw an empty list. Verify with a loaded model before claiming
  it works.
- `GET /api/tags`: installed model count and total size.
- Requests today: count the `[GIN]` lines in `~/.ollama/logs/server.log`
  whose path is a generation path (`/api/chat`, `/api/generate`,
  `/api/embed`, `/api/embeddings`, `/v1/chat/completions`,
  `/v1/completions`, `/v1/embeddings`) and whose date is today, plus their
  median latency. Listing calls (`/api/tags`, `/api/ps`) are not requests:
  opencode's auto-detect makes them all day. The log is read incrementally
  from the last offset; a smaller file means it rotated, so start over.

**LM Studio** (port from `~/.lmstudio/.internal/http-server-config.json`,
default 1234, `FXLLA_LMSTUDIO_URL`)
- `GET /api/v0/models`: entries with `state: "loaded"` are the loaded
  models, with `max_context_length`. `/v1/models` has no state field.

**ComfyUI** (ports 8000 then 8188, `FXLLA_COMFYUI_URL`)
- 8000 is the Comfy Desktop app, 8188 the `comfy` CLI default. Probe both,
  use the first that answers.
- `GET /system_stats`: version, `ram_total`, `ram_free`, and per device
  `vram_total`, `vram_free`.
- `GET /queue`: lengths of `queue_running` and `queue_pending`.

## Timing

- Probes run only when someone requests `/monitor`. No background thread
  polls an engine nobody is watching.
- One shared deadline of 1 s for all external probes in a snapshot. An
  engine that misses it shows its previous value marked `stale: true` with
  the age, or `down: timeout` if there is none.
- The result is cached for 2 s, so several monitors polling every second
  cost one round of probes.
- A `down` engine is re-probed at most every 10 s, because a refused
  connection is cheap but a filtered one waits for the whole timeout.

## Output shape

```
"externals": {
  "strata": {
    "up": true, "url": "http://127.0.0.1:8096", "auth": "ok",
    "model": "qwen3.8-flash-next-iq3_s", "context": 262144, "loaded": true,
    "state": "ready",
    "work": {"active": 0, "queued": 0},
    "live": null | {"phase", "prompt_read", "prompt_total", "generated",
                    "max_tokens", "elapsed_s", "tok_s"},
    "rates": {"answer_tok_s": null, "prefill_tok_s": null},
    "ram": {"used_mb": 110335, "total_mb": 131072},
    "totals": {"requests": 5174, "prompt_tokens": 701520462,
               "output_tokens": 3979561, "since": 1795520322},
    "recent": [<fxlla request row, see Strata rows>],
    "stale": false, "age_s": 0.4
  },
  "ollama":   {"up", "url", "loaded": [{"name", "size_mb", "vram_mb",
               "context", "expires_in_s"}], "installed": {"count",
               "size_gb"}, "today": {"requests", "median_ms"}, ...},
  "lmstudio": {"up", "url", "loaded": [{"name", "context"}], ...},
  "comfyui":  {"up", "url", "version", "queue": {"running", "pending"},
               "vram": {"used_mb", "total_mb"}, ...}
}
```

Each entry also carries `down_reason` when `up` is false.

**Strata rows** use the same shape as fxlla's own `recent` rows
(`metrics.build_sample` plus `engine_report`), so the monitor merges them
without a second code path:
- `ts`: Strata's `time`, as int epoch seconds.
- `model`: the current `model.id`. A Strata instance serves one model for
  its lifetime (a different model means a restart and a new
  `instance.boot_time`), so this is exact. Rows older than `boot_time` are
  dropped.
- `engine` and `source`: `"strata"`.
- `prompt_tokens`, `output_tokens`, `prompt_ms`, `decode_ms`, `finish`:
  copied. `cached_tokens`: Strata's `reused`. `ttft_ms`:
  `first_engine_token_s` times 1000. `duration_ms`: `duration_s` times
  1000. `tps`: `output_tokens / (decode_ms / 1000)` when `decode_ms > 0`.
- `reasoning_tokens`, `client_app`, `ram_mb`: null. Strata does not
  report them.

(Ruling after the builder check: the monitor merges on this shape, so it
must match fxlla's rows field for field.)

## Privacy

- Strata `recent` rows keep only counts, timings and `finish`. The
  request `id`, `launch_attestation`, `instance.id` and every device
  field except memory are dropped.
- No prompt text exists in any source used, and none is read.
- Ollama log lines contribute only counts and latencies. Client IPs are
  not copied.
- The Strata key never leaves the process except in the header to
  127.0.0.1. A test asserts that no `/monitor` response and no log line
  contains it.
- `schema_version` of `/monitor` stays 1. `externals` is an additive key,
  as `kb`, `graph` and `media` were in piece 2; readers ignore keys they do
  not know.

## Constraints

- Stdlib only (`urllib`, `concurrent.futures`), like the rest of the
  gateway.
- The observer is its own module, `gateway/externals.py`, with one function
  per engine that takes the raw JSON and returns the entry, so each parser
  is tested from saved fixtures without a live engine.
- Fixtures are recorded from this Mac (Strata live, Ollama `/api/tags`,
  LM Studio `/api/v0/models`) and sanitized; shapes not observed live
  (Ollama `/api/ps` loaded, ComfyUI) are built from docs and source and
  marked UNVERIFIED in the fixture name.
- English and ASCII only; no secrets in tests or fixtures.

## Left to the builder

- The thread pool size and how the shared deadline is enforced.
- How the remembered Strata key is stored (in memory only).
- Exact `down_reason` wording.
- Whether Strata's `/v1/telemetry` and `/metrics` are fetched in parallel
  or `/metrics` only when `work.active > 0`.
