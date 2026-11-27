# fxlla gateway

One OpenAI-compatible endpoint that fronts every downloaded model. Standard
library only. The CLI keeps ownership of load, download, and RAM; the gateway is
the router that orchestrates it.

## Pieces

- `fxlla_gateway.py` - the HTTP server. Aggregates downloaded models in
  `/v1/models`, and on each request routes to the model's backend, loading it on
  demand and evicting the least-recently-used backend when a load would exceed
  the RAM budget. Backends are launched through `fxlla _backend <alias> <port>`
  so the launch logic stays in the CLI.
- `metrics.py` - passive request metrics. Pure, standard-library helpers that
  derive time to first token and tokens per second from proxied traffic and
  append them to the stats time-series.

Driven through the CLI: `fxlla serve` / `fxlla unserve`. The endpoint defaults
to `127.0.0.1:8080`; keep the bind local unless you tunnel it with auth.

## Passive metrics

The gateway sees the real token stream as it proxies, so it measures usage
without a synthetic probe:

- Streamed responses (SSE): the first content delta marks time to first token,
  and deltas are counted for tokens per second (one token per delta,
  approximate). A trailing `usage` chunk (`stream_options.include_usage`), when
  present, overrides the count with the server's exact `completion_tokens`.
- Non-streamed responses: `usage.completion_tokens` is read from the body.

One sample per completed completion request is appended to `stats.jsonl` (the
same time-series `fxlla stats` and the menu bar app read), shaped as
`ts, model, engine, ram_mb, ttft_ms, tps` plus `source: "gateway"`. Recording is
best-effort: any failure is logged and swallowed so it never affects the
proxied response. Embeddings and non-generative endpoints are not timed.

## Telemetry and /monitor

`GET /monitor` (also `/v1/monitor`) returns one JSON object with everything the
gateway knows about itself. `/health` is unchanged. The top-level keys are
`schema_version` (1), `instance`, `backends`, `resident_mb`, `work`, `totals`,
`series`, `ttft_p95_ms`, `day`, `recent`, `events`, `machine`, `kb`, `graph`
and `media`.

Each finished completion request is also appended to `stats.jsonl`. The
fields are only ever added to, so older readers keep working. A field the
engine did not report is `null`; nothing is invented. The one estimate is the
delta count used for `output_tokens`, flagged by `output_estimated: true`.

- `id`: short id, unique per gateway run.
- `ts`, `model`, `engine`, `ram_mb`, `ttft_ms`, `tps`, `source`: as before.
- `prompt_tokens`: prompt size, cached tokens included.
- `cached_tokens`: prompt tokens served from the engine's prompt cache.
- `output_tokens`: tokens generated.
- `output_estimated`: true when `output_tokens` is a delta count, not the
  engine's number.
- `prompt_ms`: time the engine spent reading the prompt (engine timings only).
- `decode_ms`: time the engine spent generating (engine timings only).
- `duration_ms`: wall time from dispatch to the last byte.
- `finish`: `stop`, `length`, `tool_calls`, `client_gone` or `error`.
- `stream`: whether the client asked for a stream.
- `client_addr`: the client's remote address.
- `client_app`: tag from the User-Agent (`opencode`, `hermes`, `curl`,
  `python` or `other`).

`work.oldest_wait_s` is the number of seconds since the load queue last became
non-empty. It is not the age of the oldest blocked request. `work.waiting` is
the number of requests blocked on a model load, and `work.client_gone_total`
counts clients that disconnected mid-answer since boot.

Usage injection: to read exact token counts, the gateway adds
`stream_options.include_usage` to a streamed request when the client did not
ask for it, and removes the resulting usage-only chunk before the client sees
it. A client that asked for usage gets its stream untouched. Injection is
limited to the engines `mlx` and `omlx`, whose servers honor the option.
`mtplx` is excluded: it reports usage on its own terms and must not be sent
options it may reject. `gguf` is excluded because llama-server moves its
`timings` object to the usage-only chunk when the option is set, so stripping
that chunk would take `timings` from the client; the counts come from
`timings` instead.

`machine` holds unified memory, swap, memory pressure, GPU utilization and
in-use memory, and free disk space, read without sudo and cached for about a
second. All commands share a 2 second budget per sample; a command that is
missing (on Linux) or hangs leaves its fields `null` and `/monitor` still
answers.

### kb, graph and media sections

`kb`, `graph` and `media` are read-only views of work that does not go through
the gateway. The gateway only reads; it never writes, creates or repairs any
of these sources, and a missing or corrupt source leaves its fields empty or
`null`. A reader failure never fails `/monitor`.

- `kb`: `recent` (the last 50 records of `kb.jsonl`, newest first), `today`
  (`searches`, `adds`, `avg_search_ms`, `cold_starts`), `bases` (one row of
  `kb`, `sources`, `chunks` per knowledge base) and `db_bytes`. `bases` comes
  from one read-only SQLite query on `<store>/kb/kb.db` (1 s busy timeout,
  cached 10 s). The db is never created: when it is missing, `bases` is `[]`
  and `db_bytes` is `null`.
- `graph`: `recent` (last 50 records of `graph.jsonl`), `today` (`queries`,
  `indexes`, `avg_query_ms`) and `stats` (`files`, `defs`, `refs`,
  `indexed_at`). The gateway does not open kuzu, because of its writer lock
  and the dependency. `files`, `defs` and `refs` come from the newest
  successful `index` record, and `indexed_at` is the mtime of
  `<store>/graph/graph.kuzu`.
- `media`: built from `<store>/media/jobs/*.json` (or `$FXLLA_MEDIA_OUT/jobs`).
  `running` holds jobs with status `running`, each with `elapsed_s` and
  `eta_s` (`elapsed / step * (total - step)`, `null` without progress).
  `queued` holds queued jobs. `recent` holds the last 20 finished jobs, newest
  finish first. `today` is `jobs`, `done`, `failed` and `gpu_s`, the sum of
  `finished - started` over today's finished jobs. A queued or running job
  whose pid is gone is reported as `died` in `recent` (counted as failed
  today), by the same rule as `jobs._reap` but without writing to the record.

Notes and caveats:

- The event logs are appended oldest first. The reader takes the tail of the
  file, so `recent` is in append order, newest first, not sorted by `ts`.
- "Today" is the local calendar date. The `today` aggregates of `kb` and
  `graph` are computed from the newest 5000 records of the log, cached for
  2 s. With more than 5000 records in a day they cover only the newest 5000,
  so the counts are a lower bound. Only the live file is read, not the
  rotated `.jsonl.1`, so right after a rotation they can undercount.
- The jobs directory is listed at most once per 2 s.
- Privacy: a media item exposes only `id`, `job`, `media`, `model`, `steps`,
  `width`, `height`, `status`, `created`, `started`, `finished`,
  `peak_rss_mb`, `progress`, `caller` and `error`, plus `elapsed_s` and
  `eta_s` for running jobs. `error` is cut to its first line and 200
  characters, with the home directory replaced by `~`. The job's `argv`,
  `summary`, `output`, `log` and `pid` are never copied: they carry the prompt
  and home-directory paths, and `/monitor` has no authentication. kb and graph
  records are likewise reduced to a fixed field list and carry no query or
  chunk text.

`/monitor` has no authentication. On a loopback bind that is harmless; on a
LAN bind (`FXLLA_HOST=0.0.0.0`) anyone who can reach the port can read it. It
exposes model names, token counts, timings, and client addresses and app tags.
It never contains prompt or answer text, request headers other than the
User-Agent tag, or keys.

## Notes and limits (v0)

- Token counts from streamed deltas are approximate unless the server emits a
  usage chunk. RAM is the backend process RSS at the moment the request ends.
- Cold start: the first request to an unloaded model pays the load time;
  resident hot models avoid it.
- `/health` says what each backend is DOING, not only what it holds:
  `inflight`, `busy_s`, `phase`, `prompt_tokens` and `output_tokens`. `fxlla
  status` renders one of:

  ```
  working:  redteam-4bit, 25s so far - reading 181,714 tokens of context
  working:  redteam-4bit, 90s so far - generating, 1,900 tokens written
  ```

  Reading the prompt and writing the answer are separate phases with separate
  costs, and telling them apart is not cosmetic. A status that read "reading
  181,714 tokens of context" while the model was in fact writing page after
  page made a normal long answer look like a stuck prefill - and on a slow
  model (22 tok/s here) look hung, which got working turns cancelled. The phase
  flips to "generating" the moment the first token returns, and the count
  climbs live from the same delta counter the metrics already keep.

- A conversation larger than the model's window is refused with HTTP 400 and
  `type: context_overflow`, before the model is loaded. This exists because the
  alternative was measured and is worse than an error: a backend handed a
  request past its window accepts it and never answers - 180 seconds with no
  reply, no error, nothing. A session that grew that way reads as a hung chat
  rather than a full one, and compacting cannot rescue it, since compacting
  sends the whole conversation. The refusal names the two numbers so the reply
  points at the only cure, which is a new session.


- A tool call that reached the TEXT channel is promoted to a real one. mlx_lm's
  parser for this family anchors its closing tag to the end of the string
  (`<function=(.*?)</function>$`), and qwen3-coder sometimes closes a
  Llama-shaped call with a Hermes-shaped `</tool_call>`; the anchor then fails
  and the whole call arrives as prose. No client reads the text channel, so
  from the outside the model simply did not call anything. Measured at
  temperature 0, this is about wording rather than luck: "Run the shell
  command: echo hi" parsed 3/3 and "Use the bash tool to run: echo hi" parsed
  0/3. `evals/README.md` already names this class and says the remedy belongs
  to the serving layer.

  Only attempted when the request declared tools and the recovered name is one
  of them - a reply that merely discusses `<function=...>` is prose, and
  turning prose into a call is worse than missing one. Streams are repaired
  without buffering the answer: everything flows through as it arrives and only
  `data: [DONE]` is held, so the recovered call lands before the client stops
  reading and a normal reply keeps its incremental output.

- The same tool call returning the same result `FXLLA_LOOP_LIMIT` times in a
  row (default 8, 0 disables) is refused with `type: tool_loop`, naming the
  call. A local model here ran one command 240 times over 8.5 hours: it started
  a server in the foreground, so the command could never exit and the result
  was identical every time. A model with no new information retrying is not a
  malfunction, it is the only move it has, so the stop has to come from
  outside it - and from here rather than from any one client, because the loop
  is a property of the conversation and every client sends the conversation
  here.

  The result is part of what is compared, not just the call. The same command
  with a CHANGING result is progress - polling a build, watching a file grow -
  and must never be mistaken for a loop. Only a run ending at the newest
  exchange counts, so a model that tried something else in between is left
  alone. The run is also remembered HERE, across requests, because a client
  that compacts replaces the tool history with a summary and the conversation
  then shows a run of one - the 240-attempt session contained two compactions,
  which landed outside the run by luck. The memory is bounded and expires after
  30 minutes, since the thing worth bounding is time burned.

  This fires around the eighth attempt rather than the 240th, and
  before the conversation has grown large enough to trip the size check below,
  which is why it is checked first: told about its size, someone starts a new
  session and loops again; told about the repetition, they look at the call.

  The token count is estimated from characters, so it is a check on the
  impossible rather than on the marginal: a request merely near the limit is
  still sent to the backend, which is the only thing that knows for certain. A
  model whose window cannot be read is never refused - no window is not a
  small window.
