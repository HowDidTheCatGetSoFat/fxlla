# kb, graph and media telemetry for /monitor

Date: 2026-11-27. Status: design. Piece 2 of 4. Piece 1 is
`2026-11-26-gateway-telemetry-design.md`. The screen it feeds is
`2026-11-26-monitor-look.md`: the panels KB / RAG and Media, and kb rows in
Recent requests.

## Intent

The monitor must show what kb/RAG search, the code graph and media
generation are doing, as it already does for chat. Today none of these
record anything usable:

- **kb search and graph queries** run in short-lived processes: the CLI, or
  a subprocess that an MCP server spawns per call. No process writes a log.
- **Media jobs** keep a JSON record per async job. They have no progress,
  no peak RAM, and no model, steps or resolution fields. Synchronous runs
  (without `--async`) leave no record at all. Each generator's output is
  captured and thrown away, so every job log is 0 bytes.

## Approach (agreed)

- **Writers.** Each process appends one JSON line per event to its own file
  in the fxlla state directory:
  - `kb.jsonl`
  - `graph.jsonl`
  - `media.jsonl`
- **Reader.** The gateway reads those files, plus cheap on-disk stats, and
  adds the sections to `GET /monitor`.

Rejected:

- **POST each event to the gateway.** The event is lost when the gateway is
  down, and kb and media would come to depend on the gateway.
- **The monitor reads the sources itself.** That duplicates logic, and
  `/monitor` would stop being the single source.

## Shared contract

- **Helper module.** A new stdlib module, `lib/fxlla_events.py`, exposes:
  - `state_dir() -> str`. It resolves `FXLLA_STATE_DIR`, then
    `$XDG_STATE_HOME/fxlla`, then `~/.local/state/fxlla`, matching
    lib/core.sh:96.
  - `append(stream: str, record: dict) -> None`.
    - `stream` is one of `kb`, `graph`, `media`.
    - It adds `ts`, the epoch float, when absent.
    - It serializes compact JSON on one line.
    - It writes `line + "\n"` with a single `os.write` on an fd opened
      `O_WRONLY|O_APPEND|O_CREAT`, mode 0600, so concurrent processes never
      interleave inside a line.
    - It never raises: it catches everything, and on failure writes one
      line to stderr at most.
    - When the file passes 5 MB, the writer that notices renames it to
      `<stream>.jsonl.1`, replacing any older one, then appends to a fresh
      file. A race there loses at most a few lines, which is acceptable.
  - `read_tail(stream: str, max_lines: int = 200) -> list[dict]`. It
    returns the newest parseable records, newest first, reading only the
    file's tail with a seek, and it skips bad lines.
- **Import path.** Callers outside `lib/` add the repo's `lib` directory to
  `sys.path`, the way `bin/fxlla` sets `PYTHONPATH` for
  `opencode_config.py`. Each module computes the repo root from its own
  `__file__`.
- **No text.** Never record a query string, prompt, file content or chunk
  text. Paths are recorded only for the kb source being indexed and the
  graph repo being indexed, as a basename. Neither is ever a home-directory
  path.

## Record shapes

Every record has `ts` and `kind`.

- **`kb.jsonl`:**
  - `{"kind":"search","kb":str,"hits":int,"top_score":float|None,"duration_ms":int,"embed_cold":bool,"caller":"cli"|"mcp","ok":bool,"error":str|None}`
  - `{"kind":"add","kb":str,"source":basename,"chunks_added":int,"duration_ms":int,"ok":bool,"error":str|None}`
  - `{"kind":"rm",...}`, with the same basic fields.
  - `embed_cold` is true when this process had to start the embedding
    server (rag/core.py `Embedder.__enter__` launched it), and false when it
    reused one.
  - `caller` is `mcp` when invoked by `rag/rag_mcp.py`. The MCP passes an
    env var `FXLLA_CALLER=mcp` to the subprocess. Otherwise it is `cli`.
- **`graph.jsonl`:**
  - `{"kind":"query","query":"def"|"refs"|"callers"|"impact"|"unused","results":int,"duration_ms":int,"caller":...,"ok":bool,"error":str|None}`
  - `{"kind":"index","repo":basename,"files":int|None,"defs":int|None,"duration_ms":int,"ok":bool,"error":str|None}`
  - `caller` uses the same mechanism as kb, through `graph/graph_mcp.py`.
- **`media.jsonl`:**
  - One `start` record and one `end` record per generation, both async jobs
    and synchronous runs:
    `{"kind":"start"|"end","id":str,"job":bool,"media":"image"|"video"|"voice"|"edit"|"upscale","model":str|None,"steps":int|None,"width":int|None,"height":int|None,"duration_s":float|None,"peak_rss_mb":int|None,"status":"done"|"failed"|"cancelled"|None,"error":str|None,"caller":...}`
    - `id` is the job id for jobs, and a generated id for sync runs.
    - `steps`, `width` and `height` are parsed from the argv the generator
      received. Each is None when not determinable. Do not guess defaults.
    - `peak_rss_mb` comes from `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`
      in the process that waited on the generator child. That value is bytes
      on macOS and KB on Linux, so normalize it. For a sync in-process run,
      use `RUSAGE_SELF`.
    - `error` is the first line of the error, at most 200 characters.
  - **Progress, while running:**
    `{"kind":"progress","id":str,"step":int,"total":int}`
    - Write it at most once per second per job, and only when the step
      changed.

## Progress capture (behavior change)

- **Where output goes.** The job worker (media/jobs.py `run`) and
  media/generate.py's backend calls stop discarding the generator's
  stdout/stderr. The generator's combined output goes to the job's `.log`
  file, which is truncated at the start of the job. For sync runs, it is
  still captured as today, for the existing warning and error handling.
- **Parsing.**
  - A reader thread scans the stream for step progress.
  - tqdm-style `NN/MM` and `step N/M` patterns cover mflux, mlx-audio and
    ltx. Verify each against the real tool's output on this Mac. Small test
    models are fine. If a tool prints no parseable progress, it emits no
    progress record. Do not invent one.
- **What must not change.** The current behavior:
  - the `error` (last 800 characters of stderr) on failure
  - the `warnings` field
  - the output path
  - the exit codes

## Reader, in the gateway

`/monitor` gains three top-level keys, and `schema_version` stays 1:

- **`kb`:**
  - `recent`: the last 50 search, add and rm records, newest first
  - `today`: `{searches, adds, avg_search_ms, cold_starts}`
  - `bases`: `[{"kb", "sources", "chunks"}]`, plus `db_bytes`
- **`graph`:**
  - `recent`: the last 50 records
  - `today`: `{queries, indexes, avg_query_ms}`
  - `stats`: `{"files", "defs", "refs", "indexed_at"}`
- **`media`:**
  - `running`: the start records without an end, each with its latest
    progress, `elapsed_s`, and `eta_s`, where `eta_s` is
    `elapsed / step * (total - step)` and is None without progress
  - `queued`: the job records with status queued, from
    `<store>/media/jobs/*.json`
  - `recent`: the last 20 end records
  - `today`: `{jobs, done, failed, gpu_s}`, where `gpu_s` is the sum of
    `duration_s` over today's end records
- **How the stats are read.**
  - `bases` is read with one read-only SQLite query on `<store>/kb/kb.db`,
    `?mode=ro` URI, with a 1 s busy timeout. The result is cached for 10 s,
    and the gateway never creates the db.
  - `graph.stats` must not open kuzu, because of the writer lock and the
    dependency. Take `indexed_at` from the `graph.kuzu` mtime, and `files`
    and `defs` from the newest successful `index` record in `graph.jsonl`.
- **Failure.** Any reader failure leaves that section's fields None or
  empty. It never fails `/monitor`.

## Anti-goals

- **A tool failing because of telemetry.** Every append is fire-and-forget.
- **Text in the files.** Never record query, prompt, chunk or file content.
- **Changing what an MCP tool or CLI command returns** to its caller.
- **Slowing a kb search noticeably.** That rules out an extra subprocess
  and any network call. The append is one syscall.

## Constraints

- Standard library only in the writer and the reader.
- English and ASCII only, per AGENTS.md.
- Tests must pass on the Linux CI runner.
- New tests go into the existing test modules (`rag/test_rag.py`,
  `graph/test_graph.py`, `media/test_media.py`, `gateway/test_telemetry.py`,
  `gateway/test_gateway_e2e.py`), plus a new `lib/test_fxlla_events.py`.
  Add that new module to the CI unittest list and to the AGENTS.md list.

## Calls made after the builder check

These are the designer's calls, and they bind all four parts.

1. **State dir.**
   - `state_dir()` checks `FXLLA_STATE_DIR` first, as a new override that
     tests use to isolate.
   - Then `$XDG_STATE_HOME/fxlla`, then `~/.local/state/fxlla`.
   - On this Mac every process (CLI, MCP servers, gateway) resolves to
     `~/.local/state/fxlla`. `bin/fxlla serve` already forwards
     `FXLLA_STATE_DIR=$STATE_DIR` to the gateway, which is the same path.
2. **Frozen before the fan-out.**
   - The helper (`lib/fxlla_events.py`: `state_dir`, `append`, `read_tail`)
     and the caller convention are built first, as Part 0, with their own
     tests.
   - **Caller convention.** The three MCP servers set `FXLLA_CALLER=mcp` in
     the env of the subprocess they spawn. Writers read
     `os.environ.get("FXLLA_CALLER", "cli")`.
   - Part 0 is the only part that edits `.github/workflows/ci.yml` and
     AGENTS.md, adding `lib.test_fxlla_events`.
   - No part edits CHANGELOG.md or the top-level docs. The controller writes
     them at merge.
3. **"Today" and staleness.**
   - "Today" is the local date.
   - The reader computes today's aggregates from `read_tail(stream, 5000)`,
     cached for 2 s. With more than 5000 records in a day, the aggregates
     cover the newest 5000. This is documented, not hidden.
   - Rotated `.1` files are not read.
   - Media `start` records carry `pid`. A start without an end whose pid is
     no longer alive (`os.kill(pid, 0)` fails) is shown in `recent` with
     status `died`, never in `running`.
4. **Media emit ownership.**
   - **`media/generate.py`** owns `start`, `progress` and `end` for every
     generation, sync or job.
     - `jobs.run` passes `FXLLA_JOB_ID=<id>` in the env of the generator it
       spawns. generate.py then uses that as `id`, with `job: true`.
     - Without it, generate.py makes its own id, with `job: false`.
   - **`media/jobs.py`** emits an `end` only when generate.py could not:
     - cancel, with status `cancelled`
     - a reaped dead worker, with status `failed` and error "worker process
       died"
   - The reader keeps the last `end` per id.
   - **Argv parsing for model, steps, width and height** lives once, in
     generate.py, next to its existing argument parsing.
   - **The job `.log`.** jobs.run writes the generator's combined output to
     the job's `.log` instead of capturing it. The `error` field still gets
     the last 800 characters of that output, as today.
   - **Progress.** generate.py scans the backend's output in real time for
     progress, and passes the output through to its own stderr.
5. **Peak RSS.** generate.py measures `RUSAGE_CHILDREN` after a backend
   subprocess exits, or `RUSAGE_SELF` when the backend ran in-process.
   jobs.py does not measure RSS.
6. **kb.**
   - `rag/core.py`'s `Embedder` gains a boolean attribute `started`, true
     when this instance launched the server. `cmd_search` reads it for
     `embed_cold`.
   - The `rm` record is
     `{"kind":"rm","kb":str,"source":basename|None,"duration_ms":int,"ok":bool,"error":str|None}`.
7. **Graph and kb stats.**
   - `graph.stats.refs` is None unless an index record carries it.
   - When `kb.db` is missing, `bases` is `[]` and `db_bytes` is None. The
     read-only open must not create it.
8. **Where tests go.**
   - Reader unit tests: `gateway/test_telemetry.py`.
   - `/monitor` shape tests: `gateway/test_gateway_e2e.py`.
   - Each part touches only its own directory, plus that part's test
     module. The gateway reader part touches only `gateway/`.

## Left to the builder

- Exact regexes for progress, verified against real output.
- How the reader caches the file tails.
- The internal structure of each change, within these rules.
