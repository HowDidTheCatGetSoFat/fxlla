# kb, graph and media telemetry for /monitor

Date: 2026-11-27. Status: design, revision 2. Piece 2 of 4. Piece 1 is
`2026-11-26-gateway-telemetry-design.md`. The screen it feeds is
`2026-11-26-monitor-look.md`: the panels KB / RAG and Media, and kb rows in
Recent requests.

## Intent

The monitor must show what kb/RAG search, the code graph and media
generation are doing, as it already does for chat. Today none of these
record anything usable:

- **kb search and graph queries.** They run in short-lived processes: the
  CLI, or a subprocess that an MCP server spawns per call. No process
  writes a log.
- **Media jobs.** Each async job keeps a JSON record under
  `<store>/media/jobs/<id>.json`. The record has no progress, no peak RAM,
  and no model, steps or resolution fields. Synchronous runs (without
  `--async`) leave no record at all. Each generator's output is captured
  and thrown away, so every job log is 0 bytes.

## Approach

**kb and graph get new event logs.** Each process appends one JSON line
per event to `kb.jsonl` or `graph.jsonl` in the fxlla state directory.

**Media extends what already exists.**
- The job record becomes the one source of truth for every generation.
- It gains model, steps, size, peak RAM, and live progress.
- Synchronous runs write a record too.
- Nothing about media goes into a separate log. A separate log next to the
  job records would be a second source of truth that drifts. It would also
  duplicate the dead-worker detection that `jobs._reap` already does.

**The gateway reads all of it** and adds the sections to `GET /monitor`.

Rejected:
- **POST each event to the gateway.** The event is lost when the gateway is
  down, and kb and media would come to depend on the gateway.
- **The monitor reads the sources itself.** That duplicates logic, and
  `/monitor` stops being the single source.
- **A `media.jsonl` beside the job records.** It is a second source of
  truth, as explained above.

A side benefit: `fxlla media timings` and the MCP job status see
synchronous runs and the new fields for free.

## Part 0: the shared event helper (built before the others)

A new stdlib module, `lib/fxlla_events.py`:

- **`state_dir() -> str`.** It resolves, in order:
  1. `FXLLA_STATE_DIR`, a new override that tests use to isolate;
  2. `$XDG_STATE_HOME/fxlla`;
  3. `~/.local/state/fxlla`.

  On this Mac every process (CLI, MCP servers, gateway) resolves to
  `~/.local/state/fxlla`. `bin/fxlla serve` already forwards that path to
  the gateway as `FXLLA_STATE_DIR`.
- **`append(stream: str, record: dict) -> None`.**
  - `stream` is `kb` or `graph`.
  - It adds `ts`, the epoch float, when absent, and `caller`, from
    `os.environ.get("FXLLA_CALLER", "cli")`, when absent.
  - It writes one compact JSON line, `line + "\n"`, with a single
    `os.write` on an fd opened `O_WRONLY|O_APPEND|O_CREAT`, mode 0600.
    Concurrent processes therefore never interleave inside a line.
  - It never raises. It catches everything, and on failure writes at most
    one line to stderr.
  - When the file passes 5 MB, the writer that notices renames it to
    `<stream>.jsonl.1` (replacing any older one) and appends to a fresh
    file. A race there loses at most a few lines, which is acceptable.
- **`read_tail(stream: str, max_lines: int = 200) -> list[dict]`.**
  - It returns the newest parseable records, newest first.
  - It reads only the file's tail, by seeking from the end, and skips bad
    lines.
  - It does not read `.1`.
- **Tests and CI.**
  - Tests go in a new `lib/test_fxlla_events.py`. They cover a concurrent
    append from multiple processes, rotation, a bad line, a missing file,
    and a write that cannot fail loudly.
  - Part 0 is the only part that edits `.github/workflows/ci.yml` and
    AGENTS.md, to add `lib.test_fxlla_events`.
- **Import path.** Callers outside `lib/` add the repo's `lib` directory to
  `sys.path`, computing the repo root from their own `__file__`.
- **The caller convention.** `rag/rag_mcp.py`, `graph/graph_mcp.py` and
  `media/media_mcp.py` set `FXLLA_CALLER=mcp` in the env of each subprocess
  they spawn.

## Part 1: kb and graph writers

**kb**, in `rag/core.py`:
- **search:**
  `{"kind":"search","kb":str,"hits":int,"top_score":float|None,"duration_ms":int,"embed_cold":bool,"ok":bool,"error":str|None}`
- **add:**
  `{"kind":"add","kb":str,"source":basename,"chunks_added":int,"duration_ms":int,"ok":bool,"error":str|None}`
- **rm:**
  `{"kind":"rm","kb":str,"source":basename|None,"duration_ms":int,"ok":bool,"error":str|None}`
- **embed_cold.** `Embedder` gains a boolean attribute `started`, which is
  true when this instance launched the embedding server. `cmd_search`
  reads it for `embed_cold`.

**graph**, in `graph/codegraph.py`:
- **query:**
  `{"kind":"query","query":"def"|"refs"|"callers"|"impact"|"unused","results":int,"duration_ms":int,"ok":bool,"error":str|None}`
- **index:**
  `{"kind":"index","repo":basename,"files":int|None,"defs":int|None,"refs":int|None,"duration_ms":int,"ok":bool,"error":str|None}`

**Both:**
- Records are written once per command, after it finishes, whether it
  succeeded or failed.
- `error` is the first line of the error, at most 200 characters.
- Nothing printed to the caller changes, and exit codes do not change.

## Part 2: media job records

### The record

The job record (`media/jobs.py`) gains these fields. The existing ones
keep their meaning:

- `job`: bool. False for synchronous runs.
- `media`: `"image"`, `"video"`, `"voice"`, `"edit"` or `"upscale"`. This
  is today's `kind`, which stays as is.
- `model`, `steps`, `width`, `height`: each an int or str, or None when
  not determinable. Never guess a default.
  - They are parsed once, in `media/generate.py`, next to its existing
    argument parsing.
  - They are written into the record when the generation starts.
- `peak_rss_mb`: int or None.
  - It is measured in generate.py with
    `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` after a backend
    subprocess exits, or with `RUSAGE_SELF` when the backend ran
    in-process.
  - Normalize the units: macOS reports bytes and Linux reports KB.
- `progress`: `{"step": int, "total": int, "at": epoch}` or None.
  - generate.py updates it at most once per second, and only when the step
    changed.
- `caller`: `cli` or `mcp`, from `FXLLA_CALLER`.

### Who writes what

- **jobs.run** passes `FXLLA_JOB_ID=<id>` in the env of the generate.py
  process it spawns.
- **generate.py**, when `FXLLA_JOB_ID` is set, updates that record. It
  writes the new fields at start, then progress, then `peak_rss_mb` at the
  end.
  - Use the module's existing atomic tmp-and-rename write.
  - Re-read before each write, so status changes that `jobs.py` makes are
    not lost. Lost is only a field or two from a one-second progress
    update, which is acceptable.
- **A synchronous run**, without `FXLLA_JOB_ID`, is created by generate.py
  itself, through a `jobs.py` helper:
  - `job: false` and `status: "running"`
  - `pid` set to its own pid
  - at the end, `done` or `failed`, with `finished`, `output` and `error`
  - It runs under the media lock exactly as today: no change to
    concurrency.
- **Status transitions stay where they are**, in jobs.py, for async jobs.

### Output and progress capture (behavior change)

- **jobs.run** writes the generator's combined stdout and stderr to the
  job's `.log` file, truncated at job start, instead of capturing and
  discarding them.
- **The `error` field** still gets the last 800 characters of that output
  on failure, as today.
- **generate.py** reads each backend subprocess's output in real time, in
  a reader thread or an equivalent.
  - It splits on both `\n` and `\r`. tqdm-style progress bars redraw one
    line with `\r`.
  - It passes the output through unchanged.
  - It matches progress with tqdm `N/M` and `step N/M` patterns, and
    writes `progress`.
  - Verify the patterns against the real output of mflux, mlx-audio and
    ltx on this Mac. A tool that prints no parseable progress gets no
    progress: do not invent one.
- **These must not change:** the `warnings` field and its content, the
  output path, the exit codes, and the JSON that CLI and MCP callers get
  back.

### Pruning

Existing pruning applies to synchronous records too.

## Part 3: the reader, in the gateway

`/monitor` gains three top-level keys. `schema_version` stays 1.

- **`kb`:**
  - `recent`: the last 50 records from `read_tail("kb")`
  - `today`: `{searches, adds, avg_search_ms, cold_starts}`
  - `bases`: `[{"kb", "sources", "chunks"}]`, plus `db_bytes`
- **`graph`:**
  - `recent`: the last 50 records
  - `today`: `{queries, indexes, avg_query_ms}`
  - `stats`: `{files, defs, refs, indexed_at}`
- **`media`:** built from `<store>/media/jobs/*.json`.
  - `running`: the records with status `running`. A record whose pid is no
    longer alive (`os.kill(pid, 0)` fails) is reported as `died`, in
    `recent` and not in `running`. This uses the same rule as
    `jobs._reap`, without writing to the file.
    - Each running item carries `elapsed_s` and `progress`.
    - It also carries `eta_s`, which is `elapsed / step * (total - step)`,
      or None without progress.
  - `queued`: the records with status `queued`.
  - `recent`: the last 20 finished records.
  - `today`: `{jobs, done, failed, gpu_s}`, where `gpu_s` is the sum of
    `finished - started` over today's finished records.
- **Field allowlist (privacy).** A media item exposes only these fields:
  - `id`, `job`, `media`, `model`, `steps`, `width`, `height`
  - `status`, `created`, `started`, `finished`
  - `peak_rss_mb`, `progress`, `caller`
  - an `error` truncated to its first line, at most 200 characters

  Never `argv`, `summary`, `output`, `log`, or any path. Those carry the
  prompt and home-directory paths. `/monitor` is unauthenticated on a LAN
  bind. Test this with a job whose prompt is a distinctive string.
- **"Today"** is the local date. The kb and graph aggregates come from
  `read_tail(stream, 5000)`, cached for 2 s. With more than 5000 records a
  day, they cover the newest 5000; document this.
- **kb `bases`** comes from one read-only SQLite query on
  `<store>/kb/kb.db`: `file:...?mode=ro` URI, 1 s busy timeout, cached
  10 s.
  - It never creates the db.
  - When the db is missing, `bases` is `[]` and `db_bytes` is None.
- **`graph.stats`** must not open kuzu, because of the writer lock and the
  dependency.
  - `indexed_at` is the `ts` of the newest successful `index` record (None
    when there is none). The kuzu mtime is not used: any query bumps it.
  - `files`, `defs` and `refs` come from the newest successful `index`
    record.
- **The media jobs directory** is listed at most once per 2 s (cached).
  There are about 20 files.
- **Failure.** Any reader failure leaves that section's fields None or
  empty. It never fails `/monitor`.
- **Tests.** Reader unit tests go in `gateway/test_telemetry.py`, and the
  `/monitor` shape and privacy tests in `gateway/test_gateway_e2e.py`. This
  part touches only `gateway/`.

## Anti-goals

- **A tool failing or slowing because of telemetry.**
  - Every append is fire-and-forget, one syscall, with no subprocess and no
    network.
  - A job-record update failure never fails a generation.
- **Text in any telemetry output.** No query, prompt, chunk, file content
  or home path, either in the logs or in `/monitor`.
- **Changing what an MCP tool or CLI command returns** to its caller.

## Constraints

- Standard library only, in the writers and in the reader.
- English and ASCII only, per AGENTS.md.
- Tests must pass on the Linux CI runner, so no test needs mflux,
  mlx-audio, ltx or kuzu. Progress parsing is tested on captured text.
- **Ownership.**
  - Each part touches only its own directory, plus that part's test
    module:
    - Part 0: `lib/`, ci.yml and AGENTS.md
    - Part 1: `rag/`, `graph/`, and the `FXLLA_CALLER` line in their MCP
      servers
    - Part 2: `media/`
    - Part 3: `gateway/`
  - No part edits CHANGELOG.md or the top-level docs. The controller writes
    those at merge.
- **Order.** Part 0 is built and reviewed first. Parts 1, 2 and 3 then run
  in parallel, each in its own git worktree branched from Part 0. Part 3
  builds against the record shapes written here, using its own fixture
  files.

## Left to the builder

- The exact progress regexes, verified against real output.
- How the reader caches.
- The internal structure of each change, within these rules.
