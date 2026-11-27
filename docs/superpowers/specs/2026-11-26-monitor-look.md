# fxlla monitor: the approved look

Date: 2026-11-26. Status: look approved by the user. The monitor itself is
the last of four pieces:
1. gateway telemetry (`2026-11-26-gateway-telemetry-design.md`)
2. kb/RAG and media job telemetry
3. observers for external engines (Strata, Ollama, ComfyUI, LM Studio)
4. the monitor

Each piece gets its own design before it is built.

## Intent

The user wants a live terminal monitor that looks better than Strata's
`strata monitor`. Strata's screen is a text table. The user asked first for
it to look fantastic, then above all for it to show everything they need to
know. Information density wins over spectacle.

## Choice

Three directions were mocked up in the browser:
- A: a cockpit of panels with sparklines
- B: an editorial layout of large numbers and no boxes
- C: a neon layout with one full-width chart

The user picked A as the most informative, with everything visible. The big
time-series chart from C moves to its own tab (8, Velocidad). From B, it
keeps the hierarchy: dim labels, bright values.

## Home screen (tab 1, Resumen)

This is for a terminal of about 130 columns. On a narrower terminal, panels
stack instead of sitting side by side.

- **Header.** The tabs, then the gateway address, uptime, and a live marker.
  Tabs: 1 Resumen, 2 Pedidos, 3 Modelos, 4 Media, 5 KB, 6 Externos,
  7 Eventos, 8 Velocidad.
- **Machine.** RAM used and total, GPU percent and GPU in-use memory, swap
  with pressure level, and free space on the model store disk. Each has a
  color meter and a braille sparkline of the last minutes.
- **Speed.**
  - Answer tok/s, prompt-reading tok/s, TTFT, and prompt-cache hit percent.
  - Each has a sparkline and a peak: p95 for TTFT, and tokens saved for the
    cache.
- **fxlla loaded models.** One row per resident backend:
  - engine, phase (generating or reading), and live tok/s
  - output so far against `max_tokens`, and MTP acceptance with a meter
  - context used against the window, RSS, and client
  - the keep-warm countdown when idle
  - The panel header shows budget, queue length, and keep-warm.
- **Other engines.** Strata (model, tok/s, in-flight requests, context,
  total requests, memory), Ollama (loaded model or none, installed models,
  request count), ComfyUI, and LM Studio. Each shows up or down.
- **Media jobs.**
  - The running job, with step N/M, a progress bar, elapsed time, ETA, and
    RAM.
  - Queued jobs, recent successes and failures with the reason, and a daily
    summary.
- **Recent requests from all engines.**
  - Columns: time, engine, model, client, input, cached, output, reasoning,
    prompt time, generation time, total time, and finish.
  - MTP acceptance shows when present. kb searches appear here too.
- **KB / RAG.** Each base with its chunks, searches today, and latency. Also
  the code graph status and the embedding model.
- **Events.** Loaded, evicted, died and reloaded, and idle-unloaded, each
  with a time.
- **Footer.** The day's totals and the share by client app, followed by the
  key hints.

## Keys

- 1-8: tabs
- up/down: select
- enter: detail
- /: filter
- space: pause
- q: quit
- u: unload the selected model. The user saw this on the mockup and did not
  object. It is the only action the monitor takes, and the monitor design may
  still revisit it.

## Rendering notes

- 24-bit color, rounded box drawing, braille sparklines, and block meters,
  degrading to 256 colors and ASCII when the terminal cannot show them.
- The mockup HTML is kept locally under `.superpowers/brainstorm/`, which git
  does not track.
