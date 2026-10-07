# fxlla media

Local image, video, and speech generation, driven through `fxlla media`. The
CLI is a thin wrapper over the Apple Silicon toolchains; fxlla keeps the
friendly names, sane defaults, and output validation in one place.

## Pieces

- `generate.py` - the wrapper and CLI. Images go through mflux (one CLI per
  model family: `mflux-generate`, `mflux-generate-z-image-turbo`, ...); video
  goes through `ltx-2-mlx` (LTX-2.3); speech goes through mlx-audio (Chatterbox).
  Output lands under `<FXLLA_STORE>/media` and every render is validated (a zero
  exit code is not proof of a real file).
- `voice_backend.py` - the speech worker. It runs under a separate interpreter
  (`FXLLA_VOICE_PYTHON`) that has `mlx-audio` installed, so fxlla itself never
  imports it; it loads Chatterbox and writes a 24 kHz mono WAV.
- `media_mcp.py` - a minimal stdio MCP server exposing `generate_image`,
  `generate_video`, `generate_speech`, `edit_image`, and `upscale_image` so opencode and Claude Code can render
  media as a tool call.

Driven through the CLI:
`fxlla media image|video|voice|edit|upscale|models`, `fxlla media mcp`,
`fxlla media wire-opencode`.

## Images (mflux)

```sh
fxlla media models                       # list the supported image models
fxlla media image "a red sailboat at sunset" --model z-image-turbo \
    --width 1024 --height 1024 --seed 42 -q 8
```

`z-image-turbo` is the fast default. Common flags: `-q {3,4,5,6,8}` (quantize),
`--steps`, `--seed`, `--width`/`--height` or `--aspect` (fxlla resolves a
ratio to a one-megapixel width and height; mflux has no such flag), `--low-ram`,
`--metadata` (writes a sidecar JSON), `-o <path>`.

## Video (ltx-2-mlx)

```sh
fxlla media video "a red sailboat sailing at sunset, gentle waves" \
    --frames 49 --width 768 --height 512 --frame-rate 24 --seed 42
```

`ltx-2-mlx generate` requires exactly one quality stage; the wrapper defaults to
`--distilled` (the fast, verified path) and passes the mandatory frame rate
(the model was trained at 24 fps). Override the stage with
`--stage {distilled,one-stage,two-stage,two-stages-hq}`.

## Voice (mlx-audio / Chatterbox)

```sh
fxlla media voice "Hello from a local voice." --lang en --ref reference-voice.wav
```

The Chatterbox multilingual model has no built-in speaker conditionals, so a
reference voice wav is required (`--ref` or `FXLLA_VOICE_REF`); it sets the
timbre and accent. Because speech runs under `FXLLA_VOICE_PYTHON`, that
interpreter must have `mlx-audio` installed (a project venv is the usual home).

## Configuration

- `FXLLA_VIDEO_BIN` - path to the `ltx-2-mlx` binary. It commonly lives in a
  project virtualenv rather than on PATH; set it in `~/.config/fxlla/config.env`.
- `FXLLA_VOICE_PYTHON` - interpreter with `mlx-audio` installed (a project venv).
- `FXLLA_VOICE_REF` - reference voice wav for speech (sets the timbre).
- `FXLLA_VOICE_MODEL` / `FXLLA_VOICE_LANG` - TTS model and default language.
- `FXLLA_MEDIA_HF_HOME` - Hugging Face cache holding the diffusion/audio weights.
  Colon separated for several caches: the first takes new downloads, all are
  searched. `HF_HOME` accepts only one path, so each render is handed the root
  that actually holds the model it needs - pointing it at the wrong one does
  not fail, it silently downloads tens of gigabytes again.
- `FXLLA_MEDIA_MODEL` - default image model (default `z-image-turbo`).
- `FXLLA_MEDIA_OUT` - output directory (default `<FXLLA_STORE>/media`).
- `FXLLA_MEDIA_SKIP_QUALITY` - set to 1 to accept output that fails the content
  checks in `quality.py`. Per call, pass `--skip-quality` instead (an MCP client
  cannot set an environment variable on an already-running server).

A generator downloads its weights from Hugging Face on first use, which is tens of
gigabytes. `media/weights.py` checks the catalog against the cache before the
toolchain starts and refuses unless the transfer was authorized with `--yes` or
`FXLLA_ASSUME_YES=1`; `fxlla pull media:<alias>` fetches them deliberately. The
check lives in `generate.py` rather than in `bin/fxlla` because a background job
and an MCP tool call both re-invoke it directly.

`ffprobe` is optional. When it is on PATH, video output is checked for a usable
stream; without it the video checks are skipped rather than guessed, so the same
render can be accepted on one machine and flagged on another. `fxlla doctor`
reports which case you are in.

## Memory coordination

Media models share unified memory with the gateway's resident LLMs. Before each
job the wrapper asks a running gateway to unload its models
(`POST /admin/unload`), so a heavy render does not push past the GPU wired limit
and OOM; the gateway reloads on the next request. It is best-effort - no gateway
means nothing to free. Pass `--keep-models` (or set `FXLLA_MEDIA_KEEP_MODELS=1`)
for a small job that fits alongside the resident model.

## Notes and limits

- Generation is synchronous and can take tens of seconds (video longer). The MCP
  call blocks until the file is written.
- `mflux-generate-qwen-layered` (image to RGBA layers) is not a text-to-image
  sampler and is intentionally not in the model list.
- The external disks that hold the weight caches must be mounted; a "model not
  found" error is usually an unmounted volume.
