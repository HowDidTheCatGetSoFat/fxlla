#!/usr/bin/env python3
"""fxlla media: local image, video, and voice generation.

Images go through the mflux toolchain (mflux-generate, mflux-generate-z-image-turbo,
...), mapping a friendly model name to the right CLI. Video goes through
ltx-2-mlx (LTX-2.3). Both write under <FXLLA_STORE>/media by default and the
produced file is validated, since a zero exit code is not proof of a real render.

Config via environment:
  FXLLA_MEDIA_HF_HOME   HF cache(s) for the diffusion weights, ':' separated.
                        The first takes new downloads, all are searched, and
                        each render is exported the one holding what it needs
                        (HF_HOME accepts a single path).
  FXLLA_MEDIA_MODEL     default image model (default z-image-turbo)
  FXLLA_MEDIA_OUT       output directory (default <FXLLA_STORE>/media)
  FXLLA_VIDEO_BIN       path to the ltx-2-mlx binary (default: ltx-2-mlx on PATH)
  FXLLA_LORA_DIRS       colon-separated directories holding LoRAs
  FXLLA_MFLUX_BIN_DIR   directory holding the mflux CLIs (default: PATH)
  FXLLA_EDIT_BIN        path to the image-edit CLI (wins over FXLLA_MFLUX_BIN_DIR)
  FXLLA_UPSCALE_BIN     path to the upscale CLI (wins over FXLLA_MFLUX_BIN_DIR)
  FXLLA_MEDIA_KEEP_MODELS  set to keep the gateway's resident models loaded
  FXLLA_VOICE_PYTHON    interpreter that has mlx-audio (default: the uv tool
                        venv when present, else python3)
  FXLLA_VOICE_MODEL     TTS model (default YUGOROU/Chatterbox-Multilingual-MLX-4bit)
  FXLLA_VOICE_REF       reference voice wav for TTS (--ref overrides)
  FXLLA_VOICE_LANG      TTS language (default en)
  FXLLA_HOST, FXLLA_PORT  gateway asked to free its models (default 127.0.0.1:8080)

Usage (normally driven via `fxlla media`):
  generate.py image "<prompt>" [--model z-image-turbo] [--steps N] [--seed N]
                              [--width N --height N | --aspect 1:1] [-q {3,4,5,6,8}]
                              [--low-ram] [--metadata] [-o path]
  generate.py video "<prompt>" [--stage distilled] [--frames N | --seconds S]
                              [--frame-rate R] [--width N] [--height N] [--seed N]
                              [--model NAME] [--low-ram] [-o path]
  generate.py voice "<text>"  [--ref voice.wav] [--lang en] [--model NAME]
                              [--speed X] [-o path]
  generate.py edit "<prompt>" --image path [--seed N] [-q {3,4,5,6,8}] [-o path]
  generate.py upscale --image path [--scale 2x] [-o path]
  (every generator also takes --async, --keep-models, --skip-quality, --yes)
  generate.py models          list the supported image models
  generate.py voice-python    print the resolved mlx-audio interpreter
  generate.py jobs [--prune]  list background jobs
  generate.py job <id>        show one background job
  generate.py cancel <id>     cancel a running job
"""
import argparse
import base64
import glob
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.request

import jobs  # background media jobs (submit, poll, cancel)
import quality  # perceptual checks on the produced file
import weights  # consent for the weights a render would download

# Set by --skip-quality. An MCP client cannot set an environment variable on an
# already-running server, so the flag is the only override available there.
SKIP_QUALITY = False

STORE = os.environ.get("FXLLA_STORE", "")
OUT_DIR = os.environ.get("FXLLA_MEDIA_OUT") or os.path.join(STORE, "media")
DEFAULT_MODEL = os.environ.get("FXLLA_MEDIA_MODEL", "z-image-turbo")
DEFAULT_QUANTIZE = 8
VIDEO_BIN = os.environ.get("FXLLA_VIDEO_BIN", "ltx-2-mlx")
# Instruction-based image edit and diffusion upscale are separate mflux CLIs.
# One directory for the whole mflux family (image models, edit, upscale):
# image alone is eight per-model CLIs, so a single-binary knob cannot cover
# it. The specific knobs below still win for edit and upscale.
MFLUX_BIN_DIR = os.environ.get("FXLLA_MFLUX_BIN_DIR", "")
EDIT_BIN = os.environ.get("FXLLA_EDIT_BIN", "")
UPSCALE_BIN = os.environ.get("FXLLA_UPSCALE_BIN", "")


def depth_map_path(output):
    """Where mflux writes the derived depth map: "<base>_depth_map<ext>",
    beside the image (mflux/callbacks/instances/depth_saver.py).

    Computed and reported rather than left implicit, because the point of
    --save-depth-map is feeding the map into a later control step, and a
    caller that cannot find the file cannot chain anything."""
    base, ext = os.path.splitext(output)
    return "%s_depth_map%s" % (base, ext or ".png")


def resolve_output(output, kind, ext):
    """Where a render should land: OUT_DIR when nothing was asked for, a
    generated name INSIDE the path when it is an existing directory, and the
    path itself otherwise.

    Passing a directory is the obvious thing to try ("put it in ~/Downloads")
    and it used to reach the toolchain as a filename, which failed deep in a
    backend with a message about writing to a directory. Naming the file here
    is what a caller would have had to do by hand."""
    if not output:
        return os.path.join(OUT_DIR, "fxlla-%s-%d.%s" % (kind, int(time.time()), ext))
    # "~/Downloads" is what a person says and what an agent passes on. Nothing
    # here runs through a shell, so an unexpanded ~ would become a directory of
    # that literal name beside the working directory.
    output = os.path.expanduser(output)
    # A trailing separator means a directory was intended even if it does not
    # exist yet, which is worth creating rather than treating as a filename.
    if output.endswith(os.sep) and not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
    if os.path.isdir(output):
        return os.path.join(output, "fxlla-%s-%d.%s" % (kind, int(time.time()), ext))
    return output


def split_ref(ref):
    """A reference and its numeric modifiers, as a list of strings.

    Accepts "path,0.8" or ["path", "0.8"]. The comma form is what the CLI and
    the MCP use, for two reasons: argparse `nargs="+"` swallows the following
    positional (`--lora x "a cat"` ate the prompt), and a JSON array of plain
    strings cannot express a pair, so a scale sent through MCP would have
    arrived as a second, bogus reference."""
    if isinstance(ref, (list, tuple)):
        parts = [str(p) for p in ref]
    else:
        parts = [p.strip() for p in str(ref).split(",")]
    return [p for p in parts if p]


def mflux_cli(name):
    """Resolve one mflux-family CLI: FXLLA_MFLUX_BIN_DIR when set, PATH
    otherwise. A set directory missing the binary is an error worth naming -
    silently falling back to PATH would run a different install than the one
    the user pointed at, which is the quiet failure the knob exists to
    prevent."""
    if not MFLUX_BIN_DIR:
        return name
    path = os.path.join(MFLUX_BIN_DIR, name)
    if not os.path.isfile(path):
        raise ValueError(
            "FXLLA_MFLUX_BIN_DIR is set (%s) but '%s' is not in it"
            % (MFLUX_BIN_DIR, name))
    return path

def backend_capabilities(timeout_s=300):
    """mflux's own dump of which options each CLI honours, or None.

    The catalog below transcribes this by hand, and a transcription rots: over
    one week four flags were declared that the backend accepts and silently
    discards, each found by reading mflux's source after the fact. mflux
    0.19.0 publishes the contract instead - status is honored, ignored or
    conditional, read from the same constants the runtime warnings use - so the
    table can be checked rather than trusted.

    None when the backend is not installed or is too old to have the dump,
    which is the normal case on a machine that has never run `fxlla setup
    --media`; a caller treats that as "cannot check", never as "agrees"."""
    try:
        binary = mflux_cli("mflux-generate")
    except ValueError:
        return None  # a misconfigured bin dir is "cannot check", not a failure
    path = shutil.which(binary) or (binary if os.path.isfile(binary) else None)
    if not path:
        return None
    # The console script's shebang names the interpreter of the venv mflux is
    # installed in. Going through it runs the module even when that venv's
    # entry points predate the dump, which is exactly the case after an
    # in-place upgrade of an editable install.
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            first = fh.readline()
    except OSError:
        return None
    python = first[2:].strip() if first.startswith("#!") else ""
    if not os.path.isfile(python):
        return None
    try:
        proc = subprocess.run([python, "-m", "mflux.cli.capabilities"],
                              capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


# Media generation and the gateway's resident LLMs share unified memory. Before
# a job, ask a running gateway to free its models so the render has headroom;
# the gateway reloads on demand afterward. FXLLA_MEDIA_KEEP_MODELS opts out.
GATEWAY_HOST = os.environ.get("FXLLA_HOST", "127.0.0.1")
GATEWAY_PORT = os.environ.get("FXLLA_PORT", "8080")
KEEP_MODELS = os.environ.get("FXLLA_MEDIA_KEEP_MODELS", "") not in ("", "0", "false")

# Friendly name -> the mflux CLI and defaults. `base_model` is only needed
# for the multi-model `mflux-generate` binary (FLUX.1). `steps` is a sane
# default for the fast distilled models; None leaves the CLI's own default.
MODELS_CONF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "media-models.conf")


def load_models(path=None):
    """Image models from the catalog: alias -> {cli, base_model, steps, caps}.

    `caps` is what the CLI actually accepts, recorded from its own --help when
    the catalog was written. It is load-bearing rather than documentation: a
    flag a model does not support is refused by name here instead of being
    passed through to fail deep inside the backend, and the models genuinely
    differ (boogu and fibo take no LoRA at all).
    """
    models = {}
    try:
        with open(path or MODELS_CONF, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 5:
                    continue
                alias, cli, base, steps, caps = parts[:5]
                models[alias] = {
                    "cli": cli,
                    "base_model": base or None,
                    "steps": int(steps) if steps.isdigit() else None,
                    # Each token stripped: a catalog line written "lora, negative"
                    # would otherwise carry a leading space and match
                    # nothing, silently disabling the flag.
                    "caps": set(c.strip() for c in caps.split(",") if c.strip()),
                    "note": parts[5] if len(parts) > 5 else "",
                }
    except OSError:
        pass
    return models


MODELS = load_models()


# Models whose backend REJECTS a size off its grid, and the grid. Not a
# universal rule: most latent creators divide by 8 and silently accept
# anything, so a blanket multiple-of-16 check would refuse sizes that work.
# Ideogram 4 raises (mflux Ideogram4LatentCreator.DIMENSION_STEP = 16), and it
# raises after the weights are resident - a 1080-wide poster died minutes in,
# which is what this catches in a millisecond.
_DIM_STEP = {"ideogram4": 16}

# Ideogram 4's sampler presets, which fix the step count AND the guidance
# schedule - hence no --steps or --guidance on this model. Names and counts
# from mflux's Ideogram4Scheduler.PRESETS.
IDEOGRAM_PRESETS = {"V4_TURBO_12": 12, "V4_DEFAULT_20": 20, "V4_QUALITY_48": 48}

# Steps each model actually runs when nothing is passed, from mflux's
# MODEL_INFERENCE_STEPS. Reported, never sent: this is the cost signal a caller
# needs to choose between models, and without it an eight-minute render and a
# one-minute render look identical from the outside. Where fxlla pins a value
# in the catalog, that pin wins and is what gets reported.
_MFLUX_DEFAULT_STEPS = {
    "dev": 25, "schnell": 4, "krea2": 8, "qwen": 20, "fibo": 50,
    "z-image": 50, "z-image-turbo": 9, "z-controlnet": 8, "controlnet": 25,
    "depth": 25, "ernie": 50, "ernie-turbo": 8, "boogu": 4,
    "flux2-klein": 4, "ideogram4": 20, "qwen-2.1": 40, "ming": 12,
}


# --aspect is fxlla's, not mflux's: upstream mflux has no such flag, so a ratio
# resolves here to the width and height of a one-megapixel image, each rounded
# up to a multiple of 16 - a size every model's latent grid accepts, Ideogram 4
# included.
ASPECT_RATIOS = ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "18:9",
                 "9:18", "21:9", "9:21")
ASPECT_PIXELS = 1024 * 1024


def aspect_size(aspect):
    """Width and height for an aspect ratio such as "16:9"."""
    if aspect not in ASPECT_RATIOS:
        raise ValueError("unknown aspect %r; use one of %s"
                         % (aspect, ", ".join(ASPECT_RATIOS)))
    w_ratio, h_ratio = (int(n) for n in aspect.split(":"))
    ratio = w_ratio / h_ratio
    height = (ASPECT_PIXELS / ratio) ** 0.5
    return (int(math.ceil(height * ratio / 16)) * 16,
            int(math.ceil(height / 16)) * 16)


# PiD's LQ conditioning was distilled on latents noised at sigma ~ U[0.0, 0.8];
# past that it has never seen the input. mflux names the bound itself.
PID_MAX_DEGRADE_SIGMA = 0.8


def _check_pid_sigma(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("pid_degrade_sigma must be a number, got %r" % (value,))
    if not 0.0 <= number <= PID_MAX_DEGRADE_SIGMA:
        raise ValueError(
            "pid_degrade_sigma must be between 0 and %s (the range PiD was "
            "distilled on), got %s" % (PID_MAX_DEGRADE_SIGMA, value))


def _check_strength(value):
    """Reject a strength outside 0..1 here rather than deep in the backend.

    The real failure it catches is a caller reading "strength" as a percentage
    and sending 60. The legitimate case it must not catch is either end: 0
    keeps the init image and 1 ignores it, and both are things a caller means."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("strength must be a number between 0 and 1, got %r"
                         % (value,))
    if not 0.0 <= number <= 1.0:
        raise ValueError(
            "strength must be between 0 and 1 (how much of the init image is "
            "kept), got %s. mflux defaults to 0.4; a refining pass is usually "
            "0.3-0.5." % value)


def _check_dim_step(model_name, width, height):
    step = _DIM_STEP.get(model_name)
    if not step:
        return
    for label, value in (("width", width), ("height", height)):
        if value and value % step:
            raise ValueError(
                "%s needs %s to be a multiple of %d, got %d: use %d or %d"
                % (model_name, label, step, value,
                   value - value % step, value - value % step + step))


def _require_cap(spec, cap, flag, model_name=""):
    """Refuse a flag the chosen model cannot take, naming both.

    The alternative is passing it anyway: mflux exits with its own usage error
    after the caller has already waited, and nothing says which model was the
    problem or which models would have worked."""
    if cap in spec.get("caps", ()):
        return
    able = sorted(a for a, s in MODELS.items() if cap in s.get("caps", ()))
    raise ValueError(
        "model '%s' does not support %s. Models that do: %s"
        % (model_name or spec.get("cli", "?"), flag,
           ", ".join(able) if able else "(none)"))

# LTX-2.3 `generate` requires exactly one quality stage; distilled is the fast,
# verified default. frame-rate is mandatory (the model was trained at 24).
VIDEO_STAGES = ("distilled", "one-stage", "two-stage", "two-stages-hq")
DEFAULT_STAGE = "distilled"
DEFAULT_FRAME_RATE = 24

# What each stage costs and which step knob moves it, quoted from ltx-2-mlx's
# own `generate --help`. Video had no cost surface at all: the only thing a
# caller could see was "distilled, the fast path", so the pipeline that decides
# whether a clip takes one minute or twenty was picked blind, and the step
# counts underneath it were not reachable from fxlla at all.
VIDEO_STAGE_INFO = {
    "distilled": {
        "steps_flag": "stage1_steps", "stage1_steps": 30, "stage2_steps": 3,
        "note": "Half-res distilled + upscale + distilled refine, no CFG. "
                "The fast default."},
    "one-stage": {
        "steps_flag": "steps", "steps": 8,
        "note": "Dev model + CFG at full resolution, no upscaler. Better than "
                "distilled at small sizes, slower than two-stage at large."},
    "two-stage": {
        "steps_flag": "stage1_steps", "stage1_steps": 30, "stage2_steps": 3,
        "note": "Dev + CFG at half-res, upscale, distilled refine. Needs the "
                "q8 model."},
    "two-stages-hq": {
        "steps_flag": "stage1_steps", "stage1_steps": 15, "stage2_steps": 3,
        "note": "HQ two-stage (res_2s sampler for stage 1). The slowest."},
}

# Voice runs through a separate interpreter (FXLLA_VOICE_PYTHON) that has
# mlx-audio installed, so fxlla itself never imports it. Chatterbox ships no
# speaker conditionals, so a reference voice wav is required.
VOICE_PYTHON = os.environ.get("FXLLA_VOICE_PYTHON", "python3")


def resolved_voice_python():
    """The interpreter that actually has mlx-audio.

    FXLLA_VOICE_PYTHON always wins. Otherwise `fxlla setup --media` installs
    mlx-audio as a uv tool, and that venv's python is the one interpreter
    guaranteed to import it - so it is the default when present. The bare
    python3 fallback keeps the old behavior for hand-rolled venvs on PATH.
    Doctor and setup ask THIS function (via the voice-python subcommand)
    instead of re-implementing the resolution in shell."""
    explicit = os.environ.get("FXLLA_VOICE_PYTHON")
    if explicit:
        return explicit
    try:
        root = subprocess.run(["uv", "tool", "dir"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
        candidate = os.path.join(root, "mlx-audio", "bin", "python")
        if root and os.path.isfile(candidate):
            return candidate
    except Exception:
        pass
    return "python3"
VOICE_MODEL = os.environ.get("FXLLA_VOICE_MODEL", "YUGOROU/Chatterbox-Multilingual-MLX-4bit")
VOICE_REF = os.environ.get("FXLLA_VOICE_REF", "")
VOICE_LANG = os.environ.get("FXLLA_VOICE_LANG", "en")
VOICE_BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_backend.py")

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _lora_repo_ids(loras):
    """The --lora values that name a Hugging Face repo rather than a file.

    Same test build_command uses: something that names a file (extension,
    absolute, or explicitly relative) is a path; anything else with a slash is
    a repo id that mflux will resolve out of the cache.
    """
    ids = []
    for ref in loras or []:
        parts = split_ref(ref)
        if not parts:
            continue
        path = parts[0]
        if (path.endswith((".safetensors", ".ckpt"))
                or os.path.isabs(path)
                or path.startswith(("./", "../", "~"))):
            continue
        if "/" in path:
            ids.append(path)
    return ids


def _env(kind=None, model=None, extras=(), repos=()):
    """Environment for a backend, with HF_HOME resolved to ONE cache root.

    FXLLA_MEDIA_HF_HOME may list several roots. Forwarding the list verbatim
    would set HF_HOME to a path that does not exist, and the toolchain would
    quietly re-download everything; so resolve it to the root that holds what
    this render needs. Callers that know their kind and model say so - the
    default is the first root, which is where new downloads go anyway.
    """
    env = dict(os.environ)
    if not os.environ.get("FXLLA_MEDIA_HF_HOME"):
        return env
    root, unreachable = weights.hf_home_for(kind, model, extras, repos)
    env["HF_HOME"] = root
    if unreachable:
        # Say it rather than let a surprise re-download of tens of GB be the
        # first sign. See `fxlla doctor` for the same warning at rest.
        sys.stderr.write(
            "warning: these weights are cached under a different "
            "FXLLA_MEDIA_HF_HOME root than the one this render uses (%s), so "
            "they will be fetched again: %s\n" % (root, ", ".join(unreachable)))
    return env


# The catalog alias that can read an image. Every other model here is text-only,
# and `mlx_lm.server` refuses image content outright, so this one is served by
# llama-server with a multimodal projector.
VISION_MODEL = os.environ.get("FXLLA_VISION_MODEL", "vision")
VISION_QUESTION = ("Describe this image. Be specific and concrete about what is "
                   "actually there, including any text or lettering, quoted "
                   "exactly. Do not speculate about what it might be for.")


def describe_image(path, question=None, model=None, timeout_s=600):
    """What is in this image, as text, from the local vision model.

    This is the inverse of everything else in this module: it reads an image
    rather than making one, and it exists because the thing driving fxlla often
    cannot see. A generation reports `done` when the file is written, and
    nothing below the caller can tell whether it is right - a real chain here
    removed an object from a photograph, left a visible ghost of it, and every
    check passed.

    Deliberately synchronous and outside the job queue: measured at 9 s, this
    is a read, not a render, and making the caller poll for it would be the
    ceremony without the reason. It goes through the gateway rather than
    spawning its own server so the model stays resident between calls."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise ValueError("image not found: %s" % path)
    facts = quality.image_facts(path)
    if not facts.get("width"):
        raise ValueError("not a readable PNG: %s" % path)
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode("ascii")
    body = {"model": model or VISION_MODEL, "max_tokens": 700,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": question or VISION_QUESTION},
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + data}}]}]}
    url = "http://%s:%s/v1/chat/completions" % (GATEWAY_HOST, GATEWAY_PORT)
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            answer = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise RuntimeError("the vision model refused this: %s" % detail)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "no gateway at %s:%s (%s). Start one with `fxlla serve`; the vision "
            "model loads on demand and stays resident between calls."
            % (GATEWAY_HOST, GATEWAY_PORT, exc.reason))
    choices = answer.get("choices") or []
    text = (choices[0].get("message", {}).get("content") or "").strip() if choices else ""
    if not text:
        raise RuntimeError("the vision model returned nothing")
    return text


def free_gpu(reason, keep=False):
    """Ask a running gateway to unload its resident models before a heavy job.

    Best-effort: if no gateway is up (connection refused) or the request fails,
    there is nothing to free and we proceed. Skipped when the caller opts out
    via keep=True or FXLLA_MEDIA_KEEP_MODELS."""
    if keep or KEEP_MODELS:
        return
    url = "http://%s:%s/admin/unload" % (GATEWAY_HOST, GATEWAY_PORT)
    req = urllib.request.Request(url, data=b"{}",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            info = json.loads(r.read() or b"{}")
    except Exception:
        return
    freed = info.get("unloaded") or []
    if freed:
        sys.stderr.write("[media] freed gateway models before %s: %s\n"
                         % (reason, ", ".join(freed)))


def build_command(spec, prompt, output, steps=None, seed=None, width=None,
                  height=None, aspect=None, quantize=None, low_ram=False,
                  metadata=False, negative=None, prompt_file=None, loras=None,
                  lora_style=None, init_image=None, model_name="", guidance=None,
                  controls=None, controlnet_strength=None, depth_image=None,
                  save_depth=False, preset=None, strength=None,
                  pid_decode=False, pid_degrade_sigma=None):
    """Assemble the mflux argument vector for one image generation.

    Optional flags are checked against the model's declared capabilities
    first, so an unsupported one is refused by name rather than handed to a
    backend that will reject it minutes later with its own usage text."""
    cmd = [mflux_cli(spec["cli"])]
    if spec.get("base_model"):
        # --model, not --base-model. mflux documents --base-model as "when
        # using a third-party huggingface model, explicitly specify whether
        # the base model is dev or schnell" - it qualifies a --model, it does
        # not stand in for one. Passing it alone used to be tolerated and now
        # yields a config whose model_name is None, which surfaces much later
        # as "No root_path and no download_url for component: vae" and sends
        # you looking at a cache that is perfectly fine. Naming the model the
        # documented way is both correct and immune to that.
        cmd += ["--model", spec["base_model"]]
    if prompt_file:
        # mflux declares these mutually exclusive ("--prompt-file: not allowed
        # with argument --prompt"), so sending both made every --prompt-file
        # render die in argparse.
        _require_cap(spec, "prompt-file", "--prompt-file", model_name)
        if not os.path.isfile(prompt_file):
            raise ValueError("prompt file not found: %s" % prompt_file)
        cmd += ["--prompt-file", prompt_file, "--output", output]
    else:
        cmd += ["--prompt", prompt, "--output", output]
    # Every optional flag goes through the capability gate, not just the ones
    # added most recently: a model whose CLI lacks --seed or a fixed quantize
    # is exactly what this catalog exists to express, and forwarding the flag
    # anyway is the deep-backend failure the gate prevents everywhere else.
    # A DEFAULT is adapted, an explicit request is validated. Erroring on a
    # flag the caller never named would make an unsupported default look like
    # their mistake; silently dropping one they DID name is the deep-backend
    # failure this gate exists to prevent.
    if quantize is not None:
        _require_cap(spec, "quantize", "--quantize", model_name)
        cmd += ["--quantize", str(quantize)]
    elif "quantize" in spec.get("caps", ()):
        cmd += ["--quantize", str(DEFAULT_QUANTIZE)]
    if negative:
        _require_cap(spec, "negative", "--negative-prompt", model_name)
        cmd += ["--negative-prompt", negative]
    # `--image PATH [STRENGTH]`, not the deprecated `--image-path`. The
    # strength is the whole point: it is how much of the init image survives,
    # and without it a second pass over a first one either changes nothing or
    # destroys it. That knob is what makes generate-small-then-refine-large
    # possible at all, and it was unreachable while fxlla passed the flag that
    # cannot carry it.
    if init_image:
        # Two spellings, and which one a CLI has is not uniform: the depth model
        # never gained mflux's newer `--image PATH [STRENGTH]` and still takes
        # only the deprecated `--image-path`, so emitting the new flag there
        # died in argparse. Verified per CLI from mflux's own capability dump.
        legacy = "init-image-path" in spec.get("caps", ())
        _require_cap(spec, "init-image-path" if legacy else "init-image",
                     "--image-path" if legacy else "--image", model_name)
        parts = split_ref(init_image)
        path = os.path.expanduser(parts[0])
        if not os.path.isfile(path):
            raise ValueError("init image not found: %s" % parts[0])
        if legacy:
            if strength is not None or len(parts) > 1:
                raise ValueError(
                    "%s takes an init image but no strength: its CLI only has "
                    "the deprecated --image-path, which cannot carry one"
                    % model_name)
            cmd += ["--image-path", path]
        else:
            if strength is not None and len(parts) > 1:
                raise ValueError(
                    "give the strength once: either --image %s,%s or --strength %s"
                    % (parts[0], parts[1], strength))
            parts = [path, str(strength)] if strength is not None else [path] + parts[1:]
            if len(parts) > 1:
                _check_strength(parts[1])
            cmd += ["--image"] + parts
    elif strength is not None:
        raise ValueError("--strength needs an --init-image to apply to")
    # LoRAs: `--lora PATH [SCALE]`, repeatable. Until now `fxlla pull
    # civitai:<id>` could download these and nothing could apply them.
    for ref in loras or []:
        _require_cap(spec, "lora", "--lora", model_name)
        parts = split_ref(ref)
        if not parts:
            raise ValueError("empty --lora value")
        # mflux accepts a local file OR a HuggingFace repo id (org/name), and
        # a repo id contains a slash too - so "has a slash" cannot be the
        # test. Something that names a file (extension, absolute, or an
        # explicitly relative path) must exist; anything else is left for
        # mflux to resolve.
        path = parts[0]
        looks_local = (path.endswith((".safetensors", ".ckpt"))
                       or os.path.isabs(path)
                       or path.startswith(("./", "../", "~")))
        if looks_local:
            # Expand for the check AND for the argv: validating the expanded
            # path while passing the literal "~" told the caller the file was
            # fine and then handed mflux a path it cannot resolve, since
            # nothing runs through a shell.
            parts[0] = os.path.expanduser(path)
            if not os.path.isfile(parts[0]):
                parts[0] = resolve_lora_name(path)
        cmd += ["--lora"] + parts
    if lora_style:
        _require_cap(spec, "lora-style", "--lora-style", model_name)
        cmd += ["--lora-style", lora_style]
    if guidance is not None:
        _require_cap(spec, "guidance", "--guidance", model_name)
        cmd += ["--guidance", str(guidance)]
    # Controlnet comes in two shapes and they are not interchangeable: FLUX
    # takes a checkpoint plus a control image as separate repeatable flags,
    # Z-Image takes one combined "type:path[:strength]" spec. Both stack, so
    # the caps decide which form this model speaks rather than inventing a
    # lossy abstraction over the pair.
    for ctl in controls or []:
        if "control-spec" in spec.get("caps", ()):
            cmd += ["--control", str(ctl)]
        else:
            _require_cap(spec, "controlnet-image", "--control", model_name)
            parts = split_ref(ctl)
            image = os.path.expanduser(parts[0])
            if not os.path.isfile(image):
                # The two families take different forms and confusing them is
                # the likely mistake, so say which one this model speaks
                # rather than only that a file is missing.
                if ":" in parts[0] and not os.path.exists(parts[0]):
                    raise ValueError(
                        "'%s' looks like the z-controlnet form "
                        "(type:path[:strength]); model '%s' takes "
                        "IMAGE[,CHECKPOINT] instead" % (parts[0], model_name))
                raise ValueError("control image not found: %s" % parts[0])
            cmd += ["--controlnet-image-path", image]
            if len(parts) > 1:
                _require_cap(spec, "controlnet", "--controlnet-path", model_name)
                cmd += ["--controlnet-path", parts[1]]
    if controlnet_strength is not None:
        _require_cap(spec, "controlnet-strength", "--controlnet-strength", model_name)
        cmd += ["--controlnet-strength", str(controlnet_strength)]
    if depth_image:
        _require_cap(spec, "depth-image", "--depth-image", model_name)
        depth_image = os.path.expanduser(depth_image)
        if not os.path.isfile(depth_image):
            raise ValueError("depth image not found: %s" % depth_image)
        cmd += ["--depth-image-path", depth_image]
    if save_depth:
        # The derived map is the input to a later controlnet step, so it has
        # to be reachable rather than discarded inside the render.
        _require_cap(spec, "save-depth", "--save-depth-map", model_name)
        cmd += ["--save-depth-map"]
    # NVIDIA's pixel-diffusion decoder replaces the VAE decode and emits 4x the
    # requested size, so it is an upscale that happens inside the render.
    if pid_decode:
        _require_cap(spec, "pid-decode", "--pid-decode", model_name)
        cmd += ["--pid-decode"]
        if pid_degrade_sigma is not None:
            _check_pid_sigma(pid_degrade_sigma)
            cmd += ["--pid-degrade-sigma", str(pid_degrade_sigma)]
    elif pid_degrade_sigma is not None:
        raise ValueError("--pid-degrade-sigma does nothing without --pid-decode")
    if preset:
        _require_cap(spec, "preset", "--preset", model_name)
        if preset.upper() not in IDEOGRAM_PRESETS:
            raise ValueError("unknown preset %s: one of %s"
                             % (preset, ", ".join(sorted(IDEOGRAM_PRESETS))))
        cmd += ["--preset", preset.upper()]
    if steps is not None:
        _require_cap(spec, "steps", "--steps", model_name)
        cmd += ["--steps", str(steps)]
    elif spec.get("steps") is not None and "steps" in spec.get("caps", ()):
        cmd += ["--steps", str(spec["steps"])]
    if seed is not None:
        _require_cap(spec, "seed", "--seed", model_name)
        cmd += ["--seed", str(seed)]
    # Both would be a contradiction the CLI resolves by ignoring one, silently:
    # a request for 512x512 with aspect 1:1 produced a 1024x1024 file and
    # nothing said so. Refusing names the conflict; asking for either alone is
    # the ordinary case and still works.
    if aspect and (width or height):
        raise ValueError(
            "give either --aspect or --width/--height, not both: aspect %s "
            "would override the requested %sx%s" % (aspect, width or "?", height or "?"))
    if aspect:
        _require_cap(spec, "dimensions", "--aspect", model_name)
        width, height = aspect_size(aspect)
    if width or height:
        _require_cap(spec, "dimensions", "--width/--height", model_name)
    _check_dim_step(model_name, width, height)
    if width:
        cmd += ["--width", str(width)]
    if height:
        cmd += ["--height", str(height)]
    if low_ram:
        cmd += ["--low-ram"]
    if metadata:
        cmd += ["--metadata"]
    return cmd


# Ideogram 4 accepts a JSON caption instead of prose: a scene broken into
# elements, each optionally placed with a bounding box. The rules below are
# mflux's own (mflux/models/ideogram4/.../caption.py), checked here so a
# malformed caption is named before a render starts rather than surfacing as
# a schema warning mid-run. The trap worth catching is the axis order: bbox is
# [y_min, x_min, y_max, x_max] - Y FIRST - in an integer 0..1000 space, which
# nobody guesses right.
IDEOGRAM_TOP_KEYS = {"high_level_description", "style_description",
                     "compositional_deconstruction"}
IDEOGRAM_STYLE_KEYS = {"aesthetics", "lighting", "photo", "art_style",
                       "medium", "color_palette"}
IDEOGRAM_ELEMENT_KEYS = {"type", "bbox", "text", "desc", "color_palette"}
IDEOGRAM_ELEMENT_TYPES = {"obj", "text"}


def _check_color_palette(palette, path, problems, max_colors):
    """max_colors differs by context in mflux's own verifier: 16 for the
    overall style palette, 5 per element. Hardcoding 5 falsely rejected a
    valid caption, which is worse than missing one - it blocks a render that
    would have worked."""
    if not isinstance(palette, list):
        problems.append("%s: expected a list" % path)
        return
    if len(palette) > max_colors:
        problems.append("%s: at most %d colors, got %d"
                        % (path, max_colors, len(palette)))
    for i, color in enumerate(palette):
        # Uppercase is mflux's rule, not a preference: it rejects "#f2b134"
        # and accepts "#F2B134", and the difference is invisible unless
        # something says so.
        if not (isinstance(color, str) and len(color) == 7
                and color.startswith("#")
                and all(c in "0123456789ABCDEF" for c in color[1:])):
            problems.append("%s[%d]: expected an UPPERCASE #RRGGBB hex color, "
                            "got %r" % (path, i, color))


def check_ideogram_caption(text):
    """Problems with an Ideogram 4 JSON caption, empty when it is fine.

    Prose is also valid input, so anything that is not a JSON object is left
    alone: only something that parses as one is held to the schema."""
    try:
        caption = json.loads(text)
    except (ValueError, TypeError):
        return []
    if not isinstance(caption, dict):
        return []
    problems = []
    unknown = set(caption) - IDEOGRAM_TOP_KEYS
    if unknown:
        problems.append("unknown top-level keys: %s (known: %s)"
                        % (", ".join(sorted(unknown)), ", ".join(sorted(IDEOGRAM_TOP_KEYS))))
    style = caption.get("style_description")
    if isinstance(style, dict):
        bad = set(style) - IDEOGRAM_STYLE_KEYS
        if bad:
            problems.append("style_description: unknown keys: %s" % ", ".join(sorted(bad)))
        if "color_palette" in style:
            _check_color_palette(style["color_palette"],
                                 "style_description.color_palette", problems, 16)
    if "compositional_deconstruction" not in caption:
        problems.append("compositional_deconstruction: missing (mflux expects "
                        "it in a JSON caption)")
    comp = caption.get("compositional_deconstruction")
    if comp is not None and not isinstance(comp, dict):
        problems.append("compositional_deconstruction: expected an object")
        comp = None
    elements = (comp or {}).get("elements")
    if elements is not None and not isinstance(elements, list):
        problems.append("compositional_deconstruction.elements: expected a list")
        elements = None
    for i, element in enumerate(elements or []):
        path = "elements[%d]" % i
        if not isinstance(element, dict):
            problems.append("%s: expected an object" % path)
            continue
        bad = set(element) - IDEOGRAM_ELEMENT_KEYS
        if bad:
            problems.append("%s: unknown keys: %s" % (path, ", ".join(sorted(bad))))
        kind = element.get("type")
        if kind not in IDEOGRAM_ELEMENT_TYPES:
            problems.append("%s.type: expected one of %s, got %r"
                            % (path, ", ".join(sorted(IDEOGRAM_ELEMENT_TYPES)), kind))
        if kind == "text" and not isinstance(element.get("text"), str):
            problems.append("%s: a text element needs a 'text' string" % path)
        if "color_palette" in element:
            _check_color_palette(element["color_palette"],
                                 "%s.color_palette" % path, problems, 5)
        if "bbox" in element:
            bbox = element["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                problems.append("%s.bbox: expected [y_min, x_min, y_max, x_max]" % path)
                continue
            if not all(type(v) is int for v in bbox):
                problems.append("%s.bbox: all four values must be integers" % path)
                continue
            y_min, x_min, y_max, x_max = bbox
            if not all(0 <= v <= 1000 for v in bbox):
                problems.append("%s.bbox: values live in a 0..1000 space, got %s"
                                % (path, bbox))
            if y_min > y_max:
                problems.append("%s.bbox: y_min %d > y_max %d (the order is "
                                "[y_min, x_min, y_max, x_max], Y first)"
                                % (path, y_min, y_max))
            if x_min > x_max:
                problems.append("%s.bbox: x_min %d > x_max %d" % (path, x_min, x_max))
    return problems


# A structurally valid file can still be garbage: silent speech, a one-frame
# "video", a blank image. Content checks run after the container checks and are
# skippable, because a false positive would reject a render the user wanted.
def _check_quality(kind, path, checker):
    if SKIP_QUALITY or quality.skip_quality_checks():
        return
    try:
        problems = checker(path)
    except Exception as exc:  # a checker must never be the reason a render fails
        print("quality check skipped for %s: %s" % (path, exc), file=sys.stderr)
        return
    message = quality.report(kind, path, problems)
    if message:
        raise RuntimeError(message + " (pass --skip-quality, or set "
                           "FXLLA_MEDIA_SKIP_QUALITY=1, to accept it)")


def validate_output(path):
    """Fail if the output is missing, not a PNG, or implausibly small.

    mflux can exit 0 while writing nothing useful; checking the file is the only
    reliable signal. This catches empty or truncated files, not every bad image.
    """
    if not os.path.exists(path):
        raise RuntimeError("no output produced at %s" % path)
    if os.path.getsize(path) < 1024:
        raise RuntimeError("output at %s is suspiciously small; likely a failed render" % path)
    with open(path, "rb") as f:
        if f.read(8) != PNG_MAGIC:
            raise RuntimeError("output at %s is not a PNG" % path)
    _check_quality("image", path, quality.check_png)


def generate_image(prompt, model=None, steps=None, seed=None, width=None,
                   height=None, aspect=None, quantize=None, low_ram=False,
                   metadata=False, output=None, keep_models=False,
                   negative=None, prompt_file=None, loras=None,
                   lora_style=None, init_image=None, guidance=None,
                   controls=None, controlnet_strength=None, depth_image=None,
                   save_depth=False, preset=None, strength=None,
                   pid_decode=False, pid_degrade_sigma=None):
    if not prompt:
        raise ValueError("prompt is required")
    model = model or DEFAULT_MODEL
    spec = MODELS.get(model)
    if spec is None:
        raise ValueError("unknown model '%s'; try one of: %s"
                         % (model, ", ".join(sorted(MODELS))))
    # A JSON caption is only meaningful to ideogram4, and a malformed one
    # would otherwise reach the backend and come back as a schema warning
    # after the render had already started.
    if model.startswith("ideogram"):
        problems = check_ideogram_caption(prompt)
        if problems:
            raise ValueError("the JSON caption has %d problem(s):\n  %s"
                             % (len(problems), "\n  ".join(problems)))
    os.makedirs(OUT_DIR, exist_ok=True)
    output = resolve_output(output, model, "png")
    free_gpu("image", keep_models)
    cmd = build_command(spec, prompt, output, steps=steps, seed=seed,
                        width=width, height=height, aspect=aspect,
                        quantize=quantize, low_ram=low_ram,
                        metadata=metadata, negative=negative,
                        prompt_file=prompt_file, loras=loras,
                        lora_style=lora_style, init_image=init_image,
                        model_name=model, guidance=guidance, controls=controls,
                        controlnet_strength=controlnet_strength,
                        depth_image=depth_image, save_depth=save_depth,
                        preset=preset, strength=strength,
                        pid_decode=pid_decode,
                        pid_degrade_sigma=pid_degrade_sigma)
    # The extras mirror what weights.require() was asked for above, so the
    # cache root picked here covers the same repos the consent gate counted.
    # A --lora given as a repo id is a dependency too, and no catalog row
    # names it - without this the root choice is blind to where it lives and
    # mflux quietly fetches it again.
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env=_env("image", model,
                                   ("pid",) if pid_decode else (),
                                   _lora_repo_ids(loras)))
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr.strip() or "%s failed" % spec["cli"])[-800:])
    _report_warnings(proc.stderr)
    validate_output(output)
    return output


def build_video_command(prompt, output, stage=DEFAULT_STAGE, frames=None,
                        frame_rate=DEFAULT_FRAME_RATE, width=None, height=None,
                        seed=None, low_ram=False, model=None, bin_path=None,
                        images=None, steps=None, stage1_steps=None,
                        stage2_steps=None):
    """Assemble the ltx-2-mlx argument vector for one video generation.

    generate requires exactly one stage flag (distilled is the fast default) and
    frame-rate is mandatory (the model was trained at 24 fps)."""
    if stage not in VIDEO_STAGES:
        raise ValueError("unknown stage '%s'; try one of: %s"
                         % (stage, ", ".join(VIDEO_STAGES)))
    # Each stage has ONE step knob and ignores the other, per ltx's --help
    # ("--steps: Denoising steps for one-stage"). Forwarding the wrong one
    # would be accepted and discarded, which is the failure the image
    # capability gate exists to prevent - the same rule, applied to video.
    wanted = VIDEO_STAGE_INFO[stage]["steps_flag"]
    for name, value in (("steps", steps), ("stage1_steps", stage1_steps),
                        ("stage2_steps", stage2_steps)):
        if value is None:
            continue
        allowed = (name == wanted
                   or (name == "stage2_steps" and wanted == "stage1_steps"))
        if not allowed:
            raise ValueError(
                "stage '%s' ignores --%s; its step knob is --%s"
                % (stage, name.replace("_", "-"), wanted.replace("_", "-")))
    if steps is not None:
        cmd_steps = ["--steps", str(steps)]
    else:
        cmd_steps = []
    if stage1_steps is not None:
        cmd_steps += ["--stage1-steps", str(stage1_steps)]
    if stage2_steps is not None:
        cmd_steps += ["--stage2-steps", str(stage2_steps)]
    cmd = [bin_path or VIDEO_BIN, "generate", "--%s" % stage,
           "--prompt", prompt, "--output", output,
           "--frame-rate", str(frame_rate if frame_rate is not None else DEFAULT_FRAME_RATE)]
    # Image-to-video anchors. ltx takes `--image PATH [FRAME_IDX STRENGTH]`,
    # repeatable: one image anchors the opening frame, two anchor both ends,
    # which is how a transition between two stills is actually produced. A
    # missing file would otherwise surface as a generic backend failure.
    for ref in images or []:
        parts = split_ref(ref)
        if not parts:
            raise ValueError("empty --image value")
        parts[0] = os.path.expanduser(parts[0])
        if not os.path.isfile(parts[0]):
            raise ValueError("reference image not found: %s" % parts[0])
        cmd += ["--image"] + parts
    if model:
        cmd += ["--model", model]
    if frames:
        cmd += ["--frames", str(frames)]
    if width:
        cmd += ["--width", str(width)]
    if height:
        cmd += ["--height", str(height)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if low_ram:
        cmd += ["--low-ram"]
    return cmd + cmd_steps


def validate_video_output(path):
    """Fail if the output is missing, not an MP4, or implausibly small."""
    if not os.path.exists(path):
        raise RuntimeError("no output produced at %s" % path)
    if os.path.getsize(path) < 10240:
        raise RuntimeError("output at %s is suspiciously small; likely a failed render" % path)
    with open(path, "rb") as f:
        head = f.read(16)
    if b"ftyp" not in head:
        raise RuntimeError("output at %s is not an MP4" % path)
    _check_quality("video", path, quality.check_video)


def generate_video(prompt, stage=DEFAULT_STAGE, frames=None,
                   frame_rate=DEFAULT_FRAME_RATE, width=None, height=None,
                   seed=None, low_ram=False, model=None, output=None,
                   keep_models=False, images=None, steps=None,
                   stage1_steps=None, stage2_steps=None):
    if not prompt:
        raise ValueError("prompt is required")
    os.makedirs(OUT_DIR, exist_ok=True)
    output = resolve_output(output, "video", "mp4")
    free_gpu("video", keep_models)
    cmd = build_video_command(prompt, output, stage=stage, frames=frames,
                              frame_rate=frame_rate, width=width, height=height,
                              seed=seed, low_ram=low_ram, model=model,
                              images=images, steps=steps,
                              stage1_steps=stage1_steps,
                              stage2_steps=stage2_steps)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env=_env("video", model))
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr.strip() or "video generation failed")[-800:])
    _report_warnings(proc.stderr)
    validate_video_output(output)
    return output


def build_voice_command(text, output, ref, model=None, lang=None, speed=1.0,
                        python=None, backend=None):
    """Assemble the command that runs the voice backend under the mlx-audio
    interpreter. A reference voice wav is required (the model has no built-in
    speaker conditionals)."""
    if not ref:
        raise ValueError("a reference voice wav is required "
                         "(set FXLLA_VOICE_REF or pass --ref)")
    return [python or resolved_voice_python(), backend or VOICE_BACKEND,
            "--text", text, "--output", output, "--ref", ref,
            "--model", model or VOICE_MODEL, "--lang", lang or VOICE_LANG,
            "--speed", str(speed)]


def validate_wav_output(path):
    """Fail if the output is missing, not a WAV, or implausibly small."""
    if not os.path.exists(path):
        raise RuntimeError("no output produced at %s" % path)
    if os.path.getsize(path) < 1024:
        raise RuntimeError("output at %s is suspiciously small; likely a failed render" % path)
    with open(path, "rb") as f:
        head = f.read(12)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise RuntimeError("output at %s is not a WAV" % path)
    _check_quality("audio", path, quality.check_wav)


def generate_speech(text, ref=None, lang=None, model=None, speed=1.0,
                    output=None, keep_models=False):
    if not text:
        raise ValueError("text is required")
    os.makedirs(OUT_DIR, exist_ok=True)
    output = resolve_output(output, "voice", "wav")
    free_gpu("voice", keep_models)
    cmd = build_voice_command(text, output, ref or VOICE_REF, model=model,
                              lang=lang, speed=speed)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env=_env("voice", model))
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr.strip() or "voice generation failed")[-800:])
    _report_warnings(proc.stderr)
    validate_wav_output(output)
    return output


def build_edit_command(prompt, image, output, seed=None, quantize=8,
                       bin_path=None):
    """Assemble the mflux qwen-edit argument vector for one image edit.

    An input image is required: qwen-edit conditions the edit on it, so an
    empty path cannot produce anything. The image is passed via --image-paths
    (the CLI accepts one or more; a single edit uses one)."""
    if not prompt:
        raise ValueError("prompt is required")
    if not image:
        raise ValueError("an input image is required")
    cmd = [bin_path or EDIT_BIN or mflux_cli("mflux-generate-qwen-edit"),
           "--prompt", prompt, "--image-paths", image,
           "--output", output, "--quantize", str(quantize)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd


def generate_edit(prompt, image, seed=None, quantize=8, output=None,
                  keep_models=False):
    if not prompt:
        raise ValueError("prompt is required")
    if not image:
        raise ValueError("an input image is required")
    if not os.path.exists(image):
        raise ValueError("input image not found: %s" % image)
    os.makedirs(OUT_DIR, exist_ok=True)
    output = resolve_output(output, "edit", "png")
    free_gpu("edit", keep_models)
    cmd = build_edit_command(prompt, image, output, seed=seed, quantize=quantize)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=_env("edit"))
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr.strip() or "image edit failed")[-800:])
    _report_warnings(proc.stderr)
    validate_output(output)
    return output


def build_upscale_command(image, output, scale=None, bin_path=None):
    """Assemble the mflux seedvr2 argument vector for one image upscale.

    seedvr2 has no prompt; the input image is the required conditioning. --scale
    maps to the CLI's --resolution, which accepts a target shortest-edge in
    pixels or a scale factor such as 2x."""
    if not image:
        raise ValueError("an input image is required")
    cmd = [bin_path or UPSCALE_BIN or mflux_cli("mflux-upscale-seedvr2"),
           "--image-path", image, "--output", output]
    if scale is not None:
        cmd += ["--resolution", str(scale)]
    return cmd


def generate_upscale(image, scale=None, output=None, keep_models=False):
    if not image:
        raise ValueError("an input image is required")
    if not os.path.exists(image):
        raise ValueError("input image not found: %s" % image)
    os.makedirs(OUT_DIR, exist_ok=True)
    output = resolve_output(output, "upscale", "png")
    free_gpu("upscale", keep_models)
    cmd = build_upscale_command(image, output, scale=scale)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env=_env("upscale"))
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr.strip() or "image upscale failed")[-800:])
    _report_warnings(proc.stderr)
    validate_output(output)
    return output


def cmd_image(args):
    path = generate_image(
        args.prompt, model=args.model, steps=args.steps, seed=args.seed,
        width=args.width, height=args.height, aspect=args.aspect,
        quantize=args.quantize, low_ram=args.low_ram, metadata=args.metadata,
        output=args.output, keep_models=args.keep_models,
        negative=args.negative, prompt_file=args.prompt_file,
        loras=args.loras, lora_style=args.lora_style,
        init_image=args.init_image, guidance=args.guidance,
        controls=args.controls, controlnet_strength=args.controlnet_strength,
        depth_image=args.depth_image, save_depth=args.save_depth,
        preset=args.preset, strength=args.strength,
        pid_decode=args.pid_decode, pid_degrade_sigma=args.pid_degrade_sigma)
    print(path)
    # The derived depth map is a second artifact and the reason --save-depth-map
    # exists, so it goes on stdout next to the image: a caller chaining a
    # control step needs the path, not the naming convention.
    if args.save_depth:
        derived = depth_map_path(path)
        if os.path.isfile(derived):
            print(derived)
    facts = quality.image_facts(path)
    if facts.get("width"):
        print("size: %dx%d, %d KB" % (facts["width"], facts["height"],
                                      facts.get("bytes", 0) // 1024),
              file=sys.stderr)


def cmd_video(args):
    frames = args.frames
    rate = args.frame_rate if args.frame_rate is not None else DEFAULT_FRAME_RATE
    if args.seconds is not None:
        if frames is not None:
            raise ValueError("give either --seconds or --frames, not both")
        frames = max(1, int(round(args.seconds * rate)))
    path = generate_video(
        args.prompt, stage=args.stage, frames=frames, frame_rate=args.frame_rate,
        width=args.width, height=args.height, seed=args.seed, low_ram=args.low_ram,
        model=args.model, output=args.output, keep_models=args.keep_models,
        images=args.images, steps=args.steps, stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps)
    print(path)
    # The measured result, not the request: a caller reading back its own
    # --frames has no way to know what the backend actually produced, and one
    # confidently reported 2 seconds of video as 10.
    facts = quality.video_facts(path)
    if facts.get("duration_s") is not None:
        print("duration: %.2fs (%s frames at %s fps)"
              % (facts["duration_s"], facts.get("frames", "?"), facts.get("fps", "?")),
              file=sys.stderr)


def cmd_voice(args):
    path = generate_speech(
        args.text, ref=args.ref, lang=args.lang, model=args.model,
        speed=args.speed, output=args.output, keep_models=args.keep_models)
    print(path)


def cmd_edit(args):
    path = generate_edit(
        args.prompt, args.image, seed=args.seed, quantize=args.quantize,
        output=args.output, keep_models=args.keep_models)
    print(path)


def cmd_upscale(args):
    path = generate_upscale(
        args.image, scale=args.scale, output=args.output,
        keep_models=args.keep_models)
    print(path)


LORA_STYLES = ("couple", "font", "home", "identity", "illustration",
               "portrait", "ppt", "sandstorm", "sparklers", "storyboard")

# What a caller needs to BUILD an Ideogram 4 caption, not just to be told one
# exists. Without this a model has to guess the key names and the bbox
# convention, and the axis order is not guessable: Y comes first.
IDEOGRAM_PROMPT_FORMAT = {
    "when": "Pass this JSON object as the prompt for ideogram4. Plain prose "
            "also works; use JSON when you need text or objects placed.",
    "keys": {
        "high_level_description": "one sentence describing the whole image",
        "style_description": "object with aesthetics, lighting, medium, and "
                             "either art_style or photo, plus an optional "
                             "color_palette of up to 16 colors",
        "compositional_deconstruction": "object with background (a string) and "
                                        "elements (a list)",
    },
    "element": {
        "type": "obj or text",
        "bbox": "[y_min, x_min, y_max, x_max] - Y FIRST - integers in a "
                "0..1000 space, y_min <= y_max and x_min <= x_max",
        "text": "the literal characters to render (required when type is text)",
        "desc": "what it looks like",
        "color_palette": "up to 5 colors for this element",
    },
    "colors": "UPPERCASE #RRGGBB, e.g. #F2B134 (lowercase is rejected)",
    "example": {
        "high_level_description": "A vintage poster of a chrome lighter.",
        "style_description": {
            "aesthetics": "mid-century travel poster, flat vector shapes",
            "lighting": "warm rim light, soft vignette",
            "medium": "screen print",
            "art_style": "retro Swiss design",
            "color_palette": ["#0D3B45", "#F2B134"]},
        "compositional_deconstruction": {
            "background": "flat deep teal field",
            "elements": [
                {"type": "text", "bbox": [80, 120, 240, 880], "text": "ZIPPO",
                 "desc": "condensed display type", "color_palette": ["#E8E3D3"]},
                {"type": "obj", "bbox": [300, 330, 760, 670],
                 "desc": "chrome lighter, lid open, tall flame"}]}},
}


_COST_GUIDE = (
    "Steps compare a model only against ITSELF. Across models they do not: "
    "krea2 and z-image-turbo both run 8 steps and were measured on this "
    "machine 9x apart, because a step costs what the model costs. Use "
    "`observed` - real seconds from this machine's own finished jobs, keyed by "
    "model and by video:<stage>, with n samples and s_per_mp for images so it "
    "scales to your canvas. A model absent from `observed` has never been "
    "timed here: say that you do not know rather than estimating, because an "
    "invented figure was wrong by 25x. Cost here is time and GPU, never money: "
    "there is no per-render charge, so a $/$$/$$$ column is a fiction. Read "
    "this again rather than quoting an earlier reply - `observed` grows with "
    "every finished render. NEVER run renders to answer a question about "
    "cost: timing all 16 image models would be an hour of GPU and tens of GB "
    "of downloads, spent on a question. Report what is measured, name what is "
    "not, and offer to time one if they want it."
)


def cmd_models(args):
    if getattr(args, "json", False):
        print(json.dumps({
            "default": DEFAULT_MODEL,
            "cost": _COST_GUIDE,
            "observed": observed_timings(),
            "models": {n: {"cli": s["cli"], "steps": s["steps"],
                           "caps": sorted(s["caps"]), "note": s["note"],
                           # What it will actually run if nothing is passed.
                           # Cost is roughly linear in this, and the models
                           # differ by more than 10x - without it a caller
                           # cannot tell a one-minute render from an
                           # eight-minute one until it has waited for both.
                           "default_steps": s["steps"] or _MFLUX_DEFAULT_STEPS.get(n),
                           # Only where the backend enforces it; absent means
                           # any size. Stated here so a caller sizes a poster
                           # correctly instead of learning it from a crash.
                           **({"dim_step": _DIM_STEP[n]} if n in _DIM_STEP else {}),
                           **({"presets": IDEOGRAM_PRESETS}
                              if "preset" in s["caps"] else {})}
                       for n, s in MODELS.items()},
            "lora_styles": list(LORA_STYLES),
            "prompt_formats": {"ideogram4": IDEOGRAM_PROMPT_FORMAT},
            # Video belongs here too. It was left out, so the only thing a
            # caller could see about the pipeline that decides between a short
            # wait and a very long one was the word "distilled".
            "video": {
                "default_stage": DEFAULT_STAGE,
                "frame_rate": DEFAULT_FRAME_RATE,
                "stages": VIDEO_STAGE_INFO,
                "cost": "Wall time scales with stage, step count, resolution "
                        "AND duration: seconds x frame_rate is the frame count, "
                        "and every frame is sampled. A long clip at a slow "
                        "stage is the most expensive thing here - iterate "
                        "short and distilled, then commit. Measured times per "
                        "stage are in the top-level `observed`, under "
                        "video:<stage>; they are not normalised by length, so "
                        "read n and treat one sample as one sample.",
            },
        }, indent=1))
        return
    print("%-14s %-9s %s" % ("MODEL", "STEPS", "NOTE"))
    for name in sorted(MODELS):
        spec = MODELS[name]
        default = " *" if name == DEFAULT_MODEL else "  "
        # The step count it will really run, not a blank: cost is roughly
        # linear in it and these span 4 to 50, which is the difference between
        # a minute and most of ten.
        steps = spec["steps"] or _MFLUX_DEFAULT_STEPS.get(name)
        print("%-14s%s%-9s %s" % (name, default, steps if steps else "cli",
                                  spec["note"]))
    print("\nVIDEO STAGES   STEPS     NOTE")
    for name, info in VIDEO_STAGE_INFO.items():
        default = " *" if name == DEFAULT_STAGE else "  "
        steps = info.get("steps") or "%s+%s" % (info["stage1_steps"],
                                                info["stage2_steps"])
        print("%-14s%s%-9s %s" % (name, default, steps, info["note"]))
    print("")
    print_timings()
    print("\n* defaults. Capabilities, video cost and the ideogram4 caption "
          "schema: fxlla media models --json")


def print_timings():
    seen = observed_timings()
    if not seen:
        print("Nothing has been timed here yet. Times appear as renders finish;"
              "\nnone are estimated, and none come from anyone else's hardware.")
        return
    print("%-20s %-6s %-9s %s" % ("MEASURED HERE", "RUNS", "MEDIAN",
                                  "PER MEGAPIXEL"))
    for key in sorted(seen, key=lambda k: -seen[k]["median_s"]):
        entry = seen[key]
        rate = ("%.0f s" % entry["s_per_mp"]) if entry.get("s_per_mp") else "-"
        print("%-20s %-6s %-9s %s"
              % (key, entry["n"], "%.0f s" % entry["median_s"], rate))
    print("\nSteps compare a model to itself, not to another one: the same "
          "8 steps\nrun 9x apart on two of these. A model missing above has "
          "not been timed here -\nsay so rather than estimating, and never "
          "render just to fill this table in.")


def cmd_describe(args):
    print(describe_image(args.image, args.question, args.model))


def cmd_timings(args):
    """`fxlla media timings`: the measured table on its own.

    It exists because the numbers being published was not enough. Asked how
    long each model takes, a caller went to bash and wrote forty lines of
    Python to parse the job files by hand - twice - and its version reported a
    video stage as an image model. Someone reaching for the shell should find
    this instead of reimplementing it."""
    if getattr(args, "json", False):
        print(json.dumps({"observed": observed_timings(), "cost": _COST_GUIDE},
                         indent=1))
        return
    print_timings()


# Kinds whose output is a single still, so wall time scales with its area.
_ONE_CANVAS = ("image", "edit", "upscale")


def observed_timings(records=None):
    """What renders on THIS machine actually took, from the job history:
    {key: {n, median_s, s_per_mp}}. Keys are model aliases, and "video:<stage>"
    for video.

    Steps are not a portable cost signal and publishing them as one was a
    mistake: krea2 and z-image-turbo both run 8 steps and were measured here 9x
    apart, because a step costs what the model costs. A caller told "cost is
    linear in steps" concluded that 8-step krea2 was as fast as 8-step turbo
    and invented a timing table that was off by 25x. Measured seconds are the
    only honest answer, they are already sitting in the job records, and being
    machine-specific is exactly right - nobody else's hardware is the question.
    """
    by_key = {}
    for rec in (records if records is not None else jobs.listing()):
        if rec.get("status") != "done":
            continue
        started, finished = rec.get("started"), rec.get("finished")
        # `is None`, not falsiness: a timestamp of 0 is a real value, and
        # treating it as missing silently dropped whole records.
        if started is None or finished is None or finished <= started:
            continue
        argv = rec.get("argv") or []
        kind = rec.get("kind")
        if kind == "video":
            stage = next((a.lstrip("-") for a in argv
                          if a.lstrip("-") in VIDEO_STAGES), DEFAULT_STAGE)
            key = "video:" + stage
        elif kind == "image":
            key = argv[argv.index("--model") + 1] if "--model" in argv else DEFAULT_MODEL
        else:
            key = kind
        # Area normalises anything that produces ONE canvas - a render, an
        # edit, an upscale. It does NOT normalise a video, whose cost also
        # scales with the frame count, where a seconds-per-megapixel would be a
        # fresh wrong signal of exactly the kind this function exists to
        # replace. Keying this on kind == "image" alone left edit and upscale
        # reporting a bare median that does not scale to another size.
        area = _megapixels(rec) if kind in _ONE_CANVAS else None
        by_key.setdefault(key, []).append((finished - started, area))
    out = {}
    for key, samples in by_key.items():
        seconds = sorted(s for s, _ in samples)
        rates = sorted(s / mp for s, mp in samples if mp)
        entry = {"n": len(seconds), "median_s": round(_median(seconds), 1)}
        if rates:
            entry["s_per_mp"] = round(_median(rates), 1)
        out[key] = entry
    return out


def _human_seconds(seconds):
    if seconds < 90:
        return "%d s" % round(seconds)
    return "%d min" % round(seconds / 60.0)


def _expected_seconds(args, seen=None):
    """How long this request has taken here before, or None.

    Scaled by canvas where there is a per-megapixel rate, since the same model
    at four times the area is not the same wait."""
    seen = observed_timings() if seen is None else seen
    if args.cmd == "video":
        entry = seen.get("video:" + getattr(args, "stage", DEFAULT_STAGE))
        return entry["median_s"] if entry else None
    if args.cmd != "image":
        return None
    entry = seen.get(getattr(args, "model", None) or DEFAULT_MODEL)
    if not entry:
        return None
    width, height = getattr(args, "width", None), getattr(args, "height", None)
    if width and height and entry.get("s_per_mp"):
        return entry["s_per_mp"] * (width * height) / 1e6
    return entry["median_s"]


_WARNING_RE = re.compile(r"(?:^|\W)(?:User)?Warning:\s*(.+)$")


def backend_warnings(stderr):
    """What the backend said on a run that SUCCEEDED.

    Only stderr on failure was ever read, so a warning on a good render was
    captured and dropped - including mflux's own "--steps is ignored; Ideogram
    4 presets define the step count", which is the backend naming the exact
    class of bug that has now been found three times by reading its source
    instead. Filtered rather than forwarded whole: progress bars live on stderr
    too, and relaying those as warnings would train a reader to skip them."""
    seen, out = set(), []
    for line in (stderr or "").splitlines():
        match = _WARNING_RE.search(line.strip())
        if not match:
            continue
        text = match.group(1).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _report_warnings(stderr):
    for text in backend_warnings(stderr):
        print("warning: %s" % text, file=sys.stderr)


def _median(values):
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _megapixels(rec):
    """Pixel count of what a job produced, in megapixels, or None.

    Preferred from the artifact itself rather than the request: a backend that
    snapped a size to its grid produced something other than what was asked
    for, and the timing belongs to what it actually rendered."""
    output = rec.get("output")
    if output:
        facts = quality.image_facts(output)
        if facts.get("width"):
            return (facts["width"] * facts["height"]) / 1e6
    argv = rec.get("argv") or []
    try:
        width = int(argv[argv.index("--width") + 1])
        height = int(argv[argv.index("--height") + 1])
    except (ValueError, IndexError):
        return None
    return (width * height) / 1e6


def lora_dirs():
    """Where to look for LoRAs: FXLLA_LORA_DIRS (colon separated) plus the
    civitai download directory.

    Searching only the civitai directory was wrong: people train their own and
    keep them with the project that produced them, or share a directory with
    another tool, so a real collection was invisible and `list_loras` answered
    "none" to someone holding ten."""
    dirs = [d for d in os.environ.get("FXLLA_LORA_DIRS", "").split(":") if d]
    dirs.append(os.path.join(STORE, "civitai"))
    seen, out = set(), []
    for d in dirs:
        d = os.path.expanduser(d)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


# Tensor names an adapter has and a base model does not. Reading them is the
# only reliable test: filenames lie in both directions (Krea2-realism-V1 is a
# LoRA and says nothing, Krea-2-Turbo is a 26 GB base model whose name would
# match a "krea" filter), and metadata standards like modelspec.* appear on
# base models too, so they cannot be the signal either.
_LORA_TENSOR_MARKS = ("lora_a", "lora_b", "lora_down", "lora_up", "lokr_",
                      "loha_", "oft_", "lora_unet", "lora_te")
_SAFETENSORS_HEADER_CAP = 20 * 1024 * 1024


# Hidden dimension -> architecture, measured from adapters whose base is known
# either from their own metadata or from what they were trained against. The
# dimension is a property of the model an adapter was fitted to, so it survives
# renaming, and most LoRAs in the wild declare nothing.
_DIM_TO_BASE = {
    1536: "krea2",
    4608: "ideogram4",
    3840: "z-image",
    4096: "ltx2",     # with 16384; disambiguated below
    3072: "flux",     # shared with qwen, split by tensor names below
}


def infer_base_model(header):
    """Best guess at the base architecture from tensor shapes and names, or ""
    when nothing is recognised.

    Reported as a guess, never as fact: it is the only signal available for the
    majority of adapters, which declare no base at all, and a wrong pairing
    wastes a render rather than corrupting anything."""
    counts = {}
    names = []
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        names.append(key.lower())
        for dim in (spec.get("shape") or []) if isinstance(spec, dict) else []:
            if isinstance(dim, int) and dim >= 256:
                counts[dim] = counts.get(dim, 0) + 1
    if not counts:
        return ""
    present = set(counts)
    # 3072 is FLUX and Qwen both; Qwen's joint attention adds its own
    # projections, which FLUX has no equivalent of.
    if 3072 in present:
        joint = any("add_k_proj" in n or "add_q_proj" in n for n in names)
        return "qwen" if joint else "flux"
    for dim in sorted(present, key=lambda d: -counts[d]):
        base = _DIM_TO_BASE.get(dim)
        if base:
            return base
    return ""


def lora_base_model(path):
    """The base model a LoRA declares, "" when it declares none, or None when
    the file is not a LoRA at all.

    Reads only the safetensors header - a JSON block at the front of the file -
    so a 2 GB adapter costs a few kilobytes to identify. The declared base is
    what makes a collection usable: an adapter trained for krea2 does nothing
    useful on z-image, and the filename does not always say."""
    try:
        with open(path, "rb") as fh:
            length = struct.unpack("<Q", fh.read(8))[0]
            if length > _SAFETENSORS_HEADER_CAP:
                return None
            header = json.loads(fh.read(length))
    except (OSError, ValueError, struct.error):
        return None
    if not isinstance(header, dict):
        return None
    names = [k.lower() for k in header if k != "__metadata__"]
    if not any(any(m in n for m in _LORA_TENSOR_MARKS) for n in names):
        return None
    meta = header.get("__metadata__") or {}
    declared = meta.get("ss_base_model_version", "") if isinstance(meta, dict) else ""
    if declared:
        return declared
    guess = infer_base_model(header)
    return ("~" + guess) if guess else ""


def _hf_cache_loras():
    """LoRAs sitting in the Hugging Face cache, as (repo_id, megabytes).

    mflux takes a repo id directly for --lora, so one that is already cached
    is usable with no path at all - and these arrive by `hf download` or as a
    side effect of another tool, never through the civitai path. Matched by
    name (lora, dora, lightning) rather than by size: every base model is a
    pile of .safetensors too, and guessing by size would list all of them.
    """
    found, seen = [], set()
    for root in weights.cache_roots():
        for entry in _hf_cache_loras_in(os.path.join(root, "hub")):
            # The same repo can sit in two caches; offer it once. The first
            # root wins, which is the one a render would reach for anyway.
            if entry[0] in seen:
                continue
            seen.add(entry[0])
            found.append(entry)
    return found


def _hf_cache_loras_in(hub):
    """One cache's worth of the above. Split out so several can be scanned.

    Note the local name below is `files`, not `weights`: this module imports a
    module called weights, and shadowing it here would break the caller.
    """
    if not os.path.isdir(hub):
        return []
    found = []
    for name in sorted(os.listdir(hub)):
        if not name.startswith("models--"):
            continue
        repo = name[len("models--"):].replace("--", "/")
        files = glob.glob(os.path.join(hub, name, "snapshots", "*", "*.safetensors"))
        if not files:
            continue
        # Every weight file must be an adapter. Some base-model repos ship an
        # example LoRA beside the model (SDXL base does), and offering that
        # repo id as --lora would hand mflux the base model instead.
        verdicts = [lora_base_model(os.path.realpath(w)) for w in files]
        if not verdicts or any(v is None for v in verdicts):
            continue
        base = next((v for v in verdicts if v), "")
        total = 0
        for w in files:
            try:
                total += os.path.getsize(os.path.realpath(w))
            except OSError:
                continue
        found.append((repo, total // (1024 * 1024), base))
    return found


def resolve_lora_name(name):
    """A bare LoRA filename resolved against the search directories.

    `list_loras` reports a path and a name, and a caller that reads the name is
    behaving reasonably - one asked for "Krea2-realism-V1.safetensors" and got
    "LoRA not found", having been shown that exact string a moment earlier.
    Only names with no directory part are resolved: a path the caller spelled
    out is theirs to get right, and guessing at it would hide a typo."""
    if os.sep in name or (os.altsep and os.altsep in name):
        raise ValueError("LoRA not found: %s" % name)
    matches = []
    for root in lora_dirs():
        if not os.path.isdir(root):
            continue
        for base, _dirs, names in os.walk(root):
            if name in names:
                matches.append(os.path.join(base, name))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError("LoRA not found: %s (searched: %s). Names come from "
                         "`fxlla media loras`." % (name, ", ".join(lora_dirs())))
    raise ValueError("LoRA name is ambiguous: %s matches %s. Give the full path."
                     % (name, ", ".join(matches)))


def find_loras():
    """(path_or_repo, megabytes) for every LoRA fxlla can find: files under the
    search directories, plus Hugging Face repos already in the cache."""
    found = []
    for root in lora_dirs():
        if not os.path.isdir(root):
            continue
        for base, _dirs, names in os.walk(root):
            for n in sorted(names):
                if not n.endswith((".safetensors", ".ckpt")):
                    continue
                full = os.path.join(base, n)
                declared = lora_base_model(full)
                if declared is None and n.endswith(".safetensors"):
                    continue  # a base model sitting in a LoRA directory
                try:
                    found.append((full, os.path.getsize(full) // (1024 * 1024),
                                  declared or ""))
                except OSError:
                    continue
    return sorted(found) + sorted(_hf_cache_loras())


def cmd_loras(args):
    """What can actually be applied: the LoRAs found on disk plus mflux's
    built-in styles. Having one and no way to find it again is how they stayed
    unusable."""
    found = find_loras()
    if getattr(args, "json", False):
        print(json.dumps({
            "loras": [{"path": p, "mb": mb, "name": os.path.basename(p),
                       "base_model": base,
                       "source": "file" if os.path.isabs(p) else "huggingface"}
                      for p, mb, base in found],
            "styles": list(LORA_STYLES),
            "searched": lora_dirs(),
        }, indent=1))
        return
    if found:
        print("Found (use: --lora <path-or-repo>[,scale])")
        for p, mb, base in found:
            print("  %-58s %5d MB  %s" % (p, mb, base or "base not declared"))
    else:
        print("No LoRAs found. Searched:")
        for d in lora_dirs():
            print("  %s" % d)
        print("Point FXLLA_LORA_DIRS at where yours live (colon separated), "
              "or get one: fxlla pull civitai:<id>")
    print("\nBuilt-in styles (use: --lora-style <name>)")
    print("  " + ", ".join(LORA_STYLES))


def _summary(args):
    # Short label for `fxlla media jobs`, whatever the generator's input is.
    for field in ("prompt", "text", "image"):
        value = getattr(args, field, None)
        if value:
            return value if len(value) <= 60 else value[:57] + "..."
    return ""


def cmd_jobs(args):
    if getattr(args, "prune", False):
        print("removed %d finished job(s)" % jobs.prune())
        return
    records = jobs.listing()
    if getattr(args, "json", False):
        print(json.dumps(records))
        return
    if not records:
        print("(no media jobs)")
        return
    for rec in records:
        print(jobs.describe(rec))


def cmd_job(args):
    wait_s = getattr(args, "wait", None)
    rec = jobs.wait(args.id, wait_s) if wait_s else jobs.get(args.id)
    if rec is None:
        sys.exit("unknown job: %s" % args.id)
    if getattr(args, "json", False):
        print(json.dumps(rec))
        return
    print(jobs.describe(rec))
    if rec.get("error"):
        print(rec["error"])


def cmd_cancel(args):
    rec = jobs.cancel(args.id)
    if rec is None:
        sys.exit("unknown job: %s" % args.id)
    print(jobs.describe(rec))


def main():
    p = argparse.ArgumentParser(prog="fxlla-media")
    sub = p.add_subparsers(dest="cmd", required=True)
    im = sub.add_parser("image")
    im.add_argument("prompt")
    im.add_argument("--model", "-m")
    im.add_argument("--steps", type=int)
    im.add_argument("--seed", type=int)
    im.add_argument("--width", type=int)
    im.add_argument("--height", type=int)
    im.add_argument("--aspect")
    im.add_argument("--quantize", "-q", type=int)  # None: adapt to the model
    im.add_argument("--negative", "--negative-prompt", dest="negative",
                    help="what the image must NOT contain")
    im.add_argument("--prompt-file", dest="prompt_file",
                    help="read the prompt from a file (re-read per seed)")
    im.add_argument("--pid-decode", dest="pid_decode", action="store_true",
                    help="decode with NVIDIA PiD instead of the VAE; the "
                         "output is 4x the requested size. Downloads its own "
                         "weights (fxlla pull media:pid)")
    im.add_argument("--pid-degrade-sigma", dest="pid_degrade_sigma", type=float,
                    metavar="0..0.8",
                    help="noise the latent to this sigma before PiD decodes. "
                         "The default 0.0 is the input PiD saw LEAST in "
                         "training; try 0.2 if smooth areas look over-textured")
    im.add_argument("--strength", type=float, metavar="0..1",
                    help="how much of the init image survives; 0 keeps it, "
                         "1 ignores it (mflux default 0.4). A refining pass "
                         "over a first render is usually 0.3-0.5.")
    im.add_argument("--init-image", dest="init_image",
                    help="start from an existing image (img2img)")
    # `fxlla pull civitai:<id>` has been able to download LoRAs since it
    # shipped, with nothing able to apply one. PATH may be a local file or a
    # HuggingFace repo id; SCALE is optional.
    im.add_argument("--lora", action="append", dest="loras",
                    metavar="PATH[,SCALE]",
                    help="apply a LoRA; repeatable. Comma form, not spaces: "
                         "a multi-value flag would swallow the prompt.")
    im.add_argument("--guidance", type=float,
                    help="guidance scale (model default: 3.5 typical, 10 depth)")
    im.add_argument("--preset",
                    help="ideogram4 sampler preset, which sets its step count "
                         "and guidance together: %s"
                         % ", ".join("%s (%d steps)" % (n, s) for n, s
                                     in sorted(IDEOGRAM_PRESETS.items(),
                                               key=lambda kv: kv[1])))
    # Both controlnet families stack; the model's caps decide the wire form.
    im.add_argument("--control", action="append", dest="controls",
                    metavar="IMAGE[,CHECKPOINT] or TYPE:PATH[:STRENGTH]",
                    help="control input; repeatable to stack several")
    im.add_argument("--controlnet-strength", dest="controlnet_strength", type=float)
    im.add_argument("--depth-image", dest="depth_image",
                    help="use this depth map instead of deriving one")
    im.add_argument("--save-depth-map", dest="save_depth", action="store_true",
                    help="write the derived depth map so a later step can use it")
    im.add_argument("--lora-style", dest="lora_style",
                    help="one of mflux's built-in LoRA styles (see: fxlla media loras)")
    im.add_argument("--low-ram", action="store_true")
    im.add_argument("--metadata", action="store_true")
    im.add_argument("--output", "-o")

    vi = sub.add_parser("video")
    vi.add_argument("prompt")
    vi.add_argument("--stage", choices=VIDEO_STAGES, default=DEFAULT_STAGE,
                    help="pipeline, and the main cost knob: %s"
                         % "; ".join("%s (%s)" % (n, i["note"].split(".")[0])
                                     for n, i in VIDEO_STAGE_INFO.items()))
    # The step counts under each stage. Video had none of these: the pipeline
    # was the only cost control and its steps were unreachable.
    vi.add_argument("--steps", type=int,
                    help="denoising steps, one-stage only (default 8)")
    vi.add_argument("--stage1-steps", dest="stage1_steps", type=int,
                    help="stage 1 steps for the two-stage pipelines "
                         "(default 30, 15 for two-stages-hq)")
    vi.add_argument("--stage2-steps", dest="stage2_steps", type=int,
                    help="stage 2 steps for the two-stage pipelines (default 3)")
    vi.add_argument("--frames", type=int)
    # Duration is what a person asks for; frames is what the backend takes.
    # Without this the caller does the multiplication, and one got it wrong by
    # 5x while reporting success.
    vi.add_argument("--seconds", type=float,
                    help="target duration; converted to frames at the frame rate")
    vi.add_argument("--frame-rate", type=int, default=DEFAULT_FRAME_RATE)
    vi.add_argument("--width", type=int)
    vi.add_argument("--height", type=int)
    vi.add_argument("--seed", type=int)
    vi.add_argument("--model", "-m")
    # Image-to-video: PATH alone anchors the opening frame; PATH FRAME STRENGTH
    # anchors any frame. Repeat it to anchor both ends, which is what a
    # transition between two stills actually needs - describing the two images
    # in the prompt produces a different video that merely resembles them.
    vi.add_argument("--image", action="append", dest="images",
                    metavar="PATH[,FRAME[,STRENGTH]]",
                    help="reference image for image-to-video; repeatable")
    vi.add_argument("--low-ram", action="store_true")
    vi.add_argument("--output", "-o")

    vo = sub.add_parser("voice")
    vo.add_argument("text")
    vo.add_argument("--ref")
    vo.add_argument("--lang")
    vo.add_argument("--model", "-m")
    vo.add_argument("--speed", type=float, default=1.0)
    vo.add_argument("--output", "-o")

    ed = sub.add_parser("edit")
    ed.add_argument("prompt")
    ed.add_argument("--image", required=True)
    ed.add_argument("--seed", type=int)
    ed.add_argument("--quantize", "-q", type=int, default=8)
    ed.add_argument("--output", "-o")

    up = sub.add_parser("upscale")
    up.add_argument("--image", required=True)
    up.add_argument("--scale",
                    help="target shortest edge in pixels or a factor, e.g. 2x")
    up.add_argument("--output", "-o")

    for sp in (im, vi, vo, ed, up):
        sp.add_argument("--keep-models", action="store_true",
                        help="do not free the gateway's resident models first")
        # dest is not 'async': that is a Python keyword.
        sp.add_argument("--async", dest="run_async", action="store_true",
                        help="submit as a background job and print its id")
        sp.add_argument("--skip-quality", action="store_true",
                        help="accept output that fails the content checks")
        sp.add_argument("--yes", "-y", dest="yes", action="store_true",
                        help="authorize downloading any missing weights")

    ml = sub.add_parser("models")
    ml.add_argument("-j", "--json", action="store_true")
    de = sub.add_parser("describe")
    de.add_argument("--image", required=True, help="the image to look at")
    de.add_argument("question", nargs="?",
                    help="what to ask about it (default: describe it, quoting "
                         "any text)")
    de.add_argument("--model", help="vision model alias (default: FXLLA_VISION_MODEL)")
    lr = sub.add_parser("loras")
    lr.add_argument("-j", "--json", action="store_true")
    ti = sub.add_parser("timings")
    ti.add_argument("-j", "--json", action="store_true")
    # Introspection for doctor and setup: the ONE place voice-interpreter
    # resolution lives, so shell never re-implements it.
    sub.add_parser("voice-python")
    jl = sub.add_parser("jobs")
    jl.add_argument("-j", "--json", action="store_true")
    jl.add_argument("--prune", action="store_true",
                    help="delete finished job records and their logs")
    jg = sub.add_parser("job")
    jg.add_argument("id")
    jg.add_argument("-j", "--json", action="store_true")
    # Awaiting a render beats polling it: without this an agent issues one
    # status call per second and looks like a runaway loop.
    jg.add_argument("--wait", type=float, metavar="SECONDS",
                    help="block until the job finishes or this many seconds pass")
    jc = sub.add_parser("cancel")
    jc.add_argument("id")
    args = p.parse_args()
    global SKIP_QUALITY
    SKIP_QUALITY = bool(getattr(args, "skip_quality", False))
    # A generator downloads its weights on first use. Ask before that happens,
    # here rather than in the shell wrapper, because a background job and an MCP
    # tool call both re-invoke this module directly.
    if args.cmd in ("image", "video", "voice", "edit", "upscale"):
        if not getattr(args, "yes", False):
            # An OPTION can pull weights of its own, not just the model:
            # --pid-decode fetches a decoder checkpoint and a gated caption
            # encoder for any of ten models. Checking the model's alias alone
            # would pass on cached weights and then download 8 GB mid-render.
            extras = ["pid"] if getattr(args, "pid_decode", False) else []
            weights.require(args.cmd, getattr(args, "model", None) or DEFAULT_MODEL
                            if args.cmd == "image" else None, extras)
    if getattr(args, "run_async", False):
        # Reuse the invocation verbatim (minus the flag) as the job's argv, so a
        # background job runs exactly the same generator as the direct call.
        argv = [a for a in sys.argv[1:] if a != "--async"]
        print(jobs.submit(args.cmd, argv, _summary(args))["id"])
        # The expected wait, on the line after the id. A catalog is read once
        # and remembered: one caller read it at 11:33, kept quoting those
        # numbers for an hour, and never saw the measurements added at 12:05.
        # A figure that arrives WITH the submission cannot go stale that way.
        estimate = _expected_seconds(args)
        if estimate:
            print("typically %s here" % _human_seconds(estimate))
        return
    {"image": cmd_image, "video": cmd_video, "voice": cmd_voice,
     "edit": cmd_edit, "upscale": cmd_upscale, "models": cmd_models,
     "loras": cmd_loras, "timings": cmd_timings, "describe": cmd_describe,
     "voice-python": lambda _a: print(resolved_voice_python()),
     "jobs": cmd_jobs, "job": cmd_job, "cancel": cmd_cancel}[args.cmd](args)


if __name__ == "__main__":
    if not STORE or not os.path.isdir(STORE):
        sys.exit("FXLLA_STORE is not set or does not exist")
    main()
