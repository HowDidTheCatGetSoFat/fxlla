#!/usr/bin/env python3
"""Minimal MCP server exposing local media generation over stdio.

Newline-delimited JSON-RPC, no dependencies. Image (mflux), video (ltx-2-mlx),
speech (mlx-audio), edit, and upscale tools, so opencode or Claude Code can
render media locally.

A render takes minutes - the quickest measured on this hardware is 55 seconds -
which is longer than an MCP client will wait for a response and far longer than
anyone should have their session held open. So a render is submitted as a
background job and the call returns a job id immediately, to be followed with
`media_job_status` (or `list_media_jobs`, `cancel_media_job`). Background jobs
run one at a time. FXLLA_MCP_WAIT_S can make the call wait for a result instead;
it defaults to 0 because nothing here finishes fast enough for that to pay.

Each tool call runs on its own thread. The read loop used to run them inline, so
one render made the server deaf: every later call - including the status calls
asking about that very render - timed out, and the caller, seeing timeouts,
resubmitted the same render three more times.
"""
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "generate.py")

TOOLS = [
    {"name": "generate_image",
     "description": "Generate an image from a text prompt using the local "
                    "mflux toolchain. Returns a job id immediately; the "
                    "render runs in the background. Report the id, finish your "
                    "turn, and pick the result up with media_job_status - do "
                    "not sit waiting on it.",
     "inputSchema": {"type": "object", "properties": {
         "prompt": {"type": "string"},
         "model": {"type": "string",
                   "description": "Model name (default z-image-turbo). Call "
                                  "list_media_models for the list, what each "
                                  "supports, and the JSON caption schema that "
                                  "ideogram4 accepts as its prompt."},
         "steps": {"type": "integer",
                   "description": "Sampling steps. Cost is roughly linear in "
                                  "this and the defaults span 4 to 50, so "
                                  "check default_steps in list_media_models "
                                  "before raising it - and pick a distilled "
                                  "model instead of cutting steps on a slow "
                                  "one, which trades quality for the same "
                                  "wait."},
         "seed": {"type": "integer"},
         "width": {"type": "integer"},
         "height": {"type": "integer"},
         "aspect": {"type": "string", "description": "e.g. 1:1, 16:9"},
         "quantize": {"type": "integer", "enum": [3, 4, 5, 6, 8]},
         "negative": {"type": "string",
                      "description": "What the image must NOT contain."},
         "prompt_file": {"type": "string",
                         "description": "Read the prompt from this file "
                                        "instead of the prompt argument."},
         "init_image": {"type": "string",
                        "description": "Start from this image (img2img). Pair "
                                       "it with strength."},
         "strength": {"type": "number",
                      "description": "How much of init_image survives, 0 to 1 "
                                     "(default 0.4). This is what makes a "
                                     "second pass useful: render small and "
                                     "fast first, then refine that result at "
                                     "full size with 0.3-0.5 to add detail "
                                     "without losing the composition. Near 0 "
                                     "barely changes the input; near 1 ignores "
                                     "it and you may as well not pass one."},
         "loras": {"type": "array", "items": {"type": "string"},
                   "description": "LoRAs to apply. Each entry is a path, a "
                                  "bare filename as list_loras reports it, or "
                                  "a HuggingFace repo id, optionally with a "
                                  "scale after a comma: \"style.safetensors,0.8\". "
                                  "See list_media_models for which models "
                                  "support them."},
         "lora_style": {"type": "string",
                        "description": "A built-in LoRA style (storyboard, "
                                       "portrait, identity, ...); see "
                                       "list_loras."},
         "guidance": {"type": "number",
                      "description": "Guidance scale. Higher follows the prompt "
                                     "more literally. Model default is usually "
                                     "3.5, 10 for depth."},
         "pid_decode": {"type": "boolean",
                        "description": "Decode with NVIDIA's pixel-diffusion "
                                       "decoder instead of the VAE. The output "
                                       "is 4x the size you asked for, so it is "
                                       "an upscale inside the render rather "
                                       "than a step after it - do not also "
                                       "call upscale_image. It downloads ~8GB "
                                       "of its own weights on first use, one "
                                       "of them gated, so the first call is "
                                       "refused with the size unless they are "
                                       "cached. list_media_models says which "
                                       "models support it."},
         "pid_degrade_sigma": {"type": "number",
                               "description": "With pid_decode: noise the "
                                              "latent to this sigma first, 0 "
                                              "to 0.8. The default 0.0 is the "
                                              "input PiD saw LEAST in "
                                              "training, which shows up as "
                                              "invented texture on smooth "
                                              "areas like skin - try 0.2 if "
                                              "you see that."},
         "preset": {"type": "string",
                    "enum": ["V4_TURBO_12", "V4_DEFAULT_20", "V4_QUALITY_48"],
                    "description": "ideogram4 only, and the ONLY way to set its "
                                   "cost: the preset fixes both step count and "
                                   "guidance schedule, which is why that model "
                                   "takes neither steps nor guidance. 12 steps "
                                   "for a draft, 20 the default, 48 when the "
                                   "typography has to be right."},
         "controls": {"type": "array", "items": {"type": "string"},
                      "description": "Control inputs, repeatable to STACK "
                                     "several (e.g. depth + canny). Form "
                                     "depends on the model: \"image.png\" or "
                                     "\"image.png,checkpoint\" for the "
                                     "controlnet model, \"type:path[:strength]\" "
                                     "(e.g. pose:pose.png:0.8) for "
                                     "z-controlnet. Use list_media_models to "
                                     "see which a model takes."},
         "controlnet_strength": {"type": "number",
                                 "description": "Global multiplier over all controls."},
         "depth_image": {"type": "string",
                         "description": "Use this depth map instead of deriving one."},
         "save_depth": {"type": "boolean",
                        "description": "Also write the derived depth map. Its "
                                       "path is returned on a second line, so "
                                       "a later control step can use it."},
     }, "required": ["prompt"]}},
    {"name": "generate_video",
     "description": "Generate a short video from a text prompt using the local "
                    "ltx-2-mlx (LTX-2.3) toolchain. Returns the path to the MP4 "
                    "and the MEASURED duration, frame count and fps of the "
                    "result - report those, do not compute duration from the "
                    "request.",
     "inputSchema": {"type": "object", "properties": {
         "prompt": {"type": "string"},
         "stage": {"type": "string",
                   "enum": ["distilled", "one-stage", "two-stage", "two-stages-hq"],
                   "description": "The pipeline, and the main cost knob. "
                                  "distilled is the fast default; "
                                  "two-stages-hq is the slowest. See "
                                  "list_media_models under video.stages for "
                                  "what each does and its step counts."},
         "steps": {"type": "integer",
                   "description": "Denoising steps, one-stage ONLY (default "
                                  "8). The other stages ignore it and take "
                                  "stage1_steps instead."},
         "stage1_steps": {"type": "integer",
                          "description": "Stage 1 steps for the two-stage "
                                         "pipelines (default 30, 15 for "
                                         "two-stages-hq). The cost knob for "
                                         "every stage except one-stage."},
         "stage2_steps": {"type": "integer",
                          "description": "Stage 2 steps for the two-stage "
                                         "pipelines (default 3)."},
         "seconds": {"type": "number",
                     "description": "Target duration in seconds. Prefer this over "
                                    "frames; do not pass both."},
         "frames": {"type": "integer"},
         "frame_rate": {"type": "integer", "description": "Default 24 (trained fps)."},
         "width": {"type": "integer"},
         "height": {"type": "integer"},
         "seed": {"type": "integer"},
         "images": {"type": "array", "items": {"type": "string"},
                    "description": "Reference images for image-to-video, each "
                                   "\"path\" or \"path,frame,strength\". "
                                   "One anchors the opening frame; two anchor "
                                   "both ends, which is how a transition "
                                   "between two stills is produced. Describing "
                                   "the images in the prompt instead makes a "
                                   "different video that merely resembles them."},
     }, "required": ["prompt"]}},
    {"name": "generate_speech",
     "description": "Synthesize speech from text using the local mlx-audio "
                    "(Chatterbox) toolchain. Returns the path to the WAV.",
     "inputSchema": {"type": "object", "properties": {
         "text": {"type": "string"},
         "ref": {"type": "string", "description": "Reference voice wav (timbre)."},
         "lang": {"type": "string", "description": "Language code, e.g. en, es, pt."},
         "speed": {"type": "number"},
     }, "required": ["text"]}},
    {"name": "edit_image",
     "description": "Edit an existing image from a text instruction using the "
                    "local mflux qwen-edit toolchain. Returns the path to the "
                    "written PNG.",
     "inputSchema": {"type": "object", "properties": {
         "prompt": {"type": "string", "description": "The edit instruction."},
         "image": {"type": "string", "description": "Path to the input image."},
         "seed": {"type": "integer"},
         "quantize": {"type": "integer", "enum": [3, 4, 5, 6, 8]},
     }, "required": ["prompt", "image"]}},
    {"name": "upscale_image",
     "description": "Upscale an image using the local mflux seedvr2 "
                    "diffusion super-resolution. Returns the path to the PNG.",
     "inputSchema": {"type": "object", "properties": {
         "image": {"type": "string", "description": "Path to the input image."},
         "scale": {"type": "string",
                   "description": "Target shortest edge in pixels or a factor, e.g. 2x."},
     }, "required": ["image"]}},
    {"name": "media_job_status",
     "description": "Status of a background media job: the finished record "
                    "with its output path or its error, or a line saying it is "
                    "still going. Answers IMMEDIATELY. If it is still running "
                    "that is the render working, not a failure - report the "
                    "job id and finish your turn rather than calling this in a "
                    "loop, and never resubmit, which only starts a second "
                    "render competing with the first.",
     "inputSchema": {"type": "object", "properties": {
         "job_id": {"type": "string"},
         "wait_s": {"type": "number",
                    "description": "Seconds to block before answering. Default "
                                   "0, and leave it there: renders here run "
                                   "from about a minute to eight, so waiting "
                                   "holds up the person you are working for "
                                   "and changes nothing. Capped at the "
                                   "server's window (FXLLA_MCP_WAIT_S)."},
     }, "required": ["job_id"]}},
    {"name": "describe_image",
     "description": "Look at an image and get back what is in it, as text, from "
                    "a local vision model. The inverse of the generators, and "
                    "the way a model that cannot see gets to check its own "
                    "work: `done` on a render means the file was written, not "
                    "that it is right. Answers in about ten seconds - it is a "
                    "read, not a render, so there is no job to follow.\n"
                    "NEVER put the expected answer in the question. Asked "
                    "'this should say LA USINA, is the lettering correct?' the "
                    "model confirmed it and missed a whole block of gibberish "
                    "the generator had invented; asked to list every piece of "
                    "text, it quoted the gibberish immediately. To verify, make "
                    "it ENUMERATE what is there, then compare yourself.",
     "inputSchema": {"type": "object", "properties": {
         "image": {"type": "string", "description": "Path to the image."},
         "question": {"type": "string",
                      "description": "What to ask about it. Leave it out for a "
                                     "neutral description that quotes any text. "
                                     "Phrase it as 'list/describe what is "
                                     "there', never as 'is X correct?'."},
     }, "required": ["image"]}},
    {"name": "list_loras",
     "description": "LoRAs found on this machine and the built-in styles. Each entry carries base_model: the architecture it was trained for, prefixed with ~ when inferred from the weights rather than declared. Apply one only to its own base - a krea2 adapter does nothing useful on z-image. Check here before "
                    "generating: if a LoRA fits what the user asked for, offer "
                    "it rather than ignoring what they already downloaded.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "list_media_models",
     "description": "THE answer to \"how long does each model take here\" and "
                    "\"which one should I use\" - call it instead of reading "
                    "the job files yourself, which one caller did in forty "
                    "lines of shell and got a video stage listed as an image "
                    "model. Every image model and every video stage, what each "
                    "supports, and what each COSTS. `observed` is the "
                    "important part: real measured seconds from this machine's "
                    "own finished jobs, per model and per video:<stage>, with "
                    "n and s_per_mp. Quote those. Do NOT estimate a duration "
                    "from step counts - steps compare a model only to itself, "
                    "two 8-step models here run 9x apart, and estimating that "
                    "way produced a table wrong by 25x. A model missing from "
                    "`observed` has never been timed here: say so instead of "
                    "guessing. Also: per-model options, default_steps, "
                    "dim_step where sizes sit on a grid, presets where a model "
                    "has them instead of steps, the video stages, the LoRA "
                    "styles, and prompt_formats - ideogram4's JSON caption "
                    "schema with an example.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "list_media_jobs",
     "description": "List background media jobs, newest first.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "cancel_media_job",
     "description": "Cancel a queued or running background media job.",
     "inputSchema": {"type": "object", "properties": {
         "job_id": {"type": "string"},
     }, "required": ["job_id"]}},
]

# Every generator tool takes an async flag, and it defaults to on: holding a
# call open for a render that takes minutes only produces a client timeout, and
# a timeout reads as a failure, so the caller renders it again.
_ASYNC_DOC = ("Default true, and leave it: the render runs in the background "
              "and this returns a job id immediately, so the conversation is "
              "not held hostage to a render that takes minutes. Follow it with "
              "media_job_status. Set false only to hold the call open until "
              "the file is written, which a long render will outlast.")
_SKIP_QUALITY_DOC = ("Accept output that fails the content checks (silence, a "
                     "container with no frames). Use only after a check rejected "
                     "something you actually wanted.")
# The CLI has taken --output on every generator from the start; none of the MCP
# tools exposed it, so "save it in ~/Downloads" was accepted and then quietly
# ignored - the file landed in the media directory and the answer said nothing.
_OUTPUT_DOC = ("Where to write it: a file path, or a directory to have the "
               "file named inside it. Defaults to the media output directory. "
               "Pass this whenever the user says where they want it.")
for _tool in TOOLS:
    if _tool["name"].startswith(("generate_", "edit_", "upscale_")):
        _tool["inputSchema"]["properties"]["output"] = {
            "type": "string", "description": _OUTPUT_DOC}
        _tool["inputSchema"]["properties"]["async"] = {
            "type": "boolean", "description": _ASYNC_DOC}
        _tool["inputSchema"]["properties"]["skip_quality"] = {
            "type": "boolean", "description": _SKIP_QUALITY_DOC}

_IMAGE_FLAGS = [("model", "--model"), ("steps", "--steps"), ("seed", "--seed"),
                ("width", "--width"), ("height", "--height"), ("aspect", "--aspect"),
                ("quantize", "--quantize"), ("negative", "--negative"),
                ("prompt_file", "--prompt-file"), ("guidance", "--guidance"),
                ("controls", "--control"),
                ("controlnet_strength", "--controlnet-strength"),
                ("depth_image", "--depth-image"),
                ("init_image", "--init-image"), ("strength", "--strength"),
                ("loras", "--lora"),
                ("lora_style", "--lora-style"), ("preset", "--preset"),
                ("pid_degrade_sigma", "--pid-degrade-sigma"),
                ("output", "--output")]
# Boolean flags carry no value, so they cannot ride the name/flag table above.
_IMAGE_SWITCHES = [("pid_decode", "--pid-decode")]
_VIDEO_FLAGS = [("stage", "--stage"), ("seconds", "--seconds"),
                ("frames", "--frames"),
                ("frame_rate", "--frame-rate"), ("width", "--width"),
                ("height", "--height"), ("seed", "--seed"), ("model", "--model"),
                # generate.py takes --image (singular, repeatable), mirroring
                # ltx's own flag. --images does not exist in either.
                ("images", "--image"), ("steps", "--steps"),
                ("stage1_steps", "--stage1-steps"),
                ("stage2_steps", "--stage2-steps"), ("output", "--output")]
_VOICE_FLAGS = [("ref", "--ref"), ("lang", "--lang"), ("speed", "--speed"),
                ("model", "--model"), ("output", "--output")]
_EDIT_FLAGS = [("image", "--image"), ("seed", "--seed"),
               ("quantize", "--quantize"), ("output", "--output")]


# Seconds a render may hold the call open before it answers with a job id.
# ZERO by default, which is the whole point: nothing here finishes fast enough
# for waiting to pay. The quickest render measured on this hardware is 55 s and
# a poster took 473, so a wait window only ever spends the caller's turn to
# arrive at the same job id it could have had immediately - and while it waits,
# whoever is driving cannot do anything else. Raise it only if you specifically
# want short renders to come back as a path, and keep it under the calling
# client's own timeout (opencode returned a 45 s call and timed out past ~69).
WAIT_S = float(os.environ.get("FXLLA_MCP_WAIT_S", "0"))


def _exec(cmd, failure):
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    if proc.returncode != 0:
        return "error: " + (proc.stderr.strip() or failure)
    return proc.stdout.strip() or "error: no output"


def _job_record(job_id, wait_s):
    cmd = [sys.executable, MEDIA, "job", str(job_id), "--json"]
    if wait_s:
        cmd += ["--wait", str(wait_s)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout.strip())
    except ValueError:
        return None


def _still_running(job_id, status, note="", elapsed=None):
    """What to say about a job that has not landed yet.

    It tells the caller to STOP, not to wait: holding the conversation open for
    a render that takes minutes blocks whoever is driving it from doing
    anything else, and no amount of polling makes the render faster. The
    expected duration rides along, because a catalog read once at the start of
    a session is still being quoted an hour later."""
    parts = ["job %s is %s" % (job_id, status)]
    if elapsed is not None:
        parts.append("%d s elapsed" % round(elapsed))
    if note:
        parts.append(note)
    return ("%s. Tell the user the job id and how long it is expected to take, "
            "then finish your turn - do NOT wait for it and do NOT submit it "
            "again. Check it with media_job_status when they next ask, or when "
            "you have something else to do first." % ", ".join(parts))


def _spawn(cmd, args, failure):
    """Run one generation and return a path, an error, or a job id.

    Submitted as a background job unless the caller says otherwise, then
    awaited for WAIT_S. A render that outlives the window is reported as a
    running job rather than held open until the client times out - which is
    what used to happen, and the caller could not tell a timeout from a
    failure, so it resubmitted."""
    if args.get("skip_quality"):
        cmd = cmd + ["--skip-quality"]
    # Absent and explicit null both mean "use the default", which is the job
    # path; only a literal false opts out.
    background = args.get("async")
    if background is not None and not background:
        # Explicitly opted out: the caller wants the path in this response and
        # accepts that a long render may outlast its own timeout.
        proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
        if proc.returncode != 0:
            return "error: " + (proc.stderr.strip() or failure)
        return proc.stdout.strip() or "error: no output path returned"
    proc = subprocess.run(cmd + ["--async"], capture_output=True, text=True,
                          env=os.environ)
    if proc.returncode != 0:
        return "error: " + (proc.stderr.strip() or failure)
    # Line one is the id; anything after is what the submission wants to tell
    # the caller, such as how long this has taken here before.
    lines = proc.stdout.strip().splitlines()
    if not lines:
        return "error: no job id returned"
    job_id, note = lines[0].strip(), " ".join(l.strip() for l in lines[1:])
    rec = _job_record(job_id, WAIT_S)
    if rec is None:
        # The job was submitted; only reading it back failed. Returning a bare
        # id here threw away the estimate that came with the submission, which
        # is the one number guaranteed to reach a caller that will not re-read
        # the catalog.
        return _still_running(job_id, "submitted", note)
    status = rec.get("status")
    if status == "done":
        return rec.get("output") or job_id
    if status in ("failed", "cancelled"):
        return "error: job %s %s: %s" % (job_id, status,
                                         (rec.get("error") or "").strip()[-800:])
    return _still_running(job_id, status, note)


def _run(subcmd, positional, flags, args, switches=()):
    value = args.get(positional)
    if not value:
        return "error: %s is required" % positional
    cmd = [sys.executable, MEDIA, subcmd, str(value)]
    for key, flag in flags:
        val = args.get(key)
        if val is None:
            continue
        # A list argument repeats its flag rather than being stringified: one
        # --image per reference, which is what anchoring several frames needs.
        if isinstance(val, list):
            for item in val:
                cmd += [flag, str(item)]
        else:
            cmd += [flag, str(val)]
    for key, flag in switches:
        if args.get(key):
            cmd.append(flag)
    if args.get("save_depth"):
        cmd.append("--save-depth-map")
    return _spawn(cmd, args, "generation failed")


def run_job_status(args):
    job_id = args.get("job_id")
    if not job_id:
        return "error: job_id is required"
    # Capped, for the same reason the render is: a request to wait 120 s came
    # back to the model as a timeout, not as a status, and told it nothing.
    # An explicit 0 still means "answer now" rather than the default wait.
    raw = args.get("wait_s")
    wait_s = WAIT_S if raw is None else min(float(raw), WAIT_S)
    rec = _job_record(job_id, wait_s)
    if rec is None:
        return "error: unknown job: %s" % job_id
    started, finished = rec.get("started"), rec.get("finished")
    if rec.get("status") in ("queued", "running"):
        elapsed = (time.time() - started) if started else None
        return _still_running(job_id, rec["status"], elapsed=elapsed)
    # How long it actually took, computed here. Without it a caller wanting to
    # report the wait shelled out to python to subtract two timestamps - and
    # this is the one place a real measurement is guaranteed to reach a caller
    # that never re-reads the catalog.
    if started is not None and finished is not None and finished > started:
        rec = dict(rec, elapsed_s=round(finished - started, 1))
    # `done` means the file was written, not that it is right. The checks below
    # this line confirm a PNG is a well-formed PNG and stop there: an edit that
    # left a ghost of what it was told to remove reported done, clean, no
    # warnings. Whoever called this is the only part of the pipeline with eyes.
    if rec.get("status") == "done" and rec.get("output"):
        rec = dict(rec, verify="Read this file and check it did what was "
                                "asked. `done` means written, not correct.")
    # A `warnings` entry means the backend honoured the request differently
    # from how it was asked - relay it rather than reporting a clean success.
    return json.dumps(rec)


def run_describe(args):
    image = args.get("image")
    if not image:
        return "error: image is required"
    cmd = [sys.executable, MEDIA, "describe", "--image", str(image)]
    if args.get("question"):
        cmd.append(str(args["question"]))
    # Synchronous on purpose: measured at about ten seconds, so the job queue
    # would be ceremony. It also must NOT go through _spawn, which would submit
    # a background job and hand back an id for something already answered.
    return _exec(cmd, "could not describe the image")


def run_list_loras(_args):
    return _exec([sys.executable, MEDIA, "loras", "--json"],
                 "could not list loras")


def run_list_models(_args):
    """Capabilities, from the catalog. Without this a model reads the source to
    find out what it can do: one opened generate.py and media_mcp.py ten times
    in a single session doing exactly that."""
    return _exec([sys.executable, MEDIA, "models", "--json"],
                 "could not list models")


def run_list_jobs(_args):
    return _exec([sys.executable, MEDIA, "jobs", "--json"], "could not list jobs")


def run_cancel_job(args):
    job_id = args.get("job_id")
    if not job_id:
        return "error: job_id is required"
    return _exec([sys.executable, MEDIA, "cancel", str(job_id)], "unknown job")


def run_generate(args):
    return _run("image", "prompt", _IMAGE_FLAGS, args, _IMAGE_SWITCHES)


def run_generate_video(args):
    result = _run("video", "prompt", _VIDEO_FLAGS, args)
    # Hand back what the file IS, measured. A caller that computes duration
    # from its own request can be wrong and sound certain: one reported 49
    # frames at 24 fps as "about 10 seconds" (2.04) and declared an unmet
    # 4-8 second requirement satisfied. Async returns a job id, not a path,
    # so there is nothing to measure yet.
    if result.startswith("error:") or not os.path.isfile(result):
        return result
    sys.path.insert(0, HERE)
    import quality  # noqa: E402  (local module, beside this file)
    facts = quality.video_facts(result)
    if not facts.get("duration_s"):
        return result
    return ("%s\nmeasured: %.2fs, %s frames, %s fps, %sx%s"
            % (result, facts["duration_s"], facts.get("frames", "?"),
               facts.get("fps", "?"), facts.get("width", "?"),
               facts.get("height", "?")))


def run_generate_speech(args):
    return _run("voice", "text", _VOICE_FLAGS, args)


def run_edit(args):
    # qwen-edit needs both the instruction (positional) and the input image.
    if not args.get("image"):
        return "error: image is required"
    return _run("edit", "prompt", _EDIT_FLAGS, args)


def run_upscale(args):
    # seedvr2 has no prompt; the image is its required input, passed as --image.
    image = args.get("image")
    if not image:
        return "error: image is required"
    cmd = [sys.executable, MEDIA, "upscale", "--image", str(image)]
    scale = args.get("scale")
    if scale is not None:
        cmd += ["--scale", str(scale)]
    if args.get("output") is not None:
        cmd += ["--output", str(args["output"])]
    return _spawn(cmd, args, "upscale failed")


def handle(msg):
    method = msg.get("method")
    mid = msg.get("id")
    if method == "initialize":
        return _ok(mid, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fxlla-media", "version": "0.1.0"},
        })
    if method == "tools/list":
        return _ok(mid, {"tools": TOOLS})
    if method == "tools/call":
        params = msg.get("params", {})
        tool = params.get("name")
        runner = {"generate_image": run_generate,
                  "generate_video": run_generate_video,
                  "generate_speech": run_generate_speech,
                  "edit_image": run_edit,
                  "upscale_image": run_upscale,
                  "media_job_status": run_job_status,
                  "describe_image": run_describe,
                  "list_media_models": run_list_models,
                  "list_loras": run_list_loras,
                  "list_media_jobs": run_list_jobs,
                  "cancel_media_job": run_cancel_job}.get(tool)
        if runner:
            # A runner that raises on its own thread would drop the response
            # and leave the caller waiting on a request that can never arrive -
            # indistinguishable from the hang this server was fixed to avoid.
            try:
                text = runner(params.get("arguments", {}))
            except Exception as exc:  # noqa: BLE001 - report, never drop
                text = "error: %s: %s" % (type(exc).__name__, exc)
            return _ok(mid, {"content": [{"type": "text", "text": text}]})
        return _err(mid, -32601, "unknown tool")
    if method and method.startswith("notifications/"):
        return None
    if mid is not None:
        return _err(mid, -32601, "method not found")
    return None


def _ok(mid, result):
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _err(mid, code, message):
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


_WRITE = threading.Lock()


def _emit(resp):
    if resp is None:
        return
    with _WRITE:
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


def main():
    running = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        # Tool calls go to their own thread so a render never stops the server
        # from reading. Responses carry the request id, so the client matches
        # them whatever order they arrive in. initialize and tools/list stay
        # inline: they are instant, and a client waits for initialize before
        # sending anything else.
        if msg.get("method") == "tools/call":
            worker = threading.Thread(target=lambda m=msg: _emit(handle(m)),
                                      daemon=True)
            running.append(worker)
            worker.start()
            running = [t for t in running if t.is_alive()]
        else:
            _emit(handle(msg))
    # Stdin ended. Finish what is in flight first: returning here would let the
    # interpreter exit and take the daemon threads with it, so calls already
    # accepted would never be answered.
    for worker in running:
        worker.join()


if __name__ == "__main__":
    main()
