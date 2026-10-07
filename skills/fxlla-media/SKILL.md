---
name: fxlla-media
description: Generate and edit images, short videos, and speech locally with the fxlla media tools when the user asks for a picture, a clip, or spoken audio. Choose the model and its options deliberately - call list_media_models and list_loras first - instead of accepting defaults or describing what you would make.
---

# Generate media locally

fxlla exposes local generation over MCP. Produce the artifact rather than
describing it, and choose HOW to produce it: the models differ in ways that
change the result, and picking well is part of the job.

## Start by looking

Two calls cost nothing and prevent most bad output:

- `list_media_models` - every model, the options each one supports, which is
  the default, and `prompt_formats` for models whose prompt is structured.
  Never guess a model name or an option; a model that lacks an option refuses
  it by name and names the models that have it.
- `list_loras` - LoRAs the user has downloaded, plus the built-in styles. If
  one fits what they asked for, **offer it**. Someone who downloaded a LoRA
  wants it used, and it is invisible unless you look.

## Choosing a model

The default (`z-image-turbo`) is fast and fine for most things. Reach past it
deliberately:

- **Text inside the image, posters, layouts** - `ideogram4`, with its JSON
  caption when placement matters (below). `qwen` is the other strong one for
  rendered text; `qwen-2.1` is its newer version and runs 40 steps by default.
- **Design layouts that need transparency** - `ming` (RGBA output).
- **Photographic realism** - `krea2`.
- **Maximum fidelity, slow** - `dev` or `z-image`.
- **Fastest** - `z-image-turbo`, `schnell`, `boogu`, `ernie-turbo`.
- **Long, highly specified prompts** - `fibo`.
- **Composition taken from a reference image** - `controlnet` or
  `z-controlnet` (see Controls).
- **Geometry taken from a photo** - `depth`.

Say which model you chose and why, in one short clause. If the user names one,
use it.

**Cost is part of the choice**, and it is measured, not guessed.
`list_media_models` returns `observed`: real seconds from this machine's own
finished renders, per model and per video stage, with `n` and `s_per_mp`.

- **Never estimate a duration from step counts.** Steps compare a model to
  itself and to nothing else - krea2 and z-image-turbo both run 8 steps and are
  9x apart here. A model that reasoned from steps published a timing table
  wrong by 25x and stated it as fact.
- **A model missing from `observed` has never been timed here.** Say you do not
  know. An invented number is worse than an absent one, because it gets used.
- **Never render to answer a question about cost.** Asked how long each model
  takes, report the measured ones and name the rest as untimed. Benchmarking
  the catalog is an hour of GPU and tens of gigabytes of weights, spent on a
  question nobody meant that literally. Offer to time one if they want it.
- **Cost is time, not money.** Everything runs on this machine; there is no
  per-render charge and no tier. A `$`/`$$`/`$$$` column is a fiction.
- **Re-read `list_media_models` rather than quoting it from earlier in the
  conversation.** It is cheap, it changes as fxlla is updated, and `observed`
  grows every time a render finishes - one session quoted a catalog it read an
  hour earlier and never saw the measurements that had been added since.
- **Do not parse the job files to work out timings.** That is what `observed`
  is, already computed and correct: a hand-rolled version got a video stage
  listed as an image model. From a shell, `fxlla media timings`.
- When someone is iterating - trying a layout, checking a composition - render
  small or on a fast model first and commit to the full size afterwards. Nobody
  asked for eight minutes; they asked for a poster.

## Options worth setting

- `seed` - set one whenever the user might want this image again or a variation
  of it, and report it. Without a seed a good result is unrepeatable.
- `steps` - the cost knob, and roughly linear: `list_media_models` reports
  `default_steps` per model and they span 4 to 50, which on a large canvas is
  the difference between a minute and most of ten. Leave it alone unless
  quality is the complaint. **If speed is the complaint, change model, not
  steps** - a distilled model at its own default beats a slow one starved of
  steps, which pays the same wait for a worse image.
- `preset` - `ideogram4` only, and the ONLY way to set its cost: the preset
  fixes step count and guidance together, which is why that model refuses
  `steps` and `guidance` outright. `V4_TURBO_12` to iterate on a layout,
  `V4_DEFAULT_20` normally, `V4_QUALITY_48` when the typography has to land.
- `guidance` - how literally the prompt is followed. Raise it when the model
  drifts from the request, lower it when the output looks stiff or over-baked.
- `negative` - what must NOT appear: text, watermark, extra fingers, blur.
  Only on models that have guidance, and that is not a formality - a distilled
  model runs with CFG off, which makes the sampler discard the negative prompt
  entirely. `list_media_models` says which; a model without it refuses the
  option instead of ignoring it. **On those models the fix for a stray
  watermark or hallucinated text is a different seed**, not a stronger
  negative prompt.
- `aspect` OR `width`/`height`, never both - they conflict and the call is
  refused. Aspect for standard framing, dimensions for exact pixels. Some
  models only accept sizes on a grid; `list_media_models` reports `dim_step`
  for those, and 1080 wide is not a multiple of 16.
- `output` - where the file goes. A path names it, a directory has it named
  inside. **Set it whenever the user says where they want it**; without it the
  file lands in the media directory and they will go looking in the wrong
  place.
- `init_image` + `strength` - start from an existing image instead of from
  noise, keeping `strength` of it (0 to 1, default 0.4). See below: this is
  how you chain.
- `loras` - `"path,0.8"`, the bare filename `list_loras` shows, or a
  HuggingFace repo id; the scale after the comma is optional, and the option
  repeats. `lora_style` picks a built-in style.

## Ideogram 4 and placed text

`ideogram4` accepts a JSON object as its prompt, which is how you say WHERE
things go instead of hoping. Get the schema and a worked example from
`list_media_models` under `prompt_formats`. Two traps: a bbox is
`[y_min, x_min, y_max, x_max]` - **Y first** - as integers in a 0..1000 space,
and hex colors must be UPPERCASE. Each `text` element carries the literal
characters to render.

Plain prose still works. Use JSON when the user cares about layout: a poster, a
label, a sign, text in a particular corner. A malformed caption is rejected
before the render starts with the problem named, so fix that field rather than
falling back to prose.

## Getting more resolution

Two different things, and they are not interchangeable:

- `pid_decode` replaces the VAE decode with NVIDIA's pixel-diffusion decoder
  and emits **4x the size you asked for**, inside the same render. Ten of the
  models support it; `list_media_models` says which. If you use it, **do not
  also call `upscale_image`** - you already upscaled.
- `upscale_image` (SeedVR2) is a separate pass over a finished file, and works
  on any image including ones fxlla did not make.

The trap worth knowing: `pid_degrade_sigma` defaults to `0.0`, and that is the
input PiD saw **least** during training - it can invent texture on smooth areas
like skin. If a face or a sky comes back over-detailed, pass `0.2`. The valid
range is 0 to 0.8 and anything else is refused before the render.

First use downloads about 8 GB, one repository gated, so it is refused with the
size until the user authorises it or `fxlla pull media:pid` has run.

## Draft first, then refine

The cheapest way to be sure before spending: render small and fast, look at
whether the composition is right, then feed that result back as `init_image`
at full size with `strength` around 0.3-0.5. The second pass keeps the layout
and adds the detail.

    generate_image  prompt, model z-image-turbo, 512x512, seed 7
    generate_image  same prompt, same seed, 1024x1024,
                    init_image <the first path>, strength 0.45

Measured here: 19 s for the draft, 32 s for the refinement. The point is not
the total - it is that a composition you do not want is rejected after 19
seconds instead of after a full-size render, and on a slow model that gap is
minutes. **Keep the seed** across both passes; changing it changes the image
you were refining.

Use the same idea before anything expensive: a poster on `ideogram4` or a
photograph on `krea2` is worth checking as a `z-image-turbo` sketch first,
even though the final model is different - layout and framing carry over even
when the rendering does not.

## Controls, depth, and chaining

`controls` is repeatable, so several controls STACK in one call. The two
families take different forms and `list_media_models` says which: `controlnet`
takes `"image.png"` or `"image.png,checkpoint"`, `z-controlnet` takes
`"type:path[:strength]"` such as `"pose:pose.png:0.8"`.

To copy the geometry of a photo: run `depth` with `init_image` and
`save_depth: true`, which returns the derived depth map's path on a second
line; then pass that map as a control to a following generation. fxlla exposes
each step - sequencing them is yours.

## Long renders: submit, report, move on

Every render runs in the background and the call returns a **job id
immediately**. Nothing here is fast - the quickest measured render is under a
minute and a poster took eight - so:

**Submit, tell the user the job id and roughly what it will take, and finish
your turn.** Do not sit in a `media_job_status` loop. While you poll, the
person you are working for cannot ask you anything else; their messages queue
up behind a render you are not making any faster by watching it.

Pick the result up on the next turn, or when they ask, or after doing something
else useful. `media_job_status` answers instantly and `list_media_jobs` shows
everything in flight. On macOS the finished job also posts a desktop
notification, so the user learns it landed without asking you.

If your client can run a command in the background and wake you when it exits,
that is strictly better: run `fxlla media image ...` that way and you will be
told the moment it finishes, instead of either waiting or being asked.

Two things never to do:

- **Never resubmit because a call did not hand back a path.** It starts a
  second render competing with the first for the same GPU and both get slower.
  One session did this four times with the same poster.
- **Never report a path you have not seen.** Until the job says `done` there is
  no file.

`cancel_media_job` stops one. Jobs run one at a time, so a submission may sit
in `queued` while another finishes. That is expected.

Pass `async: false` only when you specifically need the path in the same call
and know the render is short; a long one will outlast your own timeout.

## Video

**Video is the most expensive thing here**, and its cost is the product of four
things: `stage`, the step count under it, the resolution, and the duration -
every frame gets sampled, so twice the seconds is twice the work. Iterate short
and on `distilled`, then commit to the real length.

`stage` is the main knob: `distilled` (fast default), `one-stage` (better at
small sizes), `two-stage`, `two-stages-hq` (slowest). Each has its own step
control and ignores the other - `one-stage` takes `steps`, the rest take
`stage1_steps` and `stage2_steps`. Passing the wrong one is refused and names
the right one. `list_media_models` reports all of it under `video`.

`generate_video` takes `seconds` for duration - prefer it over `frames`, and do
not pass both. It returns the path plus the MEASURED duration, frame count and
fps: report those, never a duration computed from your own request.

For a video built from existing images, pass `images`: one anchors the opening
frame, two anchor both ends, which is what makes a transition between two
stills. Describing the images in the prompt instead produces a different video
that merely resembles them.

## Weights and consent

Weights download on first use and are large - tens of gigabytes per model, and
some need two repositories. A render whose weights are missing is refused with
the size, and that refusal is correct: present the size to the user and let
them decide, then pre-fetch with `fxlla pull media:<alias>` or pass the
authorization. Never authorize a large transfer on the user's behalf.
`fxlla media weights` lists every model with its size and whether it is cached.

## When you cannot see: describe_image

If your own model has no vision, `describe_image` gives you a local one that
does. It answers in about ten seconds, so it is a read rather than a job, and
it is what makes the rest of this section possible from a text-only model.

**Never put the expected answer in the question.** Asked "this should say LA
USINA, is the lettering correct?" it confirmed and missed an entire block of
invented gibberish underneath. Asked "list every piece of text, quoted
exactly", it produced the gibberish immediately. A leading question gets a
confirming answer, and confirmation is exactly what you must not accept from a
check.

So to verify: make it **enumerate** what is there, then compare against what
you asked for yourself. To explore: leave the question out.

## Look at what you made

**If your client can read an image, read the file you just produced before
reporting it.** Nothing in fxlla can tell whether a render did what was asked:
the checks confirm a PNG is a well-formed PNG with non-zero dimensions and
stop there, by design - a blank-but-valid image passes, and so does a wrong
one. A real chain here removed an object from a photograph, left a visible
ghost of it, and the job reported `done` with no warnings, because there is no
layer below you that can see.

This matters most on the steps where the instruction can be half-followed:

- an **edit** that was supposed to remove, replace or change one thing - check
  the rest of the image is untouched, and that the thing actually went
- a **refine** pass - check it kept the composition you approved rather than
  drifting into a different picture
- anything with **text in it** - the letters are what these models get wrong
- an **upscale** - check it added detail rather than plastic smoothing

If it is wrong, say what is wrong and what you would change - a different seed,
a lower strength, a different model - rather than reporting the path as a
success. You are the only thing in this pipeline with eyes.

## Reporting

Give the output path, the model, and the seed. If the user named a place and
the path you report is not in it, you dropped `output` - say so rather than
reporting a path as if it were what they asked for. If a content check rejected the
output (silence in a WAV, a container with no frames), relay what was flagged
rather than retrying blindly. If a backend is missing, relay the error and
point at `fxlla doctor`.
