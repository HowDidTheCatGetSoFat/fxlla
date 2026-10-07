# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Image models `qwen-2.1` (Qwen-Image 2.1) and `ming` (Ming-Image 0.1 Design,
  RGBA output), both new in mflux 0.20 and 0.21.

- `qwen3.5-9b` and `gemma-4-12b-qat` in the catalog. The 9B is stock Qwen3.5,
  taken at Q6_K from the same publisher as the 4B and 27B already here, so the
  three sizes differ in size and nothing else - which is the only reason an
  abliterated-versus-base comparison across them means anything. The Gemma is
  Google's own QAT build: the 4-bit weights are what it was tuned to end up at
  rather than what it was rounded to afterwards. Google also ships that QAT in
  compressed-tensors, which llama.cpp cannot read; this is the GGUF one.

- Two 27B models on Qwen3.5 in the catalog, chosen because together they are an
  experiment. `qwythos` was trained without alignment; `abliterated-27b` is the
  same base with the refusal direction orthogonalized out. Same family, opposite
  routes to the same property, which is what makes the pair worth having.
  The catalog lists repository ids, not weights - whoever pulls one accepts that
  model's terms - so a row here carries no license of its own. `qwythos` is
  Apache-2.0 on the weights. Its own card says it shares a curriculum with the
  publisher's 9B, and that 9B's card describes the curriculum as distilled from
  Claude traces - the note states it that way, one model removed, because the
  27B card does not say it directly.
- `candor`, a fifth eval dimension: whether a model declines what cannot be
  done instead of complying. Four traps with four controls - a fabricated
  stdlib symbol against a real one, a self-contradicting specification against
  a satisfiable one, a request to process data never given against one that
  was. The pairing is the design: a model answering FAKE to everything scores
  perfectly on the traps alone. Format failure shows up on control and trap
  together for the three exact-match pairs; the impossible/possible pair is the
  exception, since its control runs the code and extraction drops prose, so a
  chatty model can fail that trap by itself. Refusing correctly is a
  capability, and nothing here had ever asked for it - which is exactly what a
  model with its refusal direction surgically removed should be tested on.

- `FXLLA_ROPE_STRETCH=1` serves the window a model advertises instead of the
  one it was trained for. Only meaningful where rope scaling is baked into the
  file - on a YaRN factor of 4 that is 1048576 against 262144 - and off by
  default, because the stretch buys reach at the cost of attention quality
  across a window almost nobody fills, and the KV cache for a 27B at 1M is
  about 64 GB allocated up front.
  It only raises the ceiling. Whether the scaling stays enabled is decided by
  what is actually served: above the trained window it is the reason the
  context exists, at or below it there is nothing bought. So asking for the
  stretch while `FXLLA_CTX` still caps under the trained length gets neither,
  which is the honest outcome rather than a special case.

- The eval's speed numbers were measuring the transport. Streamed probes were
  read in 8 KB blocks, and `read(8192)` blocks until it has that many bytes or
  the response ends - so every answer shorter than 8 KB, which is most of them,
  was timed as if it had arrived in one piece. TTFT became the whole-response
  time and the decode window collapsed to microseconds, which is how one run
  reported 1,865,386 tokens/s and pulled its own median up with it. Reading a
  line at a time is the granularity SSE actually has. On the same small model:
  TTFT 110 ms to 52 ms, and the rate across probes tightened from 667..892 to
  609..611 - more accurate and far less noisy at once.
  A rate is now also reported as unmeasurable rather than invented when the
  tokens did not arrive spread over time. Harness version bumped to 4, so
  speed numbers either side of this do not look comparable.

- `fxlla do` reads the INPUT of an edit as well, and reports what the
  description gained or lost besides what was asked. The gap it closes: the
  first real edit turned a wall yellow as requested and the ground from
  concrete to wooden, and nothing said so, because the check only ever asks
  whether what was PROMISED turned up. Costs one extra look per edit, and only
  when there is budget left for it. Reported, never acted on - two descriptions
  differing is a fact about the descriptions, and the same eyes can word the
  same picture differently, so the report says so rather than calling it a
  defect.

### Changed
- The image backend moves from mflux-cv, now retired, to upstream mflux 0.21.0
  (mflux-community), where its fixes were merged. `fxlla setup --media`
  installs `mflux==0.21.0`. Three things the fork had and upstream does not:
  `mage-flow` is gone from the catalog; `--aspect` is now resolved by fxlla to
  a one-megapixel width and height on the 16-pixel grid, since mflux has no such
  flag; and the catalog follows what upstream reports through
  `mflux-capabilities` - boogu and fibo take no LoRA, and ernie-turbo has no
  negative prompt or guidance, because upstream discards them.

- `/v1/models` answers in 30 ms where it took 4.7 seconds, and an editor asks
  for it at startup. The cost was never the sizes - `du` over the whole store
  is 30 ms - it was the GGUF headers. Deriving a serve plan asked four separate
  readers for one fact each, and each reopened the file and scanned until it
  found ITS key; a key that is absent is only known to be absent at the END of
  the header, so a model declaring no MTP head and no rope factor paid three
  full walks of a 12 MB metadata block. One walk collects all four facts now,
  which is 2.1 seconds down to 0.68 across fifteen models, and those facts are
  remembered against the file's mtime and size, which takes the second call to
  0.4 ms. The cache holds the FACTS and not the answer: two callers with
  different ceilings must each get their own window, since a reported context
  that is not the one being served is the failure this module exists to
  prevent.

### Fixed
- A projector whose name did not START with `mmproj` was invisible, in five
  places at once. Publishers disagree about where the word goes: one writes
  `mmproj-Model-F16.gguf`, another `Model.mmproj-Q8_0.gguf`. Every glob and
  grep was anchored at the front, so for the second convention the pull left
  the projector in the repo, the launcher would not have passed it, and the
  gateway reported the model could not see - three components agreeing on the
  wrong answer because they shared one wrong assumption. Four models on this
  machine had been fetched blind, counted by asking each repo what it ships
  rather than from memory, which had it as three. The entry-file fallback was
  worse than invisible: it excluded nothing at all, so under a UTF-8 locale it
  could serve the projector AS the model.
  All five sites now match case-insensitively as well, because the pull side
  always did: `grep -i` will fetch a `MMProj` spelling that a glob or a Python
  `in` would not find afterwards, which is the same silent text-only failure
  reached from the other direction. The two that scan a directory for the
  weights sort under C collation on both sides, so bash and Python cannot
  disagree about which file is the model.
- `--quant` matching only a projector was accepted. It downloaded that file,
  wrote a blank entry marker, and printed "Done" with instructions for starting
  a directory that has nothing to start. Non-empty was being read as usable.
- A repo with one build and one projector was refused as ambiguous. Both end in
  `.gguf`, and the ambiguity count read every `.gguf`, so `fxlla pull
  gemma-4-12b-qat` demanded a `--quant` choice between a model and a file that
  is not one. Counted over builds now, and the "Available" list offers builds
  only.
- `fxlla models` stopped listing at the first note containing an apostrophe.
  Fields were trimmed with `xargs`, which trims but also PARSES: "Google's own
  build" is an unterminated quote, xargs exits non-zero, and under `set -e` the
  listing ended there. Four of twenty-three rows were invisible, announced only
  by a stray `xargs: unterminated quote` under the table - including the two
  models added in this release. Trimming is now parameter expansion, which has
  no opinion about the text it trims.

### Changed
- A gguf build carrying a multi-token-prediction head is served with
  `--spec-type draft-mtp`, so it drafts against itself with no second model
  involved. Detected from the header key rather than the filename, because
  publishers ship both builds of the same weights at the same quant and the
  name is a convention. Measured on that pair, same context and flags, same
  prompts: 40.9 against 21.7 tokens/s counting, 31.9 against 20.6 on code,
  28.5 against 20.7 on prose. The plain build is flat across all three because
  it is memory bound; the speculating one tracks how guessable the next tokens
  are, which is the shape the mechanism predicts and the reason to believe the
  numbers. Through the gateway at a 262k window the MTP build does 36.0 on
  code, but that figure has no matched baseline - the head-to-head numbers
  above are the ones to compare, and both sides of those were taken at 32k.
  The flag was verified against the real model before any of this was wired -
  llama.cpp logs "creating MTP draft context against the target model" - since
  a flag existing and a flag working are different claims.
- `FXLLA_KEEP_WARM` reaches the gateway. It said "auto-stop after N idle
  minutes" and was read by the single-model server alone, so the multi-model
  path - the one most people run - freed memory only when the NEXT load would
  not fit, while /health reported an idle counter nothing acted on. Two 27B
  models held 45 GB between them after a quarter of an hour of silence here.
  A reaper unloads a backend past the threshold now, deciding under the lock
  and terminating outside it so a slow exit cannot block an arriving request.
  The value also had to be added to the explicit list `fxlla serve` passes the
  gateway: config.env assigns without exporting, so a user's setting never
  arrived and the gateway silently used its own default. Same omission that
  had swallowed `FXLLA_AGENT_MODEL`, now covered by a test over both.
- `FXLLA_CTX` is a ceiling now, not the window itself. Every gguf model was
  served the same number, which was wrong in both directions: it gave a 27B
  trained to 262k an 8k window, and asked a 7B for more than it had ever seen
  in training. Each model is now served what its own header says it was
  trained for, capped by `FXLLA_CTX`. The shipped default rises from 8192 to
  32768, since a ceiling is not a target; raise it to let a big-window model
  use its full window, and pay for it in RAM, because llama-server allocates
  the KV cache up front whether or not it is filled. Measured here: Qwythos
  27B at 262144 is 38.9 GB resident, 20.9 of it weights.
  Two details make this more than reading a number. A model shipped with rope
  scaling advertises the STRETCHED window as its context length - one here
  says 1048576, which is 262144 multiplied by a YaRN factor of 4 - so the
  window it was actually trained for is the one recorded beside the factor,
  and that is what gets served. And because llama.cpp honours baked-in scaling
  at every size, serving at or below the trained window now passes
  `--rope-scaling none`: below the original length the stretch is pure loss,
  which is what the model's own author warns about.
  `bin/fxlla _backend` and the gateway's `/v1/models` read this through the
  same function, so the window opencode's context meter and auto-compaction
  run against is the one the backend is actually serving.

### Added
- `fxlla do` chooses what to do, not just what to do it with. Naming an image
  that exists makes editing available, so "make the wall in shot.png yellow"
  plans an edit and "a red bicycle" plans a render, from the same command with
  nothing to select. Editing is the second operation deliberately: it produces
  an image, so the look and the check work on it unchanged, and it is where
  verifying earns the most - the failure that motivated `describe_image` in the
  first place was an edit that removed an object, left a visible ghost of it,
  and passed every mechanical check.
  What the planner says about the input path is never trusted. It is asked to
  copy one back, but fxlla extracts the image paths from the request itself and
  refuses any choice that is not one of them - the same reflex as withholding a
  failed model rather than asking a model not to pick it. A path a model repeats
  has to be right character by character, and it has already been shown here
  that being told a constraint in prose is not the same as being held to one.
  The comparison asks the filesystem rather than the strings, so the same file
  named in a different case (this machine's volume is case-insensitive) or
  through a symlink still matches. Only `seed` carries
  over to an edit; steps, guidance and aspect mean nothing to qwen-edit and
  would arrive as a TypeError rather than a message.
- `fxlla do "<what you want>"`: state an outcome instead of a command and
  fxlla picks the model, renders, looks at the result and retries once. This
  is a deliberate change of position. Leaving orchestration to opencode and
  Claude Code was right while the alternative was a client loading an MCP
  server and driving it by hand; it stops being right once the decision needs
  facts only fxlla holds - which models exist, what each one actually accepts,
  how long each takes here, and what the render came out looking like. A
  caller cannot plan well against a menu it has to guess at.
  Each step is a different kind of thing so none is trusted with a job it
  cannot do. A local model turns the intent into one concrete call chosen from
  the real catalog, and anything it invents is refused before a render spends
  four minutes proving it. The generator stays the single authority on what
  each model accepts - there is no second capability check here, because a
  duplicate would drift from the catalog the moment one was corrected. The
  vision model then ENUMERATES what is in the result and is never asked
  whether the result is correct, since a model asked "is this a red car?"
  agrees and an agreement is worth nothing. Plain code compares that
  enumeration against words the planner committed to in advance: no model
  judges another model's output, one states expectations, another reports
  sightings, and arithmetic decides whether they line up.
  A miss is reported as what it is - something the description did not
  mention, which is a fact about the description and not a verdict on the
  image - so the loop asks for one more attempt rather than discarding work.
  Bounded by both a step count and a wall clock, because one slow render can
  outlast any reasonable number of attempts on its own. Only models whose
  weights are actually cached are offered. A refusal or a backend failure
  feeds the reason into the next plan instead of ending the run, which is what
  the first live run needed: it died on a backend error from a model whose
  weights were fully present.
- `FXLLA_LORA_DIRS`: LoRA discovery searches directories you name, plus any
  LoRA already in the Hugging Face cache (mflux takes a repo id directly, so a
  cached one needs no path), not just the civitai download folder. A LoRA is
  identified by reading its safetensors header rather than by its filename,
  because filenames lie in both directions - an adapter can be called
  Krea2-realism-V1 and a 26 GB base model can be called Krea-2-Turbo - and the
  header also declares which base model it was trained for, which is what makes
  a collection usable. When an adapter declares nothing - most do not - the
  base is inferred from its architecture: the hidden dimension is a property
  of the model it was fitted to, so it survives renaming (1536 krea2, 4608
  ideogram4, 3840 z-image, 4096 ltx2), and FLUX and Qwen share 3072 so they
  are split by Qwen's joint-attention projections. An inferred base is shown
  with a ~ prefix, never as fact. Validated by recovering all 28 declared
  labels in the collection the table was measured from. A repo whose weights
  are not all adapters is skipped:
  SDXL base ships an example LoRA beside the model, and offering that repo id
  would hand the base model to --lora. That single-directory assumption answered "no
  LoRAs found" to someone holding ten, because people train their own and keep
  them beside the project that produced them. When nothing is found the output
  now lists the directories it searched, so the answer is actionable instead
  of just discouraging.
- The media skill was rewritten as a decision guide rather than a tool list.
  It now tells a model to look before choosing (`list_media_models`,
  `list_loras`), which model suits which kind of request, when each option is
  worth setting, how Ideogram 4's JSON caption works and where to get its
  schema, that controls stack and how to chain a depth pass into one, and how
  a long render reports itself. The old text predated all of that and actively
  taught polling, which is what produced 47 status calls in one real session.
- ControlNet and depth, both stackable. Their weight rows list BOTH repos
  each needs - the adapter and the base model - because listing only the
  adapter let the consent gate pass on 4 GB while mflux then pulled 58 GB of
  FLUX base weights mid-render, which is precisely the transfer the gate
  exists to stop. `--control` is repeatable, and the
  two families are not interchangeable, so the model's capabilities decide the
  wire form: FLUX takes a control image plus an optional checkpoint
  (`IMAGE[,CHECKPOINT]`), Z-Image takes one combined spec
  (`type:path[:strength]`). Passing the wrong form says which one that model
  takes instead of only reporting a missing file. `--depth-image` supplies a
  depth map and `--save-depth-map` writes the derived one, so it can feed a
  later control step. All reachable from the MCP.
- `--guidance` for image models, which was missing entirely.
- Ideogram 4 JSON captions are validated before a render starts, against
  mflux's own schema: elements typed `obj` or `text`, palettes of at most five
  `#RRGGBB` colors, and bounding boxes as `[y_min, x_min, y_max, x_max]`
  integers in a 0..1000 space, and hex colors uppercase. That axis order is
  the trap - Y comes first -
  and getting it wrong used to surface as a mid-render schema warning. Prose
  prompts are untouched: only something that parses as a JSON object is held
  to the schema.
- `list_loras` on the MCP, so downloaded LoRAs are discoverable and get
  offered rather than ignored.
- mage-flow is back in the catalog. It was dropped for being gated on
  HuggingFace, which was the wrong call: the weights are perfectly usable once
  downloaded, and the catalog note now says what the gate requires.
- Image models are a declarative catalog (`config/media-models.conf`) instead
  of a hardcoded table, and the list grew from 8 to 12 with ideogram4, fibo,
  ernie and ernie-turbo. Each row records what its CLI actually accepts, read
  from the CLI's own `--help` rather than assumed, and that column is
  load-bearing: an option a model cannot take is refused by name, listing the
  models that can, instead of being forwarded to fail deep in the backend.
  Defaults adapt to the model, only explicit requests are validated. The new
  models carry weight-catalog rows too, so naming one cannot trigger an
  ungated multi-gigabyte download - the four are 26 to 32 GB each.
- LoRA support, which makes `fxlla pull civitai:<id>` useful for the first
  time: it could download LoRAs from the day it shipped and nothing could
  apply one. `--lora PATH[,SCALE]` (repeatable, local file or HuggingFace repo
  id), `--lora-style` for mflux's built-ins, and `fxlla media loras` listing
  what is downloaded plus the styles.
- `--negative`, `--prompt-file` and `--init-image` for image generation, each
  gated by the model's capabilities, and all of them on the MCP tool too.
- `media_job_status` takes `wait_s` and blocks instead of making a caller
  poll: a real session issued 47 status calls waiting on one video, which is
  both wasteful and indistinguishable from a runaway loop.
- `list_media_models` on the MCP, and `fxlla media models --json`. Without it
  a model opened generate.py and media_mcp.py ten times in one session to
  learn what it could do.
- Image generation reports the size of what it produced, like video reports
  its measured duration.
- Image-to-video: `fxlla media video --image PATH [FRAME STRENGTH]`, repeatable,
  and `images` on the `generate_video` MCP tool. One reference anchors the
  opening frame, two anchor both ends, which is how a transition between two
  stills is actually produced - describing the images in the prompt instead
  yields a different video that merely resembles them, which is exactly what a
  model did when asked for one. A reference path that does not exist is named
  up front rather than surfacing as a backend failure minutes later.

### Added
- `fxlla setup --media` installs the media backends the way everything else
  gets installed: as uv tools at pinned versions (mflux-cv, mlx-audio from
  PyPI; ltx-2-mlx has no PyPI release, so its pin is a commit on the public
  repository). Determinism comes from pins, not from bundling bytes -
  relocatable Python venvs cannot be signed into the app. With the video
  binary on PATH and the voice interpreter resolved automatically (env
  override, then the uv tool venv, then python3 - doctor and setup ask
  generate.py's own resolution instead of re-implementing it), a fresh
  machine goes from clone to `fxlla media image` with two commands. The one
  manual piece is voice's reference wav: cloning a voice needs a voice.

- `FXLLA_MFLUX_BIN_DIR`: point the whole mflux family (image models, edit,
  upscale) at a custom directory. Image was the one media backend without a
  location knob - video and voice had theirs - because it is eight per-model
  CLIs, which a single-binary variable cannot cover; a directory covers them
  all, the specific `FXLLA_EDIT_BIN`/`FXLLA_UPSCALE_BIN` still win, and a set
  directory missing a binary errors naming both instead of silently falling
  back to PATH and running a different install than the one pointed at.
  `fxlla doctor` checks the directory when the knob is set.

### Added
- An image sent to a model that cannot read one is no longer a crash. The
  gateway reads it with a model that can and hands the chosen one a
  description: one request in, one answer out, two models used, with the
  capability behind the endpoint rather than in whatever is calling it - a
  client with no MCP support gets it too. Measured end to end at 7 s through
  coder-3b and 12 s through qwen3-coder, and the description carried lettering
  a generator had invented, which the answering model then identified.
  A model that CAN see gets the image untouched, since a description is
  strictly lossier. The reader is never asked to confirm anything, only to
  enumerate: asked "is the lettering correct?" a vision model said yes and
  missed a whole invented block; asked to list every piece of text it reported
  it at once. Two models having answered is said out loud, in the log and in
  `X-Fxlla-Vision`, because a description that goes wrong otherwise reads as
  the answering model being wrong. Descriptions are cached by image content,
  since an OpenAI client resends the whole conversation each turn and without
  it a picture is re-read every turn and described differently each time.
  `FXLLA_VISION_ROUTING`, `FXLLA_VISION_MODEL` and `FXLLA_VISION_MAX_IMAGES`.
- `describe_image` on the media MCP, the same capability for a client that
  would rather orchestrate it itself.
- fxlla can see. A `vision` model (Qwen2.5-VL 7B, gguf) joins the catalog, and
  the gguf path learned the second file a vision model needs: a multimodal
  projector. `mflux`-style discovery rather than new config - a `mmproj*.gguf`
  beside the weights is passed to `llama-server --mmproj`, and the pull brings
  every projector a repo ships because they carry their OWN quant tag, so
  `--quant Q4_K_M` selected the weights and silently left the projector behind.
  The entry marker now excludes projectors too: with `--quant Q8_0` both files
  match and the server could have been launched on the projector. Verified
  end to end - the model described a generated workshop scene in 9 s and
  correctly reported the gibberish on the licence plates, which is the check a
  render actually needs. `mlx_lm.server` raises on any non-text content, so
  vision runs on the gguf engine; that is a property of the server, not of MLX.
  The f16 projector is preferred explicitly because glob expansion is collated:
  a bare `mmproj*.gguf` picks f16 under a UTF-8 locale and Q8_0 under `C`, and
  which projector a model runs with must not depend on the environment.
- The image catalog is now checked against the backend instead of trusted.
  mflux-cv 0.18.34 publishes `mflux-capabilities`, a machine-readable dump of
  which options each CLI honours - `honored`, `ignored` or `conditional`, read
  from the same constants its runtime warnings use. A test diffs the catalog
  against it on every run: nothing declared may be one the backend discards or
  one its CLI does not have, nothing honoured may be silently unreachable, and
  the init-image spelling must be the one that CLI actually has. Re-introducing
  each of this week's four capability bugs one at a time is caught by it. The
  test skips where mflux is absent or predates the dump, since a machine that
  cannot check must not report agreement.
- `--pid-decode` and `--pid-degrade-sigma`, from mflux-cv 0.18.33 (pin moved
  there): NVIDIA's pixel-diffusion decoder replaces the VAE decode and emits
  **4x the requested size**, so it is an upscale that happens inside the render
  rather than a pass after it. Probed per CLI rather than assumed - ten of the
  sixteen image models offer it, and the six that do not refuse it by name.
  `pid_degrade_sigma` is bounded to the 0-0.8 range PiD was distilled on, and
  the skill records the trap mflux documents: the default 0.0 is the input the
  decoder saw *least* in training, which shows up as invented texture on smooth
  areas, so 0.2 is the fix when a face comes back over-detailed.
- The consent gate can now be raised by an OPTION, not only by the model. PiD
  pulls its own weights - one 2.8 GB checkpoint chosen by the model's VAE latent
  space, plus a gated 5.3 GB caption encoder - so `weights.require` takes extra
  catalog aliases and the refusal names every pre-fetch needed. Without that, a
  render whose model weights were already cached would have passed the gate and
  then fetched 8 GB mid-flight, which is the same shape as the controlnet rows
  that listed only the adapter. The `pid` catalog row fetches only the three
  distilled 4-step checkpoints: the full `nvidia/PiD` repo is 54 GB of variants.
- A finished job now says, in the record itself, that `done` means written and
  not correct, and asks the caller to read the file. The media skill gained a
  section on it. This came out of running a real three-step chain - generate a
  desk scene, remove one object with qwen-edit, upscale - where the edit left a
  visible ghost of the object it was told to remove and the pipeline reported
  `done`, no warnings, `check_png` clean. Nothing below the caller can see: the
  checks confirm a PNG is a well-formed PNG by design. The caller often can -
  opencode's `read` tool hands PNGs to the model as a data URL - and was never
  told to. The capability was there and nothing pointed at it.
- `--strength` on `fxlla media image`, and `strength` on the MCP tool: how much
  of `--init-image` survives, 0 to 1. fxlla had been passing mflux's deprecated
  `--image-path`, which cannot carry one, so a second pass over a first render
  either changed nothing or replaced it - which made the most common workflow
  shape there is, draft small then refine large, unreachable. Measured on this
  machine: a 512x512 draft in 19 s, refined to 1024x1024 at strength 0.45 in
  32 s, keeping the composition and adding the detail. The value is not the
  total but the exit: a composition you do not want costs 19 seconds to reject
  instead of a full-size render. Out-of-range values are refused before the
  render with the range named, since reading "strength" as a percentage and
  sending 60 is the mistake worth catching; 0 and 1 are both legitimate.
- A finished or failed background render posts a desktop notification on macOS
  with what it produced and how long it took (`FXLLA_MEDIA_NOTIFY=0` disables
  it). An MCP tool call is request/response and an agent turn ends when the
  model stops calling tools, so nothing on the server side can reopen a turn to
  announce a render - an assistant can only report it when asked, and a render
  that takes eight minutes is one nobody is still watching. Prompt text is
  escaped into the AppleScript literal rather than interpolated, since a prompt
  carries quotes and backslashes.
- `fxlla media timings`: the measured record on its own - median seconds and
  seconds-per-megapixel per model and per video stage, from this machine's own
  finished jobs, with an explicit "nothing has been timed yet" when there is no
  history. Publishing the numbers in the catalog had not been enough: asked how
  long each model takes, a caller went to the shell and wrote forty lines of
  Python to parse the job files by hand, twice, and its version reported a
  video stage as an image model. Whoever reaches for a shell should find this
  instead of reimplementing it.

### Changed
- The expected duration now travels WITH the submission and the measured one
  with the result, instead of only living in the catalog. A catalog is read
  once and remembered: one session read it at 11:33, quoted those numbers for
  an hour, and never saw the measurements added at 12:05 - so the numbers being
  wrong outlived the fix. Submitting a render answers with the job id and
  "typically N here", scaled by canvas where a per-megapixel rate exists, and
  `media_job_status` adds `elapsed_s` to a finished job, which a caller
  previously shelled out to Python to compute by subtracting two timestamps.
- A media render never holds the call open now: `FXLLA_MCP_WAIT_S` defaults to
  0 and the tool returns a job id immediately. The 45 second window it replaced
  was dead weight - the quickest render measured on this hardware is 55 s, so
  nothing ever landed inside it, and every submission spent 45 s of the
  caller's turn to arrive at the same job id. `media_job_status` answers
  instantly for the same reason, and a still-running job is told to report the
  id and finish rather than poll: while an agent polls, whoever it is working
  for is queued behind a render that is not going any faster for being watched.

### Fixed
- `schnell` and `dev` could not render at all: every attempt died with "No
  root_path and no download_url for component: vae" against a Hugging Face
  cache that was complete, vae included. fxlla was selecting them with
  `--base-model` alone, and mflux documents that flag as qualifying a
  third-party `--model` ("explicitly specify whether the base model is dev or
  schnell"), not as a way to name one. Passing it alone used to be tolerated;
  it now yields a config whose `model_name` is None, and the None surfaces
  thousands of lines later as the first component that has no download URL,
  which is the vae and has nothing to do with it. fxlla now names the model
  the documented way, `--model schnell`, which is both correct and immune to
  that. Reported upstream separately, because producing a silently malformed
  config where an argparse error would do costs whoever hits it an hour.
  its weights, which answered the wrong question. Whether an image CAN reach a
  model and whether its eyes are worth using are separate, and they came apart
  the first time a model shipped a vision tower inherited from its base and
  never tuned - a Qwen3.5 derivative whose own author writes that text-only
  training did not evaluate image understanding. Reading a file's existence as
  a statement about quality was inferring a claim nobody had made, so an image
  is now forwarded untouched only when the catalog gives the model role
  `vision` and the projector is present. Both, or it gets a description from a
  model chosen for the job - the safe answer by default, since granting the
  role is a deliberate act made after measuring. `FXLLA_VISION_MODEL` needs
  only the projector, because naming a model there is itself the declaration.
  A checkout whose catalog moved falls back to the projector alone: with no
  declaration to read and no reader to find, requiring one would have turned a
  working vision model into a 502.
- Clicking a render's desktop notification opened Script Editor. A plain
  `osascript` notification belongs to `com.apple.ScriptEditor2`, so the click
  landed in an app the user never invoked and which has nothing to do with the
  render. With `terminal-notifier` on PATH the notification now reveals the
  finished file in Finder; without it the notification is attributed to Finder,
  so the click at least opens a file browser. A failed render offers nothing to
  open, since there is no file.
- Three notification tests asserted the `osascript` form without pinning it, so
  they passed or failed depending on whether `terminal-notifier` happened to be
  installed - green here and red in CI, or the reverse. Each now pins the branch
  it means to exercise.
- `observed` gave no seconds-per-megapixel for `edit` or `upscale`. Both
  produce a single still, so their cost scales with its area exactly as a
  render's does; the normalisation was keyed on `kind == "image"` alone, so
  both reported a bare median that does not transfer to another size. Video
  still has none, deliberately - its cost also scales with the frame count.
- Publishing step counts as the cost signal invited the wrong inference and got
  one: told cost was "roughly linear in steps", a model concluded that 8-step
  krea2 was as fast as 8-step z-image-turbo and published a timing table
  claiming 15-25 s for a render measured here at 518 s. Steps compare a model
  only to itself - a step costs what the model costs, and those two are 9x
  apart at the same count. `list_media_models` now returns `observed`: real
  seconds from this machine's own finished jobs, per model and per
  `video:<stage>`, with sample counts and seconds-per-megapixel for images
  (not for video, whose cost also scales with frames, where an area-normalised
  figure would be a fresh wrong signal). The data was already in the job
  records and unused. A model with no history says so instead of estimating.
- Nothing told a caller what a render would cost, for image or for video, so
  the choice that decides between one minute and ten was made blind - an eight
  minute poster looked identical to a fast one until it had been waited for.
  `list_media_models` now reports `default_steps` per image model, the count it
  will really run (they span 4 to 50), and a `video` section with each stage,
  its step counts, and how duration multiplies the work. Video also gained the
  step knobs it never had: `--steps` (one-stage), `--stage1-steps` and
  `--stage2-steps` (the two-stage pipelines), each refused with the right one
  named when passed to a stage that ignores it, per ltx's own `--help`.
- The image backend pin moves to `mflux-cv==0.18.32`, which is where the other
  half of the ignored-option story landed: mflux now warns whenever an option
  the model cannot honour is dropped, keyed on the model's effective behaviour
  rather than on the flag, so omitting `--guidance` (which also disables CFG)
  warns too. Its format is what fxlla's warning filter already reads, so the
  two compose: the backend says it, and the caller hears it. 0.18.30-31 also
  bring Qwen and Ideogram saved-model fixes, and EXIF orientation on load.
- fxlla read the backend's stderr only when a render FAILED, so anything it
  said on a run that worked was captured and thrown away - including mflux's
  own `--steps is ignored; Ideogram 4 presets define the step count`. The
  backend was naming the exact class of bug that has now been found three times
  by reading its source instead, and the message was being deleted one layer
  up. Warnings from a successful render are now surfaced (filtered, so progress
  bars do not drown them), carried on the job record as `warnings`, and
  relayed by `media_job_status`. A structural test keeps every backend call
  that handles failure also reporting what it said on success.
- The same trap on the distilled models, and on the DEFAULT one: mflux declares
  `supports_guidance=False` for z-image-turbo, boogu, schnell and z-controlnet,
  which pins guidance to 0 - and guidance <= 1 makes the sampler drop the
  negative encoding entirely and skip CFG. So `--negative-prompt` was read,
  encoded and thrown away without a word on four models, while this catalog
  listed both caps and the media skill recommended negative prompts against
  watermarks and stray text. Both caps are gone from those rows, the skill now
  says the fix on a CFG-off model is a different seed, and a test enforces the
  invariant that made it findable: `negative` requires `guidance` or `preset`.
- Ideogram 4 accepted `--steps` and `--guidance` and discarded both - its
  presets fix them, which its CLI says in a warning nothing was reading. So a
  request for 12 steps waited for the preset's 20 and was told nothing. Both
  are now refused by name, and `--preset` is exposed instead: `V4_TURBO_12`,
  `V4_DEFAULT_20`, `V4_QUALITY_48`, reported with their step counts. A flag
  that is accepted and ignored is worse than one that is refused.
- No MCP generator exposed `output`, so over MCP there was no way to say where
  a file should go: "save it in ~/Downloads" was accepted and silently dropped,
  every render landed in the media directory, and the reported path looked like
  an answer. The CLI has taken `--output` on all five generators from the
  start - only the tool schemas were missing it. A `~` in the path now expands
  too, since nothing here runs through a shell and an unexpanded one would
  create a directory named `~`.
- One media render made the MCP server deaf to everything else. Tool calls were
  handled inline in the stdin read loop, so while mflux ran the server read
  nothing: later calls came back as `MCP error -32001: Request timed out`,
  including the `media_job_status` and `list_media_jobs` calls asking about
  that render. A model that cannot tell a timeout from a failure retries, and
  one submitted the same render three more times. Calls now run on their own
  thread, and a render is submitted as a background job awaited for a bounded
  window (`FXLLA_MCP_WAIT_S`, default 45 s, measured against a client that
  returned a 45 s call and timed out past ~69 s): it returns the path when it
  finishes in time and the job id with a still-running line when it does not.
  `media_job_status` caps its wait at the same window - a request to block 120
  seconds came back as a timeout, which said nothing - and a job that is still
  going says so, along with not to submit it again. Pass `async: false` for the
  old hold-the-call-open behaviour.
- A LoRA named the way `list_loras` reports it - the bare filename - was
  refused with "LoRA not found" by the same tool that had just printed it. Bare
  names now resolve against the search directories, and an ambiguous one names
  the candidates. A name containing a directory part is still taken literally,
  so a wrong path is reported rather than guessed at.
- Ideogram 4 rejects any dimension off a 16-pixel grid, and did so only after
  its weights were resident: a 1080-wide poster failed minutes in. Models that
  enforce a grid now declare it, it is checked before anything spawns, and
  `list_media_models` reports it as `dim_step` so the size can be right the
  first time. Models whose backends accept any size are unaffected.
- The app bundled the CLI without `config/media.conf`, for as long as that
  catalog has existed: an installed app had no weight catalog, so `fxlla media
  weights` was empty and the download consent gate had nothing to read. The
  build copied config files by name and drifted; it now globs `*.conf`, which
  keeps new catalogs bundled by construction (`config.env`, the one file that
  must never ship, matches neither pattern). `evals/` was missing from the
  bundle too. CI caught both.
- Documentation swept against the code: 48 confirmed staleness findings fixed
  across README.md, ROADMAP.md, AGENTS.md, SECURITY.md, app/README.md,
  docs/roadmap-remaining.md, evals/README.md and the module docstrings in
  rag/core.py, graph/codegraph.py, gateway/fxlla_gateway.py and
  media/generate.py. The recurring themes: the code graph was still described
  as Python-only in three places (it parses eight more languages via
  tree-sitter), ROADMAP carried thirteen unchecked boxes for work that shipped,
  module docstrings listed a fraction of the environment variables they read
  and omitted half the subcommands they register, the graph MCP was documented
  with three tools instead of five, and SECURITY.md named HF_TOKEN as the only
  credential sent over the network (FXLLA_CIVITAI_TOKEN is another).
- `FXLLA_STORE` no longer defaults to one author's external volume. A fresh
  clone followed the README verbatim and died at `fxlla setup`, because the
  copied config.env hard-coded a disk that exists on exactly one machine. The
  fallback is now `~/.local/share/fxlla/store` and the example ships the
  external-disk form commented out.
- The signed `.dmg` is no longer tracked in the repository. It is a release
  asset, rebuilt by `app/build.sh` plus `app/package-dmg.sh`; one slipped in
  through a `git add -A` and is now git-ignored along with the other app build
  artifacts.
- The media MCP registration was written with an EMPTY environment through
  `fxlla wire-opencode --all`, so every media tool call from an editor died
  with "FXLLA_STORE is not set". The wiring copied only exported variables,
  and config.env is read with `: "${VAR:=default}"`, which assigns without
  exporting - only `fxlla media wire-opencode` (which exports on its way
  through cmd_media) ever worked. Found by reading a real session where a
  model hit this four times and abandoned the MCP for about thirty shell
  commands.
- `--output` accepts a directory. Passing one ("put it in ~/Downloads") is
  the obvious thing to try and used to reach the toolchain as a filename,
  failing deep inside a backend; all five generators now name the file
  inside it. A trailing separator creates the directory.
- Image dimensions are no longer dropped in silence. `--aspect` together with
  `--width/--height` made aspect win without a word, so a request for 512x512
  produced a 1024x1024 file; the combination is now refused, naming both. The
  test suite had encoded the old behavior as intent and defended the bug.
- `fxlla media video` reports the MEASURED duration, frames and fps of what it
  produced, and the MCP returns them too. A model computed 49 frames at 24 fps
  as "about 10 seconds" (it is 2.04) and declared an unmet 4-8 second
  requirement satisfied. `--seconds` now asks for duration directly instead of
  making the caller do the multiplication.
- Voice through the MCP used a python without mlx-audio: `lib/core.sh`
  defaulted FXLLA_VOICE_PYTHON to `python3`, and the registration forwarded
  that default as if it were an explicit choice, beating the uv tool venv that
  actually has mlx-audio. The default now lives only in generate.py's
  resolution, where unset means unset.
- opencode's context meter now runs on each model's real window. The
  registration declared no per-model limit, so the meter and the
  auto-compaction trigger used a made-up number. The gateway now reports each
  model's context on /v1/models - the model's own config.json window for mlx,
  the served -c for gguf - and the registration writes it as the opencode
  limit, with output capped between 4k and 16k at a quarter of the window.
- Selecting the local provider in opencode no longer stops at an API key
  prompt. The local servers validate no key, but opencode asks for one when
  the provider omits it; the registration now carries a dummy `apiKey`
  (kept only if the user has not set their own).
- Embedding models no longer appear as chat models. The gateway's model list
  - which also feeds the opencode registration on `fxlla serve` - excludes
  catalog entries with role `embed`, matched by alias and by source repo, so
  `embed` cannot end up selectable in an editor again (it shipped that way
  once) and a raw API caller cannot make the gateway spawn a chat server on
  a BERT. The opencode registration now consumes the gateway's own filtered
  `/v1/models` instead of re-enumerating the store, so there is exactly one
  place that decides what is chat-servable. With a moved or missing catalog
  the filter excludes nothing: hiding a stranger's models would be worse
  than listing an extra one.

## [0.2.0] - 2026-07-31

Everything on the original roadmap is now delivered or explicitly declined
with a measurement. Highlights: the multi-model gateway with passive metrics,
RAG with a selectable embedding model and its own retrieval eval, the KuzuDB
code graph across nine languages, local media generation with async jobs and
content validation, consent gates on every download path, a signed and
notarized menu bar app that bundles the CLI, and `fxlla eval`, which scores
chat models on mechanically checked quality and measured speed.

### Added
- `fxlla eval`: score chat models on quality and speed locally, so choosing a
  model rests on data instead of the catalog's prose. Quality is 30 authored
  tasks across four dimensions - code executed against asserts in a sandbox,
  structured tool calls (with a separate column for calls that only appear in
  the text channel, which an agent stack never sees), instruction following,
  and long-context recall at 8k/16k - every check mechanical, no model judging
  another model. Speed is cold start, first request (the engines split the
  cold cost differently: llama-server loads before /health flips, mlx_lm loads
  on the first request, and the table shows both so neither lies), TTFT,
  decode and prefill tok/s, peak RSS, and tokens spent. Each model runs alone
  on a cold dedicated server; a busy eval port is refused rather than adopted.
  Every run prints a fingerprint of the rendered task set and a harness
  version: two runs are comparable exactly when both match. Results carry
  server build, weights identity, and machine identity, and go only to the
  state dir, which the harness never reads back. Measured on the first sweep
  (M5 Max, fingerprint f6ae21eaee7d, harness v3): redteam-4bit 10/10 code and
  8/8 tools at 93.5 tok/s and 42 GB resident; qwen3-coder 8/10 and 8/8 at
  131.2 tok/s and 16 GB; coder-1.5b 4/10 at 399.9 tok/s with its tool calls
  stranded in the text channel; tiny 1/10 code, which is the discrimination
  the task set was gated on. Dogfooding on those models fixed code extraction
  twice before shipping - real replies end in fenced example output (not
  Python), or in fenced usage examples (which compile), or quote compiling
  fragments mid-explanation, and each style silently lowered a correct score -
  so extraction concatenates the compiling fences that bind names, and those
  semantic changes are why the harness says v3.
- The embedding model behind `fxlla kb` is selectable. The catalog gained four
  more `embed` entries next to nomic-embed-text v1.5: bge-small (384-dim),
  bge-large and qwen3-embedding (1024-dim), and embeddinggemma (768-dim).
  `FXLLA_EMBED_MODEL` picks one by alias; unset keeps the previous model, so
  existing knowledge bases are untouched. Dimensions in the catalog notes are
  read from the GGUF headers, not from documentation.
- `fxlla kb reindex <name>` re-embeds a knowledge base with the currently
  selected model, which is what makes switching models possible: vectors from
  two models are not comparable, and the width guard previously refused the
  search without offering a way forward. It re-embeds the chunk text already in
  the store rather than re-reading sources that may have changed, in a single
  transaction, so an interrupted run leaves the base exactly as it was.
- `fxlla kb eval` scores retrieval instead of guessing at it: a golden set of 19
  questions over this repository's own documentation, reporting recall@1,
  recall@k, MRR, median query latency and a fingerprint of the corpus. On
  fingerprint 0c32ee03a384, nomic-embed-text scores recall@1 68%, recall@5 100%,
  MRR 0.809 at 8 ms per query, against bge-small at 74%, 84%, MRR 0.789 and
  5 ms - a 37 MB model ahead on the first hit and well behind by the fifth. Read
  those gaps with care: 19 queries means one query is worth 5.3 points, so the
  harness separates models that differ clearly and cannot rank two close ones.
  Since the corpus is live documentation, every edit moves every score;
  two runs are comparable only when their fingerprints match, which is why one
  is printed. `CHANGELOG.md` and `docs/JOURNAL.md` are excluded from the corpus
  for the same reason: they are where results get written down, and including
  them made each recorded number stale the moment it was recorded.

### Fixed
- `fxlla kb` no longer embeds against whatever server happens to hold the port.
  Reuse adopted any process answering /health without asking what it had loaded:
  demonstrated with zero embedding models installed, where indexing and search
  both still succeeded against a borrowed server. That was harmless while one
  model was possible and is not now, since nomic and embeddinggemma are both
  768-dim and qwen3-embedding and bge-large both 1024, so the width guard cannot
  see the difference and the store is silently poisoned. It now asks the server
  which model it loaded and refuses on a mismatch, naming both and pointing at
  `fxlla kb stop`. A server that will not identify itself is still allowed,
  since pointing `FXLLA_EMBED_PORT` at your own is legitimate.
- The embedding server is no longer started with `--pooling mean`. Every
  embedding GGUF declares its own pooling type and llama.cpp honours it: nomic
  wants mean, bge wants CLS, qwen3-embedding wants last-token. Forcing mean was
  correct only for the model that happened to be the default and silently
  degraded the rest, since a badly pooled vector is still a vector. Verified
  before removing the flag: omitting it reproduces `--pooling mean` bit for bit
  on nomic, while `--pooling cls` visibly differs. The README instructions for
  running a server by hand carried the same flag and were corrected.
- A gateway model switch no longer rounds up to the next whole second, and a
  backend that dies on start fails immediately instead of holding the 180s
  timeout. The wait loop polled once a second and never looked at the process it
  was waiting for. Measured: a switch to a small model went from 1.28s to about
  1.01s (four runs each), and a backend that had already exited went from burning
  the entire budget to 0.01s.
- `fxlla kb search` was five times slower than it needed to be. The embedding
  server is ready in about 0.17s, but the health check polled once a second, so
  the first probe missed it and every query slept a full second waiting for a
  server that was already up. Polling is now 20 ms and a cold query costs about
  0.2s instead of 1.1s. Reusing an already-running server, and leaving it alone
  afterwards, already worked before this change - it just also spawned a doomed
  process that could not bind the port; that is now skipped, and a warm query
  still costs about 0.09s.
- Concurrent `fxlla kb` commands no longer fail. Starting the server is now
  serialized with a lock file, so ten simultaneous searches start one server
  instead of ten and none of them fail: previously up to four in ten died with a
  raw traceback after losing the race to bind the port. A search whose borrowed
  server is stopped mid-request retries once instead of aborting, which matters
  most to `kb add`, where it used to leave a partial index behind.
- `fxlla kb` refuses to mix embedding widths. A server for a different model
  returns a different number of dimensions, and the cosine comparison silently
  truncated to the shorter vector: searches returned meaningless scores and an add
  wrote unrankable rows into the base for good. Both now stop with a message
  naming the two widths, before anything is written.
- Errors from the embedding port are messages rather than tracebacks. A chat
  server on `FXLLA_EMBED_PORT` produced a two-kilobyte Python traceback, which the
  MCP layer forwarded verbatim as a tool result.
- New `fxlla kb stop` stops the embedding server, since a reused one outlives the
  command that started it and previously had to be found by pid.
- `fxlla on <alias>` registered the wrong model in opencode. It read the id from
  the first entry of `/v1/models`, and `mlx_lm` now enumerates the whole Hugging
  Face cache there, so whichever unrelated model sorted first was registered as
  the local chat model. On a machine with diffusion weights cached that meant
  opencode's `local` provider pointed at an image model. The id is now derived
  from what was actually launched, per engine.
- `fxlla skills install` no longer leaves stale skills behind. It never removed
  anything, so a renamed or deleted skill kept living in `~/.claude/skills` (and
  its path stayed in opencode's `instructions`) telling the model to reach for a
  tool that had changed. Install now prunes skills it previously installed that
  the repo no longer ships, identified by a marker file so directories it did not
  create - other tools' skills, and your own `instructions` entries - are left
  untouched. It also validates each skill's frontmatter first and aborts the whole
  run if one is malformed, rather than half-installing a pack a client will
  silently ignore. New `fxlla skills status` reports each installed copy as
  current, drifted, or missing, and `fxlla doctor` summarises it.
- CodeQL analyzes the Python sources again. It was narrowed to workflow files
  early on, when the repo had no Python and CodeQL failed the run for a language
  with no source; there are now ~6,700 lines across `gateway`, `rag`, `graph`,
  and `media` that were never scanned, and the green check only covered the
  workflows. It now runs a matrix of `actions` plus `python`, on pull requests
  and on pushes to `main` (so the default branch has a baseline).
- CI stopped warning on every run: `actions/checkout` and `astral-sh/setup-uv`
  were on majors that target the deprecated Node 20, and `setup-uv` cached
  against `uv.lock`/`requirements*.txt`, neither of which exists here (this repo
  gets dependencies from `uv run --with`), so the cache never invalidated.
- `ludeeus/action-shellcheck` is pinned to a release SHA instead of tracking
  that repository's `master`, and `github/codeql-action` moved to v4 (v3 warned
  on every run that it is deprecated in December 2026). Every workflow now runs
  warning-free.

### Changed
- Code graph now uses an embedded KuzuDB graph (Cypher) instead of a flat SQLite
  store. `fxlla graph impact` is a Cypher variable-length path over a derived
  CALLS relationship, replacing the Python breadth-first walk. `fxlla graph`
  runs the backend under `uv run --with kuzu` (uv is already required);
  `FXLLA_GRAPH_PYTHON` overrides the interpreter. The `ast` extraction, the CLI,
  and the MCP tools are unchanged. Existing graphs are re-created at the new
  `<store>/graph/graph.kuzu` location on the next `fxlla graph index`.

### Added
- Large downloads need consent. Any transfer over `FXLLA_CONFIRM_ABOVE_GB` (5 GB
  by default) prompts at a terminal and refuses everywhere else, naming the size
  and how to proceed. This covers all four routes that move bytes: catalog models
  (both downloaders), `media:<alias>` weights, Civitai files, and the weights a
  render fetches on first use. That last one was the widest hole: a first
  `fxlla media image` pulled tens of gigabytes mid-render with nothing asked, and
  because a background job and an MCP tool call both re-invoke `media/generate.py`
  the check lives there rather than in the shell wrapper. `--yes` and
  `FXLLA_ASSUME_YES=1` authorize; the menu bar app passes `--yes` because its own
  dialog already showed the size. A size that cannot be read is treated as large,
  a resumed pull asks about the remainder, and a refusal leaves no directories or
  marker files behind.
- Media output is checked for content, not only for a valid container: a header
  check passes for eight seconds of silence, which is a real failure mode of these
  toolchains. `media/quality.py` adds stdlib-only checks (silence, a constant
  waveform, near-zero signal, a large DC offset, a mostly-silent clip; PNG
  dimensions; a usable video stream via `ffprobe` when installed) wired into the
  existing validators. Only unambiguous garbage is flagged - frame count and clip
  length are caller-controlled, so short clips are accepted, and a file the checks
  cannot parse gets no verdict rather than a rejection. `--skip-quality` or
  `FXLLA_MEDIA_SKIP_QUALITY=1` accepts a flagged file, and `fxlla doctor` reports
  whether `ffprobe` is available.
- Media weight catalog and pre-fetch: `config/media.conf` maps each media alias to
  its Hugging Face repositories with sizes, `fxlla media weights` shows what is
  cached, and `fxlla pull media:<alias>` fetches ahead of a render into the media
  HF cache (where the toolchains look them up by repo id, so the bandwidth cap
  does not apply). `fxlla doctor` reports how many entries are ready. Video
  correctly pulls both the LTX model and the Gemma text encoder its pipelines load.
- The app bundles the CLI and can install it on PATH: `app/build.sh` copies the
  CLI tree into `fxlla.app/Contents/Resources/cli` (never `config/config.env`,
  which holds tokens), so a `.dmg` install is self-contained. A new **Install the
  fxlla command** action in the panel symlinks it into `~/.local/bin` - an
  explicit user action, idempotent, and it refuses to replace a file or a link it
  did not create (for example a git checkout you develop against).
- Background media jobs: `--async` on any `fxlla media` generator returns a job
  id immediately instead of blocking, with `fxlla media jobs`, `job <id>`,
  `cancel <id>`, and `jobs --prune` to follow and clean up. Jobs are serialized
  (one render at a time) because they share unified memory with the gateway's
  models. Over MCP, generators take `async: true` and the new `media_job_status`,
  `list_media_jobs`, and `cancel_media_job` tools poll and manage them.
- Multi-language code graph: `fxlla graph` now indexes JavaScript, TypeScript,
  Go, Rust, Java, C/C++, and Ruby via tree-sitter in addition to Python (still
  `ast`). All languages share the KuzuDB graph, so `def`/`callers`/`impact`
  resolve by name across them. Runs under `uv run --with kuzu==0.11.3 --with
  tree-sitter --with tree-sitter-language-pack`.
- Optional sqlite-vec KNN index for `fxlla kb search`: set `FXLLA_KB_INDEX=1` to
  replace the brute-force cosine scan with a vector index rebuilt on demand from
  the chunks table. It runs `rag/core.py` under `uv run --with sqlite-vec`
  (no persistent install) and falls back to the scan when the extension is
  unavailable, so a clean clone still works. The `fxlla kb` CLI is unchanged. The
  RAG MCP server and its `wire-opencode` registration run under the same
  interpreter, so `rag_search` uses the index too.
- Image editing and upscaling: `fxlla media edit "<prompt>" --image <path>`
  (mflux-cv qwen-edit) and `fxlla media upscale --image <path>` (mflux-cv
  seedvr2), exposed over MCP as `edit_image` and `upscale_image`.
- `fxlla ram persist` / `unpersist`: install or remove a LaunchDaemon that
  re-applies the GPU wired limit at every boot, so `fxlla ram auto` survives a
  reboot.
- `fxlla pull --downloader hf`: fetch a Hugging Face repo with the official CLI
  (run via `uvx`, no persistent install) instead of the aria2c tree walk. More
  robust for xet-backed and LFS repos, but it ignores the bandwidth cap. Default
  stays aria2 (`FXLLA_DOWNLOADER`).
- Civitai as a weight source: `fxlla pull civitai:<id>` (or a civitai.com URL)
  downloads a LoRA or checkpoint from civitai.com into `<store>/civitai`,
  bandwidth-capped like every pull, authenticated with `FXLLA_CIVITAI_TOKEN`.
- Tool enablement: `fxlla wire-opencode --all` registers the provider and every
  MCP server (rag, graph, media) at once, and `fxlla skills install` installs a
  portable skill pack (`skills/`) that tells a model when to use the tools -
  retrieve from a knowledge base, walk the code graph before editing, offer
  media generation, and check availability and consent before a download.
  Installs for Claude Code (`~/.claude/skills`) and opencode (its `instructions`
  list), idempotently.
- Menu bar app media controls: a prompt, an image/video/voice picker, and a
  Generate button that runs `fxlla media` and reveals the output in Finder. A
  path-filtered macOS CI job builds the app on `app/**` changes.
- Media generation (`fxlla media image|video|voice`): local image generation
  through the mflux-cv toolchain (z-image-turbo and friends), short video through
  ltx-2-mlx (LTX-2.3), and text-to-speech through mlx-audio (Chatterbox), with
  output validation. Exposed to opencode and Claude Code as an MCP server
  (`generate_image`, `generate_video`, `generate_speech`) via
  `fxlla media wire-opencode`. Speech runs under a separate interpreter
  (`FXLLA_VOICE_PYTHON`) so fxlla itself never depends on mlx-audio.
- Machine-readable model availability: `fxlla ls --json` lists cached models
  (alias, size, engine, repo) and `fxlla avail <alias>` reports
  `{cached, known, engine, repo, size}` for any catalog or downloaded model, so
  an agent can check availability before offering a download. `fxlla on` gains
  opt-in `--pull` while staying fail-fast by default when a model is not cached.
- Shell completions: `fxlla completions <bash|zsh>` prints a completion script
  (load with `source <(fxlla completions bash)`). Completes commands, catalog
  aliases for `pull`, downloaded models for `on`/`off`/`rm`, and `kb`/`graph`
  subcommands, driven by a hidden `fxlla __complete` helper.
- Passive gateway metrics: the multi-model gateway now derives time to first
  token and tokens per second from real proxied traffic (streamed SSE or a
  response's `usage`) and appends samples to the same `stats.jsonl` time-series
  the menu bar app renders, so charts reflect actual usage rather than only a
  synthetic probe. `fxlla stats [--watch]` reads these samples when the gateway
  is serving.
- Code graph (`fxlla graph index|def|refs|callers`): index a Python codebase
  with the `ast` module into a SQLite store and navigate it structurally (where
  a symbol is defined, referenced, and which functions call it). Exposed to
  opencode and Claude Code as an MCP server (`find_definition`,
  `find_references`, `find_callers`).
- RAG knowledge bases (`fxlla kb add|search|ls|rm`): index files locally with a
  small embedding model (llama.cpp) into a SQLite store and search by cosine
  similarity. Exposed to opencode and Claude Code as an MCP server (`rag_search`)
  via `fxlla kb wire-opencode`.
- Menu bar app (`app/`, SwiftUI `MenuBarExtra`): live `fxlla status`,
  time-series charts for tokens/s, TTFT, and RAM (GB), gateway start/stop,
  model list with on-demand load, and a GPU RAM limit toggle. Builds to a
  `.app` bundle with an optional signed `.dmg`.
- Multi-model gateway (`fxlla serve` / `fxlla unserve`): one OpenAI-compatible
  endpoint fronting every downloaded model, with on-demand loading and
  least-recently-used eviction under a RAM budget. Aggregated `/v1/models`,
  streaming passthrough, and opencode registration of all local models.
- `fxlla stats [--watch] [--json]`: live RAM (server RSS), time to first token,
  and tokens per second via a small probe, appended to a rolling time-series
  (`stats.jsonl`) for the menu bar app.
- `fxlla doctor`: environment diagnostics (dependencies, PATH, store, GPU
  memory, media prerequisites, server health). The media section checks the
  image/video/voice backends and weight cache so a fresh machine sees the gap
  instead of a cryptic runtime failure.
- `fxlla pull` now fails loudly if a download leaves pending `.aria2` control
  files, instead of marking an incomplete model as complete.

### Changed
- Media generation now frees the gateway's resident models before a job so a
  heavy render does not compete with a large LLM for unified memory (they share
  it, and the overlap could OOM). The gateway exposes `POST /admin/unload` and
  reloads on demand afterward; opt out with `--keep-models` or
  `FXLLA_MEDIA_KEEP_MODELS`. MCP tool calls get this for free.
- Hardened app signing and packaging: shared `app/sign-lib.sh` checks the
  Developer ID identity before building and verifies the app is signed with the
  hardened runtime; `app/package-dmg.sh --check` validates the signing
  prerequisites, and the notarize path validates the staple and runs a
  Gatekeeper assessment. Both notarytool credential routes are documented.

### Fixed
- Configuration precedence: an exported environment variable now wins over
  `~/.config/fxlla/config.env` as documented. Sourcing `config.env` (plain
  assignments) used to clobber values exported in the shell.
- Signing checks captured command output before matching: piping into
  `grep -q` under `set -o pipefail` could SIGPIPE the producer and report a
  false negative (e.g. "not signed with the hardened runtime" on a signed app).

### Planned
- Menu bar app (SwiftUI `MenuBarExtra`), signed and notarized.
- Live metrics: tokens per second, time to first token, RAM per model.
- Per-project RAG and knowledge bases exposed as an MCP server.
- Code graph (KuzuDB) exposed as an MCP server.
- Image and video generation skills (mflux) exposed as MCP tools.
- Signed `.dmg` installer.

## [0.1.0] - 2026-07-28

First working release of the `fxlla` CLI. Validated end to end on M5 Max
(128 GB).

### Added
- `fxlla` CLI: `setup`, `models`, `pull`, `ls`, `on`, `off`, `status`, `logs`,
  `ram`, `wire-opencode`, `rm`, `config`.
- MLX engine (`mlx_lm.server`) as the default backend.
- GGUF backend via llama.cpp (`llama-server`); the engine is selected per
  model through the `.engine` marker. Quant selection with `--quant`.
- Bandwidth-capped downloads (`aria2c`, `FXLLA_RATE_MBIT`, default 25),
  resumable, cached on an external disk (`FXLLA_STORE`).
- Keep-warm watchdog that stops the server after `FXLLA_KEEP_WARM` idle minutes
  (default 10, `0` to disable).
- `fxlla ram` to raise or reset `iogpu.wired_limit_mb` for full 128 GB use.
- opencode integration through a `local` OpenAI-compatible provider. Claude
  Code is left untouched.
- Verified catalog of models across development, agentic, red team, and GGUF
  roles.

### Notes
- Compatible with the macOS system bash (3.2).
