import argparse
import contextlib
import importlib
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
media = importlib.import_module("generate")
media_mcp = importlib.import_module("media_mcp")
jobs = importlib.import_module("jobs")
quality = importlib.import_module("quality")
weights = importlib.import_module("weights")


def _tiny_png(path):
    """Write a minimal valid 1x1 PNG (> 1 KB via a padding chunk)."""
    def chunk(typ, data):
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff))
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    pad = chunk(b"tEXt", b"x" * 2000)  # push size over the 1 KB floor
    with open(path, "wb") as f:
        f.write(media.PNG_MAGIC + chunk(b"IHDR", ihdr) + pad
                + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def _write_lora(path, keys=("blocks.0.attn.lora_A.weight",), meta=None):
    """A minimal valid safetensors file whose header marks it as an adapter."""
    hdr = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}
    if meta:
        hdr["__metadata__"] = meta
    blob = json.dumps(hdr).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)) + blob + b"\x00\x00")
    return path


class TestBuildCommand(unittest.TestCase):
    def test_turbo_defaults_steps(self):
        cmd = media.build_command(media.MODELS["z-image-turbo"], "cat", "/o.png")
        self.assertEqual(cmd[0], "mflux-generate-z-image-turbo")
        self.assertIn("--steps", cmd)
        self.assertEqual(cmd[cmd.index("--steps") + 1], "8")
        self.assertIn("--prompt", cmd)
        self.assertNotIn("--base-model", cmd)

    def test_flux1_names_the_model_rather_than_qualifying_one(self):
        # mflux documents --base-model as qualifying a third-party --model
        # ("explicitly specify whether the base model is dev or schnell"), not
        # as a way to select one. Alone it produced a config with model_name
        # None, and the render died as "no download_url for component: vae"
        # against a cache that was complete.
        cmd = media.build_command(media.MODELS["schnell"], "cat", "/o.png")
        self.assertEqual(cmd[0], "mflux-generate")
        self.assertNotIn("--base-model", cmd)
        self.assertEqual(cmd[cmd.index("--model") + 1], "schnell")

    def test_explicit_steps_override(self):
        cmd = media.build_command(media.MODELS["z-image-turbo"], "cat", "/o.png", steps=3)
        self.assertEqual(cmd[cmd.index("--steps") + 1], "3")

    def test_aspect_with_width_height_is_refused(self):
        # This test used to assert the opposite - that aspect silently won and
        # the dimensions were dropped - which is how a request for 512x512
        # produced a 1024x1024 file with nothing said. The old behavior was
        # written down as intent, so the suite defended the bug.
        with self.assertRaises(ValueError):
            media.build_command(media.MODELS["z-image-turbo"], "cat", "/o.png",
                                aspect="1:1", width=512, height=512)

    def test_width_height_without_aspect(self):
        cmd = media.build_command(media.MODELS["z-image-turbo"], "cat", "/o.png",
                                  width=768, height=512)
        self.assertEqual(cmd[cmd.index("--width") + 1], "768")
        self.assertEqual(cmd[cmd.index("--height") + 1], "512")

    def test_low_ram_and_metadata_flags(self):
        cmd = media.build_command(media.MODELS["z-image-turbo"], "cat", "/o.png",
                                  low_ram=True, metadata=True)
        self.assertIn("--low-ram", cmd)
        self.assertIn("--metadata", cmd)

    def test_no_steps_default_omits_flag(self):
        cmd = media.build_command(media.MODELS["dev"], "cat", "/o.png")
        self.assertNotIn("--steps", cmd)


class TestValidateOutput(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(RuntimeError):
            media.validate_output("/no/such/file.png")

    def test_too_small(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(media.PNG_MAGIC + b"short")
            path = f.name
        try:
            with self.assertRaises(RuntimeError):
                media.validate_output(path)
        finally:
            os.unlink(path)

    def test_not_a_png(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x00" * 4096)
            path = f.name
        try:
            with self.assertRaises(RuntimeError):
                media.validate_output(path)
        finally:
            os.unlink(path)

    def test_valid_png_passes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ok.png")
            _tiny_png(path)
            media.validate_output(path)  # must not raise


class TestUnknownModel(unittest.TestCase):
    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            media.generate_image("cat", model="does-not-exist")

    def test_empty_prompt_raises(self):
        with self.assertRaises(ValueError):
            media.generate_image("")


class TestBuildVideoCommand(unittest.TestCase):
    def test_stage_and_mandatory_frame_rate(self):
        cmd = media.build_video_command("a cat", "/o.mp4", bin_path="ltx")
        self.assertEqual(cmd[:3], ["ltx", "generate", "--distilled"])
        self.assertIn("--frame-rate", cmd)
        self.assertEqual(cmd[cmd.index("--frame-rate") + 1], "24")
        self.assertIn("--prompt", cmd)

    def test_stage_override(self):
        cmd = media.build_video_command("a cat", "/o.mp4", stage="two-stages-hq", bin_path="ltx")
        self.assertIn("--two-stages-hq", cmd)
        self.assertNotIn("--distilled", cmd)

    def test_unknown_stage_raises(self):
        with self.assertRaises(ValueError):
            media.build_video_command("a cat", "/o.mp4", stage="nonsense")

    def test_optional_flags(self):
        cmd = media.build_video_command("a cat", "/o.mp4", frames=49, width=768,
                                        height=512, seed=42, low_ram=True, bin_path="ltx")
        for flag, val in (("--frames", "49"), ("--width", "768"),
                          ("--height", "512"), ("--seed", "42")):
            self.assertEqual(cmd[cmd.index(flag) + 1], val)
        self.assertIn("--low-ram", cmd)

    def test_uses_configured_bin(self):
        cmd = media.build_video_command("a cat", "/o.mp4", bin_path="/venv/bin/ltx-2-mlx")
        self.assertEqual(cmd[0], "/venv/bin/ltx-2-mlx")


class TestBuildEditCommand(unittest.TestCase):
    def test_builds_edit_invocation(self):
        cmd = media.build_edit_command("make it night", "/in.png", "/out.png",
                                       bin_path="qwen-edit")
        self.assertEqual(cmd[0], "qwen-edit")
        self.assertEqual(cmd[cmd.index("--prompt") + 1], "make it night")
        self.assertEqual(cmd[cmd.index("--image-paths") + 1], "/in.png")
        self.assertEqual(cmd[cmd.index("--output") + 1], "/out.png")
        self.assertIn("--quantize", cmd)

    def test_seed_and_quantize(self):
        cmd = media.build_edit_command("x", "/in.png", "/out.png", seed=7,
                                       quantize=4, bin_path="qwen-edit")
        self.assertEqual(cmd[cmd.index("--seed") + 1], "7")
        self.assertEqual(cmd[cmd.index("--quantize") + 1], "4")

    def test_missing_image_raises(self):
        with self.assertRaises(ValueError):
            media.build_edit_command("x", "", "/out.png")

    def test_empty_prompt_raises(self):
        with self.assertRaises(ValueError):
            media.build_edit_command("", "/in.png", "/out.png")

    def test_missing_input_path_raises(self):
        with self.assertRaises(ValueError):
            media.generate_edit("x", "/no/such/input.png")


class TestBuildUpscaleCommand(unittest.TestCase):
    def test_builds_upscale_invocation(self):
        cmd = media.build_upscale_command("/in.png", "/out.png", bin_path="seedvr2")
        self.assertEqual(cmd[0], "seedvr2")
        self.assertEqual(cmd[cmd.index("--image-path") + 1], "/in.png")
        self.assertEqual(cmd[cmd.index("--output") + 1], "/out.png")
        self.assertNotIn("--resolution", cmd)

    def test_scale_maps_to_resolution(self):
        cmd = media.build_upscale_command("/in.png", "/out.png", scale="2x",
                                          bin_path="seedvr2")
        self.assertEqual(cmd[cmd.index("--resolution") + 1], "2x")

    def test_missing_image_raises(self):
        with self.assertRaises(ValueError):
            media.build_upscale_command("", "/out.png")

    def test_missing_input_path_raises(self):
        with self.assertRaises(ValueError):
            media.generate_upscale("/no/such/input.png")


class TestValidateVideoOutput(unittest.TestCase):
    def test_missing(self):
        with self.assertRaises(RuntimeError):
            media.validate_video_output("/no/such.mp4")

    def test_not_mp4(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00" * 20480)  # big enough, but no ftyp box
            path = f.name
        try:
            with self.assertRaises(RuntimeError):
                media.validate_video_output(path)
        finally:
            os.unlink(path)

    def test_valid_mp4_passes(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 20480)
            path = f.name
        try:
            media.validate_video_output(path)  # must not raise
        finally:
            os.unlink(path)

    def test_empty_prompt_raises(self):
        with self.assertRaises(ValueError):
            media.generate_video("")


class TestFreeGpu(unittest.TestCase):
    def test_keep_skips(self):
        # keep=True must not attempt any network call; simply returns.
        media.free_gpu("image", keep=True)

    def test_no_gateway_is_swallowed(self):
        # Point at a closed port; connection refused must be caught, not raised.
        saved = (media.GATEWAY_PORT, media.KEEP_MODELS)
        try:
            media.GATEWAY_PORT = "9"  # discard port, nothing listening
            media.KEEP_MODELS = False
            media.free_gpu("video")  # must not raise
        finally:
            media.GATEWAY_PORT, media.KEEP_MODELS = saved


class TestMCP(unittest.TestCase):
    def test_initialize(self):
        r = media_mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertEqual(r["result"]["serverInfo"]["name"], "fxlla-media")

    def test_tools_list(self):
        r = media_mcp.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.assertEqual({t["name"] for t in r["result"]["tools"]},
                         {"generate_image", "generate_video", "generate_speech",
                          "edit_image", "upscale_image", "media_job_status",
                          "list_loras", "list_media_models",
                          "list_media_jobs", "cancel_media_job",
                          "describe_image"})

    def test_discovery_and_waiting_exist(self):
        # Both come from watching a real session: the model read the source
        # ten times to learn what it could do, and polled a job 47 times.
        tools = {t["name"]: t for t in
                 media_mcp.handle({"jsonrpc": "2.0", "id": 1,
                                   "method": "tools/list"})["result"]["tools"]}
        self.assertIn("list_media_models", tools)
        self.assertIn("wait_s",
                      tools["media_job_status"]["inputSchema"]["properties"])
        image = tools["generate_image"]["inputSchema"]["properties"]
        for opt in ("negative", "loras", "lora_style", "init_image"):
            self.assertIn(opt, image)

    def test_generators_expose_async(self):
        tools = {t["name"]: t for t in
                 media_mcp.handle({"jsonrpc": "2.0", "id": 10,
                                   "method": "tools/list"})["result"]["tools"]}
        for name in ("generate_image", "generate_video", "generate_speech",
                     "edit_image", "upscale_image"):
            self.assertIn("async", tools[name]["inputSchema"]["properties"], name)
        # The job tools are not generators, so they take no async flag.
        self.assertNotIn("async",
                         tools["media_job_status"]["inputSchema"]["properties"])

    def test_job_tools_require_id(self):
        self.assertTrue(media_mcp.run_job_status({}).startswith("error"))
        self.assertTrue(media_mcp.run_cancel_job({}).startswith("error"))

    def test_edit_and_upscale_require_image(self):
        tools = {t["name"]: t for t in
                 media_mcp.handle({"jsonrpc": "2.0", "id": 9,
                                   "method": "tools/list"})["result"]["tools"]}
        self.assertIn("image", tools["edit_image"]["inputSchema"]["required"])
        self.assertIn("prompt", tools["edit_image"]["inputSchema"]["required"])
        self.assertEqual(tools["upscale_image"]["inputSchema"]["required"], ["image"])

    def test_unknown_tool(self):
        r = media_mcp.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                              "params": {"name": "nope", "arguments": {}}})
        self.assertIn("error", r)

    def test_generate_requires_prompt(self):
        self.assertTrue(media_mcp.run_generate({}).startswith("error"))

    def test_generate_video_requires_prompt(self):
        self.assertTrue(media_mcp.run_generate_video({}).startswith("error"))

    def test_generate_speech_requires_text(self):
        self.assertTrue(media_mcp.run_generate_speech({}).startswith("error"))

    def test_edit_requires_image(self):
        self.assertTrue(media_mcp.run_edit({"prompt": "x"}).startswith("error"))

    def test_edit_requires_prompt(self):
        self.assertTrue(media_mcp.run_edit({"image": "/in.png"}).startswith("error"))

    def test_upscale_requires_image(self):
        self.assertTrue(media_mcp.run_upscale({}).startswith("error"))


def _write_wav(path, frames, rate=24000, width=2, channels=1):
    import array, wave
    codes = {2: "h", 4: "i"}
    with wave.open(path, "wb") as w:
        w.setnchannels(channels); w.setsampwidth(width); w.setframerate(rate)
        if width == 1:  # 8-bit WAV is unsigned, centered on 128
            w.writeframes(bytes((f + 128) & 0xFF for f in frames))
        elif width == 3:  # 24-bit little-endian
            w.writeframes(b"".join((f & 0xFFFFFF).to_bytes(3, "little") for f in frames))
        else:
            w.writeframes(array.array(codes[width], frames).tobytes())


class TestQualityAudio(unittest.TestCase):
    # Thresholds are deliberately far from real speech (measured peak ~0.95,
    # rms ~0.13), so these fixtures are unambiguous garbage.
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.rate = 24000

    def _path(self, name):
        return os.path.join(self.tmp, name)

    def _tone(self, seconds, amp=9000, freq=180, offset=0):
        import math
        n = int(self.rate * seconds)
        return [int(amp * math.sin(2 * math.pi * freq * t / self.rate)) + offset
                for t in range(n)]

    def test_pure_silence_is_rejected(self):
        p = self._path("s.wav"); _write_wav(p, [0] * self.rate * 2)
        self.assertTrue(any("silent" in x for x in quality.check_wav(p)))

    def test_click_then_silence_is_rejected(self):
        # Passes a peak check but is empty to a listener.
        p = self._path("c.wav"); _write_wav(p, [20000] * 50 + [0] * (self.rate * 3))
        self.assertTrue(any("silent for" in x for x in quality.check_wav(p)))

    def test_dc_offset_is_rejected(self):
        p = self._path("dc.wav"); _write_wav(p, self._tone(2, amp=8000, offset=12000))
        self.assertTrue(any("DC offset" in x for x in quality.check_wav(p)))

    def test_speech_like_signal_passes(self):
        # 0.4s of tone, 0.3s of silence, repeated: 43 percent pauses.
        seg = self._tone(0.4)
        gap = [0] * int(self.rate * 0.3)
        p = self._path("ok.wav"); _write_wav(p, (seg + gap) * 4)
        self.assertEqual(quality.check_wav(p), [])

    def test_every_supported_sample_width(self):
        for width in (1, 2, 3, 4):
            p = self._path("w%d.wav" % width)
            amp = {1: 60, 2: 9000, 3: 2000000, 4: 500000000}[width]
            _write_wav(p, self._tone(0.5, amp=amp), width=width)
            self.assertEqual(quality.check_wav(p), [], "width %d" % width)

    def test_24_bit_sign_extension_round_trips(self):
        # The only hand-rolled arithmetic in the module: three little-endian
        # bytes, sign-extended. Check the extremes survive it.
        extremes = [-8388608, -1, 0, 1, 8388607]
        p = self._path("s24.wav"); _write_wav(p, extremes, width=3)
        samples, full_scale, _ch, _rate = quality._samples(p)
        self.assertEqual(list(samples), extremes)
        self.assertEqual(full_scale, 8388608.0)

    def test_a_constant_waveform_is_silence_not_distortion(self):
        # All-zero bytes in an 8-bit file center to -128: full-scale by peak,
        # which used to be reported as distortion.
        p = self._path("z8.wav"); _write_wav(p, [-128] * self.rate, width=1)
        self.assertTrue(any("silent" in x for x in quality.check_wav(p)))

    def test_content_at_the_end_is_not_called_silent(self):
        # The windowed loop used to drop the trailing partial window, so the
        # verdict depended on where in the file the content sat.
        tone = self._tone(0.03, amp=30000)
        lead = [0] * int(self.rate * 0.20)
        tail = self._path("tail.wav"); _write_wav(tail, lead + tone)
        head = self._path("head.wav"); _write_wav(head, tone + lead)
        self.assertEqual(quality.check_wav(tail), quality.check_wav(head))
        self.assertEqual(quality.check_wav(tail), [])

    def test_zero_frame_file_is_reported(self):
        import wave as wavemod
        p = self._path("empty.wav")
        with wavemod.open(p, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(self.rate)
        self.assertEqual(quality.check_wav(p), ["contains no audio frames"])

    def test_unparseable_format_gets_no_verdict(self):
        # Python's wave reads PCM only. A float32 WAV is valid output this module
        # cannot measure, and "cannot measure" must never mean "reject".
        import struct as st
        n = 2000
        data = b"".join(st.pack("<f", 0.5) for _ in range(n))
        header = (b"RIFF" + st.pack("<I", 36 + len(data)) + b"WAVEfmt "
                  + st.pack("<IHHIIHH", 16, 3, 1, self.rate, self.rate * 4, 4, 32)
                  + b"data" + st.pack("<I", len(data)))
        p = self._path("f32.wav")
        with open(p, "wb") as f:
            f.write(header + data)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(quality.check_wav(p), [])

    def test_unreadable_file_gets_no_verdict(self):
        # A file this module cannot parse must not be rejected: the container
        # checks already passed, and guessing would reject valid output.
        p = self._path("bad.wav")
        with open(p, "wb") as f:
            f.write(b"RIFF____WAVEnonsense")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(quality.check_wav(p), [])

    def test_report_joins_problems_or_returns_none(self):
        self.assertIsNone(quality.report("audio", "/x.wav", []))
        one = quality.report("audio", "/x.wav", ["is silent"])
        self.assertIn("/x.wav", one)
        both = quality.report("audio", "/x.wav", ["is silent", "has a DC offset"])
        self.assertIn("; and ", both)


class TestVideoFacts(unittest.TestCase):
    # The measured truth about a produced file, so a caller cannot report its
    # own request back as the result: one called 49 frames at 24 fps "about 10
    # seconds" (2.04) and declared a 4-8 second requirement met.
    def _probe(self, payload):
        saved = quality._ffprobe
        quality._ffprobe = lambda _p: payload
        return saved

    def test_duration_frames_and_fps_are_read(self):
        saved = self._probe({"streams": [{"nb_frames": "49", "width": 512,
                                          "height": 512, "r_frame_rate": "24/1",
                                          "duration": "2.041667"}]})
        try:
            f = quality.video_facts("/x.mp4")
        finally:
            quality._ffprobe = saved
        self.assertEqual(f["duration_s"], 2.04)
        self.assertEqual(f["frames"], 49)
        self.assertEqual(f["fps"], 24.0)
        self.assertEqual((f["width"], f["height"]), (512, 512))

    def test_container_duration_is_the_fallback(self):
        saved = self._probe({"streams": [{"nb_frames": "10"}],
                             "format": {"duration": "0.5"}})
        try:
            self.assertEqual(quality.video_facts("/x.mp4")["duration_s"], 0.5)
        finally:
            quality._ffprobe = saved

    def test_no_ffprobe_yields_no_facts_rather_than_guesses(self):
        saved = self._probe(None)
        try:
            self.assertEqual(quality.video_facts("/x.mp4"), {})
        finally:
            quality._ffprobe = saved

    def test_unparseable_rate_is_omitted_not_invented(self):
        saved = self._probe({"streams": [{"r_frame_rate": "0/0", "duration": "1"}]})
        try:
            f = quality.video_facts("/x.mp4")
        finally:
            quality._ffprobe = saved
        self.assertNotIn("fps", f)
        self.assertEqual(f["duration_s"], 1.0)


class TestQualityImageVideo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _png(self, name, width, height):
        path = os.path.join(self.tmp, name)
        _tiny_png(path)
        with open(path, "r+b") as f:
            f.seek(16)
            f.write(struct.pack(">II", width, height))
        return path

    def test_zero_dimensions_rejected(self):
        self.assertTrue(any("zero dimensions" in x
                            for x in quality.check_png(self._png("z.png", 0, 0))))

    def test_small_image_is_accepted(self):
        # No arbitrary minimum: validate_output's byte floor already catches
        # truncation, and the repo's own fixtures are 1x1.
        self.assertEqual(quality.check_png(self._png("t.png", 1, 1)), [])

    def test_normal_image_passes(self):
        self.assertEqual(quality.check_png(self._png("n.png", 512, 512)), [])

    def test_headerless_file_rejected(self):
        p = os.path.join(self.tmp, "x.png")
        with open(p, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
        self.assertTrue(quality.check_png(p))

    @unittest.skipUnless(shutil.which("ffprobe") and shutil.which("ffmpeg"),
                         "needs ffmpeg/ffprobe")
    def test_real_clip_and_user_requested_short_clips_pass(self):
        # Frame count and duration are caller-controlled, so a one-frame or
        # fraction-of-a-second clip is a request, not a defect.
        import subprocess as sp
        for name, args in (
            ("clip.mp4", ["-f", "lavfi", "-i", "testsrc=size=64x64:rate=24:duration=1"]),
            ("one.mp4", ["-f", "lavfi", "-i", "color=c=red:size=64x64:rate=24",
                         "-frames:v", "1"]),
        ):
            path = os.path.join(self.tmp, name)
            sp.run(["ffmpeg", "-v", "error"] + args + ["-pix_fmt", "yuv420p", path],
                   check=True, capture_output=True)
            self.assertEqual(quality.check_video(path), [], name)

    @unittest.skipUnless(shutil.which("ffprobe") and shutil.which("ffmpeg"),
                         "needs ffmpeg/ffprobe")
    def test_container_without_video_is_reported(self):
        import subprocess as sp
        path = os.path.join(self.tmp, "audio.mp4")
        sp.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "sine=frequency=440:duration=1", path], check=True, capture_output=True)
        self.assertEqual(quality.check_video(path), ["has no video stream"])

    def test_video_without_ffprobe_makes_no_claim(self):
        # No ffprobe means nothing to inspect; guessing would be worse.
        saved = os.environ.get("PATH", "")
        os.environ["PATH"] = "/nonexistent"
        try:
            self.assertEqual(quality.check_video(os.path.join(self.tmp, "any.mp4")), [])
        finally:
            os.environ["PATH"] = saved


class TestQualityGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.png = os.path.join(self.tmp, "a.png")
        _tiny_png(self.png)

    def tearDown(self):
        os.environ.pop("FXLLA_MEDIA_SKIP_QUALITY", None)

    def test_skip_env_disables_the_gate(self):
        for value in ("1", "true", "yes"):
            os.environ["FXLLA_MEDIA_SKIP_QUALITY"] = value
            self.assertTrue(quality.skip_quality_checks(), value)
        for value in ("", "0", "false"):
            os.environ["FXLLA_MEDIA_SKIP_QUALITY"] = value
            self.assertFalse(quality.skip_quality_checks(), value)

    def test_gate_raises_with_the_escape_hatch_in_the_message(self):
        with self.assertRaises(RuntimeError) as ctx:
            media._check_quality("image", self.png, lambda p: ["is blank"])
        self.assertIn("FXLLA_MEDIA_SKIP_QUALITY", str(ctx.exception))

    def test_gate_is_silent_when_there_are_no_problems(self):
        media._check_quality("image", self.png, lambda p: [])

    def test_a_broken_checker_never_fails_the_render(self):
        def boom(_path):
            raise ValueError("checker bug")
        with contextlib.redirect_stderr(io.StringIO()):
            media._check_quality("image", self.png, boom)  # must not raise

    def test_skipped_gate_does_not_call_the_checker(self):
        os.environ["FXLLA_MEDIA_SKIP_QUALITY"] = "1"
        def boom(_path):
            raise AssertionError("checker must not run when skipped")
        media._check_quality("image", self.png, boom)


class TestJobs(unittest.TestCase):
    # Exercises the job lifecycle against a stand-in generator, so no model is
    # loaded: jobs.run shells out to jobs.GENERATE, which the tests replace.
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._saved = (jobs.JOBS_DIR, jobs.GENERATE)
        jobs.JOBS_DIR = os.path.join(self.tmp, "jobs")

    def tearDown(self):
        jobs.JOBS_DIR, jobs.GENERATE = self._saved

    def _fake_generator(self, body):
        path = os.path.join(self.tmp, "fake_gen.py")
        with open(path, "w") as fh:
            fh.write(body)
        jobs.GENERATE = path
        return path

    def _record(self, **fields):
        rec = {"id": jobs.new_id(), "kind": "image", "status": "queued",
               "argv": [], "summary": "", "output": None, "error": None,
               "pid": None, "created": 1.0, "started": None, "finished": None}
        rec.update(fields)
        jobs._write(rec)
        return rec

    def test_valid_id_rejects_path_traversal(self):
        # Ids arrive from MCP arguments and are joined into a path.
        self.assertTrue(jobs.valid_id(jobs.new_id()))
        for bad in ("../../etc/passwd", "..", "", "abc", "1785376545-XYZ", None):
            self.assertFalse(jobs.valid_id(bad), bad)

    def test_get_rejects_bad_id(self):
        self.assertIsNone(jobs.get("../secret"))

    def test_run_writes_output_on_success(self):
        self._fake_generator("print('/tmp/out.png')\n")
        rec = self._record()
        jobs.run(rec["id"])
        done = jobs._read(rec["id"])
        self.assertEqual(done["status"], "done")
        self.assertEqual(done["output"], "/tmp/out.png")
        self.assertIsNotNone(done["finished"])

    def test_run_records_failure(self):
        self._fake_generator("import sys; sys.exit('boom')\n")
        rec = self._record()
        jobs.run(rec["id"])
        failed = jobs._read(rec["id"])
        self.assertEqual(failed["status"], "failed")
        self.assertIn("boom", failed["error"])

    def test_run_fails_when_generator_prints_nothing(self):
        self._fake_generator("pass\n")
        rec = self._record()
        jobs.run(rec["id"])
        self.assertEqual(jobs._read(rec["id"])["status"], "failed")

    def test_run_skips_a_cancelled_job(self):
        self._fake_generator("print('/tmp/should-not-run.png')\n")
        rec = self._record(status="cancelled")
        jobs.run(rec["id"])
        self.assertEqual(jobs._read(rec["id"])["status"], "cancelled")

    def test_dead_worker_is_reaped(self):
        # A pid that cannot be running any more (reaped by the OS long ago).
        rec = self._record(status="running", pid=999999)
        self.assertEqual(jobs.get(rec["id"])["status"], "failed")
        self.assertIn("died", jobs._read(rec["id"])["error"])

    def test_cancel_marks_cancelled(self):
        rec = self._record(status="queued")
        self.assertEqual(jobs.cancel(rec["id"])["status"], "cancelled")

    def test_cancel_leaves_finished_jobs_alone(self):
        rec = self._record(status="done", output="/tmp/x.png")
        self.assertEqual(jobs.cancel(rec["id"])["status"], "done")

    def test_listing_is_newest_first(self):
        self._record(created=1.0, kind="image")
        self._record(created=5.0, kind="video")
        self.assertEqual([r["kind"] for r in jobs.listing()], ["video", "image"])

    def test_prune_keeps_active_jobs(self):
        self._record(status="done", created=1.0)
        self._record(status="failed", created=2.0)
        active = self._record(status="running", pid=os.getpid(), created=3.0)
        self.assertEqual(jobs.prune(), 2)
        self.assertEqual([r["id"] for r in jobs.listing()], [active["id"]])

    def test_describe_shows_last_error_line(self):
        rec = self._record(status="failed",
                           error="Traceback...\n  File x\nValueError: bad input")
        self.assertIn("ValueError: bad input", jobs.describe(rec))


class TestModelCatalog(unittest.TestCase):
    # The catalog replaced a hardcoded dict so per-model differences stop being
    # special cases. Capabilities came from probing each CLI's --help.
    def test_the_shipped_catalog_parses_and_has_the_default(self):
        models = media.load_models()
        self.assertIn(media.DEFAULT_MODEL, models)
        self.assertGreaterEqual(len(models), 8)
        for name, spec in models.items():
            self.assertTrue(spec["cli"].startswith("mflux-"), name)
            self.assertIsInstance(spec["caps"], set)

    def test_capabilities_are_not_uniform(self):
        # If every model had the same caps the mechanism would be pointless.
        # Verified against the installed CLIs: ideogram4 and boogu take no
        # init image, everything else does.
        models = media.load_models()
        with_init = {n for n, s in models.items() if "init-image" in s["caps"]}
        self.assertTrue(with_init)
        self.assertNotEqual(with_init, set(models))

    def test_comments_and_short_lines_are_skipped(self):
        import tempfile
        p = os.path.join(tempfile.mkdtemp(), "m.conf")
        open(p, "w").write("# a comment\n\nbroken|line\n"
                           "ok | mflux-generate-x |  | 4 | lora,negative | note\n")
        models = media.load_models(p)
        self.assertEqual(list(models), ["ok"])
        self.assertEqual(models["ok"]["steps"], 4)
        self.assertEqual(models["ok"]["caps"], {"lora", "negative"})
        self.assertIsNone(models["ok"]["base_model"])

    def test_an_unsupported_flag_is_refused_by_name(self):
        # The alternative is mflux exiting with its own usage text after the
        # caller already waited, naming neither the model nor an alternative.
        spec = {"cli": "mflux-generate-x", "caps": {"negative"}, "steps": None}
        with self.assertRaises(ValueError) as ctx:
            media.build_command(spec, "cat", "/o.png", loras=["x.safetensors"],
                                model_name="boogu")
        msg = str(ctx.exception)
        self.assertIn("boogu", msg)
        self.assertIn("--lora", msg)
        self.assertIn("Models that do", msg)

    def test_a_default_adapts_but_an_explicit_request_is_validated(self):
        # Erroring on a flag the caller never named would make an unsupported
        # default look like their mistake.
        spec = {"cli": "mflux-generate-x", "caps": {"negative"}, "steps": 8}
        cmd = media.build_command(spec, "cat", "/o.png", model_name="m")
        self.assertNotIn("--quantize", cmd)
        self.assertNotIn("--steps", cmd)
        with self.assertRaises(ValueError):
            media.build_command(spec, "cat", "/o.png", quantize=4, model_name="m")
        with self.assertRaises(ValueError):
            media.build_command(spec, "cat", "/o.png", steps=3, model_name="m")

    def test_every_documented_capability_is_gated(self):
        # Five of the ten were ungated: a model whose CLI lacks --seed would
        # have had it forwarded anyway, which is the failure the gate exists
        # to prevent for the others.
        full = {"cli": "x", "steps": None,
                "caps": {"negative", "prompt-file", "init-image", "lora",
                         "lora-style", "quantize", "steps", "seed",
                         "dimensions"}}
        for cap, kwargs in (
                ("quantize", {"quantize": 4}), ("steps", {"steps": 2}),
                ("seed", {"seed": 1}), ("dimensions", {"aspect": "1:1"}),
                ("dimensions", {"width": 512})):
            lacking = dict(full, caps=full["caps"] - {cap})
            with self.assertRaises(ValueError, msg="%s ungated" % cap):
                media.build_command(lacking, "c", "/o.png", model_name="m", **kwargs)
            media.build_command(full, "c", "/o.png", model_name="m", **kwargs)

    def test_a_supported_flag_passes_through(self):
        spec = {"cli": "mflux-generate-x", "caps": {"negative"}, "steps": None}
        cmd = media.build_command(spec, "cat", "/o.png", negative="blurry")
        self.assertEqual(cmd[cmd.index("--negative-prompt") + 1], "blurry")


class TestIdeogramCaption(unittest.TestCase):
    # Ideogram 4 takes a JSON caption with placed elements. The rules are
    # mflux's own; validating here names a malformed caption before a render
    # starts instead of after it, as a schema warning.
    def _caption(self, elements):
        return json.dumps({"high_level_description": "poster",
                           "compositional_deconstruction": {"elements": elements}})

    def test_prose_is_left_alone(self):
        self.assertEqual(media.check_ideogram_caption("a cat on a beach"), [])
        self.assertEqual(media.check_ideogram_caption('"just a string"'), [])

    def test_a_well_formed_caption_passes(self):
        good = self._caption([
            {"type": "text", "bbox": [100, 200, 300, 800], "text": "HOLA",
             "color_palette": ["#FF0000"]},
            {"type": "obj", "bbox": [0, 0, 1000, 1000], "desc": "sky"}])
        self.assertEqual(media.check_ideogram_caption(good), [])

    def test_axis_order_is_caught(self):
        # [y_min, x_min, y_max, x_max] - Y first - is the trap nobody guesses.
        problems = media.check_ideogram_caption(
            self._caption([{"type": "text", "bbox": [900, 100, 200, 400], "text": "X"}]))
        self.assertTrue(any("Y first" in p for p in problems))

    def test_coordinate_space_is_caught(self):
        problems = media.check_ideogram_caption(
            self._caption([{"type": "obj", "bbox": [0, 0, 1200, 50]}]))
        self.assertTrue(any("0..1000" in p for p in problems))

    def test_floats_are_rejected(self):
        problems = media.check_ideogram_caption(
            self._caption([{"type": "obj", "bbox": [0.0, 0, 100, 50]}]))
        self.assertTrue(any("integers" in p for p in problems))

    def test_bad_length_is_caught(self):
        problems = media.check_ideogram_caption(
            self._caption([{"type": "obj", "bbox": [0, 0, 100]}]))
        self.assertTrue(any("y_min, x_min, y_max, x_max" in p for p in problems))

    def test_element_type_and_text_requirement(self):
        problems = media.check_ideogram_caption(
            self._caption([{"type": "thing"}, {"type": "text"}]))
        self.assertTrue(any("obj, text" in p for p in problems))
        self.assertTrue(any("needs a 'text' string" in p for p in problems))

    def test_style_palette_allows_sixteen_but_elements_only_five(self):
        # mflux's own limits differ by context; hardcoding 5 falsely rejected
        # a valid caption, and a false rejection blocks a working render.
        sixteen = ["#%06X" % i for i in range(16)]
        ok = json.dumps({"high_level_description": "x",
                         "style_description": {"color_palette": sixteen},
                         "compositional_deconstruction": {"elements": []}})
        self.assertEqual(media.check_ideogram_caption(ok), [])
        too_many = json.dumps({
            "high_level_description": "x",
            "compositional_deconstruction": {"elements": [
                {"type": "obj", "color_palette": sixteen[:6]}]}})
        self.assertTrue(any("at most 5" in p
                            for p in media.check_ideogram_caption(too_many)))

    def test_hex_colors_must_be_uppercase(self):
        # mflux's own rule: it accepts "#F2B134" and rejects "#f2b134". The
        # difference is invisible unless something names it, and a caption
        # that looked fine here would come back with warnings from mflux.
        lower = json.dumps({"high_level_description": "x",
                            "style_description": {"color_palette": ["#f2b134"]},
                            "compositional_deconstruction": {"elements": []}})
        problems = media.check_ideogram_caption(lower)
        self.assertTrue(any("UPPERCASE" in p for p in problems))
        upper = lower.replace("#f2b134", "#F2B134")
        self.assertEqual(media.check_ideogram_caption(upper), [])

    def test_a_realistic_caption_passes_end_to_end(self):
        # The example shipped in the docs, kept honest: if the schema tightens
        # and this stops validating, the documented prompt is wrong too.
        caption = json.dumps({
            "high_level_description": "A vintage poster of a chrome lighter.",
            "style_description": {
                "aesthetics": "mid-century poster, flat vector shapes",
                "lighting": "warm rim light, soft vignette",
                "medium": "screen print",
                "art_style": "retro Swiss design",
                "color_palette": ["#0D3B45", "#F2B134", "#E8E3D3"]},
            "compositional_deconstruction": {
                "background": "flat teal field with a faint radial glow",
                "elements": [
                    {"type": "text", "bbox": [80, 120, 240, 880], "text": "ZIPPO",
                     "desc": "condensed display type", "color_palette": ["#E8E3D3"]},
                    {"type": "obj", "bbox": [300, 330, 760, 670],
                     "desc": "chrome lighter, lid open, tall flame"}]}})
        self.assertEqual(media.check_ideogram_caption(caption), [])

    def test_the_documented_example_passes_its_own_validator(self):
        # The schema handed to a model through list_media_models must agree
        # with the one enforced here, or fxlla teaches a format it rejects.
        example = json.dumps(media.IDEOGRAM_PROMPT_FORMAT["example"])
        self.assertEqual(media.check_ideogram_caption(example), [])

    def test_the_documented_schema_states_the_traps(self):
        fmt = media.IDEOGRAM_PROMPT_FORMAT
        self.assertIn("Y FIRST", fmt["element"]["bbox"])
        self.assertIn("0..1000", fmt["element"]["bbox"])
        self.assertIn("UPPERCASE", fmt["colors"])

    def test_a_missing_composition_section_is_flagged(self):
        problems = media.check_ideogram_caption(
            json.dumps({"high_level_description": "just a description"}))
        self.assertTrue(any("compositional_deconstruction" in p for p in problems))

    def test_unknown_keys_and_palette_limits(self):
        problems = media.check_ideogram_caption(json.dumps({"nonsense": 1}))
        self.assertTrue(any("unknown top-level keys" in p for p in problems))
        problems = media.check_ideogram_caption(self._caption(
            [{"type": "obj", "color_palette": ["#FFF", "#000000", "#000000",
                                               "#000000", "#000000", "#000000"]}]))
        self.assertTrue(any("at most 5 colors" in p for p in problems))
        self.assertTrue(any("#RRGGBB" in p for p in problems))

    def test_generate_image_refuses_a_bad_caption(self):
        with self.assertRaises(ValueError) as ctx:
            media.generate_image(
                self._caption([{"type": "text", "bbox": [900, 0, 100, 10], "text": "x"}]),
                model="ideogram4")
        self.assertIn("JSON caption", str(ctx.exception))


class TestControlAndDepth(unittest.TestCase):
    # Two controlnet families that are NOT interchangeable: FLUX takes a
    # checkpoint plus a control image as separate repeatable flags, Z-Image
    # takes one combined spec. Both stack.
    def _img(self):
        p = os.path.join(tempfile.mkdtemp(), "ctl.png")
        open(p, "wb").write(media.PNG_MAGIC + b"0" * 64)
        return p

    def _flux(self):
        return {"cli": "x", "steps": None,
                "caps": {"controlnet", "controlnet-image", "controlnet-strength"}}

    def _zimage(self):
        return {"cli": "x", "steps": None,
                "caps": {"control-spec", "controlnet-strength"}}

    def test_flux_form_splits_image_and_checkpoint(self):
        p = self._img()
        cmd = media.build_command(self._flux(), "c", "/o.png",
                                  controls=["%s,InstantX/FLUX.1-dev-Controlnet-Canny" % p],
                                  model_name="controlnet")
        self.assertEqual(cmd[cmd.index("--controlnet-image-path") + 1], p)
        self.assertEqual(cmd[cmd.index("--controlnet-path") + 1],
                         "InstantX/FLUX.1-dev-Controlnet-Canny")

    def test_flux_form_stacks(self):
        a, b = self._img(), self._img()
        cmd = media.build_command(self._flux(), "c", "/o.png", controls=[a, b],
                                  model_name="controlnet")
        self.assertEqual(cmd.count("--controlnet-image-path"), 2)

    def test_zimage_form_passes_the_spec_through_and_stacks(self):
        cmd = media.build_command(self._zimage(), "c", "/o.png",
                                  controls=["pose:pose.png:0.8", "depth:d.png"],
                                  model_name="z-controlnet")
        self.assertEqual(cmd.count("--control"), 2)
        self.assertIn("pose:pose.png:0.8", cmd)
        # The combined spec must NOT be split into the FLUX flags.
        self.assertNotIn("--controlnet-image-path", cmd)

    def test_a_z_image_spec_survives_a_comma(self):
        # split_ref splits on commas and the z-image spec uses colons, so a
        # path containing a comma must NOT be torn apart: that branch never
        # runs for this family, and this pins it.
        cmd = media.build_command(self._zimage(), "c", "/o.png",
                                  controls=["pose:/a,b/p.png:0.8"],
                                  model_name="z-controlnet")
        self.assertEqual(cmd[cmd.index("--control") + 1], "pose:/a,b/p.png:0.8")

    def test_a_flux_path_containing_a_colon_still_works(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "a:b.png")
        open(p, "wb").write(media.PNG_MAGIC)
        cmd = media.build_command(self._flux(), "c", "/o.png", controls=[p],
                                  model_name="controlnet")
        self.assertEqual(cmd[cmd.index("--controlnet-image-path") + 1], p)

    def test_the_wrong_form_says_which_one_this_model_takes(self):
        # Confusing the two families is the likely mistake; "file not found"
        # alone would send someone hunting for a path problem.
        with self.assertRaises(ValueError) as ctx:
            media.build_command(self._flux(), "c", "/o.png",
                                controls=["pose:p.png:0.8"], model_name="controlnet")
        self.assertIn("z-controlnet form", str(ctx.exception))
        self.assertIn("IMAGE[,CHECKPOINT]", str(ctx.exception))

    def test_a_missing_control_image_is_named(self):
        with self.assertRaises(ValueError) as ctx:
            media.build_command(self._flux(), "c", "/o.png",
                                controls=["/no/such.png"], model_name="controlnet")
        self.assertIn("/no/such.png", str(ctx.exception))

    def test_a_model_without_control_refuses(self):
        plain = {"cli": "x", "steps": None, "caps": {"negative"}}
        with self.assertRaises(ValueError) as ctx:
            media.build_command(plain, "c", "/o.png", controls=[self._img()],
                                model_name="z-image-turbo")
        self.assertIn("z-image-turbo", str(ctx.exception))

    def test_depth_map_can_be_supplied_and_saved(self):
        spec = {"cli": "x", "steps": None, "caps": {"depth-image", "save-depth"}}
        d = self._img()
        cmd = media.build_command(spec, "c", "/o.png", depth_image=d,
                                  save_depth=True, model_name="depth")
        self.assertEqual(cmd[cmd.index("--depth-image-path") + 1], d)
        self.assertIn("--save-depth-map", cmd)

    def test_prompt_file_replaces_the_prompt_flag(self):
        # mflux declares them mutually exclusive ("--prompt-file: not allowed
        # with argument --prompt"), verified against the real CLI, so sending
        # both made every --prompt-file render die in argparse.
        spec = {"cli": "x", "steps": None, "caps": {"prompt-file"}}
        f = os.path.join(tempfile.mkdtemp(), "p.txt")
        open(f, "w").write("a cat")
        cmd = media.build_command(spec, "ignored", "/o.png", prompt_file=f,
                                  model_name="m")
        self.assertIn("--prompt-file", cmd)
        self.assertNotIn("--prompt", cmd)
        self.assertEqual(cmd[cmd.index("--prompt-file") + 1], f)

    def test_the_depth_map_path_is_derivable(self):
        # mflux writes "<base>_depth_map<ext>"; the caller needs it to chain a
        # control step, so it cannot stay an internal naming convention.
        self.assertEqual(media.depth_map_path("/a/b/img.png"),
                         "/a/b/img_depth_map.png")
        self.assertEqual(media.depth_map_path("/a/b/img"),
                         "/a/b/img_depth_map.png")

    def test_guidance(self):
        spec = {"cli": "x", "steps": None, "caps": {"guidance"}}
        cmd = media.build_command(spec, "c", "/o.png", guidance=7.5, model_name="m")
        self.assertEqual(cmd[cmd.index("--guidance") + 1], "7.5")
        bare = {"cli": "x", "steps": None, "caps": set()}
        with self.assertRaises(ValueError):
            media.build_command(bare, "c", "/o.png", guidance=7.5, model_name="m")


class TestLoraDiscovery(unittest.TestCase):
    # Searching only the civitai download directory answered "none" to
    # someone holding ten: people train their own and keep them beside the
    # project that produced them.
    def test_configured_directories_are_searched(self):
        d = tempfile.mkdtemp()
        _write_lora(os.path.join(d, "style.safetensors"))
        saved = os.environ.get("FXLLA_LORA_DIRS")
        os.environ["FXLLA_LORA_DIRS"] = d
        try:
            paths = [r[0] for r in media.find_loras()]
            self.assertIn(os.path.join(d, "style.safetensors"), paths)
        finally:
            if saved is None:
                del os.environ["FXLLA_LORA_DIRS"]
            else:
                os.environ["FXLLA_LORA_DIRS"] = saved

    def test_several_directories_and_nested_files(self):
        a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
        os.makedirs(os.path.join(a, "sub"))
        _write_lora(os.path.join(a, "sub", "deep.safetensors"))
        open(os.path.join(b, "flat.ckpt"), "w").close()
        saved = os.environ.get("FXLLA_LORA_DIRS")
        os.environ["FXLLA_LORA_DIRS"] = "%s:%s" % (a, b)
        try:
            names = [os.path.basename(r[0]) for r in media.find_loras()]
            self.assertIn("deep.safetensors", names)
            self.assertIn("flat.ckpt", names)
        finally:
            if saved is None:
                del os.environ["FXLLA_LORA_DIRS"]
            else:
                os.environ["FXLLA_LORA_DIRS"] = saved

    def test_the_civitai_directory_is_always_searched(self):
        self.assertTrue(any(d.endswith("civitai") for d in media.lora_dirs()))

    def test_the_base_model_is_inferred_from_the_architecture(self):
        # Most adapters declare no base, and the hidden dimension is a property
        # of the model one was fitted to, so it survives renaming. The table
        # was measured from adapters whose base was known independently, and
        # it recovers every declared label in the collection it was built from.
        def hdr(dim, names=("blocks.0.attn.lora_A.weight",), extra=()):
            h = {n: {"dtype": "F16", "shape": [dim, 16], "data_offsets": [0, 2]}
                 for n in names}
            for n in extra:
                h[n] = {"dtype": "F16", "shape": [dim, 16], "data_offsets": [0, 2]}
            return h
        self.assertEqual(media.infer_base_model(hdr(1536)), "krea2")
        self.assertEqual(media.infer_base_model(hdr(4608)), "ideogram4")
        self.assertEqual(media.infer_base_model(hdr(3840)), "z-image")
        self.assertEqual(media.infer_base_model(hdr(4096)), "ltx2")
        self.assertEqual(media.infer_base_model(hdr(64)), "")

    def test_flux_and_qwen_share_a_dimension_and_are_split_by_names(self):
        # Both are 3072, so the dimension alone would pair an adapter with the
        # wrong model. Qwen's joint attention adds projections FLUX has no
        # equivalent of.
        def hdr(names):
            return {n: {"dtype": "F16", "shape": [3072, 16], "data_offsets": [0, 2]}
                    for n in names}
        self.assertEqual(media.infer_base_model(
            hdr(["transformer.blocks.0.attn.to_k.lora_A.weight"])), "flux")
        self.assertEqual(media.infer_base_model(
            hdr(["transformer_blocks.0.attn.add_k_proj.lora_down.weight"])), "qwen")

    def test_an_inferred_base_is_marked_as_a_guess(self):
        # A guess presented as fact would send someone hunting for why a
        # correctly-paired adapter did nothing.
        import struct as _s
        d = tempfile.mkdtemp()
        h = {"blocks.0.attn.lora_A.weight": {"dtype": "F16", "shape": [1536, 16],
                                             "data_offsets": [0, 2]}}
        blob = json.dumps(h).encode()
        p = os.path.join(d, "x.safetensors")
        with open(p, "wb") as fh:
            fh.write(_s.pack("<Q", len(blob)) + blob + b"\x00\x00")
        self.assertEqual(media.lora_base_model(p), "~krea2")

    def test_a_declared_base_beats_the_guess(self):
        import struct as _s
        d = tempfile.mkdtemp()
        h = {"blocks.0.attn.lora_A.weight": {"dtype": "F16", "shape": [1536, 16],
                                             "data_offsets": [0, 2]},
             "__metadata__": {"ss_base_model_version": "something-specific"}}
        blob = json.dumps(h).encode()
        p = os.path.join(d, "y.safetensors")
        with open(p, "wb") as fh:
            fh.write(_s.pack("<Q", len(blob)) + blob + b"\x00\x00")
        self.assertEqual(media.lora_base_model(p), "something-specific")

    def test_a_lora_is_identified_by_its_tensors_not_its_name(self):
        # Filenames lie both ways: an adapter can be called Krea2-realism-V1
        # and a 26 GB base model can be called Krea-2-Turbo. Only the tensor
        # names inside the safetensors header settle it.
        import struct as _s
        d = tempfile.mkdtemp()

        def write(name, keys, meta=None):
            hdr = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
                   for k in keys}
            if meta:
                hdr["__metadata__"] = meta
            blob = json.dumps(hdr).encode()
            p = os.path.join(d, name)
            with open(p, "wb") as fh:
                fh.write(_s.pack("<Q", len(blob)) + blob + b"\x00\x00")
            return p

        adapter = write("anything.safetensors",
                        ["blocks.0.attn.lora_A.weight", "blocks.0.attn.lora_B.weight"],
                        {"ss_base_model_version": "krea2"})
        lokr = write("lokr.safetensors", ["blocks.0.attn.gate.lokr_w1"])
        base = write("Krea-2-Turbo.safetensors", ["blocks.0.attn.qknorm.scale"])
        self.assertEqual(media.lora_base_model(adapter), "krea2")
        self.assertEqual(media.lora_base_model(lokr), "")
        self.assertIsNone(media.lora_base_model(base))
        self.assertIsNone(media.lora_base_model("/no/such/file.safetensors"))

    def test_the_declared_base_model_is_reported(self):
        # Which base a LoRA was trained for is what makes a collection usable;
        # applying a krea2 adapter to z-image does nothing good.
        import struct as _s
        d = tempfile.mkdtemp()
        hdr = {"blocks.0.lora_A.weight": {"dtype": "F16", "shape": [1],
                                          "data_offsets": [0, 2]},
               "__metadata__": {"ss_base_model_version": "krea2"}}
        blob = json.dumps(hdr).encode()
        with open(os.path.join(d, "x.safetensors"), "wb") as fh:
            fh.write(_s.pack("<Q", len(blob)) + blob + b"\x00\x00")
        saved = os.environ.get("FXLLA_LORA_DIRS")
        os.environ["FXLLA_LORA_DIRS"] = d
        try:
            rows = [r for r in media.find_loras() if r[0].startswith(d)]
        finally:
            if saved is None:
                del os.environ["FXLLA_LORA_DIRS"]
            else:
                os.environ["FXLLA_LORA_DIRS"] = saved
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][2], "krea2")

    def test_huggingface_cached_loras_are_found_by_repo_id(self):
        # mflux takes a repo id directly for --lora, so a cached one is usable
        # with no path. These arrive by download or as another tool's side
        # effect, never through the civitai folder.
        cache = tempfile.mkdtemp()
        import struct as _s

        def write(repo, keys):
            snap = os.path.join(cache, "hub", repo, "snapshots", "abc")
            os.makedirs(snap)
            hdr = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
                   for k in keys}
            blob = json.dumps(hdr).encode()
            with open(os.path.join(snap, "w.safetensors"), "wb") as fh:
                fh.write(_s.pack("<Q", len(blob)) + blob + b"\x00\x00")

        write("models--krea--Krea-2-LoRA-darkbrush", ["b.0.lora_A.weight"])
        write("models--org--Some-Base-Model", ["b.0.attn.scale"])
        saved = os.environ.get("FXLLA_MEDIA_HF_HOME")
        os.environ["FXLLA_MEDIA_HF_HOME"] = cache
        try:
            names = [r[0] for r in media._hf_cache_loras()]
        finally:
            if saved is None:
                del os.environ["FXLLA_MEDIA_HF_HOME"]
            else:
                os.environ["FXLLA_MEDIA_HF_HOME"] = saved
        self.assertIn("krea/Krea-2-LoRA-darkbrush", names)
        # The base model in the same cache is not a LoRA and must not appear,
        # which a name or size heuristic could not tell.
        self.assertNotIn("org/Some-Base-Model", names)

    def test_a_missing_hf_cache_is_not_fatal(self):
        saved = os.environ.get("FXLLA_MEDIA_HF_HOME")
        os.environ["FXLLA_MEDIA_HF_HOME"] = "/no/such/cache"
        try:
            self.assertEqual(media._hf_cache_loras(), [])
        finally:
            if saved is None:
                del os.environ["FXLLA_MEDIA_HF_HOME"]
            else:
                os.environ["FXLLA_MEDIA_HF_HOME"] = saved

    def test_a_missing_directory_is_skipped_not_fatal(self):
        saved = os.environ.get("FXLLA_LORA_DIRS")
        os.environ["FXLLA_LORA_DIRS"] = "/no/such/dir"
        try:
            media.find_loras()  # must not raise
        finally:
            if saved is None:
                del os.environ["FXLLA_LORA_DIRS"]
            else:
                os.environ["FXLLA_LORA_DIRS"] = saved


class TestLoRA(unittest.TestCase):
    # `fxlla pull civitai:<id>` could download LoRAs from the day it shipped
    # and nothing could apply one.
    def _spec(self):
        return {"cli": "mflux-generate-x", "steps": None,
                "caps": {"lora", "lora-style", "negative", "init-image"}}

    def _lora(self):
        import tempfile
        p = os.path.join(tempfile.mkdtemp(), "style.safetensors")
        open(p, "w").close()
        return p

    def test_a_path_with_scale_comma_form(self):
        # Comma, not spaces: a multi-value argparse flag swallowed the prompt,
        # and a JSON array of plain strings could not express the pair at all.
        p = self._lora()
        cmd = media.build_command(self._spec(), "c", "/o.png", loras=["%s,0.8" % p])
        i = cmd.index("--lora")
        self.assertEqual(cmd[i + 1:i + 3], [p, "0.8"])

    def test_a_tilde_path_reaches_the_backend_expanded(self):
        # Validating the expanded path while passing the literal "~" told the
        # caller the file was fine and handed mflux something it cannot
        # resolve: nothing here runs through a shell.
        import tempfile
        real = self._lora()
        home = os.path.expanduser("~")
        if not real.startswith(home):
            self.skipTest("temp dir is not under HOME")
        cmd = media.build_command(self._spec(), "c", "/o.png",
                                  loras=["~" + real[len(home):]])
        self.assertEqual(cmd[cmd.index("--lora") + 1], real)
        self.assertNotIn("~", cmd[cmd.index("--lora") + 1])

    def test_several_loras_repeat_the_flag(self):
        a, b = self._lora(), self._lora()
        cmd = media.build_command(self._spec(), "c", "/o.png", loras=[a, b])
        self.assertEqual(cmd.count("--lora"), 2)

    def test_a_huggingface_repo_id_is_accepted(self):
        # mflux takes org/name too, so only a local-looking path is checked.
        cmd = media.build_command(self._spec(), "c", "/o.png", loras=["org/name"])
        self.assertIn("org/name", cmd)

    def test_a_missing_local_lora_is_named(self):
        with self.assertRaises(ValueError) as ctx:
            media.build_command(self._spec(), "c", "/o.png",
                                loras=["/no/such.safetensors"])
        self.assertIn("/no/such.safetensors", str(ctx.exception))

    def test_built_in_styles(self):
        cmd = media.build_command(self._spec(), "c", "/o.png",
                                  lora_style="storyboard")
        self.assertEqual(cmd[cmd.index("--lora-style") + 1], "storyboard")
        self.assertIn("storyboard", media.LORA_STYLES)


class TestPromptControls(unittest.TestCase):
    def _spec(self):
        return {"cli": "mflux-generate-x", "steps": None,
                "caps": {"negative", "prompt-file", "init-image"}}

    def test_negative_prompt(self):
        cmd = media.build_command(self._spec(), "c", "/o.png", negative="text, watermark")
        self.assertEqual(cmd[cmd.index("--negative-prompt") + 1], "text, watermark")

    def test_prompt_file_must_exist(self):
        with self.assertRaises(ValueError) as ctx:
            media.build_command(self._spec(), "c", "/o.png", prompt_file="/no/p.txt")
        self.assertIn("/no/p.txt", str(ctx.exception))

    def test_init_image_reaches_the_backend(self):
        # It used to assert --image-path, mflux's deprecated flag, which cannot
        # carry a strength. Asserting it kept the limitation in place: the test
        # would have failed the moment chaining became possible.
        import tempfile
        p = os.path.join(tempfile.mkdtemp(), "in.png")
        with open(p, "wb") as fh:
            fh.write(media.PNG_MAGIC)
        cmd = media.build_command(self._spec(), "c", "/o.png", init_image=p)
        self.assertEqual(cmd[cmd.index("--image") + 1], p)


class TestJobWait(unittest.TestCase):
    # An agent with no way to await a render polls instead: one issued 47
    # status calls waiting on a single video.
    def test_returns_as_soon_as_the_job_leaves_the_active_states(self):
        import time
        calls = {"n": 0}
        saved = jobs.get

        def _fake(_jid):
            calls["n"] += 1
            return {"status": "running"} if calls["n"] < 3 else {"status": "done"}
        jobs.get = _fake
        try:
            begin = time.monotonic()
            rec = jobs.wait("id", timeout_s=10, poll_s=0.01)
        finally:
            jobs.get = saved
        self.assertEqual(rec["status"], "done")
        self.assertLess(time.monotonic() - begin, 1.0)

    def test_a_timeout_returns_the_record_rather_than_raising(self):
        saved = jobs.get
        jobs.get = lambda _jid: {"status": "running"}
        try:
            rec = jobs.wait("id", timeout_s=0.05, poll_s=0.01)
        finally:
            jobs.get = saved
        self.assertEqual(rec["status"], "running")

    def test_an_unknown_job_returns_none_immediately(self):
        saved = jobs.get
        jobs.get = lambda _jid: None
        try:
            self.assertIsNone(jobs.wait("id", timeout_s=10, poll_s=0.01))
        finally:
            jobs.get = saved


class TestImageFacts(unittest.TestCase):
    def test_dimensions_and_size_from_the_header(self):
        import struct, tempfile
        p = os.path.join(tempfile.mkdtemp(), "i.png")
        with open(p, "wb") as fh:
            fh.write(media.PNG_MAGIC + b"\x00\x00\x00\x0dIHDR"
                     + struct.pack(">II", 512, 384) + b"0" * 40)
        f = quality.image_facts(p)
        self.assertEqual((f["width"], f["height"]), (512, 384))
        self.assertGreater(f["bytes"], 0)

    def test_a_missing_file_yields_nothing(self):
        self.assertEqual(quality.image_facts("/no/such.png"), {})


class TestVideoImageAnchors(unittest.TestCase):
    # Image-to-video. Asked for "a video transitioning between these two
    # images", a model instead described them in the prompt and produced an
    # unrelated clip: without anchors the stills are never used at all.
    def _img(self):
        import tempfile
        p = os.path.join(tempfile.mkdtemp(), "ref.png")
        open(p, "wb").write(media.PNG_MAGIC + b"0" * 64)
        return p

    def test_a_bare_path_anchors_the_opening_frame(self):
        p = self._img()
        cmd = media.build_video_command("x", "/o.mp4", images=[p])
        self.assertEqual(cmd[cmd.index("--image") + 1], p)

    def test_two_anchors_repeat_the_flag(self):
        a, b = self._img(), self._img()
        cmd = media.build_video_command("x", "/o.mp4",
                                        images=["%s,0,1.0" % a, "%s,96,1.0" % b])
        self.assertEqual(cmd.count("--image"), 2)
        first = cmd.index("--image")
        self.assertEqual(cmd[first + 1:first + 4], [a, "0", "1.0"])

    def test_a_missing_reference_is_named(self):
        # Otherwise it surfaces as a generic backend failure minutes later.
        with self.assertRaises(ValueError) as ctx:
            media.build_video_command("x", "/o.mp4", images=["/no/such.png"])
        self.assertIn("/no/such.png", str(ctx.exception))

    def test_no_images_leaves_the_flag_out(self):
        self.assertNotIn("--image", media.build_video_command("x", "/o.mp4"))

    def test_the_flag_is_the_one_the_backend_accepts(self):
        # A model invented --images (plural); neither generate.py nor ltx has
        # it, so every call would have died in argparse.
        p = self._img()
        cmd = media.build_video_command("x", "/o.mp4", images=[p])
        self.assertNotIn("--images", cmd)


class TestResolveOutput(unittest.TestCase):
    # Passing a directory ("put it in ~/Downloads") used to reach the backend
    # as a filename and fail deep inside it. A real model hit exactly that and
    # abandoned the tool for ~30 shell commands.
    def test_none_lands_in_the_default_dir(self):
        p = media.resolve_output(None, "image", "png")
        self.assertTrue(p.startswith(media.OUT_DIR))
        self.assertTrue(p.endswith(".png"))

    def test_a_file_path_is_used_verbatim(self):
        self.assertEqual(media.resolve_output("/tmp/x.png", "image", "png"),
                         "/tmp/x.png")

    def test_an_existing_directory_gets_a_generated_name_inside(self):
        import tempfile
        d = tempfile.mkdtemp()
        p = media.resolve_output(d, "image", "png")
        self.assertEqual(os.path.dirname(p), d)
        self.assertTrue(p.endswith(".png"))
        self.assertNotEqual(p, d)

    def test_a_trailing_separator_means_a_directory(self):
        import tempfile
        d = os.path.join(tempfile.mkdtemp(), "new-dir") + os.sep
        p = media.resolve_output(d, "video", "mp4")
        self.assertTrue(os.path.isdir(d))
        self.assertEqual(os.path.dirname(p) + os.sep, d)

    def test_every_generator_kind_uses_it(self):
        for kind, ext in (("image", "png"), ("video", "mp4"), ("voice", "wav"),
                          ("edit", "png"), ("upscale", "png")):
            self.assertTrue(media.resolve_output(None, kind, ext).endswith("." + ext))

    def test_a_tilde_expands(self):
        # "~/Downloads" is what a person says and what an agent passes on, and
        # nothing here runs through a shell - unexpanded it would create a
        # directory literally named "~" beside the working directory.
        p = media.resolve_output("~/Downloads", "image", "png")
        self.assertTrue(p.startswith(os.path.expanduser("~/Downloads")), p)
        self.assertNotIn("~", p)


class TestAspectConflict(unittest.TestCase):
    # Measured: a request for 512x512 WITH aspect 1:1 produced 1024x1024,
    # because aspect won and the dimensions were dropped without a word.
    def test_aspect_with_dimensions_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            media.build_command(media.MODELS["z-image-turbo"], "a cat", "/o.png",
                                width=512, height=512, aspect="1:1")
        self.assertIn("512", str(ctx.exception))
        self.assertIn("1:1", str(ctx.exception))

    def test_either_alone_still_works(self):
        only_dims = media.build_command(media.MODELS["z-image-turbo"], "c", "/o.png",
                                        width=512, height=512)
        self.assertEqual(only_dims[only_dims.index("--width") + 1], "512")
        only_aspect = media.build_command(media.MODELS["z-image-turbo"], "c", "/o.png",
                                          aspect="16:9")
        # mflux has no --aspect; fxlla resolves it to a one-megapixel size.
        self.assertNotIn("--aspect", only_aspect)
        self.assertEqual(only_aspect[only_aspect.index("--width") + 1], "1376")
        self.assertEqual(only_aspect[only_aspect.index("--height") + 1], "768")

    def test_every_aspect_lands_on_the_16_grid(self):
        for aspect in media.ASPECT_RATIOS:
            width, height = media.aspect_size(aspect)
            self.assertEqual((width % 16, height % 16), (0, 0), aspect)
            self.assertLess(abs(width * height - 1024 * 1024), 1024 * 1024 * 0.05)

    def test_an_unknown_aspect_is_refused(self):
        with self.assertRaises(ValueError):
            media.build_command(media.MODELS["z-image-turbo"], "c", "/o.png",
                                aspect="5:4")


class TestDimensionGrid(unittest.TestCase):
    # Ideogram 4 raises on a size off its 16-pixel grid, and raises only after
    # its weights are resident: a 1080-wide poster died minutes in. Most other
    # latent creators divide by 8 and silently accept anything, so this must
    # stay a per-model rule and not become a blanket check.
    def test_a_size_off_the_grid_is_refused_before_spawning(self):
        with self.assertRaises(ValueError) as ctx:
            media.build_command(media.MODELS["ideogram4"], "p", "/o.png",
                                width=1080, height=1920, model_name="ideogram4")
        message = str(ctx.exception)
        self.assertIn("1080", message)
        self.assertIn("1088", message)  # names a size that works

    def test_a_size_on_the_grid_passes(self):
        cmd = media.build_command(media.MODELS["ideogram4"], "p", "/o.png",
                                  width=1088, height=1920, model_name="ideogram4")
        self.assertEqual(cmd[cmd.index("--width") + 1], "1088")

    def test_models_without_a_grid_take_any_size(self):
        cmd = media.build_command(media.MODELS["z-image-turbo"], "p", "/o.png",
                                  width=1080, height=1920,
                                  model_name="z-image-turbo")
        self.assertEqual(cmd[cmd.index("--width") + 1], "1080")

    def test_the_constraint_is_reported_by_the_catalog(self):
        # Discoverable, or the caller only learns it from a failed render.
        self.assertEqual(media._DIM_STEP.get("ideogram4"), 16)


class TestOneCanvasKindsNormalise(unittest.TestCase):
    # An edit and an upscale each produce one still, so their cost scales with
    # its area exactly as a render's does. Keying the normalisation on
    # kind == "image" left both reporting a bare median that does not scale to
    # another size - measured here at 263 s for one edit and 18 s for one
    # upscale, neither transferable without the per-megapixel rate.
    def _records(self):
        return [{"kind": k, "status": "done", "started": 0, "finished": 100,
                 "argv": [k, "--width", "1024", "--height", "1024"],
                 "output": None}
                for k in ("image", "edit", "upscale", "video")]

    def test_every_single_still_kind_gets_a_rate(self):
        seen = media.observed_timings(self._records())
        for kind in ("edit", "upscale"):
            self.assertIn("s_per_mp", seen[kind], kind)

    def test_video_still_does_not(self):
        # Its cost also scales with the frame count; an area-normalised figure
        # would be a fresh wrong signal.
        seen = media.observed_timings(self._records())
        self.assertNotIn("s_per_mp", seen["video:distilled"])


class TestBackendWarnings(unittest.TestCase):
    # stderr was read only when the render FAILED, so a warning on a good run
    # was captured and dropped - including mflux's own "--steps is ignored",
    # which is the backend naming the exact class of bug that had to be found
    # three times by reading its source instead.
    STDERR = (
        "  0%|          | 0/20 [00:00<?, ?it/s]\n"
        "/x/ideogram4_generate.py:44: UserWarning: --steps is ignored; "
        "Ideogram 4 presets define the step count.\n"
        '  warnings.warn("--steps is ignored", stacklevel=1)\n'
        " 50%|#####     | 10/20 [00:30<00:30]\n"
        "/x/ideogram4_generate.py:46: UserWarning: --guidance is ignored; "
        "Ideogram 4 presets define the guidance schedule.\n"
        "INFO: loading weights\n")

    def test_it_extracts_what_the_backend_said(self):
        found = media.backend_warnings(self.STDERR)
        self.assertEqual(len(found), 2)
        self.assertTrue(found[0].startswith("--steps is ignored"))

    def test_progress_and_logs_are_not_warnings(self):
        # Relaying all of stderr would drown the signal in progress bars and
        # train a reader to skip it.
        self.assertEqual(
            media.backend_warnings(" 50%|#####| 10/20\nINFO: loading\n"), [])

    def test_a_repeated_warning_is_reported_once(self):
        self.assertEqual(len(media.backend_warnings(self.STDERR * 3)), 2)

    def test_nothing_said_is_nothing_reported(self):
        self.assertEqual(media.backend_warnings(""), [])
        self.assertEqual(media.backend_warnings(None), [])

    def test_every_backend_call_reports_what_it_said(self):
        # The invariant, checked structurally so a sixth backend call cannot be
        # added with the failure branch and without the success one - which is
        # exactly how the first five came to drop their warnings.
        # Counted by how a GENERATOR reports failure - it raises RuntimeError
        # with the backend's stderr. Metadata queries against the same backend
        # (the capability dump) swallow a bad exit and return None instead, and
        # their stderr is not something a caller needs relayed.
        source = open(media.__file__.replace(".pyc", ".py")).read()
        failures = source.count("raise RuntimeError((proc.stderr")
        reports = source.count("_report_warnings(proc.stderr)")
        self.assertEqual(reports, failures,
                         "%d generator calls raise on failure, %d report warnings"
                         % (failures, reports))

    def test_a_finished_job_carries_them(self):
        class Proc(object):
            returncode, stdout = 0, "/out/x.png\n"
            stderr = "warning: --steps is ignored; presets define it.\n"

        jobs_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, jobs_dir, True)
        saved_dir, jobs.JOBS_DIR = jobs.JOBS_DIR, jobs_dir
        self.addCleanup(setattr, jobs, "JOBS_DIR", saved_dir)
        saved_notify, jobs.notify = jobs.notify, lambda rec: None
        self.addCleanup(setattr, jobs, "notify", saved_notify)
        saved_run, jobs.subprocess.run = jobs.subprocess.run, lambda c, **k: Proc()
        self.addCleanup(setattr, jobs.subprocess, "run", saved_run)
        jobs._write({"id": "1785500000-abcdef", "kind": "image",
                     "status": "queued", "argv": ["image", "x"], "summary": "x",
                     "output": None, "error": None, "pid": None,
                     "created": 1785500000.0, "started": None,
                     "finished": None, "log": ""})
        jobs.run("1785500000-abcdef")
        rec = jobs._read("1785500000-abcdef")
        self.assertEqual(rec["status"], "done")
        self.assertEqual(len(rec.get("warnings") or []), 1)
        self.assertIn("--steps is ignored", rec["warnings"][0])


class TestInitImageStrength(unittest.TestCase):
    # The knob that makes chaining possible. fxlla passed mflux's deprecated
    # --image-path, which cannot carry a strength, so a second pass over a
    # first render either changed nothing or destroyed it - render-small-then-
    # refine-large, the most common workflow shape there is, was unreachable.
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.png = os.path.join(self.dir, "first.png")
        with open(self.png, "wb") as fh:
            fh.write(media.PNG_MAGIC + b"0" * 64)

    def _image_args(self, **kw):
        cmd = media.build_command(media.MODELS["z-image-turbo"], "x", "/o.png",
                                  model_name="z-image-turbo", **kw)
        start = cmd.index("--image")
        end = next((i for i in range(start + 1, len(cmd))
                    if cmd[i].startswith("--")), len(cmd))
        return cmd[start + 1:end]

    def test_it_uses_the_flag_that_can_carry_a_strength(self):
        cmd = media.build_command(media.MODELS["z-image-turbo"], "x", "/o.png",
                                  model_name="z-image-turbo",
                                  init_image=self.png)
        self.assertIn("--image", cmd)
        self.assertNotIn("--image-path", cmd)  # deprecated, drops the strength

    def test_a_strength_reaches_the_backend(self):
        self.assertEqual(self._image_args(init_image=self.png, strength=0.4),
                         [self.png, "0.4"])

    def test_the_comma_form_carries_it_too(self):
        self.assertEqual(self._image_args(init_image=self.png + ",0.6"),
                         [self.png, "0.6"])

    def test_giving_it_twice_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self._image_args(init_image=self.png + ",0.6", strength=0.4)
        self.assertIn("once", str(ctx.exception))

    def test_a_percentage_is_caught_before_the_render(self):
        # The real mistake: reading "strength" as a percent and sending 60.
        with self.assertRaises(ValueError) as ctx:
            self._image_args(init_image=self.png, strength=60)
        self.assertIn("between 0 and 1", str(ctx.exception))

    def test_both_ends_are_legitimate(self):
        for value in (0, 1):
            self.assertEqual(
                self._image_args(init_image=self.png, strength=value)[1],
                str(value))

    def test_strength_without_an_image_says_so(self):
        with self.assertRaises(ValueError) as ctx:
            media.build_command(media.MODELS["z-image-turbo"], "x", "/o.png",
                                model_name="z-image-turbo", strength=0.4)
        self.assertIn("init-image", str(ctx.exception))

    def test_a_missing_file_is_named_before_spending_a_render(self):
        with self.assertRaises(ValueError) as ctx:
            self._image_args(init_image="/nope/missing.png", strength=0.4)
        self.assertIn("missing.png", str(ctx.exception))

    def test_the_legacy_flag_is_used_where_that_is_the_only_one(self):
        # mflux's newer `--image PATH [STRENGTH]` did not reach every CLI: the
        # depth model still has only the deprecated --image-path, so emitting
        # the new flag there died in argparse ("unrecognized arguments:
        # --image"). Found by diffing this catalog against mflux's own
        # capability dump, which is the point of having one.
        cmd = media.build_command(media.MODELS["depth"], "x", "/o.png",
                                  model_name="depth", init_image=self.png)
        self.assertIn("--image-path", cmd)
        self.assertNotIn("--image", cmd)

    def test_that_model_refuses_a_strength_it_cannot_carry(self):
        with self.assertRaises(ValueError) as ctx:
            media.build_command(media.MODELS["depth"], "x", "/o.png",
                                model_name="depth", init_image=self.png,
                                strength=0.4)
        self.assertIn("--image-path", str(ctx.exception))

    def test_exactly_one_init_image_cap_per_model(self):
        # The two spellings are alternatives, never both: claiming each would
        # make which flag gets emitted depend on evaluation order.
        for name, spec in media.MODELS.items():
            both = {"init-image", "init-image-path"} & spec["caps"]
            self.assertLessEqual(len(both), 1, "%s claims %s" % (name, both))

    def test_a_model_without_img2img_refuses_it(self):
        with self.assertRaises(ValueError) as ctx:
            media.build_command(media.MODELS["boogu"], "x", "/o.png",
                                model_name="boogu", init_image=self.png)
        self.assertIn("boogu", str(ctx.exception))

    def test_the_mcp_forwards_it(self):
        self.assertIn("strength",
                      dict(media_mcp._IMAGE_FLAGS))
        tool = next(t for t in media_mcp.TOOLS if t["name"] == "generate_image")
        self.assertIn("strength", tool["inputSchema"]["properties"])


class TestNegativeNeedsCfg(unittest.TestCase):
    # A negative prompt only exists through classifier-free guidance. mflux
    # drops it outright below guidance 1 (`if guidance <= 1.0: return
    # text_encodings, None`), and the distilled models declare
    # supports_guidance=False, which pins guidance to 0. So on those, a
    # negative prompt is read, encoded and discarded silently - and this
    # catalog listed it on the DEFAULT model while the skill recommended it.
    def test_negative_requires_guidance_or_a_preset(self):
        offenders = sorted(
            name for name, spec in media.MODELS.items()
            if "negative" in spec["caps"]
            and not {"guidance", "preset"} & spec["caps"])
        self.assertEqual(offenders, [], "negative without CFG: %s" % offenders)

    def test_the_distilled_models_refuse_it_by_name(self):
        for name in ("z-image-turbo", "boogu", "schnell", "z-controlnet"):
            with self.assertRaises(ValueError, msg=name) as ctx:
                media.build_command(media.MODELS[name], "p", "/o.png",
                                    negative="watermark", model_name=name)
            self.assertIn("--negative-prompt", str(ctx.exception))

    def test_a_model_with_real_cfg_still_takes_it(self):
        cmd = media.build_command(media.MODELS["krea2"], "p", "/o.png",
                                  negative="watermark", model_name="krea2")
        self.assertEqual(cmd[cmd.index("--negative-prompt") + 1], "watermark")


class TestDescribeImage(unittest.TestCase):
    # The inverse of every other call here: it reads an image instead of making
    # one, so a model that cannot see can check its own work. Deliberately
    # synchronous - measured around ten seconds, where the job queue would be
    # ceremony - and it goes through the gateway so the model stays resident.
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.png = os.path.join(self.dir, "shot.png")
        with open(self.png, "wb") as fh:
            fh.write(media.PNG_MAGIC + b"\x00\x00\x00\x0dIHDR"
                     + (64).to_bytes(4, "big") + (64).to_bytes(4, "big")
                     + b"\x00" * 40)

    def test_a_missing_file_is_named_before_any_request(self):
        with self.assertRaises(ValueError) as ctx:
            media.describe_image(os.path.join(self.dir, "nope.png"))
        self.assertIn("nope.png", str(ctx.exception))

    def test_something_that_is_not_an_image_is_refused(self):
        junk = os.path.join(self.dir, "notes.txt")
        with open(junk, "w") as fh:
            fh.write("hello")
        with self.assertRaises(ValueError):
            media.describe_image(junk)

    def test_no_gateway_says_how_to_start_one(self):
        saved = media.GATEWAY_PORT
        media.GATEWAY_PORT = "1"  # nothing listens there
        self.addCleanup(setattr, media, "GATEWAY_PORT", saved)
        with self.assertRaises(RuntimeError) as ctx:
            media.describe_image(self.png)
        self.assertIn("fxlla serve", str(ctx.exception))

    def test_the_default_question_asks_for_text_verbatim(self):
        # A generator inventing gibberish lettering is the failure this is for,
        # and it is only found by asking what is there rather than whether
        # something expected is present.
        self.assertIn("quoted", media.VISION_QUESTION.lower())
        for leading in ("correct", "should say", "is it right"):
            self.assertNotIn(leading, media.VISION_QUESTION.lower())

    def test_it_is_not_routed_through_the_job_queue(self):
        # _spawn would submit a background job and hand back an id for
        # something already answered.
        self.assertIn("describe_image", dict(
            (t["name"], t) for t in media_mcp.TOOLS))
        source = open(media_mcp.__file__.replace(".pyc", ".py")).read()
        body = source.split("def run_describe(")[1].split("\ndef ")[0]
        # A CALL, not a mention: the comment above it names _spawn to say why
        # it is wrong here, and matching prose would fail on its own reasoning.
        self.assertNotIn("_spawn(", body)
        self.assertIn("_exec(", body)

    def test_the_tool_warns_against_a_leading_question(self):
        tool = next(t for t in media_mcp.TOOLS if t["name"] == "describe_image")
        self.assertIn("NEVER put the expected answer", tool["description"])


class TestCatalogMatchesTheBackend(unittest.TestCase):
    """The catalog is transcribed; mflux publishes the same contract. Diff them.

    Over one week four capabilities were declared that the backend accepts and
    silently discards - negative and guidance on the CFG-off models, steps and
    guidance on ideogram4, negative on the FLUX family, and an init-image flag
    the depth CLI never had. Every one was found by reading mflux's source
    after something behaved oddly. mflux now publishes which options each
    CLI honours, so the table can be checked on every run instead.

    Skipped where mflux is absent or predates the dump, which is CI: a machine
    that cannot check must not report agreement.
    """

    # fxlla cap -> the flag mflux would report. Deliberately partial: caps that
    # describe fxlla's own shaping of an argument (control-spec, controlnet,
    # depth-image) have no one-to-one flag and are covered by their own tests.
    FLAGS = {"negative": "--negative-prompt", "guidance": "--guidance",
             "steps": "--steps", "seed": "--seed", "quantize": "--quantize",
             "dimensions": "--width", "lora": "--lora",
             "lora-style": "--lora-style", "prompt-file": "--prompt-file",
             "preset": "--preset", "pid-decode": "--pid-decode",
             "save-depth": "--save-depth-map",
             "controlnet-strength": "--controlnet-strength"}

    @classmethod
    def setUpClass(cls):
        cls.dump = media.backend_capabilities()
        if cls.dump is None:
            raise unittest.SkipTest("mflux capability dump unavailable")
        cls.status = {c["command"]: {o["flag"]: o["status"] for o in c["options"]}
                      for c in cls.dump["commands"]}

    def _rows(self):
        for name, spec in media.MODELS.items():
            status = self.status.get(spec["cli"])
            if status is not None:
                yield name, spec, status

    def test_every_catalog_cli_is_in_the_dump(self):
        missing = sorted(s["cli"] for s in media.MODELS.values()
                         if s["cli"] not in self.status)
        self.assertEqual(missing, [], "not reported by mflux: %s" % missing)

    def test_nothing_declared_is_discarded_by_the_backend(self):
        # The failure this whole class exists for: a flag accepted and thrown
        # away. Nothing fails, the image is plausible, and the only evidence is
        # that the option had no effect.
        bad = ["%s.%s" % (name, cap)
               for name, spec, status in self._rows()
               for cap, flag in self.FLAGS.items()
               if cap in spec["caps"] and status.get(flag) == "ignored"]
        self.assertEqual(bad, [], "declared but ignored by mflux: %s" % bad)

    def test_nothing_declared_is_absent_from_its_cli(self):
        # The depth model never gained --image; emitting it died in argparse.
        bad = ["%s.%s (no %s)" % (name, cap, flag)
               for name, spec, status in self._rows()
               for cap, flag in self.FLAGS.items()
               if cap in spec["caps"] and flag not in status]
        self.assertEqual(bad, [], "declared but absent: %s" % bad)

    def test_nothing_honoured_is_silently_omitted(self):
        # The other direction: a capability the backend has and callers cannot
        # reach. `conditional` is not a disagreement - the dump cannot know
        # --base-model, and this catalog splits such a CLI into one alias per
        # base model, which is strictly more precise.
        bad = ["%s.%s" % (name, cap)
               for name, spec, status in self._rows()
               for cap, flag in self.FLAGS.items()
               if cap not in spec["caps"] and status.get(flag) == "honored"]
        self.assertEqual(bad, [], "honoured but unreachable: %s" % bad)

    def test_the_init_image_spelling_is_the_one_that_cli_has(self):
        # --image and --image-path are alternatives, not synonyms: the second
        # is deprecated and cannot carry a strength, so where both exist the
        # newer one is claimed on purpose.
        for name, spec, status in self._rows():
            claimed = {"init-image", "init-image-path"} & spec["caps"]
            self.assertLessEqual(len(claimed), 1, "%s claims both" % name)
            if claimed:
                flag = "--image" if "init-image" in claimed else "--image-path"
                self.assertEqual(status.get(flag), "honored",
                                 "%s claims %s" % (name, flag))
            elif status.get("--image") == "honored":
                self.fail("%s could take --image and claims neither" % name)


class TestPidDecode(unittest.TestCase):
    # NVIDIA's pixel-diffusion decoder replaces the VAE decode and emits 4x the
    # requested size, so it is an upscale inside the render. It also pulls ~8 GB
    # of its own weights, one repo gated - a requirement that comes from an
    # OPTION rather than from the model, which the consent gate had no way to
    # express before.
    def test_the_flag_reaches_the_backend(self):
        cmd = media.build_command(media.MODELS["z-image-turbo"], "x", "/o.png",
                                  model_name="z-image-turbo", pid_decode=True)
        self.assertIn("--pid-decode", cmd)

    def test_a_model_without_it_is_refused_by_name(self):
        with self.assertRaises(ValueError) as ctx:
            media.build_command(media.MODELS["boogu"], "x", "/o.png",
                                model_name="boogu", pid_decode=True)
        self.assertIn("boogu", str(ctx.exception))

    def test_sigma_rides_along(self):
        cmd = media.build_command(media.MODELS["krea2"], "x", "/o.png",
                                  model_name="krea2", pid_decode=True,
                                  pid_degrade_sigma=0.2)
        self.assertEqual(cmd[cmd.index("--pid-degrade-sigma") + 1], "0.2")

    def test_sigma_outside_the_distilled_range_is_refused(self):
        # mflux names the bound itself: PiD saw sigma ~ U[0.0, 0.8].
        for value in (-0.1, 1.5):
            with self.assertRaises(ValueError, msg=str(value)):
                media.build_command(media.MODELS["krea2"], "x", "/o.png",
                                    model_name="krea2", pid_decode=True,
                                    pid_degrade_sigma=value)

    def test_both_ends_of_the_range_are_legitimate(self):
        for value in (0, 0.8):
            cmd = media.build_command(media.MODELS["krea2"], "x", "/o.png",
                                      model_name="krea2", pid_decode=True,
                                      pid_degrade_sigma=value)
            self.assertEqual(cmd[cmd.index("--pid-degrade-sigma") + 1], str(value))

    def test_sigma_without_the_decoder_says_so(self):
        with self.assertRaises(ValueError) as ctx:
            media.build_command(media.MODELS["krea2"], "x", "/o.png",
                                model_name="krea2", pid_degrade_sigma=0.2)
        self.assertIn("--pid-decode", str(ctx.exception))

    def test_the_catalog_matches_what_the_backend_offers(self):
        # Probed from each CLI's --help, not assumed: ten of sixteen.
        have = sorted(n for n, s in media.MODELS.items() if "pid-decode" in s["caps"])
        self.assertEqual(have, ["dev", "ernie", "ernie-turbo", "flux2-klein",
                                "ideogram4", "krea2", "qwen", "schnell",
                                "z-image", "z-image-turbo"])

    def test_the_mcp_forwards_the_switch(self):
        self.assertIn("pid_decode", dict(media_mcp._IMAGE_SWITCHES))
        tool = next(t for t in media_mcp.TOOLS if t["name"] == "generate_image")
        for key in ("pid_decode", "pid_degrade_sigma"):
            self.assertIn(key, tool["inputSchema"]["properties"])


class TestOptionWeightsAreGated(unittest.TestCase):
    # A weight requirement can come from an option, not just from the model.
    # Resolving only the model's alias let the gate pass on cached model weights
    # and then fetch 8 GB mid-render - the failure this module exists to stop,
    # and the same shape as the controlnet rows that listed only the adapter.
    def setUp(self):
        self.empty = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.empty, True)
        saved = os.environ.get("FXLLA_MEDIA_HF_HOME")
        os.environ["FXLLA_MEDIA_HF_HOME"] = self.empty
        self.addCleanup(
            lambda: os.environ.__setitem__("FXLLA_MEDIA_HF_HOME", saved)
            if saved is not None else os.environ.pop("FXLLA_MEDIA_HF_HOME", None))

    def test_the_option_alias_is_checked_too(self):
        missing, size = weights.missing_for("image", "z-image-turbo", ["pid"])
        self.assertIn("nvidia/PiD", missing)
        self.assertIn("google/gemma-2-2b-it", missing)
        self.assertIn("14GB", size)

    def test_without_the_option_it_is_not(self):
        missing, _ = weights.missing_for("image", "z-image-turbo")
        self.assertNotIn("nvidia/PiD", missing)

    def test_the_refusal_names_both_pre_fetches(self):
        with self.assertRaises(RuntimeError) as ctx:
            weights.require("image", "z-image-turbo", ["pid"])
        message = str(ctx.exception)
        self.assertIn("fxlla pull media:z-image-turbo", message)
        self.assertIn("fxlla pull media:pid", message)

    def _run_cli(self, *extra):
        return subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "generate.py"),
             "image", "a cat", "--model", "z-image-turbo", "--output", "/tmp/x.png"]
            + list(extra),
            capture_output=True, text=True,
            env=dict(os.environ, FXLLA_STORE="/tmp",
                     FXLLA_MEDIA_HF_HOME=self.empty, FXLLA_ASSUME_YES=""))

    def test_the_real_invocation_is_refused_not_just_the_helper(self):
        # The line that protects an actual render is the one in main() that
        # turns the flag into an extra alias. Testing missing_for alone left it
        # free to be deleted, which a mutation proved.
        proc = self._run_cli("--pid-decode")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("nvidia/PiD", proc.stdout + proc.stderr)

    def test_the_same_render_without_the_flag_does_not_mention_them(self):
        proc = self._run_cli()
        self.assertNotIn("nvidia/PiD", proc.stdout + proc.stderr)

    def test_an_unknown_extra_is_not_an_opinion(self):
        # Not in the catalog means no claim, rather than a wrong one.
        missing, _ = weights.missing_for("image", "z-image-turbo", ["nope"])
        self.assertNotIn("nope", missing)


class TestIdeogramPreset(unittest.TestCase):
    # Ideogram 4's argparse takes --steps and --guidance and then warns that
    # both are ignored, because its presets fix them. Listing those as caps let
    # a caller ask for 12 steps, wait for the preset's 20, and be told nothing.
    def test_steps_is_refused_because_it_would_be_ignored(self):
        with self.assertRaises(ValueError) as ctx:
            media.build_command(media.MODELS["ideogram4"], "p", "/o.png",
                                steps=12, model_name="ideogram4")
        self.assertIn("--steps", str(ctx.exception))

    def test_guidance_is_refused_for_the_same_reason(self):
        with self.assertRaises(ValueError):
            media.build_command(media.MODELS["ideogram4"], "p", "/o.png",
                                guidance=5.0, model_name="ideogram4")

    def test_the_preset_is_how_cost_is_set(self):
        cmd = media.build_command(media.MODELS["ideogram4"], "p", "/o.png",
                                  preset="v4_turbo_12", model_name="ideogram4")
        self.assertEqual(cmd[cmd.index("--preset") + 1], "V4_TURBO_12")

    def test_an_unknown_preset_lists_the_real_ones(self):
        with self.assertRaises(ValueError) as ctx:
            media.build_command(media.MODELS["ideogram4"], "p", "/o.png",
                                preset="fast", model_name="ideogram4")
        self.assertIn("V4_DEFAULT_20", str(ctx.exception))

    def test_no_other_model_takes_a_preset(self):
        with self.assertRaises(ValueError) as ctx:
            media.build_command(media.MODELS["krea2"], "p", "/o.png",
                                preset="V4_TURBO_12", model_name="krea2")
        self.assertIn("ideogram4", str(ctx.exception))


class TestDefaultStepsAreReported(unittest.TestCase):
    # Cost is roughly linear in steps and the defaults span 4 to 50, so a blank
    # tells a caller nothing: an eight-minute render and a one-minute one look
    # identical until both have been waited for.
    def test_every_model_reports_what_it_will_run(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            media.cmd_models(argparse.Namespace(json=True))
        models = json.loads(out.getvalue())["models"]
        missing = [n for n, m in models.items() if not m.get("default_steps")]
        self.assertEqual(missing, [], "no step count for: %s" % missing)

    def test_a_catalog_pin_wins_over_the_backend_default(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            media.cmd_models(argparse.Namespace(json=True))
        models = json.loads(out.getvalue())["models"]
        self.assertEqual(models["z-image-turbo"]["default_steps"],
                         media.MODELS["z-image-turbo"]["steps"])

    def test_the_presets_travel_with_the_model_that_has_them(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            media.cmd_models(argparse.Namespace(json=True))
        models = json.loads(out.getvalue())["models"]
        self.assertEqual(models["ideogram4"]["presets"]["V4_TURBO_12"], 12)
        self.assertNotIn("presets", models["krea2"])


class TestVideoStepKnobs(unittest.TestCase):
    # Video had no cost surface at all: the stage was the only visible control
    # and the step counts under it were unreachable, so "make it faster" had
    # exactly one answer regardless of what was actually expensive.
    def test_one_stage_takes_steps(self):
        cmd = media.build_video_command("p", "/o.mp4", stage="one-stage",
                                        bin_path="ltx", steps=12)
        self.assertEqual(cmd[cmd.index("--steps") + 1], "12")

    def test_the_two_stage_pipelines_take_stage_steps(self):
        for stage in ("distilled", "two-stage", "two-stages-hq"):
            cmd = media.build_video_command("p", "/o.mp4", stage=stage,
                                            bin_path="ltx", stage1_steps=12,
                                            stage2_steps=2)
            self.assertEqual(cmd[cmd.index("--stage1-steps") + 1], "12", stage)
            self.assertEqual(cmd[cmd.index("--stage2-steps") + 1], "2", stage)

    def test_a_knob_the_stage_ignores_is_refused_and_names_the_right_one(self):
        # Per ltx's own --help, --steps is one-stage only. Forwarding it
        # anywhere else is accepted and discarded, which is the exact failure
        # the image capability gate exists to prevent.
        with self.assertRaises(ValueError) as ctx:
            media.build_video_command("p", "/o.mp4", stage="distilled",
                                      bin_path="ltx", steps=12)
        self.assertIn("--stage1-steps", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            media.build_video_command("p", "/o.mp4", stage="one-stage",
                                      bin_path="ltx", stage1_steps=12)
        self.assertIn("--steps", str(ctx.exception))

    def test_no_step_flags_when_none_asked_for(self):
        cmd = media.build_video_command("p", "/o.mp4", bin_path="ltx")
        for flag in ("--steps", "--stage1-steps", "--stage2-steps"):
            self.assertNotIn(flag, cmd)

    def test_every_stage_publishes_its_cost(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            media.cmd_models(argparse.Namespace(json=True))
        video = json.loads(out.getvalue())["video"]
        self.assertEqual(video["default_stage"], media.DEFAULT_STAGE)
        self.assertEqual(sorted(video["stages"]), sorted(media.VIDEO_STAGES))
        for name, info in video["stages"].items():
            self.assertTrue(info.get("steps") or info.get("stage1_steps"), name)
            self.assertIn(info["steps_flag"], ("steps", "stage1_steps"), name)


class TestObservedTimings(unittest.TestCase):
    # Steps were published as the cost signal and that invited the wrong
    # inference: a caller concluded 8-step krea2 was as fast as 8-step
    # z-image-turbo (measured 9x apart) and produced a timing table off by 25x.
    # Measured seconds are the answer, and they were already on disk unused.
    def _records(self):
        return [
            {"kind": "image", "status": "done", "started": 0, "finished": 60,
             "argv": ["image", "--model", "z-image-turbo"],
             "output": "/nope.png"},
            {"kind": "image", "status": "done", "started": 0, "finished": 54,
             "argv": ["image", "--model", "z-image-turbo",
                      "--width", "1024", "--height", "1024"],
             "output": "/nope.png"},
            {"kind": "image", "status": "done", "started": 0, "finished": 520,
             "argv": ["image", "--model", "krea2",
                      "--width", "1088", "--height", "1920"],
             "output": "/nope.png"},
            # Dimensions present, so a version that normalised video by area
            # would produce an s_per_mp here instead of leaving it out.
            {"kind": "video", "status": "done", "started": 0, "finished": 55,
             "argv": ["video", "--distilled", "--width", "1280",
                      "--height", "720"], "output": "/nope.mp4"},
            {"kind": "image", "status": "running", "started": 0,
             "finished": None, "argv": ["image", "--model", "dev"]},
            # Finished, so only the status keeps it out of the numbers: a
            # render that died after eight minutes is not a timing.
            {"kind": "image", "status": "failed", "started": 0, "finished": 480,
             "argv": ["image", "--model", "fibo"], "output": None},
        ]

    def test_it_reports_measured_seconds_per_model(self):
        seen = media.observed_timings(self._records())
        self.assertEqual(seen["z-image-turbo"]["n"], 2)
        self.assertEqual(seen["z-image-turbo"]["median_s"], 57.0)
        self.assertEqual(seen["krea2"]["median_s"], 520.0)

    def test_same_steps_are_not_the_same_cost(self):
        # The whole point: these two run 8 steps each and are far apart.
        seen = media.observed_timings(self._records())
        self.assertEqual(media.MODELS["krea2"]["steps"] or 8,
                         media.MODELS["z-image-turbo"]["steps"])
        self.assertGreater(seen["krea2"]["median_s"],
                           seen["z-image-turbo"]["median_s"] * 5)

    def test_images_normalise_by_area_and_video_does_not(self):
        # A clip's cost scales with frames too, so seconds-per-megapixel for
        # video would be a fresh wrong signal.
        seen = media.observed_timings(self._records())
        self.assertIn("s_per_mp", seen["krea2"])
        self.assertNotIn("s_per_mp", seen["video:distilled"])

    def test_a_video_stage_keeps_its_own_key(self):
        seen = media.observed_timings(self._records())
        self.assertEqual(seen["video:distilled"]["n"], 1)

    def test_unfinished_jobs_are_not_counted(self):
        self.assertNotIn("dev", media.observed_timings(self._records()))

    def test_a_failed_render_is_not_a_timing(self):
        self.assertNotIn("fibo", media.observed_timings(self._records()))

    def test_no_history_reports_nothing_rather_than_a_guess(self):
        self.assertEqual(media.observed_timings([]), {})

    def test_an_estimate_scales_with_the_canvas(self):
        # Same model at four times the area is not the same wait, and a caller
        # told "57 s" for a poster would be wrong by the area ratio.
        seen = media.observed_timings(self._records())
        small = media._expected_seconds(
            argparse.Namespace(cmd="image", model="krea2",
                               width=1024, height=1024), seen)
        large = media._expected_seconds(
            argparse.Namespace(cmd="image", model="krea2",
                               width=2048, height=2048), seen)
        self.assertAlmostEqual(large / small, 4.0, places=3)

    def test_an_estimate_falls_back_to_the_median_without_a_size(self):
        seen = media.observed_timings(self._records())
        self.assertEqual(
            media._expected_seconds(
                argparse.Namespace(cmd="image", model="krea2",
                                   width=None, height=None), seen),
            seen["krea2"]["median_s"])

    def test_a_model_never_timed_here_gets_no_estimate(self):
        seen = media.observed_timings(self._records())
        self.assertIsNone(media._expected_seconds(
            argparse.Namespace(cmd="image", model="fibo",
                               width=None, height=None), seen))

    def test_timings_is_reachable_on_its_own(self):
        # It exists because publishing the numbers was not enough: a caller
        # asked how long each model takes and went to bash to parse the job
        # files by hand. Whoever reaches for a shell should find this.
        store = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, store, True)
        jobs_dir = os.path.join(store, "media", "jobs")
        os.makedirs(jobs_dir)
        with open(os.path.join(jobs_dir, "1785500000-abcdef.json"), "w") as fh:
            json.dump({"id": "1785500000-abcdef", "kind": "image",
                       "status": "done", "started": 1785500000.0,
                       "finished": 1785500123.0, "output": None,
                       "argv": ["image", "--model", "krea2"]}, fh)
        text = subprocess.check_output(
            [sys.executable, os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "generate.py"), "timings"],
            env=dict(os.environ, FXLLA_STORE=store), text=True)
        self.assertIn("krea2", text)
        self.assertIn("123 s", text)

    def test_timings_says_so_when_nothing_has_been_measured(self):
        store = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, store, True)
        text = subprocess.check_output(
            [sys.executable, os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "generate.py"), "timings"],
            env=dict(os.environ, FXLLA_STORE=store), text=True)
        self.assertIn("Nothing has been timed", text)

    def test_video_estimates_come_from_the_stage(self):
        seen = media.observed_timings(self._records())
        self.assertEqual(
            media._expected_seconds(
                argparse.Namespace(cmd="video", stage="distilled"), seen),
            55.0)

    def test_the_catalog_carries_the_measurements_and_the_warning(self):
        # Against a store with a real job record, so a catalog that reported an
        # empty `observed` would fail here rather than look correct.
        store = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, store, True)
        jobs_dir = os.path.join(store, "media", "jobs")
        os.makedirs(jobs_dir)
        with open(os.path.join(jobs_dir, "1785500000-abcdef.json"), "w") as fh:
            json.dump({"id": "1785500000-abcdef", "kind": "image",
                       "status": "done", "started": 1785500000.0,
                       "finished": 1785500123.0, "output": None,
                       "argv": ["image", "--model", "krea2",
                                "--width", "1024", "--height", "1024"]}, fh)
        payload = json.loads(_models_json_with_store(store))
        self.assertEqual(payload["observed"]["krea2"]["median_s"], 123.0)
        self.assertIn("compare a model only against ITSELF", payload["cost"])


def _models_json_with_store(store):
    """`fxlla media models --json` as a fresh process under this store, since
    the store is read at import time."""
    return subprocess.check_output(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "generate.py"), "models", "--json"],
        env=dict(os.environ, FXLLA_STORE=store), text=True)


class TestJobNotification(unittest.TestCase):
    # A tool call is request/response and an agent turn ends when the model
    # stops calling tools, so nothing in MCP can reopen a finished turn to
    # announce a render. The desktop is the only channel left.
    def setUp(self):
        self.sent = []
        self.real_run = jobs.subprocess.run
        jobs.subprocess.run = lambda cmd, **kw: self.sent.append(cmd)
        self.addCleanup(setattr, jobs.subprocess, "run", self.real_run)
        self.real_platform = jobs.sys.platform
        jobs.sys.platform = "darwin"
        self.addCleanup(setattr, jobs.sys, "platform", self.real_platform)

    def _osascript_only(self):
        """Pin the fallback branch. Without this these assertions depend on
        whether terminal-notifier happens to be installed, which made them pass
        here and fail in CI - or the reverse."""
        saved = jobs.shutil.which
        jobs.shutil.which = lambda _n: None
        self.addCleanup(setattr, jobs.shutil, "which", saved)

    def _rec(self, **over):
        rec = {"id": "1785500000-abcdef", "kind": "image", "status": "done",
               "output": "/out/fxlla-krea2-1785500000.png", "summary": "s",
               "started": 1785500000.0, "finished": 1785500123.0}
        rec.update(over)
        return rec

    def test_a_finished_render_says_what_and_how_long(self):
        self._osascript_only()
        jobs.notify(self._rec())
        script = self.sent[0][-1]
        self.assertIn("fxlla-krea2-1785500000.png", script)
        self.assertIn("123s", script)
        self.assertIn("ready", script)

    def test_a_failure_is_announced_too(self):
        # Eight minutes of waiting for nothing is what most needs saying, and
        # silence reads as still running.
        self._osascript_only()
        jobs.notify(self._rec(status="failed", output=None))
        self.assertIn("failed", self.sent[0][-1])

    def test_quotes_in_a_prompt_cannot_escape_the_script(self):
        self._osascript_only()
        jobs.notify(self._rec(status="failed", output=None,
                              summary='a "quoted" \\ prompt'))
        script = self.sent[0][-1]
        self.assertIn('\\"quoted\\"', script)
        # One unescaped quote would close the literal and leave the rest of
        # somebody's prompt to be interpreted as AppleScript.
        self.assertEqual(script.count('"') % 2, 0)

    def test_a_click_lands_on_the_file_when_it_can(self):
        # An osascript notification belongs to Script Editor, so clicking one
        # opened an app the user never invoked and which has nothing to do with
        # the render. terminal-notifier can attach the finished file instead.
        saved = jobs.shutil.which
        jobs.shutil.which = lambda n: "/usr/local/bin/" + n
        self.addCleanup(setattr, jobs.shutil, "which", saved)
        with tempfile.NamedTemporaryFile(suffix=".png") as fh:
            jobs.notify(self._rec(output=fh.name))
        cmd = self.sent[0]
        self.assertEqual(cmd[0], "terminal-notifier")
        self.assertIn("-execute", cmd)
        self.assertIn(fh.name, cmd[cmd.index("-execute") + 1])

    def test_without_it_a_click_at_least_opens_finder(self):
        saved = jobs.shutil.which
        jobs.shutil.which = lambda _n: None
        self.addCleanup(setattr, jobs.shutil, "which", saved)
        jobs.notify(self._rec())
        script = self.sent[0][-1]
        self.assertIn('application "Finder"', script)

    def test_a_failure_offers_no_file_to_open(self):
        saved = jobs.shutil.which
        jobs.shutil.which = lambda n: "/usr/local/bin/" + n
        self.addCleanup(setattr, jobs.shutil, "which", saved)
        jobs.notify(self._rec(status="failed", output=None))
        self.assertNotIn("-execute", self.sent[0])

    def test_it_stays_quiet_where_there_is_no_desktop(self):
        jobs.sys.platform = "linux"
        jobs.notify(self._rec())
        self.assertEqual(self.sent, [])

    def test_it_can_be_turned_off(self):
        saved = os.environ.get("FXLLA_MEDIA_NOTIFY")
        os.environ["FXLLA_MEDIA_NOTIFY"] = "0"
        self.addCleanup(lambda: os.environ.__setitem__("FXLLA_MEDIA_NOTIFY", saved)
                        if saved is not None
                        else os.environ.pop("FXLLA_MEDIA_NOTIFY", None))
        jobs.notify(self._rec())
        self.assertEqual(self.sent, [])

    def test_a_missing_record_is_not_a_crash(self):
        jobs.notify(None)
        self.assertEqual(self.sent, [])

    def test_the_worker_announces_both_outcomes(self):
        # The notify() calls have to be wired into run(), not merely exist.
        class Proc(object):
            def __init__(self, rc, out):
                self.returncode, self.stdout, self.stderr = rc, out, "boom"

        for rc, stdout, expected in ((0, "/out/x.png\n", "done"),
                                     (1, "", "failed")):
            jobs_dir = tempfile.mkdtemp()
            self.addCleanup(shutil.rmtree, jobs_dir, True)
            saved_dir, jobs.JOBS_DIR = jobs.JOBS_DIR, jobs_dir
            self.addCleanup(setattr, jobs, "JOBS_DIR", saved_dir)
            announced = []
            saved_notify, jobs.notify = jobs.notify, announced.append
            self.addCleanup(setattr, jobs, "notify", saved_notify)
            jobs.subprocess.run = lambda cmd, **kw: Proc(rc, stdout)
            rec = jobs._write({"id": "1785500000-abcdef", "kind": "image",
                               "status": "queued", "argv": ["image", "x"],
                               "summary": "x", "output": None, "error": None,
                               "pid": None, "created": 1785500000.0,
                               "started": None, "finished": None, "log": ""})
            jobs.run("1785500000-abcdef")
            self.assertEqual([r["status"] for r in announced], [expected])
            jobs.notify = saved_notify
            jobs.JOBS_DIR = saved_dir


class TestLoraNameResolution(unittest.TestCase):
    # list_loras reports a path AND a name; a caller that passes the name back
    # was told "LoRA not found" by the same tool that had just printed it.
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        nested = os.path.join(self.dir, "sub")
        os.makedirs(nested)
        for path in (os.path.join(self.dir, "style.safetensors"),
                     os.path.join(nested, "deep.safetensors")):
            open(path, "wb").close()
        saved = os.environ.get("FXLLA_LORA_DIRS")
        os.environ["FXLLA_LORA_DIRS"] = self.dir
        self.addCleanup(lambda: os.environ.__setitem__("FXLLA_LORA_DIRS", saved)
                        if saved is not None
                        else os.environ.pop("FXLLA_LORA_DIRS", None))

    def _lora_arg(self, ref):
        cmd = media.build_command(media.MODELS["krea2"], "p", "/o.png",
                                  loras=[ref], model_name="krea2")
        return cmd[cmd.index("--lora") + 1:cmd.index("--lora") + 3]

    def test_a_bare_name_resolves_to_its_path(self):
        self.assertEqual(self._lora_arg("style.safetensors"),
                         [os.path.join(self.dir, "style.safetensors")])

    def test_the_scale_survives_resolution(self):
        self.assertEqual(self._lora_arg("deep.safetensors,0.7"),
                         [os.path.join(self.dir, "sub", "deep.safetensors"), "0.7"])

    def test_an_unknown_name_names_where_it_looked(self):
        with self.assertRaises(ValueError) as ctx:
            self._lora_arg("nope.safetensors")
        self.assertIn(self.dir, str(ctx.exception))

    def test_a_spelled_out_path_is_not_guessed_at(self):
        # Resolving one would hide a typo in a path the caller chose - and a
        # wrong path is not a name lookup, so the answer must not read as one.
        wrong = os.path.join(self.dir, "wrong", "style.safetensors")
        with self.assertRaises(ValueError) as ctx:
            self._lora_arg(wrong)
        self.assertIn(wrong, str(ctx.exception))
        self.assertNotIn("searched", str(ctx.exception))

    def test_an_ambiguous_name_names_the_candidates(self):
        twin = os.path.join(self.dir, "sub", "style.safetensors")
        open(twin, "wb").close()
        with self.assertRaises(ValueError) as ctx:
            self._lora_arg("style.safetensors")
        self.assertIn("ambiguous", str(ctx.exception))
        self.assertIn(twin, str(ctx.exception))


class TestMfluxBinDir(unittest.TestCase):
    # The image knob: one directory for the whole mflux family, since image
    # alone is eight per-model CLIs and a single-binary knob cannot cover it.
    def test_unset_resolves_from_path(self):
        saved = media.MFLUX_BIN_DIR
        media.MFLUX_BIN_DIR = ""
        try:
            self.assertEqual(media.mflux_cli("mflux-generate"), "mflux-generate")
        finally:
            media.MFLUX_BIN_DIR = saved

    def test_set_resolves_inside_the_directory(self):
        import tempfile
        d = tempfile.mkdtemp()
        open(os.path.join(d, "mflux-generate"), "w").close()
        saved = media.MFLUX_BIN_DIR
        media.MFLUX_BIN_DIR = d
        try:
            self.assertEqual(media.mflux_cli("mflux-generate"),
                             os.path.join(d, "mflux-generate"))
        finally:
            media.MFLUX_BIN_DIR = saved

    def test_set_with_missing_binary_errors_instead_of_falling_back(self):
        # Silently falling back to PATH would run a different install than
        # the one the user pointed at - the quiet failure the knob exists to
        # prevent.
        import tempfile
        d = tempfile.mkdtemp()
        saved = media.MFLUX_BIN_DIR
        media.MFLUX_BIN_DIR = d
        try:
            with self.assertRaises(ValueError) as ctx:
                media.mflux_cli("mflux-generate-boogu")
            self.assertIn(d, str(ctx.exception))
            self.assertIn("mflux-generate-boogu", str(ctx.exception))
        finally:
            media.MFLUX_BIN_DIR = saved

    def test_image_edit_and_upscale_all_honor_the_directory(self):
        import tempfile
        d = tempfile.mkdtemp()
        for name in ("mflux-generate-z-image-turbo", "mflux-generate-qwen-edit",
                     "mflux-upscale-seedvr2"):
            open(os.path.join(d, name), "w").close()
        saved = media.MFLUX_BIN_DIR
        media.MFLUX_BIN_DIR = d
        try:
            img = media.build_command(media.MODELS["z-image-turbo"], "a cat", "/o.png")
            self.assertEqual(img[0], os.path.join(d, "mflux-generate-z-image-turbo"))
            edit = media.build_edit_command("p", "/i.png", "/o.png")
            self.assertEqual(edit[0], os.path.join(d, "mflux-generate-qwen-edit"))
            up = media.build_upscale_command("/i.png", "/o.png")
            self.assertEqual(up[0], os.path.join(d, "mflux-upscale-seedvr2"))
        finally:
            media.MFLUX_BIN_DIR = saved

    def test_specific_edit_knob_still_wins(self):
        saved_dir, saved_edit = media.MFLUX_BIN_DIR, media.EDIT_BIN
        media.MFLUX_BIN_DIR = "/nonexistent"
        media.EDIT_BIN = "/my/qwen-edit"
        try:
            cmd = media.build_edit_command("p", "/i.png", "/o.png")
            self.assertEqual(cmd[0], "/my/qwen-edit")
        finally:
            media.MFLUX_BIN_DIR, media.EDIT_BIN = saved_dir, saved_edit


class TestVoicePythonResolution(unittest.TestCase):
    # One resolution, three layers: the env override always wins, the uv tool
    # venv (installed by `fxlla setup --media`) is the default when present,
    # bare python3 keeps hand-rolled PATH venvs working.
    def test_env_override_wins(self):
        os.environ["FXLLA_VOICE_PYTHON"] = "/my/venv/bin/python"
        try:
            self.assertEqual(media.resolved_voice_python(), "/my/venv/bin/python")
        finally:
            del os.environ["FXLLA_VOICE_PYTHON"]

    def test_uv_tool_interpreter_when_present(self):
        saved_run, saved_isfile = media.subprocess.run, media.os.path.isfile

        class _Out:
            stdout = "/tools\n"
        media.subprocess.run = lambda *a, **k: _Out()
        media.os.path.isfile = lambda p: p == "/tools/mlx-audio/bin/python"
        try:
            self.assertEqual(media.resolved_voice_python(),
                             "/tools/mlx-audio/bin/python")
        finally:
            media.subprocess.run, media.os.path.isfile = saved_run, saved_isfile

    def test_falls_back_to_python3(self):
        saved = media.subprocess.run

        def _boom(*_a, **_k):
            raise OSError("no uv")
        media.subprocess.run = _boom
        try:
            self.assertEqual(media.resolved_voice_python(), "python3")
        finally:
            media.subprocess.run = saved


class TestBuildVoiceCommand(unittest.TestCase):
    def test_requires_reference(self):
        with self.assertRaises(ValueError):
            media.build_voice_command("hola", "/o.wav", ref="")

    def test_builds_backend_invocation(self):
        cmd = media.build_voice_command("hola", "/o.wav", ref="/r.wav",
                                        model="repo/x", lang="es", speed=1.1,
                                        python="/venv/bin/python", backend="/b.py")
        self.assertEqual(cmd[0], "/venv/bin/python")
        self.assertEqual(cmd[1], "/b.py")
        self.assertEqual(cmd[cmd.index("--text") + 1], "hola")
        self.assertEqual(cmd[cmd.index("--ref") + 1], "/r.wav")
        self.assertEqual(cmd[cmd.index("--lang") + 1], "es")
        self.assertEqual(cmd[cmd.index("--model") + 1], "repo/x")
        self.assertEqual(cmd[cmd.index("--speed") + 1], "1.1")

    def test_empty_text_raises(self):
        with self.assertRaises(ValueError):
            media.generate_speech("", ref="/r.wav")


class TestValidateWavOutput(unittest.TestCase):
    def test_missing(self):
        with self.assertRaises(RuntimeError):
            media.validate_wav_output("/no/such.wav")

    def test_not_wav(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"\x00" * 4096)
            path = f.name
        try:
            with self.assertRaises(RuntimeError):
                media.validate_wav_output(path)
        finally:
            os.unlink(path)

    def test_valid_wav_passes(self):
        import array, math, wave
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ok.wav")
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(24000)
                # A real tone. The previous fixture was a constant sample value,
                # which is a DC level carrying no signal, and the content checks
                # correctly call that silence.
                tone = array.array("h", (int(9000 * math.sin(2 * math.pi * 180 * t / 24000))
                                         for t in range(4096)))
                w.writeframes(tone.tobytes())
            media.validate_wav_output(path)  # must not raise


class _StopHere(Exception):
    """Sentinel: stop a generator once it has revealed what we are checking."""


class TestSeveralCacheRoots(unittest.TestCase):
    """FXLLA_MEDIA_HF_HOME may name several caches, ':' separated.

    Weights outgrow a disk. The first root takes new downloads; every root is
    searched for what is already there. The part that is easy to get wrong is
    the render: HF_HOME accepts ONE path, so pointing a job at the wrong root
    does not fail loudly - the toolchain decides the weights are missing and
    downloads tens of gigabytes again.
    """

    def setUp(self):
        import weights as w
        self.w = w
        self.saved = os.environ.get("FXLLA_MEDIA_HF_HOME")
        # Spaces on purpose: the real cache here is "/Volumes/verga - Data/...".
        self.a = os.path.join(tempfile.mkdtemp(), "first cache")
        self.b = os.path.join(tempfile.mkdtemp(), "second cache")
        os.makedirs(self.a)
        os.makedirs(self.b)

    def tearDown(self):
        if self.saved is None:
            os.environ.pop("FXLLA_MEDIA_HF_HOME", None)
        else:
            os.environ["FXLLA_MEDIA_HF_HOME"] = self.saved
        for d in (self.a, self.b):
            shutil.rmtree(os.path.dirname(d), ignore_errors=True)

    def _place(self, root, repo, size=2 * 1024 * 1024):
        snap = os.path.join(root, "hub", "models--" + repo.replace("/", "--"),
                            "snapshots", "abc")
        os.makedirs(snap, exist_ok=True)
        with open(os.path.join(snap, "model.safetensors"), "wb") as fh:
            fh.write(b"\0" * size)

    def test_roots_split_on_colon_and_keep_spaces(self):
        os.environ["FXLLA_MEDIA_HF_HOME"] = self.a + ":" + self.b
        self.assertEqual(self.w.cache_roots(), [self.a, self.b])
        self.assertEqual(self.w.write_root(), self.a)

    def test_unset_is_the_single_default(self):
        os.environ.pop("FXLLA_MEDIA_HF_HOME", None)
        self.assertEqual(self.w.cache_roots(),
                         [os.path.expanduser("~/.cache/huggingface")])

    def test_a_repo_in_a_later_root_is_found_and_named(self):
        self._place(self.b, "someorg/Far-Model")
        os.environ["FXLLA_MEDIA_HF_HOME"] = self.a + ":" + self.b
        self.assertTrue(self.w._cached("someorg/Far-Model"))
        self.assertEqual(self.w.repo_root("someorg/Far-Model"), self.b)
        self.assertIsNone(self.w.repo_root("someorg/Nowhere"))

    def test_metadata_only_is_not_cached_in_any_root(self):
        snap = os.path.join(self.b, "hub", "models--someorg--Thin",
                            "snapshots", "abc")
        os.makedirs(snap)
        with open(os.path.join(snap, "config.json"), "w") as fh:
            fh.write("{}")
        os.environ["FXLLA_MEDIA_HF_HOME"] = self.a + ":" + self.b
        self.assertFalse(self.w._cached("someorg/Thin"))

    def test_render_is_pointed_at_the_root_that_has_the_weights(self):
        # 'voice' resolves through FIXED_ALIAS to the chatterbox row, so this
        # uses whatever repos the shipped catalog names for it - no fixture
        # catalog that could drift away from the real one.
        repos = self.w._repos_for("voice")
        self.assertTrue(repos, "the catalog has no repos for voice")
        for repo in repos:
            self._place(self.b, repo)
        os.environ["FXLLA_MEDIA_HF_HOME"] = self.a + ":" + self.b
        root, unreachable = self.w.hf_home_for("voice")
        # The write root is self.a; the weights are in self.b. Handing back the
        # write root here is the silent re-download this exists to prevent.
        self.assertEqual(root, self.b)
        self.assertEqual(unreachable, [])

    def test_weights_nowhere_yet_resolve_to_the_write_root(self):
        os.environ["FXLLA_MEDIA_HF_HOME"] = self.a + ":" + self.b
        root, unreachable = self.w.hf_home_for("voice")
        self.assertEqual(root, self.a)
        self.assertEqual(unreachable, [],
                         "not downloaded yet is not the same as unreachable")

    def test_env_never_forwards_the_list_as_a_path(self):
        os.environ["FXLLA_MEDIA_HF_HOME"] = self.a + ":" + self.b
        env = media._env("voice")
        self.assertIn(env["HF_HOME"], (self.a, self.b))
        self.assertNotIn(":", env["HF_HOME"].replace(self.a, "").replace(self.b, ""))

    def test_the_big_model_beats_a_pile_of_small_extras(self):
        """Ranking roots by repo COUNT sent renders to the wrong disk.

        With --pid-decode the two extras (a decoder and a caption encoder) are
        two repos; the base model is one. Counting repos made the root holding
        the extras win, and the base model - the big one, the whole reason not
        to re-download - got fetched again.
        """
        base = self.w._repos_for("image", "z-image-turbo")
        extras = self.w._repos_for(None, None, ("pid",))
        self.assertTrue(base, "no repos for z-image-turbo in the catalog")
        self.assertGreater(len(extras), len(base),
                           "fixture assumes the extras outnumber the base repo")
        for repo in extras:                       # many, small
            self._place(self.a, repo, 2 * 1024 * 1024)
        for repo in base:                         # one, large
            self._place(self.b, repo, 24 * 1024 * 1024)
        os.environ["FXLLA_MEDIA_HF_HOME"] = self.a + ":" + self.b
        root, unreachable = self.w.hf_home_for("image", "z-image-turbo", ("pid",))
        self.assertEqual(root, self.b, "picked the root with more repos, not more bytes")
        self.assertEqual(unreachable, extras)

    def test_a_lora_named_as_a_repo_id_is_counted(self):
        """A --lora repo id is a real dependency that no catalog row names."""
        base = self.w._repos_for("image", "z-image-turbo")
        for repo in base:
            self._place(self.a, repo, 2 * 1024 * 1024)
        self._place(self.b, "someorg/CoolLora", 24 * 1024 * 1024)
        os.environ["FXLLA_MEDIA_HF_HOME"] = self.a + ":" + self.b
        # Blind to the LoRA, this picks self.a and reports nothing wrong.
        root, unreachable = self.w.hf_home_for(
            "image", "z-image-turbo", (), ["someorg/CoolLora"])
        self.assertEqual(root, self.b)
        self.assertEqual(unreachable, base)

    def test_generate_image_actually_forwards_the_lora_repo_ids(self):
        """The wiring, not just the helper.

        The first version of this test called hf_home_for() directly with the
        ids and passed even with generate_image's argument deleted - it proved
        the function worked and nothing about whether anyone called it. So
        drive generate_image and record what _env was really given.
        """
        seen = {}

        def fake_env(kind=None, model=None, extras=(), repos=()):
            seen["kind"] = kind
            seen["model"] = model
            seen["extras"] = tuple(extras)
            seen["repos"] = list(repos)
            raise _StopHere()

        # build_command validates a local LoRA path, so it has to be a real
        # file - the point of including one is that it must NOT be treated as
        # a repo id.
        local = os.path.join(self.a, "local.safetensors")
        with open(local, "wb") as fh:
            fh.write(b"\0" * 1024)

        real_env = media._env
        media._env = fake_env
        try:
            with self.assertRaises(_StopHere):
                media.generate_image(
                    "a cat", model="z-image-turbo",
                    loras=["someorg/CoolLora,0.8", local],
                    output=os.path.join(self.a, "out.png"))
        finally:
            media._env = real_env

        self.assertEqual(seen["kind"], "image")
        self.assertEqual(seen["model"], "z-image-turbo")
        self.assertEqual(seen["repos"], ["someorg/CoolLora"],
                         "the repo-id LoRA must reach the cache-root choice, "
                         "and the local file must not")

    def test_lora_repo_ids_are_told_apart_from_paths(self):
        self.assertEqual(media._lora_repo_ids(["someorg/CoolLora"]),
                         ["someorg/CoolLora"])
        # The scale rides on a COMMA, not a space - argparse nargs="+" would
        # otherwise swallow the prompt. See split_ref.
        self.assertEqual(media._lora_repo_ids(["someorg/CoolLora,0.8"]),
                         ["someorg/CoolLora"])
        self.assertEqual(media._lora_repo_ids([["someorg/CoolLora", "0.8"]]),
                         ["someorg/CoolLora"])
        for path in ["/abs/x.safetensors", "./rel/x.safetensors",
                     "~/x.safetensors", "plain.safetensors", "bare-name"]:
            self.assertEqual(media._lora_repo_ids([path]), [],
                             "%s is a file, not a repo id" % path)

    def test_bytes_are_not_double_counted_through_snapshot_symlinks(self):
        """A HF cache keeps bytes in blobs/ and links to them from snapshots/.

        Following both would count every file twice and could flip the choice
        of root, so the size walk skips links.
        """
        repo = os.path.join(self.a, "hub", "models--org--Linked")
        blobs = os.path.join(repo, "blobs")
        snaps = os.path.join(repo, "snapshots", "abc")
        os.makedirs(blobs)
        os.makedirs(snaps)
        blob = os.path.join(blobs, "deadbeef")
        with open(blob, "wb") as fh:
            fh.write(b"\0" * (4 * 1024 * 1024))
        os.symlink(blob, os.path.join(snaps, "model.safetensors"))
        self.assertEqual(self.w._weight_bytes(repo), 4 * 1024 * 1024)

    def test_loras_are_collected_from_every_root_once(self):
        def write(root, name, keys):
            snap = os.path.join(root, "hub", name, "snapshots", "abc")
            os.makedirs(snap, exist_ok=True)
            hdr = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
                   for k in keys}
            blob = json.dumps(hdr).encode()
            with open(os.path.join(snap, "w.safetensors"), "wb") as fh:
                fh.write(struct.pack("<Q", len(blob)) + blob + b"\x00\x00")

        write(self.a, "models--org--LoRA-A", ["b.0.lora_A.weight"])
        write(self.b, "models--org--LoRA-B", ["b.0.lora_A.weight"])
        # The same repo in BOTH roots must be offered once, not twice.
        write(self.a, "models--org--LoRA-Dup", ["b.0.lora_A.weight"])
        write(self.b, "models--org--LoRA-Dup", ["b.0.lora_A.weight"])
        os.environ["FXLLA_MEDIA_HF_HOME"] = self.a + ":" + self.b
        names = [r[0] for r in media._hf_cache_loras()]
        self.assertIn("org/LoRA-A", names)
        self.assertIn("org/LoRA-B", names)
        self.assertEqual(names.count("org/LoRA-Dup"), 1)


if __name__ == "__main__":
    unittest.main()
