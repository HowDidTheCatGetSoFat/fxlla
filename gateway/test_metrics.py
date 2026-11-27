import importlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
metrics = importlib.import_module("metrics")

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _sse(*objs):
    return "".join("data: " + json.dumps(o) + "\n\n" for o in objs).encode()


class TestPathFilter(unittest.TestCase):
    def test_completion_paths(self):
        self.assertTrue(metrics.is_completion_path("/v1/chat/completions"))
        self.assertTrue(metrics.is_completion_path("/v1/completions"))
        self.assertTrue(metrics.is_completion_path("/v1/chat/completions/"))
        self.assertTrue(metrics.is_completion_path("/v1/chat/completions?x=1"))

    def test_non_completion_paths(self):
        self.assertFalse(metrics.is_completion_path("/v1/embeddings"))
        self.assertFalse(metrics.is_completion_path("/v1/models"))
        self.assertFalse(metrics.is_completion_path("/health"))


class TestStreamMetrics(unittest.TestCase):
    def test_chat_deltas_counted(self):
        m = metrics.StreamMetrics(start=0.0)
        m.feed(_sse(
            {"choices": [{"delta": {"role": "assistant"}}]},   # role only, no text
            {"choices": [{"delta": {"content": "he"}}]},
            {"choices": [{"delta": {"content": "llo"}}]},
        ))
        m.feed(b"data: [DONE]\n\n")
        # feed() stamps self.first from a real monotonic clock, so measure the
        # window relative to it rather than an arbitrary end.
        ttft_ms, tps, tokens = m.result(end=m.first + 1.0)
        self.assertEqual(tokens, 2)
        self.assertIsNotNone(ttft_ms)
        self.assertEqual(tps, 2.0)

    def test_thinking_starts_the_clock_llama_cpp_shape(self):
        """A reasoning model's thinking is generated tokens too.

        completion_tokens counts it, so a clock started at the first VISIBLE
        token divides every token by only the time the answer took. Measured on
        Gemma 4 26B-A4B through llama.cpp: 563 reasoning deltas before 165
        content ones, reported as 518 tok/s from a model doing 114.
        """
        m = metrics.StreamMetrics(start=0.0)
        m.feed(_sse({"choices": [{"delta": {"reasoning_content": "thin"}}]}))
        first_thought = m.first
        self.assertIsNotNone(first_thought, "thinking did not start the clock")
        m.feed(_sse({"choices": [{"delta": {"content": "answer"}}]}))
        self.assertEqual(m.first, first_thought,
                         "the clock restarted at the visible token")
        _ttft, tps, tokens = m.result(end=m.first + 2.0)
        self.assertEqual(tokens, 2)
        self.assertEqual(tps, 1.0)

    def test_thinking_starts_the_clock_mlx_shape(self):
        """Same bug, other spelling: mlx_lm streams `reasoning`, not
        `reasoning_content`. Reading only one of the two names fixes one engine
        and leaves the other reporting 550 tok/s from a model doing 98."""
        m = metrics.StreamMetrics(start=0.0)
        m.feed(_sse({"choices": [{"delta": {"role": "assistant",
                                            "reasoning": "thin"}}]}))
        self.assertIsNotNone(m.first, "mlx_lm thinking did not start the clock")
        m.feed(_sse({"choices": [{"delta": {"content": "answer"}}]}))
        _ttft, tps, tokens = m.result(end=m.first + 2.0)
        self.assertEqual(tokens, 2)
        self.assertEqual(tps, 1.0)

    def test_a_tool_call_starts_the_clock_too(self):
        """A model can emit a whole tool call before any prose. Those are
        generated tokens the server counts, so ignoring them repeats the
        reasoning bug through a different field."""
        m = metrics.StreamMetrics(start=0.0)
        m.feed(_sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "get_weather",
                                      "arguments": '{"city":'}}]}}]}))
        self.assertIsNotNone(m.first, "a tool call did not start the clock")
        m.feed(_sse({"choices": [{"delta": {"content": "done"}}]}))
        _ttft, tps, tokens = m.result(end=m.first + 2.0)
        self.assertEqual(tokens, 2)
        self.assertEqual(tps, 1.0)

    def test_a_delta_with_no_generated_text_still_does_not_count(self):
        """The widening must not swallow role-only or empty chunks - those
        carry no tokens and would start the clock early, which is the same
        error pointed the other way."""
        m = metrics.StreamMetrics(start=0.0)
        m.feed(_sse(
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": ""}}]},
            {"choices": [{"delta": {"reasoning": ""}}]},
            {"choices": [{"delta": {}}]},
        ))
        self.assertIsNone(m.first)
        self.assertEqual(m.deltas, 0)

    def test_legacy_completion_text(self):
        m = metrics.StreamMetrics(start=0.0)
        m.feed(_sse({"choices": [{"text": "a"}]}, {"choices": [{"text": "b"}]}))
        _ttft, _tps, tokens = m.result(end=1.0)
        self.assertEqual(tokens, 2)

    def test_usage_overrides_delta_count(self):
        m = metrics.StreamMetrics(start=0.0)
        m.feed(_sse(
            {"choices": [{"delta": {"content": "x"}}]},
            {"choices": [{"delta": {"content": "y"}}]},
            {"choices": [], "usage": {"completion_tokens": 42}},
        ))
        _ttft, _tps, tokens = m.result(end=1.0)
        self.assertEqual(tokens, 42)

    def test_first_token_time_from_start(self):
        m = metrics.StreamMetrics(start=10.0)
        m.first = 10.5
        m.deltas = 5
        ttft_ms, tps, _tokens = m.result(end=11.0)
        self.assertEqual(ttft_ms, 500)
        self.assertEqual(tps, 10.0)  # 5 tokens over 0.5 s

    def test_no_tokens_yields_none(self):
        m = metrics.StreamMetrics(start=0.0)
        m.feed(_sse({"choices": [{"delta": {"role": "assistant"}}]}))
        ttft_ms, tps, tokens = m.result(end=1.0)
        self.assertEqual(tokens, 0)
        self.assertIsNone(ttft_ms)
        self.assertIsNone(tps)

    def test_split_across_chunks(self):
        m = metrics.StreamMetrics(start=0.0)
        payload = _sse({"choices": [{"delta": {"content": "hi"}}]})
        m.feed(payload[:7])
        m.feed(payload[7:])
        _ttft, _tps, tokens = m.result(end=1.0)
        self.assertEqual(tokens, 1)

    def test_malformed_json_ignored(self):
        m = metrics.StreamMetrics(start=0.0)
        m.feed(b"data: {not json}\n\n")
        m.feed(b": comment line\n\n")
        _ttft, _tps, tokens = m.result(end=1.0)
        self.assertEqual(tokens, 0)

    def test_realistic_chat_stream(self):
        # A response shaped like a real OpenAI-compatible chat stream: an opening
        # role-only chunk, content deltas, a finish_reason chunk, then [DONE].
        raw = (
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant"},'
            b'"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":"Hello"},'
            b'"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":" world"},'
            b'"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            b'data: [DONE]\n\n'
        )
        m = metrics.StreamMetrics(start=0.0)
        # deliver in odd-sized slices to exercise buffering across chunk edges
        for i in range(0, len(raw), 13):
            m.feed(raw[i:i + 13])
        _ttft, _tps, tokens = m.result(end=m.first + 1.0)
        self.assertEqual(tokens, 2)  # "Hello" and " world"; role/finish carry no text


class TestUsageFromJson(unittest.TestCase):
    def test_reads_completion_tokens(self):
        body = json.dumps({"usage": {"completion_tokens": 17}}).encode()
        self.assertEqual(metrics.usage_from_json(body), 17)

    def test_missing_usage(self):
        self.assertIsNone(metrics.usage_from_json(b'{"choices": []}'))

    def test_garbage(self):
        self.assertIsNone(metrics.usage_from_json(b"not json"))


class TestStatsFile(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k)
                       for k in ("FXLLA_STATS_FILE", "XDG_STATE_HOME")}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_explicit_env_wins(self):
        os.environ["FXLLA_STATS_FILE"] = "/custom/s.jsonl"
        os.environ["XDG_STATE_HOME"] = "/state"
        self.assertEqual(metrics.stats_file(), "/custom/s.jsonl")

    def test_xdg_state_home(self):
        os.environ["XDG_STATE_HOME"] = "/state"
        self.assertEqual(metrics.stats_file(), "/state/fxlla/stats.jsonl")

    def test_default_under_home(self):
        self.assertTrue(metrics.stats_file().endswith(
            os.path.join(".local", "state", "fxlla", "stats.jsonl")))


class TestAppendSample(unittest.TestCase):
    def test_append_and_trim(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "stats.jsonl")
            for i in range(50):
                metrics.append_sample(
                    path, metrics.build_sample(i, "m", "mlx", 100, 10, 5.0), cap=10)
            with open(path, encoding="utf-8") as f:
                lines = f.read().splitlines()
            self.assertLessEqual(len(lines), 10)
            last = json.loads(lines[-1])
            self.assertEqual(last["ts"], 49)
            self.assertEqual(last["source"], "gateway")
            self.assertEqual(last["model"], "m")

    def test_sample_shape(self):
        s = metrics.build_sample(1234, "coder", "gguf", 2048.7, None, None)
        self.assertEqual(s["ts"], 1234)
        self.assertEqual(s["ram_mb"], 2048)
        self.assertIsNone(s["ttft_ms"])
        self.assertIsNone(s["tps"])



class TestUnmeasurableRate(unittest.TestCase):
    """A rate the transport invented rather than the model produced.

    Measured on a real eval run: tok_s came back with a max of 1,865,386 and a
    median pulled up with it, because the response arrived buffered and every
    delta was parsed microseconds apart.
    """

    def _stream(self, n_deltas):
        sm = metrics.StreamMetrics(0.0)
        for i in range(n_deltas):
            sm.feed(b'data: {"choices":[{"delta":{"content":"x"}}]}\n')
        return sm

    def test_a_buffered_stream_reports_no_rate(self):
        sm = self._stream(200)
        # end essentially equal to the first-token stamp: the whole stream
        # landed in one read, so nothing about decode speed was observed.
        _ttft, tps, tokens = sm.result(sm.first + 0.0001)
        self.assertEqual(tokens, 200)
        self.assertIsNone(tps, "2,000,000 tok/s is the transport, not the model")

    def test_a_real_rate_still_comes_through(self):
        sm = self._stream(200)
        _ttft, tps, _ = sm.result(sm.first + 4.0)   # 50 tok/s
        self.assertEqual(tps, 50.0)

    def test_a_fast_but_plausible_rate_is_kept(self):
        # The bound must not throw away a genuinely quick small model.
        sm = self._stream(600)
        _ttft, tps, _ = sm.result(sm.first + 1.0)
        self.assertEqual(tps, 600.0)

    def test_one_delta_times_nothing(self):
        sm = self._stream(1)
        _ttft, tps, _ = sm.result(sm.first + 2.0)
        self.assertIsNone(tps, "a rate is measured between arrivals")

    def test_the_token_count_survives_either_way(self):
        # Losing the rate must not lose the count: tokens_spent is a separate
        # and still-correct fact.
        sm = self._stream(120)
        _ttft, tps, tokens = sm.result(sm.first + 0.00001)
        self.assertIsNone(tps)
        self.assertEqual(tokens, 120)


class TestEngineReport(unittest.TestCase):
    def test_llama_server_stream(self):
        r = metrics.engine_report(
            {"completion_tokens": 200, "prompt_tokens": 12, "total_tokens": 212,
             "prompt_tokens_details": {"cached_tokens": 0}},
            {"cache_n": 0, "prompt_n": 12, "prompt_ms": 45.795,
             "predicted_n": 200, "predicted_ms": 1860.769})
        self.assertEqual((r["prompt_tokens"], r["cached_tokens"], r["output_tokens"]), (12, 0, 200))
        self.assertEqual((r["prompt_ms"], r["decode_ms"]), (46, 1861))

    def test_llama_server_cache_counts_from_timings(self):
        r = metrics.engine_report(
            {"completion_tokens": 161, "prompt_tokens": 12,
             "prompt_tokens_details": {"cached_tokens": 8}},
            {"cache_n": 8, "prompt_n": 4, "prompt_ms": 16.361,
             "predicted_n": 161, "predicted_ms": 1482.787})
        self.assertEqual((r["prompt_tokens"], r["cached_tokens"]), (12, 8))

    def test_mlx_has_tokens_but_no_engine_timings(self):
        r = metrics.engine_report(
            {"prompt_tokens": 31, "completion_tokens": 2, "total_tokens": 33,
             "prompt_tokens_details": {"cached_tokens": 30}}, None)
        self.assertEqual((r["prompt_tokens"], r["cached_tokens"], r["output_tokens"]), (31, 30, 2))
        self.assertIsNone(r["prompt_ms"])
        self.assertIsNone(r["decode_ms"])

    def test_omlx_timings_live_inside_usage_in_seconds(self):
        r = metrics.engine_report(
            {"prompt_tokens": 22, "completion_tokens": 300,
             "prompt_tokens_details": {"cached_tokens": 0},
             "completion_tokens_details": {"reasoning_tokens": 300},
             "prompt_eval_duration": 4.05, "generation_duration": 7.03}, None)
        self.assertEqual((r["prompt_ms"], r["decode_ms"], r["reasoning_tokens"]), (4050, 7030, 300))

    def test_mtplx_reports_draft_acceptance(self):
        r = metrics.engine_report(
            {"prompt_tokens": 22, "completion_tokens": 108,
             "completion_tokens_details": {"reasoning_tokens": 24}},
            {"prompt_n": 22, "predicted_n": 108, "prompt_ms": 631.235,
             "predicted_ms": 2664.547, "draft_n": 84, "draft_n_accepted": 79})
        self.assertEqual((r["draft_tokens"], r["draft_accepted"]), (84, 79))
        self.assertEqual((r["prompt_ms"], r["decode_ms"]), (631, 2665))

    def test_nothing_reported_is_all_none(self):
        self.assertTrue(all(v is None for v in metrics.engine_report(None, None).values()))

    def test_garbage_types_are_none_not_an_exception(self):
        r = metrics.engine_report({"prompt_tokens": "12", "completion_tokens": None}, {"prompt_ms": "x"})
        self.assertIsNone(r["prompt_tokens"])
        self.assertIsNone(r["prompt_ms"])

    def test_non_finite_numbers_are_none_not_an_exception(self):
        r = metrics.engine_report(
            {"generation_duration": float("inf"), "prompt_eval_duration": 1e308},
            {"prompt_ms": float("nan"), "predicted_ms": float("-inf")})
        self.assertIsNone(r["prompt_ms"])
        self.assertIsNone(r["decode_ms"])

    def test_huge_ints_do_not_raise(self):
        big = 10 ** 400
        r = metrics.engine_report(
            {"prompt_tokens": big, "generation_duration": big}, {"prompt_ms": big})
        self.assertEqual(r["prompt_tokens"], big)
        self.assertEqual(r["prompt_ms"], big)
        self.assertEqual(r["decode_ms"], big * 1000)

USAGE_ONLY = b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
BODY = (b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')


class TestStreamKeepsReport(unittest.TestCase):
    def test_stream_keeps_usage_timings_and_finish_mtplx(self):
        sm = metrics.StreamMetrics(0.0)
        with open(FIX + "/mtplx-stream.sse", "rb") as f:
            sm.feed(f.read())
        self.assertEqual(sm.finish, "stop")
        self.assertEqual(sm.report()["draft_accepted"], 79)

    def test_stream_keeps_omlx_usage(self):
        sm = metrics.StreamMetrics(0.0)
        with open(FIX + "/omlx-stream.sse", "rb") as f:
            sm.feed(f.read())
        self.assertEqual(sm.report()["output_tokens"], 300)
        self.assertEqual(sm.finish, "length")


class TestUsageChunkStripper(unittest.TestCase):
    def test_stripper_removes_only_the_usage_only_event(self):
        s = metrics.UsageChunkStripper()
        out = s.consume(BODY + USAGE_ONLY + b"data: [DONE]\n\n") + s.finish()
        self.assertEqual(out, BODY + b"data: [DONE]\n\n")

    def test_stripper_survives_any_split_point(self):
        whole = BODY + USAGE_ONLY + b"data: [DONE]\n\n"
        for cut in range(1, len(whole)):
            s = metrics.UsageChunkStripper()
            out = s.consume(whole[:cut]) + s.consume(whole[cut:]) + s.finish()
            self.assertEqual(out, BODY + b"data: [DONE]\n\n", cut)

    def test_stripper_handles_crlf_events(self):
        s = metrics.UsageChunkStripper()
        crlf = USAGE_ONLY.replace(b"\n\n", b"\r\n\r\n")
        self.assertEqual(s.consume(BODY + crlf) + s.finish(), BODY)

    def test_stripper_forwards_usage_that_rides_with_choices(self):
        ev = b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":2}}\n\n'
        s = metrics.UsageChunkStripper()
        self.assertEqual(s.consume(ev) + s.finish(), ev)

    def test_stripper_passes_non_json_and_comments_through(self):
        raw = b": keepalive\n\ndata: not json\n\n"
        s = metrics.UsageChunkStripper()
        self.assertEqual(s.consume(raw) + s.finish(), raw)

    def test_stripper_emits_complete_events_at_once(self):
        s = metrics.UsageChunkStripper()
        self.assertEqual(s.consume(BODY + b"data: {"), BODY)


class TestEngineReportTimingsFallback(unittest.TestCase):
    def test_counts_come_from_timings_when_usage_is_absent(self):
        r = metrics.engine_report(None, {
            "cache_n": 0, "prompt_n": 12, "prompt_ms": 40.231,
            "predicted_n": 8, "predicted_ms": 70.0})
        self.assertEqual(r["prompt_tokens"], 12)
        self.assertEqual(r["output_tokens"], 8)

    def test_cached_tokens_count_toward_the_prompt(self):
        r = metrics.engine_report(None, {
            "cache_n": 8, "prompt_n": 4, "predicted_n": 3})
        self.assertEqual(r["prompt_tokens"], 12)
        self.assertEqual(r["cached_tokens"], 8)

    def test_usage_wins_over_timings(self):
        r = metrics.engine_report(
            {"prompt_tokens": 5, "completion_tokens": 2},
            {"cache_n": 0, "prompt_n": 12, "predicted_n": 8})
        self.assertEqual(r["prompt_tokens"], 5)
        self.assertEqual(r["output_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
