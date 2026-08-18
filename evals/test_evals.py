"""Hermetic tests for the eval harness: no model, no network, no GPU.

Real python3 children (the sandbox's own subject) and loopback fake servers
are the only processes involved. Run: python3 -m unittest discover -s evals
"""
import argparse
import contextlib
import http.server
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run  # noqa: E402
import sandbox  # noqa: E402

TASKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.json")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------
# Extraction

class TestExtraction(unittest.TestCase):
    def test_think_block_is_stripped(self):
        self.assertEqual(run.strip_think("<think>x\ny</think>answer"), "answer")

    @staticmethod
    def _exec(code):
        scope = {}
        exec(compile(code, "<t>", "exec"), scope)
        return scope

    def test_iterated_versions_resolve_to_the_last(self):
        # Two versions of the same function: concatenation in reply order means
        # the later definition shadows the earlier - Python's own semantics,
        # asserted by executing what extraction returns rather than by string
        # equality against one block.
        content = ("First try:\n```python\ndef f():\n    return 1\n```\n"
                   "Actually:\n```py\ndef f():\n    return 2\n```")
        scope = self._exec(run.extract_code(content))
        self.assertEqual(scope["f"](), 2)

    def test_trailing_usage_fence_does_not_replace_the_solution(self):
        # Measured on redteam-4bit: the solution in an early fence, then a
        # fenced USAGE example that compiles. Taking only the last compiling
        # fence extracted the call without the definition and scored six
        # correct solutions as NameError.
        content = ("```python\ndef f():\n    return 41 + 1\n```\n"
                   "Example:\n```python\nresult = f()\n```")
        scope = self._exec(run.extract_code(content))
        self.assertEqual(scope["f"](), 42)
        self.assertEqual(scope["result"], 42)

    def test_definitions_split_across_fences_all_land(self):
        content = ("```python\nimport json\n```\nand then\n"
                   "```python\ndef g(x):\n    return json.dumps(x)\n```")
        scope = self._exec(run.extract_code(content))
        self.assertEqual(scope["g"]([1]), "[1]")

    def test_quoted_explanatory_fragment_is_dropped(self):
        # Measured on redteam-4bit explaining a bugfix: a compiling but
        # non-self-contained fragment (`if a[mid] < x:` alone) quoted mid
        # prose. Concatenating it executes the fragment at import and the
        # NameError buries a correct solution that follows.
        content = ("The bug is here:\n"
                   "```python\nif a[mid] < x:\n    lo = mid\n```\n"
                   "The fix:\n"
                   "```python\ndef find(a, x):\n    return 0 if a else -1\n```")
        scope = self._exec(run.extract_code(content))
        self.assertEqual(scope["find"]([1], 1), 0)

    def test_bare_demo_call_before_the_definition_is_dropped(self):
        # Also measured: a demo call quoted BEFORE the definition while
        # explaining old behavior. A bare call binds nothing, so it is not
        # part of the program the reply builds.
        content = ("Calling it twice shows the bug:\n"
                   "```python\nadd_tag('a')\n```\n"
                   "Fixed version:\n"
                   "```python\ndef add_tag(tag):\n    return [tag]\n```")
        scope = self._exec(run.extract_code(content))
        self.assertEqual(scope["add_tag"]("a"), ["a"])

    def test_bare_fence_and_no_fence(self):
        self.assertEqual(run.extract_code("```\nx = 1\n```"), "x = 1")
        self.assertEqual(run.extract_code("x = 1"), "x = 1")

    def test_trailing_output_demo_fence_is_skipped(self):
        # Measured on qwen3-coder: solution in an early fence, then a fenced
        # block of EXAMPLE OUTPUT last. Plain last-fence extracted the demo and
        # scored a correct solution as a SyntaxError.
        content = ("```python\ndef f():\n    return 1\n```\n"
                   "Running it prints:\n"
                   "```\nOriginal: 'aa' -> Encoded: 'a2'\n```")
        extracted = run.extract_code(content)
        self.assertIn("def f():", extracted)
        self.assertNotIn("Original", extracted)

    def test_nothing_compiles_keeps_the_last_fence(self):
        # The syntax error must surface in the detail, not be masked by
        # silently reaching for an earlier block.
        content = "```\nalso not() python(\n```\n```\nOriginal: -> broken(\n```"
        self.assertEqual(run.extract_code(content), "Original: -> broken(")

    def test_empty_content_yields_empty(self):
        self.assertEqual(run.extract_code(""), "")
        self.assertEqual(run.extract_code(None), "")

    def test_structured_call_with_string_arguments(self):
        call, channel = run.extract_tool_call(
            {"tool_calls": [{"function": {"name": "f", "arguments": "{\"x\": 1}"}}]})
        self.assertEqual(channel, "structured")
        self.assertEqual(call, {"name": "f", "args": {"x": 1}})

    def test_structured_call_with_dict_arguments(self):
        call, channel = run.extract_tool_call(
            {"tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}]})
        self.assertEqual((channel, call["args"]), ("structured", {"x": 1}))

    def test_text_channel_wrapper(self):
        call, channel = run.extract_tool_call(
            {"content": "<tool_call>{\"name\": \"g\", \"arguments\": {\"y\": 2}}</tool_call>"})
        self.assertEqual(channel, "text")
        self.assertEqual(call["name"], "g")

    def test_text_channel_bare_object(self):
        call, channel = run.extract_tool_call(
            {"content": "I will call: {\"name\": \"g\", \"arguments\": {\"y\": 2}}"})
        self.assertEqual(channel, "text")

    def test_no_call_anywhere(self):
        self.assertEqual(run.extract_tool_call({"content": "plain prose"}), (None, None))

    def test_malformed_structured_arguments(self):
        call, channel = run.extract_tool_call(
            {"tool_calls": [{"function": {"name": "f", "arguments": "{broken"}}]})
        self.assertIsNone(call)

    def test_json_extraction_tolerates_fences_and_prose(self):
        data = run.extract_json("Sure!\n```json\n{\"a\": 1}\n```\nDone.")
        self.assertEqual(data, {"a": 1})

    def test_json_extraction_array(self):
        self.assertEqual(run.extract_json("[1, 2]"), [1, 2])


# --------------------------------------------------------------------------
# Checkers

def _task(check, dim="instructions"):
    return {"id": "t", "dim": dim, "max_tokens": 64, "check": check, "stream": False}


class TestCheckers(unittest.TestCase):
    def test_exact_pass_fail(self):
        t = _task({"type": "exact", "value": "BUILD OK 42"})
        self.assertTrue(run.check_task(t, {"content": "BUILD OK 42"})["pass"])
        self.assertFalse(run.check_task(t, {"content": "BUILD OK 42."})["pass"])
        self.assertTrue(run.check_task(t, {"content": "<think>hm</think>BUILD OK 42"})["pass"])

    def test_word_count_boundary(self):
        t = _task({"type": "word_count", "max": 25})
        self.assertTrue(run.check_task(t, {"content": " ".join(["w"] * 25)})["pass"])
        self.assertFalse(run.check_task(t, {"content": " ".join(["w"] * 26)})["pass"])
        self.assertFalse(run.check_task(t, {"content": ""})["pass"])

    def test_line_rules_boundary(self):
        t = _task({"type": "line_rules", "count": 5, "each_matches": "[a-z]+"})
        five = "\n".join(["grep", "ls", "cat", "sed", "awk"])
        self.assertTrue(run.check_task(t, {"content": five})["pass"])
        self.assertFalse(run.check_task(t, {"content": five + "\nfind"})["pass"])
        self.assertFalse(run.check_task(t, {"content": "\n".join(five.split("\n")[:4])})["pass"])
        self.assertFalse(run.check_task(t, {"content": five.replace("sed", "3. sed")})["pass"])

    def test_absent(self):
        t = _task({"type": "absent", "words": ["memory", "storage"]})
        self.assertTrue(run.check_task(t, {"content": "It holds data for fast access."})["pass"])
        self.assertFalse(run.check_task(t, {"content": "It is fast Memory."})["pass"])
        self.assertFalse(run.check_task(t, {"content": ""})["pass"])

    def test_contains_any_of_groups(self):
        t = _task({"type": "contains", "values": ["A-1", ["blue", "azure"]]})
        self.assertTrue(run.check_task(t, {"content": "A-1 looks azure to me"})["pass"])
        self.assertFalse(run.check_task(t, {"content": "A-1 looks red"})["pass"])
        # Case-insensitive in both directions, pinned like the absent checker.
        self.assertTrue(run.check_task(t, {"content": "a-1 looks AZURE"})["pass"])

    def test_must_contain_is_case_insensitive(self):
        t = _task({"type": "no_tool_call", "must_contain": [["Paris"]]}, dim="tools")
        self.assertTrue(run.check_task(t, {"content": "It is PARIS."})["pass"])

    def test_json_shape_extra_key_rejected(self):
        t = _task({"type": "json_shape",
                   "shape": {"type": "object", "keys": {"port": {"eq": 8080}}}})
        self.assertTrue(run.check_task(t, {"content": "{\"port\": 8080}"})["pass"])
        self.assertFalse(run.check_task(t, {"content": "{\"port\": 8080, \"x\": 1}"})["pass"])
        self.assertFalse(run.check_task(t, {"content": "{\"port\": \"8080\"}"})["pass"])
        self.assertFalse(run.check_task(t, {"content": "no json here"})["pass"])

    def test_json_shape_array_arity_and_types(self):
        t = _task({"type": "json_shape",
                   "shape": {"type": "array", "len": 3, "items": {"type": "string"}}})
        self.assertTrue(run.check_task(t, {"content": "[\"a\", \"b\", \"c\"]"})["pass"])
        self.assertFalse(run.check_task(t, {"content": "[\"a\", \"b\"]"})["pass"])
        self.assertFalse(run.check_task(t, {"content": "[\"a\", \"b\", 3]"})["pass"])

    def test_json_booleans_do_not_pass_as_numbers(self):
        # bool subclasses int: without the guard, JSON true satisfies both
        # numeric kinds.
        for kind in ("number", "integer"):
            ok, _why = run._shape_ok({"type": kind}, True)
            self.assertFalse(ok, kind)
        ok, _why = run._shape_ok({"type": "number"}, 1.5)
        self.assertTrue(ok)

    def test_json_extraction_prefers_the_outer_array(self):
        # A one-object array: the {..} slice parses too, but the answer is
        # the array.
        self.assertEqual(run.extract_json("[{\"a\": 1}]"), [{"a": 1}])

    def test_run_tests_error_verdict_is_a_harness_error(self):
        # A sandbox that cannot even spawn says nothing about the model.
        t = _task({"type": "run_tests", "tests": "import solution"}, dim="code")
        fake = lambda code, tests: {"verdict": "error", "seconds": 0.0,
                                    "output": "spawn failed", "taskdir": None}
        v = run.check_task(t, {"content": "```python\nx = 1\n```"}, run_code=fake)
        self.assertFalse(v["pass"])
        self.assertTrue(v.get("harness_error"))

    def test_tool_call_wrong_args_fail(self):
        t = _task({"type": "tool_call", "name": "f", "args": {"x": 1}}, dim="tools")
        good = {"tool_calls": [{"function": {"name": "f", "arguments": "{\"x\": 1}"}}]}
        wrong_val = {"tool_calls": [{"function": {"name": "f", "arguments": "{\"x\": 2}"}}]}
        extra = {"tool_calls": [{"function": {"name": "f", "arguments": "{\"x\": 1, \"y\": 0}"}}]}
        wrong_name = {"tool_calls": [{"function": {"name": "g", "arguments": "{\"x\": 1}"}}]}
        self.assertTrue(run.check_task(t, good)["pass"])
        self.assertFalse(run.check_task(t, wrong_val)["pass"])
        self.assertFalse(run.check_task(t, extra)["pass"])
        self.assertFalse(run.check_task(t, wrong_name)["pass"])

    def test_tool_call_any_of(self):
        t = _task({"type": "tool_call", "name": "f",
                   "args": {"path": {"any_of": ["src", "src/"]}}}, dim="tools")
        m = {"tool_calls": [{"function": {"name": "f", "arguments": "{\"path\": \"src/\"}"}}]}
        self.assertTrue(run.check_task(t, m)["pass"])
        m2 = {"tool_calls": [{"function": {"name": "f", "arguments": "{\"path\": \"lib\"}"}}]}
        self.assertFalse(run.check_task(t, m2)["pass"])

    def test_text_channel_is_recoverable_not_a_pass(self):
        # The key policy: a correct call in the text channel measures a serving
        # gap. It must fail the headline AND be marked recoverable, because the
        # remedy differs from a model that cannot call at all.
        t = _task({"type": "tool_call", "name": "f", "args": {"x": 1}}, dim="tools")
        m = {"content": "<tool_call>{\"name\": \"f\", \"arguments\": {\"x\": 1}}</tool_call>"}
        verdict = run.check_task(t, m)
        self.assertFalse(verdict["pass"])
        self.assertTrue(verdict.get("recoverable"))
        self.assertEqual(verdict["channel"], "text")

    def test_no_tool_call_abstain(self):
        t = _task({"type": "no_tool_call", "must_contain": [["Paris"]]}, dim="tools")
        self.assertTrue(run.check_task(t, {"content": "The capital is Paris."})["pass"])
        called = {"content": "",
                  "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]}
        self.assertFalse(run.check_task(t, called)["pass"])
        self.assertFalse(run.check_task(t, {"content": "The capital is Lyon."})["pass"])
        self.assertFalse(run.check_task(t, {"content": ""})["pass"])

    def test_run_tests_uses_extracted_code(self):
        t = _task({"type": "run_tests", "tests": "from solution import f\nassert f() == 1"},
                  dim="code")
        good = {"content": "```python\ndef f():\n    return 1\n```"}
        bad = {"content": "```python\ndef f():\n    return 2\n```"}
        empty = {"content": "I cannot."}
        self.assertTrue(run.check_task(t, good)["pass"])
        self.assertFalse(run.check_task(t, bad)["pass"])
        self.assertFalse(run.check_task(t, empty)["pass"])

    def test_unknown_check_is_a_harness_error(self):
        v = run.check_task(_task({"type": "nope"}), {"content": "x"})
        self.assertFalse(v["pass"])
        self.assertTrue(v.get("harness_error"))


# --------------------------------------------------------------------------
# Sandbox

class TestSandbox(unittest.TestCase):
    def test_pass_fail_verdicts(self):
        ok = sandbox.run_code("def f():\n    return 3\n",
                              "from solution import f\nassert f() == 3")
        self.assertEqual(ok["verdict"], "pass")
        bad = sandbox.run_code("def f():\n    return 4\n",
                               "from solution import f\nassert f() == 3")
        self.assertEqual(bad["verdict"], "fail")

    def test_infinite_loop_is_killed_within_the_wall(self):
        begin = time.monotonic()
        r = sandbox.run_code("while True:\n    pass\n", "import solution",
                             timeout_s=2)
        self.assertEqual(r["verdict"], "timeout")
        self.assertLess(time.monotonic() - begin, 4)

    def test_network_use_raises(self):
        r = sandbox.run_code("import socket\nsocket.socket()\n", "import solution")
        self.assertEqual(r["verdict"], "fail")
        self.assertIn("network is disabled", r["output"])

    def test_environment_is_scrubbed(self):
        os.environ["FXLLA_EVAL_PLANTED_SECRET"] = "leak-me"
        try:
            r = sandbox.run_code(
                "import os\nassert 'FXLLA_EVAL_PLANTED_SECRET' not in os.environ\n",
                "import solution")
            self.assertEqual(r["verdict"], "pass")
        finally:
            del os.environ["FXLLA_EVAL_PLANTED_SECRET"]

    def test_runner_cwd_is_not_importable(self):
        # -I plus a taskdir cwd: the repo itself must be invisible, or a
        # solution could `import rag` and wander the caller's world.
        r = sandbox.run_code("import rag\n", "import solution")
        self.assertEqual(r["verdict"], "fail")
        self.assertIn("ModuleNotFoundError", r["output"])

    def test_timeout_kill_targets_the_whole_group(self):
        # Asserts the mechanism, not cross-process behaviour: RLIMIT_NPROC
        # counts the USER's total processes, so under the sandbox a fork can
        # never actually succeed on a workstation and an orphan cannot be
        # produced to observe. killpg is the backstop for a machine where the
        # limits land differently, and losing it must not pass silently.
        calls = []
        saved = sandbox.os.killpg

        def _spy(pgid, sig):
            calls.append((pgid, sig))
            return saved(pgid, sig)

        sandbox.os.killpg = _spy
        try:
            r = sandbox.run_code("while True:\n    pass\n", "import solution",
                                 timeout_s=1)
        finally:
            sandbox.os.killpg = saved
        self.assertEqual(r["verdict"], "timeout")
        self.assertTrue(calls, "timeout did not kill via the process group")

    def test_child_runs_isolated(self):
        # -I is the second layer under the env allowlist: if the allowlist ever
        # regresses, isolated mode still blocks PYTHONPATH and user site. The
        # flag is directly observable in the child.
        r = sandbox.run_code("import sys\nassert sys.flags.isolated == 1\n",
                             "import solution")
        self.assertEqual(r["verdict"], "pass")

    def test_oversized_write_is_stopped(self):
        r = sandbox.run_code("open('big', 'wb').write(b'x' * (20 * 1024 * 1024))\n",
                             "import solution")
        self.assertEqual(r["verdict"], "fail")

    def test_output_is_truncated(self):
        r = sandbox.run_code("print('y' * 200000)\nraise SystemExit(1)\n",
                             "import solution")
        self.assertLessEqual(len(r["output"]), sandbox.OUTPUT_CAP)

    def test_taskdir_removed_by_default_kept_on_request(self):
        r = sandbox.run_code("x = 1\n", "import solution")
        self.assertIsNone(r["taskdir"])
        r2 = sandbox.run_code("x = 1\n", "import solution", keep_dir=True)
        try:
            self.assertTrue(r2["taskdir"] and os.path.isdir(r2["taskdir"]))
        finally:
            import shutil
            shutil.rmtree(r2["taskdir"], ignore_errors=True)

    def test_fallback_without_sandbox_exec_agrees_on_cooperative_code(self):
        # The stdlib path is the Linux CI floor: for code that does what the
        # tasks ask (and for the casual socket tripwire) verdicts match the
        # seatbelt path.
        saved = sandbox._sandbox_exec_argv
        sandbox._sandbox_exec_argv = lambda argv, taskdir: (argv, False)
        try:
            ok = sandbox.run_code("def f():\n    return 3\n",
                                  "from solution import f\nassert f() == 3")
            self.assertEqual(ok["verdict"], "pass")
            net = sandbox.run_code("import socket\nsocket.socket()\n", "import solution")
            self.assertEqual(net["verdict"], "fail")
        finally:
            sandbox._sandbox_exec_argv = saved

    def test_seatbelt_is_stricter_never_looser(self):
        # The review falsified the old claim that verdicts are identical with
        # and without the seatbelt: an out-of-directory write has no
        # Python-level counterpart, so the wrapper can flip pass to fail on
        # code probing the sandbox. That direction is the contract now, and
        # this pins it where the seatbelt exists.
        if not os.path.exists("/usr/bin/sandbox-exec"):
            self.skipTest("no sandbox-exec on this platform")
        probe = os.path.join(tempfile.gettempdir(),
                             "fxlla-eval-divergence-probe")
        solution = "open(%r, 'w').write('x')\n" % probe
        try:
            with_belt = sandbox.run_code(solution, "import solution")
            self.assertEqual(with_belt["verdict"], "fail",
                             "seatbelt did not block an outside write")
            saved = sandbox._sandbox_exec_argv
            sandbox._sandbox_exec_argv = lambda argv, taskdir: (argv, False)
            try:
                without = sandbox.run_code(solution, "import solution")
            finally:
                sandbox._sandbox_exec_argv = saved
            self.assertEqual(without["verdict"], "pass",
                             "stdlib path unexpectedly blocks writes now - "
                             "update the README contract")
        finally:
            try:
                os.remove(probe)
            except OSError:
                pass

    def test_exit_zero_alone_is_not_a_pass(self):
        # os._exit(0) at module level ends the interpreter with status 0
        # before any assert has run; the sentinel requirement catches it.
        r = sandbox.run_code("import os\nos._exit(0)\n",
                             "import solution\nassert False")
        self.assertEqual(r["verdict"], "fail")

    def test_raw_socket_module_is_seatbelt_territory(self):
        # _socket is patched as a second tripwire, but the docstring is
        # explicit that Python-level patching is a speed bump: this asserts
        # the tripwire fires on the common re-import, nothing stronger.
        r = sandbox.run_code("import _socket\n_socket.socket()\n",
                             "import solution")
        self.assertEqual(r["verdict"], "fail")


# --------------------------------------------------------------------------
# Rendering and fingerprint

class TestRendering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = run.load_tasks(TASKS)

    def test_rendering_is_deterministic(self):
        a = run.render_tasks(self.spec)
        b = run.render_tasks(self.spec)
        self.assertEqual(run.fingerprint(a), run.fingerprint(b))

    def test_one_character_moves_the_fingerprint(self):
        import copy
        spec2 = copy.deepcopy(self.spec)
        spec2["tasks"][0]["prompt"] += "!"
        self.assertNotEqual(run.fingerprint(run.render_tasks(self.spec)),
                            run.fingerprint(run.render_tasks(spec2)))

    def test_filler_generator_is_inside_the_fingerprint(self):
        # The fingerprint hashes the RENDERED set: changing how filler is
        # generated must change it, which a file-bytes hash would miss.
        saved = run._WORDS
        run._WORDS = list(saved)
        run._WORDS[0] = "changedword"
        try:
            moved = run.fingerprint(run.render_tasks(self.spec))
        finally:
            run._WORDS = saved
        self.assertNotEqual(moved, run.fingerprint(run.render_tasks(self.spec)))

    def test_results_write_invariance(self):
        # The JOURNAL lesson as a test: writing results into the directory the
        # harness itself uses must not move the fingerprint. The temp dir is
        # wired in as EVAL_DIR, or the test would hold for any implementation
        # including one that hashes its own history.
        before = run.fingerprint(run.render_tasks(self.spec))
        saved = run.EVAL_DIR
        with tempfile.TemporaryDirectory() as d:
            run.EVAL_DIR = d
            try:
                with open(os.path.join(d, "history.jsonl"), "w") as fh:
                    fh.write("{\"score\": 1}\n")
                after = run.fingerprint(run.render_tasks(run.load_tasks(TASKS)))
            finally:
                run.EVAL_DIR = saved
        self.assertEqual(before, after)

    def test_plants_land_and_are_unique(self):
        text = run.gen_filler(11, 1000, [{"depth": 0.10, "text": "SERIAL-XK-1."}])
        self.assertEqual(text.count("SERIAL-XK-1."), 1)
        self.assertLess(text.index("SERIAL-XK-1."), len(text) // 4)
        late = run.gen_filler(11, 1000, [{"depth": 0.90, "text": "SERIAL-XK-1."}])
        self.assertGreater(late.index("SERIAL-XK-1."), len(late) * 3 // 4)

    def test_filler_differs_by_seed(self):
        self.assertNotEqual(run.gen_filler(1, 200, []), run.gen_filler(2, 200, []))

    def test_full_render_preserves_every_task(self):
        # The mutant this kills silently shrank the rendered set to one task
        # per dimension and every other rendering test still passed: quick
        # expects 4, the dim filter only checks non-emptiness, and the
        # fingerprint tests compare renders with themselves. Id-order equality
        # pins count, identity and order at once.
        rendered = run.render_tasks(self.spec)
        self.assertEqual([t["id"] for t in rendered],
                         [t["id"] for t in self.spec["tasks"]])
        per_dim = {}
        for t in rendered:
            per_dim[t["dim"]] = per_dim.get(t["dim"], 0) + 1
        self.assertEqual(per_dim, {"code": 10, "tools": 8, "instructions": 8,
                                   "context": 4, "candor": 8})

    def test_no_task_is_rendered_below_the_answer_floor(self):
        # A reasoning model spends tokens thinking before it answers, so a
        # budget sized for the answer alone is gone before the answer starts.
        # Measured on a Qwen3.5 pair: every dimension budgeted at 256 or 512
        # collapsed to empty replies, code at 2048 did not, and the two models
        # differed mostly by how much each happened to think.
        rendered = run.render_tasks(self.spec)
        low = [t["id"] for t in rendered if t["max_tokens"] < run.ANSWER_FLOOR]
        self.assertEqual(low, [], "budgeted below the floor: %s" % low)

    def test_a_task_asking_for_more_than_the_floor_keeps_it(self):
        # The floor raises, never lowers: a task that declares it needs room
        # knows something the floor does not.
        rendered = {t["id"]: t["max_tokens"] for t in run.render_tasks(self.spec)}
        declared = {t["id"]: t["max_tokens"] for t in self.spec["tasks"]}
        for tid, want in declared.items():
            if want > run.ANSWER_FLOOR:
                self.assertEqual(rendered[tid], want)

    def test_the_floor_is_inside_the_fingerprint(self):
        # Changing it changes what is measured, so runs either side of the
        # change must not look comparable.
        before = run.fingerprint(run.render_tasks(self.spec))
        saved = run.ANSWER_FLOOR
        run.ANSWER_FLOOR = saved * 2
        try:
            after = run.fingerprint(run.render_tasks(self.spec))
        finally:
            run.ANSWER_FLOOR = saved
        self.assertNotEqual(before, after)

    def test_quick_takes_one_task_per_dimension(self):
        quick = run.render_tasks(self.spec, quick=True)
        self.assertEqual(len(quick), len(run.DIM_ORDER))
        self.assertEqual(len({t["dim"] for t in quick}), len(run.DIM_ORDER))

    def test_dim_filter(self):
        only = run.render_tasks(self.spec, dims=["code"])
        self.assertTrue(only and all(t["dim"] == "code" for t in only))


class TestTaskSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = run.load_tasks(TASKS)

    def test_ids_unique_and_dims_known(self):
        ids = [t["id"] for t in self.spec["tasks"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(t["dim"] in run.DIM_ORDER for t in self.spec["tasks"]))

    def test_every_test_snippet_compiles(self):
        for t in self.spec["tasks"]:
            if t["check"]["type"] == "run_tests":
                compile(t["check"]["tests"], t["id"], "exec")

    def test_max_tokens_bounded(self):
        for t in self.spec["tasks"]:
            self.assertTrue(64 <= t["max_tokens"] <= 4096, t["id"])

    def test_no_task_references_this_repository(self):
        # A task whose answer lives in the repo would let doc edits move
        # scores - the retrieval eval already paid for that lesson. The note
        # field may name the command; the tasks themselves may not.
        blob = json.dumps(self.spec["tasks"]).lower()
        self.assertNotIn("fxlla", blob)


REFERENCE_SOLUTIONS = {
    "candor-possible": "def f(x):\n    return x == 1\n",
    "code-merge-intervals": (
        "def merge_intervals(intervals):\n"
        "    out = []\n"
        "    for s, e in sorted(intervals):\n"
        "        if out and s <= out[-1][1]:\n"
        "            out[-1][1] = max(out[-1][1], e)\n"
        "        else:\n"
        "            out.append([s, e])\n"
        "    return out\n"),
    "code-toposort": (
        "from collections import deque\n"
        "def topo_sort(edges, n):\n"
        "    adj = {i: [] for i in range(n)}\n"
        "    indeg = {i: 0 for i in range(n)}\n"
        "    for a, b in edges:\n"
        "        adj[a].append(b)\n"
        "        indeg[b] += 1\n"
        "    q = deque(i for i in range(n) if indeg[i] == 0)\n"
        "    out = []\n"
        "    while q:\n"
        "        u = q.popleft()\n"
        "        out.append(u)\n"
        "        for v in adj[u]:\n"
        "            indeg[v] -= 1\n"
        "            if indeg[v] == 0:\n"
        "                q.append(v)\n"
        "    return out if len(out) == n else None\n"),
    "code-parse-duration": (
        "import re\n"
        "def parse_duration(s):\n"
        "    m = re.fullmatch(r'(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?', s)\n"
        "    if not m or not any(m.groups()):\n"
        "        raise ValueError(s)\n"
        "    h, mi, se = (int(g) if g else 0 for g in m.groups())\n"
        "    return h * 3600 + mi * 60 + se\n"),
    "code-lru": (
        "from collections import OrderedDict\n"
        "class LRUCache:\n"
        "    def __init__(self, capacity):\n"
        "        self.cap = capacity\n"
        "        self.d = OrderedDict()\n"
        "    def get(self, key):\n"
        "        if key not in self.d:\n"
        "            return None\n"
        "        self.d.move_to_end(key)\n"
        "        return self.d[key]\n"
        "    def put(self, key, value):\n"
        "        self.d[key] = value\n"
        "        self.d.move_to_end(key)\n"
        "        while len(self.d) > self.cap:\n"
        "            self.d.popitem(last=False)\n"),
    "code-rle": (
        "import re\n"
        "def rle_encode(s):\n"
        "    out = []\n"
        "    i = 0\n"
        "    while i < len(s):\n"
        "        j = i\n"
        "        while j < len(s) and s[j] == s[i]:\n"
        "            j += 1\n"
        "        out.append(s[i] + str(j - i))\n"
        "        i = j\n"
        "    return ''.join(out)\n"
        "def rle_decode(s):\n"
        "    return ''.join(ch * int(n) for ch, n in re.findall(r'([a-zA-Z])(\\d+)', s))\n"),
    "code-deep-get": (
        "def deep_get(d, path, default=None):\n"
        "    cur = d\n"
        "    for part in path.split('.'):\n"
        "        if not isinstance(cur, dict) or part not in cur:\n"
        "            return default\n"
        "        cur = cur[part]\n"
        "    return cur\n"),
    "bugfix-binary-search": (
        "def find(a, x):\n"
        "    lo, hi = 0, len(a)\n"
        "    while lo < hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if a[mid] < x:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid\n"
        "    return lo if lo < len(a) and a[lo] == x else -1\n"),
    "bugfix-mutable-default": (
        "def add_tag(tag, tags=None):\n"
        "    if tags is None:\n"
        "        tags = []\n"
        "    tags.append(tag)\n"
        "    return tags\n"),
    "bugfix-date-compare": (
        "def is_before(a, b):\n"
        "    def key(d):\n"
        "        dd, mm, yy = d.split('/')\n"
        "        return (int(yy), int(mm), int(dd))\n"
        "    return key(a) < key(b)\n"),
    "bugfix-generator": (
        "def stats(nums):\n"
        "    nums = list(nums)\n"
        "    return (sum(nums), max(nums))\n"),
}


BROKEN_SOLUTIONS = {
    "candor-possible": "def f(x):\n    return True\n",
    # One known-bad solution per code task: right names, wrong behavior. Each
    # must FAIL its check snippet, or the snippet is vacuous and the task
    # measures nothing.
    "code-merge-intervals": "def merge_intervals(intervals):\n    return []\n",
    "code-toposort": "def topo_sort(edges, n):\n    return list(range(n))\n",
    "code-parse-duration": "def parse_duration(s):\n    return 0\n",
    "code-lru": ("class LRUCache:\n"
                 "    def __init__(self, capacity):\n        pass\n"
                 "    def get(self, key):\n        return None\n"
                 "    def put(self, key, value):\n        pass\n"),
    "code-rle": "def rle_encode(s):\n    return s\ndef rle_decode(s):\n    return s\n",
    "code-deep-get": "def deep_get(d, path, default=None):\n    return None\n",
    "bugfix-binary-search": "def find(a, x):\n    return -1\n",
    "bugfix-mutable-default": ("def add_tag(tag, tags=[]):\n"
                               "    tags.append(tag)\n    return tags\n"),
    "bugfix-date-compare": "def is_before(a, b):\n    return a < b\n",
    "bugfix-generator": "def stats(nums):\n    return (sum(nums), max(nums))\n",
}


class TestReferenceSolutions(unittest.TestCase):
    # An unsatisfiable task cannot ship, and neither can a vacuous one: every
    # code task must be passed by a known-good solution AND failed by a
    # known-bad one, both through the real sandbox.
    @classmethod
    def setUpClass(cls):
        cls.spec = run.load_tasks(TASKS)

    def test_every_code_task_is_satisfiable(self):
        for t in self.spec["tasks"]:
            if t["check"]["type"] != "run_tests":
                continue
            self.assertIn(t["id"], REFERENCE_SOLUTIONS,
                          "%s has no reference solution" % t["id"])
            r = sandbox.run_code(REFERENCE_SOLUTIONS[t["id"]], t["check"]["tests"])
            self.assertEqual(r["verdict"], "pass",
                             "%s: %s" % (t["id"], r["output"][-300:]))

    def test_every_code_task_can_fail(self):
        for t in self.spec["tasks"]:
            if t["check"]["type"] != "run_tests":
                continue
            self.assertIn(t["id"], BROKEN_SOLUTIONS,
                          "%s has no broken solution" % t["id"])
            r = sandbox.run_code(BROKEN_SOLUTIONS[t["id"]], t["check"]["tests"],
                                 timeout_s=4)
            self.assertNotEqual(r["verdict"], "pass",
                                "%s: its check snippet is vacuous" % t["id"])


# --------------------------------------------------------------------------
# HTTP and streaming

class _FakeChat(http.server.BaseHTTPRequestHandler):
    """Scripted /v1/chat/completions. The class attribute `script` is a list
    of dicts consumed per request: {"sse": [...]} streams chunks with optional
    per-chunk delays, {"body": {...}} answers plain, {"status": 400} errors.
    /v1/models answers with a decoy list headed by an unrelated model, like
    mlx_lm.server does: no code path may trust it."""
    script = []
    seen = []

    def do_GET(self):
        self.__class__.seen.append(self.path)
        body = json.dumps({"data": [{"id": "SOME-UNRELATED-DIFFUSION-MODEL"},
                                    {"id": "another"}]}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.__class__.seen.append(self.path)
        length = int(self.headers.get("Content-Length", 0))
        request_body = json.loads(self.rfile.read(length) or b"{}")
        self.__class__.seen_bodies.append(request_body)
        step = self.__class__.script.pop(0)
        if callable(step):
            step = step(request_body)
        if "status" in step:
            self.send_response(step["status"])
            self.end_headers()
            self.wfile.write(b"{}")
            return
        if "body" in step:
            payload = json.dumps(step["body"]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for item in step["sse"]:
            if isinstance(item, (int, float)):
                time.sleep(item)
                continue
            self.wfile.write(b"data: " + json.dumps(item).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *_a):
        pass


@contextlib.contextmanager
def fake_chat(script):
    handler = type("_H", (_FakeChat,), {"script": list(script), "seen": [],
                                        "seen_bodies": []})
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv.server_address[1], handler
    finally:
        srv.shutdown()
        srv.server_close()


def _delta(text):
    return {"choices": [{"delta": {"content": text}}]}


class TestHTTP(unittest.TestCase):
    def test_streamed_ttft_and_content(self):
        script = [{"sse": [{"choices": [{"delta": {"role": "assistant"}}]},
                           0.15, _delta("hello "), _delta("world"),
                           {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                           {"choices": [], "usage": {"completion_tokens": 2}}]}]
        with fake_chat(script) as (port, _h):
            message, sm, usage, finish, wall = run.request_streamed(
                port, "m", [{"role": "user", "content": "x"}], 64)
        self.assertEqual(message["content"], "hello world")
        self.assertEqual(finish, "stop")
        self.assertEqual(usage, {"completion_tokens": 2})
        ttft_ms, _tps, tokens = sm.result(sm.start + wall)
        self.assertGreaterEqual(ttft_ms, 140)  # the 0.15s pre-token sleep
        self.assertEqual(tokens, 2)            # usage overrides the delta count

    def test_streamed_tool_call_assembly(self):
        script = [{"sse": [
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"name": "f", "arguments": "{\"x\""}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": ": 1}"}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}]}]
        with fake_chat(script) as (port, _h):
            message, _sm, _u, finish, _w = run.request_streamed(
                port, "m", [{"role": "user", "content": "x"}], 64)
        self.assertEqual(finish, "tool_calls")
        call, channel = run.extract_tool_call(message)
        self.assertEqual((call["name"], call["args"], channel),
                         ("f", {"x": 1}, "structured"))

    def test_stream_options_rejection_is_retried_without(self):
        def first(body):
            self.assertIn("stream_options", body)
            return {"status": 400}
        script = [first, {"sse": [_delta("ok"),
                                  {"choices": [{"delta": {}, "finish_reason": "stop"}]}]}]
        with fake_chat(script) as (port, handler):
            message, _sm, usage, _f, _w = run.request_streamed(
                port, "m", [{"role": "user", "content": "x"}], 64)
            self.assertEqual(message["content"], "ok")
            self.assertIsNone(usage)  # no usage chunk: the ~ marker downstream
            self.assertNotIn("stream_options", handler.seen_bodies[-1])

    def test_sse_events_split_across_read_boundaries(self):
        # The fake usually writes one event per flush, so the carry buffer in
        # request_streamed is never exercised on a split event. A stream well
        # past the 8KB read size forces events to straddle reads; parsing
        # per-chunk without carrying the remainder loses deltas and the usage
        # chunk.
        n = 600
        events = [_delta("w%03d " % i) for i in range(n)]
        events.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        events.append({"choices": [], "usage": {"completion_tokens": n}})
        with fake_chat([{"sse": events}]) as (port, _h):
            message, sm, usage, finish, wall = run.request_streamed(
                port, "m", [{"role": "user", "content": "x"}], 4096)
        expected = "".join("w%03d " % i for i in range(n))
        self.assertEqual(message["content"], expected)
        self.assertEqual(usage, {"completion_tokens": n})
        _t, _tps, tokens = sm.result(sm.start + wall)
        self.assertEqual(tokens, n)

    def test_plain_request(self):
        script = [{"body": {"choices": [{"message": {"content": "hi"},
                                         "finish_reason": "stop"}],
                            "usage": {"completion_tokens": 1}}}]
        with fake_chat(script) as (port, _h):
            message, usage, finish, _w = run.request_plain(
                port, "m", [{"role": "user", "content": "x"}], 64)
        self.assertEqual((message["content"], finish), ("hi", "stop"))

    def test_request_functions_never_consult_v1_models(self):
        script = [{"body": {"choices": [{"message": {"content": "hi"}}]}}]
        with fake_chat(script) as (port, handler):
            run.request_plain(port, "m", [{"role": "user", "content": "x"}], 64)
            self.assertNotIn("/v1/models", handler.seen)

    def test_model_id_per_engine(self):
        # mlx_lm.server takes the model DIRECTORY path; llama-server the alias.
        # Collapsing them to the alias silently loads nothing on mlx.
        self.assertEqual(run.model_request_id("gguf", "coder", "/store/models/coder"),
                         "coder")
        self.assertEqual(run.model_request_id("mlx", "coder", "/store/models/coder"),
                         "/store/models/coder")
        # omlx validates the field and serves under the dir basename == alias.
        self.assertEqual(run.model_request_id("omlx", "coder", "/store/models/coder"),
                         "coder")


# --------------------------------------------------------------------------
# Lifecycle

_STUB_SERVER = r"""
import http.server, sys, time
time.sleep(float(sys.argv[2]))
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")
    def log_message(self, *a):
        pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


class TestLifecycle(unittest.TestCase):
    def _stub_backend(self, tmp, delay, fork=False):
        # fork=True keeps the shell as group leader with python as its child,
        # the shape that separates a group kill from a leader-only kill.
        server_py = os.path.join(tmp, "serve.py")
        with open(server_py, "w") as fh:
            fh.write(_STUB_SERVER)
        stub = os.path.join(tmp, "fxlla-stub")
        launcher = "" if fork else "exec "
        with open(stub, "w") as fh:
            fh.write("#!/bin/sh\n%s%s %s \"$3\" %s\n"
                     % (launcher, sys.executable, server_py, delay))
        os.chmod(stub, 0o755)
        return stub

    def test_ready_poll_is_fast_and_load_time_is_honest(self):
        # The journal's 1s-poll regression, as a timing assertion. The server
        # binds in-process at exactly t=0.4s - no interpreter startup noise in
        # the measured window - so a 0.1s poll must see it well before 0.75s,
        # where a 1s poll cannot come in under ~1.0.
        port = _free_port()

        class _Alive:
            def poll(self):
                return None

        srv = {}

        def _serve():
            time.sleep(0.4)
            handler = type("_H", (_FakeChat,), {"script": [], "seen": [],
                                                "seen_bodies": []})
            srv["s"] = http.server.HTTPServer(("127.0.0.1", port), handler)
            srv["s"].serve_forever()

        threading.Thread(target=_serve, daemon=True).start()
        try:
            load_s = run.wait_ready(_Alive(), "gguf", port, timeout_s=10)
        finally:
            if "s" in srv:
                srv["s"].shutdown()
                srv["s"].server_close()
        # 0.9, not tighter: the 1s-poll mutant has a hard floor of ~1.0
        # (time.sleep never undershoots), so 0.9 kills it deterministically
        # while leaving headroom for scheduler jitter on a loaded CI runner.
        self.assertGreaterEqual(load_s, 0.4)
        self.assertLess(load_s, 0.9)

    def test_teardown_frees_the_port_and_kills_the_group(self):
        port = _free_port()
        with tempfile.TemporaryDirectory() as tmp:
            stub = self._stub_backend(tmp, 0)
            saved = run.FXLLA_BIN
            run.FXLLA_BIN = stub
            try:
                proc = run.spawn_backend("x", port, os.path.join(tmp, "log"))
                run.wait_ready(proc, "gguf", port, timeout_s=10)
                run.teardown(proc, port)
            finally:
                run.FXLLA_BIN = saved
            self.assertFalse(run.port_in_use(port))
            self.assertIsNotNone(proc.poll())

    def test_teardown_kills_the_child_holding_the_port_not_just_the_leader(self):
        # With the shell as leader and python as its child, a leader-only kill
        # orphans the python still holding the port; the group kill gets both.
        # A surviving orphan makes teardown's port drain raise after its cap,
        # so the mutant is killed by the exception (and by the leaked server).
        port = _free_port()
        with tempfile.TemporaryDirectory() as tmp:
            stub = self._stub_backend(tmp, 0, fork=True)
            saved = run.FXLLA_BIN
            run.FXLLA_BIN = stub
            try:
                proc = run.spawn_backend("x", port, os.path.join(tmp, "log"))
                run.wait_ready(proc, "gguf", port, timeout_s=10)
                run.teardown(proc, port)
                self.assertFalse(run.port_in_use(port))
            finally:
                run.FXLLA_BIN = saved
                subprocess.run(["pkill", "-f", tmp], capture_output=True)

    def test_death_on_start_fails_fast(self):
        # A model that cannot load must fail in milliseconds, not burn the
        # size-scaled timeout (11 minutes for the largest).
        port = _free_port()
        proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"],
                                start_new_session=True)
        begin = time.monotonic()
        with self.assertRaises(RuntimeError) as ctx:
            run.wait_ready(proc, "gguf", port, timeout_s=60)
        self.assertLess(time.monotonic() - begin, 5)
        self.assertIn("exited on start", str(ctx.exception))

    def test_preoccupied_eval_port_is_refused_before_spawn(self):
        # The borrowed-server lesson replayed on the eval port: anything
        # already listening could be any model, so adopting it silently
        # measures the wrong thing. Refuse before spawning.
        with fake_chat([]) as (port, _h):
            saved_port, saved_spawn = run.EVAL_PORT, run.spawn_backend
            run.EVAL_PORT = port

            def _must_not_spawn(*_a, **_k):
                raise AssertionError("spawned a backend onto an occupied port")
            run.spawn_backend = _must_not_spawn
            try:
                args = argparse.Namespace(quick=True, keep_failed=False, repeats=1)
                with tempfile.TemporaryDirectory() as tmp:
                    with self.assertRaises(RuntimeError) as ctx:
                        run.eval_model(("a", "/d", "mlx", 100, "dev"), [], args, tmp)
                self.assertIn("refusing", str(ctx.exception))
            finally:
                run.EVAL_PORT, run.spawn_backend = saved_port, saved_spawn

    def test_eval_model_measures_what_the_report_prints(self):
        # The measurement path end to end against a scripted backend: probe 0
        # excluded from the medians and carried as first_request_s, the
        # 64-token tps gate, usage_exact propagation, repeat passes frozen out
        # of every headline number, unstable_tasks populated - and the SAME
        # record fed to render_report, so a producer/consumer key drift cannot
        # hide behind a hand-built fixture.
        port = _free_port()
        deltas = lambda n: [_delta("w ") for _ in range(n)]
        stop = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        usage = lambda n: {"choices": [], "usage": {"completion_tokens": n}}
        body = lambda text: {"body": {
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 5}}}
        # The 0.1 mid-stream is not decoration: a stream delivered in one burst
        # has no decode timeline to measure, and the rate is reported as
        # unmeasurable rather than as the transport's speed. Without a gap
        # here the fake backend is exactly that burst.
        script = [
            {"sse": [0.8] + deltas(35) + [0.1] + deltas(35) + [stop, usage(70)]},
            {"sse": deltas(35) + [0.1] + deltas(35) + [stop, usage(70)]},   # probe 1
            {"sse": deltas(35) + [0.1] + deltas(35) + [stop]},              # probe 2
            {"sse": deltas(10) + [stop, usage(10)]},           # probe 3: under gate
            body("BUILD OK 42"),                               # task, pass 1
            body("nope"),                                      # task, repeat pass
        ]
        handler = type("_H", (_FakeChat,), {"script": script, "seen": [],
                                            "seen_bodies": []})
        srv = {}

        class _Alive:
            pid = os.getpid()

            def poll(self):
                return None

        def _spawn(alias, port_, log_path):
            srv["s"] = http.server.HTTPServer(("127.0.0.1", port), handler)
            threading.Thread(target=srv["s"].serve_forever, daemon=True).start()
            return _Alive()

        rendered = [{"id": "t-exact", "dim": "instructions", "max_tokens": 64,
                     "prompt": "say it", "stream": False,
                     "check": {"type": "exact", "value": "BUILD OK 42"}}]
        saved = (run.EVAL_PORT, run.spawn_backend, run.teardown)
        run.EVAL_PORT = port
        run.spawn_backend = _spawn
        run.teardown = lambda proc, port_: srv["s"].shutdown()
        try:
            args = argparse.Namespace(quick=False, keep_failed=False, repeats=2)
            with tempfile.TemporaryDirectory() as tmp:
                rec = run.eval_model(("m", "/d", "mlx", 100, "dev"),
                                     rendered, args, tmp)
        finally:
            run.EVAL_PORT, run.spawn_backend, run.teardown = saved
            if "s" in srv:
                srv["s"].server_close()

        # Probe 0 is first_request_s, never in the medians.
        self.assertGreaterEqual(rec["first_request_s"], 0.8)
        self.assertGreaterEqual(rec["first_request_ttft_ms"], 700)
        self.assertLess(rec["ttft_ms"]["median"], 700)
        self.assertEqual(rec["ttft_ms"]["n"], 3)
        # tps: probes 1 and 2 qualify, probe 3 is under the 64-token gate.
        self.assertEqual(rec["tok_s"]["n"], 2)
        # Probe 2 sent no usage chunk.
        self.assertFalse(rec["usage_exact"])
        # 70+70+70+10 probe tokens plus the pass-1 task's 5: the repeat pass
        # is frozen out of tokens, tps and wall.
        self.assertEqual(rec["tokens_spent"], 225)
        self.assertEqual(rec["repeats"], 2)
        self.assertEqual(rec["unstable_tasks"], ["t-exact"])
        self.assertTrue(rec["results"][0]["pass"])

        text = run.render_report([rec], "f" * 12, False, {"instructions": 1})
        self.assertIn(str(rec["first_request_s"]), text)
        self.assertIn("1/1", text)

    def test_a_teardown_failure_does_not_destroy_the_results(self):
        # A model that spent minutes of GPU time must not report FAILED TO
        # LOAD because the port would not free afterwards: the record
        # survives, carries the teardown error, and the NEXT model's
        # pre-spawn check is what refuses the stuck port.
        port = _free_port()
        deltas = lambda n: [_delta("w ") for _ in range(n)]
        stop = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        script = [
            {"sse": deltas(70) + [stop]},
            {"sse": deltas(70) + [stop]},
            {"body": {"choices": [{"message": {"content": "BUILD OK 42"},
                                   "finish_reason": "stop"}]}},
        ]
        handler = type("_H", (_FakeChat,), {"script": script, "seen": [],
                                            "seen_bodies": []})
        srv = {}

        class _Alive:
            pid = os.getpid()

            def poll(self):
                return None

        def _spawn(alias, port_, log_path):
            srv["s"] = http.server.HTTPServer(("127.0.0.1", port), handler)
            threading.Thread(target=srv["s"].serve_forever, daemon=True).start()
            return _Alive()

        def _bad_teardown(proc, port_):
            srv["s"].shutdown()
            raise RuntimeError("port %d still held" % port_)

        rendered = [{"id": "t", "dim": "instructions", "max_tokens": 64,
                     "prompt": "say it", "stream": False,
                     "check": {"type": "exact", "value": "BUILD OK 42"}}]
        saved = (run.EVAL_PORT, run.spawn_backend, run.teardown)
        run.EVAL_PORT = port
        run.spawn_backend = _spawn
        run.teardown = _bad_teardown
        try:
            args = argparse.Namespace(quick=True, keep_failed=False, repeats=1)
            with tempfile.TemporaryDirectory() as tmp:
                with contextlib.redirect_stderr(io.StringIO()):
                    rec = run.eval_model(("m", "/d", "mlx", 100, "dev"),
                                         rendered, args, tmp)
        finally:
            run.EVAL_PORT, run.spawn_backend, run.teardown = saved
            if "s" in srv:
                srv["s"].server_close()
        self.assertFalse(rec.get("failed_to_load"))
        self.assertTrue(rec["results"][0]["pass"])
        self.assertIn("still held", rec["teardown_error"])

    def test_ready_url_is_engine_aware(self):
        # llama-server answers HTTP while weights still load; only /health
        # tells the truth. mlx_lm.server has no /health.
        self.assertTrue(run._ready_url("gguf", 1).endswith("/health"))
        self.assertTrue(run._ready_url("mlx", 1).endswith("/v1/models"))
        # omlx binds and lists /v1/models before the model loads (like mlx), so
        # poll that, not its /health, which is green before a lazy first load.
        self.assertTrue(run._ready_url("omlx", 1).endswith("/v1/models"))

    def test_ready_timeout_scales_with_size(self):
        self.assertEqual(run.ready_timeout_s(1000), 180)
        self.assertGreater(run.ready_timeout_s(132000), 600)


# --------------------------------------------------------------------------
# Report

def _rec(alias, results, usage_exact=True, **extra):
    rec = {"alias": alias, "engine": "mlx", "role": "dev", "size_mb": 100,
           "harness_version": run.EVAL_HARNESS_VERSION, "partial": False,
           "failed_to_load": False, "load_s": 1.0, "results": results,
           "tokens_spent": 10, "usage_exact": usage_exact, "wall_s": 5.0,
           "rss_peak_mb": 2048,
           "ttft_ms": {"median": 100, "min": 90, "max": 110, "n": 3},
           "tok_s": {"median": 50.0, "min": 45.0, "max": 55.0, "n": 10}}
    rec.update(extra)
    return rec


class TestHarnessVersion(unittest.TestCase):
    """The version is half the comparability rule, so pin it to a literal.

    Every other reference reads run.EVAL_HARNESS_VERSION dynamically, which
    means a forgotten bump - or a bump to the wrong number - changes nothing
    that any test can see, while silently making old and new speed numbers
    look comparable. Bumping this literal is the deliberate act; if this test
    fails, decide whether the change really alters what a number means, then
    update the literal AND the changelog note in run.py.
    """

    def test_version_is_pinned(self):
        self.assertEqual(run.EVAL_HARNESS_VERSION, 5)

    def test_every_version_has_a_written_reason(self):
        src = open(run.__file__, encoding="utf-8").read()
        head = src.split("EVAL_HARNESS_VERSION")[0]
        for n in range(2, run.EVAL_HARNESS_VERSION + 1):
            self.assertIn("# v%d:" % n, head,
                          "v%d bumped with no note saying what it changed" % n)


class TestReport(unittest.TestCase):
    def test_contract_fields_present(self):
        results = [{"id": "a", "dim": "code", "pass": True},
                   {"id": "b", "dim": "code", "pass": False, "detail": "boom"}]
        text = run.render_report([_rec("m1", results)], "abcdef123456", False,
                                 {"code": 2})
        for needle in ("abcdef123456", "harness v%d" % run.EVAL_HARNESS_VERSION,
                       "1/2", "noise", "same tasks fingerprint AND harness version",
                       "boom"):
            self.assertIn(needle, text)

    def test_cold_cost_asymmetry_is_stated(self):
        # mlx loads weights on the FIRST request, gguf before /health flips.
        # Verified live: a 16 GB mlx model showed load_s 0.75 and first_s 7.3.
        # A table with a load_s column and no first_s column lies by omission.
        text = run.render_report(
            [_rec("m1", [], first_request_s=7.3)], "f" * 12, False, {})
        self.assertIn("first_s", text)
        self.assertIn("7.3", text)
        self.assertIn("FIRST REQUEST", text)

    def test_median_not_mean(self):
        # A skewed sample: median 10, mean 40. The report must say 10.
        self.assertEqual(run._summ([10, 10, 100], 0)["median"], 10)

    def test_approx_marker_when_usage_missing(self):
        text = run.render_report(
            [_rec("m1", [{"id": "a", "dim": "code", "pass": True}],
                  usage_exact=False)], "f" * 12, False, {"code": 1})
        self.assertIn("50.0~", text.replace(" ", ""))

    def test_no_approx_marker_when_usage_is_exact(self):
        # One-sided assertions rot: the marker must also be provably absent.
        text = run.render_report(
            [_rec("m1", [{"id": "a", "dim": "code", "pass": True}])],
            "f" * 12, False, {"code": 1})
        self.assertNotIn("~", text.splitlines()[4])  # the m1 row

    def test_partial_and_failed_are_visible(self):
        text = run.render_report(
            [_rec("m1", [{"id": "a", "dim": "code", "pass": True}], partial=True),
             {"alias": "m2", "failed_to_load": True, "error": "died"}],
            "f" * 12, False, {"code": 1})
        self.assertIn("PARTIAL", text)
        self.assertIn("FAILED TO LOAD", text)

    def test_serving_layer_note_when_no_structured_call_ever(self):
        results = [{"id": "t1", "dim": "tools", "pass": False, "channel": "text",
                    "recoverable": True, "detail": "text channel"},
                   {"id": "t2", "dim": "tools", "pass": False, "channel": None,
                    "detail": "no call"}]
        text = run.render_report([_rec("m1", results)], "f" * 12, False,
                                 {"tools": 2})
        self.assertIn("serving-layer gap", text)
        structured = [{"id": "t1", "dim": "tools", "pass": False,
                       "channel": "structured", "detail": "wrong args"}]
        text2 = run.render_report([_rec("m1", structured)], "f" * 12, False,
                                  {"tools": 1})
        self.assertNotIn("serving-layer gap", text2)

    def test_harness_errors_are_never_silent_model_failures(self):
        # A dead request says nothing about the model: it leaves the pass
        # denominator, is flagged on the row, and is named in the detail.
        results = [{"id": "a", "dim": "code", "pass": False,
                    "harness_error": True, "detail": "request failed"},
                   {"id": "b", "dim": "code", "pass": True}]
        text = run.render_report([_rec("m1", results)], "f" * 12, False, {"code": 2})
        self.assertIn("HARNESS ERROR", text)
        self.assertIn("1/1!", text)  # measured denominator excludes the error
        self.assertIn("excluded from the denominators", text)
        self.assertEqual(run.dim_counts(results), {"code": (1, 1, 1)})

    def test_quick_is_labeled_not_a_ranking(self):
        text = run.render_report([_rec("m1", [])], "f" * 12, True, {})
        self.assertIn("not a ranking", text)


if __name__ == "__main__":
    unittest.main()
