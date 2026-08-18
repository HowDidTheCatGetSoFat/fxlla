import importlib
import inspect
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The gateway reads FXLLA_STORE at import time; point it at a scratch store
# with two model dirs before importing.
_STORE = tempfile.mkdtemp(prefix="fxlla-gw-")
os.environ["FXLLA_STORE"] = _STORE
_MODELS = os.path.join(_STORE, "models")
for _name, _engine in (("mlx-model", None), ("gguf-model", "gguf"), ("omlx-model", "omlx")):
    os.makedirs(os.path.join(_MODELS, _name), exist_ok=True)
    if _engine:
        with open(os.path.join(_MODELS, _name, ".engine"), "w") as _f:
            _f.write(_engine + "\n")

gw = importlib.import_module("fxlla_gateway")


class TestDownloadedModels(unittest.TestCase):
    # Embedding models share the store but cannot chat. Serving one spawns
    # llama-server without --embeddings on a BERT, and this list feeds the
    # opencode registration: 'embed' shipped as a selectable chat model once.
    def setUp(self):
        # The loop memory is module-global and survives between tests, so one
        # test's requests can push another over the limit and make it pass or
        # fail for a reason unrelated to its name. Start every test from empty.
        gw._reset_loop_memory()

    def _store(self, dirs):
        root = tempfile.mkdtemp(prefix="fxlla-dm-")
        for name, source in dirs.items():
            os.makedirs(os.path.join(root, name))
            with open(os.path.join(root, name, ".source"), "w") as fh:
                fh.write(source + "\n")
        return root

    def test_embed_models_are_excluded_by_alias_and_by_repo(self):
        catalog = tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False)
        catalog.write(
            "# comment\n"
            "chat  | org/chat-model  | 1GB | dev   | mlx  | note\n"
            "embed | org/embed-model | 1GB | embed | gguf | note\n")
        catalog.close()
        store = self._store({
            "chat": "org/chat-model",
            "embed": "org/embed-model",              # by alias
            "embed-model": "org/embed-model",        # pulled by org/repo form
            "incomplete": "",                        # no real .source content
        })
        os.remove(os.path.join(store, "incomplete", ".source"))
        saved = (gw.MODELS_DIR, gw.CATALOG)
        gw.MODELS_DIR, gw.CATALOG = store, catalog.name
        try:
            self.assertEqual(sorted(gw.downloaded_models()), ["chat"])
        finally:
            gw.MODELS_DIR, gw.CATALOG = saved
            os.unlink(catalog.name)

    def test_model_context_per_engine(self):
        # mlx serves the model's own window (config.json); gguf serves the -c
        # llama-server was started with. Feeding either the wrong one gives
        # opencode a context meter that lies in one direction or the other.
        store = self._store({"m-mlx": "org/a", "m-gguf": "org/b", "m-bare": "org/c"})
        with open(os.path.join(store, "m-mlx", "config.json"), "w") as fh:
            fh.write('{"max_position_embeddings": 262144}')
        with open(os.path.join(store, "m-gguf", ".engine"), "w") as fh:
            fh.write("gguf\n")
        saved = gw.MODELS_DIR
        gw.MODELS_DIR = store
        try:
            self.assertEqual(gw.model_context("m-mlx"), 262144)
            self.assertEqual(gw.model_context("m-gguf"), gw.SERVED_GGUF_CTX)
            self.assertIsNone(gw.model_context("m-bare"))
        finally:
            gw.MODELS_DIR = saved

    def _ctx(self, window):
        """Patch model_context so these tests state their own window."""
        saved = gw.model_context
        gw.model_context = lambda alias: window
        self.addCleanup(lambda: setattr(gw, "model_context", saved))

    def test_a_conversation_past_the_window_is_refused_with_the_number(self):
        """The failure this prevents is a silence, not an error.

        Measured against a real backend: a request ~1.18x over the window ran
        180 s and returned nothing at all - no error, no rejection. An opencode
        session here reached ~728k tokens against a 262k window and read as a
        hung chat. The refusal has to carry the numbers, because "too big"
        sends someone to restart things while "728k against 262k" sends them
        to start a new session, which is the only cure.
        """
        self._ctx(262144)
        body = {"messages": [{"role": "user", "content": "x" * 2548664}]}
        msg = gw.oversized_prompt(body, "m")
        self.assertIsNotNone(msg)
        self.assertIn("728", msg)
        self.assertIn("262144", msg)
        self.assertIn("new session", msg)
        # Compacting is the obvious move and the one that cannot work, so the
        # message has to say so or it will be tried first.
        self.assertIn("compacting", msg.lower())

    def test_a_conversation_that_fits_is_not_refused(self):
        """The case the rule must not catch. A request near the limit still
        goes to the backend, which is the only thing that knows for sure."""
        self._ctx(262144)
        for chars in (100, 100_000, 800_000):
            body = {"messages": [{"role": "user", "content": "x" * chars}]}
            self.assertIsNone(gw.oversized_prompt(body, "m"),
                              "%d chars was refused but fits" % chars)

    def test_max_tokens_counts_against_the_window(self):
        """Room to answer is part of what has to fit."""
        self._ctx(1000)
        body = {"messages": [{"role": "user", "content": "x" * 3400}]}   # ~971
        self.assertIsNone(gw.oversized_prompt(body, "m"))
        body["max_tokens"] = 500
        self.assertIsNotNone(gw.oversized_prompt(body, "m"))

    def test_an_unknown_window_refuses_nothing(self):
        """No window is not a small window. A model whose limit cannot be read
        must keep working exactly as before rather than have one invented."""
        self._ctx(None)
        body = {"messages": [{"role": "user", "content": "x" * 9_000_000}]}
        self.assertIsNone(gw.oversized_prompt(body, "m"))

    def test_tool_schemas_count(self):
        """They ride in the prompt. Counting only message strings under-reads
        a tool-heavy agent session, which is exactly the shape that overflows."""
        self._ctx(1000)
        body = {"messages": [{"role": "user", "content": "hi"}],
                "tools": [{"function": {"name": "f", "description": "d" * 8000}}]}
        self.assertIsNotNone(gw.oversized_prompt(body, "m"))

    def test_an_attached_photo_is_not_an_overflow(self):
        """An image is not its base64 length.

        The first version of this guard counted the data: URL as prompt text,
        so one ordinary phone photo - ~300KB, ~400k base64 characters - looked
        like a 114k-token prompt and was refused. A model that cannot see has
        that image replaced by a short description before the check runs, and
        one that can see spends a bounded number of tokens on it. Either way
        the encoding length is the wrong number.
        """
        self._ctx(32768)
        photo = "data:image/jpeg;base64," + "A" * 400_000
        body = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is in this photo?"},
            {"type": "image_url", "image_url": {"url": photo}}]}]}
        self.assertIsNone(gw.oversized_prompt(body, "m"))
        # Four of them - the vision path's own default cap - still fit.
        body["messages"][0]["content"] += [
            {"type": "image_url", "image_url": {"url": photo}} for _ in range(3)]
        self.assertIsNone(gw.oversized_prompt(body, "m"))
        # But real text beside the photo is still counted.
        body["messages"].append({"role": "user", "content": "y" * 200_000})
        self.assertIsNotNone(gw.oversized_prompt(body, "m"))

    def test_the_overflow_check_runs_after_add_vision(self):
        """The order IS the fix, and nothing else here tests order.

        oversized_prompt measures the body it is given. Ahead of add_vision it
        measures a body still carrying base64, and an ordinary photo is refused
        as a 114k-token overflow. The functions are both correct in isolation;
        only their sequence in do_POST decides whether a photo works, so assert
        the sequence rather than trust the comment beside it.
        """
        src = inspect.getsource(gw.Handler.do_POST)
        vision = src.find("add_vision(")
        overflow = src.find("oversized_prompt(")
        self.assertNotEqual(vision, -1, "add_vision no longer called in do_POST")
        self.assertNotEqual(overflow, -1, "oversized_prompt no longer called in do_POST")
        self.assertLess(vision, overflow,
                        "the overflow check moved ahead of add_vision, so an "
                        "attached photo is measured as base64 and refused")

    def _exchange(self, i, command, result):
        return [{"role": "assistant", "tool_calls": [
                    {"id": "c%d" % i, "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": command})}}]},
                {"role": "tool", "tool_call_id": "c%d" % i, "content": result}]

    def _convo(self, n, command, result=lambda i: "same"):
        msgs = []
        for i in range(n):
            msgs += self._exchange(i, command, result(i))
        return {"messages": msgs}

    def test_the_same_call_with_the_same_result_is_a_loop(self):
        """The failure this exists for.

        A local model ran one command 240 times over 8.5 hours. The command
        started a server in the foreground, so it could never exit and the
        result was identical every time. A model with no new information
        retrying is not a malfunction - it is the only move it has - so the
        stop has to come from outside it.
        """
        msg = gw.looping_tool_calls(self._convo(12, "timeout 120 python3 -m src.server"))
        self.assertIsNotNone(msg)
        self.assertIn("12 times", msg)
        # Naming the call is the point: "you are looping" sends someone to
        # restart, the command sends them to the cause.
        self.assertIn("src.server", msg)

    def test_the_same_call_with_a_changing_result_is_progress(self):
        """The case the rule must not catch. Polling a build, watching a file
        grow, waiting on a queue - all of these repeat one command forever and
        are all legitimate. The result changing is what tells them apart."""
        convo = self._convo(40, "curl localhost/build",
                            result=lambda i: "progress %d%%" % i)
        self.assertIsNone(gw.looping_tool_calls(convo))

    def test_ordinary_repetition_is_left_alone(self):
        """Running a test suite a few times while editing is a workflow.

        The memory is cleared between sizes on purpose: this checks the
        CONVERSATION rule alone. Without the reset, asking n=1..7 in one loop is
        seven requests all ending on the same exchange, which the gateway
        memory rightly counts as seven attempts - correct behaviour, wrong
        question. The memory has its own tests.
        """
        for n in range(1, gw.LOOP_LIMIT):
            gw._reset_loop_memory()
            self.assertIsNone(gw.looping_tool_calls(self._convo(n, "pytest")),
                              "%d repeats was called a loop" % n)
        gw._reset_loop_memory()

    def test_a_run_that_was_broken_by_a_different_call_is_not_a_loop(self):
        """Only a run ENDING at the last exchange counts. A model that tried
        something else in between is exploring, however badly.

        Both shapes are needed, and the first one alone proves nothing: with a
        DIFFERENT call last, counting occurrences anywhere would also find one
        and pass. The second shape - the repeated call is last, but the run was
        interrupted - is the one that separates "consecutive" from "how many in
        total", which is the actual rule.
        """
        convo = self._convo(20, "pytest")
        convo["messages"] += self._exchange(99, "ls", "a b c")
        self.assertIsNone(gw.looping_tool_calls(convo))

        interrupted = self._convo(gw.LOOP_LIMIT + 4, "pytest")
        interrupted["messages"] += self._exchange(98, "ls", "a b c")
        interrupted["messages"] += self._exchange(97, "pytest", "same")
        self.assertIsNone(gw.looping_tool_calls(interrupted),
                          "counted total occurrences instead of the final run")

    def test_an_unanswered_call_is_not_counted(self):
        """A call still in flight has no result to compare, and counting it
        would let an unlucky retry look like a settled loop.

        Stated as behaviour - calls with no result after them - rather than by
        breaking tool_call_id, which is how the first version of this test was
        written. That version encoded the id-matching MECHANISM, so it passed
        for a reason unrelated to its name and broke the moment matching became
        order-based, even though the behaviour it claims to check never
        changed.
        """
        # Every call issued, none answered.
        msgs = [{"role": "assistant", "tool_calls": [
                    {"id": "c%d" % i, "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "pytest"})}}]}
                for i in range(20)]
        self.assertIsNone(gw.looping_tool_calls({"messages": msgs}))

        # And a settled run whose newest call is still open stays a run of the
        # ANSWERED ones, so an open call cannot pad the count.
        convo = self._convo(gw.LOOP_LIMIT - 1, "pytest")
        convo["messages"] += [{"role": "assistant", "tool_calls": [
            {"id": "open", "function": {
                "name": "bash",
                "arguments": json.dumps({"command": "pytest"})}}]}]
        self.assertIsNone(gw.looping_tool_calls(convo))

    def test_polling_survives_a_client_that_reuses_call_ids(self):
        """Matching results to calls by id, not by order, broke this.

        A client that reuses tool_call_id - a fixed string, or a counter that
        restarts each turn, which small local-model harnesses really do - had
        every historical call rewritten to the newest result for that id. A
        build polled twenty times with twenty different answers then looked
        like twenty identical ones. That is the exact case the rule promises
        to let through, defeated by a plausible client rather than a hostile
        one.
        """
        msgs = []
        for i in range(20):
            msgs += [
                {"role": "assistant", "tool_calls": [
                    {"id": "call_1", "function": {                # same id, always
                        "name": "bash",
                        "arguments": json.dumps({"command": "curl localhost/build"})}}]},
                {"role": "tool", "tool_call_id": "call_1",
                 "content": "progress %d%%" % (i * 5)}]
        self.assertIsNone(gw.looping_tool_calls({"messages": msgs}))

    def test_a_hostile_body_is_not_an_exception(self):
        """An id can be any JSON. Keyed by id, a list id raised TypeError out
        of the handler and the client got no reply at all - worse than the
        check simply not firing."""
        bodies = [
            {"messages": [{"role": "assistant", "tool_calls": [
                {"id": ["not", "a", "string"],
                 "function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "abc", "content": "r"}]},
            {"messages": [{"role": "assistant", "tool_calls": "nope"}]},
            {"messages": [{"role": "assistant", "tool_calls": [None, 7, "x"]}]},
            {"messages": [{"role": "assistant", "tool_calls": [{"function": "no"}]}]},
            {"messages": [{"role": "tool", "content": {"nested": ["x"]}}]},
        ]
        for body in bodies:
            gw.looping_tool_calls(body)   # must not raise

    def test_the_limit_is_the_boundary_it_names(self):
        """Exactly LOOP_LIMIT identical answers is a loop; one fewer is not."""
        self.assertIsNone(gw.looping_tool_calls(self._convo(gw.LOOP_LIMIT - 1, "pytest")))
        self.assertIsNotNone(gw.looping_tool_calls(self._convo(gw.LOOP_LIMIT, "pytest")))

    def test_a_bad_loop_limit_does_not_stop_the_gateway(self):
        """The refusal says 'set FXLLA_LOOP_LIMIT=0 to disable', which invites
        FXLLA_LOOP_LIMIT=off. A bare int() there turned a wrong guess about one
        check into a gateway that would not import."""
        saved = os.environ.get("FXLLA_LOOP_LIMIT")
        try:
            for raw in ("off", "", "8.5", "disable"):
                os.environ["FXLLA_LOOP_LIMIT"] = raw
                self.assertEqual(gw._loop_limit(), 8, "%r took the gateway down" % raw)
            os.environ["FXLLA_LOOP_LIMIT"] = "0"
            self.assertEqual(gw._loop_limit(), 0)
        finally:
            if saved is None:
                os.environ.pop("FXLLA_LOOP_LIMIT", None)
            else:
                os.environ["FXLLA_LOOP_LIMIT"] = saved

    def test_a_loop_survives_a_compaction_that_empties_the_history(self):
        """opencode replaces the tool history with a summary when it compacts,
        so a run restarts at zero and a loop of hours reads as its first try.
        Not hypothetical: the 240-attempt session contained two compactions;
        they landed outside the run by luck.
        """
        gw._reset_loop_memory()
        self.addCleanup(gw._reset_loop_memory)
        caught = None
        for turn in range(1, gw.LOOP_LIMIT + 5):
            # After turn 4 the client compacts: each request now carries only
            # the newest exchange, exactly as opencode would send it.
            msgs = (self._exchange(turn, "pytest", "same") if turn > 4
                    else self._convo(turn, "pytest")["messages"])
            if gw.looping_tool_calls({"messages": msgs}):
                caught = turn
                break
        self.assertIsNotNone(caught, "a compaction hid the loop entirely")
        self.assertLessEqual(caught, gw.LOOP_LIMIT,
                             "the memory restarted along with the conversation")

    def test_the_memory_does_not_invent_loops(self):
        """Twenty separate requests whose result changes are twenty pieces of
        progress, however identical the command."""
        gw._reset_loop_memory()
        self.addCleanup(gw._reset_loop_memory)
        for i in range(20):
            msgs = self._exchange(i, "curl localhost/build", "progress %d" % i)
            self.assertIsNone(gw.looping_tool_calls({"messages": msgs}))

    def test_the_memory_forgets_after_the_window(self):
        """The thing worth bounding is time burned. A command reached again
        tomorrow is not the failure this is about."""
        gw._reset_loop_memory()
        self.addCleanup(gw._reset_loop_memory)
        sig = ("bash", "{}", "same")
        for i in range(gw.LOOP_LIMIT + 3):
            run = gw._remember_exchange(sig, now=i * (gw._LOOP_WINDOW_S + 1))
            self.assertEqual(run, 1, "an expired attempt still counted")

    def test_the_memory_is_bounded(self):
        gw._reset_loop_memory()
        self.addCleanup(gw._reset_loop_memory)
        for i in range(gw._LOOP_MEMORY_MAX * 2):
            gw._remember_exchange(("bash", str(i), "out"))
        self.assertLessEqual(len(gw._LOOP_MEMORY), gw._LOOP_MEMORY_MAX)

    def test_the_check_can_be_turned_off(self):
        saved = gw.LOOP_LIMIT
        gw.LOOP_LIMIT = 0
        self.addCleanup(lambda: setattr(gw, "LOOP_LIMIT", saved))
        self.assertIsNone(gw.looping_tool_calls(self._convo(500, "pytest")))

    def test_a_conversation_with_no_tools_is_untouched(self):
        for body in ({}, {"messages": "nope"},
                     {"messages": [{"role": "user", "content": "hi"}]}):
            self.assertIsNone(gw.looping_tool_calls(body))

    def test_a_legacy_completions_prompt_is_measured(self):
        """/v1/completions carries its input at the top level, through the same
        proxy. Measuring only `messages` left that path unguarded."""
        self._ctx(1000)
        self.assertIsNone(gw.oversized_prompt({"prompt": "x" * 100}, "m"))
        self.assertIsNotNone(gw.oversized_prompt({"prompt": "x" * 20_000}, "m"))

    def test_a_malformed_body_is_not_refused_here(self):
        """Shape validation belongs to the backend; guessing here would turn a
        clear downstream error into a wrong one."""
        self._ctx(1000)
        for body in ({}, {"messages": "nope"}, {"messages": []},
                     {"messages": [{"role": "user"}]}):
            self.assertIsNone(gw.oversized_prompt(body, "m"))

    def test_model_context_of_a_multimodal_config(self):
        """A multimodal model nests the text settings.

        Gemma 4 keeps max_position_embeddings under text_config, beside
        vision_config, and declares nothing at the top level. Reading only the
        top level reported no window at all for it - and no window is exactly
        when opencode's context meter and auto-compaction start working from a
        number nobody supplied. Measured: gemma-4-26b was the one model of
        fifteen with a null context in /v1/models.
        """
        store = self._store({"m-mm": "org/a", "m-empty-text": "org/b"})
        with open(os.path.join(store, "m-mm", "config.json"), "w") as fh:
            fh.write('{"model_type": "gemma4", "vision_config": {"x": 1},'
                     ' "text_config": {"max_position_embeddings": 131072}}')
        # A nested block that says nothing is still no answer, not a zero.
        with open(os.path.join(store, "m-empty-text", "config.json"), "w") as fh:
            fh.write('{"text_config": {"hidden_size": 8}}')
        saved = gw.MODELS_DIR
        gw.MODELS_DIR = store
        try:
            self.assertEqual(gw.model_context("m-mm"), 131072)
            self.assertIsNone(gw.model_context("m-empty-text"))
        finally:
            gw.MODELS_DIR = saved

    def test_a_missing_catalog_excludes_nothing(self):
        # A stranger's checkout with a moved catalog must not hide their
        # models; the filter fails open to the old behavior.
        store = self._store({"m1": "org/m1"})
        saved = (gw.MODELS_DIR, gw.CATALOG)
        gw.MODELS_DIR, gw.CATALOG = store, "/nonexistent/models.conf"
        try:
            self.assertEqual(sorted(gw.downloaded_models()), ["m1"])
        finally:
            gw.MODELS_DIR, gw.CATALOG = saved


class TestEngineDetection(unittest.TestCase):
    def test_default_engine_is_mlx(self):
        self.assertEqual(gw.engine_for("mlx-model"), "mlx")

    def test_gguf_marker(self):
        self.assertEqual(gw.engine_for("gguf-model"), "gguf")

    def test_omlx_marker(self):
        self.assertEqual(gw.engine_for("omlx-model"), "omlx")

    def test_missing_model_defaults_mlx(self):
        self.assertEqual(gw.engine_for("nope"), "mlx")


class TestModelField(unittest.TestCase):
    def test_mlx_sends_path(self):
        self.assertEqual(gw.model_field_for("mlx-model"),
                         os.path.join(_MODELS, "mlx-model"))

    def test_gguf_sends_alias(self):
        self.assertEqual(gw.model_field_for("gguf-model"), "gguf-model")

    def test_omlx_sends_alias(self):
        # omlx validates the model field and serves under the dir basename, so
        # it must get the bare alias, not the path mlx gets.
        self.assertEqual(gw.model_field_for("omlx-model"), "omlx-model")


class _FakeProc:
    pid = os.getpid()

    def terminate(self):
        pass


class TestWaitReady(unittest.TestCase):
    """A model switch waits on this loop, so its poll interval is the floor on
    switching, and a backend that dies must not hold the whole timeout."""

    def _manager(self):
        return gw.Manager.__new__(gw.Manager)

    def test_poll_interval_does_not_round_up_to_a_second(self):
        # At a one-second interval a backend ready at 1.01s was reported at 2.0s.
        self.assertLessEqual(gw.READY_POLL_INTERVAL, 0.2)

    def test_a_dead_backend_fails_at_once(self):
        class _Dead:
            def poll(self):
                return 1

        start = time.monotonic()
        self.assertFalse(self._manager()._wait_ready(9, timeout=5, proc=_Dead()))
        self.assertLess(time.monotonic() - start, 1.0,
                        "burned the timeout on a backend that had already exited")

    def test_a_live_but_silent_backend_waits_out_the_budget(self):
        # The opposite case: still loading, so the loop must keep waiting.
        class _Alive:
            def poll(self):
                return None

        start = time.monotonic()
        self.assertFalse(self._manager()._wait_ready(9, timeout=0.5, proc=_Alive()))
        self.assertGreaterEqual(time.monotonic() - start, 0.5)

    def test_no_proc_behaves_as_before(self):
        start = time.monotonic()
        self.assertFalse(self._manager()._wait_ready(9, timeout=0.4))
        self.assertGreaterEqual(time.monotonic() - start, 0.4)


class TestEnsureColdLoad(unittest.TestCase):
    """The loader path must return a real model_field, not the None sentinel,
    on the very first request to a freshly-loaded model."""

    def _ensure(self, alias):
        m = gw.Manager()
        saved = (gw.downloaded_models, gw.subprocess.Popen, gw.Manager._wait_ready)
        try:
            gw.downloaded_models = lambda: {alias: {"size_mb": 1}}
            gw.subprocess.Popen = lambda *a, **k: _FakeProc()
            gw.Manager._wait_ready = lambda self, port, timeout=180, proc=None: True
            return m.ensure(alias)
        finally:
            gw.downloaded_models, gw.subprocess.Popen, gw.Manager._wait_ready = saved

    def test_gguf_first_load_returns_alias(self):
        _port, model_field = self._ensure("gguf-model")
        self.assertEqual(model_field, "gguf-model")

    def test_mlx_first_load_returns_path(self):
        _port, model_field = self._ensure("mlx-model")
        self.assertEqual(model_field, os.path.join(_MODELS, "mlx-model"))
        self.assertIsNotNone(model_field)

    def test_load_cancelled_by_concurrent_unload(self):
        m = gw.Manager()
        saved = (gw.downloaded_models, gw.subprocess.Popen, gw.Manager._wait_ready)
        try:
            gw.downloaded_models = lambda: {"gguf-model": {"size_mb": 1}}
            gw.subprocess.Popen = lambda *a, **k: _FakeProc()

            def wait_ready(self, port, timeout=180, proc=None):
                self.epoch += 1   # an unload_all races this load
                return True
            gw.Manager._wait_ready = wait_ready
            with self.assertRaises(RuntimeError):
                m.ensure("gguf-model")
            self.assertEqual(m.backends, {})   # the raced load is not registered
        finally:
            gw.downloaded_models, gw.subprocess.Popen, gw.Manager._wait_ready = saved


class _TermProc:
    def __init__(self):
        self.pid = os.getpid()
        self.terminated = False
        self.waited = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True

    def kill(self):
        pass


class TestReapIdle(unittest.TestCase):
    """Unloading a backend nobody has used for a while.

    The gateway freed memory only under pressure: a model was evicted when the
    NEXT load would not fit, and otherwise stayed forever. FXLLA_KEEP_WARM was
    read by the single-model server alone, so the multi-model path ignored it
    while /health reported an idle counter nothing acted on.
    """

    def _manager(self, ages):
        m = gw.Manager()
        procs = {}
        now = time.monotonic()
        for alias, age in ages.items():
            procs[alias] = _TermProc()
            b = gw.Backend(alias, 8100 + len(procs), procs[alias], 10, alias, "gguf")
            b.last_used = now - age
            m.backends[alias] = b
        return m, procs

    def test_an_idle_backend_is_unloaded(self):
        m, procs = self._manager({"cold": 700})
        self.assertEqual(m.reap_idle(600), ["cold"])
        self.assertEqual(m.backends, {})
        self.assertTrue(procs["cold"].terminated and procs["cold"].waited)

    def test_a_recently_used_backend_is_left_alone(self):
        # The whole risk of a reaper: unloading what someone is about to use,
        # which for a 22 GB model costs minutes to undo.
        m, procs = self._manager({"warm": 30})
        self.assertEqual(m.reap_idle(600), [])
        self.assertIn("warm", m.backends)
        self.assertFalse(procs["warm"].terminated)

    def test_only_the_idle_ones_go(self):
        m, _ = self._manager({"cold": 900, "warm": 5, "older": 1200})
        self.assertEqual(set(m.reap_idle(600)), {"cold", "older"})
        self.assertEqual(list(m.backends), ["warm"])

    def test_exactly_at_the_threshold_counts_as_idle(self):
        m, _ = self._manager({"borderline": 600})
        self.assertEqual(m.reap_idle(600), ["borderline"])

    def test_zero_means_never(self):
        # 0 is the documented "keep them forever" value, and reaping on it
        # would unload every backend on the first tick.
        m, procs = self._manager({"a": 99999})
        self.assertEqual(m.reap_idle(0), [])
        self.assertIn("a", m.backends)
        self.assertFalse(procs["a"].terminated)

    def test_a_backend_with_work_in_flight_is_never_reaped(self):
        # last_used is stamped when a request is dispatched and not again, so
        # a generation that runs longer than the keep-warm window reads as
        # idle - and the reaper terminated it mid-stream, cutting the answer
        # off for a client doing nothing wrong.
        m, procs = self._manager({"busy": 99999})
        m.backends["busy"].inflight = 1
        self.assertEqual(m.reap_idle(600), [])
        self.assertFalse(procs["busy"].terminated)

    def test_it_is_reaped_once_the_work_finishes(self):
        m, _ = self._manager({"busy": 99999})
        m.begin("busy")
        self.assertEqual(m.reap_idle(600), [])
        m.end("busy")
        # end() restamps: idleness starts when the answer finishes, not when
        # it was asked for, so the window is measured from the right moment.
        self.assertEqual(m.reap_idle(600), [])
        m.backends["busy"].last_used -= 700
        self.assertEqual(m.reap_idle(600), ["busy"])

    def test_concurrent_requests_each_hold_it(self):
        m, procs = self._manager({"busy": 99999})
        m.begin("busy"); m.begin("busy")
        m.end("busy")
        m.backends["busy"].last_used -= 700
        self.assertEqual(m.reap_idle(600), [], "one release should not free it")
        m.end("busy")
        m.backends["busy"].last_used -= 700
        self.assertEqual(m.reap_idle(600), ["busy"])

    def test_end_on_a_vanished_backend_does_not_raise(self):
        # It can be evicted for budget while a request is in flight.
        m, _ = self._manager({"gone": 5})
        m.begin("gone")
        m.backends.clear()
        m.end("gone")

    def test_a_malformed_keep_warm_costs_the_feature_not_the_process(self):
        # Read at import: a typo took the whole gateway down before it bound a
        # port, while the single-model watchdog treats the same value as an
        # ignorable nuisance and keeps serving.
        saved = os.environ.get("FXLLA_KEEP_WARM")
        self.addCleanup(lambda: os.environ.__setitem__("FXLLA_KEEP_WARM", saved)
                        if saved is not None else os.environ.pop("FXLLA_KEEP_WARM", None))
        for bad in ("10m", "", "  ", "ten", "-3"):
            os.environ["FXLLA_KEEP_WARM"] = bad
            self.assertGreaterEqual(gw._keep_warm_s(), 0, bad)
        os.environ["FXLLA_KEEP_WARM"] = "7"
        self.assertEqual(gw._keep_warm_s(), 420)

    def test_reaping_an_empty_gateway_is_a_noop(self):
        self.assertEqual(gw.Manager().reap_idle(600), [])

    def test_the_gateway_reads_the_same_variable_as_the_cli(self):
        # A user who set FXLLA_KEEP_WARM once should not have to discover it
        # governed only one of the two ways to run this.
        self.assertEqual(gw.KEEP_WARM_S % 60, 0)
        self.assertGreaterEqual(gw.KEEP_WARM_S, 0)


class TestUnloadAll(unittest.TestCase):
    def test_unload_frees_waits_and_reports(self):
        m = gw.Manager()
        p1, p2 = _TermProc(), _TermProc()
        m.backends["a"] = gw.Backend("a", 8100, p1, 10, "a", "gguf")
        m.backends["b"] = gw.Backend("b", 8101, p2, 10, "b", "gguf")
        freed = m.unload_all()
        self.assertEqual(set(freed), {"a", "b"})
        self.assertEqual(m.backends, {})
        # terminated AND waited: memory is released before returning
        self.assertTrue(p1.terminated and p1.waited)
        self.assertTrue(p2.terminated and p2.waited)

    def test_unload_bumps_epoch(self):
        m = gw.Manager()
        e = m.epoch
        m.unload_all()
        self.assertEqual(m.epoch, e + 1)

    def test_unload_empty_is_noop(self):
        m = gw.Manager()
        self.assertEqual(m.unload_all(), [])


class TestLoopback(unittest.TestCase):
    def test_loopback_addresses(self):
        for a in ("127.0.0.1", "127.0.0.5", "::1", "::ffff:127.0.0.1"):
            self.assertTrue(gw._is_loopback(a), a)

    def test_non_loopback_addresses(self):
        for a in ("10.0.0.5", "192.168.1.9", "0.0.0.0", "::"):
            self.assertFalse(gw._is_loopback(a), a)


class TestRss(unittest.TestCase):
    def test_own_process_has_rss(self):
        self.assertGreater(gw.rss_mb(os.getpid()), 0)

    def test_bogus_pid_is_zero(self):
        self.assertEqual(gw.rss_mb(-1), 0)

class TestInFlightIsVisible(unittest.TestCase):
    """"Working" and "hung" look identical from outside, and that is the whole
    question when a client is showing a spinner. A 169k-token conversation
    spends about 80 seconds in prefill before the first token; nothing reported
    it, and three turns in a row were cancelled for looking dead."""

    def _mgr(self):
        m = gw.BackendManager() if hasattr(gw, "BackendManager") else gw.MANAGER
        return m

    def test_status_says_what_is_being_worked_on(self):
        mgr = self._mgr()
        alias = "probe-model"
        backend = type("B", (), {})()
        backend.alias, backend.port, backend.size_mb = alias, 9999, 100
        backend.last_used = time.monotonic()
        backend.inflight, backend.started, backend.prompt_tokens = 0, 0.0, None
        with mgr.lock:
            mgr.backends[alias] = backend
        self.addCleanup(lambda: mgr.backends.pop(alias, None))

        idle = [b for b in mgr.status() if b["alias"] == alias][0]
        self.assertEqual(idle["inflight"], 0)
        self.assertNotIn("busy_s", idle, "an idle backend claimed to be busy")

        backend.produced = 0
        mgr.begin(alias, prompt_tokens=181714)
        busy = [b for b in mgr.status() if b["alias"] == alias][0]
        self.assertEqual(busy["inflight"], 1)
        self.assertIn("busy_s", busy)
        self.assertEqual(busy["prompt_tokens"], 181714)
        # Before any token comes back it is reading the prompt...
        self.assertEqual(busy["phase"], "reading prompt")

        # ...and once tokens flow it is generating, with a live count. This is
        # the distinction whose absence made a normal long answer read as a
        # stuck prefill - "reading 11 tokens of context" while it wrote pages.
        mgr.progress(alias, 1078)
        gen = [b for b in mgr.status() if b["alias"] == alias][0]
        self.assertEqual(gen["phase"], "generating")
        self.assertEqual(gen["output_tokens"], 1078)

        mgr.end(alias)
        done = [b for b in mgr.status() if b["alias"] == alias][0]
        self.assertEqual(done["inflight"], 0)
        self.assertNotIn("busy_s", done, "it still looked busy after answering")

    def test_produced_count_resets_between_requests(self):
        """A new turn must not inherit the last turn's token count."""
        mgr = self._mgr()
        alias = "probe-model-3"
        backend = type("B", (), {})()
        backend.alias, backend.port, backend.size_mb = alias, 9997, 100
        backend.last_used = time.monotonic()
        backend.inflight, backend.started, backend.prompt_tokens = 0, 0.0, None
        backend.produced = 0
        with mgr.lock:
            mgr.backends[alias] = backend
        self.addCleanup(lambda: mgr.backends.pop(alias, None))
        mgr.begin(alias, prompt_tokens=10)
        mgr.progress(alias, 500)
        mgr.end(alias)
        mgr.begin(alias, prompt_tokens=10)
        fresh = [b for b in mgr.status() if b["alias"] == alias][0]
        self.assertEqual(fresh["phase"], "reading prompt",
                         "the new turn kept the old token count")

    def test_an_unknown_prompt_size_is_omitted_not_zero(self):
        """Zero tokens is a claim; no answer is not."""
        mgr = self._mgr()
        alias = "probe-model-2"
        backend = type("B", (), {})()
        backend.alias, backend.port, backend.size_mb = alias, 9998, 100
        backend.last_used = time.monotonic()
        backend.inflight, backend.started, backend.prompt_tokens = 0, 0.0, None
        backend.produced = 0
        with mgr.lock:
            mgr.backends[alias] = backend
        self.addCleanup(lambda: mgr.backends.pop(alias, None))
        mgr.begin(alias, prompt_tokens=None)
        busy = [b for b in mgr.status() if b["alias"] == alias][0]
        self.assertNotIn("prompt_tokens", busy)
        mgr.end(alias)


class TestTextChannelToolCalls(unittest.TestCase):
    """A tool call the backend's own parser dropped into the text channel.

    mlx_lm anchors its closing tag to the end of the string
    (`<function=(.*?)</function>$`), and qwen3-coder sometimes closes a
    Llama-shaped call with a Hermes-shaped `</tool_call>`. The anchor then
    fails and the whole call arrives as prose, which no client can act on.
    Measured at temperature 0: "Run the shell command: echo hi" parsed 3/3 and
    "Use the bash tool to run: echo hi" parsed 0/3 - about the wording, not
    luck. evals/README.md already calls this a serving-layer problem.
    """

    TOOLS = [{"type": "function", "function": {
        "name": "bash",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "timeout": {"type": "integer"}}}}}]

    # Exactly what the model emitted, trailing stray tag included.
    EMITTED = ("<function=bash>\n<parameter=command>\necho hi\n</parameter>\n"
               "</function>\n</tool_call>")

    def test_the_real_emission_is_recovered(self):
        calls = gw.text_tool_calls(self.EMITTED, self.TOOLS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "bash")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"command": "echo hi"})
        self.assertTrue(calls[0]["id"])

    def test_declared_types_are_honoured(self):
        text = ("<function=bash><parameter=command>ls</parameter>"
                "<parameter=timeout>30</parameter></function>")
        args = json.loads(gw.text_tool_calls(text, self.TOOLS)[0]["function"]["arguments"])
        self.assertEqual(args["timeout"], 30)      # int, not "30"
        self.assertIsInstance(args["timeout"], int)

    def test_prose_about_the_syntax_is_left_alone(self):
        """Turning prose into a call is worse than missing one."""
        for text in ("You could write <function=bash> but I will not",
                     "no tags here at all", ""):
            self.assertIsNone(gw.text_tool_calls(text, self.TOOLS))

    def test_a_tool_the_request_never_offered_is_refused(self):
        text = "<function=rm><parameter=path>/</parameter></function>"
        self.assertIsNone(gw.text_tool_calls(text, self.TOOLS))

    def test_nothing_happens_without_declared_tools(self):
        self.assertIsNone(gw.text_tool_calls(self.EMITTED, None))
        self.assertIsNone(gw.text_tool_calls(self.EMITTED, []))

    def test_a_stream_is_repaired_before_done(self):
        """The client stops at [DONE], so a recovered call has to land before
        it. The line is held rather than the whole answer buffered, which is
        what keeps a normal reply incremental."""
        r = gw._StreamRescue(self.TOOLS)
        out = bytearray()
        for piece in self.EMITTED.split("\n"):
            chunk = json.dumps({"model": "m", "choices": [
                {"delta": {"content": piece + "\n"}}]}).encode()
            got, rest = r.consume(b"data: " + chunk + b"\n")
            out += got
            self.assertEqual(rest, b"")
        got, rest = r.consume(b"data: [DONE]\n")
        out += got
        self.assertNotIn(b"[DONE]", bytes(out), "[DONE] was forwarded too early")
        out += r.finish(rest)
        body = bytes(out).decode()
        self.assertIn("tool_calls", body)
        self.assertLess(body.index("tool_calls"), body.index("[DONE]"))

    def test_a_stream_that_already_called_a_tool_is_untouched(self):
        r = gw._StreamRescue(self.TOOLS)
        chunk = json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "bash", "arguments": "{}"}}]}}]}).encode()
        r.consume(b"data: " + chunk + b"\n")
        r.consume(b"data: [DONE]\n")
        tail = r.finish(b"").decode()
        self.assertNotIn("<function=", tail)
        self.assertNotIn("tool_calls", tail, "a real call was duplicated")

    def test_a_stream_with_nothing_to_repair_ends_normally(self):
        r = gw._StreamRescue(self.TOOLS)
        chunk = json.dumps({"choices": [{"delta": {"content": "just prose"}}]}).encode()
        out, rest = r.consume(b"data: " + chunk + b"\ndata: [DONE]\n")
        self.assertIn(b"just prose", out)
        self.assertIn(b"[DONE]", r.finish(rest))


class TestVisionForModelsThatCannotSee(unittest.TestCase):
    """An image reaching a text model used to be a crash.

    `mlx_lm.server` raises "Only 'text' content type is supported" on any
    non-text part, so sending a picture to a coding model failed outright. The
    gateway now reads it with a model that can and hands the chosen one a
    description: one request in, one answer out, two models used. The point is
    that the capability lives behind the endpoint rather than in whatever is
    calling it - a client with no MCP support at all still gets it.
    """

    def setUp(self):
        # The description cache lives at module scope, which makes any test
        # touching it order-dependent on every other one. Cleared per test so a
        # cached description from an earlier case cannot answer a later one.
        gw._SEEN.clear()
        del gw._SEEN_ORDER[:]
        self.addCleanup(gw._SEEN.clear)
        self.addCleanup(lambda: gw._SEEN_ORDER.__delitem__(slice(None)))

    def _body(self, model="coder", text="what is this?"):
        return {"model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}

    def _stub_reader(self, answer="a red square with the word HOLA"):
        calls = []

        def fake(part, asked):
            calls.append((part, asked))
            return "seer", answer

        saved = gw._read_image
        gw._read_image = fake
        self.addCleanup(setattr, gw, "_read_image", saved)
        return calls

    def _stub_roles(self, vision=("seer",)):
        """Which models the catalog declares as readers.

        Only the catalog is faked. _can_see is left real so that what these
        tests exercise is the actual rule - declaration and projector both -
        rather than a stub standing in for it. Each declared model gets a
        projector on disk so the pair genuinely agrees."""
        for alias in vision:
            self._make_model(alias, projector=True)
        saved_role = gw._role_aliases
        gw._role_aliases = lambda role: set(vision) if role == "vision" else set()
        self.addCleanup(setattr, gw, "_role_aliases", saved_role)

    def test_the_image_becomes_text_for_a_model_that_cannot_see(self):
        self._stub_roles()
        self._stub_reader()
        body = self._body()
        self.assertEqual(gw.add_vision(body, "coder"), "seer")
        parts = body["messages"][0]["content"]
        self.assertTrue(all(p["type"] == "text" for p in parts))
        self.assertIn("HOLA", parts[1]["text"])

    def test_the_replacement_says_it_is_a_description(self):
        # The answering model must not reply as though it had seen the image.
        self._stub_roles()
        self._stub_reader()
        body = self._body()
        gw.add_vision(body, "coder")
        said = body["messages"][0]["content"][1]["text"]
        self.assertIn("an image was attached", said)
        self.assertIn("seer", said)

    def test_a_model_that_sees_gets_the_image_itself(self):
        # A description is strictly lossier than the thing it describes.
        self._stub_roles()
        calls = self._stub_reader()
        body = self._body(model="seer")
        self.assertIsNone(gw.add_vision(body, "seer"))
        self.assertEqual(calls, [])
        self.assertEqual(body["messages"][0]["content"][1]["type"], "image_url")

    def test_a_request_without_an_image_is_untouched(self):
        self._stub_roles()
        calls = self._stub_reader()
        body = {"model": "coder", "messages": [{"role": "user", "content": "hola"}]}
        before = json.dumps(body, sort_keys=True)
        self.assertIsNone(gw.add_vision(body, "coder"))
        self.assertEqual(json.dumps(body, sort_keys=True), before)
        self.assertEqual(calls, [])

    def test_every_image_is_read_not_only_the_first(self):
        self._stub_roles()
        calls = self._stub_reader()
        body = self._body()
        body["messages"][0]["content"].append(
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}})
        gw.add_vision(body, "coder")
        self.assertEqual(len(calls), 2)

    def test_the_users_words_reach_the_reader_as_context(self):
        # Relevance without confirmation: the reader is told what was asked so
        # it covers the right things, and told to report rather than verify.
        self._stub_roles()
        calls = self._stub_reader()
        gw.add_vision(self._body(text="is the lettering right?"), "coder")
        self.assertEqual(calls[0][1], "is the lettering right?")

    def test_it_can_be_turned_off(self):
        self._stub_roles()
        calls = self._stub_reader()
        saved = gw.VISION_ROUTING
        gw.VISION_ROUTING = False
        self.addCleanup(setattr, gw, "VISION_ROUTING", saved)
        body = self._body()
        self.assertIsNone(gw.add_vision(body, "coder"))
        self.assertEqual(calls, [])

    def test_no_vision_model_is_an_error_naming_the_fix(self):
        # Silence would forward the image and surface as the backend's own
        # "Only 'text' content type is supported", which names nothing useful.
        self._stub_roles(vision=())
        saved = os.environ.pop("FXLLA_VISION_MODEL", None)
        self.addCleanup(lambda: os.environ.__setitem__("FXLLA_VISION_MODEL", saved)
                        if saved is not None else None)
        with self.assertRaises(RuntimeError) as ctx:
            gw._read_image({"type": "image_url"}, "")
        self.assertIn("role 'vision'", str(ctx.exception))

    def test_the_words_around_the_image_survive(self):
        # Only the image slot may change. An over-eager rewrite that dropped or
        # mangled the user's own text would be invisible to every assertion
        # that only inspects the image slot.
        self._stub_roles()
        self._stub_reader()
        body = self._body(text="fix the parser")
        gw.add_vision(body, "coder")
        self.assertEqual(body["messages"][0]["content"][0],
                         {"type": "text", "text": "fix the parser"})

    def test_each_image_is_replaced_in_its_own_slot(self):
        self._stub_roles()
        saved = gw._read_image
        gw._read_image = lambda part, asked: ("seer", part["image_url"]["url"])
        self.addCleanup(setattr, gw, "_read_image", saved)
        body = self._body()
        body["messages"][0]["content"].append(
            {"type": "image_url", "image_url": {"url": "SECOND"}})
        gw.add_vision(body, "coder")
        parts = body["messages"][0]["content"]
        self.assertIn("data:image/png;base64,AAAA", parts[1]["text"])
        self.assertIn("SECOND", parts[2]["text"])

    def test_a_malformed_body_is_not_reported_as_a_vision_failure(self):
        # This runs on EVERY request. Anything raising here turns a request
        # that used to be forwarded into a 502 blamed on vision.
        self._stub_roles()
        self._stub_reader()
        for messages in ([None], ["a string"], [{"content": None}],
                         [{"content": ["not a dict"]}], "not a list", None):
            body = {"model": "coder", "messages": messages}
            self.assertIsNone(gw.add_vision(body, "coder"), repr(messages))

    def test_a_non_string_text_part_does_not_abort_the_read(self):
        self._stub_roles()
        calls = self._stub_reader()
        body = self._body()
        body["messages"][0]["content"][0] = {"type": "text", "text": {"oops": 1}}
        gw.add_vision(body, "coder")
        self.assertEqual(len(calls), 1)

    def test_it_reads_the_latest_user_turn_for_context(self):
        self._stub_roles()
        calls = self._stub_reader()
        body = self._body(text="the newest question")
        body["messages"].insert(0, {"role": "user", "content": "an older one"})
        body["messages"].insert(1, {"role": "assistant", "content": "sure"})
        gw.add_vision(body, "coder")
        self.assertEqual(calls[0][1], "the newest question")

    def _make_model(self, alias, projector=False):
        """A model directory, optionally with a multimodal projector in it."""
        path = os.path.join(_MODELS, alias)
        os.makedirs(path, exist_ok=True)
        if projector:
            open(os.path.join(path, "mmproj-f16.gguf"), "wb").close()
        return path

    def test_a_projector_is_what_reaches_llama_server(self):
        # bin/fxlla passes --mmproj when it finds one, so that file alone
        # decides whether an image can physically reach the model.
        self._make_model("no-eyes")
        self._make_model("has-eyes", projector=True)
        self.assertFalse(gw._has_projector("no-eyes"))
        self.assertTrue(gw._has_projector("has-eyes"))

    def test_a_projector_named_the_other_way_round_is_found(self):
        # Two conventions in the wild: `mmproj-Model-F16.gguf` (empero-ai) and
        # `Model.mmproj-Q8_0.gguf` (mradermacher). This globbed `mmproj*.gguf`,
        # so it said "cannot see" about the second - agreeing with bin/fxlla
        # only because bin/fxlla was anchored too and had already left the file
        # in the repo. Both were wrong; agreeing did not make either right.
        path = os.path.join(_MODELS, "late-eyes")
        os.makedirs(path, exist_ok=True)
        open(os.path.join(path, "Qwen3.5-9B.Q6_K.gguf"), "wb").close()
        open(os.path.join(path, "Qwen3.5-9B.mmproj-f16.gguf"), "wb").close()
        self.assertTrue(gw._has_projector("late-eyes"))

    def test_the_projector_test_matches_case_insensitively_like_the_launcher(self):
        # This must answer the same question bin/fxlla answers, and bin/fxlla
        # matches case-insensitively. Saying "cannot see" about a projector the
        # launcher WILL pass is how a model that can see gets told it cannot.
        path = os.path.join(_MODELS, "shouty-eyes")
        os.makedirs(path, exist_ok=True)
        open(os.path.join(path, "Model.Q6_K.gguf"), "wb").close()
        open(os.path.join(path, "Model.MMProj-F16.gguf"), "wb").close()
        self.assertTrue(gw._has_projector("shouty-eyes"))

    def test_an_undeclared_projector_does_not_make_a_model_trusted(self):
        # The regression this exists for: a model that ships a vision tower it
        # inherited and never tuned. The file is there, so it COULD be handed
        # the image, but nobody declared the eyes worth using.
        self._make_model("inherited-eyes", projector=True)
        self._stub_roles()
        self.assertTrue(gw._has_projector("inherited-eyes"))
        self.assertFalse(gw._can_see("inherited-eyes"))

    def test_a_declaration_without_a_projector_is_not_enough_either(self):
        # The other direction: the catalog can claim a role the disk cannot
        # honour, and a pull that fetched no projector is exactly that.
        # _stub_roles is bypassed here on purpose - it lays down a projector
        # for what it declares, which is the very thing being withheld.
        self._make_model("claims-eyes")
        saved = gw._role_aliases
        gw._role_aliases = lambda role: {"claims-eyes"} if role == "vision" else set()
        self.addCleanup(setattr, gw, "_role_aliases", saved)
        self.assertFalse(gw._can_see("claims-eyes"))

    def test_both_together_are_what_forwards_an_image_untouched(self):
        self._make_model("real-eyes", projector=True)
        self._stub_roles(vision=["real-eyes"])
        self.assertTrue(gw._can_see("real-eyes"))

    def test_an_undeclared_projector_still_gets_a_description(self):
        # End to end: the model could have taken the image, and is handed a
        # description anyway because its vision was never declared.
        self._make_model("inherited-eyes", projector=True)
        self._stub_roles()
        calls = self._stub_reader()
        body = self._body()
        self.assertEqual(gw.add_vision(body, "inherited-eyes"), "seer")
        self.assertEqual(len(calls), 1)
        self.assertEqual(body["messages"][-1]["content"][1]["type"], "text")

    def test_a_missing_catalog_falls_back_to_the_projector(self):
        # With the catalog moved there is no declaration to read AND no reader
        # to find, so demanding one would turn a working vision model into a
        # 502. The projector is the only evidence left, so it decides.
        self._make_model("has-eyes", projector=True)
        self._make_model("no-eyes")
        saved = gw.CATALOG
        gw.CATALOG = "/nonexistent/models.conf"
        self.addCleanup(setattr, gw, "CATALOG", saved)
        self.assertTrue(gw._can_see("has-eyes"))
        self.assertFalse(gw._can_see("no-eyes"))

    def test_an_unreadable_catalog_fails_open_like_a_missing_one(self):
        # Present but unopenable is the same situation as moved: no
        # declarations to read AND no reader to find. Treating it as "the
        # catalog declares nothing" denied every model instead.
        self._make_model("has-eyes", projector=True)
        blocked = os.path.join(_STORE, "unreadable.conf")
        with open(blocked, "w") as fh:
            fh.write("vision | org/x | 7GB | vision | gguf | n\n")
        os.chmod(blocked, 0o000)
        self.addCleanup(os.chmod, blocked, 0o644)
        saved = gw.CATALOG
        gw.CATALOG = blocked
        self.addCleanup(setattr, gw, "CATALOG", saved)
        if gw._role_aliases("vision") is not None:
            self.skipTest("running as a user that can read a 0o000 file")
        self.assertTrue(gw._can_see("has-eyes"))

    def test_a_readable_catalog_that_declares_nothing_is_not_fail_open(self):
        # The other half of the same distinction: an empty answer is an
        # answer, and must not be mistaken for "could not find out".
        self._make_model("has-eyes", projector=True)
        empty = os.path.join(_STORE, "empty.conf")
        with open(empty, "w") as fh:
            fh.write("# no rows\n")
        saved = gw.CATALOG
        gw.CATALOG = empty
        self.addCleanup(setattr, gw, "CATALOG", saved)
        self.assertEqual(gw._role_aliases("vision"), set())
        self.assertFalse(gw._can_see("has-eyes"))

    def test_an_image_free_request_never_touches_the_disk(self):
        # add_vision runs on every request. Answering "is there an image" in
        # memory first keeps the common case off the filesystem entirely.
        # The alias must be one the catalog DOES declare: an undeclared one
        # makes _can_see return False on role membership alone, so the disk is
        # never reached whatever the ordering, and the test proves nothing.
        def explode(alias):
            raise AssertionError("the disk was read for a request with no image")

        self._stub_roles(vision=["seer"])
        saved = gw._has_projector
        gw._has_projector = explode
        self.addCleanup(setattr, gw, "_has_projector", saved)
        body = {"messages": [{"role": "user", "content": "no image here"}]}
        self.assertIsNone(gw.add_vision(body, "seer"))

    def test_an_override_naming_a_blind_model_is_refused(self):
        # Sending an image to a text model would surface as a vision failure
        # blaming the reader rather than the misconfiguration.
        os.makedirs(os.path.join(_MODELS, "no-eyes"), exist_ok=True)
        os.environ["FXLLA_VISION_MODEL"] = "no-eyes"
        self.addCleanup(os.environ.pop, "FXLLA_VISION_MODEL", None)
        with self.assertRaises(RuntimeError) as ctx:
            gw._vision_alias()
        self.assertIn("projector", str(ctx.exception))

    def test_the_same_image_is_read_once_across_turns(self):
        # An OpenAI client resends the whole conversation every turn. Without a
        # cache the picture from turn one is re-read on every later turn, paying
        # its cost again and describing it differently each time, so the
        # answering model watches the same image change its mind.
        self._stub_roles()
        calls = self._stub_reader()
        first = self._body()
        gw.add_vision(first, "coder")
        second = self._body()          # same image, next turn
        gw.add_vision(second, "coder")
        self.assertEqual(len(calls), 1)
        self.assertEqual(first["messages"][0]["content"][1]["text"],
                         second["messages"][0]["content"][1]["text"])

    def test_a_different_image_is_read_again(self):
        self._stub_roles()
        calls = self._stub_reader()
        gw.add_vision(self._body(), "coder")
        other = self._body()
        other["messages"][0]["content"][1]["image_url"]["url"] = "data:image/png;base64,ZZZZ"
        gw.add_vision(other, "coder")
        self.assertEqual(len(calls), 2)

    def test_the_cache_does_not_grow_without_bound(self):
        self._stub_roles()
        self._stub_reader()
        for i in range(gw._SEEN_MAX + 10):
            body = self._body()
            body["messages"][0]["content"][1]["image_url"]["url"] = "u%d" % i
            gw.add_vision(body, "coder")
        self.assertLessEqual(len(gw._SEEN), gw._SEEN_MAX)
        self.assertEqual(len(gw._SEEN), len(gw._SEEN_ORDER))

    def test_too_many_images_is_refused_with_the_limit_named(self):
        # Each is read serially with its own timeout, so a large batch holds the
        # connection for as long as all of them take.
        self._stub_roles()
        calls = self._stub_reader()
        body = self._body()
        for i in range(gw.MAX_IMAGES):
            body["messages"][0]["content"].append(
                {"type": "image_url", "image_url": {"url": "extra%d" % i}})
        with self.assertRaises(RuntimeError) as ctx:
            gw.add_vision(body, "coder")
        self.assertIn(str(gw.MAX_IMAGES), str(ctx.exception))
        self.assertEqual(calls, [], "nothing should be read once it is refused")

    def test_exactly_the_limit_is_allowed(self):
        # The legitimate case the rule must not catch: comparing a few renders.
        self._stub_roles()
        self._stub_reader()
        body = self._body()
        body["messages"][0]["content"] = [{"type": "text", "text": "compare"}] + [
            {"type": "image_url", "image_url": {"url": "n%d" % i}}
            for i in range(gw.MAX_IMAGES)]
        self.assertEqual(gw.add_vision(body, "coder"), "seer")


if __name__ == "__main__":
    unittest.main()
