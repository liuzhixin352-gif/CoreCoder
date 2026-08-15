"""Tests for core modules: config, context, session, imports."""

from corecoder import Agent, LLM, Config, ALL_TOOLS, __version__
from corecoder import session as session_module
from corecoder.context import ContextManager, estimate_tokens
from corecoder.session import save_session, load_session, list_sessions
from corecoder.tools import get_tool


def test_version():
    assert __version__ == "0.4.0"


def test_public_api_exports():
    """Users should be able to import key classes from the top-level package."""
    assert Agent is not None
    assert LLM is not None
    assert Config is not None
    assert len(ALL_TOOLS) == 11


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("CORECODER_MODEL", "test-model")
    c = Config.from_env()
    assert c.model == "test-model"


def test_config_defaults(monkeypatch):
    # This test checks code defaults, so local .env settings must not affect it.
    monkeypatch.setattr("corecoder.config._load_dotenv", lambda: None)

    # clear relevant env vars without leaking the change into other tests
    monkeypatch.delenv("CORECODER_MODEL", raising=False)
    monkeypatch.delenv("CORECODER_MAX_TOKENS", raising=False)

    c = Config.from_env()
    assert c.model == "gpt-5.5"
    assert c.max_tokens == 4096
    assert c.temperature == 0.0


# --- Context ---

def test_estimate_tokens():
    msgs = [{"role": "user", "content": "hello world"}]
    t = estimate_tokens(msgs)
    assert t > 0
    assert t < 100

def test_context_budget_reserves_tokens_for_model_output():
    ctx = ContextManager(
        max_tokens=1000,
        reserved_output_tokens=200,
    )

    assert ctx.input_budget == 800
    assert ctx._snip_at == 400
    assert ctx._summarize_at == 560
    assert ctx._collapse_at == 720

def test_context_snip():
    ctx = ContextManager(max_tokens=3000)
    msgs = [
        {"role": "tool", "tool_call_id": "t1", "content": "x\n" * 1000},
    ]
    before = estimate_tokens(msgs)
    ctx._snip_tool_outputs(msgs)
    after = estimate_tokens(msgs)
    assert after < before


def test_context_compress():
    ctx = ContextManager(max_tokens=2000)
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"msg {i} " + "a" * 200})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "b" * 2000})
    before = estimate_tokens(msgs)
    ctx.maybe_compress(msgs, None)
    after = estimate_tokens(msgs)
    assert after < before
    assert len(msgs) < 40  # should be compressed


def test_safe_split_never_orphans_a_tool_message():
    """The kept tail must not begin with a 'tool' message - it would be severed
    from the assistant tool_calls that produced it, which the API rejects."""
    ctx = ContextManager(max_tokens=1000)
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {"role": "tool", "tool_call_id": "c2", "content": "result2"},
    ]
    split = ctx._safe_split(messages, keep_recent=1)
    assert messages[split].get("role") != "tool"


def test_compress_never_leaves_an_orphan_tool_reply():
    """After summarisation every tool reply must still follow its tool_calls."""
    ctx = ContextManager(max_tokens=2000)
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"msg {i} " + "a" * 200})
        msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}"}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "b" * 800})
    ctx.maybe_compress(msgs, None)
    for i, m in enumerate(msgs):
        if m.get("role") == "tool":
            prev = msgs[i - 1]
            assert prev.get("role") == "tool" or prev.get("tool_calls"), f"orphan tool at {i}"


# --- Session ---

def test_session_save_load(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    msgs = [{"role": "user", "content": "test message"}]
    save_session(msgs, "test-model", "pytest_test_session")
    loaded = load_session("pytest_test_session")
    assert loaded is not None
    assert loaded[0] == msgs
    assert loaded[1] == "test-model"


def test_session_name_is_sanitized(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    msgs = [{"role": "user", "content": "test message"}]
    sid = save_session(msgs, "test-model", "../Research Notes!")

    assert sid == "Research-Notes"
    assert (tmp_path / "Research-Notes.json").exists()
    assert load_session("../Research Notes!") is not None


def test_session_not_found():
    assert load_session("nonexistent_session_id") is None


def test_list_sessions():
    sessions = list_sessions()
    assert isinstance(sessions, list)


# --- Cost estimation ---

def test_cost_estimation_known_model():
    from corecoder.llm import LLM
    llm = LLM.__new__(LLM)
    llm.model = "gpt-5.4"
    llm.total_prompt_tokens = 1_000_000
    llm.total_completion_tokens = 500_000
    cost = llm.estimated_cost
    assert cost is not None
    assert cost == 2.5 + 7.5  # $2.5/M in + $15/M out * 0.5M

def test_cost_estimation_unknown_model():
    from corecoder.llm import LLM
    llm = LLM.__new__(LLM)
    llm.model = "some-custom-model"
    llm.total_prompt_tokens = 1000
    llm.total_completion_tokens = 500
    assert llm.estimated_cost is None


# --- Changed files tracking ---

def test_edit_tracks_changed_files(tmp_path):
    from corecoder.tools.edit import _changed_files
    _changed_files.clear()
    edit = get_tool("edit_file")
    path = tmp_path / "sample.py"
    path.write_text("aaa\nbbb\n")
    edit.execute(file_path=str(path), old_string="aaa", new_string="zzz")
    assert any(str(path) in p for p in _changed_files)
    _changed_files.clear()


def test_write_tracks_changed_files(tmp_path):
    from corecoder.tools.edit import _changed_files
    _changed_files.clear()
    write = get_tool("write_file")
    path = tmp_path / "tracked.txt"
    write.execute(file_path=str(path), content="tracked\n")
    assert any(path.name in p for p in _changed_files)
    _changed_files.clear()


# --- Agent tool execution ---


def test_exec_tools_parallel_runs_concurrently_and_preserves_order():
    import threading
    import time

    from corecoder.tools.base import Tool

    barrier = threading.Barrier(2)

    class _SlowTool(Tool):
        name = "slow"
        description = "slow test tool"
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        def execute(self):
            barrier.wait(timeout=1)
            time.sleep(0.05)
            return "slow-result"

    class _FastTool(Tool):
        name = "fast"
        description = "fast test tool"
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        def execute(self):
            barrier.wait(timeout=1)
            return "fast-result"

    agent = Agent(
        llm=LLM.__new__(LLM),
        tools=[_SlowTool(), _FastTool()],
    )

    class _TC:
        def __init__(self, name, tool_call_id):
            self.name = name
            self.id = tool_call_id
            self.arguments = {}

    tool_calls = [
        _TC("slow", "1"),
        _TC("fast", "2"),
    ]

    assert agent._exec_tools_parallel(tool_calls) == [
        "slow-result",
        "fast-result",
    ]
def test_exec_tools_parallel_uses_async_tool_execution():
    from corecoder.tools.base import Tool

    class _AsyncTool(Tool):
        name = "async_tool"
        description = "async test tool"
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        def execute(self):
            raise AssertionError("sync execute should not be used")

        async def aexecute(self):
            return "async-result"

    agent = Agent(
        llm=LLM.__new__(LLM),
        tools=[_AsyncTool()],
    )

    class _TC:
        name = "async_tool"
        id = "1"
        arguments = {}

    assert agent._exec_tools_parallel([_TC()]) == [
        "async-result",
    ]

def test_exec_tools_parallel_times_out_slow_async_tool():
    import asyncio

    from corecoder.tools.base import Tool

    class _SlowAsyncTool(Tool):
        name = "slow_async"
        description = "slow async test tool"
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        def execute(self):
            return "sync-result"

        async def aexecute(self):
            await asyncio.sleep(0.2)
            return "async-result"

    agent = Agent(
        llm=LLM.__new__(LLM),
        tools=[_SlowAsyncTool()],
        tool_timeout=0.05,
    )

    class _TC:
        name = "slow_async"
        id = "1"
        arguments = {}

    assert agent._exec_tools_parallel([_TC()]) == [
        "Error executing slow_async: timed out after 0.05 seconds",
    ]
def test_exec_tools_parallel_timeout_does_not_cancel_siblings():
    import asyncio

    from corecoder.tools.base import Tool

    class _SlowTool(Tool):
        name = "slow"
        description = "slow tool"
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        def execute(self):
            return "sync-slow"

        async def aexecute(self):
            await asyncio.sleep(0.2)
            return "slow-result"

    class _FastTool(Tool):
        name = "fast"
        description = "fast tool"
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        def execute(self):
            return "sync-fast"

        async def aexecute(self):
            await asyncio.sleep(0)
            return "fast-result"

    agent = Agent(
        llm=LLM.__new__(LLM),
        tools=[_SlowTool(), _FastTool()],
        tool_timeout=0.05,
    )

    class _TC:
        def __init__(self, name, tool_call_id):
            self.name = name
            self.id = tool_call_id
            self.arguments = {}

    tool_calls = [
        _TC("slow", "1"),
        _TC("fast", "2"),
    ]

    assert agent._exec_tools_parallel(tool_calls) == [
        "Error executing slow: timed out after 0.05 seconds",
        "fast-result",
    ]


def test_agent_tool_scope_is_per_instance():
    """An Agent restricted to a subset of tools must not resolve tools outside it."""
    only_read = [get_tool("read_file")]
    agent = Agent(llm=LLM.__new__(LLM), tools=only_read)
    assert set(agent._tool_by_name) == {"read_file"}

    class _TC:
        name = "bash"  # a real, registered tool - but not in this agent's set
        id = "x"
        arguments = {"command": "echo hi"}

    assert "unknown tool 'bash'" in agent._exec_tool(_TC())
def test_agent_executes_mcp_tool_through_existing_tool_interface(
    tmp_path,
):
    from corecoder.mcp_tools import create_mcp_server
    from corecoder.tools.mcp import load_mcp_tools

    target = tmp_path / "example.txt"
    target.write_text(
        "hello from agent through MCP",
        encoding="utf-8",
    )

    server = create_mcp_server(
        repository_root=tmp_path
    )
    tools = load_mcp_tools(server)

    agent = Agent(
        llm=LLM.__new__(LLM),
        tools=tools,
    )

    class _TC:
        name = "read_repository_file"
        id = "mcp-1"
        arguments = {"path": "example.txt"}

    assert (
        agent._exec_tool(_TC())
        == "hello from agent through MCP"
    )

def test_exec_tool_distinguishes_bad_args_from_internal_error():
    """A TypeError raised inside a tool must not be reported as bad arguments."""
    from corecoder.tools.base import Tool

    class _Boom(Tool):
        name = "boom"
        description = "raises TypeError internally"
        parameters = {"type": "object", "properties": {}, "required": []}

        def execute(self):
            raise TypeError("internal explosion")

    agent = Agent(llm=LLM.__new__(LLM), tools=[_Boom()])

    class _BadArgs:
        name, id, arguments = "boom", "1", {"unexpected": 1}

    class _Good:
        name, id, arguments = "boom", "2", {}

    assert "bad arguments" in agent._exec_tool(_BadArgs())
    assert "Error executing boom" in agent._exec_tool(_Good())
    assert "bad arguments" not in agent._exec_tool(_Good())


def test_interrupt_backfills_missing_tool_replies():
    """A half-finished tool round must be repaired so history stays valid."""
    agent = Agent(llm=LLM.__new__(LLM), tools=[])
    agent.messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a", "content": "done"},
    ]

    class _TC:
        def __init__(self, i):
            self.id = i

    agent._answer_pending_tool_calls([_TC("a"), _TC("b")])
    replies = [m for m in agent.messages if m.get("role") == "tool"]
    ids = [m["tool_call_id"] for m in replies]
    assert sorted(ids) == ["a", "b"]
    assert ids.count("a") == 1  # the already-answered call wasn't duplicated

def test_agent_accepts_tracer():
    from corecoder.tracing import InMemoryTracer

    tracer = InMemoryTracer()

    agent = Agent(
        llm=LLM.__new__(LLM),
        tools=[],
        tracer=tracer,
    )

    assert agent.tracer is tracer

def test_agent_emits_tool_started_trace():
    from corecoder.tracing import InMemoryTracer
    from corecoder.tools.base import Tool

    tracer = InMemoryTracer()

    class _Tool(Tool):
        name = "example"
        description = "example tool"
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        def execute(self):
            return "result"

        async def aexecute(self):
            assert tracer.events[0].name == "tool.started"
            return "result"

    agent = Agent(
        llm=LLM.__new__(LLM),
        tools=[_Tool()],
        tracer=tracer,
    )

    class _TC:
        name = "example"
        id = "call-1"
        arguments = {}

    assert agent._exec_tools_parallel([_TC()]) == [
        "result",
    ]

    event = tracer.events[0]

    assert event.name == "tool.started"
    assert event.timestamp > 0
    assert event.attributes == {
        "tool_name": "example",
        "tool_call_id": "call-1",
    }

def test_agent_emits_tool_completed_trace_with_duration():
    import asyncio

    from corecoder.tracing import InMemoryTracer
    from corecoder.tools.base import Tool

    tracer = InMemoryTracer()

    class _Tool(Tool):
        name = "example"
        description = "example tool"
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        def execute(self):
            return "result"

        async def aexecute(self):
            await asyncio.sleep(0)
            return "result"

    agent = Agent(
        llm=LLM.__new__(LLM),
        tools=[_Tool()],
        tracer=tracer,
    )

    class _TC:
        name = "example"
        id = "call-1"
        arguments = {}

    assert agent._exec_tools_parallel([_TC()]) == [
        "result",
    ]

    assert [event.name for event in tracer.events] == [
        "tool.started",
        "tool.completed",
    ]

    completed = tracer.events[1]

    assert completed.timestamp > 0
    assert completed.attributes["tool_name"] == "example"
    assert completed.attributes["tool_call_id"] == "call-1"
    assert completed.attributes["duration_ms"] >= 0

def test_agent_emits_tool_failed_trace():
    from corecoder.tracing import InMemoryTracer
    from corecoder.tools.base import Tool

    tracer = InMemoryTracer()

    class _FailingTool(Tool):
        name = "failing"
        description = "failing test tool"
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        def execute(self):
            raise RuntimeError("boom")

        async def aexecute(self):
            raise RuntimeError("boom")

    agent = Agent(
        llm=LLM.__new__(LLM),
        tools=[_FailingTool()],
        tracer=tracer,
    )

    class _TC:
        name = "failing"
        id = "call-1"
        arguments = {}

    assert agent._exec_tools_parallel([_TC()]) == [
        "Error executing failing: boom",
    ]

    assert [event.name for event in tracer.events] == [
        "tool.started",
        "tool.failed",
    ]

    failed = tracer.events[1]

    assert failed.attributes["tool_name"] == "failing"
    assert failed.attributes["tool_call_id"] == "call-1"
    assert failed.attributes["error_type"] == "RuntimeError"
    assert failed.attributes["duration_ms"] >= 0

def test_agent_emits_tool_timed_out_trace():
    import asyncio

    from corecoder.tracing import InMemoryTracer
    from corecoder.tools.base import Tool

    tracer = InMemoryTracer()

    class _SlowTool(Tool):
        name = "slow"
        description = "slow test tool"
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        def execute(self):
            return "sync-result"

        async def aexecute(self):
            await asyncio.sleep(0.2)
            return "result"

    agent = Agent(
        llm=LLM.__new__(LLM),
        tools=[_SlowTool()],
        tracer=tracer,
        tool_timeout=0.05,
    )

    class _TC:
        name = "slow"
        id = "call-1"
        arguments = {}

    assert agent._exec_tools_parallel([_TC()]) == [
        "Error executing slow: timed out after 0.05 seconds",
    ]

    assert [event.name for event in tracer.events] == [
        "tool.started",
        "tool.timed_out",
    ]

    timed_out = tracer.events[1]

    assert timed_out.attributes["tool_name"] == "slow"
    assert timed_out.attributes["tool_call_id"] == "call-1"
    assert timed_out.attributes["timeout_seconds"] == 0.05
    assert timed_out.attributes["duration_ms"] >= 0

def test_agent_emits_llm_started_and_completed_trace():
    from types import SimpleNamespace

    from corecoder.tracing import InMemoryTracer

    tracer = InMemoryTracer()

    class _FakeLLM:
        def chat(self, **kwargs):
            return SimpleNamespace(
                tool_calls=[],
                message={
                    "role": "assistant",
                    "content": "done",
                },
                content="done",
            )

    agent = Agent(
        llm=_FakeLLM(),
        tools=[],
        tracer=tracer,
    )

    assert agent.chat("hello") == "done"

    llm_events = [
        event
        for event in tracer.events
        if event.name.startswith("llm.")
    ]

    assert [event.name for event in llm_events] == [
        "llm.started",
        "llm.completed",
    ]

    started = llm_events[0]
    completed = llm_events[1]

    assert started.attributes["round"] == 1
    assert completed.attributes["round"] == 1
    assert completed.attributes["duration_ms"] >= 0

def test_agent_emits_llm_failed_trace():
    import pytest

    from corecoder.tracing import InMemoryTracer

    tracer = InMemoryTracer()

    class _FailingLLM:
        def chat(self, **kwargs):
            raise RuntimeError("boom")

    agent = Agent(
        llm=_FailingLLM(),
        tools=[],
        tracer=tracer,
    )

    with pytest.raises(RuntimeError, match="boom"):
        agent.chat("hello")

    llm_events = [
        event
        for event in tracer.events
        if event.name.startswith("llm.")
    ]

    assert [event.name for event in llm_events] == [
        "llm.started",
        "llm.failed",
    ]

    failed = llm_events[1]

    assert failed.attributes["round"] == 1
    assert failed.attributes["error_type"] == "RuntimeError"
    assert failed.attributes["duration_ms"] >= 0

def test_agent_emits_started_and_completed_trace():
    from types import SimpleNamespace

    from corecoder.tracing import InMemoryTracer

    tracer = InMemoryTracer()

    class _FakeLLM:
        def chat(self, **kwargs):
            return SimpleNamespace(
                tool_calls=[],
                message={
                    "role": "assistant",
                    "content": "done",
                },
                content="done",
            )

    agent = Agent(
        llm=_FakeLLM(),
        tools=[],
        tracer=tracer,
    )

    assert agent.chat("hello") == "done"

    assert [event.name for event in tracer.events] == [
        "agent.started",
        "context.managed",
        "llm.started",
        "llm.completed",
        "agent.completed",
    ]

    started = tracer.events[0]
    completed = tracer.events[-1]

    assert "run_id" in started.attributes
    assert completed.attributes["run_id"] == started.attributes["run_id"]
    assert completed.attributes["duration_ms"] >= 0
    assert completed.attributes["llm_rounds"] == 1

def test_agent_emits_pre_llm_context_managed_trace():
    from types import SimpleNamespace

    from corecoder.tracing import InMemoryTracer

    tracer = InMemoryTracer()

    class _FakeLLM:
        def chat(self, **kwargs):
            return SimpleNamespace(
                tool_calls=[],
                message={
                    "role": "assistant",
                    "content": "done",
                },
                content="done",
            )

    agent = Agent(
        llm=_FakeLLM(),
        tools=[],
        tracer=tracer,
    )

    assert agent.chat("hello") == "done"

    context_events = [
        event
        for event in tracer.events
        if event.name == "context.managed"
    ]

    assert len(context_events) == 1

    event = context_events[0]

    assert event.attributes["phase"] == "pre_llm"
    assert event.attributes["tokens_before"] == (
        event.attributes["tokens_after"]
    )
    assert event.attributes["tokens_saved"] == 0
    assert event.attributes["applied_layers"] == ()
    assert event.attributes["high_priority_messages"] == 0
    assert event.attributes["priority_preserved_messages"] == 0
    assert "run_id" in event.attributes

def test_agent_emits_post_tool_context_managed_trace():
    from types import SimpleNamespace

    from corecoder.tracing import InMemoryTracer

    tracer = InMemoryTracer()

    class _FakeLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1

            if self.calls == 1:
                tool_call = SimpleNamespace(
                    name="missing_tool",
                    id="call-1",
                    arguments={},
                )
                return SimpleNamespace(
                    tool_calls=[tool_call],
                    message={
                        "role": "assistant",
                        "content": "",
                    },
                    content="",
                )

            return SimpleNamespace(
                tool_calls=[],
                message={
                    "role": "assistant",
                    "content": "done",
                },
                content="done",
            )

    agent = Agent(
        llm=_FakeLLM(),
        tools=[],
        tracer=tracer,
    )

    assert agent.chat("hello") == "done"

    context_events = [
        event
        for event in tracer.events
        if event.name == "context.managed"
    ]

    assert len(context_events) == 2
    assert context_events[0].attributes["phase"] == "pre_llm"
    assert context_events[1].attributes["phase"] == "post_tool"
    assert (
        context_events[0].attributes["run_id"]
        == context_events[1].attributes["run_id"]
    )


def test_agent_emits_completed_trace_when_max_rounds_reached():
    from types import SimpleNamespace

    from corecoder.tracing import InMemoryTracer

    tracer = InMemoryTracer()

    class _FakeLLM:
        def chat(self, **kwargs):
            tool_call = SimpleNamespace(
                name="missing_tool",
                id="call-1",
                arguments={},
            )
            return SimpleNamespace(
                tool_calls=[tool_call],
                message={
                    "role": "assistant",
                    "content": "",
                },
                content="",
            )

    agent = Agent(
        llm=_FakeLLM(),
        tools=[],
        tracer=tracer,
        max_rounds=2,
    )

    assert agent.chat("hello") == (
        "(reached maximum tool-call rounds)"
    )

    assert tracer.events[-1].name == "agent.completed"
    assert tracer.events[-1].attributes["llm_rounds"] == 2
    assert tracer.events[-1].attributes["duration_ms"] >= 0

def test_agent_emits_failed_trace_when_llm_fails():
    import pytest

    from corecoder.tracing import InMemoryTracer

    tracer = InMemoryTracer()

    class _FailingLLM:
        def chat(self, **kwargs):
            raise RuntimeError("boom")

    agent = Agent(
        llm=_FailingLLM(),
        tools=[],
        tracer=tracer,
    )

    with pytest.raises(RuntimeError, match="boom"):
        agent.chat("hello")

    assert [event.name for event in tracer.events] == [
        "agent.started",
        "context.managed",
        "llm.started",
        "llm.failed",
        "agent.failed",
    ]

    failed = tracer.events[-1]

    assert failed.attributes["error_type"] == "RuntimeError"
    assert failed.attributes["llm_rounds"] == 1
    assert failed.attributes["duration_ms"] >= 0

def test_tracer_failure_does_not_break_agent():
    from types import SimpleNamespace

    from corecoder.tracing import TraceEvent, Tracer

    class _FailingTracer(Tracer):
        def emit(self, event: TraceEvent) -> None:
            raise RuntimeError("tracer broke")

    class _FakeLLM:
        def chat(self, **kwargs):
            return SimpleNamespace(
                tool_calls=[],
                message={
                    "role": "assistant",
                    "content": "done",
                },
                content="done",
            )

    agent = Agent(
        llm=_FakeLLM(),
        tools=[],
        tracer=_FailingTracer(),
    )

    assert agent.chat("hello") == "done"

def test_tracer_failure_does_not_break_tool_execution():
    from corecoder.tracing import TraceEvent, Tracer
    from corecoder.tools.base import Tool

    class _FailingTracer(Tracer):
        def emit(self, event: TraceEvent) -> None:
            raise RuntimeError("tracer broke")

    class _Tool(Tool):
        name = "example"
        description = "example tool"
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        def execute(self):
            return "result"

        async def aexecute(self):
            return "result"

    agent = Agent(
        llm=LLM.__new__(LLM),
        tools=[_Tool()],
        tracer=_FailingTracer(),
    )

    class _TC:
        name = "example"
        id = "call-1"
        arguments = {}

    assert agent._exec_tools_parallel([_TC()]) == [
        "result",
    ]

def test_agent_trace_events_share_run_id_per_chat():
    from types import SimpleNamespace

    from corecoder.tracing import InMemoryTracer

    tracer = InMemoryTracer()

    class _FakeLLM:
        def chat(self, **kwargs):
            return SimpleNamespace(
                tool_calls=[],
                message={
                    "role": "assistant",
                    "content": "done",
                },
                content="done",
            )

    agent = Agent(
        llm=_FakeLLM(),
        tools=[],
        tracer=tracer,
    )

    agent.chat("first")

    first_run_ids = {
        event.attributes["run_id"]
        for event in tracer.events
    }

    assert len(first_run_ids) == 1

    first_run_id = next(iter(first_run_ids))

    tracer.events.clear()

    agent.chat("second")

    second_run_ids = {
        event.attributes["run_id"]
        for event in tracer.events
    }

    assert len(second_run_ids) == 1
    assert next(iter(second_run_ids)) != first_run_id

def test_agent_clears_active_run_id_when_tool_execution_is_interrupted():
    from types import SimpleNamespace

    import pytest

    from corecoder.tools.base import Tool

    class _InterruptingTool(Tool):
        name = "interrupting"
        description = "interrupting tool"
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        def execute(self):
            raise KeyboardInterrupt

        async def aexecute(self):
            raise KeyboardInterrupt

    class _FakeLLM:
        def chat(self, **kwargs):
            return SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        name="interrupting",
                        id="call-1",
                        arguments={},
                    )
                ],
                message={
                    "role": "assistant",
                    "content": "",
                },
                content="",
            )

    agent = Agent(
        llm=_FakeLLM(),
        tools=[_InterruptingTool()],
    )

    with pytest.raises(KeyboardInterrupt):
        agent.chat("hello")

    assert agent._active_run_id is None

def test_context_budget_rejects_reserved_output_at_or_above_window():
    import pytest
    with pytest.raises(
        ValueError,
        match="reserved_output_tokens must be less than max_tokens",
    ):
        ContextManager(
            max_tokens=1000,
            reserved_output_tokens=1000,
        )


def test_context_budget_rejects_negative_reserved_output():
    import pytest

    with pytest.raises(
        ValueError,
        match="reserved_output_tokens must be non-negative",
    ):
        ContextManager(
            max_tokens=1000,
            reserved_output_tokens=-1,
        )

def test_context_manager_records_compression_metrics():
    ctx = ContextManager(max_tokens=2000)

    messages = []
    for i in range(20):
        messages.append(
            {
                "role": "user",
                "content": f"msg {i} " + "a" * 200,
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"t{i}",
                "content": "b\n" * 1000,
            }
        )

    before = estimate_tokens(messages)

    assert ctx.maybe_compress(messages, None) is True

    metrics = ctx.last_metrics

    assert metrics.tokens_before == before
    assert metrics.tokens_after == estimate_tokens(messages)
    assert metrics.tokens_saved == (
        metrics.tokens_before - metrics.tokens_after
    )
    assert metrics.tokens_saved > 0

def test_context_manager_records_metrics_without_compression():
    ctx = ContextManager(max_tokens=2000)

    messages = [
        {
            "role": "user",
            "content": "short message",
        }
    ]

    before = estimate_tokens(messages)

    assert ctx.maybe_compress(messages, None) is False

    metrics = ctx.last_metrics

    assert metrics.tokens_before == before
    assert metrics.tokens_after == before
    assert metrics.tokens_saved == 0
    assert metrics.applied_layers == ()

def test_context_metrics_records_applied_compression_layers():
    ctx = ContextManager(max_tokens=3000)

    messages = [
        {
            "role": "tool",
            "tool_call_id": "t1",
            "content": "line\n" * 2000,
        }
    ]

    assert ctx.maybe_compress(messages, None) is True

    assert ctx.last_metrics.applied_layers == (
        "tool_snip",
    )


def test_context_manager_exposes_recent_message_preservation_policy():
    ctx = ContextManager(
        max_tokens=2000,
        keep_recent_messages=6,
    )

    assert ctx.keep_recent_messages == 6

def test_context_summarization_uses_recent_message_policy():
    ctx = ContextManager(
        max_tokens=1000,
        keep_recent_messages=6,
    )

    messages = [
        {
            "role": "user",
            "content": f"msg {i} " + "a" * 200,
        }
        for i in range(12)
    ]

    expected_recent = [
        message["content"]
        for message in messages[-6:]
    ]

    assert ctx.maybe_compress(messages, None) is True

    assert len(messages) == 8
    assert [
        message["content"]
        for message in messages[-6:]
    ] == expected_recent

def test_context_manager_rejects_non_positive_keep_recent_messages():
    import pytest

    with pytest.raises(
        ValueError,
        match="keep_recent_messages must be greater than 0",
    ):
        ContextManager(
            max_tokens=1000,
            keep_recent_messages=0,
        )

def test_context_summarization_trigger_follows_recent_message_policy():
    ctx = ContextManager(
        max_tokens=1000,
        keep_recent_messages=6,
    )

    messages = [
        {
            "role": "user",
            "content": f"msg {i} " + "a" * 250,
        }
        for i in range(9)
    ]

    assert ctx.maybe_compress(messages, None) is True

    assert "summarize" in ctx.last_metrics.applied_layers
    assert len(messages) == 8

def test_context_summarization_preserves_high_priority_message():
    ctx = ContextManager(
        max_tokens=1000,
        keep_recent_messages=4,
    )

    messages = [
        {
            "role": "user",
            "content": "Never modify production credentials.",
            "context_priority": "high",
        }
    ]

    for i in range(10):
        messages.append(
            {
                "role": "user",
                "content": f"old msg {i} " + "a" * 250,
            }
        )

    assert ctx.maybe_compress(messages, None) is True

    assert any(
        message.get("content")
        == "Never modify production credentials."
        for message in messages
    )

def test_agent_full_messages_strips_context_engineering_metadata():
    agent = Agent(
        llm=None,
        tools=[],
    )

    agent.messages.append(
        {
            "role": "user",
            "content": "Never modify production credentials.",
            "context_priority": "high",
        }
    )

    full_messages = agent._full_messages()

    assert agent.messages[0]["context_priority"] == "high"
    assert "context_priority" not in full_messages[1]

def test_context_preserves_high_priority_tool_result_with_tool_call():
    ctx = ContextManager(
        max_tokens=1000,
        keep_recent_messages=4,
    )

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "Critical repository state.",
            "context_priority": "high",
        },
    ]

    for i in range(10):
        messages.append(
            {
                "role": "user",
                "content": f"old msg {i} " + "a" * 250,
            }
        )

    assert ctx.maybe_compress(messages, None) is True

    tool_index = next(
        i
        for i, message in enumerate(messages)
        if message.get("tool_call_id") == "call-1"
    )

    assert messages[tool_index - 1]["role"] == "assistant"
    assert messages[tool_index - 1]["tool_calls"][0]["id"] == "call-1"

def test_context_preserves_entire_tool_group_when_one_result_is_high_priority():
    ctx = ContextManager(
        max_tokens=1000,
        keep_recent_messages=4,
    )

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{}",
                    },
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "git_status",
                        "arguments": "{}",
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "Ordinary file contents.",
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": "Critical repository state.",
            "context_priority": "high",
        },
    ]

    for i in range(10):
        messages.append(
            {
                "role": "user",
                "content": f"old msg {i} " + "a" * 250,
            }
        )

    assert ctx.maybe_compress(messages, None) is True

    assistant_index = next(
        i
        for i, message in enumerate(messages)
        if {
            tool_call.get("id")
            for tool_call in message.get("tool_calls", [])
        }
        == {"call-1", "call-2"}
    )

    assert messages[assistant_index + 1]["tool_call_id"] == "call-1"
    assert messages[assistant_index + 2]["tool_call_id"] == "call-2"

def test_hard_collapse_preserves_high_priority_message():
    ctx = ContextManager(
        max_tokens=1000,
        keep_recent_messages=4,
    )

    critical_content = (
        "Never modify production credentials. "
        + "critical " * 300
    )

    messages = [
        {
            "role": "user",
            "content": critical_content,
            "context_priority": "high",
        }
    ]

    for i in range(10):
        messages.append(
            {
                "role": "user",
                "content": f"msg {i} " + "a" * 250,
            }
        )

    assert ctx.maybe_compress(messages, None) is True

    assert any(
        message.get("content") == critical_content
        for message in messages
    )

def test_hard_collapse_preserves_high_priority_tool_group():
    ctx = ContextManager(
        max_tokens=1000,
        keep_recent_messages=4,
    )

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{}",
                    },
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "git_status",
                        "arguments": "{}",
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "Ordinary file contents.",
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": (
                "Critical repository state. "
                + "critical " * 300
            ),
            "context_priority": "high",
        },
    ]

    for i in range(10):
        messages.append(
            {
                "role": "user",
                "content": f"msg {i} " + "a" * 250,
            }
        )

    assert ctx.maybe_compress(messages, None) is True

    assistant_index = next(
        i
        for i, message in enumerate(messages)
        if {
            tool_call.get("id")
            for tool_call in message.get("tool_calls", [])
        }
        == {"call-1", "call-2"}
    )

    assert messages[assistant_index + 1]["tool_call_id"] == "call-1"
    assert messages[assistant_index + 2]["tool_call_id"] == "call-2"

def test_tool_snip_preserves_high_priority_tool_output():
    ctx = ContextManager(
        max_tokens=4000,
        keep_recent_messages=4,
    )

    tool_output = "important line\n" * 500

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": tool_output,
            "context_priority": "high",
        },
    ]

    assert ctx.maybe_compress(messages, None) is False

    assert messages[1]["content"] == tool_output

def test_context_metrics_records_high_priority_messages_after_compression():
    ctx = ContextManager(
        max_tokens=1000,
        keep_recent_messages=4,
    )

    messages = [
        {
            "role": "user",
            "content": "Never modify production credentials.",
            "context_priority": "high",
        }
    ]

    for i in range(10):
        messages.append(
            {
                "role": "user",
                "content": f"msg {i} " + "a" * 250,
            }
        )

    assert ctx.maybe_compress(messages, None) is True

    assert ctx.last_metrics.high_priority_messages == 1

def test_context_metrics_records_priority_preserved_messages():
    ctx = ContextManager(
        max_tokens=1000,
        keep_recent_messages=4,
    )

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{}",
                    },
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "git_status",
                        "arguments": "{}",
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "Ordinary file contents.",
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": "Critical repository state.",
            "context_priority": "high",
        },
    ]

    for i in range(10):
        messages.append(
            {
                "role": "user",
                "content": f"msg {i} " + "a" * 250,
            }
        )

    assert ctx.maybe_compress(messages, None) is True

    assert ctx.last_metrics.high_priority_messages == 1
    assert ctx.last_metrics.priority_preserved_messages == 3

def test_tool_snip_preserves_entire_high_priority_tool_group():
    ctx = ContextManager(
        max_tokens=4000,
        keep_recent_messages=4,
    )

    sibling_output = "ordinary sibling line\n" * 500

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{}",
                    },
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "git_status",
                        "arguments": "{}",
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": sibling_output,
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": "Critical repository state.",
            "context_priority": "high",
        },
    ]

    ctx.maybe_compress(messages, None)

    assert messages[1]["content"] == sibling_output

def test_eval_case_stores_name_and_prompt():
    from corecoder.eval import EvalCase

    case = EvalCase(
        name="simple-answer",
        prompt="Reply with exactly: done",
    )

    assert case.name == "simple-answer"
    assert case.prompt == "Reply with exactly: done"

def test_eval_result_stores_case_outcome():
    from corecoder.eval import EvalResult

    result = EvalResult(
        case_name="simple-answer",
        success=True,
        output="done",
    )

    assert result.case_name == "simple-answer"
    assert result.success is True
    assert result.output == "done"

def test_eval_result_stores_duration():
    from corecoder.eval import EvalResult

    result = EvalResult(
        case_name="simple-answer",
        success=True,
        output="done",
        duration_ms=12.5,
    )

    assert result.duration_ms == 12.5

def test_eval_case_stores_expected_output():
    from corecoder.eval import EvalCase

    case = EvalCase(
        name="simple-answer",
        prompt="Reply with exactly: done",
        expected_output="done",
    )

    assert case.expected_output == "done"


def test_eval_case_matches_expected_output():
    from corecoder.eval import EvalCase

    case = EvalCase(
        name="simple-answer",
        prompt="Reply with exactly: done",
        expected_output="done",
    )

    assert case.matches("done") is True
    assert case.matches("not done") is False

def test_eval_case_matches_requires_expected_output():
    import pytest

    from corecoder.eval import EvalCase

    case = EvalCase(
        name="unscored-case",
        prompt="Do the task",
    )

    with pytest.raises(
        ValueError,
        match="expected_output is required for exact matching",
    ):
        case.matches("done")

def test_eval_runner_runs_case_and_records_success():
    from corecoder.eval import EvalCase, EvalRunner

    class _FakeAgent:
        def chat(self, prompt):
            assert prompt == "Reply with exactly: done"
            return "done"

    runner = EvalRunner(agent=_FakeAgent())

    case = EvalCase(
        name="simple-answer",
        prompt="Reply with exactly: done",
        expected_output="done",
    )

    result = runner.run_case(case)

    assert result.case_name == "simple-answer"
    assert result.success is True
    assert result.output == "done"
    assert result.duration_ms >= 0

def test_eval_runner_records_failed_case():
    from corecoder.eval import EvalCase, EvalRunner

    class _FakeAgent:
        def chat(self, prompt):
            return "wrong answer"

    runner = EvalRunner(agent=_FakeAgent())

    case = EvalCase(
        name="simple-answer",
        prompt="Reply with exactly: done",
        expected_output="done",
    )

    result = runner.run_case(case)

    assert result.case_name == "simple-answer"
    assert result.success is False
    assert result.output == "wrong answer"
    assert result.duration_ms >= 0

def test_eval_runner_runs_multiple_cases_in_order():
    from corecoder.eval import EvalCase, EvalRunner

    class _FakeAgent:
        def chat(self, prompt):
            return {
                "first prompt": "first",
                "second prompt": "wrong",
            }[prompt]

    runner = EvalRunner(agent=_FakeAgent())

    cases = [
        EvalCase(
            name="first",
            prompt="first prompt",
            expected_output="first",
        ),
        EvalCase(
            name="second",
            prompt="second prompt",
            expected_output="second",
        ),
    ]

    report = runner.run(cases)
    results = report.results

    assert [result.case_name for result in results] == [
        "first",
        "second",
    ]
    assert [result.success for result in results] == [
        True,
        False,
    ]

def test_eval_report_computes_success_rate():
    from corecoder.eval import EvalReport, EvalResult

    report = EvalReport(
        results=[
            EvalResult(
                case_name="case-1",
                success=True,
                output="ok",
            ),
            EvalResult(
                case_name="case-2",
                success=False,
                output="wrong",
            ),
            EvalResult(
                case_name="case-3",
                success=True,
                output="ok",
            ),
        ]
    )

    assert report.success_rate == 2 / 3

def test_eval_report_computes_average_duration():
    from corecoder.eval import EvalReport, EvalResult

    report = EvalReport(
        results=[
            EvalResult(
                case_name="case-1",
                success=True,
                output="ok",
                duration_ms=10.0,
            ),
            EvalResult(
                case_name="case-2",
                success=False,
                output="wrong",
                duration_ms=30.0,
            ),
        ]
    )

    assert report.avg_duration_ms == 20.0

def test_eval_report_handles_empty_results():
    from corecoder.eval import EvalReport

    report = EvalReport(results=[])

    assert report.success_rate == 0.0
    assert report.avg_duration_ms == 0.0

def test_eval_runner_returns_report():
    from corecoder.eval import EvalCase, EvalReport, EvalRunner

    class _FakeAgent:
        def chat(self, prompt):
            return "done"

    runner = EvalRunner(agent=_FakeAgent())

    cases = [
        EvalCase(
            name="simple-answer",
            prompt="Reply with exactly: done",
            expected_output="done",
        )
    ]

    report = runner.run(cases)

    assert isinstance(report, EvalReport)
    assert len(report.results) == 1
    assert report.results[0].success is True
    assert report.success_rate == 1.0

def test_eval_result_stores_llm_rounds():
    from corecoder.eval import EvalResult

    result = EvalResult(
        case_name="simple-answer",
        success=True,
        output="done",
        llm_rounds=3,
    )

    assert result.llm_rounds == 3


def test_eval_runner_records_llm_rounds_from_trace():
    from types import SimpleNamespace

    from corecoder.agent import Agent
    from corecoder.eval import EvalCase, EvalRunner
    from corecoder.tracing import InMemoryTracer

    tracer = InMemoryTracer()

    class _FakeLLM:
        def chat(self, **kwargs):
            return SimpleNamespace(
                tool_calls=[],
                message={
                    "role": "assistant",
                    "content": "done",
                },
                content="done",
            )

    agent = Agent(
        llm=_FakeLLM(),
        tools=[],
        tracer=tracer,
    )
    runner = EvalRunner(agent=agent)

    case = EvalCase(
        name="simple-answer",
        prompt="Reply with exactly: done",
        expected_output="done",
    )

    result = runner.run_case(case)

    assert result.llm_rounds == 1


def test_eval_runner_llm_rounds_do_not_accumulate_across_cases():
    from types import SimpleNamespace

    from corecoder.agent import Agent
    from corecoder.eval import EvalCase, EvalRunner
    from corecoder.tracing import InMemoryTracer

    tracer = InMemoryTracer()

    class _FakeLLM:
        def chat(self, **kwargs):
            return SimpleNamespace(
                tool_calls=[],
                message={
                    "role": "assistant",
                    "content": "done",
                },
                content="done",
            )

    agent = Agent(
        llm=_FakeLLM(),
        tools=[],
        tracer=tracer,
    )
    runner = EvalRunner(agent=agent)

    first = runner.run_case(
        EvalCase(
            name="first",
            prompt="first prompt",
            expected_output="done",
        )
    )
    second = runner.run_case(
        EvalCase(
            name="second",
            prompt="second prompt",
            expected_output="done",
        )
    )

    assert first.llm_rounds == 1
    assert second.llm_rounds == 1

def test_eval_result_stores_tool_calls():
    from corecoder.eval import EvalResult

    result = EvalResult(
        case_name="tool-case",
        success=True,
        output="done",
        tool_calls=2,
    )

    assert result.tool_calls == 2

def test_eval_runner_records_tool_calls_from_trace():
    from types import SimpleNamespace

    from corecoder.agent import Agent
    from corecoder.eval import EvalCase, EvalRunner
    from corecoder.tools.base import Tool
    from corecoder.tracing import InMemoryTracer

    tracer = InMemoryTracer()

    class _Tool(Tool):
        name = "example"
        description = "example tool"
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        def execute(self):
            return "tool result"

    class _FakeLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1

            if self.calls == 1:
                tool_call = SimpleNamespace(
                    name="example",
                    id="call-1",
                    arguments={},
                )
                return SimpleNamespace(
                    tool_calls=[tool_call],
                    message={
                        "role": "assistant",
                        "content": "",
                    },
                    content="",
                )

            return SimpleNamespace(
                tool_calls=[],
                message={
                    "role": "assistant",
                    "content": "done",
                },
                content="done",
            )

    agent = Agent(
        llm=_FakeLLM(),
        tools=[_Tool()],
        tracer=tracer,
    )
    runner = EvalRunner(agent=agent)

    case = EvalCase(
        name="tool-case",
        prompt="Use the example tool",
        expected_output="done",
    )

    result = runner.run_case(case)

    assert result.success is True
    assert result.llm_rounds == 2
    assert result.tool_calls == 1

def test_eval_runner_tool_calls_do_not_accumulate_across_cases():
    from types import SimpleNamespace

    from corecoder.agent import Agent
    from corecoder.eval import EvalCase, EvalRunner
    from corecoder.tools.base import Tool
    from corecoder.tracing import InMemoryTracer

    tracer = InMemoryTracer()

    class _Tool(Tool):
        name = "example"
        description = "example tool"
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        def execute(self):
            return "tool result"

    class _FakeLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1

            if self.calls % 2 == 1:
                tool_call = SimpleNamespace(
                    name="example",
                    id=f"call-{self.calls}",
                    arguments={},
                )
                return SimpleNamespace(
                    tool_calls=[tool_call],
                    message={
                        "role": "assistant",
                        "content": "",
                    },
                    content="",
                )

            return SimpleNamespace(
                tool_calls=[],
                message={
                    "role": "assistant",
                    "content": "done",
                },
                content="done",
            )

    agent = Agent(
        llm=_FakeLLM(),
        tools=[_Tool()],
        tracer=tracer,
    )
    runner = EvalRunner(agent=agent)

    first = runner.run_case(
        EvalCase(
            name="first",
            prompt="first prompt",
            expected_output="done",
        )
    )
    second = runner.run_case(
        EvalCase(
            name="second",
            prompt="second prompt",
            expected_output="done",
        )
    )

    assert first.tool_calls == 1
    assert second.tool_calls == 1

def test_eval_result_stores_context_tokens_saved():
    from corecoder.eval import EvalResult

    result = EvalResult(
        case_name="context-case",
        success=True,
        output="done",
        context_tokens_saved=120,
    )

    assert result.context_tokens_saved == 120

def test_eval_runner_records_context_tokens_saved_from_trace():
    import time

    from corecoder.eval import EvalCase, EvalRunner
    from corecoder.tracing import InMemoryTracer, TraceEvent

    tracer = InMemoryTracer()

    class _FakeAgent:
        def __init__(self):
            self.tracer = tracer

        def chat(self, prompt):
            tracer.emit(
                TraceEvent(
                    name="context.managed",
                    timestamp=time.time(),
                    attributes={
                        "tokens_saved": 40,
                    },
                )
            )
            tracer.emit(
                TraceEvent(
                    name="context.managed",
                    timestamp=time.time(),
                    attributes={
                        "tokens_saved": 80,
                    },
                )
            )
            return "done"

    runner = EvalRunner(agent=_FakeAgent())

    case = EvalCase(
        name="context-case",
        prompt="Do the task",
        expected_output="done",
    )

    result = runner.run_case(case)

    assert result.context_tokens_saved == 120

def test_eval_report_computes_average_context_tokens_saved():
    from corecoder.eval import EvalReport, EvalResult

    report = EvalReport(
        results=[
            EvalResult(
                case_name="case-1",
                success=True,
                output="done",
                context_tokens_saved=100,
            ),
            EvalResult(
                case_name="case-2",
                success=True,
                output="done",
                context_tokens_saved=300,
            ),
        ]
    )

    assert report.avg_context_tokens_saved == 200.0

def test_eval_report_computes_average_llm_rounds():
    from corecoder.eval import EvalReport, EvalResult

    report = EvalReport(
        results=[
            EvalResult(
                case_name="case-1",
                success=True,
                output="done",
                llm_rounds=1,
            ),
            EvalResult(
                case_name="case-2",
                success=True,
                output="done",
                llm_rounds=3,
            ),
        ]
    )

    assert report.avg_llm_rounds == 2.0

def test_eval_report_computes_average_tool_calls():
    from corecoder.eval import EvalReport, EvalResult

    report = EvalReport(
        results=[
            EvalResult(
                case_name="case-1",
                success=True,
                output="done",
                tool_calls=1,
            ),
            EvalResult(
                case_name="case-2",
                success=True,
                output="done",
                tool_calls=3,
            ),
        ]
    )

    assert report.avg_tool_calls == 2.0

def test_eval_runner_records_agent_exception():
    from corecoder.eval import EvalCase, EvalRunner

    class _FailingAgent:
        def chat(self, prompt):
            raise RuntimeError("agent failed")

    runner = EvalRunner(agent=_FailingAgent())

    case = EvalCase(
        name="failing-case",
        prompt="Do the task",
        expected_output="done",
    )

    result = runner.run_case(case)

    assert result.case_name == "failing-case"
    assert result.success is False
    assert result.output == ""
    assert result.error_type == "RuntimeError"
    assert result.duration_ms >= 0

def test_eval_runner_continues_after_failed_case():
    from corecoder.eval import EvalCase, EvalRunner

    class _Agent:
        def chat(self, prompt):
            if prompt == "fail":
                raise RuntimeError("agent failed")
            return "done"

    runner = EvalRunner(agent=_Agent())

    report = runner.run(
        [
            EvalCase(
                name="failing-case",
                prompt="fail",
                expected_output="done",
            ),
            EvalCase(
                name="passing-case",
                prompt="pass",
                expected_output="done",
            ),
        ]
    )

    assert len(report.results) == 2

    assert report.results[0].success is False
    assert report.results[0].error_type == "RuntimeError"

    assert report.results[1].success is True
    assert report.results[1].error_type is None

    assert report.success_rate == 0.5

def test_eval_report_serializes_to_dict():
    from corecoder.eval import EvalReport, EvalResult

    report = EvalReport(
        results=[
            EvalResult(
                case_name="case-1",
                success=True,
                output="done",
                duration_ms=10.0,
                llm_rounds=2,
                tool_calls=1,
                context_tokens_saved=50,
            )
        ]
    )

    data = report.to_dict()

    assert data["success_rate"] == 1.0
    assert data["avg_duration_ms"] == 10.0
    assert data["avg_llm_rounds"] == 2.0
    assert data["avg_tool_calls"] == 1.0
    assert data["avg_context_tokens_saved"] == 50.0

    assert data["results"][0]["case_name"] == "case-1"
    assert data["results"][0]["success"] is True

def test_eval_report_serializes_to_json():
    import json

    from corecoder.eval import EvalReport, EvalResult

    report = EvalReport(
        results=[
            EvalResult(
                case_name="case-1",
                success=True,
                output="done",
                duration_ms=10.0,
                llm_rounds=2,
                tool_calls=1,
                context_tokens_saved=50,
            )
        ]
    )

    payload = report.to_json()
    data = json.loads(payload)

    assert data["success_rate"] == 1.0
    assert data["results"][0]["case_name"] == "case-1"
    assert data["results"][0]["output"] == "done"

def test_eval_report_saves_json_file(tmp_path):
    import json

    from corecoder.eval import EvalReport, EvalResult

    report = EvalReport(
        results=[
            EvalResult(
                case_name="case-1",
                success=True,
                output="done",
            )
        ]
    )

    path = tmp_path / "report.json"

    report.save_json(path)

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["success_rate"] == 1.0
    assert data["results"][0]["case_name"] == "case-1"