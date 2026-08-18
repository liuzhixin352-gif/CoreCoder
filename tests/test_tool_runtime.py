import asyncio
import threading

from corecoder.tool_runtime import run_tool_calls
from corecoder.tools.base import Tool

def test_run_tool_calls_runs_sync_work_concurrently_and_preserves_order():
    barrier = threading.Barrier(2)

    def execute(name: str) -> str:
        barrier.wait(timeout=1)
        return f"{name}-result"

    async def scenario():
        return await run_tool_calls(
            ["first", "second"],
            execute,
        )

    assert asyncio.run(scenario()) == [
        "first-result",
        "second-result",
    ]

def test_run_tool_calls_awaits_async_work():
    async def execute(name: str) -> str:
        await asyncio.sleep(0)
        return f"{name}-result"

    async def scenario():
        return await run_tool_calls(
            ["first", "second"],
            execute,
        )

    assert asyncio.run(scenario()) == [
        "first-result",
        "second-result",
    ]

def test_run_tool_calls_respects_max_concurrency():
    active = 0
    peak = 0
    lock = asyncio.Lock()

    async def execute(name: str) -> str:
        nonlocal active, peak

        async with lock:
            active += 1
            peak = max(peak, active)

        await asyncio.sleep(0.05)

        async with lock:
            active -= 1

        return f"{name}-result"

    async def scenario():
        return await run_tool_calls(
            ["a", "b", "c", "d"],
            execute,
            max_concurrency=2,
        )

    assert asyncio.run(scenario()) == [
        "a-result",
        "b-result",
        "c-result",
        "d-result",
    ]
    assert peak == 2


def test_tool_aexecute_bridges_sync_execute():
    class _EchoTool(Tool):
        name = "echo"
        description = "echo test tool"
        parameters = {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
            },
            "required": ["value"],
        }

        def execute(self, value: str) -> str:
            return value

    async def scenario():
        return await _EchoTool().aexecute(
            value="hello",
        )

    assert asyncio.run(scenario()) == "hello"

def test_run_tool_calls_rejects_zero_max_concurrency():
    def execute(name: str) -> str:
        return name

    async def scenario():
        return await run_tool_calls(
            [],
            execute,
            max_concurrency=0,
        )

    try:
        asyncio.run(scenario())
    except ValueError:
        return

    raise AssertionError("expected ValueError")
