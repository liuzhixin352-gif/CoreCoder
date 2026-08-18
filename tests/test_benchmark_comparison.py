"""Tests for baseline versus advanced benchmark comparison."""

import pytest

from corecoder.benchmark import (
    BenchmarkComparisonRunner,
    BenchmarkTarget,
    CORECODER_RUNTIME_BENCHMARK_V1,
)
from corecoder.budget import (
    BudgetLimits,
    BudgetTracker,
)
from corecoder.tracing import (
    InMemoryTracer,
)
from corecoder.permissions import (
    ToolPermission,
    ToolPermissionPolicy,
)
from corecoder.tools.base import Tool
from corecoder.agent import Agent
from corecoder.llm import (
    LLMResponse,
    ToolCall,
)

_EXECUTED_BENCHMARK_TOOL_CALLS: list[str] = []


class BenchmarkTool(Tool):
    """Test-only tool used to prove real Agent execution."""

    name = "benchmark_tool"
    description = "Record one deterministic benchmark tool call."
    permission = ToolPermission.READ

    parameters = {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
            },
        },
        "required": [
            "label",
        ],
    }

    def execute(
        self,
        label: str,
    ) -> str:
        _EXECUTED_BENCHMARK_TOOL_CALLS.append(
            label
        )
        return "ok"

def _tracker_factory() -> BudgetTracker:
    return BudgetTracker(
        BudgetLimits()
    )

class BaselineBenchmarkLLM:
    """Deterministic direct-only baseline LLM."""

    model = "gpt-4o"

    def chat(
        self,
        *,
        messages,
        tools,
        on_token=None,
    ) -> LLMResponse:
        assert messages

        return LLMResponse(
            content="done",
            prompt_tokens=20,
            completion_tokens=5,
            model=self.model,
        )

class DeterministicBenchmarkLLM:
    """Scripted LLM that drives the real Agent runtime."""

    model = "gpt-4o"

    def __init__(self):
        self._tool_call_index = 0

    def _next_tool_call(
        self,
        *,
        label: str,
    ) -> ToolCall:
        self._tool_call_index += 1

        return ToolCall(
            id=(
                "benchmark-call-"
                f"{self._tool_call_index}"
            ),
            name="benchmark_tool",
            arguments={
                "label": label,
            },
        )

    def chat(
        self,
        *,
        messages,
        tools,
        on_token=None,
    ) -> LLMResponse:
        assert tools

        latest_user_index = max(
            index
            for index, message in enumerate(messages)
            if message.get("role") == "user"
        )

        prompt = messages[
            latest_user_index
        ].get(
            "content",
            "",
        )

        tool_results = [
            message
            for message in messages[
                latest_user_index + 1:
            ]
            if message.get("role") == "tool"
        ]

        if "without using tools" in prompt:
            return self._final_response()

        if "exactly one tool" in prompt:
            if not tool_results:
                return self._tool_response(
                    (
                        self._next_tool_call(
                            label="single-tool",
                        ),
                    )
                )

            return self._final_response()

        if "exactly two tools" in prompt:
            if not tool_results:
                return self._tool_response(
                    (
                        self._next_tool_call(
                            label="tool-chain-1",
                        ),
                        self._next_tool_call(
                            label="tool-chain-2",
                        ),
                    )
                )

            return self._final_response()

        raise AssertionError(
            f"unexpected benchmark prompt: {prompt}"
        )

    def _tool_response(
        self,
        tool_calls: tuple[ToolCall, ...],
    ) -> LLMResponse:
        return LLMResponse(
            tool_calls=list(tool_calls),
            prompt_tokens=20,
            completion_tokens=5,
            model=self.model,
        )

    def _final_response(self) -> LLMResponse:
        return LLMResponse(
            content="done",
            prompt_tokens=20,
            completion_tokens=5,
            model=self.model,
        )

class BaselineAgent:
    """Direct-only baseline backed by the real CoreCoder Agent."""

    def __init__(self):
        self.tracer = InMemoryTracer()

        self._agent = Agent(
            llm=BaselineBenchmarkLLM(),
            tools=[],
            tracer=self.tracer,
            permission_policy=(
                ToolPermissionPolicy()
            ),
        )

    def chat(
        self,
        message: str,
        *,
        budget_tracker=None,
    ) -> str:
        assert budget_tracker is not None

        return self._agent.chat(
            message,
            budget_tracker=budget_tracker,
        )

class AdvancedAgent:
    """Benchmark target backed by the real CoreCoder Agent."""

    def __init__(self):
        self.tracer = InMemoryTracer()

        self._agent = Agent(
            llm=DeterministicBenchmarkLLM(),
            tools=[
                BenchmarkTool(),
            ],
            tracer=self.tracer,
            permission_policy=(
                ToolPermissionPolicy()
            ),
        )

    def chat(
        self,
        message: str,
        *,
        budget_tracker=None,
    ) -> str:
        assert budget_tracker is not None

        return self._agent.chat(
            message,
            budget_tracker=budget_tracker,
        )

def test_advanced_agent_executes_real_benchmark_tool():
    _EXECUTED_BENCHMARK_TOOL_CALLS.clear()

    agent = AdvancedAgent()

    result = agent.chat(
        (
            "Use exactly one tool, "
            "then return exactly 'done'."
        ),
        budget_tracker=_tracker_factory(),
    )

    assert result == "done"
    assert _EXECUTED_BENCHMARK_TOOL_CALLS == [
        "single-tool",
    ]

def test_advanced_agent_emits_real_tool_runtime_trace():
    agent = AdvancedAgent()

    result = agent.chat(
        (
            "Use exactly one tool, "
            "then return exactly 'done'."
        ),
        budget_tracker=_tracker_factory(),
    )

    assert result == "done"

    event_names = [
        event.name
        for event in agent.tracer.events
    ]

    assert event_names.count(
        "llm.started"
    ) == 2
    assert event_names.count(
        "tool.permission"
    ) == 1
    assert event_names.count(
        "tool.started"
    ) == 1
    assert event_names.count(
        "tool.completed"
    ) == 1

    permission_event = next(
        event
        for event in agent.tracer.events
        if event.name == "tool.permission"
    )

    assert (
        permission_event.attributes[
            "tool_name"
        ]
        == "benchmark_tool"
    )
    assert (
        permission_event.attributes[
            "decision"
        ]
        == "allow"
    )


def test_comparison_runs_same_suite_for_all_targets():
    comparison = BenchmarkComparisonRunner(
        suite=CORECODER_RUNTIME_BENCHMARK_V1,
        targets=(
            BenchmarkTarget(
                name="baseline",
                agent_factory=BaselineAgent,
                budget_tracker_factory=(
                    _tracker_factory
                ),
            ),
            BenchmarkTarget(
                name="advanced",
                agent_factory=AdvancedAgent,
                budget_tracker_factory=(
                    _tracker_factory
                ),
            ),
        ),
    ).run()

    assert comparison.suite_name == (
        "corecoder-runtime"
    )
    assert comparison.suite_version == "1"

    assert [
        target_run.target_name
        for target_run in comparison.runs
    ] == [
        "baseline",
        "advanced",
    ]

    for target_run in comparison.runs:
        assert target_run.run.suite_name == (
            comparison.suite_name
        )
        assert target_run.run.suite_version == (
            comparison.suite_version
        )


def test_comparison_exposes_baseline_and_advanced_gap():
    comparison = BenchmarkComparisonRunner(
        suite=CORECODER_RUNTIME_BENCHMARK_V1,
        targets=(
            BenchmarkTarget(
                name="baseline",
                agent_factory=BaselineAgent,
                budget_tracker_factory=(
                    _tracker_factory
                ),
            ),
            BenchmarkTarget(
                name="advanced",
                agent_factory=AdvancedAgent,
                budget_tracker_factory=(
                    _tracker_factory
                ),
            ),
        ),
    ).run()

    baseline = comparison.runs[0].run.report
    advanced = comparison.runs[1].run.report

    assert baseline.success_rate == pytest.approx(
        1 / 3
    )
    assert advanced.success_rate == pytest.approx(
        1.0
    )

    assert [
        result.success
        for result in baseline.results
    ] == [
        True,
        False,
        False,
    ]

    assert [
        result.success
        for result in advanced.results
    ] == [
        True,
        True,
        True,
    ]


def test_comparison_preserves_usage_metrics_per_target():
    comparison = BenchmarkComparisonRunner(
        suite=CORECODER_RUNTIME_BENCHMARK_V1,
        targets=(
            BenchmarkTarget(
                name="baseline",
                agent_factory=BaselineAgent,
                budget_tracker_factory=(
                    _tracker_factory
                ),
            ),
            BenchmarkTarget(
                name="advanced",
                agent_factory=AdvancedAgent,
                budget_tracker_factory=(
                    _tracker_factory
                ),
            ),
        ),
    ).run()

    baseline = comparison.runs[0].run.report
    advanced = comparison.runs[1].run.report

    assert baseline.total_tokens == 75
    assert baseline.total_cost_usd == pytest.approx(
        0.0003
    )

    assert advanced.total_tokens == 125
    assert advanced.total_cost_usd == pytest.approx(
        0.0005
    )


def test_comparison_creates_fresh_agents_for_each_run():
    created = []

    def factory():
        agent = BaselineAgent()
        created.append(agent)
        return agent

    runner = BenchmarkComparisonRunner(
        suite=CORECODER_RUNTIME_BENCHMARK_V1,
        targets=(
            BenchmarkTarget(
                name="baseline",
                agent_factory=factory,
                budget_tracker_factory=(
                    _tracker_factory
                ),
            ),
        ),
    )

    runner.run()
    runner.run()

    assert len(created) == 2
    assert created[0] is not created[1]


def test_comparison_rejects_duplicate_target_names():
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        BenchmarkComparisonRunner(
            suite=CORECODER_RUNTIME_BENCHMARK_V1,
            targets=(
                BenchmarkTarget(
                    name="same",
                    agent_factory=BaselineAgent,
                ),
                BenchmarkTarget(
                    name="same",
                    agent_factory=AdvancedAgent,
                ),
            ),
        )


def test_comparison_serializes_named_runs():
    comparison = BenchmarkComparisonRunner(
        suite=CORECODER_RUNTIME_BENCHMARK_V1,
        targets=(
            BenchmarkTarget(
                name="baseline",
                agent_factory=BaselineAgent,
                budget_tracker_factory=(
                    _tracker_factory
                ),
            ),
            BenchmarkTarget(
                name="advanced",
                agent_factory=AdvancedAgent,
                budget_tracker_factory=(
                    _tracker_factory
                ),
            ),
        ),
    ).run()

    payload = comparison.to_dict()

    assert payload["suite_name"] == (
        "corecoder-runtime"
    )
    assert payload["suite_version"] == "1"

    assert [
        item["target_name"]
        for item in payload["runs"]
    ] == [
        "baseline",
        "advanced",
    ]

    assert (
        payload["runs"][0]["run"]["report"][
            "success_rate"
        ]
        == pytest.approx(1 / 3)
    )

    assert (
        payload["runs"][1]["run"]["report"][
            "success_rate"
        ]
        == pytest.approx(1.0)
    )