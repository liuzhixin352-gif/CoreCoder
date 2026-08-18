"""Tests for reproducible CoreCoder benchmarks."""

import pytest

from corecoder.benchmark import (
    BenchmarkCase,
    BenchmarkRunner,
    BenchmarkSuite,
    CORECODER_RUNTIME_BENCHMARK_V1,
)
from corecoder.budget import (
    BudgetLimits,
    BudgetTracker,
)
from corecoder.eval import EvalResult
from corecoder.tracing import (
    InMemoryTracer,
    TraceEvent,
)


def test_runtime_benchmark_is_versioned_and_reproducible():
    suite = (
        CORECODER_RUNTIME_BENCHMARK_V1
    )

    assert suite.name == "corecoder-runtime"
    assert suite.version == "1"

    assert [
        case.name
        for case in suite.cases
    ] == [
        "direct-response",
        "single-tool",
        "tool-chain",
    ]

    assert suite.categories == (
        "direct",
        "tool-use",
    )


def test_benchmark_case_requires_behavioral_expectations():
    case = BenchmarkCase(
        name="tool-case",
        category="tool-use",
        prompt="use a tool",
        expected_output="done",
        expected_llm_rounds=2,
        expected_tool_calls=1,
    )

    correct = EvalResult(
        case_name="tool-case",
        success=True,
        output="done",
        llm_rounds=2,
        tool_calls=1,
    )

    skipped_tool = EvalResult(
        case_name="tool-case",
        success=True,
        output="done",
        llm_rounds=1,
        tool_calls=0,
    )

    assert case.matches_result(
        correct
    ) is True

    assert case.matches_result(
        skipped_tool
    ) is False


def test_benchmark_suite_rejects_duplicate_case_names():
    case = BenchmarkCase(
        name="duplicate",
        category="direct",
        prompt="return done",
        expected_output="done",
    )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        BenchmarkSuite(
            name="suite",
            version="1",
            cases=(
                case,
                case,
            ),
        )


def test_benchmark_case_rejects_negative_expectations():
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        BenchmarkCase(
            name="bad-case",
            category="tool-use",
            prompt="do something",
            expected_output="done",
            expected_tool_calls=-1,
        )


def _tracker_factory() -> BudgetTracker:
    return BudgetTracker(
        BudgetLimits()
    )


class ScriptedBenchmarkAgent:
    """Deterministic agent used to exercise the benchmark runner."""

    def __init__(self):
        self.tracer = InMemoryTracer()
        self.calls = 0

    def _emit(
        self,
        name: str,
    ) -> None:
        self.tracer.emit(
            TraceEvent(
                name=name,
                timestamp=0.0,
            )
        )

    def chat(
        self,
        message: str,
        *,
        budget_tracker=None,
    ) -> str:
        assert budget_tracker is not None

        self.calls += 1

        if "without using tools" in message:
            self._emit(
                "llm.started"
            )

            budget_tracker.record(
                prompt_tokens=10,
                completion_tokens=2,
                cost_usd=0.0001,
            )

            return "done"

        if "exactly one tool" in message:
            self._emit(
                "llm.started"
            )
            self._emit(
                "tool.started"
            )
            self._emit(
                "llm.started"
            )

            budget_tracker.record(
                prompt_tokens=20,
                completion_tokens=4,
                cost_usd=0.0002,
            )

            return "done"

        self._emit(
            "llm.started"
        )
        self._emit(
            "tool.started"
        )
        self._emit(
            "tool.started"
        )
        self._emit(
            "llm.started"
        )

        budget_tracker.record(
            prompt_tokens=30,
            completion_tokens=6,
            cost_usd=0.0003,
        )

        return "done"


def test_benchmark_runner_produces_reproducible_report():
    runner = BenchmarkRunner(
        suite=(
            CORECODER_RUNTIME_BENCHMARK_V1
        ),
        agent=ScriptedBenchmarkAgent(),
        budget_tracker_factory=(
            _tracker_factory
        ),
    )

    run = runner.run()

    assert run.suite_name == (
        "corecoder-runtime"
    )
    assert run.suite_version == "1"

    assert run.report.success_rate == pytest.approx(
        1.0
    )

    assert run.report.total_prompt_tokens == 60
    assert (
        run.report.total_completion_tokens
        == 12
    )
    assert run.report.total_tokens == 72
    assert run.report.total_cost_usd == pytest.approx(
        0.0006
    )

    assert [
        result.tool_calls
        for result in run.report.results
    ] == [
        0,
        1,
        2,
    ]


def test_benchmark_run_serializes_suite_metadata():
    run = BenchmarkRunner(
        suite=(
            CORECODER_RUNTIME_BENCHMARK_V1
        ),
        agent=ScriptedBenchmarkAgent(),
        budget_tracker_factory=(
            _tracker_factory
        ),
    ).run()

    payload = run.to_dict()

    assert payload["suite_name"] == (
        "corecoder-runtime"
    )
    assert payload["suite_version"] == "1"

    assert (
        payload["report"]["success_rate"]
        == pytest.approx(1.0)
    )
    assert (
        payload["report"]["total_tokens"]
        == 72
    )