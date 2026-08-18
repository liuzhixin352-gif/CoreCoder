"""Tests for evaluation token and cost metrics."""

import pytest

from corecoder.budget import (
    BudgetLimits,
    BudgetTracker,
)
from corecoder.eval import (
    EvalCase,
    EvalRunner,
)


def _new_tracker() -> BudgetTracker:
    return BudgetTracker(
        BudgetLimits()
    )


class BudgetRecordingAgent:
    """Minimal agent that records deterministic usage."""

    def __init__(self):
        self.trackers = []

    def chat(
        self,
        message: str,
        *,
        budget_tracker=None,
    ) -> str:
        assert message
        assert budget_tracker is not None

        self.trackers.append(
            budget_tracker
        )

        budget_tracker.record(
            prompt_tokens=100,
            completion_tokens=20,
            cost_usd=0.001,
        )

        return "done"


def test_eval_runner_records_token_and_cost_metrics():
    agent = BudgetRecordingAgent()

    runner = EvalRunner(
        agent=agent,
        budget_tracker_factory=_new_tracker,
    )

    result = runner.run_case(
        EvalCase(
            name="usage-case",
            prompt="do the task",
            expected_output="done",
        )
    )

    assert result.success is True
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 20
    assert result.total_tokens == 120
    assert result.cost_usd == pytest.approx(
        0.001
    )


def test_eval_runner_uses_fresh_budget_tracker_per_case():
    agent = BudgetRecordingAgent()

    runner = EvalRunner(
        agent=agent,
        budget_tracker_factory=_new_tracker,
    )

    report = runner.run(
        [
            EvalCase(
                name="first",
                prompt="first task",
                expected_output="done",
            ),
            EvalCase(
                name="second",
                prompt="second task",
                expected_output="done",
            ),
        ]
    )

    assert len(agent.trackers) == 2
    assert (
        agent.trackers[0]
        is not agent.trackers[1]
    )

    assert report.total_prompt_tokens == 200
    assert report.total_completion_tokens == 40
    assert report.total_tokens == 240
    assert report.total_cost_usd == pytest.approx(
        0.002
    )
    assert report.avg_total_tokens == pytest.approx(
        120.0
    )
    assert report.avg_cost_usd == pytest.approx(
        0.001
    )


def test_eval_report_serializes_usage_metrics():
    runner = EvalRunner(
        agent=BudgetRecordingAgent(),
        budget_tracker_factory=_new_tracker,
    )

    report = runner.run(
        [
            EvalCase(
                name="usage-case",
                prompt="do the task",
                expected_output="done",
            )
        ]
    )

    payload = report.to_dict()

    assert payload["total_prompt_tokens"] == 100
    assert payload["total_completion_tokens"] == 20
    assert payload["total_tokens"] == 120
    assert payload["total_cost_usd"] == pytest.approx(
        0.001
    )
    assert payload["avg_total_tokens"] == pytest.approx(
        120.0
    )
    assert payload["avg_cost_usd"] == pytest.approx(
        0.001
    )

    result = payload["results"][0]

    assert result["prompt_tokens"] == 100
    assert result["completion_tokens"] == 20
    assert result["total_tokens"] == 120
    assert result["cost_usd"] == pytest.approx(
        0.001
    )