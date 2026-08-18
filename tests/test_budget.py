"""Tests for token and cost budget tracking."""

import pytest

from corecoder.budget import (
    BudgetExceededError,
    BudgetLimits,
    BudgetTracker,
)


def test_budget_tracker_accumulates_usage():
    tracker = BudgetTracker(
        BudgetLimits(),
    )

    tracker.record(
        prompt_tokens=100,
        completion_tokens=20,
        cost_usd=0.01,
    )
    tracker.record(
        prompt_tokens=50,
        completion_tokens=10,
        cost_usd=0.02,
    )

    assert tracker.usage.prompt_tokens == 150
    assert tracker.usage.completion_tokens == 30
    assert tracker.usage.total_tokens == 180
    assert tracker.usage.cost_usd == pytest.approx(
        0.03,
    )


def test_budget_tracker_allows_usage_at_limit():
    tracker = BudgetTracker(
        BudgetLimits(
            max_prompt_tokens=100,
            max_completion_tokens=50,
            max_total_tokens=150,
            max_cost_usd=0.05,
        )
    )

    tracker.record(
        prompt_tokens=100,
        completion_tokens=50,
        cost_usd=0.05,
    )

    assert tracker.usage.total_tokens == 150


@pytest.mark.parametrize(
    (
        "limits",
        "usage",
        "resource",
    ),
    [
        (
            BudgetLimits(
                max_prompt_tokens=99,
            ),
            {
                "prompt_tokens": 100,
            },
            "prompt_tokens",
        ),
        (
            BudgetLimits(
                max_completion_tokens=49,
            ),
            {
                "completion_tokens": 50,
            },
            "completion_tokens",
        ),
        (
            BudgetLimits(
                max_total_tokens=149,
            ),
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
            "total_tokens",
        ),
        (
            BudgetLimits(
                max_cost_usd=0.04,
            ),
            {
                "cost_usd": 0.05,
            },
            "cost_usd",
        ),
    ],
)
def test_budget_tracker_rejects_usage_above_limit(
    limits,
    usage,
    resource,
):
    tracker = BudgetTracker(limits)

    with pytest.raises(
        BudgetExceededError,
    ) as exc_info:
        tracker.record(**usage)

    assert exc_info.value.resource == resource


@pytest.mark.parametrize(
    (
        "field",
        "value",
    ),
    [
        ("max_prompt_tokens", -1),
        ("max_completion_tokens", -1),
        ("max_total_tokens", -1),
        ("max_cost_usd", -0.01),
    ],
)
def test_budget_limits_reject_negative_values(
    field,
    value,
):
    with pytest.raises(
        ValueError,
        match="must be non-negative",
    ):
        BudgetLimits(
            **{
                field: value,
            }
        )


@pytest.mark.parametrize(
    (
        "usage",
        "message",
    ),
    [
        (
            {"prompt_tokens": -1},
            "prompt_tokens must be non-negative",
        ),
        (
            {"completion_tokens": -1},
            "completion_tokens must be non-negative",
        ),
        (
            {"cost_usd": -0.01},
            "cost_usd must be non-negative",
        ),
    ],
)
def test_budget_tracker_rejects_negative_usage(
    usage,
    message,
):
    tracker = BudgetTracker(
        BudgetLimits(),
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        tracker.record(**usage)

def test_budget_tracker_reports_remaining_cost():
    tracker = BudgetTracker(
        BudgetLimits(
            max_cost_usd=1.0,
        )
    )

    tracker.record(
        cost_usd=0.25,
    )

    assert tracker.remaining_cost_usd == pytest.approx(
        0.75,
    )


def test_budget_tracker_remaining_cost_is_none_without_cost_limit():
    tracker = BudgetTracker(
        BudgetLimits()
    )

    assert tracker.remaining_cost_usd is None
