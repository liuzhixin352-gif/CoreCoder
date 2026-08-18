"""Tests for per-response LLM cost estimation."""

import pytest

from corecoder.llm import estimate_cost_usd


def test_estimate_cost_usd_for_known_model():
    cost = estimate_cost_usd(
        "gpt-4o-mini",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )

    assert cost == pytest.approx(
        0.75,
    )


def test_estimate_cost_usd_scales_token_usage():
    cost = estimate_cost_usd(
        "gpt-4o-mini",
        prompt_tokens=1000,
        completion_tokens=500,
    )

    assert cost == pytest.approx(
        0.00045,
    )


def test_estimate_cost_usd_returns_none_for_unknown_model():
    cost = estimate_cost_usd(
        "unknown-model",
        prompt_tokens=100,
        completion_tokens=50,
    )

    assert cost is None


def test_estimate_cost_usd_allows_zero_usage():
    cost = estimate_cost_usd(
        "gpt-4o-mini",
        prompt_tokens=0,
        completion_tokens=0,
    )

    assert cost == pytest.approx(
        0.0,
    )


@pytest.mark.parametrize(
    (
        "prompt_tokens",
        "completion_tokens",
        "message",
    ),
    [
        (
            -1,
            0,
            "prompt_tokens must be non-negative",
        ),
        (
            0,
            -1,
            "completion_tokens must be non-negative",
        ),
    ],
)
def test_estimate_cost_usd_rejects_negative_usage(
    prompt_tokens,
    completion_tokens,
    message,
):
    with pytest.raises(
        ValueError,
        match=message,
    ):
        estimate_cost_usd(
            "gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def test_llm_estimated_cost_uses_accumulated_usage():
    from corecoder.llm import LLM

    llm = object.__new__(LLM)
    llm.model = "gpt-4o-mini"
    llm.total_prompt_tokens = 1000
    llm.total_completion_tokens = 500

    assert llm.estimated_cost == pytest.approx(
        0.00045,
    )
