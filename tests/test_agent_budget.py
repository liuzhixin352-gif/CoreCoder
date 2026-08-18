"""Tests for per-run agent token and cost budgets."""

import pytest

from corecoder.agent import Agent
from corecoder.budget import (
    BudgetExceededError,
    BudgetLimits,
    BudgetPricingUnavailableError,
    BudgetTracker,
)
from corecoder.llm import (
    LLMResponse,
    ToolCall,
)
from corecoder.permissions import ToolPermission
from corecoder.tools.base import Tool
from corecoder.tracing import InMemoryTracer
from corecoder.model_catalog import (
    ModelCatalog,
    ModelProfile,
)
from corecoder.model_router import (
    CapabilityModelRouter,
    RouteRequest,
)
from corecoder.routed_llm import RoutedLLM

class FakeLLM:
    model = "gpt-4o-mini"

    def __init__(
        self,
        responses: list[LLMResponse],
    ):
        self.responses = list(responses)
        self.calls = 0

    def chat(
        self,
        messages,
        tools=None,
        on_token=None,
    ):
        response = self.responses[self.calls]
        self.calls += 1
        return response


class CountingTool(Tool):
    name = "count"
    description = "Count tool executions."
    permission = ToolPermission.READ
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self):
        self.calls = 0

    def execute(self) -> str:
        self.calls += 1
        return "counted"


def test_agent_records_response_usage_in_budget():
    llm = FakeLLM(
        [
            LLMResponse(
                content="done",
                prompt_tokens=1000,
                completion_tokens=500,
            )
        ]
    )

    agent = Agent(
        llm=llm,
        tools=[],
        budget_limits=BudgetLimits(
            max_total_tokens=2000,
            max_cost_usd=0.001,
        ),
    )

    result = agent.chat("hello")

    assert result == "done"

    usage = agent.last_budget_usage

    assert usage is not None
    assert usage.prompt_tokens == 1000
    assert usage.completion_tokens == 500
    assert usage.total_tokens == 1500
    assert usage.cost_usd == pytest.approx(
        0.00045,
    )


def test_agent_uses_fresh_budget_for_each_run():
    llm = FakeLLM(
        [
            LLMResponse(
                content="first",
                prompt_tokens=6,
                completion_tokens=1,
            ),
            LLMResponse(
                content="second",
                prompt_tokens=6,
                completion_tokens=1,
            ),
        ]
    )

    agent = Agent(
        llm=llm,
        tools=[],
        budget_limits=BudgetLimits(
            max_total_tokens=10,
        ),
    )

    assert agent.chat("first") == "first"
    assert agent.last_budget_usage is not None
    assert agent.last_budget_usage.total_tokens == 7

    assert agent.chat("second") == "second"
    assert agent.last_budget_usage is not None
    assert agent.last_budget_usage.total_tokens == 7


def test_agent_stops_before_tool_execution_when_budget_exceeded():
    tool = CountingTool()

    llm = FakeLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="count",
                        arguments={},
                    )
                ],
                prompt_tokens=8,
                completion_tokens=3,
            )
        ]
    )

    agent = Agent(
        llm=llm,
        tools=[tool],
        budget_limits=BudgetLimits(
            max_total_tokens=10,
        ),
    )

    with pytest.raises(
        BudgetExceededError,
    ) as exc_info:
        agent.chat("use the tool")

    assert exc_info.value.resource == "total_tokens"
    assert llm.calls == 1
    assert tool.calls == 0

    assert agent.last_budget_usage is not None
    assert agent.last_budget_usage.total_tokens == 11


def test_agent_enforces_cost_budget():
    llm = FakeLLM(
        [
            LLMResponse(
                content="done",
                prompt_tokens=1000,
                completion_tokens=500,
            )
        ]
    )

    agent = Agent(
        llm=llm,
        tools=[],
        budget_limits=BudgetLimits(
            max_cost_usd=0.0004,
        ),
    )

    with pytest.raises(
        BudgetExceededError,
    ) as exc_info:
        agent.chat("hello")

    assert exc_info.value.resource == "cost_usd"
    assert llm.calls == 1

    assert agent.last_budget_usage is not None
    assert agent.last_budget_usage.cost_usd == pytest.approx(
        0.00045,
    )


def test_agent_cost_budget_rejects_unpriced_model_before_call():
    class UnknownModelLLM(FakeLLM):
        model = "unknown-model"

    llm = UnknownModelLLM(
        [
            LLMResponse(
                content="should not run",
            )
        ]
    )

    agent = Agent(
        llm=llm,
        tools=[],
        budget_limits=BudgetLimits(
            max_cost_usd=1.0,
        ),
    )

    with pytest.raises(
        BudgetPricingUnavailableError,
        match="Cost budget requires pricing",
    ):
        agent.chat("hello")

    assert llm.calls == 0


def test_agent_accepts_shared_budget_tracker():
    llm = FakeLLM(
        [
            LLMResponse(
                content="first",
                prompt_tokens=4,
                completion_tokens=1,
            ),
            LLMResponse(
                content="second",
                prompt_tokens=4,
                completion_tokens=1,
            ),
        ]
    )

    tracker = BudgetTracker(
        BudgetLimits(
            max_total_tokens=8,
        )
    )

    agent = Agent(
        llm=llm,
        tools=[],
    )

    assert (
        agent.chat(
            "first",
            budget_tracker=tracker,
        )
        == "first"
    )

    with pytest.raises(
        BudgetExceededError,
    ):
        agent.chat(
            "second",
            budget_tracker=tracker,
        )

    assert tracker.usage.total_tokens == 10


def test_agent_traces_budget_usage():
    tracer = InMemoryTracer()

    llm = FakeLLM(
        [
            LLMResponse(
                content="done",
                prompt_tokens=1000,
                completion_tokens=500,
            )
        ]
    )

    agent = Agent(
        llm=llm,
        tools=[],
        tracer=tracer,
        budget_limits=BudgetLimits(
            max_total_tokens=2000,
            max_cost_usd=0.001,
        ),
    )

    result = agent.chat(
        "hello",
        run_id="budget-run",
    )

    assert result == "done"

    events = tracer.events_for_run(
        "budget-run",
    )

    budget_events = [event for event in events if event.name == "budget.updated"]

    assert len(budget_events) == 1

    attributes = budget_events[0].attributes

    assert attributes["prompt_tokens"] == 1000
    assert attributes["completion_tokens"] == 500
    assert attributes["total_tokens"] == 1500
    assert attributes["cost_usd"] == pytest.approx(
        0.00045,
    )
    assert attributes["max_total_tokens"] == 2000
    assert attributes["max_cost_usd"] == pytest.approx(
        0.001,
    )


def test_agent_traces_budget_exceeded():
    tracer = InMemoryTracer()

    llm = FakeLLM(
        [
            LLMResponse(
                content="too expensive",
                prompt_tokens=8,
                completion_tokens=3,
            )
        ]
    )

    agent = Agent(
        llm=llm,
        tools=[],
        tracer=tracer,
        budget_limits=BudgetLimits(
            max_total_tokens=10,
        ),
    )

    with pytest.raises(
        BudgetExceededError,
    ):
        agent.chat(
            "hello",
            run_id="budget-run",
        )

    events = tracer.events_for_run(
        "budget-run",
    )

    exceeded = [event for event in events if event.name == "budget.exceeded"]

    assert len(exceeded) == 1

    attributes = exceeded[0].attributes

    assert attributes["resource"] == "total_tokens"
    assert attributes["limit"] == 10
    assert attributes["actual"] == 11
    assert attributes["total_tokens"] == 11

def test_agent_does_not_start_next_llm_call_when_budget_is_exhausted():
    tool = CountingTool()

    llm = FakeLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="count",
                        arguments={},
                    )
                ],
                prompt_tokens=8,
                completion_tokens=2,
            ),
            LLMResponse(
                content="must not run",
                prompt_tokens=1,
                completion_tokens=1,
            ),
        ]
    )

    agent = Agent(
        llm=llm,
        tools=[tool],
        budget_limits=BudgetLimits(
            max_total_tokens=10,
        ),
    )

    with pytest.raises(
        BudgetExceededError,
    ) as exc_info:
        agent.chat(
            "use the tool",
        )

    assert exc_info.value.resource == "total_tokens"

    assert llm.calls == 1
    assert tool.calls == 1

    assert agent.last_budget_usage is not None
    assert (
        agent.last_budget_usage.total_tokens
        == 10
    )

def test_agent_zero_token_budget_blocks_first_llm_call():
    llm = FakeLLM(
        [
            LLMResponse(
                content="must not run",
                prompt_tokens=1,
            )
        ]
    )

    agent = Agent(
        llm=llm,
        tools=[],
        budget_limits=BudgetLimits(
            max_total_tokens=0,
        ),
    )

    with pytest.raises(
        BudgetExceededError,
    ) as exc_info:
        agent.chat("hello")

    assert exc_info.value.resource == "total_tokens"
    assert llm.calls == 0

def test_agent_cost_uses_response_model():
    class RoutedFakeLLM:
        model = "gpt-4o"

        @property
        def routable_models(self):
            return (
                "gpt-4o",
                "gpt-4o-mini",
            )

        def chat(
            self,
            messages,
            tools=None,
            on_token=None,
        ):
            del messages
            del tools
            del on_token

            return LLMResponse(
                content="fallback result",
                prompt_tokens=1000,
                completion_tokens=500,
                model="gpt-4o-mini",
            )

    agent = Agent(
        llm=RoutedFakeLLM(),
        tools=[],
        budget_limits=BudgetLimits(
            max_cost_usd=0.001,
        ),
    )

    result = agent.chat(
        "hello",
    )

    assert result == "fallback result"

    assert agent.last_budget_usage is not None
    assert agent.last_budget_usage.cost_usd == pytest.approx(
        0.00045,
    )

def test_agent_cost_budget_rejects_unpriced_routed_model():
    class RoutedFakeLLM:
        def __init__(self):
            self.calls = 0

        @property
        def routable_models(self):
            return (
                "gpt-4o-mini",
                "custom-unpriced-model",
            )

        def chat(
            self,
            messages,
            tools=None,
            on_token=None,
        ):
            del messages
            del tools
            del on_token

            self.calls += 1

            return LLMResponse(
                content="must not run",
            )

    llm = RoutedFakeLLM()

    agent = Agent(
        llm=llm,
        tools=[],
        budget_limits=BudgetLimits(
            max_cost_usd=1.0,
        ),
    )

    with pytest.raises(
        BudgetPricingUnavailableError,
        match="custom-unpriced-model",
    ):
        agent.chat(
            "hello",
        )

    assert llm.calls == 0

def test_agent_reroutes_as_remaining_cost_budget_shrinks():
    tool = CountingTool()

    expensive = FakeLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="count",
                        arguments={},
                    )
                ],
                prompt_tokens=7800,
                completion_tokens=0,
            )
        ]
    )
    expensive.model = "gpt-4o"

    cheap = FakeLLM(
        [
            LLMResponse(
                content="done",
                prompt_tokens=1000,
                completion_tokens=500,
            )
        ]
    )
    cheap.model = "gpt-4o-mini"

    catalog = ModelCatalog(
        [
            ModelProfile(
                name="gpt-4o",
            ),
            ModelProfile(
                name="gpt-4o-mini",
            ),
        ]
    )

    routed_llm = RoutedLLM(
        router=CapabilityModelRouter(
            [
                "gpt-4o",
                "gpt-4o-mini",
            ],
            catalog,
        ),
        backends={
            "gpt-4o": expensive,
            "gpt-4o-mini": cheap,
        },
        route_request=RouteRequest(),
    )

    agent = Agent(
        llm=routed_llm,
        tools=[
            tool,
        ],
        budget_limits=BudgetLimits(
            max_cost_usd=0.020,
        ),
    )

    result = agent.chat(
        "use the tool and finish",
    )

    assert result == "done"

    assert expensive.calls == 1
    assert cheap.calls == 1
    assert tool.calls == 1

    assert agent.last_budget_usage is not None
    assert agent.last_budget_usage.cost_usd == pytest.approx(
        0.01995,
    )

    # Per-call routing context must not mutate
    # the shared default request.
    assert (
        routed_llm.route_request.remaining_cost_usd
        is None
    )