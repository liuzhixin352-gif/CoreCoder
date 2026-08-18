"""Tests for routed LLM execution and fallback."""

import pytest

from corecoder.llm import LLMResponse
from corecoder.model_router import (
    RouteRequest,
    StaticModelRouter,
    CapabilityModelRouter,
)
from corecoder.routed_llm import (
    ModelBackendUnavailableError,
    RoutedLLM,
    is_transient_model_error,
)
from corecoder.budget import (
    BudgetExceededError,
)
from corecoder.tracing import (
    InMemoryTracer,
    TraceEvent,
    Tracer,
)
from corecoder.model_catalog import (
    ModelCatalog,
    ModelProfile,
)

class TransientProviderError(RuntimeError):
    status_code = 503


class FakeBackend:
    def __init__(
        self,
        model: str,
        outcomes,
    ):
        self.model = model
        self.outcomes = list(outcomes)
        self.calls = 0
        self.received_messages = None
        self.received_tools = None

    def chat(
        self,
        messages,
        tools=None,
        on_token=None,
    ):
        self.calls += 1
        self.received_messages = messages
        self.received_tools = tools

        outcome = self.outcomes.pop(0)

        if isinstance(
            outcome,
            Exception,
        ):
            raise outcome

        if callable(outcome):
            return outcome(
                on_token,
            )

        return outcome


def test_routed_llm_uses_primary_model():
    primary = FakeBackend(
        "primary",
        [
            LLMResponse(
                content="primary response",
            )
        ],
    )
    fallback = FakeBackend(
        "fallback",
        [
            LLMResponse(
                content="fallback response",
            )
        ],
    )

    llm = RoutedLLM(
        router=StaticModelRouter(
            [
                "primary",
                "fallback",
            ]
        ),
        backends={
            "primary": primary,
            "fallback": fallback,
        },
    )

    response = llm.chat(
        [
            {
                "role": "user",
                "content": "hello",
            }
        ]
    )

    assert response.content == (
        "primary response"
    )
    assert primary.calls == 1
    assert fallback.calls == 0

    assert llm.last_decision is not None
    assert (
        llm.last_decision.model
        == "primary"
    )
    assert (
        llm.last_decision.is_fallback
        is False
    )


def test_routed_llm_falls_back_after_transient_failure():
    primary = FakeBackend(
        "primary",
        [
            TransientProviderError(
                "service unavailable"
            )
        ],
    )
    fallback = FakeBackend(
        "fallback",
        [
            LLMResponse(
                content="fallback response",
            )
        ],
    )

    llm = RoutedLLM(
        router=StaticModelRouter(
            [
                "primary",
                "fallback",
            ]
        ),
        backends={
            "primary": primary,
            "fallback": fallback,
        },
    )

    response = llm.chat(
        [
            {
                "role": "user",
                "content": "hello",
            }
        ]
    )

    assert response.content == (
        "fallback response"
    )
    assert primary.calls == 1
    assert fallback.calls == 1

    assert llm.last_decision is not None
    assert (
        llm.last_decision.model
        == "fallback"
    )
    assert (
        llm.last_decision.fallback_index
        == 1
    )
    assert response.model == "fallback"


def test_routed_llm_walks_multiple_fallbacks():
    first = FakeBackend(
        "first",
        [
            TransientProviderError(
                "first failed"
            )
        ],
    )
    second = FakeBackend(
        "second",
        [
            TransientProviderError(
                "second failed"
            )
        ],
    )
    third = FakeBackend(
        "third",
        [
            LLMResponse(
                content="third succeeded",
            )
        ],
    )

    llm = RoutedLLM(
        router=StaticModelRouter(
            [
                "first",
                "second",
                "third",
            ]
        ),
        backends={
            "first": first,
            "second": second,
            "third": third,
        },
    )

    response = llm.chat(
        [
            {
                "role": "user",
                "content": "hello",
            }
        ]
    )

    assert response.content == (
        "third succeeded"
    )
    assert first.calls == 1
    assert second.calls == 1
    assert third.calls == 1

    assert llm.last_decision is not None
    assert (
        llm.last_decision.model
        == "third"
    )


def test_routed_llm_reraises_after_fallbacks_exhausted():
    primary_error = (
        TransientProviderError(
            "primary failed"
        )
    )
    fallback_error = (
        TransientProviderError(
            "fallback failed"
        )
    )

    primary = FakeBackend(
        "primary",
        [
            primary_error,
        ],
    )
    fallback = FakeBackend(
        "fallback",
        [
            fallback_error,
        ],
    )

    llm = RoutedLLM(
        router=StaticModelRouter(
            [
                "primary",
                "fallback",
            ]
        ),
        backends={
            "primary": primary,
            "fallback": fallback,
        },
    )

    with pytest.raises(
        TransientProviderError,
        match="fallback failed",
    ):
        llm.chat(
            [
                {
                    "role": "user",
                    "content": "hello",
                }
            ]
        )

    assert primary.calls == 1
    assert fallback.calls == 1


def test_routed_llm_forwards_messages_and_tools():
    backend = FakeBackend(
        "primary",
        [
            LLMResponse(
                content="done",
            )
        ],
    )

    messages = [
        {
            "role": "user",
            "content": "hello",
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
            },
        }
    ]

    llm = RoutedLLM(
        router=StaticModelRouter(
            [
                "primary",
            ]
        ),
        backends={
            "primary": backend,
        },
    )

    llm.chat(
        messages,
        tools=tools,
    )

    assert (
        backend.received_messages
        is messages
    )
    assert backend.received_tools is tools


def test_routed_llm_does_not_fallback_after_streaming_text():
    fallback = FakeBackend(
        "fallback",
        [
            LLMResponse(
                content="must not run",
            )
        ],
    )

    def partial_failure(
        on_token,
    ):
        assert on_token is not None

        on_token("partial")

        raise TransientProviderError(
            "stream interrupted"
        )

    primary = FakeBackend(
        "primary",
        [
            partial_failure,
        ],
    )

    emitted = []

    llm = RoutedLLM(
        router=StaticModelRouter(
            [
                "primary",
                "fallback",
            ]
        ),
        backends={
            "primary": primary,
            "fallback": fallback,
        },
    )

    with pytest.raises(
        TransientProviderError,
        match="stream interrupted",
    ):
        llm.chat(
            [
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
            on_token=emitted.append,
        )

    assert emitted == [
        "partial",
    ]
    assert primary.calls == 1
    assert fallback.calls == 0


def test_routed_llm_rejects_missing_backend():
    llm = RoutedLLM(
        router=StaticModelRouter(
            [
                "missing",
            ]
        ),
        backends={},
    )

    with pytest.raises(
        ModelBackendUnavailableError,
        match="missing",
    ):
        llm.chat(
            [
                {
                    "role": "user",
                    "content": "hello",
                }
            ]
        )


def test_transient_error_classifier_accepts_server_error():
    error = TransientProviderError(
        "service unavailable"
    )

    assert is_transient_model_error(
        error
    )

def test_transient_error_classifier_rejects_bad_request():
    class BadRequestProviderError(
        RuntimeError
    ):
        status_code = 400

    error = BadRequestProviderError(
        "invalid request"
    )

    assert not is_transient_model_error(
        error
    )

@pytest.mark.parametrize(
    "status_code",
    [
        401,
        403,
    ],
)
def test_transient_error_classifier_rejects_auth_errors(
    status_code,
):
    class AuthProviderError(
        RuntimeError
    ):
        pass

    error = AuthProviderError(
        "authentication failed"
    )
    error.status_code = status_code

    assert not is_transient_model_error(
        error
    )

def test_transient_error_classifier_rejects_application_error():
    assert not is_transient_model_error(
        RuntimeError(
            "application bug"
        )
    )

def test_transient_error_classifier_rejects_budget_error():
    error = BudgetExceededError(
        "total_tokens",
        limit=100,
        actual=101,
    )

    assert not is_transient_model_error(
        error
    )

def test_routed_llm_does_not_fallback_on_bad_request():
    class BadRequestProviderError(
        RuntimeError
    ):
        status_code = 400

    error = BadRequestProviderError(
        "invalid tool schema"
    )

    primary = FakeBackend(
        "primary",
        [
            error,
        ],
    )
    fallback = FakeBackend(
        "fallback",
        [
            LLMResponse(
                content="must not run",
            )
        ],
    )

    llm = RoutedLLM(
        router=StaticModelRouter(
            [
                "primary",
                "fallback",
            ]
        ),
        backends={
            "primary": primary,
            "fallback": fallback,
        },
    )

    with pytest.raises(
        BadRequestProviderError,
        match="invalid tool schema",
    ):
        llm.chat(
            [
                {
                    "role": "user",
                    "content": "hello",
                }
            ]
        )

    assert primary.calls == 1
    assert fallback.calls == 0

def test_routed_llm_does_not_fallback_on_budget_error():
    error = BudgetExceededError(
        "total_tokens",
        limit=100,
        actual=101,
    )

    primary = FakeBackend(
        "primary",
        [
            error,
        ],
    )
    fallback = FakeBackend(
        "fallback",
        [
            LLMResponse(
                content="must not run",
            )
        ],
    )

    llm = RoutedLLM(
        router=StaticModelRouter(
            [
                "primary",
                "fallback",
            ]
        ),
        backends={
            "primary": primary,
            "fallback": fallback,
        },
    )

    with pytest.raises(
        BudgetExceededError,
    ):
        llm.chat(
            [
                {
                    "role": "user",
                    "content": "hello",
                }
            ]
        )

    assert primary.calls == 1
    assert fallback.calls == 0

def test_transient_error_classifier_accepts_rate_limit_status():
    class RateLimitedProviderError(
        RuntimeError
    ):
        status_code = 429

    assert is_transient_model_error(
        RateLimitedProviderError(
            "rate limited"
        )
    )

def test_routed_llm_traces_primary_success():
    tracer = InMemoryTracer()

    primary = FakeBackend(
        "primary",
        [
            LLMResponse(
                content="done",
                prompt_tokens=10,
                completion_tokens=5,
            )
        ],
    )

    llm = RoutedLLM(
        router=StaticModelRouter(
            [
                "primary",
            ]
        ),
        backends={
            "primary": primary,
        },
        route_request=RouteRequest(
            role="coder",
            required_capabilities=frozenset(
                {
                    "coding",
                }
            ),
        ),
        tracer=tracer,
    )

    response = llm.chat(
        [
            {
                "role": "user",
                "content": "hello",
            }
        ]
    )

    assert response.model == "primary"
    assert response.content == "done"

    assert [
        event.name
        for event in tracer.events
    ] == [
        "model.selected",
        "model.completed",
    ]

    selected = tracer.events[0]

    assert selected.attributes["model"] == (
        "primary"
    )
    assert selected.attributes["role"] == (
        "coder"
    )
    assert selected.attributes[
        "required_capabilities"
    ] == (
        "coding",
    )

    completed = tracer.events[1]

    assert completed.attributes[
        "prompt_tokens"
    ] == 10
    assert completed.attributes[
        "completion_tokens"
    ] == 5

def test_routed_llm_traces_fallback_sequence():
    tracer = InMemoryTracer()

    primary = FakeBackend(
        "primary",
        [
            TransientProviderError(
                "service unavailable"
            )
        ],
    )

    fallback = FakeBackend(
        "fallback",
        [
            LLMResponse(
                content="recovered",
            )
        ],
    )

    llm = RoutedLLM(
        router=StaticModelRouter(
            [
                "primary",
                "fallback",
            ]
        ),
        backends={
            "primary": primary,
            "fallback": fallback,
        },
        tracer=tracer,
    )

    response = llm.chat(
        [
            {
                "role": "user",
                "content": "hello",
            }
        ]
    )

    assert response.content == "recovered"

    assert [
        event.name
        for event in tracer.events
    ] == [
        "model.selected",
        "model.failed",
        "model.fallback",
        "model.completed",
    ]

    failed = tracer.events[1]

    assert failed.attributes["model"] == (
        "primary"
    )
    assert failed.attributes[
        "error_type"
    ] == "TransientProviderError"
    assert failed.attributes[
        "transient"
    ] is True
    assert failed.attributes[
        "status_code"
    ] == 503

    fallback_event = tracer.events[2]

    assert fallback_event.attributes[
        "from_model"
    ] == "primary"
    assert fallback_event.attributes[
        "to_model"
    ] == "fallback"
    assert fallback_event.attributes[
        "fallback_index"
    ] == 1

    completed = tracer.events[3]

    assert completed.attributes["model"] == (
        "fallback"
    )

def test_routed_llm_traces_non_transient_failure():
    tracer = InMemoryTracer()

    error = RuntimeError(
        "application bug"
    )

    primary = FakeBackend(
        "primary",
        [
            error,
        ],
    )

    fallback = FakeBackend(
        "fallback",
        [
            LLMResponse(
                content="must not run",
            )
        ],
    )

    llm = RoutedLLM(
        router=StaticModelRouter(
            [
                "primary",
                "fallback",
            ]
        ),
        backends={
            "primary": primary,
            "fallback": fallback,
        },
        tracer=tracer,
    )

    with pytest.raises(
        RuntimeError,
        match="application bug",
    ):
        llm.chat(
            [
                {
                    "role": "user",
                    "content": "hello",
                }
            ]
        )

    assert [
        event.name
        for event in tracer.events
    ] == [
        "model.selected",
        "model.failed",
    ]

    failed = tracer.events[1]

    assert failed.attributes[
        "transient"
    ] is False

    assert fallback.calls == 0

def test_routed_llm_traces_streaming_failure_without_fallback():
    tracer = InMemoryTracer()

    def partial_failure(
        on_token,
    ):
        assert on_token is not None

        on_token("partial")

        raise TransientProviderError(
            "stream interrupted"
        )

    primary = FakeBackend(
        "primary",
        [
            partial_failure,
        ],
    )

    fallback = FakeBackend(
        "fallback",
        [
            LLMResponse(
                content="must not run",
            )
        ],
    )

    llm = RoutedLLM(
        router=StaticModelRouter(
            [
                "primary",
                "fallback",
            ]
        ),
        backends={
            "primary": primary,
            "fallback": fallback,
        },
        tracer=tracer,
    )

    with pytest.raises(
        TransientProviderError,
    ):
        llm.chat(
            [
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
            on_token=lambda token: None,
        )

    assert [
        event.name
        for event in tracer.events
    ] == [
        "model.selected",
        "model.failed",
    ]

    assert tracer.events[1].attributes[
        "emitted_text"
    ] is True

    assert fallback.calls == 0

def test_routed_llm_ignores_tracer_failures():
    class FailingTracer(Tracer):
        def emit(
            self,
            event: TraceEvent,
        ) -> None:
            del event
            raise RuntimeError(
                "tracer unavailable"
            )

    backend = FakeBackend(
        "primary",
        [
            LLMResponse(
                content="still works",
            )
        ],
    )

    llm = RoutedLLM(
        router=StaticModelRouter(
            [
                "primary",
            ]
        ),
        backends={
            "primary": backend,
        },
        tracer=FailingTracer(),
    )

    response = llm.chat(
        [
            {
                "role": "user",
                "content": "hello",
            }
        ]
    )

    assert response.content == (
        "still works"
    )

def test_routed_llm_accepts_per_call_route_request():
    catalog = ModelCatalog(
        [
            ModelProfile(
                name="gpt-4o",
                capabilities=frozenset(
                    {
                        "coding",
                    }
                ),
            ),
            ModelProfile(
                name="gpt-4o-mini",
                capabilities=frozenset(
                    {
                        "coding",
                    }
                ),
            ),
        ]
    )

    primary = FakeBackend(
        "gpt-4o",
        [
            LLMResponse(
                content="expensive",
            )
        ],
    )
    cheap = FakeBackend(
        "gpt-4o-mini",
        [
            LLMResponse(
                content="cheap",
            )
        ],
    )

    llm = RoutedLLM(
        router=CapabilityModelRouter(
            [
                "gpt-4o",
                "gpt-4o-mini",
            ],
            catalog,
        ),
        backends={
            "gpt-4o": primary,
            "gpt-4o-mini": cheap,
        },
        route_request=RouteRequest(
            required_capabilities=frozenset(
                {
                    "coding",
                }
            ),
            remaining_cost_usd=0.01,
            estimated_prompt_tokens=1000,
            estimated_completion_tokens=500,
        ),
    )

    response = llm.chat(
        [
            {
                "role": "user",
                "content": "hello",
            }
        ],
        route_request=RouteRequest(
            required_capabilities=frozenset(
                {
                    "coding",
                }
            ),
            remaining_cost_usd=0.001,
            estimated_prompt_tokens=1000,
            estimated_completion_tokens=500,
        ),
    )

    assert response.content == "cheap"
    assert response.model == "gpt-4o-mini"

    assert primary.calls == 0
    assert cheap.calls == 1
