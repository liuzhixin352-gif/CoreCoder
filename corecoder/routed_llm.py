"""Routed LLM execution with transient-failure fallback."""

import time

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)

from .llm import LLMResponse
from .model_router import (
    ModelRouter,
    RouteDecision,
    RouteRequest,
)
from .tracing import TraceEvent, Tracer

class ChatModel(Protocol):
    """Minimal model backend required by routed execution."""

    model: str

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
    ) -> LLMResponse:
        """Run one model completion."""
        ...

@runtime_checkable
class RouteAwareLLM(Protocol):
    """LLM supporting request-local routing context."""

    route_request: RouteRequest

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
        *,
        route_request: RouteRequest | None = None,
    ) -> LLMResponse:
        """Run one completion with optional routing context."""
        ...


class ModelBackendUnavailableError(RuntimeError):
    """Raised when a route selects a model without a backend."""

    def __init__(
        self,
        model: str,
    ):
        self.model = model

        super().__init__(
            f"No model backend configured for {model!r}"
        )


def is_transient_model_error(
    error: Exception,
) -> bool:
    """Return whether an exhausted model error is safe to fallback."""
    if isinstance(
        error,
        (
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
        ),
    ):
        return True

    if isinstance(
        error,
        APIError,
    ):
        status_code = getattr(
            error,
            "status_code",
            None,
        )

        if (
            isinstance(status_code, int)
            and status_code >= 500
        ):
            return True

    status_code = getattr(
        error,
        "status_code",
        None,
    )

    if isinstance(
        status_code,
        int,
    ):
        if (
            status_code == 429
            or status_code >= 500
        ):
            return True

    error_name = type(
        error
    ).__name__.lower()

    transient_markers = (
        "timeout",
        "connection",
        "ratelimit",
        "rate_limit",
        "serviceunavailable",
    )

    return any(
        marker in error_name
        for marker in transient_markers
    )


class RoutedLLM:
    """Execute routed model calls with ordered fallback."""

    def __init__(
        self,
        *,
        router: ModelRouter,
        backends: dict[str, ChatModel],
        route_request: RouteRequest | None = None,
        should_fallback: (
            Callable[[Exception], bool] | None
        ) = None,
        tracer: Tracer | None = None,
    ):
        self.router = router
        self.backends = dict(backends)
        self.route_request = (
            route_request
            if route_request is not None
            else RouteRequest()
        )
        self.should_fallback = (
            should_fallback
            if should_fallback is not None
            else is_transient_model_error
        )
        self.tracer = tracer
        self.last_decision: (
            RouteDecision | None
        ) = None

    @property
    def routable_models(
        self,
    ) -> tuple[str, ...]:
        """Return models configured for routed execution."""
        return tuple(
            self.backends
        )

    def _trace_attributes(
        self,
        request: RouteRequest,
    ) -> dict[str, object]:
        """Return attributes shared by routing trace events."""
        attributes: dict[str, object] = {
            "required_capabilities": tuple(
                sorted(
                    request.required_capabilities
                )
            ),
        }

        if request.role is not None:
            attributes["role"] = (
                request.role
            )

        return attributes

    def _emit_trace(
        self,
        name: str,
        attributes: dict[str, object],
        *,
        request: RouteRequest,
    ) -> None:
        """Emit a routing trace without affecting execution."""
        if self.tracer is None:
            return

        try:
            self.tracer.emit(
                TraceEvent(
                    name=name,
                    timestamp=time.time(),
                    attributes={
                        **self._trace_attributes(
                            request,
                        ),
                        **attributes,
                    },
                )
            )
        except Exception:
            pass

    def _backend_for(
        self,
        model: str,
    ) -> ChatModel:
        """Return the configured backend for one model."""
        try:
            return self.backends[model]
        except KeyError as exc:
            raise ModelBackendUnavailableError(
                model,
            ) from exc

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
        *,
        route_request: RouteRequest | None = None,
    ) -> LLMResponse:
        """Route and execute one completion with safe fallback."""
        active_request = (
            route_request
            if route_request is not None
            else self.route_request
        )
        decision = self.router.select(
            active_request,
        )

        self._emit_trace(
            "model.selected",
            {
                "model": decision.model,
                "fallback_index": (
                    decision.fallback_index
                ),
            },
            request=active_request,
        )

        while True:
            backend = self._backend_for(
                decision.model,
            )

            attempt_started_at = (
                time.perf_counter()
            )
            emitted_text = False

            def routed_on_token(
                token: str,
            ) -> None:
                nonlocal emitted_text

                emitted_text = True

                if on_token is not None:
                    on_token(token)

            try:
                response = backend.chat(
                    messages,
                    tools=tools,
                    on_token=(
                        routed_on_token
                        if on_token is not None
                        else None
                    ),
                )
            except Exception as error:
                transient = self.should_fallback(
                    error,
                )

                status_code = getattr(
                    error,
                    "status_code",
                    None,
                )

                failed_attributes: dict[str, object] = {
                    "model": decision.model,
                    "fallback_index": (
                        decision.fallback_index
                    ),
                    "error_type": type(error).__name__,
                    "transient": transient,
                    "emitted_text": emitted_text,
                    "duration_ms": (
                        time.perf_counter()
                        - attempt_started_at
                    )
                    * 1000,
                }

                if isinstance(status_code, int):
                    failed_attributes["status_code"] = (
                        status_code
                    )

                self._emit_trace(
                    "model.failed",
                    failed_attributes,
                    request=active_request,
                )

                if emitted_text:
                    raise

                if not transient:
                    raise

                next_decision = (
                    self.router.fallback(
                        active_request,
                        decision.model,
                        error,
                    )
                )

                if next_decision is None:
                    raise

                self._emit_trace(
                    "model.fallback",
                    {
                        "from_model": decision.model,
                        "to_model": next_decision.model,
                        "fallback_index": (
                            next_decision.fallback_index
                        ),
                    },
                    request=active_request,
                )

                decision = next_decision
                continue

            response.model = decision.model
            self.last_decision = decision

            self._emit_trace(
                "model.completed",
                {
                    "model": decision.model,
                    "fallback_index": (
                        decision.fallback_index
                    ),
                    "duration_ms": (
                        time.perf_counter()
                        - attempt_started_at
                    )
                    * 1000,
                    "prompt_tokens": (
                        response.prompt_tokens
                    ),
                    "completion_tokens": (
                        response.completion_tokens
                    ),
                },
                request=active_request,
            )

            return response