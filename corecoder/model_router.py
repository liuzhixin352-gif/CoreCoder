"""Model routing primitives and deterministic routing policy."""

from dataclasses import dataclass
from typing import Protocol
from .model_catalog import ModelCatalog
from .llm import estimate_cost_usd

@dataclass(frozen=True)
class RouteRequest:
    """Context used when selecting a model."""

    role: str | None = None
    required_capabilities: frozenset[str] = (
        frozenset()
    )
    remaining_cost_usd: float | None = None
    estimated_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0

    def __post_init__(self) -> None:
        values = {
            "remaining_cost_usd": (
                self.remaining_cost_usd
            ),
            "estimated_prompt_tokens": (
                self.estimated_prompt_tokens
            ),
            "estimated_completion_tokens": (
                self.estimated_completion_tokens
            ),
        }

        for name, value in values.items():
            if (
                value is not None
                and value < 0
            ):
                raise ValueError(
                    f"{name} must be non-negative"
                )

@dataclass(frozen=True)
class RouteDecision:
    """One model-routing decision."""

    model: str
    fallback_index: int = 0

    @property
    def is_fallback(self) -> bool:
        """Return whether this decision uses a fallback model."""
        return self.fallback_index > 0


class ModelRouter(Protocol):
    """Select primary and fallback models for one request."""

    def select(
        self,
        request: RouteRequest,
    ) -> RouteDecision:
        """Select the primary model."""
        ...

    def fallback(
        self,
        request: RouteRequest,
        current_model: str,
        error: Exception,
    ) -> RouteDecision | None:
        """Select the next fallback model, if any."""
        ...

class NoRoleRouteError(RuntimeError):
    """Raised when no router is configured for a request role."""

    def __init__(
        self,
        role: str | None,
    ):
        self.role = role

        super().__init__(
            f"No model router configured for role {role!r}"
        )

class NoEligibleModelError(RuntimeError):
    """Raised when no configured model satisfies a route request."""

    def __init__(
        self,
        request: RouteRequest,
    ):
        self.request = request

        capabilities = ", ".join(
            sorted(
                request.required_capabilities
            )
        )

        if not capabilities:
            capabilities = "<none>"

        message = (
            "No configured model satisfies "
            "required capabilities: "
            f"{capabilities}"
        )

        if request.remaining_cost_usd is not None:
            message += (
                "; remaining cost budget: "
                f"{request.remaining_cost_usd}"
            )

        super().__init__(
            message
        )

class StaticModelRouter:
    """Route through a fixed ordered list of models."""

    def __init__(
        self,
        models: list[str],
    ):
        if not models:
            raise ValueError(
                "models must contain at least one model"
            )

        if any(not model.strip() for model in models):
            raise ValueError(
                "model names must be non-empty"
            )

        if len(models) != len(set(models)):
            raise ValueError(
                "models must not contain duplicates"
            )

        self.models = tuple(models)

    def select(
        self,
        request: RouteRequest,
    ) -> RouteDecision:
        """Select the first configured model."""
        del request

        return RouteDecision(
            model=self.models[0],
            fallback_index=0,
        )

    def fallback(
        self,
        request: RouteRequest,
        current_model: str,
        error: Exception,
    ) -> RouteDecision | None:
        """Return the next configured model after the current one."""
        del request
        del error

        try:
            current_index = self.models.index(
                current_model,
            )
        except ValueError as exc:
            raise ValueError(
                "current_model is not configured "
                "in this router"
            ) from exc

        next_index = current_index + 1

        if next_index >= len(self.models):
            return None

        return RouteDecision(
            model=self.models[next_index],
            fallback_index=next_index,
        )

class RoleModelRouter:
    """Delegate routing to a role-specific model router."""

    def __init__(
        self,
        routes: dict[str, ModelRouter],
        *,
        default_router: ModelRouter | None = None,
    ):
        if any(
            not role.strip()
            for role in routes
        ):
            raise ValueError(
                "role names must be non-empty"
            )

        self.routes = dict(
            routes
        )
        self.default_router = (
            default_router
        )

    def _router_for(
        self,
        request: RouteRequest,
    ) -> ModelRouter:
        """Return the router configured for one request."""
        if request.role is not None:
            router = self.routes.get(
                request.role
            )

            if router is not None:
                return router

        if self.default_router is not None:
            return self.default_router

        raise NoRoleRouteError(
            request.role,
        )

    def select(
        self,
        request: RouteRequest,
    ) -> RouteDecision:
        """Select using the request's role-specific router."""
        return self._router_for(
            request
        ).select(
            request
        )

    def fallback(
        self,
        request: RouteRequest,
        current_model: str,
        error: Exception,
    ) -> RouteDecision | None:
        """Fallback using the same role-specific router."""
        return self._router_for(
            request
        ).fallback(
            request,
            current_model,
            error,
        )

class CapabilityModelRouter:
    """Route through capability-compatible models in priority order."""

    def __init__(
        self,
        models: list[str],
        catalog: ModelCatalog,
    ):
        if not models:
            raise ValueError(
                "models must contain at least one model"
            )

        if any(
            not model.strip()
            for model in models
        ):
            raise ValueError(
                "model names must be non-empty"
            )

        if len(models) != len(set(models)):
            raise ValueError(
                "models must not contain duplicates"
            )

        for model in models:
            catalog.get(model)

        self.models = tuple(models)
        self.catalog = catalog

    def _fits_cost_budget(
        self,
        model: str,
        request: RouteRequest,
    ) -> bool:
        """Return whether estimated call cost fits remaining budget."""
        if request.remaining_cost_usd is None:
            return True

        estimated_cost = estimate_cost_usd(
            model,
            prompt_tokens=(
                request.estimated_prompt_tokens
            ),
            completion_tokens=(
                request.estimated_completion_tokens
            ),
        )

        if estimated_cost is None:
            return False

        return (
            estimated_cost
            <= request.remaining_cost_usd
        )

    def _eligible_models(
        self,
        request: RouteRequest,
    ) -> tuple[str, ...]:
        """Return compatible models in configured priority order."""
        return tuple(
            model
            for model in self.models
            if self.catalog.supports(
                model,
                request.required_capabilities,
            )
            and self._fits_cost_budget(
                model,
                request,
            )
        )

    def select(
        self,
        request: RouteRequest,
    ) -> RouteDecision:
        """Select the highest-priority compatible model."""
        eligible_models = self._eligible_models(
            request,
        )

        if not eligible_models:
            raise NoEligibleModelError(
                request,
            )

        return RouteDecision(
            model=eligible_models[0],
            fallback_index=0,
        )

    def fallback(
        self,
        request: RouteRequest,
        current_model: str,
        error: Exception,
    ) -> RouteDecision | None:
        """Select the next compatible model after the current one."""
        del error

        if current_model not in self.models:
            raise ValueError(
                "current_model is not configured "
                "in this router"
            )

        eligible_models = self._eligible_models(
            request,
        )

        try:
            current_index = (
                eligible_models.index(
                    current_model,
                )
            )
        except ValueError as exc:
            raise ValueError(
                "current_model does not satisfy "
                "the route request"
            ) from exc

        next_index = current_index + 1

        if next_index >= len(
            eligible_models
        ):
            return None

        return RouteDecision(
            model=eligible_models[
                next_index
            ],
            fallback_index=next_index,
        )
