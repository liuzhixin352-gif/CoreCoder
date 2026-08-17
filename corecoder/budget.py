"""Token and cost budget primitives for agent execution."""

from dataclasses import dataclass


class BudgetExceededError(RuntimeError):
    """Raised when recorded usage exceeds a configured budget."""

    def __init__(
        self,
        resource: str,
        *,
        limit: int | float,
        actual: int | float,
    ):
        self.resource = resource
        self.limit = limit
        self.actual = actual

        super().__init__(f"{resource} budget exceeded: limit={limit}, actual={actual}")


class BudgetPricingUnavailableError(RuntimeError):
    """Raised when a cost budget cannot price the active model."""

    def __init__(
        self,
        model: str | None,
    ):
        self.model = model

        super().__init__(f"Cost budget requires pricing for model {model!r}")


@dataclass(frozen=True)
class BudgetLimits:
    """Optional resource limits for one execution scope."""

    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_total_tokens: int | None = None
    max_cost_usd: float | None = None

    def __post_init__(self) -> None:
        values = {
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "max_total_tokens": self.max_total_tokens,
            "max_cost_usd": self.max_cost_usd,
        }

        for name, value in values.items():
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class BudgetUsage:
    """Resources consumed within one budget scope."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Return total prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens


class BudgetTracker:
    """Accumulate resource usage and enforce configured limits."""

    def __init__(
        self,
        limits: BudgetLimits,
    ):
        self.limits = limits
        self.usage = BudgetUsage()

    def record(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record incurred usage and raise if it exceeds a limit."""
        if prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")

        if completion_tokens < 0:
            raise ValueError("completion_tokens must be non-negative")

        if cost_usd < 0:
            raise ValueError("cost_usd must be non-negative")

        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens
        self.usage.cost_usd += cost_usd

        self._check_limits()

    def ensure_can_continue(self) -> None:
        """Raise when no budget remains for another LLM call."""
        checks = (
            (
                "prompt_tokens",
                self.limits.max_prompt_tokens,
                self.usage.prompt_tokens,
            ),
            (
                "completion_tokens",
                self.limits.max_completion_tokens,
                self.usage.completion_tokens,
            ),
            (
                "total_tokens",
                self.limits.max_total_tokens,
                self.usage.total_tokens,
            ),
            (
                "cost_usd",
                self.limits.max_cost_usd,
                self.usage.cost_usd,
            ),
        )

        for resource, limit, actual in checks:
            if (
                limit is not None
                and actual >= limit
            ):
                raise BudgetExceededError(
                    resource,
                    limit=limit,
                    actual=actual,
                )

    def _check_limits(self) -> None:
        checks = (
            (
                "prompt_tokens",
                self.limits.max_prompt_tokens,
                self.usage.prompt_tokens,
            ),
            (
                "completion_tokens",
                self.limits.max_completion_tokens,
                self.usage.completion_tokens,
            ),
            (
                "total_tokens",
                self.limits.max_total_tokens,
                self.usage.total_tokens,
            ),
            (
                "cost_usd",
                self.limits.max_cost_usd,
                self.usage.cost_usd,
            ),
        )

        for resource, limit, actual in checks:
            if limit is not None and actual > limit:
                raise BudgetExceededError(
                    resource,
                    limit=limit,
                    actual=actual,
                )
