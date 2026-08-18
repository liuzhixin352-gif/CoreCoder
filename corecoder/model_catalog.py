"""Model metadata and capability catalog."""

from dataclasses import dataclass, field

from .llm import get_model_pricing


@dataclass(frozen=True)
class ModelProfile:
    """Metadata describing one routable model."""

    name: str
    capabilities: frozenset[str] = field(
        default_factory=frozenset,
    )

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "model name must be non-empty"
            )

        if any(
            not capability.strip()
            for capability in self.capabilities
        ):
            raise ValueError(
                "capability names must be non-empty"
            )

    @property
    def pricing(
        self,
    ) -> tuple[float, float] | None:
        """Return input/output USD pricing per million tokens."""
        return get_model_pricing(
            self.name,
        )

    def supports(
        self,
        required_capabilities: frozenset[str],
    ) -> bool:
        """Return whether all required capabilities are supported."""
        return required_capabilities.issubset(
            self.capabilities,
        )


class ModelCatalog:
    """Lookup routable model metadata by model name."""

    def __init__(
        self,
        profiles: list[ModelProfile],
    ):
        names = [
            profile.name
            for profile in profiles
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "model profiles must not contain duplicates"
            )

        self._profiles = {
            profile.name: profile
            for profile in profiles
        }

    def get(
        self,
        model: str,
    ) -> ModelProfile:
        """Return metadata for one configured model."""
        try:
            return self._profiles[model]
        except KeyError as exc:
            raise ValueError(
                f"Unknown model profile: {model}"
            ) from exc

    def supports(
        self,
        model: str,
        required_capabilities: frozenset[str],
    ) -> bool:
        """Return whether a model satisfies capabilities."""
        return self.get(model).supports(
            required_capabilities,
        )