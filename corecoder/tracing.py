"""Structured tracing primitives for agent execution."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TraceEvent:
    """A structured event emitted during agent execution."""

    name: str
    timestamp: float
    attributes: dict[str, object] = field(
        default_factory=dict
    )


class Tracer(ABC):
    """Interface for receiving structured trace events."""

    @abstractmethod
    def emit(self, event: TraceEvent) -> None:
        """Record a trace event."""
        ...


class InMemoryTracer(Tracer):
    """Collect trace events in memory."""

    def __init__(self):
        self.events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self.events.append(event)

    def events_for_run(self, run_id: str) -> list[TraceEvent]:
        """Return events associated with one agent run."""
        return [
            event
            for event in self.events
            if event.attributes.get("run_id") == run_id
        ]