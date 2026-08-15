"""Evaluation primitives for CoreCoder agents."""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    """One reproducible agent evaluation task."""

    name: str
    prompt: str
    expected_output: str | None = None

    def matches(self, output: str) -> bool:
        if self.expected_output is None:
            raise ValueError("expected_output is required for exact matching")

        return output == self.expected_output


@dataclass(frozen=True)
class EvalResult:
    """Outcome of one agent evaluation task."""

    case_name: str
    success: bool
    output: str
    duration_ms: float = 0.0
    llm_rounds: int = 0
    tool_calls: int = 0
    context_tokens_saved: int = 0
    error_type: str | None = None


class EvalRunner:
    """Run reproducible evaluation cases against an agent."""

    def __init__(self, agent):
        self.agent = agent

    def run_case(self, case: EvalCase) -> EvalResult:
        tracer = getattr(self.agent, "tracer", None)
        trace_events = getattr(tracer, "events", None)
        trace_start = len(trace_events) if isinstance(trace_events, list) else None

        started_at = time.perf_counter()

        output = ""
        error_type = None

        try:
            output = self.agent.chat(case.prompt)
        except Exception as exc:
            error_type = type(exc).__name__

        duration_ms = (time.perf_counter() - started_at) * 1000

        llm_rounds = 0
        tool_calls = 0
        context_tokens_saved = 0

        if trace_start is not None:
            case_events = trace_events[trace_start:]

            llm_rounds = sum(1 for event in case_events if event.name == "llm.started")

            tool_calls = sum(1 for event in case_events if event.name == "tool.started")

            context_tokens_saved = sum(
                event.attributes.get("tokens_saved", 0) for event in case_events if event.name == "context.managed"
            )

        return EvalResult(
            case_name=case.name,
            success=(error_type is None and case.matches(output)),
            output=output,
            duration_ms=duration_ms,
            llm_rounds=llm_rounds,
            tool_calls=tool_calls,
            context_tokens_saved=context_tokens_saved,
            error_type=error_type,
        )

    def run(self, cases: list[EvalCase]) -> EvalReport:
        results = [self.run_case(case) for case in cases]

        return EvalReport(results=results)


@dataclass(frozen=True)
class EvalReport:
    """Aggregate results for an evaluation run."""

    results: list[EvalResult]

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0

        successes = sum(1 for result in self.results if result.success)
        return successes / len(self.results)

    @property
    def avg_duration_ms(self) -> float:
        if not self.results:
            return 0.0

        return sum(result.duration_ms for result in self.results) / len(self.results)

    @property
    def avg_context_tokens_saved(self) -> float:
        if not self.results:
            return 0.0

        return sum(result.context_tokens_saved for result in self.results) / len(self.results)

    @property
    def avg_llm_rounds(self) -> float:
        if not self.results:
            return 0.0

        return sum(result.llm_rounds for result in self.results) / len(self.results)

    @property
    def avg_tool_calls(self) -> float:
        if not self.results:
            return 0.0

        return sum(result.tool_calls for result in self.results) / len(self.results)

    def to_dict(self) -> dict:
        return {
            "success_rate": self.success_rate,
            "avg_duration_ms": self.avg_duration_ms,
            "avg_llm_rounds": self.avg_llm_rounds,
            "avg_tool_calls": self.avg_tool_calls,
            "avg_context_tokens_saved": (self.avg_context_tokens_saved),
            "results": [
                {
                    "case_name": result.case_name,
                    "success": result.success,
                    "output": result.output,
                    "duration_ms": result.duration_ms,
                    "llm_rounds": result.llm_rounds,
                    "tool_calls": result.tool_calls,
                    "context_tokens_saved": (result.context_tokens_saved),
                    "error_type": result.error_type,
                }
                for result in self.results
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            self.to_json(),
            encoding="utf-8",
        )
