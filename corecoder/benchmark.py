"""Reproducible benchmark primitives for CoreCoder."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from .budget import BudgetTracker
from .eval import (
    EvalCase,
    EvalReport,
    EvalResult,
    EvalRunner,
)


@dataclass(frozen=True)
class BenchmarkCase:
    """One versioned benchmark scenario."""

    name: str
    category: str
    prompt: str
    expected_output: str
    expected_llm_rounds: int | None = None
    expected_tool_calls: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "category",
            "prompt",
        ):
            value = getattr(
                self,
                field_name,
            )

            if not value.strip():
                raise ValueError(
                    f"{field_name} must be non-empty"
                )

        if (
            self.expected_llm_rounds
            is not None
            and self.expected_llm_rounds < 0
        ):
            raise ValueError(
                "expected_llm_rounds must be non-negative"
            )

        if (
            self.expected_tool_calls
            is not None
            and self.expected_tool_calls < 0
        ):
            raise ValueError(
                "expected_tool_calls must be non-negative"
            )

    def to_eval_case(self) -> EvalCase:
        """Convert to the underlying evaluation case."""
        return EvalCase(
            name=self.name,
            prompt=self.prompt,
            expected_output=self.expected_output,
        )

    def matches_result(
        self,
        result: EvalResult,
    ) -> bool:
        """Return whether output and behavioral expectations pass."""
        if not result.success:
            return False

        if (
            self.expected_llm_rounds
            is not None
            and result.llm_rounds
            != self.expected_llm_rounds
        ):
            return False

        if (
            self.expected_tool_calls
            is not None
            and result.tool_calls
            != self.expected_tool_calls
        ):
            return False

        return True


@dataclass(frozen=True)
class BenchmarkSuite:
    """A named, versioned collection of benchmark cases."""

    name: str
    version: str
    cases: tuple[BenchmarkCase, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "name must be non-empty"
            )

        if not self.version.strip():
            raise ValueError(
                "version must be non-empty"
            )

        if not self.cases:
            raise ValueError(
                "benchmark suite must contain at least one case"
            )

        names = [
            case.name
            for case in self.cases
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "benchmark case names must be unique"
            )

    @property
    def categories(self) -> tuple[str, ...]:
        """Return categories in first-seen order."""
        return tuple(
            dict.fromkeys(
                case.category
                for case in self.cases
            )
        )

    def to_eval_cases(
        self,
    ) -> list[EvalCase]:
        """Convert all benchmark cases to eval cases."""
        return [
            case.to_eval_case()
            for case in self.cases
        ]


@dataclass(frozen=True)
class BenchmarkRun:
    """Results from one benchmark suite execution."""

    suite_name: str
    suite_version: str
    report: EvalReport

    def to_dict(self) -> dict:
        """Return a serializable benchmark report."""
        return {
            "suite_name": self.suite_name,
            "suite_version": self.suite_version,
            "report": self.report.to_dict(),
        }

    def to_json(self) -> str:
        """Serialize the benchmark report to JSON."""
        return json.dumps(
            self.to_dict()
        )

    def save_json(
        self,
        path: str | Path,
    ) -> None:
        """Save the benchmark report as UTF-8 JSON."""
        Path(path).write_text(
            self.to_json(),
            encoding="utf-8",
        )


class BenchmarkRunner:
    """Run one reproducible suite against an agent."""

    def __init__(
        self,
        *,
        suite: BenchmarkSuite,
        agent,
        budget_tracker_factory: (
            Callable[[], BudgetTracker] | None
        ) = None,
    ):
        self.suite = suite
        self.agent = agent
        self.budget_tracker_factory = (
            budget_tracker_factory
        )

    def run(self) -> BenchmarkRun:
        """Execute every case and apply benchmark expectations."""
        eval_runner = EvalRunner(
            agent=self.agent,
            budget_tracker_factory=(
                self.budget_tracker_factory
            ),
        )

        raw_report = eval_runner.run(
            self.suite.to_eval_cases()
        )

        results = [
            replace(
                result,
                success=case.matches_result(
                    result
                ),
            )
            for case, result in zip(
                self.suite.cases,
                raw_report.results,
                strict=True,
            )
        ]

        return BenchmarkRun(
            suite_name=self.suite.name,
            suite_version=self.suite.version,
            report=EvalReport(
                results=results,
            ),
        )

@dataclass(frozen=True)
class BenchmarkTarget:
    """One named implementation evaluated by a benchmark suite."""

    name: str
    agent_factory: Callable[[], object]
    budget_tracker_factory: (
        Callable[[], BudgetTracker] | None
    ) = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "benchmark target name must be non-empty"
            )


@dataclass(frozen=True)
class BenchmarkTargetRun:
    """One named target's benchmark result."""

    target_name: str
    run: BenchmarkRun

    def to_dict(self) -> dict:
        """Return a serializable target result."""
        return {
            "target_name": self.target_name,
            "run": self.run.to_dict(),
        }

def _markdown_cell(
    value: object,
) -> str:
    """Escape one value for use inside a Markdown table cell."""
    return (
        str(value)
        .replace(
            "\n",
            " ",
        )
        .replace(
            "|",
            r"\|",
        )
    )

@dataclass(frozen=True)
class BenchmarkComparison:
    """Results from evaluating multiple targets on one suite."""

    suite_name: str
    suite_version: str
    runs: tuple[BenchmarkTargetRun, ...]

    def to_dict(self) -> dict:
        """Return a serializable comparison."""
        return {
            "suite_name": self.suite_name,
            "suite_version": self.suite_version,
            "runs": [
                target_run.to_dict()
                for target_run in self.runs
            ],
        }

    def to_json(self) -> str:
        """Serialize the comparison to JSON."""
        return json.dumps(
            self.to_dict()
        )

    def to_markdown(self) -> str:
        """Render an interview-friendly Markdown comparison report."""
        lines = [
            "# Benchmark Comparison",
            "",
            f"- Suite: `{self.suite_name}`",
            f"- Version: `{self.suite_version}`",
            "",
            "## Summary",
            "",
            (
                "| Target | Success | Avg latency (ms) | "
                "Avg LLM rounds | Avg tool calls | "
                "Total tokens | Total cost (USD) |"
            ),
            (
                "|---|---:|---:|---:|---:|---:|---:|"
            ),
        ]

        for target_run in self.runs:
            report = target_run.run.report

            lines.append(
                "| "
                f"{_markdown_cell(target_run.target_name)} | "
                f"{report.success_rate:.1%} | "
                f"{report.avg_duration_ms:.2f} | "
                f"{report.avg_llm_rounds:.2f} | "
                f"{report.avg_tool_calls:.2f} | "
                f"{report.total_tokens} | "
                f"${report.total_cost_usd:.6f} |"
            )

        lines.extend(
            [
                "",
                "## Cases",
                "",
                (
                    "| Target | Case | Result | "
                    "Latency (ms) | LLM rounds | "
                    "Tool calls | Tokens | "
                    "Cost (USD) | Error |"
                ),
                (
                    "|---|---|---|---:|---:|---:|"
                    "---:|---:|---|"
                ),
            ]
        )

        for target_run in self.runs:
            for result in target_run.run.report.results:
                error = (
                    result.error_type
                    if result.error_type is not None
                    else "-"
                )

                lines.append(
                    "| "
                    f"{_markdown_cell(target_run.target_name)} | "
                    f"{_markdown_cell(result.case_name)} | "
                    f"{'PASS' if result.success else 'FAIL'} | "
                    f"{result.duration_ms:.2f} | "
                    f"{result.llm_rounds} | "
                    f"{result.tool_calls} | "
                    f"{result.total_tokens} | "
                    f"${result.cost_usd:.6f} | "
                    f"{_markdown_cell(error)} |"
                )

        return "\n".join(lines) + "\n"
    def save_json(
        self,
        path: str | Path,
    ) -> None:
        """Save the comparison as UTF-8 JSON."""
        Path(path).write_text(
            self.to_json(),
            encoding="utf-8",
        )

    def save_markdown(
        self,
        path: str | Path,
    ) -> None:
        """Save the comparison as UTF-8 Markdown."""
        Path(path).write_text(
            self.to_markdown(),
            encoding="utf-8",
        )


class BenchmarkComparisonRunner:
    """Evaluate multiple implementations on the same suite."""

    def __init__(
        self,
        *,
        suite: BenchmarkSuite,
        targets: tuple[BenchmarkTarget, ...],
    ):
        if not targets:
            raise ValueError(
                "benchmark comparison requires at least one target"
            )

        names = [
            target.name
            for target in targets
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "benchmark target names must be unique"
            )

        self.suite = suite
        self.targets = targets

    def run(self) -> BenchmarkComparison:
        """Run every target against the exact same suite."""
        runs = tuple(
            BenchmarkTargetRun(
                target_name=target.name,
                run=BenchmarkRunner(
                    suite=self.suite,
                    agent=target.agent_factory(),
                    budget_tracker_factory=(
                        target.budget_tracker_factory
                    ),
                ).run(),
            )
            for target in self.targets
        )

        return BenchmarkComparison(
            suite_name=self.suite.name,
            suite_version=self.suite.version,
            runs=runs,
        )

CORECODER_RUNTIME_BENCHMARK_V1 = BenchmarkSuite(
    name="corecoder-runtime",
    version="1",
    cases=(
        BenchmarkCase(
            name="direct-response",
            category="direct",
            prompt=(
                "Return exactly 'done' "
                "without using tools."
            ),
            expected_output="done",
            expected_llm_rounds=1,
            expected_tool_calls=0,
        ),
        BenchmarkCase(
            name="single-tool",
            category="tool-use",
            prompt=(
                "Use exactly one tool, "
                "then return exactly 'done'."
            ),
            expected_output="done",
            expected_llm_rounds=2,
            expected_tool_calls=1,
        ),
        BenchmarkCase(
            name="tool-chain",
            category="tool-use",
            prompt=(
                "Use exactly two tools, "
                "then return exactly 'done'."
            ),
            expected_output="done",
            expected_tool_calls=2,
        ),
    ),
)