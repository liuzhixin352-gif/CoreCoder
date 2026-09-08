"""Deterministic benchmark for CoreCoder context compression."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from corecoder.context import (
    ContextManager,
    estimate_tokens,
)


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    tool_groups: int
    lines_per_tool: int
    user_repeat: int


@dataclass(frozen=True)
class WorkloadResult:
    name: str
    tokens_before: int
    tokens_after: int
    reduction_percent: float
    applied_layers: tuple[str, ...]
    priority_preserved: bool
    protocol_valid: bool


def make_tool_output(
    group: int,
    lines: int,
    *,
    high_priority: bool,
) -> str:
    output = []

    if high_priority:
        output.append(
            f"CRITICAL_CONTEXT_MARKER_{group}"
        )

    for line_number in range(lines):
        output.append(
            "corecoder/module_"
            f"{group}.py:{line_number}: "
            "repository inspection output containing "
            "implementation details, diagnostics, "
            "dependencies, and source-code context "
            "for the current repair task."
        )

    return "\n".join(output)


def build_messages(
    spec: WorkloadSpec,
) -> list[dict]:
    messages: list[dict] = []

    for group in range(spec.tool_groups):
        call_id = f"call-{group}"

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Investigate repository component {group}. "
                    + (
                        "Preserve relevant implementation context. "
                        * spec.user_repeat
                    )
                ),
            }
        )

        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": (
                                '{"path": '
                                f'"corecoder/module_{group}.py"'
                                "}"
                            ),
                        },
                    }
                ],
            }
        )

        high_priority = group == 0

        tool_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": make_tool_output(
                group,
                spec.lines_per_tool,
                high_priority=high_priority,
            ),
        }

        if high_priority:
            tool_message["context_priority"] = "high"

        messages.append(tool_message)

        messages.append(
            {
                "role": "assistant",
                "content": (
                    "Repository evidence collected. "
                    "Continue analyzing dependencies and "
                    "implementation behavior."
                ),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": (
                "Using the repository evidence above, "
                "continue the repair task while preserving "
                "critical implementation details."
            ),
        }
    )

    return messages


def protocol_is_valid(
    messages: list[dict],
) -> bool:
    seen_tool_calls: set[str] = set()

    for message in messages:
        if message.get("role") == "assistant":
            for tool_call in (
                message.get("tool_calls") or []
            ):
                call_id = tool_call.get("id")

                if call_id:
                    seen_tool_calls.add(call_id)

        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")

            if call_id not in seen_tool_calls:
                return False

    return True


def priority_context_preserved(
    messages: list[dict],
) -> bool:
    return any(
        "CRITICAL_CONTEXT_MARKER_0"
        in str(message.get("content", ""))
        for message in messages
    )


def run_workload(
    spec: WorkloadSpec,
) -> WorkloadResult:
    messages = build_messages(spec)

    tokens_before = estimate_tokens(
        messages
    )

    manager = ContextManager(
        max_tokens=16_000,
        reserved_output_tokens=2_000,
        keep_recent_messages=8,
    )

    manager.maybe_compress(
        messages,
        llm=None,
    )

    tokens_after = estimate_tokens(
        messages
    )

    reduction_percent = (
        (
            tokens_before
            - tokens_after
        )
        / tokens_before
        * 100
        if tokens_before
        else 0.0
    )

    metrics = manager.last_metrics

    if metrics is None:
        raise RuntimeError(
            "ContextManager did not produce metrics"
        )

    return WorkloadResult(
        name=spec.name,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        reduction_percent=reduction_percent,
        applied_layers=metrics.applied_layers,
        priority_preserved=(
            priority_context_preserved(
                messages
            )
        ),
        protocol_valid=protocol_is_valid(
            messages
        ),
    )


def main() -> None:
    specs = [
        WorkloadSpec(
            name=f"context-{index:02d}",
            tool_groups=groups,
            lines_per_tool=lines,
            user_repeat=user_repeat,
        )
        for index, (
            groups,
            lines,
            user_repeat,
        ) in enumerate(
            [
                (8, 40, 4),
                (8, 50, 5),
                (8, 60, 6),
                (10, 40, 4),
                (10, 50, 5),
                (10, 60, 6),
                (12, 40, 4),
                (12, 50, 5),
                (12, 60, 6),
                (14, 40, 4),
                (14, 50, 5),
                (14, 60, 6),
                (16, 40, 4),
                (16, 50, 5),
                (16, 60, 6),
            ],
            start=1,
        )
    ]

    results = [
        run_workload(spec)
        for spec in specs
    ]

    reductions = [
        result.reduction_percent
        for result in results
    ]

    total_before = sum(
        result.tokens_before
        for result in results
    )
    total_after = sum(
        result.tokens_after
        for result in results
    )

    aggregate_reduction = (
        (
            total_before
            - total_after
        )
        / total_before
        * 100
    )

    layer_counts = {
        "tool_snip": 0,
        "summarize": 0,
        "hard_collapse": 0,
    }

    for result in results:
        for layer in result.applied_layers:
            if layer in layer_counts:
                layer_counts[layer] += 1

    priority_passes = sum(
        result.priority_preserved
        for result in results
    )

    protocol_passes = sum(
        result.protocol_valid
        for result in results
    )

    print("Context Compression Benchmark")
    print("=============================")
    print(f"cases:                 {len(results)}")
    print(
        "input_budget:          "
        "14000 estimated tokens"
    )
    print()

    avg_tokens_before = statistics.mean(
        result.tokens_before
        for result in results
    )
    avg_tokens_after = statistics.mean(
        result.tokens_after
        for result in results
    )

    print(f"avg_tokens_before:     {avg_tokens_before:.1f}")
    print(f"avg_tokens_after:      {avg_tokens_after:.1f}")

    print(
        "avg_reduction:         "
        f"{statistics.mean(reductions):.1f}%"
    )
    print(
        "median_reduction:      "
        f"{statistics.median(reductions):.1f}%"
    )
    print(
        "aggregate_reduction:   "
        f"{aggregate_reduction:.1f}%"
    )
    print(
        "min_reduction:         "
        f"{min(reductions):.1f}%"
    )
    print(
        "max_reduction:         "
        f"{max(reductions):.1f}%"
    )
    print()

    print(
        "tool_snip_cases:       "
        f"{layer_counts['tool_snip']}/{len(results)}"
    )
    print(
        "summarize_cases:       "
        f"{layer_counts['summarize']}/{len(results)}"
    )
    print(
        "hard_collapse_cases:   "
        f"{layer_counts['hard_collapse']}/{len(results)}"
    )
    print()

    print(
        "priority_preservation: "
        f"{priority_passes}/{len(results)}"
    )
    print(
        "protocol_integrity:    "
        f"{protocol_passes}/{len(results)}"
    )
    print()

    print("Per-case results")
    print("----------------")

    for result in results:
        layers = (
            ",".join(
                result.applied_layers
            )
            or "none"
        )

        print(
            f"{result.name}: "
            f"{result.tokens_before}"
            " -> "
            f"{result.tokens_after} "
            f"({result.reduction_percent:.1f}%) "
            f"[{layers}]"
        )


if __name__ == "__main__":
    main()
