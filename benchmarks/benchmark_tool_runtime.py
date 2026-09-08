"""Benchmark sequential vs bounded-concurrent tool execution."""

import argparse
import asyncio
import statistics
import time

from corecoder.tool_runtime import run_tool_calls


async def simulated_tool(item: int, delay_seconds: float) -> int:
    """Simulate an independent I/O-bound tool call."""
    await asyncio.sleep(delay_seconds)
    return item


async def run_sequential(
    items: list[int],
    delay_seconds: float,
) -> list[int]:
    results = []

    for item in items:
        results.append(
            await simulated_tool(
                item,
                delay_seconds,
            )
        )

    return results


async def run_concurrent(
    items: list[int],
    delay_seconds: float,
    max_concurrency: int,
) -> list[int]:
    async def execute(item: int) -> int:
        return await simulated_tool(
            item,
            delay_seconds,
        )

    return await run_tool_calls(
        items,
        execute,
        max_concurrency=max_concurrency,
    )


def percentile_95(values: list[float]) -> float:
    """Return a simple nearest-rank p95."""
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            int(len(ordered) * 0.95) - 1,
        ),
    )
    return ordered[index]


async def measure_once(
    *,
    mode: str,
    items: list[int],
    delay_seconds: float,
    max_concurrency: int,
) -> float:
    started_at = time.perf_counter()

    if mode == "sequential":
        results = await run_sequential(
            items,
            delay_seconds,
        )
    else:
        results = await run_concurrent(
            items,
            delay_seconds,
            max_concurrency,
        )

    elapsed = time.perf_counter() - started_at

    if results != items:
        raise RuntimeError(
            "tool results did not preserve input order"
        )

    return elapsed


async def benchmark(
    *,
    tool_calls: int,
    delay_seconds: float,
    repetitions: int,
    max_concurrency: int,
) -> None:
    items = list(range(tool_calls))

    # Warm up both paths once so first-run overhead has less influence.
    await measure_once(
        mode="sequential",
        items=items,
        delay_seconds=delay_seconds,
        max_concurrency=max_concurrency,
    )
    await measure_once(
        mode="concurrent",
        items=items,
        delay_seconds=delay_seconds,
        max_concurrency=max_concurrency,
    )

    sequential_times = []
    concurrent_times = []

    for _ in range(repetitions):
        sequential_times.append(
            await measure_once(
                mode="sequential",
                items=items,
                delay_seconds=delay_seconds,
                max_concurrency=max_concurrency,
            )
        )

        concurrent_times.append(
            await measure_once(
                mode="concurrent",
                items=items,
                delay_seconds=delay_seconds,
                max_concurrency=max_concurrency,
            )
        )

    sequential_mean = statistics.mean(
        sequential_times
    )
    concurrent_mean = statistics.mean(
        concurrent_times
    )

    speedup = (
        sequential_mean / concurrent_mean
    )

    latency_reduction = (
        (
            sequential_mean
            - concurrent_mean
        )
        / sequential_mean
        * 100
    )

    print("Tool Runtime Benchmark")
    print("======================")
    print(f"tool_calls:        {tool_calls}")
    print(f"delay_per_tool:    {delay_seconds:.3f}s")
    print(f"repetitions:       {repetitions}")
    print(f"max_concurrency:   {max_concurrency}")
    print()
    print(
        "sequential_mean:  "
        f"{sequential_mean:.4f}s"
    )
    print(
        "sequential_p50:   "
        f"{statistics.median(sequential_times):.4f}s"
    )
    print(
        "sequential_p95:   "
        f"{percentile_95(sequential_times):.4f}s"
    )
    print()
    print(
        "concurrent_mean:  "
        f"{concurrent_mean:.4f}s"
    )
    print(
        "concurrent_p50:   "
        f"{statistics.median(concurrent_times):.4f}s"
    )
    print(
        "concurrent_p95:   "
        f"{percentile_95(concurrent_times):.4f}s"
    )
    print()
    print(f"speedup:           {speedup:.2f}x")
    print(
        "latency_reduction:"
        f" {latency_reduction:.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tool-calls",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    if args.tool_calls <= 0:
        parser.error("--tool-calls must be greater than 0")

    if args.delay <= 0:
        parser.error("--delay must be greater than 0")

    if args.repetitions <= 0:
        parser.error("--repetitions must be greater than 0")

    if args.max_concurrency <= 0:
        parser.error(
            "--max-concurrency must be greater than 0"
        )

    asyncio.run(
        benchmark(
            tool_calls=args.tool_calls,
            delay_seconds=args.delay,
            repetitions=args.repetitions,
            max_concurrency=args.max_concurrency,
        )
    )


if __name__ == "__main__":
    main()