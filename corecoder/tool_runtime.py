import asyncio
import inspect
from collections.abc import Callable, Sequence
from typing import TypeVar


T = TypeVar("T")
R = TypeVar("R")


async def run_tool_calls(
    items: Sequence[T],
    execute: Callable[[T], R],
    *,
    max_concurrency: int = 8,
) -> list[R]:
    if max_concurrency <= 0:
        raise ValueError(
            "max_concurrency must be greater than 0"
        )
    semaphore = asyncio.Semaphore(max_concurrency)
    is_async = inspect.iscoroutinefunction(execute)

    async def run_one(item: T) -> R:
        async with semaphore:
            if is_async:
                return await execute(item)

            return await asyncio.to_thread(
                execute,
                item,
            )

    return list(
        await asyncio.gather(
            *(run_one(item) for item in items)
        )
    )
