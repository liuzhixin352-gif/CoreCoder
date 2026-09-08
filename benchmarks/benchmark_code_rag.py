"""Repository-level retrieval benchmark for CoreCoder Code RAG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from corecoder.code_rag import (
    LexicalCodeIndex,
    index_repository,
)


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_paths: frozenset[str]


CASES = [
    RetrievalCase(
        query="tool permission approval allow ask deny",
        relevant_paths=frozenset({
            "corecoder/permissions.py",
            "corecoder/agent.py",
        }),
    ),
    RetrievalCase(
        query="token cost budget usage remaining cost limit",
        relevant_paths=frozenset({
            "corecoder/budget.py",
        }),
    ),
    RetrievalCase(
        query="capability role budget model routing selection",
        relevant_paths=frozenset({
            "corecoder/model_router.py",
        }),
    ),
    RetrievalCase(
        query="transient model fallback timeout rate limit streaming",
        relevant_paths=frozenset({
            "corecoder/routed_llm.py",
        }),
    ),
    RetrievalCase(
        query="context compression tool output summary collapse",
        relevant_paths=frozenset({
            "corecoder/context.py",
        }),
    ),
    RetrievalCase(
        query="agent llm tool call loop execution",
        relevant_paths=frozenset({
            "corecoder/agent.py",
        }),
    ),
    RetrievalCase(
        query="bounded concurrent asynchronous tool calls semaphore",
        relevant_paths=frozenset({
            "corecoder/tool_runtime.py",
        }),
    ),
    RetrievalCase(
        query="planner coder reviewer role multi agent",
        relevant_paths=frozenset({
            "corecoder/multi_agent.py",
        }),
    ),
    RetrievalCase(
        query="workflow state checkpoint resume repair",
        relevant_paths=frozenset({
            "corecoder/issue_orchestration.py",
        }),
    ),
    RetrievalCase(
        query="save load workflow checkpoint json",
        relevant_paths=frozenset({
            "corecoder/issue_orchestration.py",
        }),
    ),
    RetrievalCase(
        query="repository identity verification mismatch worktree",
        relevant_paths=frozenset({
            "corecoder/repository_guard.py",
        }),
    ),
    RetrievalCase(
        query="repair branch create dedicated branch",
        relevant_paths=frozenset({
            "corecoder/repair_branch.py",
        }),
    ),
    RetrievalCase(
        query="post repair validation tests passed failed count",
        relevant_paths=frozenset({
            "corecoder/post_repair_validation.py",
        }),
    ),
    RetrievalCase(
        query="CI failure log prompt retry check runs",
        relevant_paths=frozenset({
            "corecoder/repair_ci.py",
        }),
    ),
    RetrievalCase(
        query="structured trace event tracer timestamp attributes",
        relevant_paths=frozenset({
            "corecoder/tracing.py",
        }),
    ),
    RetrievalCase(
        query="FastAPI agent chat service request response",
        relevant_paths=frozenset({
            "corecoder/agent_service.py",
            "corecoder/service.py",
        }),
    ),
    RetrievalCase(
        query="repository code chunk lexical index search",
        relevant_paths=frozenset({
            "corecoder/code_rag.py",
        }),
    ),
    RetrievalCase(
        query="model catalog capabilities pricing profile",
        relevant_paths=frozenset({
            "corecoder/model_catalog.py",
        }),
    ),
    RetrievalCase(
        query="evaluation success latency tool calls token cost",
        relevant_paths=frozenset({
            "corecoder/eval.py",
        }),
    ),
    RetrievalCase(
        query="deterministic benchmark direct response tool chain",
        relevant_paths=frozenset({
            "corecoder/benchmark.py",
        }),
    ),
]


def first_relevant_rank(
    paths: list[str],
    relevant_paths: frozenset[str],
) -> int | None:
    for rank, path in enumerate(
        paths,
        start=1,
    ):
        if path in relevant_paths:
            return rank

    return None


def percentage(
    count: int,
    total: int,
) -> float:
    return (
        count / total * 100
        if total
        else 0.0
    )


def main() -> None:
    repo_root = Path(".")

    chunks = index_repository(
        repo_root
    )
    index = LexicalCodeIndex(
        chunks
    )

    ranks: list[int | None] = []

    print("Code RAG Retrieval Benchmark")
    print("============================")
    print(f"cases:          {len(CASES)}")
    print(f"indexed_chunks: {len(chunks)}")
    print()

    for case_number, case in enumerate(
        CASES,
        start=1,
    ):
        results = index.search(
            case.query,
            top_k=5,
        )

        paths = [
            result.chunk.path
            for result in results
        ]

        rank = first_relevant_rank(
            paths,
            case.relevant_paths,
        )

        ranks.append(rank)

        rank_text = (
            str(rank)
            if rank is not None
            else "MISS"
        )

        print(
            f"{case_number:02d}. "
            f"rank={rank_text:<4} "
            f"query={case.query}"
        )

        for result_rank, result in enumerate(
            results,
            start=1,
        ):
            print(
                "    "
                f"{result_rank}. "
                f"{result.chunk.path}"
                f"::{result.chunk.symbol} "
                f"score={result.score:.1f}"
            )

    hit_1 = sum(
        rank is not None and rank <= 1
        for rank in ranks
    )
    hit_3 = sum(
        rank is not None and rank <= 3
        for rank in ranks
    )
    hit_5 = sum(
        rank is not None and rank <= 5
        for rank in ranks
    )

    reciprocal_ranks = [
        (
            1.0 / rank
            if rank is not None
            else 0.0
        )
        for rank in ranks
    ]

    mrr = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )

    print()
    print("Summary")
    print("-------")
    print(
        f"Hit@1: {hit_1}/{len(CASES)} "
        f"({percentage(hit_1, len(CASES)):.1f}%)"
    )
    print(
        f"Hit@3: {hit_3}/{len(CASES)} "
        f"({percentage(hit_3, len(CASES)):.1f}%)"
    )
    print(
        f"Hit@5: {hit_5}/{len(CASES)} "
        f"({percentage(hit_5, len(CASES)):.1f}%)"
    )
    print(f"MRR:   {mrr:.3f}")


if __name__ == "__main__":
    main()