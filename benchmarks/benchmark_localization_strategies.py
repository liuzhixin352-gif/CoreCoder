"""Compare Agent-driven and Code-RAG-assisted code localization."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from corecoder.agent import Agent
from corecoder.config import Config
from corecoder.llm import LLM, LiteLLM
from corecoder.tools.code_search import CodeSearchTool
from corecoder.tools.grep import GrepTool
from corecoder.tools.read import ReadFileTool
from corecoder.tools.repo_map import RepoMapTool


@dataclass(frozen=True)
class Case:
    case_id: str
    repo: str
    query: str
    relevant_paths: frozenset[str]


@dataclass
class Result:
    case_id: str
    repo: str
    strategy: str
    predicted_paths: list[str]
    rank: int | None
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    error: str | None = None


def _normalize_path(path: str) -> str:
    path = path.replace("\\", "/").strip()

    while path.startswith("./"):
        path = path[2:]

    return path


def _git(repo_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_dataset(path: Path) -> tuple[dict, list[Case]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    cases = [
        Case(
            case_id=str(raw["id"]),
            repo=str(raw["repo"]),
            query=str(raw["query"]),
            relevant_paths=frozenset(
                _normalize_path(path)
                for path in raw["relevant_paths"]
            ),
        )
        for raw in data["cases"]
    ]

    return data, cases


def _validate_repositories(
    data: dict,
    repos_root: Path,
) -> dict[str, Path]:
    result: dict[str, Path] = {}

    for repo_name, config in data["repositories"].items():
        repo_dir = repos_root / config["directory"]

        if not repo_dir.is_dir():
            raise RuntimeError(
                f"{repo_name}: missing repository {repo_dir}"
            )

        actual_commit = _git(
            repo_dir,
            "rev-parse",
            "HEAD",
        )

        expected_commit = config["commit"]

        if actual_commit != expected_commit:
            raise RuntimeError(
                f"{repo_name}: commit mismatch\n"
                f"expected: {expected_commit}\n"
                f"actual:   {actual_commit}"
            )

        status = _git(
            repo_dir,
            "status",
            "--porcelain",
        )

        if status:
            raise RuntimeError(
                f"{repo_name}: working tree is not clean"
            )

        result[repo_name] = repo_dir

    return result


def _build_llm(config: Config):
    llm_class = (
        LiteLLM
        if config.provider == "litellm"
        else LLM
    )

    return llm_class(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=0.0,
        max_tokens=1024,
    )


def _prompt(
    *,
    query: str,
    strategy: str,
) -> str:
    common = f"""
You are performing CODE LOCALIZATION ONLY.

Task:
{query}

Do not modify any files.
Do not run tests.
Identify the repository-relative Python files most likely to contain
the production implementation relevant to the task.

Return at most 5 UNIQUE file paths, ordered from most likely to least likely.

Your final response MUST be JSON only, exactly in this shape:

{{"paths": ["path/to/file.py", "another/file.py"]}}
""".strip()

    if strategy == "agent":
        instructions = """
Required search process:
1. First call repo_map(path=".", max_files=200).
2. Then use grep and read_file to investigate relevant candidates.
3. Base the final ranked paths on repository evidence.
""".strip()

    elif strategy == "rag":
        instructions = f"""
Required search process:
1. First call code_search with:
   query={json.dumps(query)}
   path="."
   top_k=5
2. Use grep and read_file only when useful to verify or refine candidates.
3. Base the final ranked paths on repository evidence.
""".strip()

    else:
        raise ValueError(
            f"unknown strategy: {strategy}"
        )

    return common + "\n\n" + instructions


def _parse_paths(text: str) -> list[str]:
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"no JSON object in response: {text!r}"
        )

    payload = json.loads(
        text[start : end + 1]
    )

    raw_paths = payload.get("paths")

    if not isinstance(raw_paths, list):
        raise ValueError(
            "response does not contain a paths list"
        )

    paths: list[str] = []

    for raw_path in raw_paths:
        if not isinstance(raw_path, str):
            continue

        path = _normalize_path(raw_path)

        if path and path not in paths:
            paths.append(path)

    return paths[:5]


def _first_relevant_rank(
    predicted_paths: list[str],
    relevant_paths: frozenset[str],
) -> int | None:
    for rank, path in enumerate(
        predicted_paths,
        start=1,
    ):
        if path in relevant_paths:
            return rank

    return None


def _run_case(
    *,
    llm,
    case: Case,
    repo_dir: Path,
    strategy: str,
    max_context_tokens: int,
) -> Result:
    if strategy == "agent":
        tools = [
            RepoMapTool(),
            GrepTool(),
            ReadFileTool(),
        ]
    else:
        tools = [
            CodeSearchTool(),
            GrepTool(),
            ReadFileTool(),
        ]

    agent = Agent(
        llm=llm,
        tools=tools,
        max_context_tokens=max_context_tokens,
        max_rounds=8,
    )

    tool_calls: list[str] = []

    def on_tool(
        name: str,
        arguments: dict,
    ) -> None:
        tool_calls.append(name)

    prompt_tokens_before = (
        llm.total_prompt_tokens
    )
    completion_tokens_before = (
        llm.total_completion_tokens
    )

    old_cwd = Path.cwd()
    started = time.perf_counter()

    try:
        os.chdir(repo_dir)

        answer = agent.chat(
            _prompt(
                query=case.query,
                strategy=strategy,
            ),
            on_tool=on_tool,
        )

        predicted_paths = _parse_paths(
            answer
        )

        error = None

    except Exception as exc:
        predicted_paths = []
        error = (
            f"{type(exc).__name__}: {exc}"
        )

    finally:
        os.chdir(old_cwd)

    latency = (
        time.perf_counter() - started
    )

    return Result(
        case_id=case.case_id,
        repo=case.repo,
        strategy=strategy,
        predicted_paths=predicted_paths,
        rank=_first_relevant_rank(
            predicted_paths,
            case.relevant_paths,
        ),
        tool_calls=len(tool_calls),
        prompt_tokens=(
            llm.total_prompt_tokens
            - prompt_tokens_before
        ),
        completion_tokens=(
            llm.total_completion_tokens
            - completion_tokens_before
        ),
        latency_seconds=latency,
        error=error,
    )


def _print_summary(
    results: list[Result],
    strategy: str,
) -> None:
    rows = [
        result
        for result in results
        if result.strategy == strategy
    ]

    total = len(rows)

    ranks = [
        result.rank
        for result in rows
    ]

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

    mrr = sum(
        1.0 / rank
        for rank in ranks
        if rank is not None
    ) / total

    avg_tools = sum(
        row.tool_calls
        for row in rows
    ) / total

    avg_tokens = sum(
        row.prompt_tokens
        + row.completion_tokens
        for row in rows
    ) / total

    avg_latency = sum(
        row.latency_seconds
        for row in rows
    ) / total

    errors = sum(
        row.error is not None
        for row in rows
    )

    print()
    print(strategy)
    print("-" * len(strategy))
    print(
        f"Hit@1: {hit_1}/{total} "
        f"({hit_1 / total:.1%})"
    )
    print(
        f"Hit@3: {hit_3}/{total} "
        f"({hit_3 / total:.1%})"
    )
    print(
        f"Hit@5: {hit_5}/{total} "
        f"({hit_5 / total:.1%})"
    )
    print(f"MRR: {mrr:.3f}")
    print(
        f"Avg tool calls: {avg_tools:.2f}"
    )
    print(
        f"Avg tokens: {avg_tokens:.1f}"
    )
    print(
        f"Avg latency: {avg_latency:.2f}s"
    )
    print(f"Errors: {errors}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repos-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "external_code_rag_cases.json"
        ),
    )

    parser.add_argument(
        "--cases-per-repo",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "localization_strategy_results.json"
        ),
    )

    args = parser.parse_args()

    if not 1 <= args.cases_per_repo <= 20:
        raise ValueError(
            "cases-per-repo must be between 1 and 20"
        )

    data, all_cases = _load_dataset(
        args.dataset
    )

    repo_dirs = _validate_repositories(
        data,
        args.repos_root,
    )

    selected: list[Case] = []

    for repo_name in (
        "flask",
        "requests",
        "pytest",
    ):
        repo_cases = [
            case
            for case in all_cases
            if case.repo == repo_name
        ]

        selected.extend(
            repo_cases[
                : args.cases_per_repo
            ]
        )

    config = Config.from_env()

    print(
        f"Model: {config.model}"
    )
    print(
        f"Cases: {len(selected)}"
    )

    llm = _build_llm(config)

    results: list[Result] = []

    for index, case in enumerate(
        selected,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(selected)}] "
            f"{case.case_id}: {case.query}"
        )

        for strategy in (
            "agent",
            "rag",
        ):
            result = _run_case(
                llm=llm,
                case=case,
                repo_dir=repo_dirs[
                    case.repo
                ],
                strategy=strategy,
                max_context_tokens=(
                    config.max_context_tokens
                ),
            )

            results.append(result)

            print(
                f"  {strategy:<5} "
                f"rank={result.rank} "
                f"tools={result.tool_calls} "
                f"tokens="
                f"{result.prompt_tokens + result.completion_tokens} "
                f"latency={result.latency_seconds:.2f}s"
            )

            if result.error:
                print(
                    f"        ERROR: "
                    f"{result.error}"
                )

    args.output.write_text(
        json.dumps(
            [
                result.__dict__
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    _print_summary(
        results,
        "agent",
    )
    _print_summary(
        results,
        "rag",
    )

    print()
    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()