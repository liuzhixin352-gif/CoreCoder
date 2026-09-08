"""Held-out cross-repository Code RAG retrieval benchmark.

Compares:
1. Baseline lexical ranking reconstructed from raw CodeSearchResult.score.
2. Current source-aware ranking returned by LexicalCodeIndex.search().

The query set and ground truth live in external_code_rag_cases.json and
should be frozen before inspecting benchmark results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corecoder.code_rag import LexicalCodeIndex, index_repository


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    repo: str
    query: str
    relevant_paths: frozenset[str]


@dataclass(frozen=True)
class RankPair:
    case: RetrievalCase
    baseline_rank: int | None
    source_aware_rank: int | None


@dataclass(frozen=True)
class Metrics:
    total: int
    hit_1: int
    hit_3: int
    hit_5: int
    mrr: float

    @property
    def hit_1_rate(self) -> float:
        return _rate(self.hit_1, self.total)

    @property
    def hit_3_rate(self) -> float:
        return _rate(self.hit_3, self.total)

    @property
    def hit_5_rate(self) -> float:
        return _rate(self.hit_5, self.total)


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total * 100.0


def _git(repo_dir: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def _load_dataset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("dataset root must be a JSON object")

    return data


def _validate_dataset(data: dict[str, Any]) -> list[RetrievalCase]:
    repositories = data.get("repositories")
    raw_cases = data.get("cases")

    if not isinstance(repositories, dict):
        raise ValueError("dataset.repositories must be an object")

    if not isinstance(raw_cases, list):
        raise ValueError("dataset.cases must be a list")

    seen_ids: set[str] = set()
    cases: list[RetrievalCase] = []

    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("every case must be an object")

        case_id = str(raw_case["id"])
        repo = str(raw_case["repo"])
        query = str(raw_case["query"]).strip()
        relevant_paths = frozenset(
            _normalize_path(str(path))
            for path in raw_case["relevant_paths"]
        )

        if case_id in seen_ids:
            raise ValueError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)

        if repo not in repositories:
            raise ValueError(
                f"{case_id}: unknown repository {repo!r}"
            )

        if not query:
            raise ValueError(f"{case_id}: empty query")

        if not relevant_paths:
            raise ValueError(
                f"{case_id}: no relevant paths"
            )

        cases.append(
            RetrievalCase(
                case_id=case_id,
                repo=repo,
                query=query,
                relevant_paths=relevant_paths,
            )
        )

    counts = Counter(case.repo for case in cases)
    expected_repos = set(repositories)

    if set(counts) != expected_repos:
        raise ValueError(
            "dataset cases do not cover exactly the declared repositories"
        )

    for repo in sorted(expected_repos):
        if counts[repo] != 20:
            raise ValueError(
                f"{repo}: expected 20 frozen cases, got {counts[repo]}"
            )

    if len(cases) != 60:
        raise ValueError(
            f"expected 60 frozen cases, got {len(cases)}"
        )

    return cases


def _validate_repository(
    repos_root: Path,
    repo_name: str,
    repo_config: dict[str, Any],
    repo_cases: list[RetrievalCase],
) -> Path:
    repo_dir = repos_root / str(repo_config["directory"])

    if not repo_dir.is_dir():
        raise RuntimeError(
            f"{repo_name}: repository directory missing: {repo_dir}"
        )

    expected_commit = str(repo_config["commit"])
    actual_commit = _git(repo_dir, "rev-parse", "HEAD")

    if actual_commit != expected_commit:
        raise RuntimeError(
            f"{repo_name}: commit mismatch: "
            f"expected {expected_commit}, got {actual_commit}"
        )

    status = _git(repo_dir, "status", "--porcelain")

    if status:
        raise RuntimeError(
            f"{repo_name}: working tree is not clean:\n{status}"
        )

    missing_paths: list[str] = []

    for case in repo_cases:
        for relevant_path in case.relevant_paths:
            if not (repo_dir / Path(relevant_path)).is_file():
                missing_paths.append(
                    f"{case.case_id}: {relevant_path}"
                )

    if missing_paths:
        rendered = "\n".join(
            f"  - {item}"
            for item in sorted(set(missing_paths))
        )
        raise RuntimeError(
            f"{repo_name}: ground-truth files are missing:\n{rendered}"
        )

    return repo_dir


def _baseline_sort(results: list[Any]) -> list[Any]:
    """Reconstruct the pre-reranking lexical ordering."""
    return sorted(
        results,
        key=lambda result: (
            -result.score,
            _normalize_path(result.chunk.path),
            result.chunk.start_line,
            result.chunk.end_line,
            result.chunk.symbol,
        ),
    )


def _first_relevant_rank(
    results: list[Any],
    relevant_paths: frozenset[str],
) -> int | None:
    for rank, result in enumerate(results, start=1):
        path = _normalize_path(result.chunk.path)
        if path in relevant_paths:
            return rank
    return None


def _evaluate_repo(
    repo_dir: Path,
    cases: list[RetrievalCase],
) -> tuple[list[RankPair], int]:
    chunks = index_repository(repo_dir)
    index = LexicalCodeIndex(chunks)

    rank_pairs: list[RankPair] = []

    for case in cases:
        # Request every possible matching result. The current search order is
        # the source-aware ranking; the raw score remains the lexical score,
        # allowing the old baseline ordering to be reconstructed without
        # changing production code.
        source_aware_results = list(
            index.search(
                case.query,
                top_k=max(1, len(chunks)),
            )
        )
        baseline_results = _baseline_sort(
            source_aware_results
        )

        rank_pairs.append(
            RankPair(
                case=case,
                baseline_rank=_first_relevant_rank(
                    baseline_results,
                    case.relevant_paths,
                ),
                source_aware_rank=_first_relevant_rank(
                    source_aware_results,
                    case.relevant_paths,
                ),
            )
        )

    return rank_pairs, len(chunks)


def _metrics(
    pairs: list[RankPair],
    *,
    source_aware: bool,
) -> Metrics:
    ranks = [
        (
            pair.source_aware_rank
            if source_aware
            else pair.baseline_rank
        )
        for pair in pairs
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
    reciprocal_rank_sum = sum(
        1.0 / rank
        for rank in ranks
        if rank is not None
    )

    total = len(ranks)
    mrr = (
        reciprocal_rank_sum / total
        if total
        else 0.0
    )

    return Metrics(
        total=total,
        hit_1=hit_1,
        hit_3=hit_3,
        hit_5=hit_5,
        mrr=mrr,
    )


def _rank_text(rank: int | None) -> str:
    if rank is None:
        return "MISS"
    return str(rank)


def _print_metrics(
    label: str,
    metrics: Metrics,
) -> None:
    print(
        f"{label:<14} "
        f"Hit@1 {metrics.hit_1:>2}/{metrics.total:<2} "
        f"({metrics.hit_1_rate:>5.1f}%)  "
        f"Hit@3 {metrics.hit_3:>2}/{metrics.total:<2} "
        f"({metrics.hit_3_rate:>5.1f}%)  "
        f"Hit@5 {metrics.hit_5:>2}/{metrics.total:<2} "
        f"({metrics.hit_5_rate:>5.1f}%)  "
        f"MRR {metrics.mrr:.3f}"
    )


def _print_case_results(pairs: list[RankPair]) -> None:
    print()
    print("Per-case ranks")
    print("--------------")

    for pair in pairs:
        baseline = _rank_text(pair.baseline_rank)
        source = _rank_text(pair.source_aware_rank)

        print(
            f"{pair.case.case_id:<12} "
            f"baseline={baseline:<5} "
            f"source_aware={source:<5} "
            f"{pair.case.query}"
        )


def _print_failures(pairs: list[RankPair]) -> None:
    failures = [
        pair
        for pair in pairs
        if (
            pair.source_aware_rank is None
            or pair.source_aware_rank > 5
        )
    ]

    print()
    print("Source-aware misses beyond Top-5")
    print("-------------------------------")

    if not failures:
        print("none")
        return

    for pair in failures:
        print(
            f"{pair.case.case_id}: "
            f"rank={_rank_text(pair.source_aware_rank)} "
            f"query={pair.case.query}"
        )
        print(
            "    relevant="
            + ", ".join(
                sorted(pair.case.relevant_paths)
            )
        )


def parse_args() -> argparse.Namespace:
    default_dataset = (
        Path(__file__).resolve().parent
        / "external_code_rag_cases.json"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 60-query cross-repository "
            "Code RAG benchmark."
        )
    )
    parser.add_argument(
        "--repos-root",
        type=Path,
        required=True,
        help=(
            "Directory containing flask/, requests/, and pytest/."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help="Path to the frozen benchmark dataset JSON.",
    )
    parser.add_argument(
        "--show-cases",
        action="store_true",
        help="Print all 60 per-case baseline and source-aware ranks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = _load_dataset(args.dataset)
    cases = _validate_dataset(data)
    repositories = data["repositories"]

    cases_by_repo: dict[str, list[RetrievalCase]] = defaultdict(list)
    for case in cases:
        cases_by_repo[case.repo].append(case)

    all_pairs: list[RankPair] = []
    chunk_counts: dict[str, int] = {}

    dataset_sha256 = hashlib.sha256(
        args.dataset.read_bytes()
    ).hexdigest()

    print("External Code RAG Held-out Benchmark")
    print("====================================")
    print(f"dataset:     {args.dataset}")
    print(f"sha256:      {dataset_sha256}")
    print(f"repos_root:  {args.repos_root}")
    print(f"cases:       {len(cases)}")
    print(
        "protocol:    whole-repository Python indexing, "
        "frozen external queries"
    )
    print()

    # Validate every repository and every frozen ground-truth path before
    # evaluating even one query. This prevents accidental peeking at partial
    # benchmark results and then editing the held-out dataset.
    repo_dirs: dict[str, Path] = {}

    for repo_name in ("flask", "requests", "pytest"):
        repo_dirs[repo_name] = _validate_repository(
            args.repos_root,
            repo_name,
            repositories[repo_name],
            cases_by_repo[repo_name],
        )

    print("Preflight: all repositories, commits, clean trees, and labels OK")
    print()

    for repo_name in ("flask", "requests", "pytest"):
        repo_cases = cases_by_repo[repo_name]
        repo_config = repositories[repo_name]
        repo_dir = repo_dirs[repo_name]

        print(
            f"Indexing {repo_name:<8} "
            f"@ {repo_config['commit'][:8]} ..."
        )

        pairs, chunk_count = _evaluate_repo(
            repo_dir,
            repo_cases,
        )
        chunk_counts[repo_name] = chunk_count
        all_pairs.extend(pairs)

        baseline = _metrics(
            pairs,
            source_aware=False,
        )
        source = _metrics(
            pairs,
            source_aware=True,
        )

        print(f"  indexed_chunks: {chunk_count}")
        _print_metrics("  baseline", baseline)
        _print_metrics("  source-aware", source)
        print()

    overall_baseline = _metrics(
        all_pairs,
        source_aware=False,
    )
    overall_source = _metrics(
        all_pairs,
        source_aware=True,
    )

    print("Overall")
    print("-------")
    _print_metrics("baseline", overall_baseline)
    _print_metrics("source-aware", overall_source)

    print()
    print(
        "Hit@5 delta: "
        f"{overall_source.hit_5_rate - overall_baseline.hit_5_rate:+.1f} pp"
    )
    print(
        "MRR delta:   "
        f"{overall_source.mrr - overall_baseline.mrr:+.3f}"
    )

    _print_failures(all_pairs)

    if args.show_cases:
        _print_case_results(all_pairs)


if __name__ == "__main__":
    main()
