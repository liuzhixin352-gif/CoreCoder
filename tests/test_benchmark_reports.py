"""Tests for benchmark report rendering."""

import json

from corecoder.benchmark import (
    BenchmarkComparison,
    BenchmarkRun,
    BenchmarkTargetRun,
)
from corecoder.eval import (
    EvalReport,
    EvalResult,
)


def _comparison() -> BenchmarkComparison:
    baseline = BenchmarkRun(
        suite_name="corecoder-runtime",
        suite_version="1",
        report=EvalReport(
            results=[
                EvalResult(
                    case_name="direct-response",
                    success=True,
                    output="done",
                    duration_ms=10.0,
                    llm_rounds=1,
                    tool_calls=0,
                    prompt_tokens=10,
                    completion_tokens=2,
                    total_tokens=12,
                    cost_usd=0.0001,
                ),
                EvalResult(
                    case_name="single-tool",
                    success=False,
                    output="done",
                    duration_ms=20.0,
                    llm_rounds=1,
                    tool_calls=0,
                    prompt_tokens=10,
                    completion_tokens=2,
                    total_tokens=12,
                    cost_usd=0.0001,
                ),
                EvalResult(
                    case_name="tool-chain",
                    success=False,
                    output="done",
                    duration_ms=30.0,
                    llm_rounds=1,
                    tool_calls=0,
                    prompt_tokens=10,
                    completion_tokens=2,
                    total_tokens=12,
                    cost_usd=0.0001,
                ),
            ]
        ),
    )

    advanced = BenchmarkRun(
        suite_name="corecoder-runtime",
        suite_version="1",
        report=EvalReport(
            results=[
                EvalResult(
                    case_name="direct-response",
                    success=True,
                    output="done",
                    duration_ms=15.0,
                    llm_rounds=1,
                    tool_calls=0,
                    prompt_tokens=10,
                    completion_tokens=2,
                    total_tokens=12,
                    cost_usd=0.0001,
                ),
                EvalResult(
                    case_name="single-tool",
                    success=True,
                    output="done",
                    duration_ms=25.0,
                    llm_rounds=2,
                    tool_calls=1,
                    prompt_tokens=20,
                    completion_tokens=4,
                    total_tokens=24,
                    cost_usd=0.0002,
                ),
                EvalResult(
                    case_name="tool-chain",
                    success=True,
                    output="done",
                    duration_ms=35.0,
                    llm_rounds=2,
                    tool_calls=2,
                    prompt_tokens=30,
                    completion_tokens=6,
                    total_tokens=36,
                    cost_usd=0.0003,
                ),
            ]
        ),
    )

    return BenchmarkComparison(
        suite_name="corecoder-runtime",
        suite_version="1",
        runs=(
            BenchmarkTargetRun(
                target_name="baseline",
                run=baseline,
            ),
            BenchmarkTargetRun(
                target_name="advanced",
                run=advanced,
            ),
        ),
    )


def test_comparison_markdown_contains_summary_table():
    markdown = _comparison().to_markdown()

    assert "# Benchmark Comparison" in markdown
    assert "Suite: `corecoder-runtime`" in markdown
    assert "Version: `1`" in markdown

    assert (
        "| baseline | 33.3% | 20.00 |"
        in markdown
    )

    assert (
        "| advanced | 100.0% | 25.00 |"
        in markdown
    )

    assert "$0.000300" in markdown
    assert "$0.000600" in markdown


def test_comparison_markdown_contains_case_results():
    markdown = _comparison().to_markdown()

    assert (
        "| baseline | direct-response | PASS |"
        in markdown
    )

    assert (
        "| baseline | single-tool | FAIL |"
        in markdown
    )

    assert (
        "| advanced | single-tool | PASS |"
        in markdown
    )

    assert (
        "| advanced | tool-chain | PASS |"
        in markdown
    )


def test_comparison_save_markdown(tmp_path):
    comparison = _comparison()

    path = tmp_path / "benchmark.md"

    comparison.save_markdown(
        path
    )

    assert path.read_text(
        encoding="utf-8"
    ) == comparison.to_markdown()


def test_comparison_json_and_markdown_share_suite_metadata(
    tmp_path,
):
    comparison = _comparison()

    json_path = tmp_path / "benchmark.json"
    markdown_path = tmp_path / "benchmark.md"

    comparison.save_json(
        json_path
    )
    comparison.save_markdown(
        markdown_path
    )

    payload = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )

    markdown = markdown_path.read_text(
        encoding="utf-8"
    )

    assert payload["suite_name"] == (
        "corecoder-runtime"
    )
    assert payload["suite_version"] == "1"

    assert "corecoder-runtime" in markdown
    assert "Version: `1`" in markdown


def test_markdown_escapes_table_cell_characters():
    comparison = BenchmarkComparison(
        suite_name="suite",
        suite_version="1",
        runs=(
            BenchmarkTargetRun(
                target_name="base|line",
                run=BenchmarkRun(
                    suite_name="suite",
                    suite_version="1",
                    report=EvalReport(
                        results=[
                            EvalResult(
                                case_name="case|one",
                                success=False,
                                output="",
                                error_type=(
                                    "Bad|Error"
                                ),
                            )
                        ]
                    ),
                ),
            ),
        ),
    )

    markdown = comparison.to_markdown()

    assert r"base\|line" in markdown
    assert r"case\|one" in markdown
    assert r"Bad\|Error" in markdown