"""Structured pytest execution tool."""
import re
import json
import subprocess
import sys
import time
from pathlib import Path

from .base import Tool

_PYTEST_COUNT_PATTERNS = {
    "passed_count": r"(\d+)\s+passed\b",
    "failed_count": r"(\d+)\s+failed\b",
    "error_count": r"(\d+)\s+errors?\b",
    "skipped_count": r"(\d+)\s+skipped\b",
    "xfailed_count": r"(\d+)\s+xfailed\b",
    "xpassed_count": r"(\d+)\s+xpassed\b",
}

_PYTEST_EXIT_STATUS = {
    0: "passed",
    1: "failed",
    2: "interrupted",
    3: "internal_error",
    4: "usage_error",
    5: "no_tests",
    6: "warning_limit_exceeded",
}


def _status_from_exit_code(exit_code: int) -> str:
    """Convert a pytest exit code into a structured status."""
    return _PYTEST_EXIT_STATUS.get(exit_code, "error")


def _parse_pytest_counts(output: str) -> dict[str, int]:
    """Extract test outcome counts from pytest's terminal summary."""
    counts = {name: 0 for name in _PYTEST_COUNT_PATTERNS}

    for name, pattern in _PYTEST_COUNT_PATTERNS.items():
        matches = re.findall(pattern, output, flags=re.IGNORECASE)
        if matches:
            counts[name] = int(matches[-1])

    return counts
def _parse_failed_tests(
    output: str,
    test_path: str | Path | None = None,
) -> list[str]:
    """Extract failed pytest node IDs from the short test summary."""
    failed_tests: list[str] = []

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if not line.startswith("FAILED "):
            continue

        details = line.removeprefix("FAILED ").strip()

        if " - " in details:
            node_id, _reason = details.split(" - ", 1)
        else:
            node_id = details

        node_id = node_id.strip()

        # When pytest runs a single absolute file outside its root directory,
        # the summary may contain only "::test_name".
        if node_id.startswith("::") and test_path is not None:
            target = Path(test_path)

            if target.is_file():
                node_id = f"{target}{node_id}"

        if node_id and node_id not in failed_tests:
            failed_tests.append(node_id)

    return failed_tests

class RunTestsTool(Tool):
    """Run pytest and return a structured result."""

    name = "run_tests"

    description = (
        "Run pytest for a file or directory and return a structured result. "
        "Use this after editing code to verify whether the tests pass."
    )

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Test file or directory to run (default: tests)",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum execution time in seconds (default: 120)",
            },
        },
        "required": [],
    }

    def execute(self, path: str = "tests", timeout: int = 120) -> str:
        """Run pytest and return the result as JSON text."""
        target = Path(path).expanduser().resolve()

        if not target.exists():
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Test path not found: {path}",
                },
                ensure_ascii=False,
                indent=2,
            )

        command = [
            sys.executable,
            "-m",
            "pytest",
            str(target),
            "-q",
        ]

        start_time = time.perf_counter()

        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=Path.cwd(),
            )
        except subprocess.TimeoutExpired:
            return json.dumps(
                {
                    "status": "timeout",
                    "timeout_seconds": timeout,
                },
                ensure_ascii=False,
                indent=2,
            )
        except Exception as exc:
            return json.dumps(
                {
                    "status": "error",
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )

        duration = time.perf_counter() - start_time

        output = proc.stdout
        if proc.stderr:
            output += f"\n[stderr]\n{proc.stderr}"

        counts = _parse_pytest_counts(output)
        failed_tests = _parse_failed_tests(output,target)

        if len(output) > 15_000:
            output = (
                output[:6000]
                + f"\n\n... truncated ({len(output)} chars total) ...\n\n"
                + output[-3000:]
            )

        result = {
            "status": _status_from_exit_code(proc.returncode),
            "exit_code": proc.returncode,
            "duration_seconds": round(duration, 2),
            "test_path": str(target),
            "passed_count": counts["passed_count"],
            "failed_count": counts["failed_count"],
            "error_count": counts["error_count"],
            "skipped_count": counts["skipped_count"],
            "xfailed_count": counts["xfailed_count"],
            "xpassed_count": counts["xpassed_count"],
            "failed_tests": failed_tests,
            "output": output.strip() or "(no output)",
        }

        return json.dumps(result, ensure_ascii=False, indent=2)