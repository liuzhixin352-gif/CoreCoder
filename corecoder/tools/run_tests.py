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


def _parse_pytest_counts(output: str) -> dict[str, int]:
    """Extract test outcome counts from pytest's terminal summary."""
    counts = {name: 0 for name in _PYTEST_COUNT_PATTERNS}

    for name, pattern in _PYTEST_COUNT_PATTERNS.items():
        matches = re.findall(pattern, output, flags=re.IGNORECASE)
        if matches:
            counts[name] = int(matches[-1])

    return counts


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

        if len(output) > 15_000:
            output = (
                output[:6000]
                + f"\n\n... truncated ({len(output)} chars total) ...\n\n"
                + output[-3000:]
            )

        result = {
            "status": "passed" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "duration_seconds": round(duration, 2),
            "test_path": str(target),
            "passed_count": counts["passed_count"],
            "failed_count": counts["failed_count"],
            "error_count": counts["error_count"],
            "skipped_count": counts["skipped_count"],
            "xfailed_count": counts["xfailed_count"],
            "xpassed_count": counts["xpassed_count"],
            "output": output.strip() or "(no output)",
        }

        return json.dumps(result, ensure_ascii=False, indent=2)