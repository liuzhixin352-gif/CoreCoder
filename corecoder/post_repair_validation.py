"""Run mandatory tests after an Issue repair."""

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_PYTHON_EXECUTABLE = sys.executable

_PYTEST_COUNT_PATTERNS = {
    "passed_count": r"(\d+)\s+passed\b",
    "failed_count": r"(\d+)\s+failed\b",
    "error_count": r"(\d+)\s+errors?\b",
}

_PYTEST_EXIT_STATUS = {
    0: "passed",
    1: "failed",
    2: "interrupted",
    3: "internal_error",
    4: "usage_error",
    5: "no_tests",
}


@dataclass(frozen=True)
class PostRepairValidation:
    """Result of mandatory post-repair test execution."""

    command: tuple[str, ...]
    status: str
    exit_code: int
    passed_count: int
    failed_count: int
    error_count: int
    output: str

    @property
    def passed(self) -> bool:
        """Return whether the mandatory validation passed."""
        return (
            self.status == "passed"
            and self.exit_code == 0
        )


class PostRepairValidationError(RuntimeError):
    """Raised when mandatory validation cannot be executed."""


def _run_pytest(
    test_path: str | Path,
    *,
    cwd: str | Path | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """Execute pytest for mandatory post-repair validation."""
    command = [
        _PYTHON_EXECUTABLE,
        "-m",
        "pytest",
        str(test_path),
        "-q",
    ]

    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _parse_count(
    output: str,
    pattern: str,
) -> int:
    """Return the last matching pytest summary count."""
    matches = re.findall(
        pattern,
        output,
        flags=re.IGNORECASE,
    )

    if not matches:
        return 0

    return int(matches[-1])


def run_post_repair_validation(
    test_path: str | Path = "tests",
    *,
    cwd: str | Path | None = None,
    timeout: int = 120,
) -> PostRepairValidation:
    """Run and summarize mandatory post-repair tests."""
    try:
        result = _run_pytest(
            test_path,
            cwd=cwd,
            timeout=timeout,
        )
    except (
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        raise PostRepairValidationError(
            "unable to run post-repair validation"
        ) from error

    output = result.stdout

    if result.stderr:
        if output:
            output += "\n"

        output += f"[stderr]\n{result.stderr}"

    output = output.strip()

    command = tuple(
        str(part)
        for part in result.args
    )

    return PostRepairValidation(
        command=command,
        status=_PYTEST_EXIT_STATUS.get(
            result.returncode,
            "error",
        ),
        exit_code=result.returncode,
        passed_count=_parse_count(
            output,
            _PYTEST_COUNT_PATTERNS["passed_count"],
        ),
        failed_count=_parse_count(
            output,
            _PYTEST_COUNT_PATTERNS["failed_count"],
        ),
        error_count=_parse_count(
            output,
            _PYTEST_COUNT_PATTERNS["error_count"],
        ),
        output=output or "(no output)",
    )