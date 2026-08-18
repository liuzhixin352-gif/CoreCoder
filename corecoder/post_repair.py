"""Inspect Git changes produced by an Issue repair."""

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PostRepairSummary:
    """Git result produced by one Issue repair."""

    branch: str
    changes: tuple[str, ...]

    @property
    def has_changes(self) -> bool:
        """Return whether the repair changed repository files."""
        return bool(self.changes)


class PostRepairSummaryError(RuntimeError):
    """Raised when the post-repair Git result cannot be inspected."""


def collect_post_repair_summary(
    branch: str,
    cwd: str | Path | None = None,
) -> PostRepairSummary:
    """Inspect Git changes left by an Issue repair."""
    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        raise PostRepairSummaryError(
            "unable to inspect post-repair Git changes"
        ) from error

    if result.returncode != 0:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
        )
        message = "unable to inspect post-repair Git changes"

        if detail:
            message = f"{message}: {detail}"

        raise PostRepairSummaryError(message)

    changes = tuple(
        line
        for line in result.stdout.splitlines()
        if line
    )

    return PostRepairSummary(
        branch=branch,
        changes=changes,
    )