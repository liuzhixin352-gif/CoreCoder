"""Create dedicated Git branches for Issue repairs."""

import re
import subprocess
from pathlib import Path

from .issue_task import IssueTask

_MAX_BRANCH_LENGTH = 80
_UNSAFE_TITLE_PATTERN = re.compile(r"[^a-z0-9]+")


class RepairBranchError(RuntimeError):
    """Raised when a dedicated repair branch cannot be prepared."""


def build_repair_branch_name(
    issue_task: IssueTask,
) -> str:
    """Return a safe dedicated branch name for an Issue repair."""
    if issue_task.issue_number is None:
        raise RepairBranchError(
            "Issue number is required to create a repair branch"
        )

    prefix = f"devpilot/issue-{issue_task.issue_number}"

    ascii_title = (
        issue_task.title
        .strip()
        .casefold()
        .encode("ascii", errors="ignore")
        .decode("ascii")
    )
    slug = _UNSAFE_TITLE_PATTERN.sub(
        "-",
        ascii_title,
    ).strip("-")

    if not slug:
        return prefix

    available_length = (
        _MAX_BRANCH_LENGTH
        - len(prefix)
        - 1
    )
    slug = slug[:available_length].rstrip("-")

    if not slug:
        return prefix

    return f"{prefix}-{slug}"


def create_repair_branch(
    issue_task: IssueTask,
    cwd: str | Path | None = None,
) -> str:
    """Create and switch to a dedicated branch for an Issue repair."""
    branch_name = build_repair_branch_name(issue_task)

    try:
        result = subprocess.run(
            [
                "git",
                "switch",
                "-c",
                branch_name,
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
        raise RepairBranchError(
            f"unable to create repair branch {branch_name}"
        ) from error

    if result.returncode != 0:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
        )
        message = (
            f"unable to create repair branch {branch_name}"
        )

        if detail:
            message = f"{message}: {detail}"

        raise RepairBranchError(message)

    return branch_name