"""Create a Git commit for a validated Issue repair."""

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepairCommit:
    """Git commit created for one validated Issue repair."""

    sha: str
    message: str


class RepairCommitError(RuntimeError):
    """Raised when a validated repair cannot be committed."""


def build_repair_commit_message(
    issue_number: int,
    issue_title: str,
) -> str:
    """Build a deterministic commit message for an Issue repair."""
    normalized_title = " ".join(
        issue_title.split()
    )

    if not normalized_title:
        normalized_title = "GitHub Issue repair"

    return (
        f"Fix #{issue_number}: "
        f"{normalized_title}"
    )


def _run_git(
    command: list[str],
    *,
    cwd: str | Path | None,
    timeout: int,
    error_message: str,
) -> subprocess.CompletedProcess:
    """Run one Git command or raise a repair commit error."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        raise RepairCommitError(
            error_message
        ) from error

    if result.returncode != 0:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
        )

        message = error_message

        if detail:
            message = f"{message}: {detail}"

        raise RepairCommitError(message)

    return result


def create_repair_commit(
    issue_number: int,
    issue_title: str,
    *,
    cwd: str | Path | None = None,
    timeout: int = 30,
) -> RepairCommit:
    """Stage and commit all validated repair changes."""
    message = build_repair_commit_message(
        issue_number,
        issue_title,
    )

    _run_git(
        [
            "git",
            "add",
            "--all",
        ],
        cwd=cwd,
        timeout=timeout,
        error_message=(
            "unable to stage repair changes"
        ),
    )

    _run_git(
        [
            "git",
            "commit",
            "--message",
            message,
        ],
        cwd=cwd,
        timeout=timeout,
        error_message=(
            "unable to commit repair changes"
        ),
    )

    sha_result = _run_git(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=cwd,
        timeout=timeout,
        error_message=(
            "unable to read repair commit SHA"
        ),
    )

    commit_sha = sha_result.stdout.strip()

    if not commit_sha:
        raise RepairCommitError(
            "unable to read repair commit SHA: "
            "Git returned an empty SHA"
        )

    return RepairCommit(
        sha=commit_sha,
        message=message,
    )