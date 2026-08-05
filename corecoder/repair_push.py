"""Push a validated Issue repair branch."""

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepairPush:
    """Remote branch created for one validated repair."""

    remote: str
    branch: str
    commit_sha: str


class RepairPushError(RuntimeError):
    """Raised when a validated repair branch cannot be pushed."""


def _run_git(
    command: list[str],
    *,
    cwd: str | Path | None,
    timeout: int,
    error_message: str,
) -> subprocess.CompletedProcess:
    """Run one Git command or raise a repair push error."""
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
        raise RepairPushError(
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

        raise RepairPushError(message)

    return result


def push_repair_branch(
    branch: str,
    commit_sha: str,
    *,
    cwd: str | Path | None = None,
    timeout: int = 60,
) -> RepairPush:
    """Push a validated repair branch to origin."""
    remote = "origin"

    verification_result = _run_git(
        [
            "git",
            "rev-parse",
            "--verify",
            f"refs/heads/{branch}",
        ],
        cwd=cwd,
        timeout=timeout,
        error_message=(
            f"unable to verify repair branch {branch}"
        ),
    )

    actual_sha = verification_result.stdout.strip()

    if actual_sha != commit_sha:
        raise RepairPushError(
            f"repair branch {branch} does not point "
            "to the validated repair commit"
        )

    _run_git(
        [
            "git",
            "push",
            "--set-upstream",
            remote,
            branch,
        ],
        cwd=cwd,
        timeout=timeout,
        error_message=(
            f"unable to push repair branch {branch}"
        ),
    )

    return RepairPush(
        remote=remote,
        branch=branch,
        commit_sha=commit_sha,
    )