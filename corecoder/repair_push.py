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

    push_command = [
    "git",
    "push",
    "--set-upstream",
    remote,
    branch,
    ]
    push_error_message = (
        f"unable to push repair branch {branch}"
    )

    try:
        _run_git(
            push_command,
            cwd=cwd,
            timeout=timeout,
            error_message=push_error_message,
        )
    except RepairPushError as error:
        error_text = str(error).lower()
        transient_network_errors = (
            "connection was reset",
            "failed to connect to",
            "could not connect to server",
        )

        if not any(
            marker in error_text
            for marker in transient_network_errors
        ):
            raise

        _run_git(
            push_command,
            cwd=cwd,
            timeout=timeout,
            error_message=push_error_message,
        )

    return RepairPush(
        remote=remote,
        branch=branch,
        commit_sha=commit_sha,
    )