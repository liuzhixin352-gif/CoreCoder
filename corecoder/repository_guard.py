"""Repository identity checks for GitHub Issue workflows."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .github_issue import parse_github_issue_url
from .issue_task import IssueTask


_REPOSITORY_COMPONENT_PATTERN = re.compile(
    r"^[A-Za-z0-9_.-]+$"
)


class RepositoryGuardError(ValueError):
    """Raised when repository preflight input is invalid."""


@dataclass(frozen=True)
class GitHubRepository:
    """A normalized GitHub owner/repository reference."""

    owner: str
    repo: str

    @property
    def full_name(self) -> str:
        """Return the normalized owner/repository name."""
        return f"{self.owner}/{self.repo}"

    def same_repository_name(
        self,
        other: "GitHubRepository",
    ) -> bool:
        """Return whether two references share the repository name."""
        return self.repo.casefold() == other.repo.casefold()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GitHubRepository):
            return NotImplemented

        return (
            self.owner.casefold(),
            self.repo.casefold(),
        ) == (
            other.owner.casefold(),
            other.repo.casefold(),
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.owner.casefold(),
                self.repo.casefold(),
            )
        )


@dataclass(frozen=True)
class RepositoryPreflight:
    """Result of comparing an Issue repository with local remotes."""

    target: GitHubRepository
    local_repositories: tuple[GitHubRepository, ...]
    status: str

    @property
    def compatible(self) -> bool:
        """Return whether the current repository can handle the Issue."""
        return self.status in {
            "exact",
            "fork",
        }

@dataclass(frozen=True)
class WorktreePreflight:
    """Result of inspecting the current Git worktree."""

    status: str
    changes: tuple[str, ...]

    @property
    def ready_for_repair(self) -> bool:
        """Return whether a repair can safely modify the worktree."""
        return self.status == "clean"




def _validate_repository_component(
    value: str,
    *,
    field: str,
) -> str:
    value = value.strip()

    if (
        not value
        or not _REPOSITORY_COMPONENT_PATTERN.fullmatch(value)
    ):
        raise RepositoryGuardError(
            f"invalid GitHub repository {field}"
        )

    return value


def parse_github_repository_remote(
    remote_url: str,
) -> GitHubRepository | None:
    """Parse a supported GitHub remote URL.

    Non-GitHub remotes return None.
    """
    if not isinstance(remote_url, str):
        raise RepositoryGuardError(
            "remote_url must be a string"
        )

    value = remote_url.strip()

    if not value:
        raise RepositoryGuardError(
            "remote_url must not be empty"
        )

    path: str

    if value.startswith("git@github.com:"):
        path = value.removeprefix("git@github.com:")
    else:
        parsed = urlparse(value)

        host = (parsed.hostname or "").casefold()

        if host not in {
            "github.com",
            "www.github.com",
        }:
            return None

        path = parsed.path.lstrip("/")

    path = path.rstrip("/")

    if path.endswith(".git"):
        path = path[:-4]

    parts = path.split("/")

    if len(parts) != 2:
        raise RepositoryGuardError(
            "invalid GitHub repository remote URL"
        )

    owner = _validate_repository_component(
        parts[0],
        field="owner",
    )
    repo = _validate_repository_component(
        parts[1],
        field="name",
    )

    return GitHubRepository(
        owner=owner,
        repo=repo,
    )

def check_worktree(
    cwd: str | Path | None = None,
) -> WorktreePreflight:
    """Inspect whether the current Git worktree is clean."""
    try:
        repository_check = subprocess.run(
            [
                "git",
                "rev-parse",
                "--is-inside-work-tree",
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        raise RepositoryGuardError(
            "unable to inspect Git worktree"
        ) from error

    if (
        repository_check.returncode != 0
        or repository_check.stdout.strip().casefold() != "true"
    ):
        return WorktreePreflight(
            status="not_repository",
            changes=(),
        )

    try:
        status_check = subprocess.run(
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
            timeout=5,
            check=False,
        )
    except (
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        raise RepositoryGuardError(
            "unable to inspect Git worktree"
        ) from error

    if status_check.returncode != 0:
        raise RepositoryGuardError(
            "unable to inspect Git worktree"
        )

    changes = tuple(
        line
        for line in status_check.stdout.splitlines()
        if line
    )

    return WorktreePreflight(
        status="dirty" if changes else "clean",
        changes=changes,
    )

def get_local_github_repositories(
    cwd: str | Path | None = None,
) -> tuple[GitHubRepository, ...]:
    """Return unique GitHub repositories configured as local remotes."""
    try:
        completed = subprocess.run(
            [
                "git",
                "remote",
                "-v",
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (
        OSError,
        subprocess.TimeoutExpired,
    ):
        return ()

    if completed.returncode != 0:
        return ()

    repositories = []
    seen = set()

    for line in completed.stdout.splitlines():
        columns = line.split()

        if len(columns) < 2:
            continue

        try:
            repository = parse_github_repository_remote(
                columns[1]
            )
        except RepositoryGuardError:
            continue

        if repository is None:
            continue

        if repository in seen:
            continue

        seen.add(repository)
        repositories.append(repository)

    return tuple(repositories)


def check_issue_repository(
    task: IssueTask,
    *,
    cwd: str | Path | None = None,
) -> RepositoryPreflight:
    """Compare an Issue repository with local GitHub remotes."""
    if not isinstance(task, IssueTask):
        raise RepositoryGuardError(
            "task must be an IssueTask"
        )

    if not task.issue_url:
        raise RepositoryGuardError(
            "task must include an issue_url"
        )

    reference = parse_github_issue_url(
        task.issue_url
    )

    target = GitHubRepository(
        owner=reference.owner,
        repo=reference.repo,
    )

    local_repositories = get_local_github_repositories(
        cwd=cwd
    )

    if target in local_repositories:
        status = "exact"
    elif any(
        target.same_repository_name(repository)
        for repository in local_repositories
    ):
        status = "fork"
    elif local_repositories:
        status = "mismatch"
    else:
        status = "unknown"

    return RepositoryPreflight(
        target=target,
        local_repositories=local_repositories,
        status=status,
    )