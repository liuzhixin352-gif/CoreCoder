"""Tests for repository identity preflight checks."""

import subprocess

import pytest
import corecoder.repository_guard as repository_guard
from corecoder.issue_task import IssueTask
from corecoder.repository_guard import (
    GitHubRepository,
    RepositoryGuardError,
    check_issue_repository,
    get_local_github_repositories,
    parse_github_repository_remote,
)


def _task(
    issue_url=(
        "https://github.com/example/project/issues/21"
    ),
):
    return IssueTask(
        title="Fix repository scan limit",
        issue_number=21,
        issue_url=issue_url,
    )


@pytest.mark.parametrize(
    "remote_url",
    [
        "https://github.com/example/project.git",
        "https://github.com/example/project",
        "git@github.com:example/project.git",
        "ssh://git@github.com/example/project.git",
    ],
)
def test_parse_supported_github_remote_urls(remote_url):
    repository = parse_github_repository_remote(
        remote_url
    )

    assert repository == GitHubRepository(
        owner="example",
        repo="project",
    )
    assert repository.full_name == "example/project"


def test_parse_remote_is_case_insensitive_for_identity():
    first = parse_github_repository_remote(
        "https://github.com/Example/Project.git"
    )
    second = parse_github_repository_remote(
        "git@github.com:example/project.git"
    )

    assert first == second
    assert hash(first) == hash(second)


def test_parse_non_github_remote_returns_none():
    assert (
        parse_github_repository_remote(
            "https://gitlab.com/example/project.git"
        )
        is None
    )


@pytest.mark.parametrize(
    "remote_url",
    [
        "",
        "https://github.com/example",
        "https://github.com/example/project/extra",
        "git@github.com:example",
    ],
)
def test_parse_invalid_remote_rejected(remote_url):
    with pytest.raises(RepositoryGuardError):
        parse_github_repository_remote(remote_url)


def test_get_local_repositories_deduplicates_remotes(
    monkeypatch,
):
    output = (
        "origin https://github.com/example/project.git (fetch)\n"
        "origin https://github.com/example/project.git (push)\n"
        "upstream git@github.com:upstream/project.git (fetch)\n"
        "mirror https://gitlab.com/example/project.git (fetch)\n"
    )

    completed = subprocess.CompletedProcess(
        args=["git", "remote", "-v"],
        returncode=0,
        stdout=output,
        stderr="",
    )

    monkeypatch.setattr(
        "corecoder.repository_guard.subprocess.run",
        lambda *args, **kwargs: completed,
    )

    repositories = get_local_github_repositories()

    assert repositories == (
        GitHubRepository(
            owner="example",
            repo="project",
        ),
        GitHubRepository(
            owner="upstream",
            repo="project",
        ),
    )


def test_get_local_repositories_handles_git_failure(
    monkeypatch,
):
    completed = subprocess.CompletedProcess(
        args=["git", "remote", "-v"],
        returncode=128,
        stdout="",
        stderr="not a git repository",
    )

    monkeypatch.setattr(
        "corecoder.repository_guard.subprocess.run",
        lambda *args, **kwargs: completed,
    )

    assert get_local_github_repositories() == ()


def test_get_local_repositories_handles_missing_git(
    monkeypatch,
):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(
        "corecoder.repository_guard.subprocess.run",
        fake_run,
    )

    assert get_local_github_repositories() == ()


@pytest.mark.parametrize(
    ("local_repositories", "expected_status"),
    [
        (
            (
                GitHubRepository(
                    owner="example",
                    repo="project",
                ),
            ),
            "exact",
        ),
        (
            (
                GitHubRepository(
                    owner="my-fork",
                    repo="project",
                ),
            ),
            "fork",
        ),
        (
            (
                GitHubRepository(
                    owner="example",
                    repo="different-project",
                ),
            ),
            "mismatch",
        ),
        (
            (),
            "unknown",
        ),
    ],
)
def test_check_issue_repository_status(
    monkeypatch,
    local_repositories,
    expected_status,
):
    monkeypatch.setattr(
        "corecoder.repository_guard."
        "get_local_github_repositories",
        lambda cwd=None: local_repositories,
    )

    result = check_issue_repository(
        _task()
    )

    assert result.target.full_name == "example/project"
    assert result.local_repositories == local_repositories
    assert result.status == expected_status
    assert result.compatible is (
        expected_status in {"exact", "fork"}
    )


def test_check_issue_repository_rejects_invalid_task():
    with pytest.raises(
        RepositoryGuardError,
        match="task must be an IssueTask",
    ):
        check_issue_repository("not a task")


def test_check_issue_repository_requires_issue_url():
    with pytest.raises(
        RepositoryGuardError,
        match="task must include an issue_url",
    ):
        check_issue_repository(
            IssueTask(title="Missing URL")
        )

def test_check_worktree_reports_clean(monkeypatch):
    responses = iter(
        [
            subprocess.CompletedProcess(
                args=[
                    "git",
                    "rev-parse",
                    "--is-inside-work-tree",
                ],
                returncode=0,
                stdout="true\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[
                    "git",
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ],
                returncode=0,
                stdout="",
                stderr="",
            ),
        ]
    )
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return next(responses)

    monkeypatch.setattr(
        repository_guard.subprocess,
        "run",
        fake_run,
    )

    result = repository_guard.check_worktree(
        cwd="example-repository"
    )

    assert result.status == "clean"
    assert result.changes == ()
    assert result.ready_for_repair is True

    assert [
        args
        for args, _ in calls
    ] == [
        [
            "git",
            "rev-parse",
            "--is-inside-work-tree",
        ],
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
    ]
    assert all(
        kwargs["cwd"] == "example-repository"
        for _, kwargs in calls
    )


def test_check_worktree_reports_dirty_changes(monkeypatch):
    responses = iter(
        [
            subprocess.CompletedProcess(
                args=[
                    "git",
                    "rev-parse",
                    "--is-inside-work-tree",
                ],
                returncode=0,
                stdout="true\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[
                    "git",
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ],
                returncode=0,
                stdout=(
                    " M corecoder/cli.py\n"
                    "M  tests/test_issue_cli.py\n"
                    "?? scratch.txt\n"
                ),
                stderr="",
            ),
        ]
    )

    monkeypatch.setattr(
        repository_guard.subprocess,
        "run",
        lambda *args, **kwargs: next(responses),
    )

    result = repository_guard.check_worktree()

    assert result.status == "dirty"
    assert result.changes == (
        " M corecoder/cli.py",
        "M  tests/test_issue_cli.py",
        "?? scratch.txt",
    )
    assert result.ready_for_repair is False


def test_check_worktree_reports_not_repository(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)

        return subprocess.CompletedProcess(
            args=args,
            returncode=128,
            stdout="",
            stderr=(
                "fatal: not a git repository "
                "(or any parent up to mount point)"
            ),
        )

    monkeypatch.setattr(
        repository_guard.subprocess,
        "run",
        fake_run,
    )

    result = repository_guard.check_worktree()

    assert result.status == "not_repository"
    assert result.changes == ()
    assert result.ready_for_repair is False

    # A failed rev-parse must stop before git status is run.
    assert calls == [
        [
            "git",
            "rev-parse",
            "--is-inside-work-tree",
        ]
    ]


def test_check_worktree_reports_git_execution_failure(
    monkeypatch,
):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("git executable not found")

    monkeypatch.setattr(
        repository_guard.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        RepositoryGuardError,
        match="unable to inspect Git worktree",
    ):
        repository_guard.check_worktree()
