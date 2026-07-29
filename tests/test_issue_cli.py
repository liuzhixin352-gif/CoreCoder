"""Tests for the GitHub Issue CLI workflow."""

import sys

import pytest

import corecoder.cli as cli
from corecoder.config import Config
from corecoder.github_issue import (
    GitHubIssueFetchError,
    GitHubIssueReferenceError,
)
from corecoder.issue_task import IssueTask
from corecoder.issue_workflow import IssueWorkflowError
from corecoder.repository_guard import (
    GitHubRepository,
    RepositoryGuardError,
    RepositoryPreflight,
)

_ISSUE_URL = "https://github.com/example/project/issues/21"


def _sample_task() -> IssueTask:
    """Return a representative structured Issue task."""
    return IssueTask(
        title="Fix repository scan limit",
        body="Update corecoder/tools/repo_map.py.",
        labels=["bug", "agent"],
        issue_number=21,
        issue_url=_ISSUE_URL,
        acceptance_criteria=[
            "max_files must limit scanned files",
            "the result must report truncated",
        ],
        referenced_files=[
            "corecoder/tools/repo_map.py",
        ],
        ambiguities=[],
    )

def _repository_preflight(
    status: str = "exact",
) -> RepositoryPreflight:
    """Return a representative repository preflight result."""
    return RepositoryPreflight(
        target=GitHubRepository(
            owner="example",
            repo="project",
        ),
        local_repositories=(
            GitHubRepository(
                owner="example",
                repo="project",
            ),
        ),
        status=status,
    )

def _patch_runtime(
    monkeypatch,
    captured: dict | None = None,
) -> None:
    """Replace configuration, LLM, and Agent construction."""
    monkeypatch.setattr(
        cli.Config,
        "from_env",
        classmethod(
            lambda cls: Config(
                model="test-model",
                api_key="test-api-key",
            )
        ),
    )

    monkeypatch.setattr(
        cli,
        "LLM",
        lambda **kwargs: object(),
    )

    def fake_agent(**kwargs):
        if captured is not None:
            captured["agent_kwargs"] = kwargs

        return object()

    monkeypatch.setattr(
        cli,
        "Agent",
        fake_agent,
    )


def test_parse_args_accepts_issue_dry_run(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--issue",
            _ISSUE_URL,
            "--dry-run",
        ],
    )

    args = cli._parse_args()

    assert args.issue == _ISSUE_URL
    assert args.dry_run is True
    assert args.prompt is None


def test_parse_args_rejects_dry_run_without_issue(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit) as error:
        cli._parse_args()

    assert error.value.code == 2
    assert "--dry-run requires --issue" in capsys.readouterr().err


def test_parse_args_rejects_prompt_and_issue_together(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--prompt",
            "hello",
            "--issue",
            _ISSUE_URL,
        ],
    )

    with pytest.raises(SystemExit) as error:
        cli._parse_args()

    assert error.value.code == 2

    stderr = capsys.readouterr().err

    assert "not allowed with argument" in stderr
    assert "--issue" in stderr


@pytest.mark.parametrize(
    ("extra_args", "expected_dry_run"),
    [
        ([], False),
        (["--dry-run"], True),
    ],
)
def test_main_runs_issue_workflow(
    monkeypatch,
    extra_args,
    expected_dry_run,
):
    captured = {}
    _patch_runtime(monkeypatch,captured)

    task = _sample_task()
    monkeypatch.setattr(
        cli,
        "check_issue_repository",
        lambda task: _repository_preflight(),
    )

    def fake_fetch_github_issue(**kwargs):
        captured["fetch_kwargs"] = kwargs
        return task

    def fake_build_issue_repair_prompt(
        received_task,
        *,
        dry_run=False,
        max_files=200,
    ):
        captured["task"] = received_task
        captured["dry_run"] = dry_run
        captured["max_files"] = max_files
        return "generated workflow prompt"

    def fake_run_once(agent, prompt):
        captured["agent"] = agent
        captured["prompt"] = prompt

    monkeypatch.setattr(
        cli,
        "fetch_github_issue",
        fake_fetch_github_issue,
    )
    monkeypatch.setattr(
        cli,
        "build_issue_repair_prompt",
        fake_build_issue_repair_prompt,
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        fake_run_once,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--issue",
            _ISSUE_URL,
            *extra_args,
        ],
    )

    cli.main()

    assert captured["fetch_kwargs"] == {
        "issue_url": _ISSUE_URL,
    }
    assert captured["task"] is task
    assert captured["dry_run"] is expected_dry_run
    assert captured["max_files"] == 200
    assert captured["prompt"] == "generated workflow prompt"
    agent_tools = captured["agent_kwargs"]["tools"]

    if expected_dry_run:
        assert [
            tool.name
            for tool in agent_tools
        ] == [
            "read_file",
            "glob",
            "grep",
            "repo_map",
        ]
    else:
        assert agent_tools is None


@pytest.mark.parametrize(
    "workflow_error",
    [
        GitHubIssueReferenceError("invalid Issue URL"),
        GitHubIssueFetchError("Issue not found"),
    ],
)
def test_main_reports_issue_fetch_errors(
    monkeypatch,
    workflow_error,
):
    _patch_runtime(monkeypatch)

    printed = []

    def fake_fetch_github_issue(**kwargs):
        raise workflow_error

    def fail_run_once(agent, prompt):
        pytest.fail("_run_once must not run after an Issue fetch error")

    monkeypatch.setattr(
        cli,
        "fetch_github_issue",
        fake_fetch_github_issue,
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        fail_run_once,
    )
    monkeypatch.setattr(
        cli.console,
        "print",
        lambda *args, **kwargs: printed.append(
            " ".join(str(value) for value in args)
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--issue",
            _ISSUE_URL,
        ],
    )

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 1
    assert any(
        "Issue workflow error:" in message
        for message in printed
    )
    assert any(
        str(workflow_error) in message
        for message in printed
    )


def test_main_reports_prompt_construction_error(
    monkeypatch,
):
    _patch_runtime(monkeypatch)

    printed = []

    monkeypatch.setattr(
        cli,
        "fetch_github_issue",
        lambda **kwargs: _sample_task(),
    )

    monkeypatch.setattr(
        cli,
        "check_issue_repository",
        lambda task: _repository_preflight(),
    )

    def fake_build_issue_repair_prompt(*args, **kwargs):
        raise IssueWorkflowError("invalid workflow configuration")

    monkeypatch.setattr(
        cli,
        "build_issue_repair_prompt",
        fake_build_issue_repair_prompt,
    )
    monkeypatch.setattr(
        cli.console,
        "print",
        lambda *args, **kwargs: printed.append(
            " ".join(str(value) for value in args)
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--issue",
            _ISSUE_URL,
        ],
    )

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 1
    assert any(
        "invalid workflow configuration" in message
        for message in printed
    )


def test_main_preserves_normal_prompt_mode(monkeypatch):
    _patch_runtime(monkeypatch)

    captured = {}

    def fail_fetch_github_issue(**kwargs):
        pytest.fail(
            "Normal prompt mode must not fetch a GitHub Issue"
        )

    def fake_run_once(agent, prompt):
        captured["prompt"] = prompt

    monkeypatch.setattr(
        cli,
        "fetch_github_issue",
        fail_fetch_github_issue,
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        fake_run_once,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--prompt",
            "Explain this repository",
        ],
    )

    cli.main()

    assert captured["prompt"] == "Explain this repository"


def test_main_preserves_interactive_mode(monkeypatch):
    _patch_runtime(monkeypatch)

    captured = {}

    def fake_repl(agent, config):
        captured["agent"] = agent
        captured["config"] = config

    monkeypatch.setattr(
        cli,
        "_repl",
        fake_repl,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["corecoder"],
    )

    cli.main()

    assert captured["agent"] is not None
    assert captured["config"].api_key == "test-api-key"


def test_dry_run_tool_profile_excludes_dangerous_tools():
    tools = cli._tools_for_issue_workflow(
        dry_run=True,
    )
    tool_names = {
        tool.name
        for tool in tools
    }

    assert tool_names == {
        "read_file",
        "glob",
        "grep",
        "repo_map",
    }

    assert tool_names.isdisjoint(
        {
            "bash",
            "write_file",
            "edit_file",
            "agent",
            "run_tests",
            "parse_issue",
            "fetch_issue",
        }
    )


def test_repair_tool_profile_uses_default_tools():
    assert (
        cli._tools_for_issue_workflow(
            dry_run=False,
        )
        is None
    )

def test_main_stops_before_runtime_on_repository_mismatch(
    monkeypatch,
):
    task = _sample_task()
    printed = []

    monkeypatch.setattr(
        cli,
        "fetch_github_issue",
        lambda **kwargs: task,
    )

    mismatch = RepositoryPreflight(
        target=GitHubRepository(
            owner="example",
            repo="project",
        ),
        local_repositories=(
            GitHubRepository(
                owner="other",
                repo="different-project",
            ),
        ),
        status="mismatch",
    )

    monkeypatch.setattr(
        cli,
        "check_issue_repository",
        lambda received_task: mismatch,
    )

    monkeypatch.setattr(
        cli.Config,
        "from_env",
        classmethod(
            lambda cls: pytest.fail(
                "Configuration must not be loaded "
                "after a repository mismatch"
            )
        ),
    )

    monkeypatch.setattr(
        cli,
        "LLM",
        lambda **kwargs: pytest.fail(
            "LLM must not be created "
            "after a repository mismatch"
        ),
    )

    monkeypatch.setattr(
        cli,
        "Agent",
        lambda **kwargs: pytest.fail(
            "Agent must not be created "
            "after a repository mismatch"
        ),
    )

    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda *args, **kwargs: pytest.fail(
            "Workflow must not run "
            "after a repository mismatch"
        ),
    )

    monkeypatch.setattr(
        cli.console,
        "print",
        lambda *args, **kwargs: printed.append(
            " ".join(str(value) for value in args)
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--issue",
            _ISSUE_URL,
        ],
    )

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 1

    output = "\n".join(printed)

    assert "Repository mismatch:" in output
    assert "example/project" in output
    assert "other/different-project" in output

def test_main_allows_matching_fork(monkeypatch):
    captured = {}
    _patch_runtime(monkeypatch, captured)

    task = _sample_task()

    fork_preflight = RepositoryPreflight(
        target=GitHubRepository(
            owner="upstream",
            repo="project",
        ),
        local_repositories=(
            GitHubRepository(
                owner="my-account",
                repo="project",
            ),
        ),
        status="fork",
    )

    monkeypatch.setattr(
        cli,
        "fetch_github_issue",
        lambda **kwargs: task,
    )
    monkeypatch.setattr(
        cli,
        "check_issue_repository",
        lambda received_task: fork_preflight,
    )
    monkeypatch.setattr(
        cli,
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "fork workflow prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: captured.update(
            {"prompt": prompt}
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--issue",
            _ISSUE_URL,
            "--dry-run",
        ],
    )

    cli.main()

    assert captured["prompt"] == "fork workflow prompt"


def test_main_reports_repository_preflight_error(
    monkeypatch,
):
    task = _sample_task()
    printed = []

    monkeypatch.setattr(
        cli,
        "fetch_github_issue",
        lambda **kwargs: task,
    )

    def fake_check_issue_repository(task):
        raise RepositoryGuardError(
            "unable to determine Issue repository"
        )

    monkeypatch.setattr(
        cli,
        "check_issue_repository",
        fake_check_issue_repository,
    )

    monkeypatch.setattr(
        cli.console,
        "print",
        lambda *args, **kwargs: printed.append(
            " ".join(str(value) for value in args)
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--issue",
            _ISSUE_URL,
        ],
    )

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 1
    assert any(
        "unable to determine Issue repository" in message
        for message in printed
    )