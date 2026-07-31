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
from corecoder.repair_branch import RepairBranchError
from corecoder.repository_guard import (
    GitHubRepository,
    RepositoryGuardError,
    RepositoryPreflight,
    WorktreePreflight,
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

def _unknown_repository_preflight() -> RepositoryPreflight:
    """Return an unverified repository preflight result."""
    return RepositoryPreflight(
        target=GitHubRepository(
            owner="example",
            repo="project",
        ),
        local_repositories=(),
        status="unknown",
    )

def _worktree_preflight(
    status: str = "clean",
    changes: tuple[str, ...] = (),
) -> WorktreePreflight:
    """Return a representative Git worktree preflight result."""
    return WorktreePreflight(
        status=status,
        changes=changes,
    )

def test_main_blocks_unknown_repository_in_repair_mode(
    monkeypatch,
):
    task = _sample_task()
    printed = []

    monkeypatch.setattr(
        cli,
        "fetch_github_issue",
        lambda **kwargs: task,
    )
    monkeypatch.setattr(
        cli,
        "check_issue_repository",
        lambda received_task: _unknown_repository_preflight(),
    )

    monkeypatch.setattr(
        cli.Config,
        "from_env",
        classmethod(
            lambda cls: pytest.fail(
                "Configuration must not be loaded when an "
                "unverified repair is blocked"
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "LLM",
        lambda **kwargs: pytest.fail(
            "LLM must not be created when an "
            "unverified repair is blocked"
        ),
    )
    monkeypatch.setattr(
        cli,
        "Agent",
        lambda **kwargs: pytest.fail(
            "Agent must not be created when an "
            "unverified repair is blocked"
        ),
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda *args, **kwargs: pytest.fail(
            "Workflow must not run when an "
            "unverified repair is blocked"
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
    assert "Repository verification required:" in output
    assert "Real Issue repair is blocked by default." in output
    assert "--allow-unverified-repository" in output

def test_main_allows_unknown_repository_in_dry_run(
    monkeypatch,
):
    captured = {}
    printed = []
    _patch_runtime(monkeypatch, captured)

    monkeypatch.setattr(
        cli,
        "fetch_github_issue",
        lambda **kwargs: _sample_task(),
    )
    monkeypatch.setattr(
        cli,
        "check_issue_repository",
        lambda received_task: _unknown_repository_preflight(),
    )
    monkeypatch.setattr(
        cli,
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "unknown dry-run prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: captured.update(
            {"prompt": prompt}
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
            "--dry-run",
        ],
    )

    cli.main()

    assert captured["prompt"] == "unknown dry-run prompt"

    tool_names = {
        tool.name
        for tool in captured["agent_kwargs"]["tools"]
    }
    assert tool_names == {
        "read_file",
        "glob",
        "grep",
        "repo_map",
    }

    output = "\n".join(printed)
    assert "Continuing in read-only dry-run mode." in output

def test_main_allows_unknown_repository_with_override(
    monkeypatch,
):
    captured = {}
    printed = []
    _patch_runtime(monkeypatch, captured)

    monkeypatch.setattr(
        cli,
        "fetch_github_issue",
        lambda **kwargs: _sample_task(),
    )
    monkeypatch.setattr(
        cli,
        "check_issue_repository",
        lambda received_task: _unknown_repository_preflight(),
    )
    monkeypatch.setattr(
        cli,
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "unverified repair prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: captured.update(
            {"prompt": prompt}
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
            "--allow-unverified-repository",
        ],
    )

    cli.main()

    assert captured["prompt"] == "unverified repair prompt"

    # Repair mode receives the normal complete tool profile.
    assert captured["agent_kwargs"]["tools"] is None

    output = "\n".join(printed)
    assert "Repository verification override:" in output
    assert "--allow-unverified-repository was provided." in output

def test_main_blocks_dirty_worktree_before_runtime(
    monkeypatch,
):
    task = _sample_task()
    printed = []

    monkeypatch.setattr(
        cli,
        "fetch_github_issue",
        lambda **kwargs: task,
    )
    monkeypatch.setattr(
        cli,
        "check_issue_repository",
        lambda received_task: _repository_preflight(),
    )
    monkeypatch.setattr(
        cli,
        "check_worktree",
        lambda: _worktree_preflight(
            status="dirty",
            changes=(
                " M corecoder/cli.py",
                "?? scratch.txt",
            ),
        ),
        raising=False,
    )

    monkeypatch.setattr(
        cli.Config,
        "from_env",
        classmethod(
            lambda cls: pytest.fail(
                "Configuration must not be loaded when "
                "a dirty worktree blocks repair"
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "LLM",
        lambda **kwargs: pytest.fail(
            "LLM must not be created when "
            "a dirty worktree blocks repair"
        ),
    )
    monkeypatch.setattr(
        cli,
        "Agent",
        lambda **kwargs: pytest.fail(
            "Agent must not be created when "
            "a dirty worktree blocks repair"
        ),
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda *args, **kwargs: pytest.fail(
            "Workflow must not run when "
            "a dirty worktree blocks repair"
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
    assert "Clean worktree required:" in output
    assert "uncommitted changes" in output
    assert "corecoder/cli.py" in output
    assert "scratch.txt" in output

def test_main_creates_repair_branch_before_runtime(
    monkeypatch,
):
    task = _sample_task()
    captured = {}
    events = []
    printed = []

    _patch_runtime(monkeypatch, captured)

    monkeypatch.setattr(
        cli,
        "fetch_github_issue",
        lambda **kwargs: task,
    )
    monkeypatch.setattr(
        cli,
        "check_issue_repository",
        lambda received_task: _repository_preflight(),
    )

    def fake_check_worktree():
        events.append("worktree")
        return _worktree_preflight()

    def fake_create_repair_branch(received_task):
        assert received_task is task
        assert events == ["worktree"]
        events.append("branch")
        return "devpilot/issue-21-fix-repository-scan-limit"

    def fake_config_from_env(cls):
        assert events == [
            "worktree",
            "branch",
        ]
        events.append("config")
        return Config(
            model="test-model",
            api_key="test-api-key",
        )

    monkeypatch.setattr(
        cli,
        "check_worktree",
        fake_check_worktree,
    )
    monkeypatch.setattr(
        cli,
        "create_repair_branch",
        fake_create_repair_branch,
        raising=False,
    )
    monkeypatch.setattr(
        cli.Config,
        "from_env",
        classmethod(fake_config_from_env),
    )
    monkeypatch.setattr(
        cli,
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "repair branch prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: captured.update(
            {"prompt": prompt}
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

    cli.main()

    assert events == [
        "worktree",
        "branch",
        "config",
    ]
    assert captured["prompt"] == "repair branch prompt"

    output = "\n".join(printed)
    assert "Repair branch created:" in output
    assert (
        "devpilot/issue-21-fix-repository-scan-limit"
        in output
    )


def test_main_skips_repair_branch_in_dry_run(
    monkeypatch,
):
    captured = {}

    _patch_runtime(monkeypatch, captured)

    monkeypatch.setattr(
        cli,
        "fetch_github_issue",
        lambda **kwargs: _sample_task(),
    )
    monkeypatch.setattr(
        cli,
        "check_issue_repository",
        lambda received_task: _repository_preflight(),
    )
    monkeypatch.setattr(
        cli,
        "create_repair_branch",
        lambda issue_task: pytest.fail(
            "Dry-run must not create a repair branch"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "dry-run branch prompt",
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

    assert captured["prompt"] == "dry-run branch prompt"


def test_main_reports_repair_branch_error_before_runtime(
    monkeypatch,
):
    task = _sample_task()
    printed = []

    monkeypatch.setattr(
        cli,
        "fetch_github_issue",
        lambda **kwargs: task,
    )
    monkeypatch.setattr(
        cli,
        "check_issue_repository",
        lambda received_task: _repository_preflight(),
    )
    monkeypatch.setattr(
        cli,
        "check_worktree",
        lambda: _worktree_preflight(),
    )

    def fake_create_repair_branch(received_task):
        raise RepairBranchError(
            "unable to create repair branch"
        )

    monkeypatch.setattr(
        cli,
        "create_repair_branch",
        fake_create_repair_branch,
        raising=False,
    )
    monkeypatch.setattr(
        cli.Config,
        "from_env",
        classmethod(
            lambda cls: pytest.fail(
                "Configuration must not be loaded when "
                "repair branch creation fails"
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "LLM",
        lambda **kwargs: pytest.fail(
            "LLM must not be created when "
            "repair branch creation fails"
        ),
    )
    monkeypatch.setattr(
        cli,
        "Agent",
        lambda **kwargs: pytest.fail(
            "Agent must not be created when "
            "repair branch creation fails"
        ),
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda *args, **kwargs: pytest.fail(
            "Workflow must not run when "
            "repair branch creation fails"
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
    assert "Issue workflow error:" in output
    assert "unable to create repair branch" in output

def test_main_blocks_repair_outside_git_worktree(
    monkeypatch,
):
    task = _sample_task()
    printed = []

    monkeypatch.setattr(
        cli,
        "fetch_github_issue",
        lambda **kwargs: task,
    )
    monkeypatch.setattr(
        cli,
        "check_issue_repository",
        lambda received_task: _repository_preflight(),
    )
    monkeypatch.setattr(
        cli,
        "check_worktree",
        lambda: _worktree_preflight(
            status="not_repository",
        ),
        raising=False,
    )

    monkeypatch.setattr(
        cli.Config,
        "from_env",
        classmethod(
            lambda cls: pytest.fail(
                "Configuration must not be loaded outside "
                "a Git worktree"
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "LLM",
        lambda **kwargs: pytest.fail(
            "LLM must not be created outside "
            "a Git worktree"
        ),
    )
    monkeypatch.setattr(
        cli,
        "Agent",
        lambda **kwargs: pytest.fail(
            "Agent must not be created outside "
            "a Git worktree"
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
    assert "Git worktree required:" in output
    assert "inside a Git worktree" in output


def test_main_checks_clean_worktree_before_repair(
    monkeypatch,
):
    captured = {}
    worktree_calls = []

    _patch_runtime(monkeypatch, captured)

    monkeypatch.setattr(
        cli,
        "fetch_github_issue",
        lambda **kwargs: _sample_task(),
    )
    monkeypatch.setattr(
        cli,
        "check_issue_repository",
        lambda received_task: _repository_preflight(),
    )

    def fake_check_worktree():
        worktree_calls.append(True)
        return _worktree_preflight()

    monkeypatch.setattr(
        cli,
        "check_worktree",
        fake_check_worktree,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "clean repair prompt",
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
        ],
    )

    cli.main()

    assert worktree_calls == [True]
    assert captured["prompt"] == "clean repair prompt"


def test_main_skips_worktree_check_in_dry_run(
    monkeypatch,
):
    captured = {}

    _patch_runtime(monkeypatch, captured)

    monkeypatch.setattr(
        cli,
        "fetch_github_issue",
        lambda **kwargs: _sample_task(),
    )
    monkeypatch.setattr(
        cli,
        "check_issue_repository",
        lambda received_task: _repository_preflight(),
    )
    monkeypatch.setattr(
        cli,
        "check_worktree",
        lambda: pytest.fail(
            "Dry-run must not inspect worktree cleanliness"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "dry-run prompt",
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

    assert captured["prompt"] == "dry-run prompt"

def _patch_runtime(
    monkeypatch,
    captured: dict | None = None,
) -> None:
    """Replace configuration, LLM, and Agent construction."""
    monkeypatch.setattr(
        cli,
        "check_worktree",
        lambda: _worktree_preflight(),
    )
    monkeypatch.setattr(
        cli,
        "create_repair_branch",
        lambda issue_task: (
            "devpilot/issue-21-fix-repository-scan-limit"
        ),
        raising=False,
    )
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


def test_parse_args_accepts_allow_unverified_repository_with_issue(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--issue",
            _ISSUE_URL,
            "--allow-unverified-repository",
        ],
    )

    args = cli._parse_args()

    assert args.issue == _ISSUE_URL
    assert args.allow_unverified_repository is True


def test_parse_args_rejects_allow_unverified_repository_without_issue(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--allow-unverified-repository",
        ],
    )

    with pytest.raises(SystemExit) as error:
        cli._parse_args()

    assert error.value.code == 2
    assert (
        "--allow-unverified-repository requires --issue"
        in capsys.readouterr().err
    )


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

@pytest.mark.parametrize(
    "extra_args",
    [
        [],
        ["--allow-unverified-repository"],
    ],
)
def test_main_stops_before_runtime_on_repository_mismatch(
    monkeypatch,
    extra_args,
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
            *extra_args,
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


# ---------------------------------------------------------------------------
# _brief preview formatting
# ---------------------------------------------------------------------------

SHORT_URL = "https://example.com/short"
LONG_URL = (
    "https://github.com/python/cpython/issues/"
    "123456789012345678901234567890"
)


def test_brief_short_value():
    """Short string values are not truncated."""
    result = cli._brief({"url": SHORT_URL})
    # repr(SHORT_URL) = "'https://example.com/short'"  (28 chars, well under 40)
    assert "'https://example.com/short'" in result
    assert result == "url='https://example.com/short'"
    assert len(result) <= 80


def test_brief_long_url():
    """A single long URL is safely truncated with a closing quote preserved."""
    result = cli._brief({"issue_url": LONG_URL})
    # The repr of LONG_URL exceeds 40 characters; the inner content should
    # be shortened with "..." while the outer quotes remain intact.
    assert result.startswith("issue_url=")
    assert result.count("'") == 2, "closing quote must be present"
    assert "..." in result
    assert len(result) <= 80


def test_brief_multiple_args():
    """Multiple arguments remain distinguishable in the preview."""
    result = cli._brief(
        {
            "owner": "python",
            "repo": "cpython",
            "issue_url": LONG_URL,
        }
    )
    assert result.startswith("owner=")
    assert "'python'" in result
    assert "'cpython'" in result
    assert "issue_url=" in result
    # Every value keeps its quotes
    assert result.count("'") >= 4
    assert len(result) <= 80


def test_brief_small_maxlen():
    """With a small maxlen the result is truncated and ends with '...'."""
    result = cli._brief({"issue_url": LONG_URL}, maxlen=20)
    assert len(result) <= 20
    assert result.endswith("...")


def test_brief_multiple_args_respects_maxlen():
    """Multiple args are still capped by maxlen."""
    result = cli._brief(
        {
            "owner": "python",
            "repo": "cpython",
            "issue_url": LONG_URL,
            "extra": "x" * 100,
        },
        maxlen=50,
    )
    assert len(result) <= 50
    assert result.endswith("...")


def test_brief_empty():
    """Empty kwargs produce an empty string."""
    assert cli._brief({}) == ""


def test_brief_non_string_values():
    """Non-string values (int, float, None, bool) are displayed unchanged."""
    result = cli._brief(
        {
            "count": 42,
            "ratio": 3.14,
            "enabled": True,
            "callback": None,
        }
    )
    assert "count=42" in result
    assert "ratio=3.14" in result
    assert "enabled=True" in result
    assert "callback=None" in result
    assert len(result) <= 80


def test_brief_small_maxlen_does_not_leave_open_quote():
    result = cli._brief(
        {
            "issue_url": LONG_URL,
        },
        maxlen=20,
    )

    assert len(result) <= 20
    assert result.endswith("...")
    assert result.count("'") % 2 == 0

def test_brief_truncation_handles_escaped_quotes():
    value = (
        'He said "don\'t" and then continued '
        "with a very long explanation"
    )

    result = cli._brief(
        {
            "text": value,
        },
        maxlen=30,
    )

    assert len(result) <= 30
    assert result == "text=..."
    assert not result.endswith("'...")
    assert not result.endswith('"...')