"""Tests for the GtHub Issue CLI workflow."""

import sys
from json import JSONDecodeError
import pytest
from corecoder.repair_ci import (
    RepairCIStatus,
    RepairCheckRun,
    RepairCIStatusError,
)
import corecoder.cli as cli
from corecoder.config import Config
from corecoder.github_issue import (
    GitHubIssueFetchError,
    GitHubIssueReferenceError,
)
from corecoder.issue_task import IssueTask
from corecoder.issue_workflow import IssueWorkflowError
from corecoder.post_repair import (
    PostRepairSummary,
    PostRepairSummaryError,
)
from corecoder.issue_orchestration import (
    WorkflowCheckpoint,
    WorkflowState,
)
from corecoder.post_repair_validation import (
    PostRepairValidation,
    PostRepairValidationError,
)
from corecoder.repair_commit import (
    RepairCommit,
    RepairCommitError,
)
from corecoder.repair_push import (
    RepairPush,
    RepairPushError,
)
from corecoder.repair_pr import (
    RepairPullRequest,
    RepairPullRequestError,
)
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

    def fake_get_current_branch():
        assert events == [
            "worktree",
        ]
        events.append("base_branch")
        return "devpilot-v1"

    def fake_create_repair_branch(received_task):
        assert received_task is task
        assert events == [
        "worktree",
        "base_branch",
        ]
        events.append("branch")
        return "devpilot/issue-21-fix-repository-scan-limit"

    def fake_config_from_env(cls):
        assert events == [
            "worktree",
            "base_branch",
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
    "get_current_branch",
    fake_get_current_branch,
    raising=False,
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
        "base_branch",
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

    monkeypatch.setattr(
        cli,
        "get_current_branch",
        lambda: "devpilot-v1",
        raising=False,
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

def test_main_prints_post_repair_changes_after_repair(
    monkeypatch,
):
    captured = {}
    events = []
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
        lambda received_task: _repository_preflight(),
    )
    monkeypatch.setattr(
        cli,
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "post-repair prompt",
    )

    def fake_run_once(agent, prompt):
        events.append("run")
        captured["prompt"] = prompt

    def fake_collect_post_repair_summary(branch):
        assert events == ["run"]
        events.append("summary")

        return PostRepairSummary(
            branch=branch,
            changes=(
                " M corecoder/cli.py",
                "?? tests/test_example.py",
            ),
        )

    monkeypatch.setattr(
        cli,
        "_run_once",
        fake_run_once,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        fake_collect_post_repair_summary,
        raising=False,
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
        "run",
        "summary",
    ]
    assert captured["prompt"] == "post-repair prompt"

    output = "\n".join(printed)
    assert "Post-repair summary" in output
    assert "Repair branch:" in output
    assert (
        "devpilot/issue-21-fix-repository-scan-limit"
        in output
    )
    assert "Changed files:" in output
    assert " M corecoder/cli.py" in output
    assert "?? tests/test_example.py" in output


def test_main_reports_when_repair_produces_no_changes(
    monkeypatch,
):
    printed = []

    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "no-change prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        lambda branch: PostRepairSummary(
            branch=branch,
            changes=(),
        ),
        raising=False,
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

    output = "\n".join(printed)
    assert "Post-repair summary" in output
    assert "No repository changes were produced." in output


def test_main_skips_post_repair_summary_in_dry_run(
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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "dry-run summary prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: captured.update(
            {"prompt": prompt}
        ),
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        lambda branch: pytest.fail(
            "Dry-run must not collect a post-repair summary"
        ),
        raising=False,
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

    assert captured["prompt"] == "dry-run summary prompt"


def test_main_reports_post_repair_summary_error(
    monkeypatch,
):
    events = []
    printed = []

    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "summary error prompt",
    )

    def fake_run_once(agent, prompt):
        events.append("run")

    def fake_collect_post_repair_summary(branch):
        assert events == ["run"]

        raise PostRepairSummaryError(
            "unable to inspect post-repair Git changes"
        )

    monkeypatch.setattr(
        cli,
        "_run_once",
        fake_run_once,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        fake_collect_post_repair_summary,
        raising=False,
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
    assert events == ["run"]

    output = "\n".join(printed)
    assert "Post-repair summary error:" in output
    assert "unable to inspect post-repair Git changes" in output

def test_main_runs_post_repair_validation_after_changed_summary(
    monkeypatch,
):
    events = []
    printed = []

    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "validation prompt",
    )

    def fake_run_once(agent, prompt):
        events.append("run")

    def fake_collect_post_repair_summary(branch):
        assert events == ["run"]
        events.append("summary")

        return PostRepairSummary(
            branch=branch,
            changes=(
                " M corecoder/cli.py",
            ),
        )

    def fake_run_post_repair_validation():
        assert events == [
            "run",
            "summary",
        ]
        events.append("validation")

        return PostRepairValidation(
            command=(
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
            ),
            status="passed",
            exit_code=0,
            passed_count=264,
            failed_count=0,
            error_count=0,
            output="264 passed in 38.28s",
        )

    monkeypatch.setattr(
        cli,
        "_run_once",
        fake_run_once,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        fake_collect_post_repair_summary,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "run_post_repair_validation",
        fake_run_post_repair_validation,
        raising=False,
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
        "run",
        "summary",
        "validation",
    ]

    output = "\n".join(printed)
    assert "Post-repair validation" in output
    assert "Command:" in output
    assert "pytest" in output
    assert "Status:" in output
    assert "passed" in output
    assert "Passed:" in output
    assert "264" in output


def test_main_skips_post_repair_validation_without_changes(
    monkeypatch,
):
    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "no-change validation prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        lambda branch: PostRepairSummary(
            branch=branch,
            changes=(),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "run_post_repair_validation",
        lambda: pytest.fail(
            "Validation must not run without repository changes"
        ),
        raising=False,
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


def test_main_skips_post_repair_validation_in_dry_run(
    monkeypatch,
):
    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "dry-run validation prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
    )
    monkeypatch.setattr(
        cli,
        "run_post_repair_validation",
        lambda: pytest.fail(
            "Dry-run must not run post-repair validation"
        ),
        raising=False,
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


def test_main_exits_when_post_repair_validation_fails(
    monkeypatch,
):
    printed = []

    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "failed validation prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        lambda branch: PostRepairSummary(
            branch=branch,
            changes=(
                " M corecoder/cli.py",
            ),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "run_post_repair_validation",
        lambda: PostRepairValidation(
            command=(
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
            ),
            status="failed",
            exit_code=1,
            passed_count=10,
            failed_count=2,
            error_count=0,
            output="2 failed, 10 passed",
        ),
        raising=False,
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
    assert "Post-repair validation" in output
    assert "Status:" in output
    assert "failed" in output
    assert "Passed:" in output
    assert "10" in output
    assert "Failed:" in output
    assert "2" in output


def test_main_reports_post_repair_validation_execution_error(
    monkeypatch,
):
    printed = []

    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "validation error prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        lambda branch: PostRepairSummary(
            branch=branch,
            changes=(
                " M corecoder/cli.py",
            ),
        ),
        raising=False,
    )

    def fake_run_post_repair_validation():
        raise PostRepairValidationError(
            "unable to run post-repair validation"
        )

    monkeypatch.setattr(
        cli,
        "run_post_repair_validation",
        fake_run_post_repair_validation,
        raising=False,
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
    assert "Post-repair validation error:" in output
    assert "unable to run post-repair validation" in output

def test_main_creates_commit_after_successful_validation(
    monkeypatch,
):
    events = []
    printed = []
    commit_sha = (
        "0123456789abcdef"
        "0123456789abcdef"
        "01234567"
    )

    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "commit prompt",
    )

    def fake_run_once(agent, prompt):
        events.append("run")

    def fake_collect_post_repair_summary(branch):
        assert events == ["run"]
        events.append("summary")

        return PostRepairSummary(
            branch=branch,
            changes=(
                " M corecoder/cli.py",
            ),
        )

    def fake_run_post_repair_validation():
        assert events == [
            "run",
            "summary",
        ]
        events.append("validation")

        return PostRepairValidation(
            command=(
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
            ),
            status="passed",
            exit_code=0,
            passed_count=275,
            failed_count=0,
            error_count=0,
            output="275 passed",
        )

    def fake_create_repair_commit(
        issue_number,
        issue_title,
    ):
        assert events == [
            "run",
            "summary",
            "validation",
        ]
        assert issue_number == 21
        assert issue_title == "Fix repository scan limit"

        events.append("commit")

        return RepairCommit(
            sha=commit_sha,
            message=(
                "Fix #21: Fix repository scan limit"
            ),
        )

    monkeypatch.setattr(
        cli,
        "_run_once",
        fake_run_once,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        fake_collect_post_repair_summary,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "run_post_repair_validation",
        fake_run_post_repair_validation,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "create_repair_commit",
        fake_create_repair_commit,
        raising=False,
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
        "run",
        "summary",
        "validation",
        "commit",
    ]

    output = "\n".join(printed)
    assert "Repair commit created" in output
    assert "Commit:" in output
    assert commit_sha in output
    assert "Message:" in output
    assert (
        "Fix #21: Fix repository scan limit"
        in output
    )
def test_main_pushes_repair_branch_after_commit(
    monkeypatch,
):
    events = []
    printed = []
    branch = (
        "devpilot/"
        "issue-21-fix-repository-scan-limit"
    )
    commit_sha = (
        "0123456789abcdef"
        "0123456789abcdef"
        "01234567"
    )

    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "push prompt",
    )

    def fake_run_once(agent, prompt):
        events.append("run")

    def fake_collect_post_repair_summary(
        received_branch,
    ):
        assert received_branch == branch
        assert events == ["run"]
        events.append("summary")

        return PostRepairSummary(
            branch=received_branch,
            changes=(
                " M corecoder/cli.py",
            ),
        )

    def fake_run_post_repair_validation():
        assert events == [
            "run",
            "summary",
        ]
        events.append("validation")

        return PostRepairValidation(
            command=(
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
            ),
            status="passed",
            exit_code=0,
            passed_count=288,
            failed_count=0,
            error_count=0,
            output="288 passed",
        )

    def fake_create_repair_commit(
        issue_number,
        issue_title,
    ):
        assert events == [
            "run",
            "summary",
            "validation",
        ]
        events.append("commit")

        return RepairCommit(
            sha=commit_sha,
            message=(
                "Fix #21: "
                "Fix repository scan limit"
            ),
        )

    def fake_push_repair_branch(
        received_branch,
        received_commit_sha,
    ):
        assert events == [
            "run",
            "summary",
            "validation",
            "commit",
        ]
        assert received_branch == branch
        assert received_commit_sha == commit_sha
        events.append("push")

        return RepairPush(
            remote="origin",
            branch=received_branch,
            commit_sha=received_commit_sha,
        )

    def fake_create_repair_pull_request(
    repository,
    issue_number,
    issue_title,
    received_repair_push,
    base_branch,
):
        assert events == [
            "run",
            "summary",
            "validation",
            "commit",
            "push",
        ]
        assert repository == "example/project"
        assert issue_number == 21
        assert issue_title == "Fix repository scan limit"
        assert received_repair_push == RepairPush(
            remote="origin",
            branch=branch,
            commit_sha=commit_sha,
        )
        assert base_branch == "devpilot-v1"

        events.append("pull_request")

        return RepairPullRequest(
            repository=repository,
            number=42,
            url=(
                "https://github.com/"
                "example/project/pull/42"
            ),
            title=(
                "Fix #21: "
                "Fix repository scan limit"
            ),
            base_branch=base_branch,
            head_branch=branch,
            commit_sha=commit_sha,
        )

    def fake_wait_for_repair_ci_status(
        repository,
        received_commit_sha,
    ):
        assert events == [
            "run",
            "summary",
            "validation",
            "commit",
            "push",
            "pull_request",
        ]
        assert repository == "example/project"
        assert received_commit_sha == commit_sha

        events.append("ci_status")

        return RepairCIStatus(
            repository=repository,
            commit_sha=received_commit_sha,
            state="success",
            check_runs=(
                RepairCheckRun(
                    name="tests",
                    status="completed",
                    conclusion="success",
                    details_url=(
                        "https://github.com/"
                        "example/project/actions/runs/1"
                    ),
                ),
            ),
        )

    monkeypatch.setattr(
        cli,
        "_run_once",
        fake_run_once,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        fake_collect_post_repair_summary,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "run_post_repair_validation",
        fake_run_post_repair_validation,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "create_repair_commit",
        fake_create_repair_commit,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "push_repair_branch",
        fake_push_repair_branch,
        raising=False,
    )
    monkeypatch.setattr(
    cli,
    "create_repair_pull_request",
    fake_create_repair_pull_request,
    raising=False,
    )

    monkeypatch.setattr(
    cli,
    "wait_for_repair_ci_status",
    fake_wait_for_repair_ci_status,
    raising=False,
    )

    monkeypatch.setattr(
        cli,
        "get_current_branch",
        lambda: "devpilot-v1",
        raising=False,
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
        "run",
        "summary",
        "validation",
        "commit",
        "push",
        "pull_request",
        "ci_status",
    ]
    output = "\n".join(printed)

    assert "Repair pull request created" in output
    assert "Pull request:" in output
    assert "#42" in output
    assert "URL:" in output
    assert (
        "https://github.com/example/project/pull/42"
        in output
    )
    assert "Base:" in output
    assert "devpilot-v1" in output
    assert "Head:" in output

    assert "Repair branch pushed" in output
    assert "Remote:" in output
    assert "origin" in output
    assert "Branch:" in output
    assert branch in output
    assert "Commit:" in output
    assert commit_sha in output
    assert "Repair CI status" in output
    assert "State:" in output
    assert "success" in output
    assert "Check runs:" in output
    assert "tests" in output
    assert "completed" in output

    def fake_fetch_repair_ci_status(
    repository,
    received_commit_sha,
    ):
        assert events == [
            "run",
            "summary",
            "validation",
            "commit",
            "push",
            "pull_request",
        ]
        assert repository == "example/project"
        assert received_commit_sha == commit_sha

        events.append("ci_status")

        return RepairCIStatus(
            repository=repository,
            commit_sha=received_commit_sha,
            state="success",
            check_runs=(
                RepairCheckRun(
                    name="tests",
                    status="completed",
                    conclusion="success",
                    details_url=(
                        "https://github.com/"
                        "example/project/actions/runs/1"
                    ),
                ),
            ),
        )



def test_main_reports_repair_push_error(
    monkeypatch,
):
    printed = []

    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "push error prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        lambda branch: PostRepairSummary(
            branch=branch,
            changes=(
                " M corecoder/cli.py",
            ),
        ),
        raising=False,
    )

    def fake_push_repair_branch(
        branch,
        commit_sha,
    ):
        raise RepairPushError(
            "unable to push repair branch: "
            "connection was reset"
        )

    monkeypatch.setattr(
        cli,
        "push_repair_branch",
        fake_push_repair_branch,
        raising=False,
    )



    monkeypatch.setattr(
    cli,
    "get_current_branch",
    lambda: "devpilot-v1",
    raising=False,
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
    assert "Repair push error:" in output
    assert "unable to push repair branch" in output
    assert "connection was reset" in output

def test_main_reports_repair_pull_request_error(
    monkeypatch,
):
    printed = []

    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "pull request error prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        lambda branch: PostRepairSummary(
            branch=branch,
            changes=(
                " M corecoder/cli.py",
            ),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "get_current_branch",
        lambda: "devpilot-v1",
        raising=False,
    )

    def fake_create_repair_pull_request(
        repository,
        issue_number,
        issue_title,
        repair_push,
        base_branch,
    ):
        assert repository == "example/project"
        assert issue_number == 21
        assert issue_title == "Fix repository scan limit"
        assert repair_push.remote == "origin"
        assert repair_push.branch == (
            "devpilot/"
            "issue-21-fix-repository-scan-limit"
        )
        assert base_branch == "devpilot-v1"

        raise RepairPullRequestError(
            "GitHub API request failed with "
            "HTTP status 422"
        )

    monkeypatch.setattr(
        cli,
        "create_repair_pull_request",
        fake_create_repair_pull_request,
        raising=False,
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
    assert "Repair branch pushed" in output
    assert "Repair pull request error:" in output
    assert "GitHub API request failed" in output
    assert "HTTP status 422" in output
    assert "Repair pull request created" not in output

def test_main_reports_repair_ci_status_error(
    monkeypatch,
):
    printed = []

    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "CI status error prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        lambda branch: PostRepairSummary(
            branch=branch,
            changes=(
                " M corecoder/cli.py",
            ),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "get_current_branch",
        lambda: "devpilot-v1",
        raising=False,
    )

    def fake_wait_for_repair_ci_status(
        repository,
        commit_sha,
    ):
        assert repository == "example/project"
        assert commit_sha == (
            "0123456789abcdef"
            "0123456789abcdef"
            "01234567"
        )

        raise RepairCIStatusError(
            "GitHub API request timed out after "
            "20 seconds"
        )

    monkeypatch.setattr(
    cli,
    "wait_for_repair_ci_status",
    fake_wait_for_repair_ci_status,
    raising=False,
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

    assert "Repair pull request created" in output
    assert "Repair CI status error:" in output
    assert "GitHub API request timed out" in output
    assert "20 seconds" in output
    assert "Repair CI status\n" not in output

def test_main_skips_repair_commit_without_changes(
    monkeypatch,
):
    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "no-change commit prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        lambda branch: PostRepairSummary(
            branch=branch,
            changes=(),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "create_repair_commit",
        lambda *args, **kwargs: pytest.fail(
            "Commit must not run without changes"
        ),
        raising=False,
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


def test_main_skips_repair_commit_in_dry_run(
    monkeypatch,
):
    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "dry-run commit prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
    )
    monkeypatch.setattr(
        cli,
        "create_repair_commit",
        lambda *args, **kwargs: pytest.fail(
            "Dry-run must not create a commit"
        ),
        raising=False,
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


def test_main_skips_repair_commit_when_validation_fails(
    monkeypatch,
):
    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "failed validation commit prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        lambda branch: PostRepairSummary(
            branch=branch,
            changes=(
                " M corecoder/cli.py",
            ),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "run_post_repair_validation",
        lambda: PostRepairValidation(
            command=(
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
            ),
            status="failed",
            exit_code=1,
            passed_count=10,
            failed_count=2,
            error_count=0,
            output="2 failed, 10 passed",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "create_repair_commit",
        lambda *args, **kwargs: pytest.fail(
            "Failed validation must not create a commit"
        ),
        raising=False,
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


def test_main_reports_repair_commit_error(
    monkeypatch,
):
    printed = []

    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "commit error prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        lambda branch: PostRepairSummary(
            branch=branch,
            changes=(
                " M corecoder/cli.py",
            ),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "run_post_repair_validation",
        lambda: PostRepairValidation(
            command=(
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
            ),
            status="passed",
            exit_code=0,
            passed_count=275,
            failed_count=0,
            error_count=0,
            output="275 passed",
        ),
        raising=False,
    )

    def fake_create_repair_commit(
        issue_number,
        issue_title,
    ):
        raise RepairCommitError(
            "unable to commit repair changes: "
            "pre-commit hook failed"
        )

    monkeypatch.setattr(
        cli,
        "create_repair_commit",
        fake_create_repair_commit,
        raising=False,
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
    assert "Repair commit error:" in output
    assert "unable to commit repair changes" in output
    assert "pre-commit hook failed" in output

def _patch_runtime(
    monkeypatch,
    captured: dict | None = None,
) -> None:
    """Replace configuration, LLM, and Agent construction."""
    def fake_create_repair_pull_request(
        repository,
        issue_number,
        issue_title,
        repair_push,
        base_branch,
    ):
            return RepairPullRequest(
                repository=repository,
                number=42,
                url=(
                    f"https://github.com/"
                    f"{repository}/pull/42"
                ),
                title=(
                    f"Fix #{issue_number}: "
                    f"{issue_title}"
                ),
                base_branch=base_branch,
                head_branch=repair_push.branch,
                commit_sha=repair_push.commit_sha,
            )

    def fake_fetch_repair_ci_status(
    repository,
    commit_sha,
    ):
        return RepairCIStatus(
            repository=repository,
            commit_sha=commit_sha,
            state="no_checks",
            check_runs=(),
        )

    def fake_wait_for_repair_ci_status(
            repository,
            commit_sha,
        ):
            return RepairCIStatus(
                repository=repository,
                commit_sha=commit_sha,
                state="no_checks",
                check_runs=(),
            )
    monkeypatch.setattr(
    cli,
    "_request_commit_approval",
    lambda summary, validation: True,
    )
    monkeypatch.setattr(
        cli,
        "check_worktree",
        lambda: _worktree_preflight(),
    )

    monkeypatch.setattr(
        cli,
        "get_current_branch",
        lambda: "devpilot-v1",
        raising=False,
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
        cli,
        "collect_post_repair_summary",
        lambda branch: PostRepairSummary(
            branch=branch,
            changes=(),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "run_post_repair_validation",
        lambda: PostRepairValidation(
            command=(
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
            ),
            status="passed",
            exit_code=0,
            passed_count=264,
            failed_count=0,
            error_count=0,
            output="264 passed",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "create_repair_commit",
        lambda issue_number, issue_title: RepairCommit(
            sha=(
                "0123456789abcdef"
                "0123456789abcdef"
                "01234567"
            ),
            message=(
                f"Fix #{issue_number}: "
                f"{issue_title}"
            ),
        ),
        raising=False,
    )
    monkeypatch.setattr(
    cli,
    "push_repair_branch",
    lambda branch, commit_sha: RepairPush(
        remote="origin",
        branch=branch,
        commit_sha=commit_sha,
    ),
    raising=False,
    )
    monkeypatch.setattr(
    cli,
    "create_repair_pull_request",
    fake_create_repair_pull_request,
    raising=False,
    )


    monkeypatch.setattr(
    cli,
    "fetch_repair_ci_status",
    fake_fetch_repair_ci_status,
    raising=False,
    )

    monkeypatch.setattr(
    cli,
    "wait_for_repair_ci_status",
    fake_wait_for_repair_ci_status,
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

def test_parse_args_rejects_resume_workflow_without_issue(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--resume-workflow",
        ],
    )

    with pytest.raises(SystemExit) as error:
        cli._parse_args()

    assert error.value.code == 2
    assert (
        "--resume-workflow requires --issue"
        in capsys.readouterr().err
    )

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

def test_main_enables_tool_permission_policy_by_default(
    monkeypatch,
):
    from corecoder.permissions import ToolPermissionPolicy

    captured = {}
    _patch_runtime(monkeypatch, captured)

    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
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

    assert isinstance(
        captured["agent_kwargs"]["permission_policy"],
        ToolPermissionPolicy,
    )

def test_main_wires_tool_approval_callback(
    monkeypatch,
):
    captured = {}
    _patch_runtime(monkeypatch, captured)

    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
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

    assert (
        captured["agent_kwargs"]["request_tool_approval"]
        is cli._request_tool_approval
    )


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

def test_main_skips_repair_push_without_changes(
    monkeypatch,
):
    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "no-change push prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        lambda branch: PostRepairSummary(
            branch=branch,
            changes=(),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "push_repair_branch",
        lambda *args, **kwargs: pytest.fail(
            "Push must not run without repository changes"
        ),
        raising=False,
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


def test_main_skips_repair_push_in_dry_run(
    monkeypatch,
):
    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "dry-run push prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
    )
    monkeypatch.setattr(
        cli,
        "push_repair_branch",
        lambda *args, **kwargs: pytest.fail(
            "Dry-run must not push a repair branch"
        ),
        raising=False,
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


def test_main_skips_repair_push_when_validation_fails(
    monkeypatch,
):
    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "failed validation push prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        lambda branch: PostRepairSummary(
            branch=branch,
            changes=(
                " M corecoder/cli.py",
            ),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "run_post_repair_validation",
        lambda: PostRepairValidation(
            command=(
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
            ),
            status="failed",
            exit_code=1,
            passed_count=10,
            failed_count=1,
            error_count=0,
            output="1 failed, 10 passed",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "push_repair_branch",
        lambda *args, **kwargs: pytest.fail(
            "Failed validation must not push a repair branch"
        ),
        raising=False,
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


def test_main_skips_repair_push_when_commit_fails(
    monkeypatch,
):
    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "failed commit push prompt",
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: None,
    )
    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        lambda branch: PostRepairSummary(
            branch=branch,
            changes=(
                " M corecoder/cli.py",
            ),
        ),
        raising=False,
    )

    def fake_create_repair_commit(
        issue_number,
        issue_title,
    ):
        raise RepairCommitError(
            "unable to commit repair changes"
        )

    monkeypatch.setattr(
        cli,
        "create_repair_commit",
        fake_create_repair_commit,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "push_repair_branch",
        lambda *args, **kwargs: pytest.fail(
            "Failed commit must not push a repair branch"
        ),
        raising=False,
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


@pytest.mark.parametrize(
    (
        "retry_state",
        "retry_has_changes",
        "expected_exit_code",
    ),
    [
        ("success", True, None),
        ("failure", True, 1),
        ("success", False, 1),
    ],
)
def test_main_handles_repair_ci_retry_result(
    monkeypatch,
    retry_state,
    retry_has_changes,
    expected_exit_code,
):
    printed = []
    prompts = []
    summary_branches = []
    validation_calls = []
    commit_calls = []
    push_calls = []
    ci_wait_calls = []
    ci_log_calls = []

    commit_sha = (
        "0123456789abcdef"
        "0123456789abcdef"
        "01234567"
    )

    _patch_runtime(monkeypatch)
    default_push_repair_branch = cli.push_repair_branch

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "Initial issue repair prompt",
    )

    def fake_build_repair_ci_failure_prompt(
    ci_status,
    *,
    check_logs=None,
    ):
        assert ci_status.state == "failure"
        assert ci_status.repository == "example/project"
        assert ci_status.commit_sha == commit_sha
        assert check_logs == {
            "tests": (
                "FAILED tests/test_example.py::test_failure\n"
                "AssertionError: expected 1, got 2"
            ),
        }
        return "Remote CI failure repair prompt"

    monkeypatch.setattr(
        cli,
        "build_repair_ci_failure_prompt",
        fake_build_repair_ci_failure_prompt,
        raising=False,
    )

    def fake_fetch_repair_check_log(
    repository,
    check_run,
    ):
        ci_log_calls.append(
            (repository, check_run.name)
        )
        assert repository == "example/project"
        assert check_run.conclusion == "failure"

        return (
            "FAILED tests/test_example.py::test_failure\n"
            "AssertionError: expected 1, got 2"
        )

    monkeypatch.setattr(
        cli,
        "fetch_repair_check_log",
        fake_fetch_repair_check_log,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: prompts.append(prompt),
    )
    def fake_collect_post_repair_summary(branch):
            summary_branches.append(branch)

            if (
                len(summary_branches) == 2
                and not retry_has_changes
            ):
                return PostRepairSummary(
                    branch=branch,
                    changes=(),
                )

            return PostRepairSummary(
                branch=branch,
                changes=(
                    " M corecoder/cli.py",
                ),
            )

    monkeypatch.setattr(
        cli,
        "collect_post_repair_summary",
        fake_collect_post_repair_summary,
        raising=False,
    )

    monkeypatch.setattr(
        cli,
        "get_current_branch",
        lambda: "devpilot-v1",
        raising=False,
    )

    def fake_run_post_repair_validation():
        validation_calls.append("validation")
        return PostRepairValidation(
            command=(
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
            ),
            status="passed",
            exit_code=0,
            passed_count=342,
            failed_count=0,
            error_count=0,
            output="342 passed",
        )

    monkeypatch.setattr(
        cli,
        "run_post_repair_validation",
        fake_run_post_repair_validation,
    )

    def fake_create_repair_commit(
    issue_number,
    issue_title,
    ):
        commit_calls.append(
            (issue_number, issue_title)
        )
        return RepairCommit(
            sha=commit_sha,
            message=(
                f"Fix #{issue_number}: "
                f"{issue_title}"
            ),
        )

    monkeypatch.setattr(
        cli,
        "create_repair_commit",
        fake_create_repair_commit,
        raising=False,
    )

    def fake_push_repair_branch(
    branch,
    received_commit_sha,
    ):
        push_calls.append(
            (branch, received_commit_sha)
        )
        return default_push_repair_branch(
            branch,
            received_commit_sha,
        )

    monkeypatch.setattr(
        cli,
        "push_repair_branch",
        fake_push_repair_branch,
        raising=False,
    )

    def fake_wait_for_repair_ci_status(
    repository,
    received_commit_sha,
    ):
        ci_wait_calls.append(
            (repository, received_commit_sha)
        )

        assert repository == "example/project"
        assert received_commit_sha == commit_sha

        if len(ci_wait_calls) == 1:
            return RepairCIStatus(
                repository=repository,
                commit_sha=received_commit_sha,
                state="failure",
                check_runs=(
                    RepairCheckRun(
                        name="tests",
                        status="completed",
                        conclusion="failure",
                        details_url=(
                            "https://github.com/"
                            "example/project/actions/runs/1"
                        ),
                    ),
                ),
            )

        return RepairCIStatus(
            repository=repository,
            commit_sha=received_commit_sha,
            state=retry_state,
            check_runs=(
                RepairCheckRun(
                    name="tests",
                    status="completed",
                    conclusion=retry_state,
                    details_url=(
                        "https://github.com/"
                        "example/project/actions/runs/2"
                    ),
                ),
            ),
        )

    monkeypatch.setattr(
        cli,
        "wait_for_repair_ci_status",
        fake_wait_for_repair_ci_status,
        raising=False,
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

    if expected_exit_code is None:
        cli.main()
    else:
        with pytest.raises(SystemExit) as error:
            cli.main()

        assert error.value.code == expected_exit_code

    output = "\n".join(printed)

    assert "Repair pull request created" in output
    assert "Repair CI status" in output
    assert "State:" in output
    assert "failure" in output
    assert "tests" in output
    assert "completed" in output
    assert (
        "https://github.com/"
        "example/project/actions/runs/1"
        in output
    )
    assert "Repair CI status error:" not in output

    assert prompts == [
        "Initial issue repair prompt",
        "Remote CI failure repair prompt",
    ]

    assert summary_branches == [
    "devpilot/issue-21-fix-repository-scan-limit",
    "devpilot/issue-21-fix-repository-scan-limit",
    ]

    expected_retry_count = (
    2 if retry_has_changes else 1
    )

    assert len(validation_calls) == expected_retry_count
    assert len(commit_calls) == expected_retry_count
    assert len(push_calls) == expected_retry_count
    assert len(ci_wait_calls) == expected_retry_count

    if retry_has_changes:
        assert (
            "https://github.com/"
            "example/project/actions/runs/2"
            in output
        )
    assert ci_log_calls == [
    ("example/project", "tests"),
    ]
def test_main_warns_when_repair_ci_log_fetch_fails(
    monkeypatch,
):
    printed = []
    captured = {}

    _patch_runtime(monkeypatch)

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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "issue repair prompt",
    )

    def fake_fetch_repair_check_log(
        repository,
        check_run,
    ):
        raise RepairCIStatusError(
            "unable to fetch CI log"
        )

    monkeypatch.setattr(
        cli,
        "fetch_repair_check_log",
        fake_fetch_repair_check_log,
        raising=False,
    )

    def fake_build_repair_ci_failure_prompt(
        ci_status,
        *,
        check_logs=None,
    ):
        captured["check_logs"] = check_logs
        return "CI failure repair prompt"

    monkeypatch.setattr(
        cli,
        "build_repair_ci_failure_prompt",
        fake_build_repair_ci_failure_prompt,
        raising=False,
    )

    def fake_run_issue_workflow(**kwargs):
        ci_status = RepairCIStatus(
            repository="example/project",
            commit_sha="0123456789abcdef",
            state="failure",
            check_runs=(
                RepairCheckRun(
                    name="tests",
                    status="completed",
                    conclusion="failure",
                    details_url=None,
                ),
            ),
        )

        prompt = kwargs[
            "build_ci_failure_prompt"
        ](
            ci_status
        )

        assert prompt == "CI failure repair prompt"

        class Result:
            summary = PostRepairSummary(
                branch=(
                    "devpilot/"
                    "issue-21-fix-repository-scan-limit"
                ),
                changes=(),
            )
            validation = None
            commit = None

        return Result()

    monkeypatch.setattr(
        cli,
        "run_issue_workflow",
        fake_run_issue_workflow,
        raising=False,
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

    assert captured["check_logs"] == {}

    output = "\n".join(printed)
    assert "Repair CI log warning:" in output
    assert "unable to fetch CI log" in output

def test_main_delegates_issue_dry_run_to_orchestrator(
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
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "dry-run prompt",
    )

    def fake_run_issue_workflow(**kwargs):
        captured["workflow_kwargs"] = kwargs
        return None

    monkeypatch.setattr(
        cli,
        "run_issue_workflow",
        fake_run_issue_workflow,
        raising=False,
    )

    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda *args, **kwargs: pytest.fail(
            "CLI must delegate dry-run execution "
            "to run_issue_workflow"
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

    workflow_kwargs = captured["workflow_kwargs"]

    assert workflow_kwargs["issue_prompt"] == (
        "dry-run prompt"
    )
    assert workflow_kwargs["dry_run"] is True
    assert callable(workflow_kwargs["run_agent"])


def test_main_delegates_issue_repair_to_orchestrator(
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
        lambda: _worktree_preflight(),
    )
    monkeypatch.setattr(
        cli,
        "get_current_branch",
        lambda: "devpilot-v1",
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "create_repair_branch",
        lambda received_task: (
            "devpilot/issue-21-fix-repository-scan-limit"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "repair prompt",
    )

    def fake_run_issue_workflow(**kwargs):
        captured["workflow_kwargs"] = kwargs
        return None

    monkeypatch.setattr(
        cli,
        "run_issue_workflow",
        fake_run_issue_workflow,
        raising=False,
    )

    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda *args, **kwargs: pytest.fail(
            "CLI must delegate repair execution "
            "to run_issue_workflow"
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

    workflow_kwargs = captured["workflow_kwargs"]

    assert workflow_kwargs["workflow_id"] == "example/project#21"
    assert callable(workflow_kwargs["save_checkpoint"])
    assert (
    workflow_kwargs["request_commit_approval"]
    is cli._request_commit_approval
    )

def test_main_resumes_issue_workflow_without_recreating_repair_branch(
    monkeypatch,
):
    captured = {}

    _patch_runtime(monkeypatch, captured)

    checkpoint = WorkflowCheckpoint(
        workflow_id="example/project#21",
        state=WorkflowState.VALIDATE,
        repair_branch=(
            "devpilot/issue-21-fix-repository-scan-limit"
        ),
        repair_base_branch="devpilot-v1",
        summary=PostRepairSummary(
            branch=(
                "devpilot/issue-21-fix-repository-scan-limit"
            ),
            changes=(" M corecoder/example.py",),
        ),
    )

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
        lambda: _worktree_preflight(
            status="dirty",
            changes=(" M corecoder/example.py",),
        ),
    )

    monkeypatch.setattr(
        cli,
        "load_workflow_checkpoint",
        lambda path: checkpoint,
        raising=False,
    )

    monkeypatch.setattr(
    cli,
    "get_current_branch",
    lambda: checkpoint.repair_branch,
    )

    monkeypatch.setattr(
        cli,
        "create_repair_branch",
        lambda task: pytest.fail(
            "Resume must not create a new repair branch"
        ),
        raising=False,
    )

    monkeypatch.setattr(
        cli,
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "repair prompt",
    )

    def fake_run_issue_workflow(**kwargs):
        captured["workflow_kwargs"] = kwargs
        return None

    monkeypatch.setattr(
        cli,
        "run_issue_workflow",
        fake_run_issue_workflow,
        raising=False,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--issue",
            _ISSUE_URL,
            "--resume-workflow",
        ],
    )

    cli.main()

    workflow_kwargs = captured["workflow_kwargs"]

    assert workflow_kwargs["repair_branch"] == (
        checkpoint.repair_branch
    )
    assert workflow_kwargs["repair_base_branch"] == (
    checkpoint.repair_base_branch
    )
    assert workflow_kwargs["checkpoint"] == checkpoint

def test_main_rejects_resume_checkpoint_for_different_workflow(
    monkeypatch,
):
    captured = {}

    _patch_runtime(monkeypatch, captured)

    checkpoint = WorkflowCheckpoint(
        workflow_id="example/project#999",
        state=WorkflowState.VALIDATE,
        repair_branch=(
            "devpilot/issue-999-fix-repository-scan-limit"
        ),
        repair_base_branch="devpilot-v1",
    )

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
        lambda: _worktree_preflight(status="dirty"),
    )
    monkeypatch.setattr(
        cli,
        "load_workflow_checkpoint",
        lambda path: checkpoint,
    )
    monkeypatch.setattr(
        cli,
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "repair prompt",
    )
    monkeypatch.setattr(
        cli,
        "run_issue_workflow",
        lambda **kwargs: None,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--issue",
            _ISSUE_URL,
            "--resume-workflow",
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

def test_main_rejects_resume_checkpoint_without_base_branch(
    monkeypatch,
):
    captured = {}

    _patch_runtime(monkeypatch, captured)

    checkpoint = WorkflowCheckpoint(
        workflow_id="example/project#21",
        state=WorkflowState.VALIDATE,
        repair_branch=(
            "devpilot/issue-21-fix-repository-scan-limit"
        ),
    )

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
        lambda: _worktree_preflight(status="dirty"),
    )
    monkeypatch.setattr(
        cli,
        "load_workflow_checkpoint",
        lambda path: checkpoint,
    )
    monkeypatch.setattr(
        cli,
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "repair prompt",
    )
    monkeypatch.setattr(
        cli,
        "run_issue_workflow",
        lambda **kwargs: None,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--issue",
            _ISSUE_URL,
            "--resume-workflow",
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

def test_main_rejects_resume_checkpoint_on_wrong_current_branch(
    monkeypatch,
):
    captured = {}

    _patch_runtime(monkeypatch, captured)

    checkpoint = WorkflowCheckpoint(
        workflow_id="example/project#21",
        state=WorkflowState.VALIDATE,
        repair_branch=(
            "devpilot/issue-21-fix-repository-scan-limit"
        ),
        repair_base_branch="devpilot-v1",
    )

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
        lambda: _worktree_preflight(status="dirty"),
    )
    monkeypatch.setattr(
        cli,
        "load_workflow_checkpoint",
        lambda path: checkpoint,
    )
    monkeypatch.setattr(
        cli,
        "get_current_branch",
        lambda: "devpilot-v1",
    )
    monkeypatch.setattr(
        cli,
        "build_issue_repair_prompt",
        lambda *args, **kwargs: "repair prompt",
    )
    monkeypatch.setattr(
        cli,
        "run_issue_workflow",
        lambda **kwargs: None,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--issue",
            _ISSUE_URL,
            "--resume-workflow",
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

def test_main_reports_missing_resume_checkpoint(
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
        lambda: _worktree_preflight(status="clean"),
    )
    monkeypatch.setattr(
        cli,
        "load_workflow_checkpoint",
        lambda path: (_ for _ in ()).throw(
            FileNotFoundError("checkpoint missing")
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--issue",
            _ISSUE_URL,
            "--resume-workflow",
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

def test_main_reports_invalid_resume_checkpoint(
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
        lambda: _worktree_preflight(status="clean"),
    )
    monkeypatch.setattr(
        cli,
        "load_workflow_checkpoint",
        lambda path: (_ for _ in ()).throw(
            JSONDecodeError(
                "invalid checkpoint",
                "{",
                0,
            )
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corecoder",
            "--issue",
            _ISSUE_URL,
            "--resume-workflow",
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

def test_request_commit_approval_accepts_approve(monkeypatch):
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=101,
        failed_count=0,
        error_count=0,
        output="101 passed",
    )

    monkeypatch.setattr(
        cli,
        "pt_prompt",
        lambda prompt: "approve",
    )

    assert cli._request_commit_approval(
        summary,
        validation,
    ) is True

def test_request_commit_approval_accepts_reject(monkeypatch):
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=101,
        failed_count=0,
        error_count=0,
        output="101 passed",
    )

    monkeypatch.setattr(
        cli,
        "pt_prompt",
        lambda prompt: "reject",
    )

    assert cli._request_commit_approval(
        summary,
        validation,
    ) is False

def test_request_tool_approval_accepts_approve(monkeypatch):
    from corecoder.permissions import (
        ToolApprovalRequest,
        ToolPermission,
    )

    request = ToolApprovalRequest(
        tool_name="write_file",
        permission=ToolPermission.WRITE,
        arguments={
            "path": "example.py",
            "content": "hello",
        },
    )

    monkeypatch.setattr(
        cli,
        "pt_prompt",
        lambda prompt: "approve",
    )

    assert cli._request_tool_approval(request) is True

def test_request_tool_approval_accepts_reject(monkeypatch):
    from corecoder.permissions import (
        ToolApprovalRequest,
        ToolPermission,
    )

    request = ToolApprovalRequest(
        tool_name="write_file",
        permission=ToolPermission.WRITE,
        arguments={
            "path": "example.py",
            "content": "hello",
        },
    )

    monkeypatch.setattr(
        cli,
        "pt_prompt",
        lambda prompt: "reject",
    )

    assert cli._request_tool_approval(request) is False
