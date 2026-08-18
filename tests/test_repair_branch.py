"""Tests for dedicated Git branches used by Issue repairs."""

import pytest
import subprocess
import corecoder.repair_branch as repair_branch
from corecoder.issue_task import IssueTask
from corecoder.repair_branch import (
    RepairBranchError,
    build_repair_branch_name,
    create_repair_branch,
    get_current_branch,
)


def _task(
    *,
    title: str = "Fix repository scan limit",
    issue_number: int | None = 21,
) -> IssueTask:
    """Return a representative Issue task."""
    return IssueTask(
        title=title,
        issue_number=issue_number,
    )

def test_get_current_branch_reads_checked_out_branch(
    monkeypatch,
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout="devpilot-v1\n",
            stderr="",
        )

    monkeypatch.setattr(
        repair_branch.subprocess,
        "run",
        fake_run,
    )

    result = get_current_branch()

    assert result == "devpilot-v1"
    assert calls == [
        (
            [
                "git",
                "branch",
                "--show-current",
            ],
            {
                "cwd": None,
                "capture_output": True,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "timeout": 10,
                "check": False,
            },
        )
    ]

def test_get_current_branch_reports_git_failure(
    monkeypatch,
):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            returncode=128,
            stdout="",
            stderr="fatal: not a git repository",
        )

    monkeypatch.setattr(
        repair_branch.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        RepairBranchError,
        match="unable to read current Git branch",
    ):
        get_current_branch()

def test_get_current_branch_rejects_detached_head(
    monkeypatch,
):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout="\n",
            stderr="",
        )

    monkeypatch.setattr(
        repair_branch.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        RepairBranchError,
        match="current Git checkout is not on a branch",
    ):
        get_current_branch()

def test_build_repair_branch_name_uses_number_and_title():
    result = build_repair_branch_name(_task())

    assert (
        result
        == "devpilot/issue-21-fix-repository-scan-limit"
    )


def test_build_repair_branch_name_normalizes_punctuation():
    result = build_repair_branch_name(
        _task(
            title=" Fix: CLI preview / quoting!!! ",
        )
    )

    assert result == "devpilot/issue-21-fix-cli-preview-quoting"


def test_build_repair_branch_name_limits_total_length():
    result = build_repair_branch_name(
        _task(
            title="Very long repair title " * 20,
        )
    )

    assert result.startswith("devpilot/issue-21-")
    assert len(result) <= 80
    assert not result.endswith("-")


def test_build_repair_branch_name_handles_non_ascii_title():
    result = build_repair_branch_name(
        _task(
            title="修复中文标题",
        )
    )

    assert result == "devpilot/issue-21"


def test_build_repair_branch_name_requires_issue_number():
    with pytest.raises(
        RepairBranchError,
        match="Issue number is required",
    ):
        build_repair_branch_name(
            _task(issue_number=None)
        )

def test_create_repair_branch_switches_to_new_branch(
    monkeypatch,
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(
        repair_branch.subprocess,
        "run",
        fake_run,
        raising=False,
    )

    result = create_repair_branch(_task())

    assert (
        result
        == "devpilot/issue-21-fix-repository-scan-limit"
    )
    assert calls == [
        (
            [
                "git",
                "switch",
                "-c",
                "devpilot/issue-21-fix-repository-scan-limit",
            ],
            {
                "cwd": None,
                "capture_output": True,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "timeout": 10,
                "check": False,
            },
        )
    ]


def test_create_repair_branch_uses_requested_directory(
    monkeypatch,
    tmp_path,
):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs

        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(
        repair_branch.subprocess,
        "run",
        fake_run,
        raising=False,
    )

    create_repair_branch(
        _task(),
        cwd=tmp_path,
    )

    assert captured["kwargs"]["cwd"] == tmp_path


def test_create_repair_branch_reports_git_failure(
    monkeypatch,
):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            returncode=128,
            stdout="",
            stderr="fatal: a branch with that name already exists",
        )

    monkeypatch.setattr(
        repair_branch.subprocess,
        "run",
        fake_run,
        raising=False,
    )

    with pytest.raises(
        RepairBranchError,
        match="unable to create repair branch",
    ):
        create_repair_branch(_task())


@pytest.mark.parametrize(
    "execution_error",
    [
        FileNotFoundError("git executable not found"),
        subprocess.TimeoutExpired(
            cmd=["git", "switch"],
            timeout=10,
        ),
    ],
)
def test_create_repair_branch_reports_execution_error(
    monkeypatch,
    execution_error,
):
    def fake_run(command, **kwargs):
        raise execution_error

    monkeypatch.setattr(
        repair_branch.subprocess,
        "run",
        fake_run,
        raising=False,
    )

    with pytest.raises(
        RepairBranchError,
        match="unable to create repair branch",
    ):
        create_repair_branch(_task())