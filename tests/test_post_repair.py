"""Tests for post-repair Git result inspection."""

import subprocess

import pytest

import corecoder.post_repair as post_repair
from corecoder.post_repair import (
    PostRepairSummary,
    PostRepairSummaryError,
    collect_post_repair_summary,
)

_BRANCH = "devpilot/issue-21-fix-repository-scan-limit"


def test_collect_post_repair_summary_reports_clean_result(
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
        post_repair.subprocess,
        "run",
        fake_run,
    )

    result = collect_post_repair_summary(_BRANCH)

    assert result == PostRepairSummary(
        branch=_BRANCH,
        changes=(),
    )
    assert result.has_changes is False
    assert calls == [
        (
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
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


def test_collect_post_repair_summary_preserves_changes(
    monkeypatch,
):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=(
                " M corecoder/cli.py\n"
                "?? tests/test_example.py\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(
        post_repair.subprocess,
        "run",
        fake_run,
    )

    result = collect_post_repair_summary(_BRANCH)

    assert result.branch == _BRANCH
    assert result.changes == (
        " M corecoder/cli.py",
        "?? tests/test_example.py",
    )
    assert result.has_changes is True


def test_collect_post_repair_summary_uses_requested_directory(
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
        post_repair.subprocess,
        "run",
        fake_run,
    )

    collect_post_repair_summary(
        _BRANCH,
        cwd=tmp_path,
    )

    assert captured["kwargs"]["cwd"] == tmp_path


def test_collect_post_repair_summary_reports_git_failure(
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
        post_repair.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        PostRepairSummaryError,
        match="unable to inspect post-repair Git changes",
    ):
        collect_post_repair_summary(_BRANCH)


@pytest.mark.parametrize(
    "execution_error",
    [
        FileNotFoundError("git executable not found"),
        subprocess.TimeoutExpired(
            cmd=["git", "status"],
            timeout=10,
        ),
    ],
)
def test_collect_post_repair_summary_reports_execution_error(
    monkeypatch,
    execution_error,
):
    def fake_run(command, **kwargs):
        raise execution_error

    monkeypatch.setattr(
        post_repair.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        PostRepairSummaryError,
        match="unable to inspect post-repair Git changes",
    ):
        collect_post_repair_summary(_BRANCH)