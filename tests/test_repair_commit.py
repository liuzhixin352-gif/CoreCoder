"""Tests for automatic validated repair commits."""

import subprocess

import pytest

import corecoder.repair_commit as repair_commit
from corecoder.repair_commit import (
    RepairCommit,
    RepairCommitError,
    build_repair_commit_message,
    create_repair_commit,
)


def _completed_process(
    command,
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
):
    return subprocess.CompletedProcess(
        command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_build_repair_commit_message_normalizes_title():
    message = build_repair_commit_message(
        18,
        "  Handle   empty\nconfiguration files  ",
    )

    assert message == (
        "Fix #18: Handle empty configuration files"
    )


def test_build_repair_commit_message_uses_fallback_title():
    message = build_repair_commit_message(
        18,
        " \n\t ",
    )

    assert message == "Fix #18: GitHub Issue repair"


def test_create_repair_commit_runs_expected_git_commands(
    monkeypatch,
):
    calls = []
    commit_sha = (
        "0123456789abcdef"
        "0123456789abcdef"
        "01234567"
    )

    responses = iter(
        [
            {
                "returncode": 0,
                "stdout": "",
                "stderr": "",
            },
            {
                "returncode": 0,
                "stdout": (
                    "[devpilot/issue-18 0123456] "
                    "Fix #18: Handle empty files\n"
                ),
                "stderr": "",
            },
            {
                "returncode": 0,
                "stdout": f"{commit_sha}\n",
                "stderr": "",
            },
        ]
    )

    def fake_run(command, **kwargs):
        calls.append(
            (
                command,
                kwargs,
            )
        )
        response = next(responses)

        return _completed_process(
            command,
            **response,
        )

    monkeypatch.setattr(
        repair_commit.subprocess,
        "run",
        fake_run,
    )

    result = create_repair_commit(
        18,
        "Handle empty files",
    )

    message = "Fix #18: Handle empty files"

    assert result == RepairCommit(
        sha=commit_sha,
        message=message,
    )

    assert [
        command
        for command, _kwargs in calls
    ] == [
        [
            "git",
            "add",
            "--all",
        ],
        [
            "git",
            "commit",
            "--message",
            message,
        ],
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
    ]

    expected_kwargs = {
        "cwd": None,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 30,
        "check": False,
    }

    assert all(
        kwargs == expected_kwargs
        for _command, kwargs in calls
    )


def test_create_repair_commit_reports_staging_failure(
    monkeypatch,
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)

        return _completed_process(
            command,
            returncode=128,
            stderr="fatal: not a git repository",
        )

    monkeypatch.setattr(
        repair_commit.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        RepairCommitError,
        match="unable to stage repair changes",
    ):
        create_repair_commit(
            18,
            "Handle empty files",
        )

    assert calls == [
        [
            "git",
            "add",
            "--all",
        ],
    ]


def test_create_repair_commit_reports_commit_failure(
    monkeypatch,
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)

        if command[:2] == [
            "git",
            "add",
        ]:
            return _completed_process(command)

        return _completed_process(
            command,
            returncode=1,
            stderr="pre-commit hook failed",
        )

    monkeypatch.setattr(
        repair_commit.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        RepairCommitError,
        match="unable to commit repair changes",
    ):
        create_repair_commit(
            18,
            "Handle empty files",
        )

    assert len(calls) == 2


def test_create_repair_commit_reports_sha_failure(
    monkeypatch,
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)

        if command[:2] == [
            "git",
            "rev-parse",
        ]:
            return _completed_process(
                command,
                returncode=128,
                stderr="fatal: ambiguous argument HEAD",
            )

        return _completed_process(command)

    monkeypatch.setattr(
        repair_commit.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        RepairCommitError,
        match="unable to read repair commit SHA",
    ):
        create_repair_commit(
            18,
            "Handle empty files",
        )

    assert len(calls) == 3


@pytest.mark.parametrize(
    "execution_error",
    [
        OSError("git executable not found"),
        subprocess.TimeoutExpired(
            cmd=[
                "git",
                "add",
                "--all",
            ],
            timeout=30,
        ),
    ],
)
def test_create_repair_commit_reports_execution_error(
    monkeypatch,
    execution_error,
):
    def fake_run(command, **kwargs):
        raise execution_error

    monkeypatch.setattr(
        repair_commit.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        RepairCommitError,
        match="unable to stage repair changes",
    ):
        create_repair_commit(
            18,
            "Handle empty files",
        )