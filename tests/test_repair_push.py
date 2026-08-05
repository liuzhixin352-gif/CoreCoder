"""Tests for automatically pushing validated repair branches."""

import subprocess

import pytest

import corecoder.repair_push as repair_push
from corecoder.repair_push import (
    RepairPush,
    RepairPushError,
    push_repair_branch,
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


def test_push_repair_branch_verifies_and_pushes_branch(
    monkeypatch,
):
    calls = []
    branch = "devpilot/issue-21-fix-scan-limit"
    commit_sha = (
        "0123456789abcdef"
        "0123456789abcdef"
        "01234567"
    )

    responses = iter(
        [
            _completed_process(
                [],
                stdout=f"{commit_sha}\n",
            ),
            _completed_process(
                [],
                stdout=(
                    "branch set up to track "
                    "'origin/devpilot/issue-21-fix-scan-limit'\n"
                ),
            ),
        ]
    )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        result = next(responses)
        result.args = command
        return result

    monkeypatch.setattr(
        repair_push.subprocess,
        "run",
        fake_run,
    )

    result = push_repair_branch(
        branch,
        commit_sha,
    )

    assert result == RepairPush(
        remote="origin",
        branch=branch,
        commit_sha=commit_sha,
    )

    assert [
        command
        for command, _kwargs in calls
    ] == [
        [
            "git",
            "rev-parse",
            "--verify",
            f"refs/heads/{branch}",
        ],
        [
            "git",
            "push",
            "--set-upstream",
            "origin",
            branch,
        ],
    ]

    expected_kwargs = {
        "cwd": None,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 60,
        "check": False,
    }

    assert all(
        kwargs == expected_kwargs
        for _command, kwargs in calls
    )


def test_push_repair_branch_rejects_commit_mismatch(
    monkeypatch,
):
    calls = []
    expected_sha = "a" * 40
    actual_sha = "b" * 40

    def fake_run(command, **kwargs):
        calls.append(command)

        return _completed_process(
            command,
            stdout=f"{actual_sha}\n",
        )

    monkeypatch.setattr(
        repair_push.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        RepairPushError,
        match=(
            "does not point to the validated "
            "repair commit"
        ),
    ):
        push_repair_branch(
            "devpilot/issue-21-fix-scan-limit",
            expected_sha,
        )

    assert len(calls) == 1


def test_push_repair_branch_reports_verification_failure(
    monkeypatch,
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)

        return _completed_process(
            command,
            returncode=128,
            stderr="fatal: unknown revision",
        )

    monkeypatch.setattr(
        repair_push.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        RepairPushError,
        match="unable to verify repair branch",
    ):
        push_repair_branch(
            "devpilot/issue-21-fix-scan-limit",
            "a" * 40,
        )

    assert len(calls) == 1


def test_push_repair_branch_reports_push_failure(
    monkeypatch,
):
    calls = []
    commit_sha = "a" * 40

    def fake_run(command, **kwargs):
        calls.append(command)

        if command[:2] == [
            "git",
            "rev-parse",
        ]:
            return _completed_process(
                command,
                stdout=f"{commit_sha}\n",
            )

        return _completed_process(
            command,
            returncode=1,
            stderr="fatal: unable to access remote",
        )

    monkeypatch.setattr(
        repair_push.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        RepairPushError,
        match="unable to push repair branch",
    ):
        push_repair_branch(
            "devpilot/issue-21-fix-scan-limit",
            commit_sha,
        )

    assert len(calls) == 2


@pytest.mark.parametrize(
    "execution_error",
    [
        OSError("git executable not found"),
        subprocess.TimeoutExpired(
            cmd=[
                "git",
                "rev-parse",
                "--verify",
                (
                    "refs/heads/"
                    "devpilot/issue-21-fix-scan-limit"
                ),
            ],
            timeout=60,
        ),
    ],
)
def test_push_repair_branch_reports_execution_error(
    monkeypatch,
    execution_error,
):
    def fake_run(command, **kwargs):
        raise execution_error

    monkeypatch.setattr(
        repair_push.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        RepairPushError,
        match="unable to verify repair branch",
    ):
        push_repair_branch(
            "devpilot/issue-21-fix-scan-limit",
            "a" * 40,
        )