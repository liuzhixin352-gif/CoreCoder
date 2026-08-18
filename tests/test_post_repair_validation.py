"""Tests for mandatory post-repair validation."""

import subprocess
import sys

import pytest

import corecoder.post_repair_validation as validation
from corecoder.post_repair_validation import (
    PostRepairValidation,
    PostRepairValidationError,
    run_post_repair_validation,
)


def _completed_process(
    *,
    returncode: int,
    stdout: str,
    stderr: str = "",
) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
    ]

    return subprocess.CompletedProcess(
        command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_run_pytest_invokes_expected_command(
    monkeypatch,
):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs

        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout="264 passed in 38.28s\n",
            stderr="",
        )

    monkeypatch.setattr(
        validation.subprocess,
        "run",
        fake_run,
    )

    result = validation._run_pytest("tests")

    assert result.returncode == 0
    assert captured["command"] == [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
    ]
    assert captured["kwargs"] == {
        "cwd": None,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 120,
        "check": False,
    }


def test_run_post_repair_validation_reports_passed_tests(
    monkeypatch,
):
    monkeypatch.setattr(
        validation,
        "_run_pytest",
        lambda *args, **kwargs: _completed_process(
            returncode=0,
            stdout="264 passed in 38.28s\n",
        ),
    )

    result = run_post_repair_validation()

    assert result == PostRepairValidation(
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
    assert result.passed is True


def test_run_post_repair_validation_reports_failed_tests(
    monkeypatch,
):
    monkeypatch.setattr(
        validation,
        "_run_pytest",
        lambda *args, **kwargs: _completed_process(
            returncode=1,
            stdout=(
                "================ short test summary ================\n"
                "FAILED tests/test_example.py::test_example\n"
                "2 failed, 10 passed in 1.25s\n"
            ),
        ),
    )

    result = run_post_repair_validation()

    assert result.status == "failed"
    assert result.exit_code == 1
    assert result.passed_count == 10
    assert result.failed_count == 2
    assert result.error_count == 0
    assert result.passed is False


def test_run_post_repair_validation_reports_no_tests(
    monkeypatch,
):
    monkeypatch.setattr(
        validation,
        "_run_pytest",
        lambda *args, **kwargs: _completed_process(
            returncode=5,
            stdout="no tests ran in 0.01s\n",
        ),
    )

    result = run_post_repair_validation()

    assert result.status == "no_tests"
    assert result.exit_code == 5
    assert result.passed_count == 0
    assert result.failed_count == 0
    assert result.error_count == 0
    assert result.passed is False


@pytest.mark.parametrize(
    "execution_error",
    [
        FileNotFoundError("python executable not found"),
        subprocess.TimeoutExpired(
            cmd=[
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
            ],
            timeout=120,
        ),
    ],
)
def test_run_post_repair_validation_reports_execution_error(
    monkeypatch,
    execution_error,
):
    def fake_run_pytest(*args, **kwargs):
        raise execution_error

    monkeypatch.setattr(
        validation,
        "_run_pytest",
        fake_run_pytest,
    )

    with pytest.raises(
        PostRepairValidationError,
        match="unable to run post-repair validation",
    ):
        run_post_repair_validation()