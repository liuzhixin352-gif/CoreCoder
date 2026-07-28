"""Tests for the structured pytest execution tool."""

import json

from corecoder.tools.run_tests import (
    RunTestsTool,
    _parse_failed_tests,
    _parse_pytest_counts,
    _status_from_exit_code,
    )


def _execute_json(tool: RunTestsTool, **kwargs) -> dict:
    """Execute the tool and decode its JSON response."""
    return json.loads(tool.execute(**kwargs))

def test_status_from_exit_code():
    assert _status_from_exit_code(0) == "passed"
    assert _status_from_exit_code(1) == "failed"
    assert _status_from_exit_code(2) == "interrupted"
    assert _status_from_exit_code(3) == "internal_error"
    assert _status_from_exit_code(4) == "usage_error"
    assert _status_from_exit_code(5) == "no_tests"
    assert _status_from_exit_code(6) == "warning_limit_exceeded"
    assert _status_from_exit_code(99) == "error"

def test_parse_pytest_counts():
    output = (
        "2 failed, 3 passed, 1 skipped, "
        "4 errors, 1 xfailed, 1 xpassed in 0.50s"
    )

    result = _parse_pytest_counts(output)

    assert result == {
        "passed_count": 3,
        "failed_count": 2,
        "error_count": 4,
        "skipped_count": 1,
        "xfailed_count": 1,
        "xpassed_count": 1,
    }
def test_parse_failed_tests():
    output = (
        "================ short test summary info ================\n"
        "FAILED tests/test_math.py::test_add - assert 2 == 3\n"
        "FAILED tests/test_user.py::test_login - AssertionError\n"
        "2 failed in 0.25s\n"
    )

    result = _parse_failed_tests(output)

    assert result == [
        "tests/test_math.py::test_add",
        "tests/test_user.py::test_login",
    ]
def test_parse_failed_tests_restores_missing_file_path(tmp_path):
    test_file = tmp_path / "test_failing.py"
    test_file.write_text(
        "def test_wrong_result():\n"
        "    assert False\n",
        encoding="utf-8",
    )

    output = (
        "================ short test summary info ================\n"
        "FAILED ::test_wrong_result - AssertionError\n"
        "1 failed in 0.25s\n"
    )

    result = _parse_failed_tests(output, test_file)

    assert result == [
        f"{test_file}::test_wrong_result",
    ]
#路径不存在时，返回结构化error
def test_run_tests_missing_path(tmp_path):
    tool = RunTestsTool()
    missing_path = tmp_path / "does_not_exist"

    result = _execute_json(tool, path=str(missing_path))

    assert result["status"] == "error"
    assert "not found" in result["error"].lower()

#测试通过时，status=passed、exit_code=0
def test_run_tests_passing_file(tmp_path):
    tool = RunTestsTool()
    test_file = tmp_path / "test_passing.py"
    test_file.write_text(
        "def test_addition():\n"
        "    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )

    result = _execute_json(tool, path=str(test_file), timeout=30)

    assert result["status"] == "passed"
    assert result["exit_code"] == 0
    assert "1 passed" in result["output"]
    assert result["passed_count"] == 1
    assert result["failed_count"] == 0
    assert result["error_count"] == 0
    assert result["failed_tests"] == []
    assert result["duration_seconds"] >= 0

#测试失败时，status=failed、exit_code非0
def test_run_tests_failing_file(tmp_path):
    tool = RunTestsTool()
    test_file = tmp_path / "test_failing.py"
    test_file.write_text(
        "def test_wrong_result():\n"
        "    assert 1 + 1 == 3\n",
        encoding="utf-8",
    )

    result = _execute_json(tool, path=str(test_file), timeout=30)

    assert result["status"] == "failed"
    assert result["exit_code"] != 0
    assert "1 failed" in result["output"]
    assert result["passed_count"] == 0
    assert result["failed_count"] == 1
    assert result["error_count"] == 0
    assert len(result["failed_tests"]) == 1
    assert result["failed_tests"][0].endswith(
    "test_failing.py::test_wrong_result"
)
#测试运行过久时，返回timeout
def test_run_tests_timeout(tmp_path):
    tool = RunTestsTool()
    test_file = tmp_path / "test_slow.py"
    test_file.write_text(
        "import time\n\n"
        "def test_slow():\n"
        "    time.sleep(10)\n",
        encoding="utf-8",
    )

    result = _execute_json(tool, path=str(test_file), timeout=1)

    assert result["status"] == "timeout"
    assert result["timeout_seconds"] == 1

def test_run_tests_no_tests_collected(tmp_path):
    tool = RunTestsTool()

    result = _execute_json(tool, path=str(tmp_path), timeout=30)

    assert result["status"] == "no_tests"
    assert result["exit_code"] == 5
    assert result["passed_count"] == 0
    assert result["failed_count"] == 0
    assert result["error_count"] == 0
    assert result["failed_tests"] == []
    assert "no tests" in result["output"].lower()