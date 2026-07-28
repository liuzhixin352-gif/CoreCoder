"""Tests for the structured pytest execution tool."""

import json

from corecoder.tools.run_tests import RunTestsTool


def _execute_json(tool: RunTestsTool, **kwargs) -> dict:
    """Execute the tool and decode its JSON response."""
    return json.loads(tool.execute(**kwargs))

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