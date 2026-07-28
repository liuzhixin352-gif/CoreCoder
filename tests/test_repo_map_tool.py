"""Tests for the repository mapping tool."""

import json

from corecoder.tools.repo_map import RepoMapTool


def _execute_json(tool: RepoMapTool, **kwargs) -> dict:
    """Execute the tool and decode its JSON response."""
    return json.loads(tool.execute(**kwargs))


def test_repo_map_missing_path(tmp_path):
    tool = RepoMapTool()
    missing_path = tmp_path / "does_not_exist"

    result = _execute_json(tool, path=str(missing_path))

    assert result["status"] == "error"
    assert "not found" in result["error"].lower()


def test_repo_map_rejects_non_python_file(tmp_path):
    tool = RepoMapTool()
    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello\n", encoding="utf-8")

    result = _execute_json(tool, path=str(text_file))

    assert result["status"] == "error"
    assert "not a python file" in result["error"].lower()


def test_repo_map_rejects_invalid_max_files(tmp_path):
    tool = RepoMapTool()

    result = _execute_json(
        tool,
        path=str(tmp_path),
        max_files=0,
    )

    assert result["status"] == "error"
    assert "at least 1" in result["error"].lower()


def test_repo_map_extracts_top_level_symbols(tmp_path):
    tool = RepoMapTool()
    source_file = tmp_path / "sample.py"
    source_file.write_text(
        "import os\n"
        "import json as json_module\n"
        "from pathlib import Path\n"
        "from .helpers import helper\n"
        "\n"
        "class Example:\n"
        "    def method(self):\n"
        "        return 1\n"
        "\n"
        "def helper_function():\n"
        "    return Path('.')\n"
        "\n"
        "async def async_helper():\n"
        "    return os.getcwd()\n",
        encoding="utf-8",
    )

    result = _execute_json(tool, path=str(source_file))

    assert result["status"] == "ok"
    assert result["file_count"] == 1
    assert result["parsed_file_count"] == 1
    assert result["truncated"] is False
    assert result["syntax_errors"] == []
    assert result["read_errors"] == []

    file_info = result["files"][0]

    assert file_info["path"] == "sample.py"
    assert file_info["classes"] == ["Example"]
    assert file_info["functions"] == [
        "helper_function",
        "async_helper",
    ]
    assert file_info["imports"] == [
        "os",
        "json",
        "pathlib",
        ".helpers",
    ]

    # Class methods should not be reported as top-level functions.
    assert "method" not in file_info["functions"]


def test_repo_map_records_syntax_errors(tmp_path):
    tool = RepoMapTool()
    broken_file = tmp_path / "broken.py"
    broken_file.write_text(
        "def broken(:\n"
        "    pass\n",
        encoding="utf-8",
    )

    result = _execute_json(tool, path=str(tmp_path))

    assert result["status"] == "ok"
    assert result["file_count"] == 1
    assert result["parsed_file_count"] == 0
    assert result["files"] == []
    assert len(result["syntax_errors"]) == 1

    error = result["syntax_errors"][0]

    assert error["path"] == "broken.py"
    assert error["line"] == 1
    assert error["message"]


def test_repo_map_skips_ignored_directories(tmp_path):
    tool = RepoMapTool()

    visible_file = tmp_path / "visible.py"
    visible_file.write_text("VALUE = 1\n", encoding="utf-8")

    ignored_directories = [
        ".venv",
        "__pycache__",
        "node_modules",
    ]

    for directory_name in ignored_directories:
        ignored_directory = tmp_path / directory_name
        ignored_directory.mkdir()
        (ignored_directory / "ignored.py").write_text(
            "VALUE = 2\n",
            encoding="utf-8",
        )

    result = _execute_json(tool, path=str(tmp_path))

    assert result["status"] == "ok"
    assert result["file_count"] == 1
    assert result["parsed_file_count"] == 1
    assert [item["path"] for item in result["files"]] == [
        "visible.py",
    ]


def test_repo_map_respects_max_files(tmp_path):
    tool = RepoMapTool()

    for file_name in ["a.py", "b.py", "c.py"]:
        (tmp_path / file_name).write_text(
            f'FILE_NAME = "{file_name}"\n',
            encoding="utf-8",
        )

    result = _execute_json(
        tool,
        path=str(tmp_path),
        max_files=2,
    )

    assert result["status"] == "ok"
    assert result["file_count"] == 2
    assert result["parsed_file_count"] == 2
    assert result["truncated"] is True
    assert [item["path"] for item in result["files"]] == [
        "a.py",
        "b.py",
    ]


def test_repo_map_scans_single_python_file(tmp_path):
    tool = RepoMapTool()
    source_file = tmp_path / "single.py"
    source_file.write_text(
        "class SingleClass:\n"
        "    pass\n",
        encoding="utf-8",
    )

    result = _execute_json(
        tool,
        path=str(source_file),
        max_files=1,
    )

    assert result["status"] == "ok"
    assert result["file_count"] == 1
    assert result["parsed_file_count"] == 1
    assert result["truncated"] is False
    assert result["files"][0]["path"] == "single.py"
    assert result["files"][0]["classes"] == ["SingleClass"]