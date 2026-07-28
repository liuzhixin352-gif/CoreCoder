"""Tests for the structured issue parsing tool."""

import json

from corecoder.tools.parse_issue import ParseIssueTool


def _execute(**kwargs) -> dict:
    """Execute ParseIssueTool and decode its JSON result."""
    return json.loads(ParseIssueTool().execute(**kwargs))


def test_parse_issue_tool_schema():
    tool = ParseIssueTool()
    schema = tool.schema()

    assert tool.name == "parse_issue"
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "parse_issue"
    assert schema["function"]["parameters"]["required"] == ["title"]


def test_parse_issue_tool_returns_structured_task():
    result = _execute(
        title="  Fix structured test reporting  ",
        body="""
        Update corecoder/tools/run_tests.py.

        - [ ] Return failed test names
        - [x] Preserve existing JSON fields
        """,
        labels=["bug", "agent", "bug"],
        issue_number=12,
        issue_url="https://example.com/issues/12",
    )

    assert result["status"] == "ok"

    task = result["task"]

    assert task["title"] == "Fix structured test reporting"
    assert task["labels"] == ["bug", "agent"]
    assert task["issue_number"] == 12
    assert task["issue_url"] == "https://example.com/issues/12"
    assert task["acceptance_criteria"] == [
        "Return failed test names",
        "Preserve existing JSON fields",
    ]
    assert task["referenced_files"] == [
        "corecoder/tools/run_tests.py",
    ]
    assert task["ambiguities"] == []


def test_parse_issue_tool_allows_empty_optional_fields():
    result = _execute(title="Improve issue handling")

    assert result["status"] == "ok"
    assert result["task"]["title"] == "Improve issue handling"
    assert result["task"]["body"] == ""
    assert result["task"]["labels"] == []
    assert result["task"]["issue_number"] is None
    assert result["task"]["issue_url"] is None
    assert result["task"]["ambiguities"] == [
        "Issue body is empty",
        "No explicit Markdown acceptance criteria found",
    ]


def test_parse_issue_tool_records_empty_title_ambiguity():
    result = _execute(
        title="   ",
        body="- [ ] The issue is repaired",
    )

    assert result["status"] == "ok"
    assert result["task"]["title"] == ""
    assert result["task"]["acceptance_criteria"] == [
        "The issue is repaired",
    ]
    assert result["task"]["ambiguities"] == [
        "Issue title is empty",
    ]


def test_parse_issue_tool_rejects_non_string_title():
    result = _execute(title=123)

    assert result == {
        "status": "error",
        "error": "title must be a string",
    }


def test_parse_issue_tool_rejects_non_string_body():
    result = _execute(
        title="Test issue",
        body=["not", "a", "string"],
    )

    assert result == {
        "status": "error",
        "error": "body must be a string",
    }


def test_parse_issue_tool_rejects_non_list_labels():
    result = _execute(
        title="Test issue",
        labels="bug",
    )

    assert result == {
        "status": "error",
        "error": "labels must be an array of strings",
    }


def test_parse_issue_tool_rejects_non_string_label_items():
    result = _execute(
        title="Test issue",
        labels=["bug", 123],
    )

    assert result == {
        "status": "error",
        "error": "labels must contain only strings",
    }


def test_parse_issue_tool_rejects_invalid_issue_metadata():
    tool = ParseIssueTool()

    invalid_number = json.loads(
        tool.execute(
            title="Test issue",
            issue_number=True,
        )
    )

    invalid_url = json.loads(
        tool.execute(
            title="Test issue",
            issue_url=123,
        )
    )

    assert invalid_number == {
        "status": "error",
        "error": "issue_number must be an integer",
    }

    assert invalid_url == {
        "status": "error",
        "error": "issue_url must be a string",
    }