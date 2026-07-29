"""Tests for the GitHub issue fetching tool."""

import json

import pytest

from corecoder.github_issue import (
    GitHubIssueFetchError,
    GitHubIssueReferenceError,
)
from corecoder.issue_task import IssueTask
from corecoder.tools.fetch_issue import FetchIssueTool


def _execute(**kwargs) -> dict:
    """Execute FetchIssueTool and decode its JSON result."""
    return json.loads(FetchIssueTool().execute(**kwargs))


def _sample_task() -> IssueTask:
    """Return a representative structured issue task."""
    return IssueTask(
        title="Fix structured test reporting",
        body=(
            "Update corecoder/tools/run_tests.py.\n\n"
            "- [ ] Return failed test names\n"
            "- [ ] Preserve existing JSON fields"
        ),
        labels=["bug", "agent"],
        issue_number=12,
        issue_url=(
            "https://github.com/octocat/"
            "Hello-World/issues/12"
        ),
        acceptance_criteria=[
            "Return failed test names",
            "Preserve existing JSON fields",
        ],
        referenced_files=[
            "corecoder/tools/run_tests.py",
        ],
        ambiguities=[],
    )


def test_fetch_issue_tool_schema():
    tool = FetchIssueTool()
    schema = tool.schema()

    assert tool.name == "fetch_issue"
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "fetch_issue"

    parameters = schema["function"]["parameters"]

    assert parameters["required"] == []
    assert set(parameters["properties"]) == {
        "issue_url",
        "repository",
        "issue_number",
        "timeout",
    }

    assert "token" not in parameters["properties"]
    assert "GITHUB_TOKEN" not in parameters["properties"]
    assert "GH_TOKEN" not in parameters["properties"]


def test_fetch_issue_tool_forwards_issue_url(monkeypatch):
    captured = {}

    def fake_fetch_github_issue(**kwargs):
        captured.update(kwargs)
        return _sample_task()

    monkeypatch.setattr(
        "corecoder.tools.fetch_issue.fetch_github_issue",
        fake_fetch_github_issue,
    )

    result = _execute(
        issue_url=(
            "https://github.com/octocat/"
            "Hello-World/issues/12"
        ),
        timeout=9,
    )

    assert captured == {
        "issue_url": (
            "https://github.com/octocat/"
            "Hello-World/issues/12"
        ),
        "repository": None,
        "issue_number": None,
        "timeout": 9,
    }

    assert result["status"] == "ok"

    task = result["task"]

    assert task["title"] == "Fix structured test reporting"
    assert task["labels"] == ["bug", "agent"]
    assert task["issue_number"] == 12
    assert task["acceptance_criteria"] == [
        "Return failed test names",
        "Preserve existing JSON fields",
    ]
    assert task["referenced_files"] == [
        "corecoder/tools/run_tests.py",
    ]
    assert task["ambiguities"] == []


def test_fetch_issue_tool_forwards_repository_and_number(
    monkeypatch,
):
    captured = {}

    def fake_fetch_github_issue(**kwargs):
        captured.update(kwargs)
        return _sample_task()

    monkeypatch.setattr(
        "corecoder.tools.fetch_issue.fetch_github_issue",
        fake_fetch_github_issue,
    )

    result = _execute(
        repository="octocat/Hello-World",
        issue_number=12,
    )

    assert captured == {
        "issue_url": None,
        "repository": "octocat/Hello-World",
        "issue_number": 12,
        "timeout": 20,
    }

    assert result["status"] == "ok"
    assert result["task"]["issue_number"] == 12


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (
            GitHubIssueReferenceError(
                "provide issue_url or both repository and issue_number"
            ),
            "provide issue_url or both repository and issue_number",
        ),
        (
            GitHubIssueFetchError(
                "GitHub issue not found or not accessible"
            ),
            "GitHub issue not found or not accessible",
        ),
    ],
)
def test_fetch_issue_tool_converts_expected_errors(
    monkeypatch,
    error,
    expected_message,
):
    def fake_fetch_github_issue(**kwargs):
        raise error

    monkeypatch.setattr(
        "corecoder.tools.fetch_issue.fetch_github_issue",
        fake_fetch_github_issue,
    )

    result = _execute(
        repository="octocat/Hello-World",
        issue_number=999,
    )

    assert result == {
        "status": "error",
        "error": expected_message,
    }


def test_fetch_issue_tool_hides_unexpected_exception(
    monkeypatch,
):
    def fake_fetch_github_issue(**kwargs):
        raise RuntimeError("private internal details")

    monkeypatch.setattr(
        "corecoder.tools.fetch_issue.fetch_github_issue",
        fake_fetch_github_issue,
    )

    result = _execute(
        repository="octocat/Hello-World",
        issue_number=12,
    )

    assert result == {
        "status": "error",
        "error": (
            "Unexpected error while fetching the GitHub issue"
        ),
    }

    assert "private internal details" not in result["error"]


def test_fetch_issue_tool_preserves_unicode(monkeypatch):
    task = IssueTask(
        title="修复 GitHub Issue 获取功能",
        body="- [ ] 正确读取问题",
        labels=["缺陷"],
        issue_number=8,
        issue_url="https://github.com/example/project/issues/8",
        acceptance_criteria=["正确读取问题"],
    )

    def fake_fetch_github_issue(**kwargs):
        return task

    monkeypatch.setattr(
        "corecoder.tools.fetch_issue.fetch_github_issue",
        fake_fetch_github_issue,
    )

    result = _execute(
        repository="example/project",
        issue_number=8,
    )

    assert result["status"] == "ok"
    assert result["task"]["title"] == "修复 GitHub Issue 获取功能"
    assert result["task"]["labels"] == ["缺陷"]
    assert result["task"]["acceptance_criteria"] == [
        "正确读取问题",
    ]