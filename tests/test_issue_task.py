"""Tests for structured issue task parsing."""

import json

from corecoder.issue_task import IssueTask, parse_issue_task


def test_parse_issue_task_extracts_structured_fields():
    body = """
    Update corecoder/tools/run_tests.py.

    ## Acceptance criteria

    - [ ] Return failed test names
    - [x] Keep existing JSON fields
    """

    task = parse_issue_task(
        title="  Fix structured test reporting  ",
        body=body,
        labels=["bug", "agent", "bug"],
        issue_number=12,
        issue_url="https://example.com/issues/12",
    )

    assert task.title == "Fix structured test reporting"
    assert task.body == body.strip()
    assert task.labels == ["bug", "agent"]
    assert task.issue_number == 12
    assert task.issue_url == "https://example.com/issues/12"
    assert task.acceptance_criteria == [
        "Return failed test names",
        "Keep existing JSON fields",
    ]
    assert task.referenced_files == [
        "corecoder/tools/run_tests.py",
    ]
    assert task.ambiguities == []


def test_empty_issue_records_ambiguities():
    task = parse_issue_task(
        title="   ",
        body="   ",
    )

    assert task.title == ""
    assert task.body == ""
    assert task.acceptance_criteria == []
    assert task.referenced_files == []
    assert task.ambiguities == [
        "Issue title is empty",
        "Issue body is empty",
        "No explicit Markdown acceptance criteria found",
    ]


def test_issue_without_checkboxes_records_ambiguity():
    task = parse_issue_task(
        title="Fix the agent loop",
        body="Please update corecoder/agent.py.",
    )

    assert task.acceptance_criteria == []
    assert task.referenced_files == ["corecoder/agent.py"]
    assert task.ambiguities == [
        "No explicit Markdown acceptance criteria found",
    ]


def test_checkbox_variants_are_extracted_and_deduplicated():
    task = parse_issue_task(
        title="Improve parser",
        body="""
        - [ ] First requirement
        * [x] Second requirement
        - [X] Third requirement
        - [ ] First requirement
        """,
    )

    assert task.acceptance_criteria == [
        "First requirement",
        "Second requirement",
        "Third requirement",
    ]


def test_referenced_files_preserve_order_and_remove_duplicates():
    task = parse_issue_task(
        title="Update docs/guide.md",
        body="""
        Change corecoder/issue_task.py and tests/test_issue_task.py.
        Also review corecoder/issue_task.py.

        - [ ] All referenced files are updated
        """,
    )

    assert task.referenced_files == [
        "docs/guide.md",
        "corecoder/issue_task.py",
        "tests/test_issue_task.py",
    ]


def test_labels_are_trimmed_and_deduplicated():
    task = parse_issue_task(
        title="Normalize labels",
        body="- [ ] Labels are normalized",
        labels=[
            " bug ",
            "",
            "bug",
            "enhancement",
            " enhancement ",
        ],
    )

    assert task.labels == [
        "bug",
        "enhancement",
    ]


def test_issue_task_to_dict():
    task = IssueTask(
        title="Fix parser",
        labels=["bug"],
        issue_number=7,
        acceptance_criteria=["Parser works"],
    )

    result = task.to_dict()

    assert result["title"] == "Fix parser"
    assert result["labels"] == ["bug"]
    assert result["issue_number"] == 7
    assert result["acceptance_criteria"] == ["Parser works"]
    assert result["referenced_files"] == []
    assert result["ambiguities"] == []


def test_issue_task_to_json_preserves_unicode():
    task = IssueTask(
        title="修复 Issue 解析器",
        labels=["缺陷"],
    )

    result = json.loads(task.to_json())

    assert result["title"] == "修复 Issue 解析器"
    assert result["labels"] == ["缺陷"]