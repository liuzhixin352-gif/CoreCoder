"""Tool for converting raw issue fields into a structured repair task."""

import json

from corecoder.issue_task import parse_issue_task
from corecoder.permissions import ToolPermission

from .base import Tool


def _error_result(message: str) -> str:
    """Return a structured tool error as JSON text."""
    return json.dumps(
        {
            "status": "error",
            "error": message,
        },
        ensure_ascii=False,
        indent=2,
    )


class ParseIssueTool(Tool):
    """Convert issue fields into a structured repair task."""

    name = "parse_issue"
    permission = ToolPermission.READ

    description = (
        "Convert a software issue title, body, labels, number, and URL into "
        "a structured repair task. The result includes acceptance criteria, "
        "referenced files, and ambiguities. Use this before investigating "
        "or modifying code for an issue."
    )

    parameters = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Issue title",
            },
            "body": {
                "type": "string",
                "description": "Issue body or description (default: empty)",
            },
            "labels": {
                "type": "array",
                "items": {
                    "type": "string",
                },
                "description": "Issue labels (default: empty list)",
            },
            "issue_number": {
                "type": "integer",
                "description": "GitHub issue number",
            },
            "issue_url": {
                "type": "string",
                "description": "GitHub issue URL",
            },
        },
        "required": ["title"],
    }

    def execute(
        self,
        title: str,
        body: str = "",
        labels: list[str] | None = None,
        issue_number: int | None = None,
        issue_url: str | None = None,
    ) -> str:
        """Parse issue fields and return a structured JSON result."""
        if not isinstance(title, str):
            return _error_result("title must be a string")

        if not isinstance(body, str):
            return _error_result("body must be a string")

        if labels is not None:
            if not isinstance(labels, list):
                return _error_result("labels must be an array of strings")

            if any(not isinstance(label, str) for label in labels):
                return _error_result("labels must contain only strings")

        if issue_number is not None:
            if isinstance(issue_number, bool) or not isinstance(
                issue_number,
                int,
            ):
                return _error_result("issue_number must be an integer")

        if issue_url is not None and not isinstance(issue_url, str):
            return _error_result("issue_url must be a string")

        task = parse_issue_task(
            title=title,
            body=body,
            labels=labels,
            issue_number=issue_number,
            issue_url=issue_url,
        )

        result = {
            "status": "ok",
            "task": task.to_dict(),
        }

        return json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )