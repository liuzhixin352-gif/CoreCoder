"""Tool for fetching and structuring a real GitHub issue."""

from __future__ import annotations

import json

from corecoder.github_issue import (
    GitHubIssueFetchError,
    GitHubIssueReferenceError,
    fetch_github_issue,
)

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


class FetchIssueTool(Tool):
    """Fetch a GitHub issue and return a structured repair task."""

    name = "fetch_issue"

    description = (
        "Fetch a real GitHub Issue by its URL, or by repository and issue "
        "number, then convert it into a structured repair task. The result "
        "includes the title, body, labels, acceptance criteria, referenced "
        "files, and ambiguities. Use this before investigating or modifying "
        "code for a GitHub Issue. Do not use this for pull requests."
    )

    parameters = {
        "type": "object",
        "properties": {
            "issue_url": {
                "type": "string",
                "description": (
                    "Full GitHub Issue URL, for example "
                    "https://github.com/owner/repo/issues/12. "
                    "Do not also provide repository or issue_number."
                ),
            },
            "repository": {
                "type": "string",
                "description": (
                    "GitHub repository in owner/repo form. "
                    "Must be provided together with issue_number."
                ),
            },
            "issue_number": {
                "type": "integer",
                "description": (
                    "Positive GitHub Issue number. "
                    "Must be provided together with repository."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": (
                    "Maximum GitHub API request time in seconds "
                    "(default: 20)"
                ),
            },
        },
        "required": [],
    }

    def execute(
        self,
        issue_url: str | None = None,
        repository: str | None = None,
        issue_number: int | None = None,
        timeout: int = 20,
    ) -> str:
        """Fetch a GitHub Issue and return structured JSON."""
        try:
            task = fetch_github_issue(
                issue_url=issue_url,
                repository=repository,
                issue_number=issue_number,
                timeout=timeout,
            )
        except (
            GitHubIssueReferenceError,
            GitHubIssueFetchError,
        ) as error:
            return _error_result(str(error))
        except Exception:
            return _error_result(
                "Unexpected error while fetching the GitHub issue"
            )

        return json.dumps(
            {
                "status": "ok",
                "task": task.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )