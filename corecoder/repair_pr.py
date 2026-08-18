"""Create a pull request for a validated Issue repair."""

import json
import socket
from dataclasses import dataclass
from urllib.request import Request, urlopen
from urllib.error import HTTPError,URLError

from .github_issue import (
    _github_request_headers,
    _github_token,
    _http_error_message,
    parse_repository_issue,
)
from .repair_push import RepairPush


@dataclass(frozen=True)
class RepairPullRequest:
    """Pull request created for one validated repair."""

    repository: str
    number: int
    url: str
    title: str
    base_branch: str
    head_branch: str
    commit_sha: str


class RepairPullRequestError(RuntimeError):
    """Raised when a repair pull request cannot be created."""


def create_repair_pull_request(
    repository: str,
    issue_number: int,
    issue_title: str,
    repair_push: RepairPush,
    base_branch: str,
    *,
    timeout: int = 20,
) -> RepairPullRequest:
    """Create a pull request for a pushed repair branch."""
    reference = parse_repository_issue(
        repository,
        issue_number,
    )
    if _github_token() is None:
        raise RepairPullRequestError(
            "GitHub token is required to create "
            "a repair pull request"
        )

    normalized_title = issue_title.strip()

    if not normalized_title:
        normalized_title = "GitHub Issue repair"

    pull_request_title = (
        f"Fix #{issue_number}: {normalized_title}"
    )
    pull_request_body = (
        "## Summary\n\n"
        f"Automated repair for GitHub Issue "
        f"#{issue_number}.\n\n"
        f"Validated commit: "
        f"`{repair_push.commit_sha}`\n\n"
        f"Closes #{issue_number}"
    )

    request_payload = {
        "title": pull_request_title,
        "body": pull_request_body,
        "head": repair_push.branch,
        "base": base_branch,
        "maintainer_can_modify": True,
        "draft": False,
    }

    request = Request(
        (
            "https://api.github.com/repos/"
            f"{reference.repository}/pulls"
        ),
        data=json.dumps(
            request_payload
        ).encode("utf-8"),
        headers=_github_request_headers(),
        method="POST",
    )

    try:
        with urlopen(
            request,
            timeout=timeout,
        ) as response:
            raw_response = response.read()
    except HTTPError as error:
        raise RepairPullRequestError(
            _http_error_message(error)
        ) from error
    except (TimeoutError, socket.timeout) as error:
        raise RepairPullRequestError(
            f"GitHub API request timed out after "
            f"{timeout} seconds"
        ) from error
    except URLError as error:
        reason = error.reason

        if isinstance(
            reason,
            (TimeoutError, socket.timeout),
        ):
            raise RepairPullRequestError(
                f"GitHub API request timed out after "
                f"{timeout} seconds"
            ) from error

        raise RepairPullRequestError(
            f"GitHub API network error: {reason}"
        ) from error
    except OSError as error:
        raise RepairPullRequestError(
            f"GitHub API network error: {error}"
        ) from error

    try:
        decoded_response = raw_response.decode(
            "utf-8"
        )
        payload = json.loads(decoded_response)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise RepairPullRequestError(
            "GitHub API returned invalid JSON"
        ) from error

    if not isinstance(payload, dict):
        raise RepairPullRequestError(
            "GitHub API returned an invalid "
            "response object"
        )

    try:
        pull_request_number = payload["number"]
        pull_request_url = payload["html_url"]
        pull_request_title = payload["title"]
        pull_request_base = payload["base"]
        pull_request_head = payload["head"]

        if not isinstance(
            pull_request_base,
            dict,
        ) or not isinstance(
            pull_request_head,
            dict,
        ):
            raise TypeError

        pull_request_base_branch = (
            pull_request_base["ref"]
        )
        pull_request_head_branch = (
            pull_request_head["ref"]
        )
        pull_request_head_sha = (
            pull_request_head["sha"]
        )
    except (KeyError, TypeError) as error:
        raise RepairPullRequestError(
            "GitHub API returned invalid "
            "pull request data"
        ) from error

    if (
        isinstance(pull_request_number, bool)
        or not isinstance(
            pull_request_number,
            int,
        )
        or pull_request_number < 1
        or not isinstance(
            pull_request_url,
            str,
        )
        or not pull_request_url.strip()
        or not isinstance(
            pull_request_title,
            str,
        )
        or not pull_request_title.strip()
        or not isinstance(
            pull_request_base_branch,
            str,
        )
        or not pull_request_base_branch.strip()
        or not isinstance(
            pull_request_head_branch,
            str,
        )
        or not pull_request_head_branch.strip()
        or not isinstance(
            pull_request_head_sha,
            str,
        )
        or not pull_request_head_sha.strip()
    ):
        raise RepairPullRequestError(
            "GitHub API returned invalid "
            "pull request data"
        )

    if (
        pull_request_base_branch != base_branch
        or pull_request_head_branch
        != repair_push.branch
    ):
        raise RepairPullRequestError(
            "pull request branches do not match "
            "the requested repair branches"
        )

    if pull_request_head_sha != repair_push.commit_sha:
        raise RepairPullRequestError(
            "pull request head does not match "
            "the validated repair commit"
        )

    return RepairPullRequest(
        repository=reference.repository,
        number=pull_request_number,
        url=pull_request_url,
        title=pull_request_title,
        base_branch=pull_request_base_branch,
        head_branch=pull_request_head_branch,
        commit_sha=pull_request_head_sha,
    )