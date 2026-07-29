"""GitHub issue reference parsing and validation."""

from __future__ import annotations

import json
import os
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen

from corecoder.issue_task import IssueTask,parse_issue_task

_REPOSITORY_PATTERN = re.compile(
    r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)$"
)

_ISSUE_PATH_PATTERN = re.compile(
    r"^/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+)/"
    r"issues/"
    r"(?P<number>[1-9]\d*)"
    r"/?$"
)

_GITHUB_API_VERSION = "2026-03-10"
_GITHUB_USER_AGENT = "CoreCoder-DevPilot"
_DEFAULT_TIMEOUT_SECONDS = 20


class GitHubIssueReferenceError(ValueError):
    """Raised when a GitHub issue reference is invalid."""

class GitHubIssueFetchError(RuntimeError):
    """Raised when a GitHub issue cannot be fetched.""" 

@dataclass(frozen=True)
class GitHubIssueReference:
    """Normalized location of one GitHub issue."""

    owner: str
    repo: str
    issue_number: int

    @property
    def repository(self) -> str:
        """Return the repository in owner/name form."""
        return f"{self.owner}/{self.repo}"

    @property
    def issue_url(self) -> str:
        """Return the canonical GitHub issue URL."""
        return (
            f"https://github.com/{self.repository}"
            f"/issues/{self.issue_number}"
        )

    @property
    def api_url(self) -> str:
        """Return the GitHub REST API URL."""
        return (
            f"https://api.github.com/repos/{self.repository}"
            f"/issues/{self.issue_number}"
        )

    def to_dict(self) -> dict:
        """Return the normalized reference as a dictionary."""
        return {
            "owner": self.owner,
            "repo": self.repo,
            "repository": self.repository,
            "issue_number": self.issue_number,
            "issue_url": self.issue_url,
            "api_url": self.api_url,
        }


def parse_github_issue_url(
    issue_url: str,
) -> GitHubIssueReference:
    """Parse a public github.com issue URL."""
    if not isinstance(issue_url, str):
        raise GitHubIssueReferenceError(
            "issue_url must be a string"
        )

    cleaned_url = issue_url.strip()

    if not cleaned_url:
        raise GitHubIssueReferenceError(
            "issue_url must not be empty"
        )

    parsed = urlparse(cleaned_url)
    hostname = (parsed.hostname or "").lower()

    if parsed.scheme.lower() != "https":
        raise GitHubIssueReferenceError(
            "issue_url must use https"
        )

    if hostname not in {"github.com", "www.github.com"}:
        raise GitHubIssueReferenceError(
            "issue_url must point to github.com"
        )

    match = _ISSUE_PATH_PATTERN.fullmatch(parsed.path)

    if not match:
        raise GitHubIssueReferenceError(
            "issue_url must have the form "
            "https://github.com/owner/repo/issues/number"
        )

    return GitHubIssueReference(
        owner=match.group("owner"),
        repo=match.group("repo"),
        issue_number=int(match.group("number")),
    )


def parse_repository_issue(
    repository: str,
    issue_number: int,
) -> GitHubIssueReference:
    """Build an issue reference from repository and issue number."""
    if not isinstance(repository, str):
        raise GitHubIssueReferenceError(
            "repository must be a string"
        )

    cleaned_repository = repository.strip()
    match = _REPOSITORY_PATTERN.fullmatch(cleaned_repository)

    if not match:
        raise GitHubIssueReferenceError(
            "repository must have the form owner/repo"
        )

    if isinstance(issue_number, bool) or not isinstance(
        issue_number,
        int,
    ):
        raise GitHubIssueReferenceError(
            "issue_number must be an integer"
        )

    if issue_number < 1:
        raise GitHubIssueReferenceError(
            "issue_number must be greater than zero"
        )

    return GitHubIssueReference(
        owner=match.group("owner"),
        repo=match.group("repo"),
        issue_number=issue_number,
    )


def resolve_github_issue_reference(
    *,
    issue_url: str | None = None,
    repository: str | None = None,
    issue_number: int | None = None,
) -> GitHubIssueReference:
    """Resolve exactly one supported form of GitHub issue input."""
    if issue_url is not None:
        if repository is not None or issue_number is not None:
            raise GitHubIssueReferenceError(
                "provide either issue_url or "
                "repository with issue_number, not both"
            )

        return parse_github_issue_url(issue_url)

    if repository is None or issue_number is None:
        raise GitHubIssueReferenceError(
            "provide issue_url or both repository and issue_number"
        )

    return parse_repository_issue(
        repository=repository,
        issue_number=issue_number,
    )

def _github_token() -> str | None:
    """Return a GitHub token from supported environment variables."""
    for variable_name in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = os.getenv(variable_name, "").strip()

        if value:
            return value

    return None


def _github_request_headers() -> dict[str, str]:
    """Build headers for the GitHub REST API."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": _GITHUB_USER_AGENT,
        "X-GitHub-Api-Version": _GITHUB_API_VERSION,
    }

    token = _github_token()

    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def _read_http_error_detail(error: HTTPError) -> str | None:
    """Read a safe message from a GitHub HTTP error response."""
    try:
        raw_body = error.read().decode(
            "utf-8",
            errors="replace",
        )
        payload = json.loads(raw_body)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    message = payload.get("message")

    if isinstance(message, str) and message.strip():
        return message.strip()

    return None


def _http_error_message(error: HTTPError) -> str:
    """Convert a GitHub HTTP error into a useful message."""
    detail = _read_http_error_detail(error)

    if error.code == 401:
        message = (
            "GitHub authentication failed; check GITHUB_TOKEN "
            "or GH_TOKEN"
        )
    elif error.code == 403:
        remaining = error.headers.get("X-RateLimit-Remaining")

        if remaining == "0":
            message = "GitHub API rate limit exceeded"
        else:
            message = "GitHub API access forbidden"
    elif error.code == 404:
        message = "GitHub issue not found or not accessible"
    elif error.code == 410:
        message = "GitHub issue is no longer available"
    else:
        message = (
            f"GitHub API request failed with HTTP status "
            f"{error.code}"
        )

    if detail:
        return f"{message}: {detail}"

    return message


def _extract_label_names(payload: dict) -> list[str]:
    """Extract label names from a GitHub issue response."""
    raw_labels = payload.get("labels", [])

    if raw_labels is None:
        return []

    if not isinstance(raw_labels, list):
        raise GitHubIssueFetchError(
            "GitHub API returned invalid labels data"
        )

    labels: list[str] = []

    for raw_label in raw_labels:
        if isinstance(raw_label, str):
            name = raw_label.strip()
        elif isinstance(raw_label, dict):
            raw_name = raw_label.get("name")
            name = (
                raw_name.strip()
                if isinstance(raw_name, str)
                else ""
            )
        else:
            name = ""

        if name and name not in labels:
            labels.append(name)

    return labels


def _issue_task_from_payload(
    payload: dict,
    reference: GitHubIssueReference,
) -> IssueTask:
    """Convert a GitHub API response into an IssueTask."""
    if "pull_request" in payload:
        raise GitHubIssueFetchError(
            "The GitHub reference points to a pull request, "
            "not an issue"
        )

    title = payload.get("title")

    if not isinstance(title, str):
        raise GitHubIssueFetchError(
            "GitHub API response is missing a valid title"
        )

    body = payload.get("body")

    if body is None:
        body = ""
    elif not isinstance(body, str):
        raise GitHubIssueFetchError(
            "GitHub API response contains an invalid body"
        )

    number = payload.get("number")

    if isinstance(number, bool) or not isinstance(number, int):
        raise GitHubIssueFetchError(
            "GitHub API response is missing a valid issue number"
        )

    html_url = payload.get("html_url")

    if not isinstance(html_url, str) or not html_url.strip():
        html_url = reference.issue_url

    return parse_issue_task(
        title=title,
        body=body,
        labels=_extract_label_names(payload),
        issue_number=number,
        issue_url=html_url,
    )


def fetch_github_issue(
    *,
    issue_url: str | None = None,
    repository: str | None = None,
    issue_number: int | None = None,
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
) -> IssueTask:
    """Fetch one GitHub issue and return a structured IssueTask."""
    reference = resolve_github_issue_reference(
        issue_url=issue_url,
        repository=repository,
        issue_number=issue_number,
    )

    if isinstance(timeout, bool) or not isinstance(timeout, int):
        raise GitHubIssueFetchError(
            "timeout must be an integer"
        )

    if timeout < 1:
        raise GitHubIssueFetchError(
            "timeout must be greater than zero"
        )

    request = Request(
        reference.api_url,
        headers=_github_request_headers(),
        method="GET",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            raw_response = response.read()
    except HTTPError as error:
        raise GitHubIssueFetchError(
            _http_error_message(error)
        ) from error
    except (TimeoutError, socket.timeout) as error:
        raise GitHubIssueFetchError(
            f"GitHub API request timed out after {timeout} seconds"
        ) from error
    except URLError as error:
        reason = error.reason

        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise GitHubIssueFetchError(
                f"GitHub API request timed out after "
                f"{timeout} seconds"
            ) from error

        raise GitHubIssueFetchError(
            f"GitHub API network error: {reason}"
        ) from error
    except OSError as error:
        raise GitHubIssueFetchError(
            f"GitHub API network error: {error}"
        ) from error

    try:
        decoded_response = raw_response.decode("utf-8")
        payload = json.loads(decoded_response)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GitHubIssueFetchError(
            "GitHub API returned invalid JSON"
        ) from error

    if not isinstance(payload, dict):
        raise GitHubIssueFetchError(
            "GitHub API returned an invalid response object"
        )

    return _issue_task_from_payload(
        payload=payload,
        reference=reference,
    )