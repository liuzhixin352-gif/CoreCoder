"""Tests for GitHub issue reference parsing."""

import json
import socket
from io import BytesIO
from urllib.error import HTTPError,URLError

import pytest




from corecoder.github_issue import (
    GitHubIssueReference,
    GitHubIssueReferenceError,
    GitHubIssueFetchError,
    fetch_github_issue,
    parse_github_issue_url,
    parse_repository_issue,
    resolve_github_issue_reference,
)


def test_github_issue_reference_properties():
    reference = GitHubIssueReference(
        owner="octocat",
        repo="Hello-World",
        issue_number=7,
    )

    assert reference.repository == "octocat/Hello-World"
    assert reference.issue_url == (
        "https://github.com/octocat/Hello-World/issues/7"
    )
    assert reference.api_url == (
        "https://api.github.com/repos/octocat/Hello-World/issues/7"
    )

    assert reference.to_dict() == {
        "owner": "octocat",
        "repo": "Hello-World",
        "repository": "octocat/Hello-World",
        "issue_number": 7,
        "issue_url": (
            "https://github.com/octocat/Hello-World/issues/7"
        ),
        "api_url": (
            "https://api.github.com/repos/"
            "octocat/Hello-World/issues/7"
        ),
    }


def test_parse_github_issue_url():
    reference = parse_github_issue_url(
        "https://github.com/octocat/Hello-World/issues/12"
    )

    assert reference.owner == "octocat"
    assert reference.repo == "Hello-World"
    assert reference.issue_number == 12


def test_parse_github_issue_url_accepts_www_and_trailing_slash():
    reference = parse_github_issue_url(
        "https://www.github.com/octocat/Hello-World/issues/12/"
    )

    assert reference.repository == "octocat/Hello-World"
    assert reference.issue_number == 12


@pytest.mark.parametrize(
    ("issue_url", "expected_message"),
    [
        (123, "issue_url must be a string"),
        ("   ", "issue_url must not be empty"),
        (
            "http://github.com/octocat/Hello-World/issues/1",
            "issue_url must use https",
        ),
        (
            "https://example.com/octocat/Hello-World/issues/1",
            "issue_url must point to github.com",
        ),
        (
            "https://github.com/octocat/Hello-World/pull/1",
            "issue_url must have the form",
        ),
        (
            "https://github.com/octocat/Hello-World/issues/0",
            "issue_url must have the form",
        ),
    ],
)
def test_parse_github_issue_url_rejects_invalid_inputs(
    issue_url,
    expected_message,
):
    with pytest.raises(
        GitHubIssueReferenceError,
        match=expected_message,
    ):
        parse_github_issue_url(issue_url)


def test_parse_repository_issue():
    reference = parse_repository_issue(
        repository="  octocat/Hello-World  ",
        issue_number=21,
    )

    assert reference.owner == "octocat"
    assert reference.repo == "Hello-World"
    assert reference.issue_number == 21


@pytest.mark.parametrize(
    ("repository", "issue_number", "expected_message"),
    [
        (123, 1, "repository must be a string"),
        ("octocat", 1, "repository must have the form owner/repo"),
        (
            "octocat/Hello-World/extra",
            1,
            "repository must have the form owner/repo",
        ),
        (
            "octocat/Hello-World",
            True,
            "issue_number must be an integer",
        ),
        (
            "octocat/Hello-World",
            "1",
            "issue_number must be an integer",
        ),
        (
            "octocat/Hello-World",
            0,
            "issue_number must be greater than zero",
        ),
    ],
)
def test_parse_repository_issue_rejects_invalid_inputs(
    repository,
    issue_number,
    expected_message,
):
    with pytest.raises(
        GitHubIssueReferenceError,
        match=expected_message,
    ):
        parse_repository_issue(
            repository=repository,
            issue_number=issue_number,
        )


def test_resolve_github_issue_reference_from_url():
    reference = resolve_github_issue_reference(
        issue_url=(
            "https://github.com/octocat/"
            "Hello-World/issues/9"
        )
    )

    assert reference.repository == "octocat/Hello-World"
    assert reference.issue_number == 9


def test_resolve_github_issue_reference_from_repository():
    reference = resolve_github_issue_reference(
        repository="octocat/Hello-World",
        issue_number=9,
    )

    assert reference.repository == "octocat/Hello-World"
    assert reference.issue_number == 9


def test_resolve_rejects_both_input_forms():
    with pytest.raises(
        GitHubIssueReferenceError,
        match="provide either issue_url or",
    ):
        resolve_github_issue_reference(
            issue_url=(
                "https://github.com/octocat/"
                "Hello-World/issues/9"
            ),
            repository="octocat/Hello-World",
            issue_number=9,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"issue_number": 9},
        {"repository": "octocat/Hello-World"},
    ],
)

def test_resolve_rejects_incomplete_input(kwargs):
    with pytest.raises(
        GitHubIssueReferenceError,
        match="provide issue_url or both repository and issue_number",
    ):
        resolve_github_issue_reference(**kwargs)

class _FakeResponse:
    """Minimal context-managed HTTP response for tests."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return self._body


def _issue_payload(**overrides) -> dict:
    """Return a representative GitHub issue API payload."""
    payload = {
        "title": "Fix structured test reporting",
        "body": (
            "Update corecoder/tools/run_tests.py.\n\n"
            "- [ ] Return failed test names\n"
            "- [ ] Preserve existing JSON fields"
        ),
        "number": 12,
        "html_url": (
            "https://github.com/octocat/Hello-World/issues/12"
        ),
        "labels": [
            {"name": "bug"},
            "agent",
            {"name": "bug"},
        ],
    }
    payload.update(overrides)
    return payload


def _mock_json_response(
    monkeypatch,
    payload,
    captured: dict | None = None,
) -> None:
    """Replace urlopen with a successful JSON response."""

    def fake_urlopen(request, timeout):
        if captured is not None:
            captured["request"] = request
            captured["timeout"] = timeout

        body = json.dumps(payload).encode("utf-8")
        return _FakeResponse(body)

    monkeypatch.setattr(
        "corecoder.github_issue.urlopen",
        fake_urlopen,
    )


def test_fetch_github_issue_returns_structured_task(monkeypatch):
    _mock_json_response(
        monkeypatch,
        _issue_payload(),
    )

    task = fetch_github_issue(
        issue_url=(
            "https://github.com/octocat/"
            "Hello-World/issues/12"
        )
    )

    assert task.title == "Fix structured test reporting"
    assert task.issue_number == 12
    assert task.issue_url == (
        "https://github.com/octocat/Hello-World/issues/12"
    )
    assert task.labels == ["bug", "agent"]
    assert task.acceptance_criteria == [
        "Return failed test names",
        "Preserve existing JSON fields",
    ]
    assert task.referenced_files == [
        "corecoder/tools/run_tests.py",
    ]
    assert task.ambiguities == []





def test_fetch_github_issue_builds_expected_request(
monkeypatch,
):
    captured = {}

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    _mock_json_response(
        monkeypatch,
        _issue_payload(),
        captured,
    )

    fetch_github_issue(
        repository="octocat/Hello-World",
        issue_number=12,
        timeout=9,
    )

    request = captured["request"]
    headers = {
        name.lower(): value
        for name, value in request.header_items()
    }

    assert request.full_url == (
        "https://api.github.com/repos/"
        "octocat/Hello-World/issues/12"
    )
    assert request.get_method() == "GET"
    assert captured["timeout"] == 9
    assert headers["accept"] == "application/vnd.github+json"
    assert headers["user-agent"] == "CoreCoder-DevPilot"
    assert headers["x-github-api-version"] == "2026-03-10"
    assert "authorization" not in headers



def test_fetch_github_issue_uses_github_token(monkeypatch):
    captured = {}

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "primary-secret-token",
    )
    monkeypatch.setenv(
        "GH_TOKEN",
        "fallback-secret-token",
    )

    _mock_json_response(
        monkeypatch,
        _issue_payload(),
        captured,
    )

    fetch_github_issue(
        repository="octocat/Hello-World",
        issue_number=12,
    )

    headers = {
        name.lower(): value
        for name, value in captured["request"].header_items()
    }

    assert headers["authorization"] == (
        "Bearer primary-secret-token"
    )



def test_fetch_github_issue_falls_back_to_gh_token(
    monkeypatch,
):
    captured = {}

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv(
        "GH_TOKEN",
        "fallback-secret-token",
    )

    _mock_json_response(
        monkeypatch,
        _issue_payload(),
        captured,
    )

    fetch_github_issue(
        repository="octocat/Hello-World",
        issue_number=12,
    )

    headers = {
        name.lower(): value
        for name, value in captured["request"].header_items()
    }

    assert headers["authorization"] == (
        "Bearer fallback-secret-token"
    )


def test_fetch_github_issue_uses_reference_url_as_fallback(
monkeypatch,
):
    payload = _issue_payload()
    payload.pop("html_url")

    _mock_json_response(
        monkeypatch,
        payload,
    )

    task = fetch_github_issue(
        repository="octocat/Hello-World",
        issue_number=12,
    )

    assert task.issue_url == (
        "https://github.com/octocat/Hello-World/issues/12"
    )


def test_fetch_github_issue_rejects_pull_request(
monkeypatch,
):
    payload = _issue_payload(
        pull_request={
            "url": (
                "https://api.github.com/repos/"
                "octocat/Hello-World/pulls/12"
            )
        }
    )

    _mock_json_response(
        monkeypatch,
        payload,
    )

    with pytest.raises(
        GitHubIssueFetchError,
        match="points to a pull request",
    ):
        fetch_github_issue(
            repository="octocat/Hello-World",
            issue_number=12,
        )


@pytest.mark.parametrize(
(
    "status_code",
    "headers",
    "api_message",
    "expected_message",
),
[
    (
        401,
        {},
        "Bad credentials",
        "GitHub authentication failed",
    ),
    (
        403,
        {"X-RateLimit-Remaining": "0"},
        "API rate limit exceeded",
        "GitHub API rate limit exceeded",
    ),
    (
        403,
        {"X-RateLimit-Remaining": "42"},
        "Resource not accessible",
        "GitHub API access forbidden",
    ),
    (
        404,
        {},
        "Not Found",
        "GitHub issue not found or not accessible",
    ),
    (
        410,
        {},
        "Gone",
        "GitHub issue is no longer available",
    ),
],
)



def test_fetch_github_issue_converts_http_errors(
    monkeypatch,
    status_code,
    headers,
    api_message,
    expected_message,
):
    def fake_urlopen(request, timeout):
        body = json.dumps(
            {"message": api_message}
        ).encode("utf-8")

        raise HTTPError(
            url=request.full_url,
            code=status_code,
            msg=api_message,
            hdrs=headers,
            fp=BytesIO(body),
        )

    monkeypatch.setattr(
        "corecoder.github_issue.urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        GitHubIssueFetchError,
        match=expected_message,
    ):
        fetch_github_issue(
            repository="octocat/Hello-World",
            issue_number=12,
        )



def test_fetch_github_issue_handles_socket_timeout(
monkeypatch,
):
    def fake_urlopen(request, timeout):
        raise socket.timeout("timed out")

    monkeypatch.setattr(
        "corecoder.github_issue.urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        GitHubIssueFetchError,
        match="timed out after 7 seconds",
    ):
        fetch_github_issue(
            repository="octocat/Hello-World",
            issue_number=12,
            timeout=7,
        )


def test_fetch_github_issue_handles_url_error(
    monkeypatch,
):
    def fake_urlopen(request, timeout):
        raise URLError("connection reset")

    monkeypatch.setattr(
        "corecoder.github_issue.urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        GitHubIssueFetchError,
        match="GitHub API network error",
    ):
        fetch_github_issue(
            repository="octocat/Hello-World",
            issue_number=12,
        )


def test_fetch_github_issue_rejects_invalid_json(
monkeypatch,
):
    def fake_urlopen(request, timeout):
        return _FakeResponse(b"not valid json")

    monkeypatch.setattr(
        "corecoder.github_issue.urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        GitHubIssueFetchError,
        match="returned invalid JSON",
    ):
        fetch_github_issue(
            repository="octocat/Hello-World",
            issue_number=12,
        )


def test_fetch_github_issue_rejects_non_object_response(
    monkeypatch,
):
    _mock_json_response(
        monkeypatch,
        ["not", "an", "object"],
    )

    with pytest.raises(
        GitHubIssueFetchError,
        match="invalid response object",
    ):
        fetch_github_issue(
            repository="octocat/Hello-World",
            issue_number=12,
        )


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        (
            {"title": None},
            "missing a valid title",
        ),
        (
            {"body": ["invalid"]},
            "contains an invalid body",
        ),
        (
            {"number": True},
            "missing a valid issue number",
        ),
        (
            {"labels": "bug"},
            "returned invalid labels data",
        ),
    ],
)


def test_fetch_github_issue_rejects_invalid_fields(
    monkeypatch,
    overrides,
    expected_message,
):
    _mock_json_response(
        monkeypatch,
        _issue_payload(**overrides),
    )

    with pytest.raises(
        GitHubIssueFetchError,
        match=expected_message,
    ):
        fetch_github_issue(
            repository="octocat/Hello-World",
            issue_number=12,
        )


@pytest.mark.parametrize(
("timeout", "expected_message"),
[
    (True, "timeout must be an integer"),
    ("20", "timeout must be an integer"),
    (0, "timeout must be greater than zero"),
    (-1, "timeout must be greater than zero"),
],
)



def test_fetch_github_issue_rejects_invalid_timeout(
    timeout,
    expected_message,
):
    with pytest.raises(
        GitHubIssueFetchError,
        match=expected_message,
    ):
        fetch_github_issue(
            repository="octocat/Hello-World",
            issue_number=12,
            timeout=timeout,
        )