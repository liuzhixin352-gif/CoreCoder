"""Tests for automatically creating repair pull requests."""

import json
import pytest
import io
import socket
import corecoder.repair_pr as repair_pr
from urllib.error import HTTPError,URLError
from corecoder.repair_pr import (
    RepairPullRequest,
    RepairPullRequestError,
    create_repair_pull_request,
)
from corecoder.repair_push import RepairPush


class _FakeResponse:
    """Minimal context-managed HTTP response."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(
        self,
        exception_type,
        exception,
        traceback,
    ):
        return False

    def read(self) -> bytes:
        return self._body

class _RawResponse:
    """Context-managed HTTP response with raw bytes."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(
        self,
        exception_type,
        exception,
        traceback,
    ):
        return False

    def read(self) -> bytes:
        return self._body

def test_create_repair_pull_request_posts_validated_branch(
    monkeypatch,
):
    repository = "liuzhixin352-gif/CoreCoder"
    base_branch = "devpilot-v1"
    head_branch = "devpilot/issue-21-fix-scan-limit"
    commit_sha = "a" * 40
    captured = {}

    repair_push = RepairPush(
        remote="origin",
        branch=head_branch,
        commit_sha=commit_sha,
    )

    expected_title = (
        "Fix #21: Fix repository scan limit"
    )
    expected_body = (
        "## Summary\n\n"
        "Automated repair for GitHub Issue #21.\n\n"
        f"Validated commit: `{commit_sha}`\n\n"
        "Closes #21"
    )

    response_payload = {
        "number": 42,
        "html_url": (
            "https://github.com/"
            "liuzhixin352-gif/CoreCoder/pull/42"
        ),
        "title": expected_title,
        "base": {
            "ref": base_branch,
        },
        "head": {
            "ref": head_branch,
            "sha": commit_sha,
        },
    }

    def fake_urlopen(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout

        return _FakeResponse(response_payload)

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )
    monkeypatch.setattr(
        repair_pr,
        "urlopen",
        fake_urlopen,
        raising=False,
    )

    result = create_repair_pull_request(
        repository,
        21,
        "Fix repository scan limit",
        repair_push,
        base_branch,
    )

    assert result == RepairPullRequest(
        repository=repository,
        number=42,
        url=response_payload["html_url"],
        title=expected_title,
        base_branch=base_branch,
        head_branch=head_branch,
        commit_sha=commit_sha,
    )

    request = captured["request"]

    assert request.full_url == (
        "https://api.github.com/repos/"
        "liuzhixin352-gif/CoreCoder/pulls"
    )
    assert request.get_method() == "POST"
    assert captured["timeout"] == 20

    headers = {
        name.lower(): value
        for name, value in request.header_items()
    }

    assert headers["authorization"] == (
        "Bearer test-token"
    )
    assert headers["accept"] == (
        "application/vnd.github+json"
    )
    assert headers["x-github-api-version"] == (
        "2026-03-10"
    )

    assert request.data is not None

    request_payload = json.loads(
        request.data.decode("utf-8")
    )

    assert request_payload == {
        "title": expected_title,
        "body": expected_body,
        "head": head_branch,
        "base": base_branch,
        "maintainer_can_modify": True,
        "draft": False,
    }


def test_create_repair_pull_request_requires_github_token(
    monkeypatch,
):
    repair_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha="a" * 40,
    )

    monkeypatch.delenv(
        "GITHUB_TOKEN",
        raising=False,
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )

    def fail_urlopen(*args, **kwargs):
        raise AssertionError(
            "network request must not run without a GitHub token"
        )

    monkeypatch.setattr(
        repair_pr,
        "urlopen",
        fail_urlopen,
    )

    with pytest.raises(
        RepairPullRequestError,
        match="GitHub token is required",
    ):
        create_repair_pull_request(
            "liuzhixin352-gif/CoreCoder",
            21,
            "Fix repository scan limit",
            repair_push,
            "devpilot-v1",
        )

def test_create_repair_pull_request_reports_http_error(
    monkeypatch,
):
    repair_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha="a" * 40,
    )

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )

    def fail_urlopen(request, *, timeout):
        error_body = json.dumps(
            {
                "message": (
                    "A pull request already exists "
                    "for this branch"
                ),
            }
        ).encode("utf-8")

        raise HTTPError(
            request.full_url,
            422,
            "Unprocessable Entity",
            None,
            io.BytesIO(error_body),
        )

    monkeypatch.setattr(
        repair_pr,
        "urlopen",
        fail_urlopen,
    )

    with pytest.raises(
        RepairPullRequestError,
        match=(
            "HTTP status 422.*"
            "A pull request already exists"
        ),
    ):
        create_repair_pull_request(
            "liuzhixin352-gif/CoreCoder",
            21,
            "Fix repository scan limit",
            repair_push,
            "devpilot-v1",
        )

@pytest.mark.parametrize(
    (
        "execution_error",
        "expected_message",
    ),
    [
        (
            socket.timeout("timed out"),
            (
                "GitHub API request timed out "
                "after 20 seconds"
            ),
        ),
        (
            URLError("connection reset"),
            (
                "GitHub API network error: "
                "connection reset"
            ),
        ),
        (
            OSError("network unavailable"),
            (
                "GitHub API network error: "
                "network unavailable"
            ),
        ),
    ],
)
def test_create_repair_pull_request_reports_network_error(
    monkeypatch,
    execution_error,
    expected_message,
):
    repair_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha="a" * 40,
    )

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )

    def fail_urlopen(request, *, timeout):
        raise execution_error

    monkeypatch.setattr(
        repair_pr,
        "urlopen",
        fail_urlopen,
    )

    with pytest.raises(
        RepairPullRequestError,
        match=expected_message,
    ):
        create_repair_pull_request(
            "liuzhixin352-gif/CoreCoder",
            21,
            "Fix repository scan limit",
            repair_push,
            "devpilot-v1",
        )

def test_create_repair_pull_request_reports_invalid_json(
    monkeypatch,
):
    repair_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha="a" * 40,
    )

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )

    def fake_urlopen(request, *, timeout):
        return _RawResponse(
            b"this is not valid JSON"
        )

    monkeypatch.setattr(
        repair_pr,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        RepairPullRequestError,
        match="GitHub API returned invalid JSON",
    ):
        create_repair_pull_request(
            "liuzhixin352-gif/CoreCoder",
            21,
            "Fix repository scan limit",
            repair_push,
            "devpilot-v1",
        )

def test_create_repair_pull_request_rejects_non_object_response(
    monkeypatch,
):
    repair_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha="a" * 40,
    )

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )

    def fake_urlopen(request, *, timeout):
        return _RawResponse(b"[]")

    monkeypatch.setattr(
        repair_pr,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        RepairPullRequestError,
        match=(
            "GitHub API returned an invalid "
            "response object"
        ),
    ):
        create_repair_pull_request(
            "liuzhixin352-gif/CoreCoder",
            21,
            "Fix repository scan limit",
            repair_push,
            "devpilot-v1",
        )


def test_create_repair_pull_request_rejects_head_commit_mismatch(
    monkeypatch,
):
    repository = "liuzhixin352-gif/CoreCoder"
    base_branch = "devpilot-v1"
    head_branch = "devpilot/issue-21-fix-scan-limit"
    validated_commit_sha = "a" * 40
    unexpected_commit_sha = "b" * 40

    repair_push = RepairPush(
        remote="origin",
        branch=head_branch,
        commit_sha=validated_commit_sha,
    )

    response_payload = {
        "number": 42,
        "html_url": (
            "https://github.com/"
            "liuzhixin352-gif/CoreCoder/pull/42"
        ),
        "title": (
            "Fix #21: Fix repository scan limit"
        ),
        "base": {
            "ref": base_branch,
        },
        "head": {
            "ref": head_branch,
            "sha": unexpected_commit_sha,
        },
    }

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )

    def fake_urlopen(request, *, timeout):
        return _FakeResponse(response_payload)

    monkeypatch.setattr(
        repair_pr,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        RepairPullRequestError,
        match=(
            "pull request head does not match "
            "the validated repair commit"
        ),
    ):
        create_repair_pull_request(
            repository,
            21,
            "Fix repository scan limit",
            repair_push,
            base_branch,
        )

def test_create_repair_pull_request_rejects_invalid_pr_data(
    monkeypatch,
):
    repair_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha="a" * 40,
    )

    response_payload = {
        "number": 42,
        "html_url": (
            "https://github.com/"
            "liuzhixin352-gif/CoreCoder/pull/42"
        ),
        "title": (
            "Fix #21: Fix repository scan limit"
        ),
        "base": {
            "ref": "devpilot-v1",
        },
        "head": {},
    }

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )

    def fake_urlopen(request, *, timeout):
        return _FakeResponse(response_payload)

    monkeypatch.setattr(
        repair_pr,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        RepairPullRequestError,
        match=(
            "GitHub API returned invalid "
            "pull request data"
        ),
    ):
        create_repair_pull_request(
            "liuzhixin352-gif/CoreCoder",
            21,
            "Fix repository scan limit",
            repair_push,
            "devpilot-v1",
        )

@pytest.mark.parametrize(
    (
        "response_base_branch",
        "response_head_branch",
    ),
    [
        (
            "main",
            "devpilot/issue-21-fix-scan-limit",
        ),
        (
            "devpilot-v1",
            "devpilot/issue-99-unexpected",
        ),
    ],
)
def test_create_repair_pull_request_rejects_branch_mismatch(
    monkeypatch,
    response_base_branch,
    response_head_branch,
):
    repository = "liuzhixin352-gif/CoreCoder"
    expected_base_branch = "devpilot-v1"
    expected_head_branch = (
        "devpilot/issue-21-fix-scan-limit"
    )
    commit_sha = "a" * 40

    repair_push = RepairPush(
        remote="origin",
        branch=expected_head_branch,
        commit_sha=commit_sha,
    )

    response_payload = {
        "number": 42,
        "html_url": (
            "https://github.com/"
            "liuzhixin352-gif/CoreCoder/pull/42"
        ),
        "title": (
            "Fix #21: Fix repository scan limit"
        ),
        "base": {
            "ref": response_base_branch,
        },
        "head": {
            "ref": response_head_branch,
            "sha": commit_sha,
        },
    }

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )

    def fake_urlopen(request, *, timeout):
        return _FakeResponse(response_payload)

    monkeypatch.setattr(
        repair_pr,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        RepairPullRequestError,
        match=(
            "pull request branches do not match "
            "the requested repair branches"
        ),
    ):
        create_repair_pull_request(
            repository,
            21,
            "Fix repository scan limit",
            repair_push,
            expected_base_branch,
        )
