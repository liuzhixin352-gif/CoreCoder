"""Tests for repair pull request CI status queries."""
import io
import json
import pytest
import socket
from corecoder import repair_ci
from urllib.error import HTTPError, URLError


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

class _RawResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return False

    def read(self):
        return self._payload

def test_fetch_repair_ci_status_reports_success(
    monkeypatch,
):
    commit_sha = (
        "0123456789abcdef"
        "0123456789abcdef"
        "01234567"
    )
    captured = {}
    payload = {
        "total_count": 1,
        "check_runs": [
            {
                "name": "tests",
                "status": "completed",
                "conclusion": "success",
                "details_url": (
                    "https://github.com/"
                    "example/project/actions/runs/1"
                ),
            },
        ],
    }

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout

        return _FakeResponse(payload)

    monkeypatch.setattr(
        repair_ci,
        "urlopen",
        fake_urlopen,
    )

    result = repair_ci.fetch_repair_ci_status(
        "example/project",
        commit_sha,
    )

    request = captured["request"]

    assert request.full_url == (
        "https://api.github.com/repos/"
        "example/project/commits/"
        f"{commit_sha}/check-runs"
        "?filter=latest&per_page=100"
    )
    assert request.get_method() == "GET"
    assert captured["timeout"] == 20
    assert request.get_header(
        "Authorization"
    ) == "Bearer test-token"

    assert result.repository == "example/project"
    assert result.commit_sha == commit_sha
    assert result.state == "success"
    assert len(result.check_runs) == 1

    check_run = result.check_runs[0]

    assert check_run.name == "tests"
    assert check_run.status == "completed"
    assert check_run.conclusion == "success"
    assert check_run.details_url == (
        "https://github.com/"
        "example/project/actions/runs/1"
    )

@pytest.mark.parametrize(
    ("check_runs", "expected_state"),
    [
        (
            [
                {
                    "name": "tests",
                    "status": "in_progress",
                    "conclusion": None,
                    "details_url": None,
                },
            ],
            "pending",
        ),
        (
            [
                {
                    "name": "tests",
                    "status": "completed",
                    "conclusion": "failure",
                    "details_url": None,
                },
            ],
            "failure",
        ),
        (
            [],
            "no_checks",
        ),
    ],
)
def test_fetch_repair_ci_status_combines_states(
    monkeypatch,
    check_runs,
    expected_state,
):
    commit_sha = (
        "0123456789abcdef"
        "0123456789abcdef"
        "01234567"
    )

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )
    monkeypatch.setattr(
        repair_ci,
        "urlopen",
        lambda request, timeout: _FakeResponse(
            {
                "total_count": len(check_runs),
                "check_runs": check_runs,
            }
        ),
    )

    result = repair_ci.fetch_repair_ci_status(
        "example/project",
        commit_sha,
    )

    assert result.state == expected_state
    assert len(result.check_runs) == len(check_runs)

def test_fetch_repair_ci_status_requires_token(
    monkeypatch,
):
    commit_sha = (
        "0123456789abcdef"
        "0123456789abcdef"
        "01234567"
    )

    monkeypatch.delenv(
        "GITHUB_TOKEN",
        raising=False,
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )
    monkeypatch.setattr(
        repair_ci,
        "urlopen",
        lambda request, timeout: pytest.fail(
            "Network must not be called "
            "without a GitHub token"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="GitHub token is required",
    ):
        repair_ci.fetch_repair_ci_status(
            "example/project",
            commit_sha,
        )

def test_fetch_repair_ci_status_converts_http_error(
    monkeypatch,
):
    commit_sha = (
        "0123456789abcdef"
        "0123456789abcdef"
        "01234567"
    )
    error = HTTPError(
        url=(
            "https://api.github.com/repos/"
            "example/project/commits/"
            f"{commit_sha}/check-runs"
        ),
        code=403,
        msg="Forbidden",
        hdrs={},
        fp=io.BytesIO(
            b'{"message":'
            b'"Resource not accessible by integration"}'
        ),
    )

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )

    def fake_urlopen(request, timeout):
        raise error

    monkeypatch.setattr(
        repair_ci,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        repair_ci.RepairCIStatusError,
        match="GitHub API access forbidden",
    ):
        repair_ci.fetch_repair_ci_status(
            "example/project",
            commit_sha,
        )

@pytest.mark.parametrize(
    "network_error",
    [
        socket.timeout(),
        URLError(socket.timeout()),
    ],
)
def test_fetch_repair_ci_status_converts_timeout(
    monkeypatch,
    network_error,
):
    commit_sha = (
        "0123456789abcdef"
        "0123456789abcdef"
        "01234567"
    )

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )

    def fake_urlopen(request, timeout):
        raise network_error

    monkeypatch.setattr(
        repair_ci,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        repair_ci.RepairCIStatusError,
        match="timed out after 20 seconds",
    ):
        repair_ci.fetch_repair_ci_status(
            "example/project",
            commit_sha,
        )


def test_fetch_repair_ci_status_rejects_invalid_json(
    monkeypatch,
):
    commit_sha = (
        "0123456789abcdef"
        "0123456789abcdef"
        "01234567"
    )

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )
    monkeypatch.setattr(
        repair_ci,
        "urlopen",
        lambda request, timeout: _RawResponse(
            b"not valid JSON"
        ),
    )

    with pytest.raises(
        repair_ci.RepairCIStatusError,
        match="GitHub API returned invalid JSON",
    ):
        repair_ci.fetch_repair_ci_status(
            "example/project",
            commit_sha,
        )

def test_fetch_repair_ci_status_rejects_non_object_response(
    monkeypatch,
):
    commit_sha = (
        "0123456789abcdef"
        "0123456789abcdef"
        "01234567"
    )

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )
    monkeypatch.setattr(
        repair_ci,
        "urlopen",
        lambda request, timeout: _FakeResponse(
            []
        ),
    )

    with pytest.raises(
        repair_ci.RepairCIStatusError,
        match=(
            "GitHub API returned an invalid "
            "response object"
        ),
    ):
        repair_ci.fetch_repair_ci_status(
            "example/project",
            commit_sha,
        )

def test_fetch_repair_ci_status_rejects_missing_check_runs(
    monkeypatch,
):
    commit_sha = (
        "0123456789abcdef"
        "0123456789abcdef"
        "01234567"
    )

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )
    monkeypatch.setattr(
        repair_ci,
        "urlopen",
        lambda request, timeout: _FakeResponse(
            {
                "total_count": 0,
            }
        ),
    )

    with pytest.raises(
        repair_ci.RepairCIStatusError,
        match=(
            "GitHub API returned invalid "
            "check run data"
        ),
    ):
        repair_ci.fetch_repair_ci_status(
            "example/project",
            commit_sha,
        )

@pytest.mark.parametrize(
    "check_run",
    [
        {
            "name": "",
            "status": "completed",
            "conclusion": "success",
            "details_url": None,
        },
        {
            "name": "tests",
            "status": 123,
            "conclusion": "success",
            "details_url": None,
        },
        {
            "name": "tests",
            "status": "completed",
            "conclusion": 123,
            "details_url": None,
        },
        {
            "name": "tests",
            "status": "completed",
            "conclusion": "success",
            "details_url": 123,
        },
    ],
)
def test_fetch_repair_ci_status_rejects_invalid_check_run_fields(
    monkeypatch,
    check_run,
):
    commit_sha = (
        "0123456789abcdef"
        "0123456789abcdef"
        "01234567"
    )

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )
    monkeypatch.setattr(
        repair_ci,
        "urlopen",
        lambda request, timeout: _FakeResponse(
            {
                "total_count": 1,
                "check_runs": [check_run],
            }
        ),
    )

    with pytest.raises(
        repair_ci.RepairCIStatusError,
        match=(
            "GitHub API returned invalid "
            "check run data"
        ),
    ):
        repair_ci.fetch_repair_ci_status(
            "example/project",
            commit_sha,
        )

@pytest.mark.parametrize(
    "check_run",
    [
        {
            "name": "tests",
            "status": "unknown",
            "conclusion": None,
            "details_url": None,
        },
        {
            "name": "tests",
            "status": "completed",
            "conclusion": None,
            "details_url": None,
        },
        {
            "name": "tests",
            "status": "queued",
            "conclusion": "success",
            "details_url": None,
        },
        {
            "name": "tests",
            "status": "completed",
            "conclusion": "unknown",
            "details_url": None,
        },
    ],
)
def test_fetch_repair_ci_status_rejects_invalid_check_run_state(
    monkeypatch,
    check_run,
):
    commit_sha = (
        "0123456789abcdef"
        "0123456789abcdef"
        "01234567"
    )

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )
    monkeypatch.setattr(
        repair_ci,
        "urlopen",
        lambda request, timeout: _FakeResponse(
            {
                "total_count": 1,
                "check_runs": [check_run],
            }
        ),
    )

    with pytest.raises(
        repair_ci.RepairCIStatusError,
        match=(
            "GitHub API returned invalid "
            "check run data"
        ),
    ):
        repair_ci.fetch_repair_ci_status(
            "example/project",
            commit_sha,
        )

def test_fetch_repair_ci_status_converts_os_error(
    monkeypatch,
):
    commit_sha = (
        "0123456789abcdef"
        "0123456789abcdef"
        "01234567"
    )

    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "GH_TOKEN",
        raising=False,
    )

    def fake_urlopen(request, timeout):
        raise OSError("connection was reset")

    monkeypatch.setattr(
        repair_ci,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        repair_ci.RepairCIStatusError,
        match=(
            "GitHub API network error: "
            "connection was reset"
        ),
    ):
        repair_ci.fetch_repair_ci_status(
            "example/project",
            commit_sha,
        )
