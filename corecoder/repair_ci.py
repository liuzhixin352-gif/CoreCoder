"""Query CI status for a repair pull request."""

import socket
import json
from dataclasses import dataclass
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from .github_issue import (
    _github_request_headers,
    _github_token,
    _http_error_message,
)
from time import monotonic, sleep

class RepairCIStatusError(RuntimeError):
    """Raised when repair CI status cannot be queried."""
_VALID_CHECK_RUN_STATUSES = frozenset(
    {
        "queued",
        "in_progress",
        "completed",
        "waiting",
        "requested",
        "pending",
    }
)

_VALID_CHECK_RUN_CONCLUSIONS = frozenset(
    {
        "action_required",
        "cancelled",
        "failure",
        "neutral",
        "success",
        "skipped",
        "stale",
        "timed_out",
    }
)

@dataclass(frozen=True)
class RepairCheckRun:
    """One GitHub check run for a repair commit."""

    name: str
    status: str
    conclusion: str | None
    details_url: str | None


@dataclass(frozen=True)
class RepairCIStatus:
    """Combined CI status for one repair commit."""

    repository: str
    commit_sha: str
    state: str
    check_runs: tuple[RepairCheckRun, ...]

def _parse_check_run(
    check_run: object,
) -> RepairCheckRun:
    """Validate and parse one GitHub check run."""
    if not isinstance(check_run, dict):
        raise RepairCIStatusError(
            "GitHub API returned invalid "
            "check run data"
        )

    name = check_run.get("name")
    status = check_run.get("status")
    conclusion = check_run.get("conclusion")
    details_url = check_run.get("details_url")

    if (
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(status, str)
        or not status.strip()
        or (
            conclusion is not None
            and (
                not isinstance(conclusion, str)
                or not conclusion.strip()
            )
        )
        or (
            details_url is not None
            and (
                not isinstance(details_url, str)
                or not details_url.strip()
            )
        )
    ):
        raise RepairCIStatusError(
            "GitHub API returned invalid "
            "check run data"
        )
    if status not in _VALID_CHECK_RUN_STATUSES:
        raise RepairCIStatusError(
            "GitHub API returned invalid "
            "check run data"
        )

    if status == "completed":
        if conclusion not in _VALID_CHECK_RUN_CONCLUSIONS:
            raise RepairCIStatusError(
                "GitHub API returned invalid "
                "check run data"
            )
    elif conclusion is not None:
        raise RepairCIStatusError(
            "GitHub API returned invalid "
            "check run data"
        )

    return RepairCheckRun(
        name=name,
        status=status,
        conclusion=conclusion,
        details_url=details_url,
    )

def _combined_state(
    check_runs: tuple[RepairCheckRun, ...],
) -> str:
    """Return the combined state for all check runs."""
    if not check_runs:
        return "no_checks"

    if any(
        check_run.status != "completed"
        for check_run in check_runs
    ):
        return "pending"

    if all(
        check_run.conclusion == "success"
        for check_run in check_runs
    ):
        return "success"

    return "failure"


def fetch_repair_ci_status(
    repository: str,
    commit_sha: str,
    *,
    timeout: int = 20,
) -> RepairCIStatus:
    """Fetch the latest GitHub check runs for a commit."""
    if _github_token() is None:
        raise RepairCIStatusError(
            "GitHub token is required to query "
            "repair CI status"
        )
    request = Request(
        (
            "https://api.github.com/repos/"
            f"{repository}/commits/{commit_sha}/check-runs"
            "?filter=latest&per_page=100"
        ),
        headers=_github_request_headers(),
        method="GET",
    )

    try:
        with urlopen(
            request,
            timeout=timeout,
        ) as response:
            raw_response = response.read()
    except HTTPError as error:
        raise RepairCIStatusError(
            _http_error_message(error)
        ) from error
    except (TimeoutError, socket.timeout) as error:
        raise RepairCIStatusError(
            f"GitHub API request timed out after "
            f"{timeout} seconds"
        ) from error
    except URLError as error:
        reason = error.reason

        if isinstance(
            reason,
            (TimeoutError, socket.timeout),
        ):
            raise RepairCIStatusError(
                f"GitHub API request timed out after "
                f"{timeout} seconds"
            ) from error

        raise RepairCIStatusError(
            f"GitHub API network error: {reason}"
        ) from error
    except OSError as error:
        raise RepairCIStatusError(
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
        raise RepairCIStatusError(
            "GitHub API returned invalid JSON"
        ) from error

    if not isinstance(payload, dict):
        raise RepairCIStatusError(
            "GitHub API returned an invalid "
            "response object"
        )

    try:
        raw_check_runs = payload["check_runs"]
    except KeyError as error:
        raise RepairCIStatusError(
            "GitHub API returned invalid "
            "check run data"
        ) from error

    if not isinstance(raw_check_runs, list):
        raise RepairCIStatusError(
            "GitHub API returned invalid "
            "check run data"
        )

    check_runs = tuple(
    _parse_check_run(check_run)
    for check_run in raw_check_runs
    )

    return RepairCIStatus(
        repository=repository,
        commit_sha=commit_sha,
        state=_combined_state(check_runs),
        check_runs=check_runs,
    )

def wait_for_repair_ci_status(
    repository: str,
    commit_sha: str,
    *,
    timeout: int = 300,
    poll_interval: int = 5,
) -> RepairCIStatus:
    """Poll GitHub until repair CI reaches a final state."""
    if timeout <= 0:
        raise RepairCIStatusError(
            "CI polling timeout must be "
            "greater than zero"
        )

    if poll_interval <= 0:
        raise RepairCIStatusError(
            "CI polling interval must be "
            "greater than zero"
        )
    deadline = monotonic() + timeout

    while True:
        ci_status = fetch_repair_ci_status(
            repository,
            commit_sha,
        )

        if ci_status.state in {
            "success",
            "failure",
        }:
            return ci_status

        current_time = monotonic()

        if current_time >= deadline:
            raise RepairCIStatusError(
                "Timed out waiting for repair CI "
                f"status after {timeout} seconds"
            )

        remaining_timeout = deadline - current_time

        sleep(
            min(
                poll_interval,
                remaining_timeout,
            )
        )
