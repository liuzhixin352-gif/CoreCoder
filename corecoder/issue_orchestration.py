"""DevPilot GitHub Issue workflow orchestration."""

from collections.abc import Callable
from dataclasses import dataclass
from .repair_commit import RepairCommit
from .post_repair import PostRepairSummary
from .post_repair_validation import PostRepairValidation
from .repair_push import RepairPush
from .repair_pr import RepairPullRequest
from .repair_ci import RepairCIStatus

@dataclass(frozen=True)
class IssueWorkflowResult:
    """Structured result from one DevPilot Issue workflow run."""

    summary: PostRepairSummary
    validation: PostRepairValidation | None
    commit: RepairCommit | None
    push: RepairPush | None
    pull_request: RepairPullRequest | None
    ci_status: RepairCIStatus | None = None
    ci_retry_summary: PostRepairSummary | None = None
    ci_retry_validation: PostRepairValidation | None = None
    ci_retry_commit: RepairCommit | None = None
    ci_retry_push: RepairPush | None = None
    ci_retry_status: RepairCIStatus | None = None

def run_issue_workflow(
    *,
    issue_prompt: str,
    dry_run: bool,
    run_agent: Callable[[str], None],
    repair_branch: str | None = None,
    collect_summary: (
        Callable[[str], PostRepairSummary] | None
    ) = None,
    run_validation: (
        Callable[[], PostRepairValidation] | None
    ) = None,
    create_commit: Callable[[], object] | None = None,
    push_commit: (
        Callable[[str, str], RepairPush] | None
    ) = None,
    create_pull_request: (
        Callable[[RepairPush], RepairPullRequest] | None
    ) = None,
    wait_for_ci: (
        Callable[[RepairPullRequest], RepairCIStatus] | None
    ) = None,
    build_ci_failure_prompt: (
        Callable[[RepairCIStatus], str] | None
    ) = None,
    wait_for_retry_ci: (
        Callable[[RepairPush], RepairCIStatus] | None
    ) = None,
) -> IssueWorkflowResult | None:
    """Run the DevPilot Issue workflow."""
    run_agent(issue_prompt)

    ci_retry_summary = None
    ci_retry_validation = None
    ci_retry_commit = None
    ci_retry_status = None

    if dry_run:
        return None

    if repair_branch is None:
        raise ValueError(
            "repair_branch is required for repair workflow"
        )

    if collect_summary is None:
        raise ValueError(
            "collect_summary is required for repair workflow"
        )

    summary = collect_summary(repair_branch)

    if not summary.has_changes:
        return IssueWorkflowResult(
            summary=summary,
            validation=None,
            commit=None,
            push=None,
            pull_request=None,
        )

    if run_validation is None:
        raise ValueError(
            "run_validation is required for repair changes"
        )

    validation = run_validation()

    if not validation.passed:
        return IssueWorkflowResult(
            summary=summary,
            validation=validation,
            commit=None,
            push=None,
            pull_request=None,
        )

    if create_commit is None:
        raise ValueError(
            "create_commit is required after passed validation"
        )

    commit = create_commit()

    if push_commit is None:
        raise ValueError(
            "push_commit is required after commit creation"
        )

    push = push_commit(
    repair_branch,
    commit.sha,
)

    if create_pull_request is None:
        raise ValueError(
            "create_pull_request is required after push"
        )

    pull_request = create_pull_request(push)

    if wait_for_ci is None:
        raise ValueError(
            "wait_for_ci is required after pull request creation"
        )

    ci_status = wait_for_ci(pull_request)
    ci_retry_summary = None
    ci_retry_push = None
    if ci_status.state == "failure":
        if build_ci_failure_prompt is None:
            raise ValueError(
                "build_ci_failure_prompt is required "
                "after CI failure"
            )

        ci_failure_prompt = build_ci_failure_prompt(
            ci_status
        )
        run_agent(ci_failure_prompt)

        ci_retry_summary = collect_summary(
            repair_branch
        )
        if not ci_retry_summary.has_changes:
            raise ValueError(
                "CI repair produced no repository changes"
            )
        ci_retry_validation = run_validation()

        if not ci_retry_validation.passed:
            raise ValueError(
                "CI repair validation failed"
            )

        ci_retry_commit = create_commit()

        ci_retry_push = push_commit(
            repair_branch,
            ci_retry_commit.sha,
        )

        if wait_for_retry_ci is None:
            raise ValueError(
                "wait_for_retry_ci is required after CI retry push"
            )

        ci_retry_status = wait_for_retry_ci(
            ci_retry_push
        )
        if ci_retry_status.state == "failure":
            raise ValueError(
                "CI retry failed"
            )
    return IssueWorkflowResult(
    summary=summary,
    validation=validation,
    commit=commit,
    push=push,
    pull_request=pull_request,
    ci_status=ci_status,
    ci_retry_summary=ci_retry_summary,
    ci_retry_validation=ci_retry_validation,
    ci_retry_commit=ci_retry_commit,
    ci_retry_push=ci_retry_push,
    ci_retry_status=ci_retry_status,
)