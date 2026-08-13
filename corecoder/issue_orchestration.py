"""DevPilot GitHub Issue workflow orchestration."""
import json
from typing import Self
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


from .repair_commit import RepairCommit
from .post_repair import PostRepairSummary
from .post_repair_validation import PostRepairValidation
from .repair_push import RepairPush
from .repair_pr import RepairPullRequest
from .repair_ci import RepairCheckRun, RepairCIStatus


class WorkflowState(str, Enum):
    """Explicit stages of a DevPilot Issue workflow."""

    REPAIR = "repair"
    COLLECT_SUMMARY = "collect_summary"
    VALIDATE = "validate"
    COMMIT = "commit"
    PUSH = "push"
    CREATE_PR = "create_pr"
    WAIT_CI = "wait_ci"
    CI_REPAIR = "ci_repair"
    CI_RETRY_SUMMARY = "ci_retry_summary"
    CI_RETRY_VALIDATE = "ci_retry_validate"
    CI_RETRY_COMMIT = "ci_retry_commit"
    CI_RETRY_PUSH = "ci_retry_push"
    WAIT_RETRY_CI = "wait_retry_ci"
    COMPLETED = "completed"


_ALLOWED_TRANSITIONS = {
    WorkflowState.REPAIR: frozenset({
        WorkflowState.COLLECT_SUMMARY,
    }),
    WorkflowState.COLLECT_SUMMARY: frozenset({
        WorkflowState.VALIDATE,
    }),
    WorkflowState.VALIDATE: frozenset({
        WorkflowState.COMMIT,
    }),
    WorkflowState.COMMIT: frozenset({
        WorkflowState.PUSH,
    }),
    WorkflowState.PUSH: frozenset({
        WorkflowState.CREATE_PR,
    }),
    WorkflowState.CREATE_PR: frozenset({
        WorkflowState.WAIT_CI,
    }),
    WorkflowState.WAIT_CI: frozenset({
        WorkflowState.CI_REPAIR,
        WorkflowState.COMPLETED,
    }),
    WorkflowState.CI_REPAIR: frozenset({
        WorkflowState.CI_RETRY_SUMMARY,
    }),
    WorkflowState.CI_RETRY_SUMMARY: frozenset({
        WorkflowState.CI_RETRY_VALIDATE,
    }),
    WorkflowState.CI_RETRY_VALIDATE: frozenset({
        WorkflowState.CI_RETRY_COMMIT,
    }),
    WorkflowState.CI_RETRY_COMMIT: frozenset({
        WorkflowState.CI_RETRY_PUSH,
    }),
    WorkflowState.CI_RETRY_PUSH: frozenset({
        WorkflowState.WAIT_RETRY_CI,
    }),
    WorkflowState.WAIT_RETRY_CI: frozenset({
        WorkflowState.COMPLETED,
    }),
    WorkflowState.COMPLETED: frozenset(),
}


@dataclass
class WorkflowStateMachine:
    """Track the current state of an Issue workflow."""

    state: WorkflowState = WorkflowState.REPAIR

    def transition(self, state: WorkflowState) -> None:
        allowed_states = _ALLOWED_TRANSITIONS[self.state]

        if state not in allowed_states:
            raise ValueError(
                "Invalid workflow transition: "
                f"{self.state.value} -> {state.value}"
            )

        self.state = state

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: "WorkflowCheckpoint",
    ) -> Self:
        return cls(state=checkpoint.state)


@dataclass(frozen=True)
class WorkflowCheckpoint:
    """Minimal durable state for an Issue workflow."""

    workflow_id: str
    state: WorkflowState
    repair_branch: str
    repair_base_branch: str | None = None
    summary: PostRepairSummary | None = None
    validation: PostRepairValidation | None = None
    commit: RepairCommit | None = None
    push: RepairPush | None = None
    pull_request: RepairPullRequest | None = None
    ci_status: RepairCIStatus | None = None
    ci_retry_summary: PostRepairSummary | None = None
    ci_retry_validation: PostRepairValidation | None = None
    ci_retry_commit: RepairCommit | None = None
    ci_retry_push: RepairPush | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "workflow_id": self.workflow_id,
            "state": self.state.value,
            "repair_branch": self.repair_branch,
        }
        if self.repair_base_branch is not None:
            data["repair_base_branch"] = self.repair_base_branch
        if self.repair_base_branch is not None:
            data["repair_base_branch"] = self.repair_base_branch

        if self.summary is not None:
            data["summary"] = {
                "branch": self.summary.branch,
                "changes": list(self.summary.changes),
            }

        if self.validation is not None:
            data["validation"] = {
                "command": list(self.validation.command),
                "status": self.validation.status,
                "exit_code": self.validation.exit_code,
                "passed_count": self.validation.passed_count,
                "failed_count": self.validation.failed_count,
                "error_count": self.validation.error_count,
                "output": self.validation.output,
            }
        if self.commit is not None:
            data["commit"] = {
                "sha": self.commit.sha,
                "message": self.commit.message,
            }
        if self.push is not None:
            data["push"] = {
                "remote": self.push.remote,
                "branch": self.push.branch,
                "commit_sha": self.push.commit_sha,
            }
        if self.pull_request is not None:
            data["pull_request"] = {
                "repository": self.pull_request.repository,
                "number": self.pull_request.number,
                "url": self.pull_request.url,
                "title": self.pull_request.title,
                "base_branch": self.pull_request.base_branch,
                "head_branch": self.pull_request.head_branch,
                "commit_sha": self.pull_request.commit_sha,
            }
        if self.ci_status is not None:
            data["ci_status"] = {
                "repository": self.ci_status.repository,
                "commit_sha": self.ci_status.commit_sha,
                "state": self.ci_status.state,
                "check_runs": [
                    {
                        "name": check_run.name,
                        "status": check_run.status,
                        "conclusion": check_run.conclusion,
                        "details_url": check_run.details_url,
                    }
                    for check_run in self.ci_status.check_runs
                ],
            }
        if self.ci_retry_summary is not None:
            data["ci_retry_summary"] = {
                "branch": self.ci_retry_summary.branch,
                "changes": list(self.ci_retry_summary.changes),
            }
        if self.ci_retry_validation is not None:
            data["ci_retry_validation"] = {
                "command": list(self.ci_retry_validation.command),
                "status": self.ci_retry_validation.status,
                "exit_code": self.ci_retry_validation.exit_code,
                "passed_count": self.ci_retry_validation.passed_count,
                "failed_count": self.ci_retry_validation.failed_count,
                "error_count": self.ci_retry_validation.error_count,
                "output": self.ci_retry_validation.output,
            }
        if self.ci_retry_commit is not None:
            data["ci_retry_commit"] = {
                "sha": self.ci_retry_commit.sha,
                "message": self.ci_retry_commit.message,
            }
        if self.ci_retry_push is not None:
            data["ci_retry_push"] = {
                "remote": self.ci_retry_push.remote,
                "branch": self.ci_retry_push.branch,
                "commit_sha": self.ci_retry_push.commit_sha,
            }
        return data
    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> Self:
        summary_data = data.get("summary")
        summary = None

        if isinstance(summary_data, dict):
            summary = PostRepairSummary(
                branch=summary_data["branch"],
                changes=tuple(summary_data["changes"]),
            )

        validation_data = data.get("validation")
        validation = None

        if isinstance(validation_data, dict):
            validation = PostRepairValidation(
                command=tuple(validation_data["command"]),
                status=validation_data["status"],
                exit_code=validation_data["exit_code"],
                passed_count=validation_data["passed_count"],
                failed_count=validation_data["failed_count"],
                error_count=validation_data["error_count"],
                output=validation_data["output"],
            )
        commit_data = data.get("commit")
        commit = None

        if isinstance(commit_data, dict):
            commit = RepairCommit(
                sha=commit_data["sha"],
                message=commit_data["message"],
            )
        push_data = data.get("push")
        push = None

        if isinstance(push_data, dict):
            push = RepairPush(
                remote=push_data["remote"],
                branch=push_data["branch"],
                commit_sha=push_data["commit_sha"],
            )
        pull_request_data = data.get("pull_request")
        pull_request = None

        if isinstance(pull_request_data, dict):
            pull_request = RepairPullRequest(
                repository=pull_request_data["repository"],
                number=pull_request_data["number"],
                url=pull_request_data["url"],
                title=pull_request_data["title"],
                base_branch=pull_request_data["base_branch"],
                head_branch=pull_request_data["head_branch"],
                commit_sha=pull_request_data["commit_sha"],
            )

        ci_status_data = data.get("ci_status")
        ci_status = None

        if isinstance(ci_status_data, dict):
            check_runs = tuple(
                RepairCheckRun(
                    name=check_run_data["name"],
                    status=check_run_data["status"],
                    conclusion=check_run_data["conclusion"],
                    details_url=check_run_data["details_url"],
                )
                for check_run_data in ci_status_data["check_runs"]
            )

            ci_status = RepairCIStatus(
                repository=ci_status_data["repository"],
                commit_sha=ci_status_data["commit_sha"],
                state=ci_status_data["state"],
                check_runs=check_runs,
            )

        ci_retry_summary_data = data.get("ci_retry_summary")
        ci_retry_summary = None

        if isinstance(ci_retry_summary_data, dict):
            ci_retry_summary = PostRepairSummary(
                branch=ci_retry_summary_data["branch"],
                changes=tuple(ci_retry_summary_data["changes"]),
            )
        ci_retry_validation_data = data.get("ci_retry_validation")
        ci_retry_validation = None

        if isinstance(ci_retry_validation_data, dict):
            ci_retry_validation = PostRepairValidation(
                command=tuple(ci_retry_validation_data["command"]),
                status=ci_retry_validation_data["status"],
                exit_code=ci_retry_validation_data["exit_code"],
                passed_count=ci_retry_validation_data["passed_count"],
                failed_count=ci_retry_validation_data["failed_count"],
                error_count=ci_retry_validation_data["error_count"],
                output=ci_retry_validation_data["output"],
            )
        ci_retry_commit_data = data.get("ci_retry_commit")
        ci_retry_commit = None

        if isinstance(ci_retry_commit_data, dict):
            ci_retry_commit = RepairCommit(
                sha=ci_retry_commit_data["sha"],
                message=ci_retry_commit_data["message"],
            )
        ci_retry_push_data = data.get("ci_retry_push")
        ci_retry_push = None

        if isinstance(ci_retry_push_data, dict):
            ci_retry_push = RepairPush(
                remote=ci_retry_push_data["remote"],
                branch=ci_retry_push_data["branch"],
                commit_sha=ci_retry_push_data["commit_sha"],
            )
        return cls(
            workflow_id=data["workflow_id"],
            state=WorkflowState(data["state"]),
            repair_branch=data["repair_branch"],
            repair_base_branch=data.get("repair_base_branch"),
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
        )


def save_workflow_checkpoint(
    checkpoint: WorkflowCheckpoint,
    path: Path,
) -> None:
    """Persist a workflow checkpoint as JSON."""
    path.write_text(
        json.dumps(checkpoint.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )


def load_workflow_checkpoint(
    path: Path,
) -> WorkflowCheckpoint:
    """Load a workflow checkpoint from JSON."""
    data = json.loads(
        path.read_text(encoding="utf-8")
    )
    return WorkflowCheckpoint.from_dict(data)


@dataclass(frozen=True)
class IssueWorkflowResult:
    """Structured result from one DevPilot Issue workflow run."""
    state: WorkflowState
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
    repair_base_branch: str | None = None,
    checkpoint: WorkflowCheckpoint | None = None,
    workflow_id: str | None = None,
    save_checkpoint: (
    Callable[[WorkflowCheckpoint], None] | None
    ) = None,

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
    if checkpoint is None:
        machine = WorkflowStateMachine()
        run_agent(issue_prompt)
    else:
        machine = WorkflowStateMachine.from_checkpoint(checkpoint)

        if machine.state not in {
            WorkflowState.VALIDATE,
            WorkflowState.COMMIT,
            WorkflowState.PUSH,
            WorkflowState.CREATE_PR,
            WorkflowState.WAIT_CI,
            WorkflowState.CI_REPAIR,
            WorkflowState.CI_RETRY_SUMMARY,
            WorkflowState.CI_RETRY_VALIDATE,
            WorkflowState.CI_RETRY_COMMIT,
            WorkflowState.CI_RETRY_PUSH,
            WorkflowState.WAIT_RETRY_CI,
        }:
            raise ValueError(
                "only VALIDATE, COMMIT, PUSH, CREATE_PR, WAIT_CI, "
                "CI_REPAIR, CI_RETRY_SUMMARY, CI_RETRY_VALIDATE, "
                "CI_RETRY_COMMIT, or CI_RETRY_PUSH checkpoint resume "
                "is currently supported"
            )

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
    if (
        checkpoint is not None
        and checkpoint.repair_branch != repair_branch
    ):
        raise ValueError(
            "checkpoint repair branch does not match workflow"
        )

    if checkpoint is not None:
        if checkpoint.summary is None:
            raise ValueError(
                "summary is required to resume from "
                f"{machine.state.value.upper()}"
            )

        summary = checkpoint.summary
    else:
        if collect_summary is None:
            raise ValueError(
                "collect_summary is required for repair workflow"
            )

        machine.transition(WorkflowState.COLLECT_SUMMARY)

        summary = collect_summary(repair_branch)

        if not summary.has_changes:
            return IssueWorkflowResult(
                state=machine.state,
                summary=summary,
                validation=None,
                commit=None,
                push=None,
                pull_request=None,
            )

    if machine.state in {
        WorkflowState.COMMIT,
        WorkflowState.PUSH,
        WorkflowState.CREATE_PR,
        WorkflowState.WAIT_CI,
        WorkflowState.CI_REPAIR,
        WorkflowState.CI_RETRY_SUMMARY,
        WorkflowState.CI_RETRY_VALIDATE,
        WorkflowState.CI_RETRY_COMMIT,
        WorkflowState.CI_RETRY_PUSH,
        WorkflowState.WAIT_RETRY_CI,

    }:
        if checkpoint is None or checkpoint.validation is None:
            raise ValueError(
                "validation is required to resume from "
                f"{machine.state.value.upper()}"
            )

        validation = checkpoint.validation
    else:
        if run_validation is None:
            raise ValueError(
                "run_validation is required for repair changes"
            )

        if machine.state == WorkflowState.COLLECT_SUMMARY:
            machine.transition(WorkflowState.VALIDATE)

            if save_checkpoint is not None:
                if workflow_id is None:
                    raise ValueError(
                        "workflow_id is required when save_checkpoint is provided"
                    )

                save_checkpoint(
                    WorkflowCheckpoint(
                        workflow_id=workflow_id,
                        state=machine.state,
                        repair_branch=repair_branch,
                        repair_base_branch=repair_base_branch,
                        summary=summary,
                    )
                )

        validation = run_validation()

        if not validation.passed:
            return IssueWorkflowResult(
                state=machine.state,
                summary=summary,
                validation=validation,
                commit=None,
                push=None,
                pull_request=None,
            )
    if machine.state in {
        WorkflowState.PUSH,
        WorkflowState.CREATE_PR,
        WorkflowState.WAIT_CI,
        WorkflowState.CI_REPAIR,
        WorkflowState.CI_RETRY_SUMMARY,
        WorkflowState.CI_RETRY_VALIDATE,
        WorkflowState.CI_RETRY_COMMIT,
        WorkflowState.CI_RETRY_PUSH,
        WorkflowState.WAIT_RETRY_CI,
        }:
        if checkpoint is None or checkpoint.commit is None:
            raise ValueError(
                "commit is required to resume from "
                f"{machine.state.value.upper()}"
            )

        commit = checkpoint.commit
    else:
        if create_commit is None:
            raise ValueError(
                "create_commit is required after passed validation"
            )

        if machine.state == WorkflowState.VALIDATE:
            machine.transition(WorkflowState.COMMIT)

            if save_checkpoint is not None:
                if workflow_id is None:
                    raise ValueError(
                        "workflow_id is required when save_checkpoint is provided"
                    )

                save_checkpoint(
                    WorkflowCheckpoint(
                        workflow_id=workflow_id,
                        state=machine.state,
                        repair_branch=repair_branch,
                        repair_base_branch=repair_base_branch,
                        summary=summary,
                        validation=validation,
                    )
                )

        commit = create_commit()

    if machine.state in {
        WorkflowState.CREATE_PR,
        WorkflowState.WAIT_CI,
        WorkflowState.CI_REPAIR,
        WorkflowState.CI_RETRY_SUMMARY,
        WorkflowState.CI_RETRY_VALIDATE,
        WorkflowState.CI_RETRY_COMMIT,
        WorkflowState.CI_RETRY_PUSH,
        WorkflowState.WAIT_RETRY_CI,
    }:
        if checkpoint is None or checkpoint.push is None:
            raise ValueError(
                "push is required to resume from "
                f"{machine.state.value.upper()}"
            )

        push = checkpoint.push
    else:
        if push_commit is None:
            raise ValueError(
                "push_commit is required after commit creation"
            )

        if machine.state == WorkflowState.COMMIT:
            machine.transition(WorkflowState.PUSH)

            if save_checkpoint is not None:
                if workflow_id is None:
                    raise ValueError(
                        "workflow_id is required when save_checkpoint is provided"
                    )

                save_checkpoint(
                    WorkflowCheckpoint(
                        workflow_id=workflow_id,
                        state=machine.state,
                        repair_branch=repair_branch,
                        repair_base_branch=repair_base_branch,
                        summary=summary,
                        validation=validation,
                        commit=commit,
                    )
                )

        push = push_commit(
            repair_branch,
            commit.sha,
        )

    if machine.state in {
        WorkflowState.WAIT_CI,
        WorkflowState.CI_REPAIR,
        WorkflowState.CI_RETRY_SUMMARY,
        WorkflowState.CI_RETRY_VALIDATE,
        WorkflowState.CI_RETRY_COMMIT,
        WorkflowState.CI_RETRY_PUSH,
        WorkflowState.WAIT_RETRY_CI,
    }:
        if checkpoint is None or checkpoint.pull_request is None:
            raise ValueError(
                "pull request is required to resume from "
                f"{machine.state.value.upper()}"
            )

        pull_request = checkpoint.pull_request
    else:
        if create_pull_request is None:
            raise ValueError(
                "create_pull_request is required after push"
            )

        if machine.state == WorkflowState.PUSH:
            machine.transition(WorkflowState.CREATE_PR)

            if save_checkpoint is not None:
                if workflow_id is None:
                    raise ValueError(
                        "workflow_id is required when save_checkpoint is provided"
                    )

                save_checkpoint(
                    WorkflowCheckpoint(
                        workflow_id=workflow_id,
                        state=machine.state,
                        repair_branch=repair_branch,
                        repair_base_branch=repair_base_branch,
                        summary=summary,
                        validation=validation,
                        commit=commit,
                        push=push,
                    )
                )

        pull_request = create_pull_request(push)

    if machine.state in {
        WorkflowState.CI_REPAIR,
        WorkflowState.CI_RETRY_SUMMARY,
        WorkflowState.CI_RETRY_VALIDATE,
        WorkflowState.CI_RETRY_COMMIT,
        WorkflowState.CI_RETRY_PUSH,
        WorkflowState.WAIT_RETRY_CI,
    }:
        if checkpoint is None or checkpoint.ci_status is None:
            raise ValueError(
                "CI status is required to resume from "
                f"{machine.state.value.upper()}"
            )

        ci_status = checkpoint.ci_status
    else:
        if wait_for_ci is None:
            raise ValueError(
                "wait_for_ci is required after pull request creation"
            )

        if machine.state == WorkflowState.CREATE_PR:
            machine.transition(WorkflowState.WAIT_CI)

            if save_checkpoint is not None:
                if workflow_id is None:
                    raise ValueError(
                        "workflow_id is required when save_checkpoint is provided"
                    )

                save_checkpoint(
                    WorkflowCheckpoint(
                        workflow_id=workflow_id,
                        state=machine.state,
                        repair_branch=repair_branch,
                        repair_base_branch=repair_base_branch,
                        summary=summary,
                        validation=validation,
                        commit=commit,
                        push=push,
                        pull_request=pull_request,
                    )
                )

        ci_status = wait_for_ci(pull_request)

    ci_retry_summary = None
    ci_retry_push = None
    if ci_status.state == "failure":
        if machine.state == WorkflowState.WAIT_CI:
            machine.transition(WorkflowState.CI_REPAIR)

            if save_checkpoint is not None:
                if workflow_id is None:
                    raise ValueError(
                        "workflow_id is required when save_checkpoint is provided"
                    )

                save_checkpoint(
                    WorkflowCheckpoint(
                        workflow_id=workflow_id,
                        state=machine.state,
                        repair_branch=repair_branch,
                        repair_base_branch=repair_base_branch,
                        summary=summary,
                        validation=validation,
                        commit=commit,
                        push=push,
                        pull_request=pull_request,
                        ci_status=ci_status,
                    )
                )

        if machine.state == WorkflowState.CI_REPAIR:
            if build_ci_failure_prompt is None:
                raise ValueError(
                    "build_ci_failure_prompt is required "
                    "after CI failure"
                )

            ci_failure_prompt = build_ci_failure_prompt(
                ci_status
            )
            run_agent(ci_failure_prompt)
            machine.transition(
                WorkflowState.CI_RETRY_SUMMARY
            )

            if save_checkpoint is not None:
                if workflow_id is None:
                    raise ValueError(
                        "workflow_id is required when save_checkpoint is provided"
                    )

                save_checkpoint(
                    WorkflowCheckpoint(
                        workflow_id=workflow_id,
                        state=machine.state,
                        repair_branch=repair_branch,
                        repair_base_branch=repair_base_branch,
                        summary=summary,
                        validation=validation,
                        commit=commit,
                        push=push,
                        pull_request=pull_request,
                        ci_status=ci_status,
                    )
                )

        if machine.state in {
            WorkflowState.CI_RETRY_VALIDATE,
            WorkflowState.CI_RETRY_COMMIT,
            WorkflowState.CI_RETRY_PUSH,
            WorkflowState.WAIT_RETRY_CI,
        }:
            if (
                checkpoint is None
                or checkpoint.ci_retry_summary is None
            ):
                raise ValueError(
                    "CI retry summary is required to resume from "
                    f"{machine.state.value.upper()}"
                )

            ci_retry_summary = checkpoint.ci_retry_summary
        else:
            ci_retry_summary = collect_summary(
                repair_branch
            )

            if not ci_retry_summary.has_changes:
                raise ValueError(
                    "CI repair produced no repository changes"
                )

            machine.transition(
                WorkflowState.CI_RETRY_VALIDATE
            )

            if save_checkpoint is not None:
                if workflow_id is None:
                    raise ValueError(
                        "workflow_id is required when save_checkpoint is provided"
                    )

                save_checkpoint(
                    WorkflowCheckpoint(
                        workflow_id=workflow_id,
                        state=machine.state,
                        repair_branch=repair_branch,
                        repair_base_branch=repair_base_branch,
                        summary=summary,
                        validation=validation,
                        commit=commit,
                        push=push,
                        pull_request=pull_request,
                        ci_status=ci_status,
                        ci_retry_summary=ci_retry_summary,
                    )
                )

        if machine.state in {
            WorkflowState.CI_RETRY_COMMIT,
            WorkflowState.CI_RETRY_PUSH,
            WorkflowState.WAIT_RETRY_CI,
        }:
            if (
                checkpoint is None
                or checkpoint.ci_retry_validation is None
            ):
                raise ValueError(
                    "CI retry validation is required to resume from "
                    f"{machine.state.value.upper()}"
                )

            ci_retry_validation = checkpoint.ci_retry_validation
        else:
            ci_retry_validation = run_validation()

            if not ci_retry_validation.passed:
                raise ValueError(
                    "CI repair validation failed"
                )
            machine.transition(
                WorkflowState.CI_RETRY_COMMIT
            )

            if save_checkpoint is not None:
                if workflow_id is None:
                    raise ValueError(
                        "workflow_id is required when save_checkpoint is provided"
                    )

                save_checkpoint(
                    WorkflowCheckpoint(
                        workflow_id=workflow_id,
                        state=machine.state,
                        repair_branch=repair_branch,
                        repair_base_branch=repair_base_branch,
                        summary=summary,
                        validation=validation,
                        commit=commit,
                        push=push,
                        pull_request=pull_request,
                        ci_status=ci_status,
                        ci_retry_summary=ci_retry_summary,
                        ci_retry_validation=ci_retry_validation,
                    )
                )
        if machine.state in {
            WorkflowState.CI_RETRY_PUSH,
            WorkflowState.WAIT_RETRY_CI,
        }:
            if (
                checkpoint is None
                or checkpoint.ci_retry_commit is None
            ):
                raise ValueError(
                    "CI retry commit is required to resume from "
                    f"{machine.state.value.upper()}"
                )

            ci_retry_commit = checkpoint.ci_retry_commit
        else:
            ci_retry_commit = create_commit()
            machine.transition(
                WorkflowState.CI_RETRY_PUSH
            )

            if save_checkpoint is not None:
                if workflow_id is None:
                    raise ValueError(
                        "workflow_id is required when save_checkpoint is provided"
                    )

                save_checkpoint(
                    WorkflowCheckpoint(
                        workflow_id=workflow_id,
                        state=machine.state,
                        repair_branch=repair_branch,
                        repair_base_branch=repair_base_branch,
                        summary=summary,
                        validation=validation,
                        commit=commit,
                        push=push,
                        pull_request=pull_request,
                        ci_status=ci_status,
                        ci_retry_summary=ci_retry_summary,
                        ci_retry_validation=ci_retry_validation,
                        ci_retry_commit=ci_retry_commit,
                    )
                )


        if machine.state == WorkflowState.WAIT_RETRY_CI:
            if (
                checkpoint is None
                or checkpoint.ci_retry_push is None
            ):
                raise ValueError(
                    "CI retry push is required to resume from "
                    "WAIT_RETRY_CI"
                )

            ci_retry_push = checkpoint.ci_retry_push
        else:
            ci_retry_push = push_commit(
                repair_branch,
                ci_retry_commit.sha,
            )

        if wait_for_retry_ci is None:
            raise ValueError(
                "wait_for_retry_ci is required after CI retry push"
            )
        if machine.state == WorkflowState.CI_RETRY_PUSH:
            machine.transition(WorkflowState.WAIT_RETRY_CI)

            if save_checkpoint is not None:
                if workflow_id is None:
                    raise ValueError(
                        "workflow_id is required when save_checkpoint is provided"
                    )

                save_checkpoint(
                    WorkflowCheckpoint(
                        workflow_id=workflow_id,
                        state=machine.state,
                        repair_branch=repair_branch,
                        repair_base_branch=repair_base_branch,
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
                    )
                )

        ci_retry_status = wait_for_retry_ci(
            ci_retry_push
        )

        if ci_retry_status.state == "failure":
            raise ValueError(
                "CI retry failed"
            )
        machine.transition(WorkflowState.COMPLETED)
    else:
        machine.transition(WorkflowState.COMPLETED)

    return IssueWorkflowResult(
        state=machine.state,
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
