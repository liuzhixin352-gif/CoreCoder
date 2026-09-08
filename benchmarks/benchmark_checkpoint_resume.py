"""Deterministic checkpoint/resume reliability benchmark.

Exercises every currently supported resume state through the real
run_issue_workflow orchestration function. Each scenario:

1. serializes a checkpoint to JSON and loads it back,
2. resumes the workflow from that state,
3. persists every later checkpoint transition through the same JSON path,
4. requires the workflow to reach COMPLETED, and
5. verifies exact callback counts so already-completed side effects are not
   executed again.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

from corecoder.issue_orchestration import (
    WorkflowCheckpoint,
    WorkflowState,
    load_workflow_checkpoint,
    run_issue_workflow,
    save_workflow_checkpoint,
)
from corecoder.post_repair import PostRepairSummary
from corecoder.post_repair_validation import PostRepairValidation
from corecoder.repair_ci import RepairCIStatus
from corecoder.repair_commit import RepairCommit
from corecoder.repair_pr import RepairPullRequest
from corecoder.repair_push import RepairPush


BRANCH = "devpilot/benchmark-resume"
BASE_BRANCH = "main"
REPOSITORY = "example/project"

MAIN_SUMMARY = PostRepairSummary(
    branch=BRANCH,
    changes=(" M corecoder/example.py",),
)
RETRY_SUMMARY = PostRepairSummary(
    branch=BRANCH,
    changes=(" M corecoder/ci_fix.py",),
)

MAIN_VALIDATION = PostRepairValidation(
    command=("python", "-m", "pytest", "-q"),
    status="passed",
    exit_code=0,
    passed_count=1,
    failed_count=0,
    error_count=0,
    output="1 passed",
)
RETRY_VALIDATION = PostRepairValidation(
    command=("python", "-m", "pytest", "-q"),
    status="passed",
    exit_code=0,
    passed_count=1,
    failed_count=0,
    error_count=0,
    output="1 passed",
)

MAIN_COMMIT = RepairCommit(
    sha="main-commit",
    message="Benchmark main commit",
)
RETRY_COMMIT = RepairCommit(
    sha="retry-commit",
    message="Benchmark retry commit",
)

MAIN_PUSH = RepairPush(
    remote="origin",
    branch=BRANCH,
    commit_sha=MAIN_COMMIT.sha,
)
RETRY_PUSH = RepairPush(
    remote="origin",
    branch=BRANCH,
    commit_sha=RETRY_COMMIT.sha,
)

PULL_REQUEST = RepairPullRequest(
    repository=REPOSITORY,
    number=1,
    url="https://example.invalid/pr/1",
    title="Benchmark PR",
    base_branch=BASE_BRANCH,
    head_branch=BRANCH,
    commit_sha=MAIN_COMMIT.sha,
)

CI_FAILURE = RepairCIStatus(
    repository=REPOSITORY,
    commit_sha=MAIN_COMMIT.sha,
    state="failure",
    check_runs=(),
)
CI_SUCCESS = RepairCIStatus(
    repository=REPOSITORY,
    commit_sha=MAIN_COMMIT.sha,
    state="success",
    check_runs=(),
)
RETRY_CI_SUCCESS = RepairCIStatus(
    repository=REPOSITORY,
    commit_sha=RETRY_COMMIT.sha,
    state="success",
    check_runs=(),
)


STATES = (
    WorkflowState.VALIDATE,
    WorkflowState.WAIT_COMMIT_APPROVAL,
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
)

RETRY_STATES = frozenset(
    {
        WorkflowState.CI_REPAIR,
        WorkflowState.CI_RETRY_SUMMARY,
        WorkflowState.CI_RETRY_VALIDATE,
        WorkflowState.CI_RETRY_COMMIT,
        WorkflowState.CI_RETRY_PUSH,
        WorkflowState.WAIT_RETRY_CI,
    }
)

# Exact callback contract for each resume state.
# A zero count is important: it proves an already-completed side effect is
# not executed again after resume.
EXPECTED_CALLS = {
    WorkflowState.VALIDATE: {
        "run_validation": 1,
        "approval": 1,
        "create_commit": 1,
        "push_commit": 1,
        "create_pr": 1,
        "wait_for_ci": 1,
    },
    WorkflowState.WAIT_COMMIT_APPROVAL: {
        "approval": 1,
        "create_commit": 1,
        "push_commit": 1,
        "create_pr": 1,
        "wait_for_ci": 1,
    },
    WorkflowState.COMMIT: {
        "create_commit": 1,
        "push_commit": 1,
        "create_pr": 1,
        "wait_for_ci": 1,
    },
    WorkflowState.PUSH: {
        "push_commit": 1,
        "create_pr": 1,
        "wait_for_ci": 1,
    },
    WorkflowState.CREATE_PR: {
        "create_pr": 1,
        "wait_for_ci": 1,
    },
    WorkflowState.WAIT_CI: {
        "wait_for_ci": 1,
    },
    WorkflowState.CI_REPAIR: {
        "build_ci_prompt": 1,
        "run_agent": 1,
        "collect_summary": 1,
        "run_validation": 1,
        "create_commit": 1,
        "push_commit": 1,
        "wait_for_retry_ci": 1,
    },
    WorkflowState.CI_RETRY_SUMMARY: {
        "collect_summary": 1,
        "run_validation": 1,
        "create_commit": 1,
        "push_commit": 1,
        "wait_for_retry_ci": 1,
    },
    WorkflowState.CI_RETRY_VALIDATE: {
        "run_validation": 1,
        "create_commit": 1,
        "push_commit": 1,
        "wait_for_retry_ci": 1,
    },
    WorkflowState.CI_RETRY_COMMIT: {
        "create_commit": 1,
        "push_commit": 1,
        "wait_for_retry_ci": 1,
    },
    WorkflowState.CI_RETRY_PUSH: {
        "push_commit": 1,
        "wait_for_retry_ci": 1,
    },
    WorkflowState.WAIT_RETRY_CI: {
        "wait_for_retry_ci": 1,
    },
}

CALL_NAMES = (
    "run_agent",
    "collect_summary",
    "run_validation",
    "approval",
    "create_commit",
    "push_commit",
    "create_pr",
    "wait_for_ci",
    "build_ci_prompt",
    "wait_for_retry_ci",
)


def build_checkpoint(state: WorkflowState) -> WorkflowCheckpoint:
    """Build the minimum valid durable artifacts for one resume state."""
    kwargs: dict[str, object] = {
        "workflow_id": f"benchmark-{state.value}",
        "state": state,
        "repair_branch": BRANCH,
        "repair_base_branch": BASE_BRANCH,
        "summary": MAIN_SUMMARY,
    }

    if state in {
        WorkflowState.WAIT_COMMIT_APPROVAL,
        WorkflowState.COMMIT,
        WorkflowState.PUSH,
        WorkflowState.CREATE_PR,
        WorkflowState.WAIT_CI,
        *RETRY_STATES,
    }:
        kwargs["validation"] = MAIN_VALIDATION

    if state in {
        WorkflowState.PUSH,
        WorkflowState.CREATE_PR,
        WorkflowState.WAIT_CI,
        *RETRY_STATES,
    }:
        kwargs["commit"] = MAIN_COMMIT

    if state in {
        WorkflowState.CREATE_PR,
        WorkflowState.WAIT_CI,
        *RETRY_STATES,
    }:
        kwargs["push"] = MAIN_PUSH

    if state in {
        WorkflowState.WAIT_CI,
        *RETRY_STATES,
    }:
        kwargs["pull_request"] = PULL_REQUEST

    if state in RETRY_STATES:
        kwargs["ci_status"] = CI_FAILURE

    if state in {
        WorkflowState.CI_RETRY_VALIDATE,
        WorkflowState.CI_RETRY_COMMIT,
        WorkflowState.CI_RETRY_PUSH,
        WorkflowState.WAIT_RETRY_CI,
    }:
        kwargs["ci_retry_summary"] = RETRY_SUMMARY

    if state in {
        WorkflowState.CI_RETRY_COMMIT,
        WorkflowState.CI_RETRY_PUSH,
        WorkflowState.WAIT_RETRY_CI,
    }:
        kwargs["ci_retry_validation"] = RETRY_VALIDATION

    if state in {
        WorkflowState.CI_RETRY_PUSH,
        WorkflowState.WAIT_RETRY_CI,
    }:
        kwargs["ci_retry_commit"] = RETRY_COMMIT

    if state is WorkflowState.WAIT_RETRY_CI:
        kwargs["ci_retry_push"] = RETRY_PUSH

    return WorkflowCheckpoint(**kwargs)


def normalized_calls(calls: Counter[str]) -> dict[str, int]:
    """Return all tracked callbacks, including explicit zero counts."""
    return {name: calls[name] for name in CALL_NAMES}


def expected_calls(state: WorkflowState) -> dict[str, int]:
    expected = Counter(EXPECTED_CALLS[state])
    return {name: expected[name] for name in CALL_NAMES}


def run_case(state: WorkflowState) -> tuple[bool, str, Counter[str], int]:
    """Run one persisted resume scenario."""
    calls: Counter[str] = Counter()
    checkpoint_roundtrips = 0

    checkpoint = build_checkpoint(state)

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_path = Path(tmp) / "workflow.json"

        save_workflow_checkpoint(checkpoint, checkpoint_path)
        restored = load_workflow_checkpoint(checkpoint_path)
        checkpoint_roundtrips += 1

        if restored != checkpoint:
            return (
                False,
                "initial checkpoint JSON round-trip mismatch",
                calls,
                checkpoint_roundtrips,
            )

        def persist(next_checkpoint: WorkflowCheckpoint) -> None:
            nonlocal checkpoint_roundtrips
            save_workflow_checkpoint(next_checkpoint, checkpoint_path)
            loaded = load_workflow_checkpoint(checkpoint_path)
            checkpoint_roundtrips += 1
            if loaded != next_checkpoint:
                raise AssertionError(
                    "checkpoint JSON round-trip mismatch during resume"
                )

        def run_agent(prompt: str) -> None:
            calls["run_agent"] += 1

        def collect_summary(branch: str) -> PostRepairSummary:
            calls["collect_summary"] += 1
            return RETRY_SUMMARY if state in RETRY_STATES else MAIN_SUMMARY

        def run_validation() -> PostRepairValidation:
            calls["run_validation"] += 1
            return (
                RETRY_VALIDATION
                if state in RETRY_STATES
                else MAIN_VALIDATION
            )

        def approval(
            summary: PostRepairSummary,
            validation: PostRepairValidation,
        ) -> bool:
            calls["approval"] += 1
            return True

        def create_commit() -> RepairCommit:
            calls["create_commit"] += 1
            return RETRY_COMMIT if state in RETRY_STATES else MAIN_COMMIT

        def push_commit(branch: str, sha: str) -> RepairPush:
            calls["push_commit"] += 1
            return RETRY_PUSH if sha == RETRY_COMMIT.sha else MAIN_PUSH

        def create_pr(push: RepairPush) -> RepairPullRequest:
            calls["create_pr"] += 1
            return PULL_REQUEST

        def wait_for_ci(pr: RepairPullRequest) -> RepairCIStatus:
            calls["wait_for_ci"] += 1
            return CI_SUCCESS

        def build_ci_prompt(ci_status: RepairCIStatus) -> str:
            calls["build_ci_prompt"] += 1
            return "Repair the deterministic CI failure."

        def wait_for_retry_ci(push: RepairPush) -> RepairCIStatus:
            calls["wait_for_retry_ci"] += 1
            return RETRY_CI_SUCCESS

        result = run_issue_workflow(
            issue_prompt="Benchmark repair.",
            dry_run=False,
            run_agent=run_agent,
            repair_branch=BRANCH,
            repair_base_branch=BASE_BRANCH,
            checkpoint=restored,
            workflow_id=restored.workflow_id,
            save_checkpoint=persist,
            collect_summary=collect_summary,
            run_validation=run_validation,
            request_commit_approval=approval,
            create_commit=create_commit,
            push_commit=push_commit,
            create_pull_request=create_pr,
            wait_for_ci=wait_for_ci,
            build_ci_failure_prompt=build_ci_prompt,
            wait_for_retry_ci=wait_for_retry_ci,
        )

    if result is None or result.state is not WorkflowState.COMPLETED:
        return (
            False,
            "resume did not reach COMPLETED",
            calls,
            checkpoint_roundtrips,
        )

    actual = normalized_calls(calls)
    expected = expected_calls(state)

    if actual != expected:
        differences = [
            f"{name}: expected={expected[name]} actual={actual[name]}"
            for name in CALL_NAMES
            if expected[name] != actual[name]
        ]
        return (
            False,
            "callback contract mismatch: " + "; ".join(differences),
            calls,
            checkpoint_roundtrips,
        )

    return True, "ok", calls, checkpoint_roundtrips


def compact_calls(calls: Counter[str]) -> str:
    active = [
        f"{name}={calls[name]}"
        for name in CALL_NAMES
        if calls[name]
    ]
    return ", ".join(active) if active else "none"


def main() -> int:
    print("Checkpoint / Resume Reliability Benchmark")
    print("=========================================")
    print(f"scenarios: {len(STATES)}")
    print(
        "protocol: JSON checkpoint round-trip + real workflow resume "
        "+ exact side-effect callback contract"
    )
    print()

    passed = 0
    total_roundtrips = 0

    for state in STATES:
        try:
            success, detail, calls, roundtrips = run_case(state)
        except Exception as exc:
            success = False
            detail = f"{type(exc).__name__}: {exc}"
            calls = Counter()
            roundtrips = 0

        total_roundtrips += roundtrips

        if success:
            passed += 1

        print(
            f"{state.value:<22} "
            f"{'PASS' if success else 'FAIL'} "
            f"roundtrips={roundtrips:<2} "
            f"calls=[{compact_calls(calls)}]"
        )
        if not success:
            print(f"    {detail}")

    total = len(STATES)

    print()
    print("Summary")
    print("-------")
    print(f"completed_contracts: {passed}/{total}")
    print(f"resume_rate:         {passed / total * 100:.1f}%")
    print(f"checkpoint_roundtrips: {total_roundtrips}")
    print(
        "duplicate_side_effect_contract: "
        f"{'PASS' if passed == total else 'FAIL'}"
    )

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
