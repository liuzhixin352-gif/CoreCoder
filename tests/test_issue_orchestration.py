"""Tests for DevPilot Issue workflow orchestration."""

import json
import pytest
from corecoder import issue_orchestration
from corecoder.post_repair import PostRepairSummary
from corecoder.post_repair_validation import PostRepairValidation
from corecoder.repair_commit import RepairCommit
from corecoder.repair_push import RepairPush
from corecoder.repair_pr import RepairPullRequest
from corecoder.repair_ci import RepairCIStatus


def test_workflow_state_has_stable_string_values():
    assert issue_orchestration.WorkflowState.REPAIR.value == "repair"
    assert (
        issue_orchestration.WorkflowState.COLLECT_SUMMARY.value
        == "collect_summary"
    )
    assert issue_orchestration.WorkflowState.VALIDATE.value == "validate"
    assert (
    issue_orchestration.WorkflowState.CI_RETRY_SUMMARY.value
    == "ci_retry_summary"
    )
    assert issue_orchestration.WorkflowState.COMPLETED.value == "completed"


def test_workflow_state_machine_starts_at_repair_and_transitions():
    machine = issue_orchestration.WorkflowStateMachine()

    assert machine.state == issue_orchestration.WorkflowState.REPAIR

    machine.transition(
        issue_orchestration.WorkflowState.COLLECT_SUMMARY
    )

    assert (
        machine.state
        == issue_orchestration.WorkflowState.COLLECT_SUMMARY
    )


def test_workflow_state_machine_rejects_invalid_transition():
    machine = issue_orchestration.WorkflowStateMachine()

    with pytest.raises(
        ValueError,
        match="Invalid workflow transition",
    ):
        machine.transition(
            issue_orchestration.WorkflowState.COMMIT
        )


def test_workflow_checkpoint_tracks_identity_state_and_branch():
    checkpoint = issue_orchestration.WorkflowCheckpoint(
    workflow_id="issue-21",
    state=issue_orchestration.WorkflowState.VALIDATE,
    repair_branch="devpilot/issue-21-fix-scan-limit",
    repair_base_branch="devpilot-v1",
    )

    assert checkpoint.workflow_id == "issue-21"
    assert checkpoint.state == issue_orchestration.WorkflowState.VALIDATE
    assert (
        checkpoint.repair_branch
        == "devpilot/issue-21-fix-scan-limit"
    )
    assert checkpoint.repair_base_branch == "devpilot-v1"


def test_workflow_checkpoint_serializes_to_dict():
    summary = PostRepairSummary(
    branch="devpilot/issue-21-fix-scan-limit",
    changes=(" M corecoder/example.py",),
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.VALIDATE,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        repair_base_branch="devpilot-v1",
        summary=summary,

    )

    assert checkpoint.to_dict() == {
    "workflow_id": "issue-21",
    "state": "validate",
    "repair_branch": "devpilot/issue-21-fix-scan-limit",
    "repair_base_branch": "devpilot-v1",
    "summary": {
        "branch": "devpilot/issue-21-fix-scan-limit",
        "changes": [" M corecoder/example.py"],
    },
    }


def test_workflow_checkpoint_restores_from_dict():
    checkpoint = issue_orchestration.WorkflowCheckpoint.from_dict(
        {
            "workflow_id": "issue-21",
            "state": "validate",
            "repair_branch": "devpilot/issue-21-fix-scan-limit",
            "repair_base_branch": "devpilot-v1",
            "summary": {
            "branch": "devpilot/issue-21-fix-scan-limit",
            "changes": [" M corecoder/example.py"],
            },
        }
    )

    assert checkpoint == issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.VALIDATE,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        repair_base_branch="devpilot-v1",
        summary=PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
        ),
    )

def test_workflow_checkpoint_round_trips_through_dict():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    push = RepairPush(
    remote="origin",
    branch="devpilot/issue-21-fix-scan-limit",
    commit_sha="0123456789abcdef",
    )

    pull_request = RepairPullRequest(
    repository="liuzhixin352-gif/CoreCoder",
    number=35,
    url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
    title="Add workflow checkpoint resume",
    base_branch="devpilot-v1",
    head_branch="devpilot/issue-21-fix-scan-limit",
    commit_sha="0123456789abcdef",
)

    ci_status = RepairCIStatus(
        repository="liuzhixin352-gif/CoreCoder",
        commit_sha="0123456789abcdef",
        state="failure",
        check_runs=(),
    )
    ci_retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )
    ci_retry_validation = PostRepairValidation(
    command=("python", "-m", "pytest", "tests", "-q"),
    status="passed",
    exit_code=0,
    passed_count=391,
    failed_count=0,
    error_count=0,
    output="391 passed",
    )

    ci_retry_commit = RepairCommit(
    sha="fedcba9876543210",
    message="Fix CI failure",
    )
    ci_retry_push = RepairPush(
    remote="origin",
    branch="devpilot/issue-21-fix-scan-limit",
    commit_sha="fedcba9876543210",
    )

    original = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_RETRY_VALIDATE,
        repair_branch="devpilot/issue-21-fix-scan-limit",
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

    restored = issue_orchestration.WorkflowCheckpoint.from_dict(
        original.to_dict()
    )

    assert restored == original

def test_save_workflow_checkpoint_writes_json(tmp_path):
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.VALIDATE,
        repair_branch="devpilot/issue-21-fix-scan-limit",
    )
    checkpoint_path = tmp_path / "workflow-checkpoint.json"

    issue_orchestration.save_workflow_checkpoint(
        checkpoint,
        checkpoint_path,
    )

    saved_data = json.loads(
        checkpoint_path.read_text(encoding="utf-8")
    )

    assert saved_data == checkpoint.to_dict()

def test_load_workflow_checkpoint_reads_json(tmp_path):
    checkpoint_path = tmp_path / "workflow-checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "workflow_id": "issue-21",
                "state": "validate",
                "repair_branch": "devpilot/issue-21-fix-scan-limit",
            }
        ),
        encoding="utf-8",
    )

    checkpoint = issue_orchestration.load_workflow_checkpoint(
        checkpoint_path
    )

    assert checkpoint == issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.VALIDATE,
        repair_branch="devpilot/issue-21-fix-scan-limit",
    )

def test_workflow_checkpoint_save_load_preserves_summary(tmp_path):
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    original = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.VALIDATE,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
    )
    checkpoint_path = tmp_path / "workflow-checkpoint.json"

    issue_orchestration.save_workflow_checkpoint(
        original,
        checkpoint_path,
    )
    restored = issue_orchestration.load_workflow_checkpoint(
        checkpoint_path
    )

    assert restored == original

def test_workflow_state_machine_restores_from_checkpoint():
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.VALIDATE,
        repair_branch="devpilot/issue-21-fix-scan-limit",
    )

    machine = (
        issue_orchestration.WorkflowStateMachine.from_checkpoint(
            checkpoint
        )
    )

    assert machine.state == issue_orchestration.WorkflowState.VALIDATE

def test_restored_workflow_state_machine_continues_from_checkpoint():
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.VALIDATE,
        repair_branch="devpilot/issue-21-fix-scan-limit",
    )

    machine = (
        issue_orchestration.WorkflowStateMachine.from_checkpoint(
            checkpoint
        )
    )

    machine.transition(
        issue_orchestration.WorkflowState.COMMIT
    )

    assert machine.state == issue_orchestration.WorkflowState.COMMIT

def test_workflow_checkpoint_tracks_summary_for_resume():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.VALIDATE,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
    )

    assert checkpoint.summary == summary

def test_run_issue_workflow_resumes_from_validate_checkpoint():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.VALIDATE,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="failed",
        exit_code=1,
        passed_count=390,
        failed_count=1,
        error_count=0,
        output="1 failed, 390 passed",
    )

    def run_agent(prompt):
        pytest.fail("Agent must not rerun when resuming from VALIDATE")

    def collect_summary(branch):
        pytest.fail(
            "Summary must come from checkpoint when resuming "
            "from VALIDATE"
        )

    result = issue_orchestration.run_issue_workflow(
        issue_prompt="Repair this Issue.",
        dry_run=False,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        run_agent=run_agent,
        collect_summary=collect_summary,
        run_validation=lambda: validation,
        checkpoint=checkpoint,
    )

    assert result.summary == summary
    assert result.validation == validation
    assert result.state == issue_orchestration.WorkflowState.VALIDATE


def test_run_issue_workflow_rejects_validate_checkpoint_without_summary():
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.VALIDATE,
        repair_branch="devpilot/issue-21-fix-scan-limit",
    )

    with pytest.raises(
        ValueError,
        match="summary is required to resume from VALIDATE",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            collect_summary=lambda branch: None,
            run_validation=lambda: None,
            checkpoint=checkpoint,
        )

def test_run_issue_workflow_rejects_checkpoint_for_different_branch():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.VALIDATE,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
    )

    with pytest.raises(
        ValueError,
        match="checkpoint repair branch does not match workflow",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-22-other-fix",
            run_agent=lambda prompt: None,
            collect_summary=lambda branch: None,
            run_validation=lambda: None,
            checkpoint=checkpoint,
        )


def test_run_issue_workflow_rejects_unsupported_checkpoint_state():
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.COMPLETED,
        repair_branch="devpilot/issue-21-fix-scan-limit",
    )

    with pytest.raises(
        ValueError,
        match="checkpoint resume is currently supported",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            checkpoint=checkpoint,
        )

def test_workflow_checkpoint_tracks_validation_for_resume():
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.COMMIT,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        validation=validation,
    )

    assert checkpoint.validation == validation


def test_workflow_checkpoint_serializes_validation():
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.COMMIT,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        validation=validation,
    )

    assert checkpoint.to_dict()["validation"] == {
        "command": ["python", "-m", "pytest", "tests", "-q"],
        "status": "passed",
        "exit_code": 0,
        "passed_count": 391,
        "failed_count": 0,
        "error_count": 0,
        "output": "391 passed",
    }

def test_workflow_checkpoint_restores_validation_from_dict():
    checkpoint = issue_orchestration.WorkflowCheckpoint.from_dict(
        {
            "workflow_id": "issue-21",
            "state": "commit",
            "repair_branch": "devpilot/issue-21-fix-scan-limit",
            "validation": {
                "command": [
                    "python",
                    "-m",
                    "pytest",
                    "tests",
                    "-q",
                ],
                "status": "passed",
                "exit_code": 0,
                "passed_count": 391,
                "failed_count": 0,
                "error_count": 0,
                "output": "391 passed",
            },
        }
    )

    assert checkpoint.validation == PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )

def test_run_issue_workflow_resumes_from_commit_checkpoint():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.COMMIT,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
        validation=validation,
    )

    def run_agent(prompt):
        pytest.fail("Agent must not rerun when resuming from COMMIT")

    def collect_summary(branch):
        pytest.fail(
            "Summary must come from checkpoint when resuming "
            "from COMMIT"
        )

    def run_validation():
        pytest.fail(
            "Validation must come from checkpoint when resuming "
            "from COMMIT"
        )

    def create_commit():
        raise RuntimeError("commit callback reached")

    with pytest.raises(
        RuntimeError,
        match="commit callback reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=run_agent,
            collect_summary=collect_summary,
            run_validation=run_validation,
            create_commit=create_commit,
            checkpoint=checkpoint,
        )

def test_run_issue_workflow_rejects_commit_checkpoint_without_validation():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.COMMIT,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
    )

    with pytest.raises(
        ValueError,
        match="validation is required to resume from COMMIT",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            checkpoint=checkpoint,
        )

def test_workflow_checkpoint_tracks_commit_for_resume():
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.PUSH,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        commit=commit,
    )

    assert checkpoint.commit == commit


def test_workflow_checkpoint_serializes_commit():
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.PUSH,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        commit=commit,
    )

    assert checkpoint.to_dict()["commit"] == {
        "sha": "0123456789abcdef",
        "message": "Fix #21: Fix repository scan limit",
    }

def test_workflow_checkpoint_restores_commit_from_dict():
    checkpoint = issue_orchestration.WorkflowCheckpoint.from_dict(
        {
            "workflow_id": "issue-21",
            "state": "push",
            "repair_branch": "devpilot/issue-21-fix-scan-limit",
            "commit": {
                "sha": "0123456789abcdef",
                "message": "Fix #21: Fix repository scan limit",
            },
        }
    )

    assert checkpoint.commit == RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

def test_run_issue_workflow_resumes_from_push_checkpoint():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.PUSH,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
        validation=validation,
        commit=commit,
    )

    def run_agent(prompt):
        pytest.fail("Agent must not rerun when resuming from PUSH")

    def collect_summary(branch):
        pytest.fail(
            "Summary must come from checkpoint when resuming "
            "from PUSH"
        )

    def run_validation():
        pytest.fail(
            "Validation must come from checkpoint when resuming "
            "from PUSH"
        )

    def create_commit():
        pytest.fail(
            "Commit must come from checkpoint when resuming "
            "from PUSH"
        )

    def push_commit(branch, commit_sha):
        assert branch == "devpilot/issue-21-fix-scan-limit"
        assert commit_sha == commit.sha
        raise RuntimeError("push callback reached")

    with pytest.raises(
        RuntimeError,
        match="push callback reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=run_agent,
            collect_summary=collect_summary,
            run_validation=run_validation,
            create_commit=create_commit,
            push_commit=push_commit,
            checkpoint=checkpoint,
        )

def test_run_issue_workflow_rejects_push_checkpoint_without_commit():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.PUSH,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
        validation=validation,
    )

    with pytest.raises(
        ValueError,
        match="commit is required to resume from PUSH",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            checkpoint=checkpoint,
        )

def test_workflow_checkpoint_tracks_push_for_resume():
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha="0123456789abcdef",
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CREATE_PR,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        push=push,
    )

    assert checkpoint.push == push

def test_workflow_checkpoint_serializes_push():
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha="0123456789abcdef",
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CREATE_PR,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        push=push,
    )

    assert checkpoint.to_dict()["push"] == {
        "remote": "origin",
        "branch": "devpilot/issue-21-fix-scan-limit",
        "commit_sha": "0123456789abcdef",
    }
def test_workflow_checkpoint_restores_push_from_dict():
    checkpoint = issue_orchestration.WorkflowCheckpoint.from_dict(
        {
            "workflow_id": "issue-21",
            "state": "create_pr",
            "repair_branch": "devpilot/issue-21-fix-scan-limit",
            "push": {
                "remote": "origin",
                "branch": "devpilot/issue-21-fix-scan-limit",
                "commit_sha": "0123456789abcdef",
            },
        }
    )

    assert checkpoint.push == RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha="0123456789abcdef",
    )

def test_run_issue_workflow_resumes_from_create_pr_checkpoint():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CREATE_PR,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
        validation=validation,
        commit=commit,
        push=push,
    )

    def fail(*args, **kwargs):
        pytest.fail(
            "Earlier workflow stages must not rerun "
            "when resuming from CREATE_PR"
        )

    def create_pull_request(saved_push):
        assert saved_push == push
        raise RuntimeError("create PR callback reached")

    with pytest.raises(
        RuntimeError,
        match="create PR callback reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=fail,
            collect_summary=fail,
            run_validation=fail,
            create_commit=fail,
            push_commit=fail,
            create_pull_request=create_pull_request,
            checkpoint=checkpoint,
        )

def test_run_issue_workflow_rejects_create_pr_checkpoint_without_push():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CREATE_PR,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
        validation=validation,
        commit=commit,
    )

    with pytest.raises(
        ValueError,
        match="push is required to resume from CREATE_PR",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            checkpoint=checkpoint,
        )

def test_workflow_checkpoint_tracks_pull_request_for_resume():
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha="0123456789abcdef",
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.WAIT_CI,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        pull_request=pull_request,
    )

    assert checkpoint.pull_request == pull_request

def test_workflow_checkpoint_serializes_pull_request():
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha="0123456789abcdef",
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.WAIT_CI,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        pull_request=pull_request,
    )

    assert checkpoint.to_dict()["pull_request"] == {
        "repository": "liuzhixin352-gif/CoreCoder",
        "number": 35,
        "url": "https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        "title": "Add workflow checkpoint resume",
        "base_branch": "devpilot-v1",
        "head_branch": "devpilot/issue-21-fix-scan-limit",
        "commit_sha": "0123456789abcdef",
    }

def test_workflow_checkpoint_restores_pull_request_from_dict():
    checkpoint = issue_orchestration.WorkflowCheckpoint.from_dict(
        {
            "workflow_id": "issue-21",
            "state": "wait_ci",
            "repair_branch": "devpilot/issue-21-fix-scan-limit",
            "pull_request": {
                "repository": "liuzhixin352-gif/CoreCoder",
                "number": 35,
                "url": "https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
                "title": "Add workflow checkpoint resume",
                "base_branch": "devpilot-v1",
                "head_branch": "devpilot/issue-21-fix-scan-limit",
                "commit_sha": "0123456789abcdef",
            },
        }
    )

    assert checkpoint.pull_request == RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha="0123456789abcdef",
    )

def test_run_issue_workflow_resumes_from_wait_ci_checkpoint():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.WAIT_CI,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
        validation=validation,
        commit=commit,
        push=push,
        pull_request=pull_request,
    )

    def fail(*args, **kwargs):
        pytest.fail(
            "Earlier workflow stages must not rerun "
            "when resuming from WAIT_CI"
        )

    def wait_for_ci(saved_pull_request):
        assert saved_pull_request == pull_request
        raise RuntimeError("wait CI callback reached")

    with pytest.raises(
        RuntimeError,
        match="wait CI callback reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=fail,
            collect_summary=fail,
            run_validation=fail,
            create_commit=fail,
            push_commit=fail,
            create_pull_request=fail,
            wait_for_ci=wait_for_ci,
            checkpoint=checkpoint,
        )
def test_run_issue_workflow_rejects_wait_ci_checkpoint_without_pull_request():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.WAIT_CI,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
        validation=validation,
        commit=commit,
        push=push,
    )

    with pytest.raises(
        ValueError,
        match="pull request is required to resume from WAIT_CI",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            checkpoint=checkpoint,
        )

def test_workflow_checkpoint_tracks_ci_status_for_resume():
    ci_status = RepairCIStatus(
        repository="liuzhixin352-gif/CoreCoder",
        commit_sha="0123456789abcdef",
        state="failure",
        check_runs=(),
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_REPAIR,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        ci_status=ci_status,
    )

    assert checkpoint.ci_status == ci_status

def test_workflow_checkpoint_serializes_ci_status():
    ci_status = RepairCIStatus(
        repository="liuzhixin352-gif/CoreCoder",
        commit_sha="0123456789abcdef",
        state="failure",
        check_runs=(),
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_REPAIR,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        ci_status=ci_status,
    )

    assert checkpoint.to_dict()["ci_status"] == {
        "repository": "liuzhixin352-gif/CoreCoder",
        "commit_sha": "0123456789abcdef",
        "state": "failure",
        "check_runs": [],
    }
def test_workflow_checkpoint_restores_ci_status_from_dict():
    checkpoint = issue_orchestration.WorkflowCheckpoint.from_dict(
        {
            "workflow_id": "issue-21",
            "state": "ci_repair",
            "repair_branch": "devpilot/issue-21-fix-scan-limit",
            "ci_status": {
                "repository": "liuzhixin352-gif/CoreCoder",
                "commit_sha": "0123456789abcdef",
                "state": "failure",
                "check_runs": [
                    {
                        "name": "tests",
                        "status": "completed",
                        "conclusion": "failure",
                        "details_url": "https://example.com/checks/1",
                    }
                ],
            },
        }
    )

    assert checkpoint.ci_status is not None
    assert checkpoint.ci_status.repository == "liuzhixin352-gif/CoreCoder"
    assert checkpoint.ci_status.commit_sha == "0123456789abcdef"
    assert checkpoint.ci_status.state == "failure"

    assert len(checkpoint.ci_status.check_runs) == 1
    check_run = checkpoint.ci_status.check_runs[0]
    assert check_run.name == "tests"
    assert check_run.status == "completed"
    assert check_run.conclusion == "failure"
    assert check_run.details_url == "https://example.com/checks/1"

def test_run_issue_workflow_resumes_from_ci_repair_checkpoint():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    ci_status = RepairCIStatus(
        repository="liuzhixin352-gif/CoreCoder",
        commit_sha=commit.sha,
        state="failure",
        check_runs=(),
    )
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_REPAIR,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
        validation=validation,
        commit=commit,
        push=push,
        pull_request=pull_request,
        ci_status=ci_status,
    )

    def fail(*args, **kwargs):
        pytest.fail(
            "Earlier workflow stages must not rerun "
            "when resuming from CI_REPAIR"
        )

    def build_ci_failure_prompt(saved_ci_status):
        assert saved_ci_status == ci_status
        return "Repair the failing CI."

    def run_agent(prompt):
        assert prompt == "Repair the failing CI."
        raise RuntimeError("CI repair agent reached")

    with pytest.raises(
        RuntimeError,
        match="CI repair agent reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=run_agent,
            collect_summary=fail,
            run_validation=fail,
            create_commit=fail,
            push_commit=fail,
            create_pull_request=fail,
            wait_for_ci=fail,
            build_ci_failure_prompt=build_ci_failure_prompt,
            checkpoint=checkpoint,
        )

def test_run_issue_workflow_rejects_ci_repair_checkpoint_without_ci_status():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_REPAIR,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
        validation=validation,
        commit=commit,
        push=push,
        pull_request=pull_request,
    )

    with pytest.raises(
        ValueError,
        match="CI status is required to resume from CI_REPAIR",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            checkpoint=checkpoint,
        )
def test_run_issue_workflow_resumes_from_ci_retry_summary_checkpoint():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    ci_status = RepairCIStatus(
        repository="liuzhixin352-gif/CoreCoder",
        commit_sha=commit.sha,
        state="failure",
        check_runs=(),
    )
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_RETRY_SUMMARY,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
        validation=validation,
        commit=commit,
        push=push,
        pull_request=pull_request,
        ci_status=ci_status,
    )

    def run_agent(prompt):
        pytest.fail(
            "Agent must not rerun when resuming "
            "from CI_RETRY_SUMMARY"
        )

    def collect_summary(branch):
        assert branch == "devpilot/issue-21-fix-scan-limit"
        raise RuntimeError("CI retry summary callback reached")

    with pytest.raises(
        RuntimeError,
        match="CI retry summary callback reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=run_agent,
            collect_summary=collect_summary,
            checkpoint=checkpoint,
        )
def test_workflow_checkpoint_tracks_ci_retry_summary_for_resume():
    ci_retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_RETRY_VALIDATE,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        ci_retry_summary=ci_retry_summary,
    )

    assert checkpoint.ci_retry_summary == ci_retry_summary

def test_workflow_checkpoint_serializes_ci_retry_summary():
    ci_retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_RETRY_VALIDATE,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        ci_retry_summary=ci_retry_summary,
    )

    assert checkpoint.to_dict()["ci_retry_summary"] == {
        "branch": "devpilot/issue-21-fix-scan-limit",
        "changes": [" M corecoder/ci_fix.py"],
    }

def test_workflow_checkpoint_restores_ci_retry_summary_from_dict():
    checkpoint = issue_orchestration.WorkflowCheckpoint.from_dict(
        {
            "workflow_id": "issue-21",
            "state": "ci_retry_validate",
            "repair_branch": "devpilot/issue-21-fix-scan-limit",
            "ci_retry_summary": {
                "branch": "devpilot/issue-21-fix-scan-limit",
                "changes": [" M corecoder/ci_fix.py"],
            },
        }
    )

    assert checkpoint.ci_retry_summary == PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )

def test_run_issue_workflow_resumes_from_ci_retry_validate_checkpoint():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    ci_status = RepairCIStatus(
        repository="liuzhixin352-gif/CoreCoder",
        commit_sha=commit.sha,
        state="failure",
        check_runs=(),
    )
    ci_retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )
    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_RETRY_VALIDATE,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
        validation=validation,
        commit=commit,
        push=push,
        pull_request=pull_request,
        ci_status=ci_status,
        ci_retry_summary=ci_retry_summary,
    )

    def fail(*args, **kwargs):
        pytest.fail(
            "Earlier workflow stages must not rerun "
            "when resuming from CI_RETRY_VALIDATE"
        )

    def run_validation():
        return PostRepairValidation(
            command=("python", "-m", "pytest", "tests", "-q"),
            status="failed",
            exit_code=1,
            passed_count=390,
            failed_count=1,
            error_count=0,
            output="1 failed, 390 passed",
        )

    with pytest.raises(
        ValueError,
        match="CI repair validation failed",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=fail,
            collect_summary=fail,
            run_validation=run_validation,
            create_commit=fail,
            push_commit=fail,
            create_pull_request=fail,
            wait_for_ci=fail,
            build_ci_failure_prompt=fail,
            checkpoint=checkpoint,
        )

def test_run_issue_workflow_rejects_ci_retry_validate_checkpoint_without_summary():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    ci_status = RepairCIStatus(
        repository="liuzhixin352-gif/CoreCoder",
        commit_sha=commit.sha,
        state="failure",
        check_runs=(),
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_RETRY_VALIDATE,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
        validation=validation,
        commit=commit,
        push=push,
        pull_request=pull_request,
        ci_status=ci_status,
    )

    with pytest.raises(
        ValueError,
        match="CI retry summary is required to resume from CI_RETRY_VALIDATE",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            checkpoint=checkpoint,
        )

def test_workflow_checkpoint_tracks_ci_retry_validation_for_resume():
    ci_retry_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_RETRY_COMMIT,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        ci_retry_validation=ci_retry_validation,
    )

    assert checkpoint.ci_retry_validation == ci_retry_validation

def test_workflow_checkpoint_serializes_ci_retry_validation():
    ci_retry_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_RETRY_COMMIT,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        ci_retry_validation=ci_retry_validation,
    )

    assert checkpoint.to_dict()["ci_retry_validation"] == {
        "command": ["python", "-m", "pytest", "tests", "-q"],
        "status": "passed",
        "exit_code": 0,
        "passed_count": 391,
        "failed_count": 0,
        "error_count": 0,
        "output": "391 passed",
    }

def test_workflow_checkpoint_restores_ci_retry_validation_from_dict():
    checkpoint = issue_orchestration.WorkflowCheckpoint.from_dict(
        {
            "workflow_id": "issue-21",
            "state": "ci_retry_commit",
            "repair_branch": "devpilot/issue-21-fix-scan-limit",
            "ci_retry_validation": {
                "command": ["python", "-m", "pytest", "tests", "-q"],
                "status": "passed",
                "exit_code": 0,
                "passed_count": 391,
                "failed_count": 0,
                "error_count": 0,
                "output": "391 passed",
            },
        }
    )

    assert checkpoint.ci_retry_validation == PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )

def test_run_issue_workflow_resumes_from_ci_retry_commit_checkpoint():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    ci_status = RepairCIStatus(
        repository="liuzhixin352-gif/CoreCoder",
        commit_sha=commit.sha,
        state="failure",
        check_runs=(),
    )
    ci_retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )
    ci_retry_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_RETRY_COMMIT,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        summary=summary,
        validation=validation,
        commit=commit,
        push=push,
        pull_request=pull_request,
        ci_status=ci_status,
        ci_retry_summary=ci_retry_summary,
        ci_retry_validation=ci_retry_validation,
    )

    def fail(*args, **kwargs):
        pytest.fail(
            "Earlier workflow stages must not rerun "
            "when resuming from CI_RETRY_COMMIT"
        )

    def create_commit():
        raise RuntimeError("CI retry commit callback reached")

    with pytest.raises(
        RuntimeError,
        match="CI retry commit callback reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=fail,
            collect_summary=fail,
            run_validation=fail,
            create_commit=create_commit,
            push_commit=fail,
            create_pull_request=fail,
            wait_for_ci=fail,
            build_ci_failure_prompt=fail,
            checkpoint=checkpoint,
        )

def test_workflow_checkpoint_tracks_ci_retry_commit_for_resume():
    ci_retry_commit = RepairCommit(
        sha="fedcba9876543210",
        message="Fix CI failure",
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_RETRY_PUSH,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        ci_retry_commit=ci_retry_commit,
    )

    assert checkpoint.ci_retry_commit == ci_retry_commit

def test_workflow_checkpoint_serializes_ci_retry_commit():
    ci_retry_commit = RepairCommit(
        sha="fedcba9876543210",
        message="Fix CI failure",
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_RETRY_PUSH,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        ci_retry_commit=ci_retry_commit,
    )

    assert checkpoint.to_dict()["ci_retry_commit"] == {
        "sha": "fedcba9876543210",
        "message": "Fix CI failure",
    }

def test_workflow_checkpoint_restores_ci_retry_commit_from_dict():
    checkpoint = issue_orchestration.WorkflowCheckpoint.from_dict(
        {
            "workflow_id": "issue-21",
            "state": "ci_retry_push",
            "repair_branch": "devpilot/issue-21-fix-scan-limit",
            "ci_retry_commit": {
                "sha": "fedcba9876543210",
                "message": "Fix CI failure",
            },
        }
    )

    assert checkpoint.ci_retry_commit == RepairCommit(
        sha="fedcba9876543210",
        message="Fix CI failure",
    )
def test_run_issue_workflow_resumes_from_ci_retry_push_checkpoint():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    ci_status = RepairCIStatus(
        repository="liuzhixin352-gif/CoreCoder",
        commit_sha=commit.sha,
        state="failure",
        check_runs=(),
    )
    ci_retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )
    ci_retry_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    ci_retry_commit = RepairCommit(
        sha="fedcba9876543210",
        message="Fix CI failure",
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_RETRY_PUSH,
        repair_branch="devpilot/issue-21-fix-scan-limit",
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

    def fail(*args, **kwargs):
        pytest.fail(
            "Earlier workflow stages must not rerun "
            "when resuming from CI_RETRY_PUSH"
        )

    def push_commit(branch, commit_sha):
        assert branch == "devpilot/issue-21-fix-scan-limit"
        assert commit_sha == ci_retry_commit.sha
        raise RuntimeError("CI retry push callback reached")

    with pytest.raises(
        RuntimeError,
        match="CI retry push callback reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=fail,
            collect_summary=fail,
            run_validation=fail,
            create_commit=fail,
            push_commit=push_commit,
            create_pull_request=fail,
            wait_for_ci=fail,
            build_ci_failure_prompt=fail,
            checkpoint=checkpoint,
        )
def test_workflow_checkpoint_tracks_ci_retry_push_for_resume():
    ci_retry_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha="fedcba9876543210",
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.WAIT_RETRY_CI,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        ci_retry_push=ci_retry_push,
    )

    assert checkpoint.ci_retry_push == ci_retry_push

def test_workflow_checkpoint_serializes_ci_retry_push():
    ci_retry_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha="fedcba9876543210",
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.WAIT_RETRY_CI,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        ci_retry_push=ci_retry_push,
    )

    assert checkpoint.to_dict()["ci_retry_push"] == {
        "remote": "origin",
        "branch": "devpilot/issue-21-fix-scan-limit",
        "commit_sha": "fedcba9876543210",
    }

def test_workflow_checkpoint_restores_ci_retry_push_from_dict():
    checkpoint = issue_orchestration.WorkflowCheckpoint.from_dict(
        {
            "workflow_id": "issue-21",
            "state": "wait_retry_ci",
            "repair_branch": "devpilot/issue-21-fix-scan-limit",
            "ci_retry_push": {
                "remote": "origin",
                "branch": "devpilot/issue-21-fix-scan-limit",
                "commit_sha": "fedcba9876543210",
            },
        }
    )

    assert checkpoint.ci_retry_push == RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha="fedcba9876543210",
    )

def test_run_issue_workflow_resumes_from_wait_retry_ci_checkpoint():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    ci_status = RepairCIStatus(
        repository="liuzhixin352-gif/CoreCoder",
        commit_sha=commit.sha,
        state="failure",
        check_runs=(),
    )
    ci_retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )
    ci_retry_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    ci_retry_commit = RepairCommit(
        sha="fedcba9876543210",
        message="Fix CI failure",
    )
    ci_retry_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=ci_retry_commit.sha,
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.WAIT_RETRY_CI,
        repair_branch="devpilot/issue-21-fix-scan-limit",
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

    def fail(*args, **kwargs):
        pytest.fail(
            "Earlier workflow stages must not rerun "
            "when resuming from WAIT_RETRY_CI"
        )

    def wait_for_retry_ci(saved_retry_push):
        assert saved_retry_push == ci_retry_push
        raise RuntimeError("wait retry CI callback reached")

    with pytest.raises(
        RuntimeError,
        match="wait retry CI callback reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=fail,
            collect_summary=fail,
            run_validation=fail,
            create_commit=fail,
            push_commit=fail,
            create_pull_request=fail,
            wait_for_ci=fail,
            build_ci_failure_prompt=fail,
            wait_for_retry_ci=wait_for_retry_ci,
            checkpoint=checkpoint,
        )
def test_run_issue_workflow_rejects_wait_retry_ci_checkpoint_without_retry_push():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    ci_status = RepairCIStatus(
        repository="liuzhixin352-gif/CoreCoder",
        commit_sha=commit.sha,
        state="failure",
        check_runs=(),
    )
    ci_retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )
    ci_retry_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    ci_retry_commit = RepairCommit(
        sha="fedcba9876543210",
        message="Fix CI failure",
    )

    checkpoint = issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.WAIT_RETRY_CI,
        repair_branch="devpilot/issue-21-fix-scan-limit",
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

    with pytest.raises(
        ValueError,
        match="CI retry push is required to resume from WAIT_RETRY_CI",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            repair_base_branch="devpilot-v1",
            run_agent=lambda prompt: None,
            checkpoint=checkpoint,
        )
def test_run_issue_workflow_emits_validate_checkpoint_after_summary():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    checkpoints = []

    def run_validation():
        raise RuntimeError("validation reached")

    with pytest.raises(
        RuntimeError,
        match="validation reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            repair_base_branch="devpilot-v1",
            workflow_id="issue-21",
            run_agent=lambda prompt: None,
            collect_summary=lambda branch: summary,
            run_validation=run_validation,
            save_checkpoint=checkpoints.append,
        )

    assert checkpoints == [
        issue_orchestration.WorkflowCheckpoint(
            workflow_id="issue-21",
            state=issue_orchestration.WorkflowState.VALIDATE,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            repair_base_branch="devpilot-v1",
            summary=summary,
        )
    ]

def test_run_issue_workflow_emits_commit_checkpoint_after_validation():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    checkpoints = []

    def create_commit():
        raise RuntimeError("commit reached")

    with pytest.raises(
        RuntimeError,
        match="commit reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            repair_base_branch="devpilot-v1",
            workflow_id="issue-21",
            run_agent=lambda prompt: None,
            collect_summary=lambda branch: summary,
            run_validation=lambda: validation,
            create_commit=create_commit,
            save_checkpoint=checkpoints.append,
        )

    assert [checkpoint.state for checkpoint in checkpoints] == [
        issue_orchestration.WorkflowState.VALIDATE,
        issue_orchestration.WorkflowState.COMMIT,
    ]

    assert checkpoints[-1] == issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.COMMIT,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        repair_base_branch="devpilot-v1",
        summary=summary,
        validation=validation,
    )

def test_run_issue_workflow_emits_push_checkpoint_after_commit():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    checkpoints = []

    def push_commit(branch, commit_sha):
        raise RuntimeError("push reached")

    with pytest.raises(
        RuntimeError,
        match="push reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            repair_base_branch="devpilot-v1",
            workflow_id="issue-21",
            run_agent=lambda prompt: None,
            collect_summary=lambda branch: summary,
            run_validation=lambda: validation,
            create_commit=lambda: commit,
            push_commit=push_commit,
            save_checkpoint=checkpoints.append,
        )

    assert [checkpoint.state for checkpoint in checkpoints] == [
        issue_orchestration.WorkflowState.VALIDATE,
        issue_orchestration.WorkflowState.COMMIT,
        issue_orchestration.WorkflowState.PUSH,
    ]

    assert checkpoints[-1] == issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.PUSH,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        repair_base_branch="devpilot-v1",
        summary=summary,
        validation=validation,
        commit=commit,
    )
def test_run_issue_workflow_emits_create_pr_checkpoint_after_push():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    checkpoints = []

    def create_pull_request(saved_push):
        assert saved_push == push
        raise RuntimeError("create PR reached")

    with pytest.raises(
        RuntimeError,
        match="create PR reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            repair_base_branch="devpilot-v1",
            workflow_id="issue-21",
            run_agent=lambda prompt: None,
            collect_summary=lambda branch: summary,
            run_validation=lambda: validation,
            create_commit=lambda: commit,
            push_commit=lambda branch, commit_sha: push,
            create_pull_request=create_pull_request,
            save_checkpoint=checkpoints.append,
        )

    assert [checkpoint.state for checkpoint in checkpoints] == [
        issue_orchestration.WorkflowState.VALIDATE,
        issue_orchestration.WorkflowState.COMMIT,
        issue_orchestration.WorkflowState.PUSH,
        issue_orchestration.WorkflowState.CREATE_PR,
    ]

    assert checkpoints[-1] == issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CREATE_PR,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        repair_base_branch="devpilot-v1",
        summary=summary,
        validation=validation,
        commit=commit,
        push=push,
    )
def test_run_issue_workflow_emits_wait_ci_checkpoint_after_pull_request():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    checkpoints = []

    def wait_for_ci(saved_pull_request):
        assert saved_pull_request == pull_request
        raise RuntimeError("wait CI reached")

    with pytest.raises(
        RuntimeError,
        match="wait CI reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            repair_base_branch="devpilot-v1",
            workflow_id="issue-21",
            run_agent=lambda prompt: None,
            collect_summary=lambda branch: summary,
            run_validation=lambda: validation,
            create_commit=lambda: commit,
            push_commit=lambda branch, commit_sha: push,
            create_pull_request=lambda saved_push: pull_request,
            wait_for_ci=wait_for_ci,
            save_checkpoint=checkpoints.append,
        )

    assert [checkpoint.state for checkpoint in checkpoints] == [
        issue_orchestration.WorkflowState.VALIDATE,
        issue_orchestration.WorkflowState.COMMIT,
        issue_orchestration.WorkflowState.PUSH,
        issue_orchestration.WorkflowState.CREATE_PR,
        issue_orchestration.WorkflowState.WAIT_CI,
    ]

    assert checkpoints[-1] == issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.WAIT_CI,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        repair_base_branch="devpilot-v1",
        summary=summary,
        validation=validation,
        commit=commit,
        push=push,
        pull_request=pull_request,
    )
def test_run_issue_workflow_emits_ci_repair_checkpoint_after_ci_failure():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    ci_status = RepairCIStatus(
        repository="liuzhixin352-gif/CoreCoder",
        commit_sha=commit.sha,
        state="failure",
        check_runs=(),
    )
    checkpoints = []

    def build_ci_failure_prompt(saved_ci_status):
        assert saved_ci_status == ci_status
        raise RuntimeError("CI repair prompt reached")

    with pytest.raises(
        RuntimeError,
        match="CI repair prompt reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            repair_base_branch="devpilot-v1",
            workflow_id="issue-21",
            run_agent=lambda prompt: None,
            collect_summary=lambda branch: summary,
            run_validation=lambda: validation,
            create_commit=lambda: commit,
            push_commit=lambda branch, commit_sha: push,
            create_pull_request=lambda saved_push: pull_request,
            wait_for_ci=lambda saved_pull_request: ci_status,
            build_ci_failure_prompt=build_ci_failure_prompt,
            save_checkpoint=checkpoints.append,
        )

    assert [checkpoint.state for checkpoint in checkpoints] == [
        issue_orchestration.WorkflowState.VALIDATE,
        issue_orchestration.WorkflowState.COMMIT,
        issue_orchestration.WorkflowState.PUSH,
        issue_orchestration.WorkflowState.CREATE_PR,
        issue_orchestration.WorkflowState.WAIT_CI,
        issue_orchestration.WorkflowState.CI_REPAIR,
    ]

    assert checkpoints[-1] == issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_REPAIR,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        repair_base_branch="devpilot-v1",
        summary=summary,
        validation=validation,
        commit=commit,
        push=push,
        pull_request=pull_request,
        ci_status=ci_status,
    )

def test_run_issue_workflow_emits_ci_retry_summary_checkpoint_after_ci_repair():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    ci_status = RepairCIStatus(
        repository="liuzhixin352-gif/CoreCoder",
        commit_sha=commit.sha,
        state="failure",
        check_runs=(),
    )
    checkpoints = []
    collect_calls = 0

    def collect_summary(branch):
        nonlocal collect_calls
        collect_calls += 1

        if collect_calls == 1:
            return summary

        raise RuntimeError("retry summary reached")


    with pytest.raises(
        RuntimeError,
        match="retry summary reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            repair_base_branch="devpilot-v1",
            workflow_id="issue-21",
            run_agent=lambda prompt: None,
            collect_summary=collect_summary,
            run_validation=lambda: validation,
            create_commit=lambda: commit,
            push_commit=lambda branch, commit_sha: push,
            create_pull_request=lambda saved_push: pull_request,
            wait_for_ci=lambda saved_pull_request: ci_status,
            build_ci_failure_prompt=lambda saved_ci_status: (
                "Repair CI failure."
            ),
            save_checkpoint=checkpoints.append,
        )

    assert checkpoints[-1] == issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_RETRY_SUMMARY,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        repair_base_branch="devpilot-v1",
        summary=summary,
        validation=validation,
        commit=commit,
        push=push,
        pull_request=pull_request,
        ci_status=ci_status,
    )

def test_run_issue_workflow_emits_ci_retry_validate_checkpoint_after_retry_summary():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    ci_retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    ci_status = RepairCIStatus(
        repository="liuzhixin352-gif/CoreCoder",
        commit_sha=commit.sha,
        state="failure",
        check_runs=(),
    )

    checkpoints = []
    collect_calls = 0
    validation_calls = 0

    def collect_summary(branch):
        nonlocal collect_calls
        collect_calls += 1

        if collect_calls == 1:
            return summary

        return ci_retry_summary

    def run_validation():
        nonlocal validation_calls
        validation_calls += 1

        if validation_calls == 1:
            return validation

        raise RuntimeError("retry validation reached")

    with pytest.raises(
        RuntimeError,
        match="retry validation reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            repair_base_branch="devpilot-v1",
            workflow_id="issue-21",
            run_agent=lambda prompt: None,
            collect_summary=collect_summary,
            run_validation=run_validation,
            create_commit=lambda: commit,
            push_commit=lambda branch, commit_sha: push,
            create_pull_request=lambda saved_push: pull_request,
            wait_for_ci=lambda saved_pull_request: ci_status,
            build_ci_failure_prompt=lambda saved_ci_status: (
                "Repair CI failure."
            ),
            save_checkpoint=checkpoints.append,
        )

    assert checkpoints[-1] == issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_RETRY_VALIDATE,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        repair_base_branch="devpilot-v1",
        summary=summary,
        validation=validation,
        commit=commit,
        push=push,
        pull_request=pull_request,
        ci_status=ci_status,
        ci_retry_summary=ci_retry_summary,
    )

def test_run_issue_workflow_emits_ci_retry_commit_checkpoint_after_retry_validation():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    ci_retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    ci_retry_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=392,
        failed_count=0,
        error_count=0,
        output="392 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    ci_status = RepairCIStatus(
        repository="liuzhixin352-gif/CoreCoder",
        commit_sha=commit.sha,
        state="failure",
        check_runs=(),
    )

    checkpoints = []
    collect_calls = 0
    validation_calls = 0
    commit_calls = 0

    def collect_summary(branch):
        nonlocal collect_calls
        collect_calls += 1
        return summary if collect_calls == 1 else ci_retry_summary

    def run_validation():
        nonlocal validation_calls
        validation_calls += 1
        return validation if validation_calls == 1 else ci_retry_validation

    def create_commit():
        nonlocal commit_calls
        commit_calls += 1

        if commit_calls == 1:
            return commit

        raise RuntimeError("retry commit reached")

    with pytest.raises(
        RuntimeError,
        match="retry commit reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            repair_base_branch="devpilot-v1",
            workflow_id="issue-21",
            run_agent=lambda prompt: None,
            collect_summary=collect_summary,
            run_validation=run_validation,
            create_commit=create_commit,
            push_commit=lambda branch, commit_sha: push,
            create_pull_request=lambda saved_push: pull_request,
            wait_for_ci=lambda saved_pull_request: ci_status,
            build_ci_failure_prompt=lambda saved_ci_status: (
                "Repair CI failure."
            ),
            save_checkpoint=checkpoints.append,
        )

    assert checkpoints[-1] == issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_RETRY_COMMIT,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        repair_base_branch="devpilot-v1",
        summary=summary,
        validation=validation,
        commit=commit,
        push=push,
        pull_request=pull_request,
        ci_status=ci_status,
        ci_retry_summary=ci_retry_summary,
        ci_retry_validation=ci_retry_validation,
    )

def test_run_issue_workflow_emits_ci_retry_push_checkpoint_after_retry_commit():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    ci_retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    ci_retry_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=392,
        failed_count=0,
        error_count=0,
        output="392 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    ci_retry_commit = RepairCommit(
        sha="fedcba9876543210",
        message="Fix CI failure",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    ci_status = RepairCIStatus(
        repository="liuzhixin352-gif/CoreCoder",
        commit_sha=commit.sha,
        state="failure",
        check_runs=(),
    )

    checkpoints = []
    collect_calls = 0
    validation_calls = 0
    commit_calls = 0
    push_calls = 0

    def collect_summary(branch):
        nonlocal collect_calls
        collect_calls += 1
        return summary if collect_calls == 1 else ci_retry_summary

    def run_validation():
        nonlocal validation_calls
        validation_calls += 1
        return validation if validation_calls == 1 else ci_retry_validation

    def create_commit():
        nonlocal commit_calls
        commit_calls += 1
        return commit if commit_calls == 1 else ci_retry_commit

    def push_commit(branch, commit_sha):
        nonlocal push_calls
        push_calls += 1

        if push_calls == 1:
            return push

        assert commit_sha == ci_retry_commit.sha
        raise RuntimeError("retry push reached")

    with pytest.raises(
        RuntimeError,
        match="retry push reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            repair_base_branch="devpilot-v1",
            workflow_id="issue-21",
            run_agent=lambda prompt: None,
            collect_summary=collect_summary,
            run_validation=run_validation,
            create_commit=create_commit,
            push_commit=push_commit,
            create_pull_request=lambda saved_push: pull_request,
            wait_for_ci=lambda saved_pull_request: ci_status,
            build_ci_failure_prompt=lambda saved_ci_status: (
                "Repair CI failure."
            ),
            save_checkpoint=checkpoints.append,
        )

    assert checkpoints[-1] == issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.CI_RETRY_PUSH,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        repair_base_branch="devpilot-v1",
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

def test_run_issue_workflow_emits_wait_retry_ci_checkpoint_after_retry_push():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    ci_retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )
    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=391,
        failed_count=0,
        error_count=0,
        output="391 passed",
    )
    ci_retry_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=392,
        failed_count=0,
        error_count=0,
        output="392 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    ci_retry_commit = RepairCommit(
        sha="fedcba9876543210",
        message="Fix CI failure",
    )
    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    ci_retry_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=ci_retry_commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="liuzhixin352-gif/CoreCoder",
        number=35,
        url="https://github.com/liuzhixin352-gif/CoreCoder/pull/35",
        title="Add workflow checkpoint resume",
        base_branch="devpilot-v1",
        head_branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    ci_status = RepairCIStatus(
        repository="liuzhixin352-gif/CoreCoder",
        commit_sha=commit.sha,
        state="failure",
        check_runs=(),
    )

    checkpoints = []
    collect_calls = 0
    validation_calls = 0
    commit_calls = 0
    push_calls = 0

    def collect_summary(branch):
        nonlocal collect_calls
        collect_calls += 1
        return summary if collect_calls == 1 else ci_retry_summary

    def run_validation():
        nonlocal validation_calls
        validation_calls += 1
        return validation if validation_calls == 1 else ci_retry_validation

    def create_commit():
        nonlocal commit_calls
        commit_calls += 1
        return commit if commit_calls == 1 else ci_retry_commit

    def push_commit(branch, commit_sha):
        nonlocal push_calls
        push_calls += 1
        return push if push_calls == 1 else ci_retry_push

    def wait_for_retry_ci(saved_retry_push):
        assert saved_retry_push == ci_retry_push
        raise RuntimeError("wait retry CI reached")

    with pytest.raises(
        RuntimeError,
        match="wait retry CI reached",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            repair_base_branch="devpilot-v1",
            workflow_id="issue-21",
            run_agent=lambda prompt: None,
            collect_summary=collect_summary,
            run_validation=run_validation,
            create_commit=create_commit,
            push_commit=push_commit,
            create_pull_request=lambda saved_push: pull_request,
            wait_for_ci=lambda saved_pull_request: ci_status,
            build_ci_failure_prompt=lambda saved_ci_status: (
                "Repair CI failure."
            ),
            wait_for_retry_ci=wait_for_retry_ci,
            save_checkpoint=checkpoints.append,
        )

    assert checkpoints[-1] == issue_orchestration.WorkflowCheckpoint(
        workflow_id="issue-21",
        state=issue_orchestration.WorkflowState.WAIT_RETRY_CI,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        repair_base_branch="devpilot-v1",
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


def test_run_issue_workflow_dry_run_runs_agent_once():
    prompts = []

    result = issue_orchestration.run_issue_workflow(
        issue_prompt="Analyze and repair this Issue.",
        dry_run=True,
        run_agent=prompts.append,
    )

    assert prompts == [
        "Analyze and repair this Issue.",
    ]
    assert result is None


def test_run_issue_workflow_repair_collects_post_repair_summary():
    calls = []

    expected_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )
    expected_validation = PostRepairValidation(
    command=("python", "-m", "pytest", "tests", "-q"),
    status="passed",
    exit_code=0,
    passed_count=360,
    failed_count=0,
    error_count=0,
    output="360 passed",
    )
    expected_commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    expected_push = RepairPush(
    remote="origin",
    branch="devpilot/issue-21-fix-scan-limit",
    commit_sha=expected_commit.sha,
    )
    expected_pull_request = RepairPullRequest(
    repository="example/project",
    number=42,
    url="https://github.com/example/project/pull/42",
    title="Fix #21: Fix repository scan limit",
    base_branch="devpilot-v1",
    head_branch=expected_push.branch,
    commit_sha=expected_push.commit_sha,
    )
    expected_ci_status = RepairCIStatus(
    repository=expected_pull_request.repository,
    commit_sha=expected_pull_request.commit_sha,
    state="success",
    check_runs=(),
    )
    def run_agent(prompt):
        calls.append(("agent", prompt))

    def collect_summary(branch):
        calls.append(("summary", branch))
        return expected_summary

    def run_validation():
        calls.append(("validation", None))
        return expected_validation

    def create_commit():
        calls.append(("commit", None))
        return expected_commit
    def push_commit(branch, commit_sha):
        calls.append(
            ("push", branch, commit_sha)
        )
        return expected_push
    def create_pull_request(push):
        calls.append(("pull_request", push))
        return expected_pull_request
    def wait_for_ci(pull_request):
        calls.append(("ci", pull_request))
        return expected_ci_status

    result = issue_orchestration.run_issue_workflow(
        issue_prompt="Repair this Issue.",
        dry_run=False,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        run_agent=run_agent,
        collect_summary=collect_summary,
        run_validation=run_validation,
        create_commit=create_commit,
        push_commit=push_commit,
        create_pull_request=create_pull_request,
        wait_for_ci=wait_for_ci,
    )

    assert calls == [
        ("agent", "Repair this Issue."),
        (
            "summary",
            "devpilot/issue-21-fix-scan-limit",
        ),
        ("validation", None),
        ("commit", None),
        (
            "push",
            "devpilot/issue-21-fix-scan-limit",
            expected_commit.sha,
        ),
        ("pull_request", expected_push),
        ("ci", expected_pull_request),
    ]
    assert result.summary == expected_summary
    assert result.validation == expected_validation
    assert result.commit == expected_commit
    assert result.push == expected_push
    assert result.pull_request == expected_pull_request
    assert result.ci_status == expected_ci_status
    assert result.state == issue_orchestration.WorkflowState.COMPLETED


def test_run_issue_workflow_repair_skips_validation_without_changes():
    prompts = []

    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(),
    )

    def run_validation():
        pytest.fail(
            "Validation must not run when repair "
            "produced no repository changes"
        )

    result = issue_orchestration.run_issue_workflow(
        issue_prompt="Repair this Issue.",
        dry_run=False,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        run_agent=prompts.append,
        collect_summary=lambda branch: summary,
        run_validation=run_validation,
    )

    assert prompts == [
        "Repair this Issue.",
    ]
    assert result.summary == summary
    assert result.validation is None
    assert (
    result.state
    == issue_orchestration.WorkflowState.COLLECT_SUMMARY
    )


def test_run_issue_workflow_repair_requires_validation_for_changes():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    with pytest.raises(
        ValueError,
        match="run_validation is required for repair changes",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            collect_summary=lambda branch: summary,
        )


def test_run_issue_workflow_repair_returns_validation_result():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )
    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
    remote="origin",
    branch="devpilot/issue-21-fix-scan-limit",
    commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
    repository="example/project",
    number=42,
    url="https://github.com/example/project/pull/42",
    title="Fix #21: Fix repository scan limit",
    base_branch="devpilot-v1",
    head_branch=push.branch,
    commit_sha=push.commit_sha,
    )
    ci_status = RepairCIStatus(
    repository=pull_request.repository,
    commit_sha=pull_request.commit_sha,
    state="success",
    check_runs=(),
    )
    result = issue_orchestration.run_issue_workflow(
    issue_prompt="Repair this Issue.",
    dry_run=False,
    repair_branch="devpilot/issue-21-fix-scan-limit",
    run_agent=lambda prompt: None,
    collect_summary=lambda branch: summary,
    run_validation=lambda: validation,
    create_commit=lambda: commit,
    push_commit=lambda branch, commit_sha: push,
    create_pull_request=lambda received_push: pull_request,
    wait_for_ci=lambda received_pull_request: ci_status,
    )

    assert result.summary == summary
    assert result.validation == validation
    assert result.commit == commit
    assert result.push == push
    assert result.pull_request == pull_request
    assert result.ci_status == ci_status


def test_run_issue_workflow_stops_after_failed_validation():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="failed",
        exit_code=1,
        passed_count=359,
        failed_count=1,
        error_count=0,
        output="1 failed, 359 passed",
    )

    def create_commit():
        pytest.fail(
            "Commit must not run after failed validation"
        )

    result = issue_orchestration.run_issue_workflow(
        issue_prompt="Repair this Issue.",
        dry_run=False,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        run_agent=lambda prompt: None,
        collect_summary=lambda branch: summary,
        run_validation=lambda: validation,
        create_commit=create_commit,
    )

    assert result.summary == summary
    assert result.validation == validation
    assert result.state == issue_orchestration.WorkflowState.VALIDATE


def test_run_issue_workflow_creates_commit_after_passed_validation():
    calls = []

    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )
    push = RepairPush(
    remote="origin",
    branch="devpilot/issue-21-fix-scan-limit",
    commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="example/project",
        number=42,
        url="https://github.com/example/project/pull/42",
        title="Fix #21: Fix repository scan limit",
        base_branch="devpilot-v1",
        head_branch=push.branch,
        commit_sha=push.commit_sha,
    )
    ci_status = RepairCIStatus(
    repository=pull_request.repository,
    commit_sha=pull_request.commit_sha,
    state="success",
    check_runs=(),
    )
    def create_commit():
        calls.append("commit")
        return commit

    result = issue_orchestration.run_issue_workflow(
        issue_prompt="Repair this Issue.",
        dry_run=False,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        run_agent=lambda prompt: None,
        collect_summary=lambda branch: summary,
        run_validation=lambda: validation,
        create_commit=create_commit,
        push_commit=lambda branch, commit_sha: push,
        create_pull_request=lambda received_push: pull_request,
        wait_for_ci=lambda received_pull_request: ci_status,
    )

    assert calls == ["commit"]
    assert result.summary == summary
    assert result.validation == validation
    assert result.commit == commit
    assert result.push == push
    assert result.pull_request == pull_request
    assert result.ci_status == ci_status


def test_run_issue_workflow_requires_commit_after_passed_validation():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    with pytest.raises(
        ValueError,
        match="create_commit is required after passed validation",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            collect_summary=lambda branch: summary,
            run_validation=lambda: validation,
        )

def test_run_issue_workflow_pushes_commit_after_creation():
    calls = []

    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )
    pull_request = RepairPullRequest(
        repository="example/project",
        number=42,
        url="https://github.com/example/project/pull/42",
        title="Fix #21: Fix repository scan limit",
        base_branch="devpilot-v1",
        head_branch=push.branch,
        commit_sha=push.commit_sha,
    )
    ci_status = RepairCIStatus(
    repository=pull_request.repository,
    commit_sha=pull_request.commit_sha,
    state="success",
    check_runs=(),
    )
    def push_commit(branch, commit_sha):
        calls.append((branch, commit_sha))
        return push

    result = issue_orchestration.run_issue_workflow(
        issue_prompt="Repair this Issue.",
        dry_run=False,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        run_agent=lambda prompt: None,
        collect_summary=lambda branch: summary,
        run_validation=lambda: validation,
        create_commit=lambda: commit,
        push_commit=push_commit,
        create_pull_request=lambda received_push: pull_request,
        wait_for_ci=lambda received_pull_request: ci_status,
    )

    assert calls == [
        (
            "devpilot/issue-21-fix-scan-limit",
            commit.sha,
        ),
    ]
    assert result.commit == commit
    assert result.push == push
    assert result.pull_request == pull_request
    assert result.ci_status == ci_status



def test_run_issue_workflow_requires_push_after_commit():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    with pytest.raises(
        ValueError,
        match="push_commit is required after commit creation",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            collect_summary=lambda branch: summary,
            run_validation=lambda: validation,
            create_commit=lambda: commit,
        )


def test_run_issue_workflow_creates_pull_request_after_push():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )

    pull_request = RepairPullRequest(
        repository="example/project",
        number=42,
        url="https://github.com/example/project/pull/42",
        title="Fix #21: Fix repository scan limit",
        base_branch="devpilot-v1",
        head_branch=push.branch,
        commit_sha=push.commit_sha,
    )
    ci_status = RepairCIStatus(
    repository=pull_request.repository,
    commit_sha=pull_request.commit_sha,
    state="success",
    check_runs=(),
    )

    calls = []

    def create_pull_request(received_push):
        calls.append(received_push)
        return pull_request

    result = issue_orchestration.run_issue_workflow(
        issue_prompt="Repair this Issue.",
        dry_run=False,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        run_agent=lambda prompt: None,
        collect_summary=lambda branch: summary,
        run_validation=lambda: validation,
        create_commit=lambda: commit,
        push_commit=lambda branch, commit_sha: push,
        create_pull_request=create_pull_request,
        wait_for_ci=lambda received_pull_request: ci_status,
    )

    assert calls == [push]
    assert result.push == push
    assert result.pull_request == pull_request
    assert result.ci_status == ci_status


def test_run_issue_workflow_requires_pull_request_after_push():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )

    with pytest.raises(
        ValueError,
        match="create_pull_request is required after push",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            collect_summary=lambda branch: summary,
            run_validation=lambda: validation,
            create_commit=lambda: commit,
            push_commit=lambda branch, commit_sha: push,
        )


def test_run_issue_workflow_waits_for_ci_after_pull_request():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )

    pull_request = RepairPullRequest(
        repository="example/project",
        number=42,
        url="https://github.com/example/project/pull/42",
        title="Fix #21: Fix repository scan limit",
        base_branch="devpilot-v1",
        head_branch=push.branch,
        commit_sha=push.commit_sha,
    )

    ci_status = RepairCIStatus(
        repository=pull_request.repository,
        commit_sha=pull_request.commit_sha,
        state="success",
        check_runs=(),
    )

    calls = []

    def wait_for_ci(received_pull_request):
        calls.append(received_pull_request)
        return ci_status

    result = issue_orchestration.run_issue_workflow(
        issue_prompt="Repair this Issue.",
        dry_run=False,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        run_agent=lambda prompt: None,
        collect_summary=lambda branch: summary,
        run_validation=lambda: validation,
        create_commit=lambda: commit,
        push_commit=lambda branch, commit_sha: push,
        create_pull_request=lambda received_push: pull_request,
        wait_for_ci=wait_for_ci,
    )

    assert calls == [pull_request]
    assert result.pull_request == pull_request
    assert result.ci_status == ci_status


def test_run_issue_workflow_requires_ci_after_pull_request():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )

    pull_request = RepairPullRequest(
        repository="example/project",
        number=42,
        url="https://github.com/example/project/pull/42",
        title="Fix #21: Fix repository scan limit",
        base_branch="devpilot-v1",
        head_branch=push.branch,
        commit_sha=push.commit_sha,
    )

    with pytest.raises(
        ValueError,
        match="wait_for_ci is required after pull request creation",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            collect_summary=lambda branch: summary,
            run_validation=lambda: validation,
            create_commit=lambda: commit,
            push_commit=lambda branch, commit_sha: push,
            create_pull_request=lambda received_push: pull_request,
        )


def test_run_issue_workflow_runs_agent_again_after_ci_failure():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )

    pull_request = RepairPullRequest(
        repository="example/project",
        number=42,
        url="https://github.com/example/project/pull/42",
        title="Fix #21: Fix repository scan limit",
        base_branch="devpilot-v1",
        head_branch=push.branch,
        commit_sha=push.commit_sha,
    )

    ci_status = RepairCIStatus(
        repository=pull_request.repository,
        commit_sha=pull_request.commit_sha,
        state="failure",
        check_runs=(),
    )
    retry_ci_status = RepairCIStatus(
        repository=pull_request.repository,
        commit_sha=push.commit_sha,
        state="success",
        check_runs=(),
    )

    prompts = []
    prompt_statuses = []

    def build_ci_failure_prompt(received_ci_status):
        prompt_statuses.append(received_ci_status)
        return "Repair the failed CI."

    result = issue_orchestration.run_issue_workflow(
        issue_prompt="Repair this Issue.",
        dry_run=False,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        run_agent=prompts.append,
        collect_summary=lambda branch: summary,
        run_validation=lambda: validation,
        create_commit=lambda: commit,
        push_commit=lambda branch, commit_sha: push,
        create_pull_request=lambda received_push: pull_request,
        wait_for_ci=lambda received_pull_request: ci_status,
        build_ci_failure_prompt=build_ci_failure_prompt,
        wait_for_retry_ci=lambda received_push: retry_ci_status,
    )

    assert prompt_statuses == [ci_status]
    assert prompts == [
        "Repair this Issue.",
        "Repair the failed CI.",
    ]
    assert result.ci_status == ci_status
    assert result.ci_retry_status == retry_ci_status


def test_run_issue_workflow_requires_failure_prompt_after_ci_failure():
    summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )

    pull_request = RepairPullRequest(
        repository="example/project",
        number=42,
        url="https://github.com/example/project/pull/42",
        title="Fix #21: Fix repository scan limit",
        base_branch="devpilot-v1",
        head_branch=push.branch,
        commit_sha=push.commit_sha,
    )

    ci_status = RepairCIStatus(
        repository=pull_request.repository,
        commit_sha=pull_request.commit_sha,
        state="failure",
        check_runs=(),
    )

    with pytest.raises(
        ValueError,
        match=(
            "build_ci_failure_prompt is required "
            "after CI failure"
        ),
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            collect_summary=lambda branch: summary,
            run_validation=lambda: validation,
            create_commit=lambda: commit,
            push_commit=lambda branch, commit_sha: push,
            create_pull_request=lambda received_push: pull_request,
            wait_for_ci=lambda received_pull_request: ci_status,
        )


def test_run_issue_workflow_collects_summary_after_ci_repair():
    initial_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )

    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )

    pull_request = RepairPullRequest(
        repository="example/project",
        number=42,
        url="https://github.com/example/project/pull/42",
        title="Fix #21: Fix repository scan limit",
        base_branch="devpilot-v1",
        head_branch=push.branch,
        commit_sha=push.commit_sha,
    )

    ci_status = RepairCIStatus(
        repository=pull_request.repository,
        commit_sha=pull_request.commit_sha,
        state="failure",
        check_runs=(),
    )
    retry_ci_status = RepairCIStatus(
    repository=pull_request.repository,
    commit_sha=push.commit_sha,
    state="success",
    check_runs=(),
    )

    summaries = [
        initial_summary,
        retry_summary,
    ]
    summary_calls = []

    def collect_summary(branch):
        summary_calls.append(branch)
        return summaries[len(summary_calls) - 1]

    result = issue_orchestration.run_issue_workflow(
        issue_prompt="Repair this Issue.",
        dry_run=False,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        run_agent=lambda prompt: None,
        collect_summary=collect_summary,
        run_validation=lambda: validation,
        create_commit=lambda: commit,
        push_commit=lambda branch, commit_sha: push,
        create_pull_request=lambda received_push: pull_request,
        wait_for_ci=lambda received_pull_request: ci_status,
        build_ci_failure_prompt=lambda status: (
            "Repair the failed CI."
        ),
        wait_for_retry_ci=lambda received_push: retry_ci_status,
    )

    assert summary_calls == [
        "devpilot/issue-21-fix-scan-limit",
        "devpilot/issue-21-fix-scan-limit",
    ]
    assert result.ci_retry_summary == retry_summary
    assert result.ci_retry_status == retry_ci_status


def test_run_issue_workflow_rejects_ci_repair_without_changes():
    initial_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(),
    )

    validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )

    pull_request = RepairPullRequest(
        repository="example/project",
        number=42,
        url="https://github.com/example/project/pull/42",
        title="Fix #21: Fix repository scan limit",
        base_branch="devpilot-v1",
        head_branch=push.branch,
        commit_sha=push.commit_sha,
    )

    ci_status = RepairCIStatus(
        repository=pull_request.repository,
        commit_sha=pull_request.commit_sha,
        state="failure",
        check_runs=(),
    )

    summaries = [
        initial_summary,
        retry_summary,
    ]
    summary_calls = []

    def collect_summary(branch):
        summary_calls.append(branch)
        return summaries[len(summary_calls) - 1]

    with pytest.raises(
        ValueError,
        match="CI repair produced no repository changes",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            collect_summary=collect_summary,
            run_validation=lambda: validation,
            create_commit=lambda: commit,
            push_commit=lambda branch, commit_sha: push,
            create_pull_request=lambda received_push: pull_request,
            wait_for_ci=lambda received_pull_request: ci_status,
            build_ci_failure_prompt=lambda status: (
                "Repair the failed CI."
            ),
        )


def test_run_issue_workflow_validates_ci_repair_changes():
    initial_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )

    initial_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    retry_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=361,
        failed_count=0,
        error_count=0,
        output="361 passed",
    )

    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )

    pull_request = RepairPullRequest(
        repository="example/project",
        number=42,
        url="https://github.com/example/project/pull/42",
        title="Fix #21: Fix repository scan limit",
        base_branch="devpilot-v1",
        head_branch=push.branch,
        commit_sha=push.commit_sha,
    )

    ci_status = RepairCIStatus(
        repository=pull_request.repository,
        commit_sha=pull_request.commit_sha,
        state="failure",
        check_runs=(),
    )
    retry_ci_status = RepairCIStatus(
        repository=pull_request.repository,
        commit_sha=push.commit_sha,
        state="success",
        check_runs=(),
    )
    summaries = [
        initial_summary,
        retry_summary,
    ]
    summary_calls = []

    def collect_summary(branch):
        summary_calls.append(branch)
        return summaries[len(summary_calls) - 1]

    validations = [
        initial_validation,
        retry_validation,
    ]
    validation_calls = []

    def run_validation():
        validation_calls.append(None)
        return validations[len(validation_calls) - 1]

    result = issue_orchestration.run_issue_workflow(
        issue_prompt="Repair this Issue.",
        dry_run=False,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        run_agent=lambda prompt: None,
        collect_summary=collect_summary,
        run_validation=run_validation,
        create_commit=lambda: commit,
        push_commit=lambda branch, commit_sha: push,
        create_pull_request=lambda received_push: pull_request,
        wait_for_ci=lambda received_pull_request: ci_status,
        build_ci_failure_prompt=lambda status: (
            "Repair the failed CI."
        ),
        wait_for_retry_ci=lambda received_push: retry_ci_status,
    )

    assert validation_calls == [None, None]
    assert result.ci_retry_validation == retry_validation
    assert result.ci_retry_status == retry_ci_status


def test_run_issue_workflow_stops_after_failed_ci_retry_validation():
    initial_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )

    initial_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    retry_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="failed",
        exit_code=1,
        passed_count=359,
        failed_count=1,
        error_count=0,
        output="1 failed, 359 passed",
    )

    commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=commit.sha,
    )

    pull_request = RepairPullRequest(
        repository="example/project",
        number=42,
        url="https://github.com/example/project/pull/42",
        title="Fix #21: Fix repository scan limit",
        base_branch="devpilot-v1",
        head_branch=push.branch,
        commit_sha=push.commit_sha,
    )

    ci_status = RepairCIStatus(
        repository=pull_request.repository,
        commit_sha=pull_request.commit_sha,
        state="failure",
        check_runs=(),
    )

    summaries = [
        initial_summary,
        retry_summary,
    ]
    summary_calls = []

    def collect_summary(branch):
        summary_calls.append(branch)
        return summaries[len(summary_calls) - 1]

    validations = [
        initial_validation,
        retry_validation,
    ]
    validation_calls = []

    def run_validation():
        validation_calls.append(None)
        return validations[len(validation_calls) - 1]

    with pytest.raises(
        ValueError,
        match="CI repair validation failed",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            collect_summary=collect_summary,
            run_validation=run_validation,
            create_commit=lambda: commit,
            push_commit=lambda branch, commit_sha: push,
            create_pull_request=lambda received_push: pull_request,
            wait_for_ci=lambda received_pull_request: ci_status,
            build_ci_failure_prompt=lambda status: (
                "Repair the failed CI."
            ),
        )


def test_run_issue_workflow_creates_commit_after_ci_retry_validation():
    initial_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )

    initial_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    retry_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=361,
        failed_count=0,
        error_count=0,
        output="361 passed",
    )

    initial_commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    retry_commit = RepairCommit(
        sha="fedcba9876543210",
        message="Fix #21: Fix repository scan limit",
    )

    push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=initial_commit.sha,
    )

    pull_request = RepairPullRequest(
        repository="example/project",
        number=42,
        url="https://github.com/example/project/pull/42",
        title="Fix #21: Fix repository scan limit",
        base_branch="devpilot-v1",
        head_branch=push.branch,
        commit_sha=push.commit_sha,
    )

    ci_status = RepairCIStatus(
        repository=pull_request.repository,
        commit_sha=pull_request.commit_sha,
        state="failure",
        check_runs=(),
    )
    retry_ci_status = RepairCIStatus(
        repository=pull_request.repository,
        commit_sha=retry_commit.sha,
        state="success",
        check_runs=(),
    )
    summaries = [
        initial_summary,
        retry_summary,
    ]
    summary_calls = []

    def collect_summary(branch):
        summary_calls.append(branch)
        return summaries[len(summary_calls) - 1]

    validations = [
        initial_validation,
        retry_validation,
    ]
    validation_calls = []

    def run_validation():
        validation_calls.append(None)
        return validations[len(validation_calls) - 1]

    commits = [
        initial_commit,
        retry_commit,
    ]
    commit_calls = []

    def create_commit():
        commit_calls.append(None)
        return commits[len(commit_calls) - 1]

    result = issue_orchestration.run_issue_workflow(
        issue_prompt="Repair this Issue.",
        dry_run=False,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        run_agent=lambda prompt: None,
        collect_summary=collect_summary,
        run_validation=run_validation,
        create_commit=create_commit,
        push_commit=lambda branch, commit_sha: push,
        create_pull_request=lambda received_push: pull_request,
        wait_for_ci=lambda received_pull_request: ci_status,
        build_ci_failure_prompt=lambda status: (
            "Repair the failed CI."
        ),
        wait_for_retry_ci=lambda received_push: retry_ci_status,
    )

    assert commit_calls == [None, None]
    assert result.ci_retry_commit == retry_commit
    assert result.ci_retry_status == retry_ci_status


def test_run_issue_workflow_pushes_ci_retry_commit():
    initial_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )

    initial_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    retry_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=361,
        failed_count=0,
        error_count=0,
        output="361 passed",
    )

    initial_commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    retry_commit = RepairCommit(
        sha="fedcba9876543210",
        message="Fix #21: Fix repository scan limit",
    )

    initial_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=initial_commit.sha,
    )

    retry_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=retry_commit.sha,
    )

    pull_request = RepairPullRequest(
        repository="example/project",
        number=42,
        url="https://github.com/example/project/pull/42",
        title="Fix #21: Fix repository scan limit",
        base_branch="devpilot-v1",
        head_branch=initial_push.branch,
        commit_sha=initial_push.commit_sha,
    )
    retry_ci_status = RepairCIStatus(
            repository=pull_request.repository,
            commit_sha=retry_push.commit_sha,
            state="success",
            check_runs=(),
        )
    ci_status = RepairCIStatus(
        repository=pull_request.repository,
        commit_sha=pull_request.commit_sha,
        state="failure",
        check_runs=(),
    )

    summaries = [
        initial_summary,
        retry_summary,
    ]
    summary_calls = []

    def collect_summary(branch):
        summary_calls.append(branch)
        return summaries[len(summary_calls) - 1]

    validations = [
        initial_validation,
        retry_validation,
    ]
    validation_calls = []

    def run_validation():
        validation_calls.append(None)
        return validations[len(validation_calls) - 1]

    commits = [
        initial_commit,
        retry_commit,
    ]
    commit_calls = []

    def create_commit():
        commit_calls.append(None)
        return commits[len(commit_calls) - 1]

    pushes = [
        initial_push,
        retry_push,
    ]
    push_calls = []

    def push_commit(branch, commit_sha):
        push_calls.append((branch, commit_sha))
        return pushes[len(push_calls) - 1]

    result = issue_orchestration.run_issue_workflow(
        issue_prompt="Repair this Issue.",
        dry_run=False,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        run_agent=lambda prompt: None,
        collect_summary=collect_summary,
        run_validation=run_validation,
        create_commit=create_commit,
        push_commit=push_commit,
        create_pull_request=lambda received_push: pull_request,
        wait_for_ci=lambda received_pull_request: ci_status,
        build_ci_failure_prompt=lambda status: (
            "Repair the failed CI."
        ),
        wait_for_retry_ci=lambda received_push: retry_ci_status,
    )

    assert push_calls == [
        (
            "devpilot/issue-21-fix-scan-limit",
            initial_commit.sha,
        ),
        (
            "devpilot/issue-21-fix-scan-limit",
            retry_commit.sha,
        ),
    ]
    assert result.ci_retry_push == retry_push
    assert result.ci_retry_status == retry_ci_status


def test_run_issue_workflow_waits_for_ci_after_retry_push():
    initial_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )

    initial_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    retry_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=361,
        failed_count=0,
        error_count=0,
        output="361 passed",
    )

    initial_commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    retry_commit = RepairCommit(
        sha="fedcba9876543210",
        message="Fix #21: Fix repository scan limit",
    )

    initial_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=initial_commit.sha,
    )

    retry_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=retry_commit.sha,
    )

    pull_request = RepairPullRequest(
        repository="example/project",
        number=42,
        url="https://github.com/example/project/pull/42",
        title="Fix #21: Fix repository scan limit",
        base_branch="devpilot-v1",
        head_branch=initial_push.branch,
        commit_sha=initial_push.commit_sha,
    )

    initial_ci_status = RepairCIStatus(
    repository=pull_request.repository,
    commit_sha=initial_push.commit_sha,
    state="failure",
    check_runs=(),
    )

    retry_ci_status = RepairCIStatus(
        repository=pull_request.repository,
        commit_sha=retry_push.commit_sha,
        state="success",
        check_runs=(),
    )

    summaries = [
        initial_summary,
        retry_summary,
    ]
    summary_calls = []

    def collect_summary(branch):
        summary_calls.append(branch)
        return summaries[len(summary_calls) - 1]

    validations = [
        initial_validation,
        retry_validation,
    ]
    validation_calls = []

    def run_validation():
        validation_calls.append(None)
        return validations[len(validation_calls) - 1]

    commits = [
        initial_commit,
        retry_commit,
    ]
    commit_calls = []

    def create_commit():
        commit_calls.append(None)
        return commits[len(commit_calls) - 1]

    pushes = [
        initial_push,
        retry_push,
    ]
    push_calls = []

    def push_commit(branch, commit_sha):
        push_calls.append((branch, commit_sha))
        return pushes[len(push_calls) - 1]

    retry_ci_calls = []

    def wait_for_retry_ci(received_push):
        retry_ci_calls.append(received_push)
        return retry_ci_status

    result = issue_orchestration.run_issue_workflow(
        issue_prompt="Repair this Issue.",
        dry_run=False,
        repair_branch="devpilot/issue-21-fix-scan-limit",
        run_agent=lambda prompt: None,
        collect_summary=collect_summary,
        run_validation=run_validation,
        create_commit=create_commit,
        push_commit=push_commit,
        create_pull_request=lambda received_push: pull_request,
        wait_for_ci=lambda received_pull_request: (
            initial_ci_status
        ),
        wait_for_retry_ci=wait_for_retry_ci,
        build_ci_failure_prompt=lambda status: (
            "Repair the failed CI."
        ),
    )

    assert retry_ci_calls == [retry_push]
    assert result.ci_retry_status == retry_ci_status


def test_run_issue_workflow_requires_retry_ci_after_retry_push():
    initial_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )

    initial_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    retry_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=361,
        failed_count=0,
        error_count=0,
        output="361 passed",
    )

    initial_commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    retry_commit = RepairCommit(
        sha="fedcba9876543210",
        message="Fix #21: Fix repository scan limit",
    )

    initial_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=initial_commit.sha,
    )

    retry_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=retry_commit.sha,
    )

    pull_request = RepairPullRequest(
        repository="example/project",
        number=42,
        url="https://github.com/example/project/pull/42",
        title="Fix #21: Fix repository scan limit",
        base_branch="devpilot-v1",
        head_branch=initial_push.branch,
        commit_sha=initial_push.commit_sha,
    )

    ci_status = RepairCIStatus(
        repository=pull_request.repository,
        commit_sha=pull_request.commit_sha,
        state="failure",
        check_runs=(),
    )

    summaries = [
        initial_summary,
        retry_summary,
    ]
    summary_calls = []

    def collect_summary(branch):
        summary_calls.append(branch)
        return summaries[len(summary_calls) - 1]

    validations = [
        initial_validation,
        retry_validation,
    ]
    validation_calls = []

    def run_validation():
        validation_calls.append(None)
        return validations[len(validation_calls) - 1]

    commits = [
        initial_commit,
        retry_commit,
    ]
    commit_calls = []

    def create_commit():
        commit_calls.append(None)
        return commits[len(commit_calls) - 1]

    pushes = [
        initial_push,
        retry_push,
    ]
    push_calls = []

    def push_commit(branch, commit_sha):
        push_calls.append((branch, commit_sha))
        return pushes[len(push_calls) - 1]

    with pytest.raises(
    ValueError,
    match="wait_for_retry_ci is required after CI retry push",
):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            collect_summary=collect_summary,
            run_validation=run_validation,
            create_commit=create_commit,
            push_commit=push_commit,
            create_pull_request=lambda received_push: pull_request,
            wait_for_ci=lambda received_pull_request: ci_status,
            build_ci_failure_prompt=lambda status: (
                "Repair the failed CI."
            ),
        )

    assert push_calls == [
        (
            "devpilot/issue-21-fix-scan-limit",
            initial_commit.sha,
        ),
        (
            "devpilot/issue-21-fix-scan-limit",
            retry_commit.sha,
        ),
    ]


def test_run_issue_workflow_fails_when_ci_retry_fails():
    initial_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/example.py",),
    )

    retry_summary = PostRepairSummary(
        branch="devpilot/issue-21-fix-scan-limit",
        changes=(" M corecoder/ci_fix.py",),
    )

    initial_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=360,
        failed_count=0,
        error_count=0,
        output="360 passed",
    )

    retry_validation = PostRepairValidation(
        command=("python", "-m", "pytest", "tests", "-q"),
        status="passed",
        exit_code=0,
        passed_count=361,
        failed_count=0,
        error_count=0,
        output="361 passed",
    )

    initial_commit = RepairCommit(
        sha="0123456789abcdef",
        message="Fix #21: Fix repository scan limit",
    )

    retry_commit = RepairCommit(
        sha="fedcba9876543210",
        message="Fix #21: Fix repository scan limit",
    )

    initial_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=initial_commit.sha,
    )

    retry_push = RepairPush(
        remote="origin",
        branch="devpilot/issue-21-fix-scan-limit",
        commit_sha=retry_commit.sha,
    )

    pull_request = RepairPullRequest(
        repository="example/project",
        number=42,
        url="https://github.com/example/project/pull/42",
        title="Fix #21: Fix repository scan limit",
        base_branch="devpilot-v1",
        head_branch=initial_push.branch,
        commit_sha=initial_push.commit_sha,
    )

    initial_ci_status = RepairCIStatus(
        repository=pull_request.repository,
        commit_sha=initial_push.commit_sha,
        state="failure",
        check_runs=(),
    )

    retry_ci_status = RepairCIStatus(
        repository=pull_request.repository,
        commit_sha=retry_push.commit_sha,
        state="failure",
        check_runs=(),
    )

    summaries = [
        initial_summary,
        retry_summary,
    ]
    summary_calls = []

    def collect_summary(branch):
        summary_calls.append(branch)
        return summaries[len(summary_calls) - 1]

    validations = [
        initial_validation,
        retry_validation,
    ]
    validation_calls = []

    def run_validation():
        validation_calls.append(None)
        return validations[len(validation_calls) - 1]

    commits = [
        initial_commit,
        retry_commit,
    ]
    commit_calls = []

    def create_commit():
        commit_calls.append(None)
        return commits[len(commit_calls) - 1]

    pushes = [
        initial_push,
        retry_push,
    ]
    push_calls = []

    def push_commit(branch, commit_sha):
        push_calls.append((branch, commit_sha))
        return pushes[len(push_calls) - 1]

    with pytest.raises(
        ValueError,
        match="CI retry failed",
    ):
        issue_orchestration.run_issue_workflow(
            issue_prompt="Repair this Issue.",
            dry_run=False,
            repair_branch="devpilot/issue-21-fix-scan-limit",
            run_agent=lambda prompt: None,
            collect_summary=collect_summary,
            run_validation=run_validation,
            create_commit=create_commit,
            push_commit=push_commit,
            create_pull_request=lambda received_push: pull_request,
            wait_for_ci=lambda received_pull_request: (
                initial_ci_status
            ),
            build_ci_failure_prompt=lambda status: (
                "Repair the failed CI."
            ),
            wait_for_retry_ci=lambda received_push: (
                retry_ci_status
            ),
        )
