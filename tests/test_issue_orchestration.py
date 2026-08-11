"""Tests for DevPilot Issue workflow orchestration."""
import pytest
from corecoder import issue_orchestration
from corecoder.post_repair import PostRepairSummary
from corecoder.post_repair_validation import PostRepairValidation
from corecoder.repair_commit import RepairCommit
from corecoder.repair_push import RepairPush
from corecoder.repair_pr import RepairPullRequest
from corecoder.repair_ci import RepairCIStatus

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