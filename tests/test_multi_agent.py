"""Tests for Planner/Coder/Reviewer orchestration."""

import pytest

from corecoder.multi_agent import (
    AgentCoder,
    AgentPlanner,
    AgentReviewer,
    MultiAgentOrchestrator,
    MultiAgentRole,
    ReviewOutputError,
    ReviewResult,
    create_multi_agent_orchestrator,
    create_role_agent,
    tools_for_role,
)
from corecoder.permissions import (
    ToolPermission,
    ToolPermissionPolicy,
)
from corecoder.tools.base import Tool
from corecoder.llm import LLMResponse
from corecoder.tracing import InMemoryTracer
from corecoder.budget import (
    BudgetExceededError,
    BudgetLimits,
    BudgetTracker,
)
from corecoder.model_router import (
    RoleModelRouter,
    RouteRequest,
    StaticModelRouter,
)
from corecoder.routed_llm import RoutedLLM

def test_multi_agent_runs_planner_coder_reviewer_in_order():
    calls = []

    class FakePlanner:
        def plan(self, task: str) -> str:
            calls.append(
                ("planner", task),
            )
            return "implementation plan"

    class FakeCoder:
        def code(
            self,
            task: str,
            plan: str,
        ) -> str:
            calls.append(
                ("coder", task, plan),
            )
            return "implementation"

    class FakeReviewer:
        def review(
            self,
            task: str,
            plan: str,
            implementation: str,
        ) -> ReviewResult:
            calls.append(
                (
                    "reviewer",
                    task,
                    plan,
                    implementation,
                )
            )
            return ReviewResult(
                approved=True,
            )

    orchestrator = MultiAgentOrchestrator(
        planner=FakePlanner(),
        coder=FakeCoder(),
        reviewer=FakeReviewer(),
    )

    result = orchestrator.run(
        "Implement permission checks",
    )

    assert result.plan == "implementation plan"
    assert result.implementation == "implementation"
    assert result.review == ReviewResult(
        approved=True,
    )

    assert calls == [
        (
            "planner",
            "Implement permission checks",
        ),
        (
            "coder",
            "Implement permission checks",
            "implementation plan",
        ),
        (
            "reviewer",
            "Implement permission checks",
            "implementation plan",
            "implementation",
        ),
    ]


def test_multi_agent_revises_code_after_reviewer_rejection():
    calls = []

    class FakePlanner:
        def plan(self, task: str) -> str:
            calls.append(
                ("planner", task),
            )
            return "implementation plan"

    class FakeCoder:
        def __init__(self):
            self.attempt = 0

        def code(
            self,
            task: str,
            plan: str,
            *,
            feedback: str | None = None,
        ) -> str:
            self.attempt += 1

            calls.append(
                (
                    "coder",
                    self.attempt,
                    task,
                    plan,
                    feedback,
                )
            )

            return f"implementation-v{self.attempt}"

    class FakeReviewer:
        def __init__(self):
            self.attempt = 0

        def review(
            self,
            task: str,
            plan: str,
            implementation: str,
        ) -> ReviewResult:
            self.attempt += 1

            calls.append(
                (
                    "reviewer",
                    self.attempt,
                    implementation,
                )
            )

            if self.attempt == 1:
                return ReviewResult(
                    approved=False,
                    feedback="Add permission tests",
                )

            return ReviewResult(
                approved=True,
            )

    orchestrator = MultiAgentOrchestrator(
        planner=FakePlanner(),
        coder=FakeCoder(),
        reviewer=FakeReviewer(),
    )

    result = orchestrator.run(
        "Implement permission checks",
    )

    assert result.implementation == "implementation-v2"
    assert result.review == ReviewResult(
        approved=True,
    )

    assert calls == [
        (
            "planner",
            "Implement permission checks",
        ),
        (
            "coder",
            1,
            "Implement permission checks",
            "implementation plan",
            None,
        ),
        (
            "reviewer",
            1,
            "implementation-v1",
        ),
        (
            "coder",
            2,
            "Implement permission checks",
            "implementation plan",
            "Add permission tests",
        ),
        (
            "reviewer",
            2,
            "implementation-v2",
        ),
    ]


def test_multi_agent_stops_after_max_review_rounds():
    coder_feedback = []
    reviewed_implementations = []

    class FakePlanner:
        def plan(self, task: str) -> str:
            return "implementation plan"

    class FakeCoder:
        def __init__(self):
            self.attempt = 0

        def code(
            self,
            task: str,
            plan: str,
            *,
            feedback: str | None = None,
        ) -> str:
            self.attempt += 1
            coder_feedback.append(feedback)
            return f"implementation-v{self.attempt}"

    class FakeReviewer:
        def __init__(self):
            self.attempt = 0

        def review(
            self,
            task: str,
            plan: str,
            implementation: str,
        ) -> ReviewResult:
            self.attempt += 1
            reviewed_implementations.append(
                implementation,
            )
            return ReviewResult(
                approved=False,
                feedback=f"feedback-{self.attempt}",
            )

    orchestrator = MultiAgentOrchestrator(
        planner=FakePlanner(),
        coder=FakeCoder(),
        reviewer=FakeReviewer(),
        max_review_rounds=3,
    )

    result = orchestrator.run(
        "Implement permission checks",
    )

    assert result.implementation == "implementation-v3"
    assert result.review == ReviewResult(
        approved=False,
        feedback="feedback-3",
    )

    assert coder_feedback == [
        None,
        "feedback-1",
        "feedback-2",
    ]
    assert reviewed_implementations == [
        "implementation-v1",
        "implementation-v2",
        "implementation-v3",
    ]


def test_multi_agent_rejects_non_positive_max_review_rounds():
    class FakePlanner:
        def plan(self, task: str) -> str:
            return "plan"

    class FakeCoder:
        def code(
            self,
            task: str,
            plan: str,
            *,
            feedback: str | None = None,
        ) -> str:
            return "implementation"

    class FakeReviewer:
        def review(
            self,
            task: str,
            plan: str,
            implementation: str,
        ) -> ReviewResult:
            return ReviewResult(
                approved=True,
            )

    with pytest.raises(
        ValueError,
        match="max_review_rounds must be at least 1",
    ):
        MultiAgentOrchestrator(
            planner=FakePlanner(),
            coder=FakeCoder(),
            reviewer=FakeReviewer(),
            max_review_rounds=0,
        )


def test_agent_planner_uses_agent_to_create_plan():
    messages = []

    class FakeAgent:
        def chat(self, message: str) -> str:
            messages.append(message)
            return "1. inspect code\n2. implement change\n3. run tests"

    planner = AgentPlanner(
        FakeAgent(),
    )

    result = planner.plan(
        "Implement permission checks",
    )

    assert result == ("1. inspect code\n2. implement change\n3. run tests")

    assert len(messages) == 1

    prompt = messages[0]

    assert "Planner" in prompt
    assert "Implement permission checks" in prompt
    assert "Coder" in prompt


def test_agent_coder_uses_plan_and_reviewer_feedback():
    messages = []

    class FakeAgent:
        def chat(self, message: str) -> str:
            messages.append(message)
            return f"implementation-{len(messages)}"

    coder = AgentCoder(
        FakeAgent(),
    )

    first = coder.code(
        "Implement permission checks",
        "Inspect permissions.py and add tests",
    )

    second = coder.code(
        "Implement permission checks",
        "Inspect permissions.py and add tests",
        feedback="Handle the denied-permission case",
    )

    assert first == "implementation-1"
    assert second == "implementation-2"
    assert len(messages) == 2

    first_prompt = messages[0]

    assert "Coder" in first_prompt
    assert "Implement permission checks" in first_prompt
    assert "Inspect permissions.py and add tests" in first_prompt
    assert "Reviewer feedback:" not in first_prompt

    second_prompt = messages[1]

    assert "Coder" in second_prompt
    assert "Implement permission checks" in second_prompt
    assert "Inspect permissions.py and add tests" in second_prompt
    assert "Reviewer feedback:" in second_prompt
    assert "Handle the denied-permission case" in second_prompt
    assert "Revise" in second_prompt


def test_agent_reviewer_parses_structured_review():
    messages = []
    responses = [
        ('{"approved": false, "feedback": "Add denied-permission tests"}'),
        ('{"approved": true, "feedback": ""}'),
    ]

    class FakeAgent:
        def chat(self, message: str) -> str:
            messages.append(message)
            return responses[len(messages) - 1]

    reviewer = AgentReviewer(
        FakeAgent(),
    )

    rejected = reviewer.review(
        "Implement permission checks",
        "Update permission handling and add tests",
        "initial implementation",
    )

    approved = reviewer.review(
        "Implement permission checks",
        "Update permission handling and add tests",
        "revised implementation",
    )

    assert rejected == ReviewResult(
        approved=False,
        feedback="Add denied-permission tests",
    )

    assert approved == ReviewResult(
        approved=True,
        feedback="",
    )

    assert len(messages) == 2

    first_prompt = messages[0]

    assert "Reviewer" in first_prompt
    assert "Implement permission checks" in first_prompt
    assert "Update permission handling and add tests" in first_prompt
    assert "initial implementation" in first_prompt
    assert '"approved"' in first_prompt
    assert '"feedback"' in first_prompt
    assert "JSON only" in first_prompt


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"approved": false}',
        '{"approved": "yes", "feedback": ""}',
        ('```json\n{"approved": true, "feedback": ""}\n```'),
    ],
)
def test_agent_reviewer_rejects_malformed_output(
    response,
):
    class FakeAgent:
        def chat(self, message: str) -> str:
            return response

    reviewer = AgentReviewer(
        FakeAgent(),
    )

    with pytest.raises(ReviewOutputError):
        reviewer.review(
            "Implement permission checks",
            "Update permission handling",
            "implementation",
        )


def test_tools_for_role_enforces_role_capabilities():
    class FakeTool(Tool):
        def __init__(
            self,
            name: str,
            permission: ToolPermission,
        ):
            self.name = name
            self.permission = permission

        def execute(self):
            return None

    tools = [
        FakeTool(
            "read",
            ToolPermission.READ,
        ),
        FakeTool(
            "write",
            ToolPermission.WRITE,
        ),
        FakeTool(
            "run_tests",
            ToolPermission.EXECUTE,
        ),
        FakeTool(
            "unknown",
            ToolPermission.UNKNOWN,
        ),
    ]

    planner_tools = tools_for_role(
        MultiAgentRole.PLANNER,
        tools,
    )
    coder_tools = tools_for_role(
        MultiAgentRole.CODER,
        tools,
    )
    reviewer_tools = tools_for_role(
        MultiAgentRole.REVIEWER,
        tools,
    )

    assert [tool.name for tool in planner_tools] == [
        "read",
    ]

    assert [tool.name for tool in coder_tools] == [
        "read",
        "write",
        "run_tests",
    ]

    assert [tool.name for tool in reviewer_tools] == [
        "read",
        "run_tests",
    ]


def test_create_role_agent_builds_isolated_role_agents(
    monkeypatch,
):
    import corecoder.multi_agent as multi_agent_module

    created = []

    class FakeAgent:
        def __init__(
            self,
            *,
            llm,
            tools,
            max_context_tokens,
            permission_policy,
            request_tool_approval,
            tracer,
            route_role,
        ):
            self.llm = llm
            self.tools = tools
            self.max_context_tokens = max_context_tokens
            self.permission_policy = permission_policy
            self.messages = []
            self.request_tool_approval = request_tool_approval
            self.tracer = tracer
            self.route_role = route_role
            created.append(self)

    monkeypatch.setattr(
        multi_agent_module,
        "Agent",
        FakeAgent,
    )

    class FakeTool(Tool):
        def __init__(
            self,
            name: str,
            permission: ToolPermission,
        ):
            self.name = name
            self.permission = permission

        def execute(self):
            return None

    tools = [
        FakeTool(
            "read",
            ToolPermission.READ,
        ),
        FakeTool(
            "write",
            ToolPermission.WRITE,
        ),
        FakeTool(
            "run_tests",
            ToolPermission.EXECUTE,
        ),
        FakeTool(
            "unknown",
            ToolPermission.UNKNOWN,
        ),
    ]

    llm = object()
    permission_policy = ToolPermissionPolicy()

    planner = create_role_agent(
        MultiAgentRole.PLANNER,
        llm=llm,
        tools=tools,
        max_context_tokens=1234,
        permission_policy=permission_policy,
    )

    coder = create_role_agent(
        MultiAgentRole.CODER,
        llm=llm,
        tools=tools,
        max_context_tokens=1234,
        permission_policy=permission_policy,
    )

    reviewer = create_role_agent(
        MultiAgentRole.REVIEWER,
        llm=llm,
        tools=tools,
        max_context_tokens=1234,
        permission_policy=permission_policy,
    )

    assert planner is not coder
    assert coder is not reviewer
    assert planner is not reviewer

    assert planner.llm is llm
    assert coder.llm is llm
    assert reviewer.llm is llm

    assert [tool.name for tool in planner.tools] == [
        "read",
    ]

    assert [tool.name for tool in coder.tools] == [
        "read",
        "write",
        "run_tests",
    ]

    assert [tool.name for tool in reviewer.tools] == [
        "read",
        "run_tests",
    ]

    assert planner.max_context_tokens == 1234
    assert coder.max_context_tokens == 1234
    assert reviewer.max_context_tokens == 1234

    assert planner.permission_policy is permission_policy
    assert coder.permission_policy is permission_policy
    assert reviewer.permission_policy is permission_policy

    assert len(created) == 3

    assert planner.route_role == "planner"
    assert coder.route_role == "coder"
    assert reviewer.route_role == "reviewer"


def test_create_multi_agent_orchestrator_wires_role_agents(
    monkeypatch,
):
    import corecoder.multi_agent as multi_agent_module

    created = []

    class FakeRoleAgent:
        def __init__(self, role):
            self.role = role

        def chat(self, message: str) -> str:
            if self.role is MultiAgentRole.PLANNER:
                return "implementation plan"

            if self.role is MultiAgentRole.CODER:
                return "implementation"

            return '{"approved": true, "feedback": ""}'

    def fake_create_role_agent(
        role,
        *,
        llm,
        tools,
        max_context_tokens,
        permission_policy,
        request_tool_approval,
        tracer,
    ):
        created.append(
            {
                "role": role,
                "llm": llm,
                "tools": tools,
                "max_context_tokens": max_context_tokens,
                "permission_policy": permission_policy,
                "request_tool_approval": request_tool_approval,
                "tracer": tracer,
            }
        )

        return FakeRoleAgent(role)

    monkeypatch.setattr(
        multi_agent_module,
        "create_role_agent",
        fake_create_role_agent,
    )

    llm = object()
    tools = []

    orchestrator = create_multi_agent_orchestrator(
        llm=llm,
        tools=tools,
        max_context_tokens=1234,
        max_review_rounds=3,
    )

    result = orchestrator.run(
        "Implement permission checks",
    )

    assert result.plan == "implementation plan"
    assert result.implementation == "implementation"
    assert result.review == ReviewResult(
        approved=True,
        feedback="",
    )

    assert [item["role"] for item in created] == [
        MultiAgentRole.PLANNER,
        MultiAgentRole.CODER,
        MultiAgentRole.REVIEWER,
    ]

    assert all(item["llm"] is llm for item in created)

    assert all(item["tools"] is tools for item in created)

    assert all(item["max_context_tokens"] == 1234 for item in created)

    policies = [item["permission_policy"] for item in created]

    assert isinstance(
        policies[0],
        ToolPermissionPolicy,
    )
    assert policies[0] is policies[1]
    assert policies[1] is policies[2]

    assert orchestrator.max_review_rounds == 3


def test_create_multi_agent_orchestrator_propagates_approval_callback(
    monkeypatch,
):
    import corecoder.multi_agent as multi_agent_module

    callbacks = []

    class FakeRoleAgent:
        def __init__(self, role):
            self.role = role

        def chat(self, message: str) -> str:
            if self.role is MultiAgentRole.PLANNER:
                return "plan"

            if self.role is MultiAgentRole.CODER:
                return "implementation"

            return '{"approved": true, "feedback": ""}'

    def fake_create_role_agent(
        role,
        *,
        llm,
        tools,
        max_context_tokens,
        permission_policy,
        request_tool_approval,
        tracer,
    ):
        callbacks.append(
            request_tool_approval,
        )
        return FakeRoleAgent(role)

    monkeypatch.setattr(
        multi_agent_module,
        "create_role_agent",
        fake_create_role_agent,
    )

    def approve_tool(request):
        return True

    orchestrator = create_multi_agent_orchestrator(
        llm=object(),
        tools=[],
        request_tool_approval=approve_tool,
    )

    result = orchestrator.run("Implement change")

    assert result.review.approved is True

    assert callbacks == [
        approve_tool,
        approve_tool,
        approve_tool,
    ]


def test_role_agent_adds_role_metadata_to_traces():
    tracer = InMemoryTracer()

    class FakeLLM:
        def chat(
            self,
            messages,
            tools=None,
            on_token=None,
        ):
            return LLMResponse(
                content="done",
            )

    agent = create_role_agent(
        MultiAgentRole.PLANNER,
        llm=FakeLLM(),
        tools=[],
        tracer=tracer,
    )

    result = agent.chat(
        "Create an implementation plan",
        run_id="run-123",
    )

    assert result == "done"

    events = tracer.events_for_run(
        "run-123",
    )

    assert events

    assert {event.name for event in events} >= {
        "agent.started",
        "llm.started",
        "llm.completed",
        "agent.completed",
    }

    assert all(event.attributes["role"] == "planner" for event in events)


def test_multi_agent_workflow_traces_all_roles():
    tracer = InMemoryTracer()

    class FakeLLM:
        def chat(
            self,
            messages,
            tools=None,
            on_token=None,
        ):
            prompt = messages[-1]["content"]

            if "You are the Planner" in prompt:
                return LLMResponse(
                    content="implementation plan",
                )

            if "You are the Coder" in prompt:
                return LLMResponse(
                    content="implementation",
                )

            return LLMResponse(
                content=('{"approved": true, "feedback": ""}'),
            )

    orchestrator = create_multi_agent_orchestrator(
        llm=FakeLLM(),
        tools=[],
        tracer=tracer,
    )

    result = orchestrator.run(
        "Implement permission checks",
    )

    assert result.plan == "implementation plan"
    assert result.implementation == "implementation"
    assert result.review == ReviewResult(
        approved=True,
        feedback="",
    )

    roles = {event.attributes.get("role") for event in tracer.events if "role" in event.attributes}

    assert roles == {
        "planner",
        "coder",
        "reviewer",
    }

    for role in roles:
        role_events = [event for event in tracer.events if event.attributes.get("role") == role]

        assert {event.name for event in role_events} >= {
            "agent.started",
            "llm.started",
            "llm.completed",
            "agent.completed",
        }


def test_multi_agent_integration_revises_rejected_implementation():
    tracer = InMemoryTracer()

    class FakeLLM:
        def chat(
            self,
            messages,
            tools=None,
            on_token=None,
        ):
            prompt = messages[-1]["content"]

            if "You are the Planner" in prompt:
                return LLMResponse(
                    content="implementation plan",
                )

            if "You are the Coder" in prompt:
                if "Reviewer feedback:" in prompt:
                    assert "Add denied-permission tests" in prompt

                    return LLMResponse(
                        content="revised implementation",
                    )

                return LLMResponse(
                    content="initial implementation",
                )

            if "initial implementation" in prompt:
                return LLMResponse(
                    content=('{"approved": false, "feedback": "Add denied-permission tests"}'),
                )

            assert "revised implementation" in prompt

            return LLMResponse(
                content=('{"approved": true, "feedback": ""}'),
            )

    orchestrator = create_multi_agent_orchestrator(
        llm=FakeLLM(),
        tools=[],
        tracer=tracer,
        max_review_rounds=2,
    )

    result = orchestrator.run(
        "Implement permission checks",
    )

    assert result.plan == "implementation plan"
    assert result.implementation == "revised implementation"
    assert result.review == ReviewResult(
        approved=True,
        feedback="",
    )

    started_roles = [event.attributes["role"] for event in tracer.events if event.name == "agent.started"]

    assert started_roles == [
        "planner",
        "coder",
        "reviewer",
        "coder",
        "reviewer",
    ]

    run_ids = {event.attributes["run_id"] for event in tracer.events if event.name == "agent.started"}

    assert len(run_ids) == 5


def test_multi_agent_integration_stops_at_max_review_rounds():
    tracer = InMemoryTracer()

    class FakeLLM:
        def __init__(self):
            self.coder_attempt = 0
            self.reviewer_attempt = 0

        def chat(
            self,
            messages,
            tools=None,
            on_token=None,
        ):
            prompt = messages[-1]["content"]

            if "You are the Planner" in prompt:
                return LLMResponse(
                    content="implementation plan",
                )

            if "You are the Coder" in prompt:
                self.coder_attempt += 1

                return LLMResponse(
                    content=(f"implementation-v{self.coder_attempt}"),
                )

            self.reviewer_attempt += 1

            return LLMResponse(
                content=(f'{{"approved": false, "feedback": "feedback-{self.reviewer_attempt}"}}'),
            )

    llm = FakeLLM()

    orchestrator = create_multi_agent_orchestrator(
        llm=llm,
        tools=[],
        tracer=tracer,
        max_review_rounds=3,
    )

    result = orchestrator.run(
        "Implement permission checks",
    )

    assert result.implementation == "implementation-v3"
    assert result.review == ReviewResult(
        approved=False,
        feedback="feedback-3",
    )

    assert llm.coder_attempt == 3
    assert llm.reviewer_attempt == 3

    started_roles = [event.attributes["role"] for event in tracer.events if event.name == "agent.started"]

    assert started_roles == [
        "planner",
        "coder",
        "reviewer",
        "coder",
        "reviewer",
        "coder",
        "reviewer",
    ]

    run_ids = {event.attributes["run_id"] for event in tracer.events if event.name == "agent.started"}

    assert len(run_ids) == 7


def test_reviewer_role_allows_only_safe_execute_tools():
    class FakeTool(Tool):
        def __init__(
            self,
            name: str,
            permission: ToolPermission,
        ):
            self.name = name
            self.permission = permission

        def execute(self):
            return None

    tools = [
        FakeTool(
            "read",
            ToolPermission.READ,
        ),
        FakeTool(
            "run_tests",
            ToolPermission.EXECUTE,
        ),
        FakeTool(
            "bash",
            ToolPermission.EXECUTE,
        ),
        FakeTool(
            "agent",
            ToolPermission.EXECUTE,
        ),
    ]

    reviewer_tools = tools_for_role(
        MultiAgentRole.REVIEWER,
        tools,
    )

    assert [tool.name for tool in reviewer_tools] == [
        "read",
        "run_tests",
    ]


def test_tools_for_role_rejects_unknown_role():
    with pytest.raises(
        ValueError,
        match="Unsupported multi-agent role",
    ):
        tools_for_role(
            "unknown-role",
            [],
        )


def test_create_role_agent_uses_default_permission_policy():
    agent = create_role_agent(
        MultiAgentRole.CODER,
        llm=object(),
        tools=[],
    )

    assert isinstance(
        agent.permission_policy,
        ToolPermissionPolicy,
    )


def test_agent_reviewer_requires_feedback_when_rejected():
    class FakeAgent:
        def chat(self, message: str) -> str:
            return '{"approved": false, "feedback": "   "}'

    reviewer = AgentReviewer(
        FakeAgent(),
    )

    with pytest.raises(
        ReviewOutputError,
        match="feedback must be non-empty",
    ):
        reviewer.review(
            "Implement permission checks",
            "implementation plan",
            "implementation",
        )


def test_multi_agent_workflow_shares_budget_across_roles():
    tracer = InMemoryTracer()

    class FakeLLM:
        model = "gpt-4o-mini"

        def __init__(self):
            self.calls = 0

        def chat(
            self,
            messages,
            tools=None,
            on_token=None,
        ):
            self.calls += 1

            prompt = messages[-1]["content"]

            if "You are the Planner" in prompt:
                return LLMResponse(
                    content="implementation plan",
                    prompt_tokens=4,
                    completion_tokens=1,
                )

            if "You are the Coder" in prompt:
                return LLMResponse(
                    content="implementation",
                    prompt_tokens=4,
                    completion_tokens=1,
                )

            return LLMResponse(
                content=('{"approved": true, "feedback": ""}'),
                prompt_tokens=4,
                completion_tokens=1,
            )

    llm = FakeLLM()

    orchestrator = create_multi_agent_orchestrator(
        llm=llm,
        tools=[],
        tracer=tracer,
        budget_limits=BudgetLimits(
            max_total_tokens=12,
        ),
    )

    with pytest.raises(
        BudgetExceededError,
    ) as exc_info:
        orchestrator.run(
            "Implement permission checks",
        )

    assert exc_info.value.resource == "total_tokens"
    assert llm.calls == 3

    assert orchestrator.last_budget_usage is not None
    assert orchestrator.last_budget_usage.total_tokens == 15

    exceeded_events = [event for event in tracer.events if event.name == "budget.exceeded"]

    assert len(exceeded_events) == 1
    assert exceeded_events[0].attributes["role"] == "reviewer"
    assert exceeded_events[0].attributes["total_tokens"] == 15


def test_multi_agent_workflow_uses_fresh_budget_each_run():
    class FakeLLM:
        model = "gpt-4o-mini"

        def chat(
            self,
            messages,
            tools=None,
            on_token=None,
        ):
            prompt = messages[-1]["content"]

            if "You are the Planner" in prompt:
                content = "implementation plan"
            elif "You are the Coder" in prompt:
                content = "implementation"
            else:
                content = '{"approved": true, "feedback": ""}'

            return LLMResponse(
                content=content,
                prompt_tokens=4,
                completion_tokens=1,
            )

    orchestrator = create_multi_agent_orchestrator(
        llm=FakeLLM(),
        tools=[],
        budget_limits=BudgetLimits(
            max_total_tokens=15,
        ),
    )

    first = orchestrator.run(
        "First task",
    )

    assert first.review.approved is True
    assert orchestrator.last_budget_usage is not None
    assert orchestrator.last_budget_usage.total_tokens == 15

    second = orchestrator.run(
        "Second task",
    )

    assert second.review.approved is True
    assert orchestrator.last_budget_usage is not None
    assert orchestrator.last_budget_usage.total_tokens == 15


def test_multi_agent_workflow_accepts_external_budget_tracker():
    class FakeLLM:
        model = "gpt-4o-mini"

        def chat(
            self,
            messages,
            tools=None,
            on_token=None,
        ):
            prompt = messages[-1]["content"]

            if "You are the Planner" in prompt:
                content = "implementation plan"
            elif "You are the Coder" in prompt:
                content = "implementation"
            else:
                content = '{"approved": true, "feedback": ""}'

            return LLMResponse(
                content=content,
                prompt_tokens=4,
                completion_tokens=1,
            )

    tracker = BudgetTracker(
        BudgetLimits(
            max_total_tokens=15,
        )
    )

    orchestrator = create_multi_agent_orchestrator(
        llm=FakeLLM(),
        tools=[],
        budget_limits=BudgetLimits(
            max_total_tokens=100,
        ),
    )

    result = orchestrator.run(
        "Implement permission checks",
        budget_tracker=tracker,
    )

    assert result.review.approved is True

    assert tracker.usage.total_tokens == 15

    assert orchestrator.last_budget_usage is tracker.usage


def test_multi_agent_revision_loop_shares_workflow_budget():
    class FakeLLM:
        model = "gpt-4o-mini"

        def __init__(self):
            self.calls = 0

        def chat(
            self,
            messages,
            tools=None,
            on_token=None,
        ):
            self.calls += 1

            prompt = messages[-1]["content"]

            if "You are the Planner" in prompt:
                content = "implementation plan"

            elif "You are the Coder" in prompt:
                if "Reviewer feedback:" in prompt:
                    content = "revised implementation"
                else:
                    content = "initial implementation"

            elif "initial implementation" in prompt:
                content = '{"approved": false, "feedback": "Revise it"}'

            else:
                content = '{"approved": true, "feedback": ""}'

            return LLMResponse(
                content=content,
                prompt_tokens=4,
                completion_tokens=1,
            )

    llm = FakeLLM()

    orchestrator = create_multi_agent_orchestrator(
        llm=llm,
        tools=[],
        budget_limits=BudgetLimits(
            max_total_tokens=25,
        ),
        max_review_rounds=2,
    )

    result = orchestrator.run(
        "Implement permission checks",
    )

    assert result.implementation == ("revised implementation")
    assert result.review.approved is True

    assert llm.calls == 5

    assert orchestrator.last_budget_usage is not None
    assert orchestrator.last_budget_usage.total_tokens == 25


def test_multi_agent_stops_following_roles_after_budget_exceeded():
    tracer = InMemoryTracer()

    class FakeLLM:
        model = "gpt-4o-mini"

        def __init__(self):
            self.calls = 0

        def chat(
            self,
            messages,
            tools=None,
            on_token=None,
        ):
            self.calls += 1

            prompt = messages[-1]["content"]

            if "You are the Planner" in prompt:
                return LLMResponse(
                    content="implementation plan",
                    prompt_tokens=4,
                    completion_tokens=1,
                )

            if "You are the Coder" in prompt:
                return LLMResponse(
                    content="implementation",
                    prompt_tokens=6,
                    completion_tokens=1,
                )

            raise AssertionError("Reviewer must not run after the Coder exceeds budget")

    llm = FakeLLM()

    orchestrator = create_multi_agent_orchestrator(
        llm=llm,
        tools=[],
        tracer=tracer,
        budget_limits=BudgetLimits(
            max_total_tokens=10,
        ),
    )

    with pytest.raises(
        BudgetExceededError,
    ) as exc_info:
        orchestrator.run(
            "Implement permission checks",
        )

    assert exc_info.value.resource == "total_tokens"

    assert llm.calls == 2

    assert orchestrator.last_budget_usage is not None
    assert orchestrator.last_budget_usage.total_tokens == 12

    started_roles = [event.attributes["role"] for event in tracer.events if event.name == "agent.started"]

    assert started_roles == [
        "planner",
        "coder",
    ]

def test_multi_agent_routes_shared_llm_by_role():
    class RoleBackend:
        def __init__(
            self,
            model: str,
            response: LLMResponse,
        ):
            self.model = model
            self.response = response
            self.calls = 0

        def chat(
            self,
            messages,
            tools=None,
            on_token=None,
        ):
            del messages
            del tools
            del on_token

            self.calls += 1

            return self.response

    planner_backend = RoleBackend(
        "gpt-4o-mini",
        LLMResponse(
            content="implementation plan",
            prompt_tokens=10,
            completion_tokens=1,
        ),
    )

    coder_backend = RoleBackend(
        "gpt-4o",
        LLMResponse(
            content="implementation",
            prompt_tokens=20,
            completion_tokens=2,
        ),
    )

    reviewer_backend = RoleBackend(
        "gpt-4.1-mini",
        LLMResponse(
            content=(
                '{"approved": true, '
                '"feedback": ""}'
            ),
            prompt_tokens=30,
            completion_tokens=3,
        ),
    )

    routed_llm = RoutedLLM(
        router=RoleModelRouter(
            {
                "planner": StaticModelRouter(
                    [
                        "gpt-4o-mini",
                    ]
                ),
                "coder": StaticModelRouter(
                    [
                        "gpt-4o",
                    ]
                ),
                "reviewer": StaticModelRouter(
                    [
                        "gpt-4.1-mini",
                    ]
                ),
            }
        ),
        backends={
            "gpt-4o-mini": planner_backend,
            "gpt-4o": coder_backend,
            "gpt-4.1-mini": reviewer_backend,
        },
        route_request=RouteRequest(),
    )

    orchestrator = create_multi_agent_orchestrator(
        llm=routed_llm,
        tools=[],
        budget_limits=BudgetLimits(
            max_cost_usd=1.0,
        ),
    )

    result = orchestrator.run(
        "Implement the feature",
    )

    assert result.plan == (
        "implementation plan"
    )
    assert result.implementation == (
        "implementation"
    )
    assert result.review.approved is True

    assert planner_backend.calls == 1
    assert coder_backend.calls == 1
    assert reviewer_backend.calls == 1

    assert (
        routed_llm.route_request.role
        is None
    )

    assert (
        orchestrator.last_budget_usage
        is not None
    )
    assert (
        orchestrator.last_budget_usage.total_tokens
        == 66
    )