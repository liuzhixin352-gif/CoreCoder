"""Planner/Coder/Reviewer multi-agent orchestration."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .agent import Agent
from .llm import LLM
from .permissions import (
    ToolApprovalRequest,
    ToolPermission,
    ToolPermissionPolicy,
)
from .tools.base import Tool
from .tracing import TraceEvent, Tracer
from .budget import (
    BudgetLimits,
    BudgetTracker,
    BudgetUsage,
)


class MultiAgentRole(str, Enum):
    """Roles in the Planner/Coder/Reviewer workflow."""

    PLANNER = "planner"
    CODER = "coder"
    REVIEWER = "reviewer"


_REVIEWER_EXECUTE_TOOLS = {
    "run_tests",
}


class RoleTracer(Tracer):
    """Add multi-agent role metadata to trace events."""

    def __init__(
        self,
        role: MultiAgentRole,
        tracer: Tracer,
    ):
        self.role = role
        self.tracer = tracer

    def emit(self, event: TraceEvent) -> None:
        """Forward a trace event with role metadata."""
        self.tracer.emit(
            TraceEvent(
                name=event.name,
                timestamp=event.timestamp,
                attributes={
                    **event.attributes,
                    "role": self.role.value,
                },
            )
        )


def tools_for_role(
    role: MultiAgentRole,
    tools: list[Tool],
) -> list[Tool]:
    """Return the tools allowed for one multi-agent role."""
    if role is MultiAgentRole.PLANNER:
        return [tool for tool in tools if tool.permission is ToolPermission.READ]

    if role is MultiAgentRole.CODER:
        return [
            tool
            for tool in tools
            if tool.permission
            in {
                ToolPermission.READ,
                ToolPermission.WRITE,
                ToolPermission.EXECUTE,
            }
        ]

    if role is MultiAgentRole.REVIEWER:
        return [
            tool
            for tool in tools
            if (
                tool.permission is ToolPermission.READ
                or (tool.permission is ToolPermission.EXECUTE and tool.name in _REVIEWER_EXECUTE_TOOLS)
            )
        ]

    raise ValueError(f"Unsupported multi-agent role: {role!r}")


def create_role_agent(
    role: MultiAgentRole,
    *,
    llm: LLM,
    tools: list[Tool],
    max_context_tokens: int = 128_000,
    permission_policy: ToolPermissionPolicy | None = None,
    request_tool_approval: (Callable[[ToolApprovalRequest], bool] | None) = None,
    tracer: Tracer | None = None,
) -> Agent:
    """Create an isolated CoreCoder agent for one workflow role."""

    resolved_permission_policy = permission_policy if permission_policy is not None else ToolPermissionPolicy()

    return Agent(
        llm=llm,
        tools=tools_for_role(
            role,
            tools,
        ),
        max_context_tokens=max_context_tokens,
        permission_policy=resolved_permission_policy,
        request_tool_approval=request_tool_approval,
        tracer=(
            RoleTracer(
                role,
                tracer,
            )
            if tracer is not None
            else None
        ),
        route_role=role.value,
    )


class RoleAgent(Protocol):
    """Minimal agent interface required by role adapters."""

    def chat(
        self,
        message: str,
        *,
        budget_tracker: BudgetTracker | None = None,
    ) -> str:
        """Run one role-specific agent turn."""
        ...


class Planner(Protocol):
    """Produce an implementation plan for a task."""

    def plan(
        self,
        task: str,
        *,
        budget_tracker: BudgetTracker | None = None,
    ) -> str:
        """Return a plan for the task."""
        ...


class AgentPlanner:
    """Planner role backed by a CoreCoder-compatible agent."""

    def __init__(self, agent: RoleAgent):
        self.agent = agent

    def plan(
        self,
        task: str,
        *,
        budget_tracker: BudgetTracker | None = None,
    ) -> str:
        """Ask the backing agent to produce an implementation plan."""
        prompt = (
            "You are the Planner in a software engineering "
            "multi-agent workflow.\n"
            "Analyze the task and produce a concise implementation "
            "plan for the Coder.\n\n"
            f"Task:\n{task}"
        )

        if budget_tracker is None:
            return self.agent.chat(prompt)

        return self.agent.chat(
            prompt,
            budget_tracker=budget_tracker,
        )


class Coder(Protocol):
    """Implement a task from a plan."""

    def code(
        self,
        task: str,
        plan: str,
        *,
        feedback: str | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> str:
        """Return the implementation result."""
        ...


class AgentCoder:
    """Coder role backed by a CoreCoder-compatible agent."""

    def __init__(self, agent: RoleAgent):
        self.agent = agent

    def code(
        self,
        task: str,
        plan: str,
        *,
        feedback: str | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> str:
        """Ask the backing agent to implement the task."""
        prompt = (
            "You are the Coder in a software engineering "
            "multi-agent workflow.\n"
            "Implement the task by following the Planner's plan.\n\n"
            f"Task:\n{task}\n\n"
            f"Plan:\n{plan}"
        )

        if feedback:
            prompt += f"\n\nReviewer feedback:\n{feedback}\n\nRevise the implementation to address this feedback."

        if budget_tracker is None:
            return self.agent.chat(prompt)

        return self.agent.chat(
            prompt,
            budget_tracker=budget_tracker,
        )


class ReviewOutputError(ValueError):
    """Raised when reviewer output violates the structured contract."""


@dataclass(frozen=True)
class ReviewResult:
    """Result of reviewing an implementation."""

    approved: bool
    feedback: str = ""


class Reviewer(Protocol):
    """Review an implementation against its task and plan."""

    def review(
        self,
        task: str,
        plan: str,
        implementation: str,
        *,
        budget_tracker: BudgetTracker | None = None,
    ) -> ReviewResult:
        """Return the review result."""
        ...


class AgentReviewer:
    """Reviewer role backed by a CoreCoder-compatible agent."""

    def __init__(self, agent: RoleAgent):
        self.agent = agent

    def review(
        self,
        task: str,
        plan: str,
        implementation: str,
        *,
        budget_tracker: BudgetTracker | None = None,
    ) -> ReviewResult:
        """Ask the backing agent to review an implementation."""
        prompt = (
            "You are the Reviewer in a software engineering "
            "multi-agent workflow.\n"
            "Review the implementation against the task and plan.\n"
            "Return JSON only, with exactly this shape:\n"
            '{"approved": true, "feedback": ""}\n'
            "Set approved to false and explain required changes in "
            "feedback when the implementation needs revision.\n\n"
            f"Task:\n{task}\n\n"
            f"Plan:\n{plan}\n\n"
            f"Implementation:\n{implementation}"
        )

        if budget_tracker is None:
            raw_response = self.agent.chat(prompt)
        else:
            raw_response = self.agent.chat(
                prompt,
                budget_tracker=budget_tracker,
            )

        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ReviewOutputError("Reviewer output must be valid JSON") from exc

        if not isinstance(payload, dict):
            raise ReviewOutputError("Reviewer output must be a JSON object")

        if set(payload) != {
            "approved",
            "feedback",
        }:
            raise ReviewOutputError("Reviewer output must contain exactly 'approved' and 'feedback'")

        approved = payload["approved"]
        feedback = payload["feedback"]

        if not isinstance(approved, bool):
            raise ReviewOutputError("Reviewer 'approved' must be a boolean")

        if not isinstance(feedback, str):
            raise ReviewOutputError("Reviewer 'feedback' must be a string")

        if not approved and not feedback.strip():
            raise ReviewOutputError("Reviewer feedback must be non-empty when approval is rejected")

        return ReviewResult(
            approved=approved,
            feedback=feedback,
        )


@dataclass(frozen=True)
class MultiAgentResult:
    """Result of one multi-agent workflow."""

    plan: str
    implementation: str
    review: ReviewResult


class MultiAgentOrchestrator:
    """Coordinate Planner, Coder, and Reviewer roles."""

    def __init__(
        self,
        *,
        planner: Planner,
        coder: Coder,
        reviewer: Reviewer,
        max_review_rounds: int = 2,
        budget_limits: BudgetLimits | None = None,
    ):
        if max_review_rounds < 1:
            raise ValueError("max_review_rounds must be at least 1")

        self.planner = planner
        self.coder = coder
        self.reviewer = reviewer
        self.max_review_rounds = max_review_rounds
        self.budget_limits = budget_limits
        self.last_budget_usage: BudgetUsage | None = None

    def run(
        self,
        task: str,
        *,
        budget_tracker: BudgetTracker | None = None,
    ) -> MultiAgentResult:
        """Run one Planner -> Coder -> Reviewer workflow."""
        resolved_budget_tracker = budget_tracker

        if resolved_budget_tracker is None and self.budget_limits is not None:
            resolved_budget_tracker = BudgetTracker(
                self.budget_limits,
            )

        self.last_budget_usage = (
            resolved_budget_tracker.usage
            if resolved_budget_tracker is not None
            else None
        )

        if resolved_budget_tracker is None:
            plan = self.planner.plan(task)
        else:
            plan = self.planner.plan(
                task,
                budget_tracker=resolved_budget_tracker,
            )

        if resolved_budget_tracker is None:
            implementation = self.coder.code(
                task,
                plan,
            )
        else:
            implementation = self.coder.code(
                task,
                plan,
                budget_tracker=resolved_budget_tracker,
            )

        if resolved_budget_tracker is None:
            review = self.reviewer.review(
                task,
                plan,
                implementation,
            )
        else:
            review = self.reviewer.review(
                task,
                plan,
                implementation,
                budget_tracker=resolved_budget_tracker,
            )

        for _ in range(1, self.max_review_rounds):
            if review.approved:
                break

            if resolved_budget_tracker is None:
                implementation = self.coder.code(
                    task,
                    plan,
                    feedback=review.feedback,
                )
            else:
                implementation = self.coder.code(
                    task,
                    plan,
                    feedback=review.feedback,
                    budget_tracker=resolved_budget_tracker,
                )

            if resolved_budget_tracker is None:
                review = self.reviewer.review(
                    task,
                    plan,
                    implementation,
                )
            else:
                review = self.reviewer.review(
                    task,
                    plan,
                    implementation,
                    budget_tracker=resolved_budget_tracker,
                )

        return MultiAgentResult(
            plan=plan,
            implementation=implementation,
            review=review,
        )


def create_multi_agent_orchestrator(
    *,
    llm: LLM,
    tools: list[Tool],
    max_context_tokens: int = 128_000,
    permission_policy: ToolPermissionPolicy | None = None,
    request_tool_approval: (Callable[[ToolApprovalRequest], bool] | None) = None,
    max_review_rounds: int = 2,
    budget_limits: BudgetLimits | None = None,
    tracer: Tracer | None = None,
) -> MultiAgentOrchestrator:
    """Create a production Planner/Coder/Reviewer workflow."""
    resolved_permission_policy = permission_policy if permission_policy is not None else ToolPermissionPolicy()

    planner_agent = create_role_agent(
        MultiAgentRole.PLANNER,
        llm=llm,
        tools=tools,
        max_context_tokens=max_context_tokens,
        permission_policy=resolved_permission_policy,
        request_tool_approval=request_tool_approval,
        tracer=tracer,
    )

    coder_agent = create_role_agent(
        MultiAgentRole.CODER,
        llm=llm,
        tools=tools,
        max_context_tokens=max_context_tokens,
        permission_policy=resolved_permission_policy,
        request_tool_approval=request_tool_approval,
        tracer=tracer,
    )

    reviewer_agent = create_role_agent(
        MultiAgentRole.REVIEWER,
        llm=llm,
        tools=tools,
        max_context_tokens=max_context_tokens,
        permission_policy=resolved_permission_policy,
        request_tool_approval=request_tool_approval,
        tracer=tracer,
    )

    return MultiAgentOrchestrator(
        planner=AgentPlanner(planner_agent),
        coder=AgentCoder(coder_agent),
        reviewer=AgentReviewer(reviewer_agent),
        max_review_rounds=max_review_rounds,
        budget_limits=budget_limits,
    )
