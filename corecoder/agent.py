"""Core agent loop.

This is the heart of CoreCoder.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.
"""

import asyncio
import inspect
import time

from collections.abc import Callable
from uuid import uuid4
from .tracing import TraceEvent, Tracer
from .tool_runtime import run_tool_calls
from .llm import LLM
from .tools import ALL_TOOLS
from .tools.base import Tool
from .tools.agent import AgentTool
from .prompt import system_prompt
from .context import ContextManager
from .permissions import (
    PermissionDecision,
    ToolApprovalRequest,
    ToolPermissionPolicy,
)

class Agent:
    def __init__(
    self,
    llm: LLM,
    tools: list[Tool] | None = None,
    max_context_tokens: int = 128_000,
    max_rounds: int = 50,
    tool_timeout: float | None = None,
    tracer: Tracer | None = None,
    permission_policy: ToolPermissionPolicy | None = None,
    request_tool_approval: (
        Callable[[ToolApprovalRequest], bool] | None
    ) = None,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self._tool_by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self.tool_timeout = tool_timeout
        self.tracer = tracer
        self.permission_policy = permission_policy
        self.request_tool_approval = request_tool_approval
        self._active_run_id: str | None = None
        self._system = system_prompt(self.tools)


        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        messages = [
            {
                key: value
                for key, value in message.items()
                if key != "context_priority"
            }
            for message in self.messages
        ]

        return [{"role": "system", "content": self._system}] + messages

    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    def chat(
        self,
        user_input: str,
        on_token=None,
        on_tool=None,
        *,
        run_id: str | None = None,
    ) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
        self._active_run_id = run_id or uuid4().hex
        agent_started_at = time.perf_counter()

        if self.tracer is not None:
            self._emit_trace(
                TraceEvent(
                    name="agent.started",
                    timestamp=time.time(),
                )
            )
        self.messages.append({"role": "user", "content": user_input})
        self.context.maybe_compress(self.messages, self.llm)
        self._emit_context_trace("pre_llm")

        for round_number in range(1, self.max_rounds + 1):
            llm_started_at = time.perf_counter()

            if self.tracer is not None:
                self._emit_trace(
                    TraceEvent(
                        name="llm.started",
                        timestamp=time.time(),
                        attributes={
                            "round": round_number,
                        },
                    )
                )

            try:
                resp = self.llm.chat(
                    messages=self._full_messages(),
                    tools=self._tool_schemas(),
                    on_token=on_token,
                )
            except Exception as e:
                if self.tracer is not None:
                    self._emit_trace(
                        TraceEvent(
                            name="llm.failed",
                            timestamp=time.time(),
                            attributes={
                                "round": round_number,
                                "error_type": type(e).__name__,
                                "duration_ms": (
                                    time.perf_counter() - llm_started_at
                                )
                                * 1000,
                            },
                        )
                    )

                    self._emit_trace(
                                    TraceEvent(
                                        name="agent.failed",
                                        timestamp=time.time(),
                                        attributes={
                                         "error_type": type(e).__name__,
                                        "llm_rounds": round_number,
                                         "duration_ms": (
                                        time.perf_counter() - agent_started_at
                                         )
                                         * 1000,
                                         },
                                                    )
                                                )
                self._active_run_id = None
                raise

            if self.tracer is not None:
                self._emit_trace(
                    TraceEvent(
                        name="llm.completed",
                        timestamp=time.time(),
                        attributes={
                            "round": round_number,
                            "duration_ms": (
                                time.perf_counter() - llm_started_at
                            )
                            * 1000,
                        },
                    )
                )

            # no tool calls -> LLM is done, return text
            if not resp.tool_calls:
                self.messages.append(resp.message)

                if self.tracer is not None:
                    self._emit_trace(
                        TraceEvent(
                            name="agent.completed",
                            timestamp=time.time(),
                            attributes={
                                "duration_ms": (
                                    time.perf_counter() - agent_started_at
                                )
                                * 1000,
                                "llm_rounds": round_number,
                            },
                        )
                    )

                self._active_run_id = None
                return resp.content

            # tool calls -> execute (parallel when multiple, like Claude Code's
            # StreamingToolExecutor which runs independent tools concurrently)
            self.messages.append(resp.message)

            try:
                results = self._exec_tools_parallel(
                    resp.tool_calls,
                    on_tool,
                )

                for tc, result in zip(resp.tool_calls, results):
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }

                    tool = self._tool_by_name.get(tc.name)
                    if tool is not None and tool.context_priority is not None:
                        tool_message["context_priority"] = tool.context_priority

                    self.messages.append(tool_message)
            except KeyboardInterrupt:
                # Ctrl+C mid-execution would leave the assistant tool_calls
                # message without replies, poisoning the next request; backfill
                self._answer_pending_tool_calls(resp.tool_calls)
                self._active_run_id = None
                raise

            # compress if tool outputs are big
            self.context.maybe_compress(self.messages, self.llm)
            self._emit_context_trace("post_tool")

        if self.tracer is not None:
            self._emit_trace(
                TraceEvent(
                    name="agent.completed",
                    timestamp=time.time(),
                    attributes={
                        "duration_ms": (
                            time.perf_counter() - agent_started_at
                        )
                        * 1000,
                        "llm_rounds": self.max_rounds,
                    },
                )
            )
        self._active_run_id = None
        return "(reached maximum tool-call rounds)"

    def _permission_error(
        self,
        tool: Tool,
        tool_name: str,
        arguments: dict,
        tool_call_id: str,
    ) -> str | None:
        if self.permission_policy is None:
            return None

        decision = self.permission_policy.evaluate(tool.permission)

        approval = None

        if decision is PermissionDecision.ASK:
            if self.request_tool_approval is None:
                approval = "required"
            else:
                request = ToolApprovalRequest(
                    tool_name=tool_name,
                    permission=tool.permission,
                    arguments=dict(arguments),
                )

                if self.request_tool_approval(request):
                    approval = "approved"
                else:
                    approval = "rejected"

        if self.tracer is not None:
            self._emit_trace(
                TraceEvent(
                    name="tool.permission",
                    timestamp=time.time(),
                    attributes={
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "permission": tool.permission.value,
                        "decision": decision.value,
                        "approval": approval,
                    },
                )
            )
        if decision is PermissionDecision.ASK:
            if approval == "required":
                return f"Error: approval required for tool '{tool_name}'"

            if approval == "rejected":
                return f"Error: approval rejected for tool '{tool_name}'"

        if decision is PermissionDecision.DENY:
            return f"Error: permission denied for tool '{tool_name}'"

        return None

    def _exec_tool(self, tc) -> str:
        """Execute a single tool call, returning the result string."""
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            return f"Error: unknown tool '{tc.name}'"
        # validate arguments first so a TypeError raised *inside* the tool isn't
        # mislabelled as a bad-arguments error from the caller
        try:
            inspect.signature(tool.execute).bind(**tc.arguments)
        except TypeError as e:
            return f"Error: bad arguments for {tc.name}: {e}"
        permission_error = self._permission_error(
            tool,
            tc.name,
            tc.arguments,
            tc.id,
        )
        if permission_error is not None:
            return permission_error
        try:
            return tool.execute(**tc.arguments)
        except Exception as e:
            return f"Error executing {tc.name}: {e}"

    async def _exec_tool_async(self, tc) -> str:
        """Execute a single tool call asynchronously."""
        started_at = time.perf_counter()
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            return f"Error: unknown tool '{tc.name}'"

        try:
            inspect.signature(tool.execute).bind(**tc.arguments)
        except TypeError as e:
            return f"Error: bad arguments for {tc.name}: {e}"
        permission_error = self._permission_error(
            tool,
            tc.name,
            tc.arguments,
            tc.id,
        )
        if permission_error is not None:
            return permission_error

        if self.tracer is not None:
            self._emit_trace(
                TraceEvent(
                    name="tool.started",
                    timestamp=time.time(),
                    attributes={
                        "tool_name": tc.name,
                        "tool_call_id": tc.id,
                    },
                )
            )

        try:
            if self.tool_timeout is None:
                result = await tool.aexecute(**tc.arguments)
            else:
                result = await asyncio.wait_for(
                    tool.aexecute(**tc.arguments),
                    timeout=self.tool_timeout,
                )
        except asyncio.TimeoutError:
            if self.tracer is not None:
                self._emit_trace(
                    TraceEvent(
                        name="tool.timed_out",
                        timestamp=time.time(),
                        attributes={
                            "tool_name": tc.name,
                            "tool_call_id": tc.id,
                            "timeout_seconds": self.tool_timeout,
                            "duration_ms": (
                                time.perf_counter() - started_at
                            )
                            * 1000,
                        },
                    )
                )

            return (
                f"Error executing {tc.name}: "
                f"timed out after {self.tool_timeout} seconds"
            )
        except Exception as e:
            if self.tracer is not None:
                self._emit_trace(
                    TraceEvent(
                        name="tool.failed",
                        timestamp=time.time(),
                        attributes={
                            "tool_name": tc.name,
                            "tool_call_id": tc.id,
                            "error_type": type(e).__name__,
                            "duration_ms": (
                                time.perf_counter() - started_at
                            )
                            * 1000,
                        },
                    )
                )

            return f"Error executing {tc.name}: {e}"

        if self.tracer is not None:
            self._emit_trace(
                TraceEvent(
                    name="tool.completed",
                    timestamp=time.time(),
                    attributes={
                        "tool_name": tc.name,
                        "tool_call_id": tc.id,
                        "duration_ms": (
                            time.perf_counter() - started_at
                        )
                        * 1000,
                    },
                )
            )

        return result

    def _exec_tools_parallel(self, tool_calls, on_tool=None) -> list[str]:
        """Run tool calls concurrently through the async runtime.

        This is inspired by Claude Code's StreamingToolExecutor which starts
        executing tools while the model is still generating.  We simplify to:
        when the model returns N tool calls at once, run them in parallel.
        """
        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        return asyncio.run(
            run_tool_calls(
                tool_calls,
                self._exec_tool_async,
                max_concurrency=8,
            )
        )

    def _answer_pending_tool_calls(self, tool_calls):
        """Backfill a tool reply for every call that didn't get one.

        OpenAI-compatible APIs reject a request where an assistant message has
        tool_calls without a matching tool reply for each id, so this keeps the
        history valid when execution is interrupted partway through.
        """
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for tc in tool_calls:
            if tc.id not in answered:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "[interrupted]",
                })

    def reset(self):
        """Clear conversation history."""
        self.messages.clear()

    def _emit_context_trace(self, phase: str) -> None:
        metrics = self.context.last_metrics
        if metrics is None:
            return

        self._emit_trace(
            TraceEvent(
                name="context.managed",
                timestamp=time.time(),
                attributes={
                    "phase": phase,
                    "tokens_before": metrics.tokens_before,
                    "tokens_after": metrics.tokens_after,
                    "tokens_saved": metrics.tokens_saved,
                    "applied_layers": metrics.applied_layers,
                    "high_priority_messages": (
                        metrics.high_priority_messages
                    ),
                    "priority_preserved_messages": (
                        metrics.priority_preserved_messages
                    ),
                },
            )
        )

    def _emit_trace(self, event: TraceEvent) -> None:
        """Emit a trace event without affecting agent execution."""
        if self.tracer is None:
            return

        if self._active_run_id is not None:
            event = TraceEvent(
                name=event.name,
                timestamp=event.timestamp,
                attributes={
                    **event.attributes,
                    "run_id": self._active_run_id,
                },
            )

        try:
            self.tracer.emit(event)
        except Exception:
            pass
