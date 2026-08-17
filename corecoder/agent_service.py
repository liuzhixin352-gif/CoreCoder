"""Service boundary for CoreCoder agent execution."""

from collections.abc import Callable
from threading import Lock
from typing import Protocol

from .agent import Agent
from .config import Config
from .llm import LLM, LiteLLM
from .permissions import ToolPermissionPolicy


class ChatAgent(Protocol):
    """Minimal agent interface required by AgentService."""

    def chat(
        self,
        message: str,
        *,
        run_id: str | None = None,
    ) -> str:
        """Run one chat turn."""
        ...


class AgentService:
    """Application service for agent chat execution."""

    def __init__(
        self,
        agent: ChatAgent | None = None,
        *,
        agent_factory: Callable[[], ChatAgent] | None = None,
    ):
        self.agent = agent
        self.agent_factory = agent_factory
        self._agents_by_session: dict[str, ChatAgent] = {}
        self._session_state_lock = Lock()
        self._locks_by_session = {}
        self._agent_lock = Lock()

    def chat(
        self,
        message: str,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> str:
        """Run one chat turn for a session."""
        if self.agent_factory is None:
            if self.agent is None:
                raise RuntimeError("Agent service is not configured")

            with self._agent_lock:
                if run_id is None:
                    return self.agent.chat(message)

                return self.agent.chat(
                    message,
                    run_id=run_id,
                )

        if session_id is None:
            raise ValueError("session_id is required when using agent_factory")

        with self._session_state_lock:
            agent = self._agents_by_session.get(session_id)

            if agent is None:
                agent = self.agent_factory()
                self._agents_by_session[session_id] = agent

            session_lock = self._locks_by_session.get(session_id)

            if session_lock is None:
                session_lock = Lock()
                self._locks_by_session[session_id] = session_lock

        with session_lock:
            if run_id is None:
                return agent.chat(message)

            return agent.chat(
                message,
                run_id=run_id,
            )


def create_agent_service(
    config: Config | None = None,
) -> AgentService:
    """Create the production agent service."""

    resolved_config = config or Config.from_env()

    def agent_factory() -> ChatAgent:
        llm_cls = LiteLLM if resolved_config.provider == "litellm" else LLM

        llm = llm_cls(
            model=resolved_config.model,
            api_key=resolved_config.api_key,
            base_url=resolved_config.base_url,
            temperature=resolved_config.temperature,
            max_tokens=resolved_config.max_tokens,
        )

        return Agent(
            llm=llm,
            max_context_tokens=resolved_config.max_context_tokens,
            permission_policy=ToolPermissionPolicy(),
        )

    return AgentService(
        agent_factory=agent_factory,
    )
