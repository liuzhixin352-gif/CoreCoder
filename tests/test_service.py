"""Tests for the FastAPI agent service."""

from fastapi.testclient import TestClient

from corecoder.service import (
    ChatRequest,
    ChatResponse,
    app,
    get_agent_service,
)
from corecoder.agent_service import (
    AgentService,
    create_agent_service,
)
from corecoder.config import Config
from corecoder.permissions import ToolPermissionPolicy
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

def test_health_endpoint_returns_ok():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }

def test_chat_request_preserves_message_and_session_id():
    request = ChatRequest(
        message="Find the permission implementation",
        session_id="session-123",
    )

    assert request.message == "Find the permission implementation"
    assert request.session_id == "session-123"


def test_chat_response_preserves_response():
    response = ChatResponse(
        response="Permission is implemented in permissions.py",
        session_id="session-123",
        run_id="run-123",
    )

    assert response.response == (
        "Permission is implemented in permissions.py"
    )
    assert response.session_id == "session-123"
    assert response.run_id == "run-123"

def test_chat_endpoint_returns_service_response():
    class FakeAgentService:
        def chat(
            self,
            message: str,
            *,
            session_id: str,
            run_id: str,
        ) -> str:
            assert message == "Find the permission implementation"
            assert session_id == "session-123"
            assert run_id
            return "Permission is implemented in permissions.py"

    app.dependency_overrides[get_agent_service] = (
        lambda: FakeAgentService()
    )

    try:
        client = TestClient(app)

        response = client.post(
            "/chat",
            json={
                "message": "Find the permission implementation",
                "session_id": "session-123",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()

    assert payload["response"] == (
        "Permission is implemented in permissions.py"
    )
    assert payload["session_id"] == "session-123"
    assert payload["run_id"]

def test_agent_service_delegates_chat_to_agent():
    class FakeAgent:
        def __init__(self):
            self.messages = []

        def chat(self, message: str) -> str:
            self.messages.append(message)
            return "agent response"

    agent = FakeAgent()
    service = AgentService(agent)

    result = service.chat("hello")

    assert result == "agent response"
    assert agent.messages == ["hello"]

def test_agent_service_isolates_agents_between_sessions():
    created_agents = []

    class FakeAgent:
        def __init__(self):
            self.messages = []
            created_agents.append(self)

        def chat(self, message: str) -> str:
            self.messages.append(message)
            return message

    service = AgentService(
        agent_factory=FakeAgent,
    )

    first = service.chat(
        "message-a",
        session_id="session-a",
    )
    second = service.chat(
        "message-b",
        session_id="session-b",
    )

    assert first == "message-a"
    assert second == "message-b"

    assert len(created_agents) == 2
    assert created_agents[0].messages == ["message-a"]
    assert created_agents[1].messages == ["message-b"]

def test_agent_service_reuses_agent_within_same_session():
    created_agents = []

    class FakeAgent:
        def __init__(self):
            self.messages = []
            created_agents.append(self)

        def chat(self, message: str) -> str:
            self.messages.append(message)
            return "|".join(self.messages)

    service = AgentService(
        agent_factory=FakeAgent,
    )

    first = service.chat(
        "first",
        session_id="session-a",
    )
    second = service.chat(
        "second",
        session_id="session-a",
    )

    assert first == "first"
    assert second == "first|second"

    assert len(created_agents) == 1
    assert created_agents[0].messages == [
        "first",
        "second",
    ]

def test_chat_endpoint_creates_session_id_when_missing():
    seen_session_ids = []

    class FakeAgentService:
        def chat(
            self,
            message: str,
            *,
            session_id: str,
            run_id: str,
        ) -> str:
            assert message == "hello"
            assert run_id
            seen_session_ids.append(session_id)
            return "agent response"

    app.dependency_overrides[get_agent_service] = (
        lambda: FakeAgentService()
    )

    try:
        client = TestClient(app)

        response = client.post(
            "/chat",
            json={
                "message": "hello",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    payload = response.json()

    assert payload["response"] == "agent response"
    assert payload["session_id"]
    assert payload["run_id"]
    assert seen_session_ids == [
        payload["session_id"],
    ]

def test_get_agent_service_reuses_single_service_instance(
    monkeypatch,
):
    import corecoder.service as service_module

    created_services = []

    class FakeAgentService:
        pass

    def fake_create_agent_service():
        service = FakeAgentService()
        created_services.append(service)
        return service

    monkeypatch.setattr(
        service_module,
        "create_agent_service",
        fake_create_agent_service,
    )

    service_module.get_agent_service.cache_clear()

    try:
        first = service_module.get_agent_service()
        second = service_module.get_agent_service()
    finally:
        service_module.get_agent_service.cache_clear()

    assert first is second
    assert len(created_services) == 1

def test_create_agent_service_builds_agent_from_config(
    monkeypatch,
):
    import corecoder.agent_service as service_module

    created = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            created["llm_kwargs"] = kwargs

    class FakeAgent:
        def __init__(
            self,
            *,
            llm,
            max_context_tokens,
            permission_policy,
        ):
            created["agent_llm"] = llm
            created["max_context_tokens"] = max_context_tokens
            created["permission_policy"] = permission_policy

        def chat(self, message: str) -> str:
            return f"agent:{message}"

    monkeypatch.setattr(
        service_module,
        "LLM",
        FakeLLM,
    )
    monkeypatch.setattr(
        service_module,
        "Agent",
        FakeAgent,
    )

    config = Config(
        model="test-model",
        api_key="test-key",
        base_url="https://example.test/v1",
        max_tokens=321,
        temperature=0.25,
        max_context_tokens=456,
        provider="openai",
    )

    service = create_agent_service(config)

    result = service.chat(
        "hello",
        session_id="session-1",
    )

    assert result == "agent:hello"

    assert created["llm_kwargs"] == {
        "model": "test-model",
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
        "temperature": 0.25,
        "max_tokens": 321,
    }
    assert created["max_context_tokens"] == 456
    assert isinstance(
        created["permission_policy"],
        ToolPermissionPolicy,
    )

def test_create_agent_service_uses_litellm_provider(
    monkeypatch,
):
    import corecoder.agent_service as service_module

    created = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            raise AssertionError("LLM should not be used")

    class FakeLiteLLM:
        def __init__(self, **kwargs):
            created["llm_kwargs"] = kwargs

    class FakeAgent:
        def __init__(
            self,
            *,
            llm,
            max_context_tokens,
            permission_policy,
        ):
            created["agent_llm"] = llm

        def chat(self, message: str) -> str:
            return f"agent:{message}"

    monkeypatch.setattr(
        service_module,
        "LLM",
        FakeLLM,
    )
    monkeypatch.setattr(
        service_module,
        "LiteLLM",
        FakeLiteLLM,
    )
    monkeypatch.setattr(
        service_module,
        "Agent",
        FakeAgent,
    )

    config = Config(
        model="anthropic/test-model",
        api_key="test-key",
        base_url=None,
        max_tokens=321,
        temperature=0.25,
        max_context_tokens=456,
        provider="litellm",
    )

    service = create_agent_service(config)

    result = service.chat(
        "hello",
        session_id="session-1",
    )

    assert result == "agent:hello"

    assert created["llm_kwargs"] == {
        "model": "anthropic/test-model",
        "api_key": "test-key",
        "base_url": None,
        "temperature": 0.25,
        "max_tokens": 321,
    }
    assert isinstance(
        created["agent_llm"],
        FakeLiteLLM,
    )

def test_chat_endpoint_maps_agent_failure_to_structured_500():
    class FailingAgentService:
        def chat(
            self,
            message: str,
            *,
            session_id: str,
            run_id: str,
        ) -> str:
            raise RuntimeError("secret provider failure")

    app.dependency_overrides[get_agent_service] = (
        lambda: FailingAgentService()
    )

    try:
        client = TestClient(
            app,
            raise_server_exceptions=False,
        )

        response = client.post(
            "/chat",
            json={
                "message": "hello",
                "session_id": "session-123",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "agent_execution_failed",
            "message": "Agent execution failed",
        }
    }

def test_chat_endpoint_preserves_validation_errors():
    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = client.post(
        "/chat",
        json={
            "session_id": "session-123",
        },
    )

    assert response.status_code == 422

    payload = response.json()
    assert payload["detail"]

def test_agent_service_forwards_run_id_to_agent():
    received = {}

    class FakeAgent:
        def chat(
            self,
            message: str,
            *,
            run_id: str,
        ) -> str:
            received["message"] = message
            received["run_id"] = run_id
            return "agent response"

    service = AgentService(
        FakeAgent(),
    )

    result = service.chat(
        "hello",
        run_id="run-123",
    )

    assert result == "agent response"
    assert received == {
        "message": "hello",
        "run_id": "run-123",
    }

def test_agent_service_correlates_run_id_with_agent_traces():
    from corecoder.agent import Agent
    from corecoder.llm import LLMResponse
    from corecoder.tracing import InMemoryTracer

    tracer = InMemoryTracer()

    class FakeLLM:
        def chat(
            self,
            messages,
            tools=None,
            on_token=None,
        ):
            return LLMResponse(
                content="agent response",
            )

    def create_agent():
        return Agent(
            llm=FakeLLM(),
            tools=[],
            tracer=tracer,
        )

    service = AgentService(
        agent_factory=create_agent,
    )

    result = service.chat(
        "hello",
        session_id="session-1",
        run_id="run-123",
    )

    assert result == "agent response"

    events = tracer.events_for_run("run-123")

    assert events
    assert {
        event.name
        for event in events
    } >= {
        "agent.started",
        "llm.started",
        "llm.completed",
        "agent.completed",
    }

def test_agent_generates_run_id_when_not_provided():
    from corecoder.agent import Agent
    from corecoder.llm import LLMResponse
    from corecoder.tracing import InMemoryTracer

    tracer = InMemoryTracer()

    class FakeLLM:
        def chat(
            self,
            messages,
            tools=None,
            on_token=None,
        ):
            return LLMResponse(
                content="agent response",
            )

    agent = Agent(
        llm=FakeLLM(),
        tools=[],
        tracer=tracer,
    )

    result = agent.chat("hello")

    assert result == "agent response"

    run_ids = {
        event.attributes["run_id"]
        for event in tracer.events
        if "run_id" in event.attributes
    }

    assert len(run_ids) == 1

    run_id = next(iter(run_ids))

    assert run_id
    assert tracer.events_for_run(run_id)

def test_chat_api_reuses_agent_within_same_session(
    monkeypatch,
):
    import corecoder.service as service_module

    created_agents = []

    class FakeAgent:
        def __init__(self):
            self.messages = []
            created_agents.append(self)

        def chat(
            self,
            message: str,
            *,
            run_id: str | None = None,
        ) -> str:
            self.messages.append(message)
            return "|".join(self.messages)

    def fake_create_agent_service():
        return AgentService(
            agent_factory=FakeAgent,
        )

    monkeypatch.setattr(
        service_module,
        "create_agent_service",
        fake_create_agent_service,
    )

    service_module.get_agent_service.cache_clear()

    try:
        client = TestClient(app)

        first = client.post(
            "/chat",
            json={
                "message": "first",
                "session_id": "session-a",
            },
        )

        second = client.post(
            "/chat",
            json={
                "message": "second",
                "session_id": "session-a",
            },
        )
    finally:
        service_module.get_agent_service.cache_clear()

    assert first.status_code == 200
    assert second.status_code == 200

    first_payload = first.json()
    second_payload = second.json()

    assert first_payload["response"] == "first"
    assert second_payload["response"] == "first|second"

    assert first_payload["session_id"] == "session-a"
    assert second_payload["session_id"] == "session-a"

    assert first_payload["run_id"]
    assert second_payload["run_id"]
    assert first_payload["run_id"] != second_payload["run_id"]

    assert len(created_agents) == 1

def test_chat_api_isolates_different_sessions(
    monkeypatch,
):
    import corecoder.service as service_module

    created_agents = []

    class FakeAgent:
        def __init__(self):
            self.messages = []
            created_agents.append(self)

        def chat(
            self,
            message: str,
            *,
            run_id: str | None = None,
        ) -> str:
            self.messages.append(message)
            return "|".join(self.messages)

    def fake_create_agent_service():
        return AgentService(
            agent_factory=FakeAgent,
        )

    monkeypatch.setattr(
        service_module,
        "create_agent_service",
        fake_create_agent_service,
    )

    service_module.get_agent_service.cache_clear()

    try:
        client = TestClient(app)

        first = client.post(
            "/chat",
            json={
                "message": "message-a",
                "session_id": "session-a",
            },
        )

        second = client.post(
            "/chat",
            json={
                "message": "message-b",
                "session_id": "session-b",
            },
        )
    finally:
        service_module.get_agent_service.cache_clear()

    assert first.status_code == 200
    assert second.status_code == 200

    first_payload = first.json()
    second_payload = second.json()

    assert first_payload["response"] == "message-a"
    assert second_payload["response"] == "message-b"

    assert first_payload["session_id"] == "session-a"
    assert second_payload["session_id"] == "session-b"

    assert first_payload["run_id"]
    assert second_payload["run_id"]
    assert first_payload["run_id"] != second_payload["run_id"]

    assert len(created_agents) == 2

    assert created_agents[0].messages == [
        "message-a",
    ]
    assert created_agents[1].messages == [
        "message-b",
    ]

def test_chat_api_generated_session_can_be_reused(
    monkeypatch,
):
    import corecoder.service as service_module

    created_agents = []

    class FakeAgent:
        def __init__(self):
            self.messages = []
            created_agents.append(self)

        def chat(
            self,
            message: str,
            *,
            run_id: str | None = None,
        ) -> str:
            self.messages.append(message)
            return "|".join(self.messages)

    def fake_create_agent_service():
        return AgentService(
            agent_factory=FakeAgent,
        )

    monkeypatch.setattr(
        service_module,
        "create_agent_service",
        fake_create_agent_service,
    )

    service_module.get_agent_service.cache_clear()

    try:
        client = TestClient(app)

        first = client.post(
            "/chat",
            json={
                "message": "first",
            },
        )

        first_payload = first.json()
        session_id = first_payload["session_id"]

        second = client.post(
            "/chat",
            json={
                "message": "second",
                "session_id": session_id,
            },
        )
    finally:
        service_module.get_agent_service.cache_clear()

    assert first.status_code == 200
    assert second.status_code == 200

    second_payload = second.json()

    assert session_id
    assert first_payload["response"] == "first"
    assert second_payload["response"] == "first|second"

    assert second_payload["session_id"] == session_id

    assert first_payload["run_id"]
    assert second_payload["run_id"]
    assert first_payload["run_id"] != second_payload["run_id"]

    assert len(created_agents) == 1
    assert created_agents[0].messages == [
        "first",
        "second",
    ]

def test_service_main_runs_uvicorn(monkeypatch):
    import corecoder.service as service_module

    captured = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        service_module.uvicorn,
        "run",
        fake_run,
    )

    service_module.main()

    assert captured == {
        "app": "corecoder.service:app",
        "kwargs": {
            "host": "127.0.0.1",
            "port": 8000,
        },
    }

def test_agent_service_serializes_calls_within_same_session():
    first_entered = Event()
    allow_first_to_finish = Event()
    second_entered = Event()

    call_guard = Lock()
    call_count = 0

    class FakeAgent:
        def chat(
            self,
            message: str,
            *,
            run_id: str | None = None,
        ) -> str:
            nonlocal call_count

            with call_guard:
                call_count += 1
                current_call = call_count

            if current_call == 1:
                first_entered.set()
                allow_first_to_finish.wait(timeout=2)
            else:
                second_entered.set()

            return message

    service = AgentService(
        agent_factory=FakeAgent,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            service.chat,
            "first",
            session_id="session-a",
        )

        assert first_entered.wait(timeout=1)

        second = executor.submit(
            service.chat,
            "second",
            session_id="session-a",
        )

        overlapped = second_entered.wait(timeout=0.2)

        allow_first_to_finish.set()

        assert first.result(timeout=1) == "first"
        assert second.result(timeout=1) == "second"

    assert overlapped is False

def test_agent_service_allows_parallel_calls_across_sessions():
    first_entered = Event()
    second_entered = Event()
    release_calls = Event()

    class FakeAgent:
        def chat(
            self,
            message: str,
            *,
            run_id: str | None = None,
        ) -> str:
            if message == "first":
                first_entered.set()
            else:
                second_entered.set()

            release_calls.wait(timeout=2)
            return message

    service = AgentService(
        agent_factory=FakeAgent,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            service.chat,
            "first",
            session_id="session-a",
        )

        assert first_entered.wait(timeout=1)

        second = executor.submit(
            service.chat,
            "second",
            session_id="session-b",
        )

        overlapped = second_entered.wait(timeout=1)

        release_calls.set()

        assert first.result(timeout=1) == "first"
        assert second.result(timeout=1) == "second"

    assert overlapped is True

def test_unrelated_http_failure_is_not_mapped_as_agent_failure():
    route_count = len(app.router.routes)

    def unrelated_failure():
        raise RuntimeError("unrelated failure")

    app.add_api_route(
        "/_test-unrelated-failure",
        unrelated_failure,
        methods=["GET"],
    )

    try:
        client = TestClient(
            app,
            raise_server_exceptions=False,
        )

        response = client.get(
            "/_test-unrelated-failure",
        )
    finally:
        del app.router.routes[route_count:]
        app.openapi_schema = None

    assert response.status_code == 500
    assert response.text == "Internal Server Error"