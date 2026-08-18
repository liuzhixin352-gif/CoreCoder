"""FastAPI service for CoreCoder."""

from functools import lru_cache
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .agent_service import (
    AgentService,
    create_agent_service,
)


class ChatRequest(BaseModel):
    """Request payload for an agent chat turn."""

    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    """Response payload for an agent chat turn."""

    response: str
    session_id: str
    run_id: str


@lru_cache(maxsize=1)
def get_agent_service() -> AgentService:
    """Return the process-wide agent service."""
    return create_agent_service()


class AgentExecutionError(RuntimeError):
    """Raised when an agent chat execution fails."""


app = FastAPI(
    title="CoreCoder Agent Service",
)


@app.exception_handler(AgentExecutionError)
async def handle_agent_execution_error(
    request: Request,
    exc: AgentExecutionError,
) -> JSONResponse:
    """Map agent execution failures to a stable API error."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "agent_execution_failed",
                "message": "Agent execution failed",
            }
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status."""
    return {
        "status": "ok",
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    service=Depends(get_agent_service),
) -> ChatResponse:
    """Run one agent chat turn."""
    session_id = request.session_id or uuid4().hex
    run_id = uuid4().hex

    try:
        response = service.chat(
            request.message,
            session_id=session_id,
            run_id=run_id,
        )
    except Exception as exc:
        raise AgentExecutionError("Agent execution failed") from exc

    return ChatResponse(
        response=response,
        session_id=session_id,
        run_id=run_id,
    )


def main() -> None:
    """Run the CoreCoder HTTP service."""
    uvicorn.run(
        "corecoder.service:app",
        host="127.0.0.1",
        port=8000,
    )
