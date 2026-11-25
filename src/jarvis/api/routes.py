"""HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from jarvis.ai.brain import Brain
from jarvis.api.schemas import ChatRequest, ChatResponse, HealthResponse

router = APIRouter()


def get_brain(request: Request) -> Brain:
    return request.app.state.brain


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    status = await request.app.state.client.health()
    healthy = status.reachable and status.model_available
    return HealthResponse(
        status="ok" if healthy else "degraded",
        ollama_reachable=status.reachable,
        model=status.model,
        model_available=status.model_available,
        detail=status.detail,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, brain: Brain = Depends(get_brain)) -> ChatResponse:
    reply = await brain.respond(payload.message)
    return ChatResponse(reply=reply, history_messages=brain.history_length)


@router.post("/reset", status_code=204)
async def reset(brain: Brain = Depends(get_brain)) -> Response:
    brain.reset()
    return Response(status_code=204)
