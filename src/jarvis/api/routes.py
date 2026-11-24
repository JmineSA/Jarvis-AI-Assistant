"""HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request, Response

from jarvis.ai.brain import Brain
from jarvis.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthCheckModel,
    HealthResponse,
    MemoryHitModel,
    MemorySearchResponse,
    RememberRequest,
    StatusResponse,
)
from jarvis.monitor.health import collect_health

router = APIRouter()


def get_brain(request: Request) -> Brain:
    return request.app.state.brain


def get_memory(request: Request):
    """The memory store, or None when memory is switched off."""
    return getattr(request.app.state, "memory", None)


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


# --------------------------------------------------------------------- status


@router.get("/status", response_model=StatusResponse)
async def status(request: Request, brain: Brain = Depends(get_brain)) -> StatusResponse:
    """The health of every part of JARVIS, plus a reading of the machine."""
    settings = request.app.state.settings
    report = await collect_health(
        settings,
        request.app.state.client,
        brain=brain,
        memory=getattr(request.app.state, "memory", None),
    )
    return StatusResponse(
        status=report.status,
        checks=[HealthCheckModel(**check.as_dict()) for check in report.checks],
        snapshot=report.snapshot.as_dict() if report.snapshot else None,
        memory=report.extras.get("memory"),
        history_messages=int(report.extras.get("history_messages", 0) or 0),
        model=str(report.extras.get("model", "")),
        api=str(report.extras.get("api", "")),
    )


# --------------------------------------------------------------------- memory


@router.get("/memory", response_model=MemorySearchResponse)
async def memory_search(
    request: Request,
    q: str = Query("", max_length=400),
    limit: int = Query(20, ge=1, le=100),
) -> MemorySearchResponse:
    """Search the memory. Without a query the newest memories are returned."""
    store = get_memory(request)
    if store is None:
        return MemorySearchResponse(query=q, count=0, results=[])
    if q.strip():
        hits = store.search(q, limit=limit)
    else:
        hits = store.recent_messages(limit=limit)
    return MemorySearchResponse(
        query=q,
        count=len(hits),
        results=[MemoryHitModel(**hit.as_dict()) for hit in hits],
    )


@router.get("/memory/facts", response_model=MemorySearchResponse)
async def memory_facts(request: Request) -> MemorySearchResponse:
    """Every stored fact."""
    store = get_memory(request)
    if store is None:
        return MemorySearchResponse(count=0, results=[])
    facts = store.all_facts()
    return MemorySearchResponse(
        count=len(facts),
        results=[MemoryHitModel(**fact.as_dict()) for fact in facts],
    )


@router.post("/memory/facts", response_model=MemoryHitModel)
async def remember(
    payload: RememberRequest, request: Request
) -> MemoryHitModel:
    """Store a fact by hand, without asking the model."""
    store = get_memory(request)
    if store is None:
        raise ValueError("Memory is switched off.")
    memory_id, _created = store.remember_fact(payload.value, payload.key, source="user")
    hit = store.get(memory_id)
    if hit is None:  # pragma: no cover - the row was just written
        raise ValueError("The fact could not be stored.")
    return MemoryHitModel(**hit.as_dict())


@router.delete("/memory/{memory_id}")
async def forget(memory_id: int, request: Request) -> dict[str, object]:
    """Delete one memory."""
    store = get_memory(request)
    if store is None:
        return {"deleted": False, "detail": "Memory is switched off."}
    deleted = store.forget(memory_id)
    return {"deleted": deleted, "id": memory_id}


@router.post("/memory/clear")
async def clear_memory(request: Request) -> dict[str, object]:
    """Delete every message of this session. The facts are kept."""
    store = get_memory(request)
    if store is None:
        return {"deleted": 0, "detail": "Memory is switched off."}
    return {"deleted": store.clear_session()}
