"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=4000)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class ChatResponse(BaseModel):
    reply: str
    history_messages: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama_reachable: bool
    model: str
    model_available: bool
    detail: str = ""


class HealthCheckModel(BaseModel):
    name: str
    status: Literal["ok", "warn", "error", "unknown"]
    detail: str = ""


class StatusResponse(BaseModel):
    """Everything the web interface shows in its side panel."""

    status: Literal["ok", "warn", "error", "unknown"]
    checks: list[HealthCheckModel] = Field(default_factory=list)
    snapshot: dict[str, object] | None = None
    memory: dict[str, object] | None = None
    history_messages: int = 0
    model: str = ""
    api: str = ""


class MemoryHitModel(BaseModel):
    id: int
    kind: Literal["message", "fact"]
    role: str
    content: str
    score: float = 0.0
    created_at: str = ""
    key: str | None = None
    date_label: str = ""


class MemorySearchResponse(BaseModel):
    query: str = ""
    count: int = 0
    results: list[MemoryHitModel] = Field(default_factory=list)


class RememberRequest(BaseModel):
    value: str = Field(..., min_length=1, max_length=600)
    key: str | None = Field(None, max_length=80)

    @field_validator("value")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value
