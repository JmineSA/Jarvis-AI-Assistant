"""Async client for a local Ollama server.

Only this module knows how to talk to Ollama's HTTP API. Every failure is
translated into one of the OllamaError subclasses so callers never have to
handle raw httpx exceptions.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from jarvis.config import OllamaSettings

logger = logging.getLogger(__name__)

Message = dict[str, str]


class OllamaError(Exception):
    """Base class for all Ollama client errors."""


class OllamaUnavailableError(OllamaError):
    """Ollama could not be reached (not installed, not running, wrong host)."""


class OllamaTimeoutError(OllamaError):
    """Ollama accepted the request but did not answer in time."""


class OllamaResponseError(OllamaError):
    """Ollama answered with an error or an unexpected payload."""


@dataclass(frozen=True)
class HealthStatus:
    reachable: bool
    model: str
    model_available: bool
    detail: str = ""


def _model_matches(wanted: str, installed: str) -> bool:
    if wanted == installed:
        return True
    # "llama3.2" means "llama3.2:latest"
    return ":" not in wanted and installed == f"{wanted}:latest"


class OllamaClient:
    def __init__(
        self,
        host: str,
        model: str,
        timeout: float = 120.0,
        temperature: float = 0.4,
        num_ctx: int = 4096,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._num_ctx = num_ctx
        self._client = httpx.AsyncClient(
            base_url=host,
            timeout=httpx.Timeout(timeout, connect=5.0),
            transport=transport,
        )

    @classmethod
    def from_settings(cls, settings: OllamaSettings) -> OllamaClient:
        return cls(
            host=settings.host,
            model=settings.model,
            timeout=settings.timeout_seconds,
            temperature=settings.temperature,
            num_ctx=settings.num_ctx,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ health

    async def health(self) -> HealthStatus:
        """Check that Ollama is reachable and the configured model is pulled."""
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            models = [m.get("name", "") for m in response.json().get("models", [])]
        except httpx.HTTPError as exc:
            return HealthStatus(False, self.model, False, f"Ollama unreachable: {exc!s}")
        except (ValueError, AttributeError, TypeError):
            return HealthStatus(True, self.model, False, "Unexpected reply from Ollama.")

        if any(_model_matches(self.model, name) for name in models):
            return HealthStatus(True, self.model, True)
        return HealthStatus(
            True,
            self.model,
            False,
            f"Model '{self.model}' is not installed. Run: ollama pull {self.model}",
        )

    # -------------------------------------------------------------------- chat

    def _payload(self, messages: list[Message], stream: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": self._temperature, "num_ctx": self._num_ctx},
        }

    def _check_status(self, response: httpx.Response) -> None:
        if response.status_code == 404:
            raise OllamaResponseError(
                f"Model '{self.model}' was not found. Run: ollama pull {self.model}"
            )
        if response.status_code >= 400:
            raise OllamaResponseError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )

    async def chat(self, messages: list[Message]) -> str:
        """Send the conversation and return the assistant's full reply."""
        payload = self._payload(messages, stream=False)
        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.ConnectTimeout as exc:
            raise OllamaUnavailableError("Timed out connecting to Ollama.") from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError("Ollama took too long to respond.") from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(
                "Cannot reach Ollama. Is it running? Try: ollama serve"
            ) from exc

        self._check_status(response)

        try:
            content = response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaResponseError("Unexpected response format from Ollama.") from exc

        if not isinstance(content, str) or not content.strip():
            raise OllamaResponseError("Ollama returned an empty reply.")
        return content.strip()

    async def chat_stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Yield the reply in pieces as the model produces it."""
        payload = self._payload(messages, stream=True)
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    self._check_status(response)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError as exc:
                        raise OllamaResponseError("Malformed streaming data.") from exc
                    if "error" in chunk:
                        raise OllamaResponseError(str(chunk["error"]))
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
        except httpx.ConnectTimeout as exc:
            raise OllamaUnavailableError("Timed out connecting to Ollama.") from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError("Ollama took too long to respond.") from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(
                "Cannot reach Ollama. Is it running? Try: ollama serve"
            ) from exc
