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


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LlmStats:
    """What Ollama reports about one request (seconds and token counts)."""

    total_s: float = 0.0
    load_s: float = 0.0
    prompt_tokens: int = 0
    prompt_s: float = 0.0
    output_tokens: int = 0
    output_s: float = 0.0

    @classmethod
    def from_response(cls, data: dict[str, Any]) -> LlmStats:
        def seconds(key: str) -> float:
            value = data.get(key)
            return float(value) / 1e9 if isinstance(value, (int, float)) else 0.0

        def count(key: str) -> int:
            value = data.get(key)
            return int(value) if isinstance(value, (int, float)) else 0

        return cls(
            seconds("total_duration"), seconds("load_duration"),
            count("prompt_eval_count"), seconds("prompt_eval_duration"),
            count("eval_count"), seconds("eval_duration"),
        )

    def summary(self) -> str:
        return (
            f"{self.total_s:.1f}s total (load {self.load_s:.1f}s, prompt {self.prompt_tokens} tokens "
            f"in {self.prompt_s:.1f}s, reply {self.output_tokens} tokens in {self.output_s:.1f}s)"
        )


@dataclass(frozen=True)
class ChatResult:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()


def parse_tool_calls(raw: Any) -> tuple[ToolCall, ...]:
    """Read the model's tool requests, tolerating the odd shapes small models produce."""
    calls: list[ToolCall] = []
    for item in raw if isinstance(raw, list) else []:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):  # JSON text instead of an object
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        calls.append(
            ToolCall(function["name"], arguments if isinstance(arguments, dict) else {})
        )
    return tuple(calls)


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
        keep_alive: str = "30m",
        num_predict: int = 256,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive
        self._num_predict = num_predict
        self.last_stats: LlmStats | None = None
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
            keep_alive=settings.keep_alive,
            num_predict=settings.num_predict,
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
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": self._temperature,
                "num_ctx": self._num_ctx,
                "num_predict": self._num_predict,
            },
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

    async def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to /api/chat and return the reply's "message" object."""
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
            body = response.json()
            message = body["message"]
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaResponseError("Unexpected response format from Ollama.") from exc
        if not isinstance(message, dict):
            raise OllamaResponseError("Unexpected response format from Ollama.")
        if isinstance(body, dict):
            self.last_stats = LlmStats.from_response(body)
            logger.info("Model: %s", self.last_stats.summary())
        return message

    async def chat(self, messages: list[Message]) -> str:
        """Send the conversation and return the assistant's full reply."""
        message = await self._post_chat(self._payload(messages, stream=False))
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OllamaResponseError("Ollama returned an empty reply.")
        return content.strip()

    async def chat_with_tools(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> ChatResult:
        """Like chat(), but the model may answer with tool requests instead of text."""
        payload = self._payload(messages, stream=False)
        payload["tools"] = tools
        message = await self._post_chat(payload)
        content = message.get("content")
        return ChatResult(
            content=content.strip() if isinstance(content, str) else "",
            tool_calls=parse_tool_calls(message.get("tool_calls")),
        )

    async def warm_up(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> None:
        """Load the model and cache the fixed start of the prompt, so the first real reply is quick.

        Never raises: if Ollama is not ready, JARVIS simply starts cold.
        """
        payload = self._payload(messages, stream=False)
        payload["options"] = {**payload["options"], "num_predict": 1}
        if tools:
            payload["tools"] = tools
        try:
            await self._post_chat(payload)
        except OllamaError as exc:
            logger.info("Warm-up skipped: %s", exc)

    async def processor(self) -> str | None:
        """Where the loaded model is running, e.g. "GPU 100%" or "CPU"; None if not loaded."""
        try:
            response = await self._client.get("/api/ps")
            response.raise_for_status()
            models = response.json().get("models", [])
        except (httpx.HTTPError, ValueError, AttributeError):
            return None
        for item in models:
            if isinstance(item, dict) and _model_matches(self.model, str(item.get("name", ""))):
                size = item.get("size") or 0
                vram = item.get("size_vram") or 0
                if not size or not vram:
                    return "CPU"
                percent = round(100 * vram / size)
                return f"GPU {percent}%" if percent >= 99 else f"GPU {percent}% / CPU {100 - percent}%"
        return None

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
