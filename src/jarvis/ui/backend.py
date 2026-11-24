"""The two ways the web interface can reach JARVIS.

* :class:`ApiBackend` talks to a running ``python -m jarvis`` server over
  HTTP. The server keeps the conversation and the memory.
* :class:`LocalBackend` runs the brain in the interface process itself, so the
  web interface works even when no server is running.

Both expose the same small interface, so the interface code never has to care
which one is in use.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

import httpx

from jarvis.ai.brain import Brain
from jarvis.config import Settings
from jarvis.memory.store import MemoryStore
from jarvis.monitor.health import collect_health
from jarvis.monitor.system import SystemMonitor

logger = logging.getLogger(__name__)


class Backend(Protocol):
    """What the interface needs from whatever is answering."""

    label: str

    def chat(self, message: str) -> str: ...

    def status(self) -> dict: ...

    def memory_search(self, query: str, limit: int = 20) -> list[dict]: ...

    def memory_facts(self) -> list[dict]: ...

    def remember(self, value: str, key: str | None = None) -> dict | None: ...

    def forget(self, memory_id: int) -> bool: ...

    def clear_memory(self) -> int: ...

    def reset(self) -> None: ...


class ApiBackend:
    """Talks to a running JARVIS API server."""

    def __init__(self, base_url: str, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.label = f"API server ({self.base_url})"
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def available(self) -> bool:
        """True when the server answers a health check."""
        try:
            response = self._client.get("/health", timeout=2.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def chat(self, message: str) -> str:
        response = self._client.post("/chat", json={"message": message})
        response.raise_for_status()
        return str(response.json().get("reply", ""))

    def status(self) -> dict:
        response = self._client.get("/status")
        response.raise_for_status()
        return dict(response.json())

    def memory_search(self, query: str, limit: int = 20) -> list[dict]:
        response = self._client.get("/memory", params={"q": query, "limit": limit})
        response.raise_for_status()
        return list(response.json().get("results", []))

    def memory_facts(self) -> list[dict]:
        response = self._client.get("/memory/facts")
        response.raise_for_status()
        return list(response.json().get("results", []))

    def remember(self, value: str, key: str | None = None) -> dict | None:
        payload: dict[str, str] = {"value": value}
        if key:
            payload["key"] = key
        response = self._client.post("/memory/facts", json=payload)
        if response.status_code >= 400:
            return None
        return dict(response.json())

    def forget(self, memory_id: int) -> bool:
        response = self._client.delete(f"/memory/{memory_id}")
        if response.status_code >= 400:
            return False
        return bool(response.json().get("deleted"))

    def clear_memory(self) -> int:
        response = self._client.post("/memory/clear")
        if response.status_code >= 400:
            return 0
        return int(response.json().get("deleted", 0))

    def reset(self) -> None:
        self._client.post("/reset")


class LocalBackend:
    """Runs the brain inside the interface process."""

    def __init__(
        self,
        settings: Settings,
        brain: Brain,
        memory: MemoryStore | None = None,
        monitor: SystemMonitor | None = None,
    ) -> None:
        self.settings = settings
        self._brain = brain
        self._memory = memory
        self._monitor = monitor or SystemMonitor()
        self.label = "local brain (in-process)"
        # One loop for the life of the interface. The brain's lock belongs to
        # the loop that first used it, so the loop must not be recreated.
        self._loop = asyncio.new_event_loop()

    def chat(self, message: str) -> str:
        return self._loop.run_until_complete(self._brain.respond(message))

    def status(self) -> dict:
        report = self._loop.run_until_complete(
            collect_health(
                self.settings,
                self._brain_client(),
                brain=self._brain,
                memory=self._memory,
            )
        )
        return report.as_dict()

    def _brain_client(self):
        return getattr(self._brain, "_client", None)

    def memory_search(self, query: str, limit: int = 20) -> list[dict]:
        if self._memory is None:
            return []
        hits = (
            self._memory.search(query, limit=limit)
            if query.strip()
            else self._memory.recent_messages(limit=limit)
        )
        return [hit.as_dict() for hit in hits]

    def memory_facts(self) -> list[dict]:
        if self._memory is None:
            return []
        return [fact.as_dict() for fact in self._memory.all_facts()]

    def remember(self, value: str, key: str | None = None) -> dict | None:
        if self._memory is None:
            return None
        memory_id, _created = self._memory.remember_fact(value, key, source="user")
        hit = self._memory.get(memory_id)
        return hit.as_dict() if hit else None

    def forget(self, memory_id: int) -> bool:
        if self._memory is None:
            return False
        return self._memory.forget(memory_id)

    def clear_memory(self) -> int:
        if self._memory is None:
            return 0
        return self._memory.clear_session()

    def reset(self) -> None:
        self._brain.reset()


def build_local_backend(settings: Settings, session_id: str = "default") -> LocalBackend:
    """Build a brain and a memory store for use inside the interface."""
    from jarvis.ai.ollama_client import OllamaClient
    from jarvis.wiring import build_brain, build_memory

    client = OllamaClient.from_settings(settings.ollama)
    memory = build_memory(settings, session_id=session_id)
    brain = build_brain(settings, client, memory=memory, session_id=session_id)
    return LocalBackend(settings, brain, memory=memory)


def pick_backend(settings: Settings) -> Backend:
    """Choose a backend according to ``ui.backend`` in the settings.

    ``auto`` uses the running API server when it answers, and falls back to the
    local brain when it does not.
    """
    mode = settings.ui.backend
    if mode in ("auto", "api"):
        api = ApiBackend(settings.ui.api_url)
        if api.available():
            logger.info("Using the JARVIS API server at %s", api.base_url)
            return api
        if mode == "api":
            logger.warning("The API server at %s did not answer.", api.base_url)
    logger.info("Using the local brain.")
    return build_local_backend(settings)
