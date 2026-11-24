import pytest

from jarvis.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Default settings for tests.

    Memory is switched off so the tests never touch the real database in
    ``data/``. The tests that exercise memory build their own settings with a
    temporary database instead.
    """
    settings = Settings()
    settings.memory.enabled = False
    return settings


class FakeClient:
    """Stands in for OllamaClient so tests never need a running Ollama."""

    def __init__(self, replies=None, error=None, health_status=None, tool_calls=None, tool_content=""):
        self.tool_calls = tool_calls          # list of ToolCall, or None for a plain reply
        self.tool_content = tool_content
        self.tools_offered = None
        self.replies = list(replies or ["Very good, sir."])
        self.error = error
        self.health_status = health_status
        self.calls: list[list[dict]] = []
        self.closed = False

    async def chat(self, messages):
        self.calls.append([dict(m) for m in messages])
        if self.error:
            raise self.error
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    async def chat_with_tools(self, messages, tools):
        """Same as chat(), or a scripted tool request when `tool_calls` is set."""
        from jarvis.ai.ollama_client import ChatResult

        self.tools_offered = tools
        if self.tool_calls is not None:
            self.calls.append([dict(m) for m in messages])
            return ChatResult(self.tool_content, tuple(self.tool_calls))
        return ChatResult(await self.chat(messages))

    async def health(self):
        return self.health_status

    async def aclose(self):
        self.closed = True


@pytest.fixture
def fake_client_cls():
    return FakeClient
