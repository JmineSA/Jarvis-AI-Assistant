import pytest

from jarvis.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings()


class FakeClient:
    """Stands in for OllamaClient so tests never need a running Ollama."""

    def __init__(self, replies=None, error=None, health_status=None):
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

    async def health(self):
        return self.health_status

    async def aclose(self):
        self.closed = True


@pytest.fixture
def fake_client_cls():
    return FakeClient
