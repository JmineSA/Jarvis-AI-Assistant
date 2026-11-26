import pytest

from jarvis.config import PROJECT_ROOT, Settings


def real_db_state():
    """(exists, size, modified time) of the real data/jarvis.db, to prove tests leave it alone."""
    path = PROJECT_ROOT / "data" / "jarvis.db"
    if not path.exists():
        return (False, 0, 0.0)
    info = path.stat()
    return (True, info.st_size, info.st_mtime)


REAL_DB_AT_START = real_db_state()  # taken when pytest starts, before any test runs


@pytest.fixture(scope="session", autouse=True)
def real_db_unchanged():
    """Returns a check that your real data/jarvis.db is exactly as it was before any test ran."""
    before = real_db_state()

    def check():
        return real_db_state() == before, before, real_db_state()

    return check


@pytest.fixture(autouse=True)
def _no_real_start_menu_lookup(monkeypatch):
    """Tests never read this computer's Start menu or start PowerShell."""
    from jarvis.tools import startmenu

    monkeypatch.setattr(startmenu, "shortcut_folders", lambda env=None: [])
    monkeypatch.setattr(startmenu, "_store_apps_via_powershell", lambda: [])


@pytest.fixture
def settings(tmp_path) -> Settings:
    # Tests must never touch the real data/jarvis.db.
    return Settings.model_validate({"memory": {"database_path": str(tmp_path / "jarvis.db")}})


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
