"""Tests for the status and memory endpoints (Phases 4 and 5)."""

from fastapi.testclient import TestClient

from jarvis.ai.ollama_client import HealthStatus
from jarvis.api.app import create_app
from jarvis.config import Settings


def make_app(settings, client):
    return TestClient(create_app(settings, client=client))


def make_settings(tmp_path):
    """Settings with the memory in a temporary file."""
    settings = Settings()
    settings.memory.database_path = str(tmp_path / "memory.db")
    return settings


def test_status_reports_the_checks(settings, fake_client_cls):
    client = fake_client_cls(health_status=HealthStatus(True, "m", True))
    with make_app(settings, client) as http:
        body = http.get("/status").json()
    assert body["status"] in {"ok", "warn", "error", "unknown"}
    assert body["checks"]
    assert body["snapshot"] is not None
    assert body["model"] == settings.ollama.model


def test_status_includes_the_memory_counts(tmp_path, fake_client_cls):
    settings = make_settings(tmp_path)
    client = fake_client_cls(health_status=HealthStatus(True, "m", True))
    with make_app(settings, client) as http:
        body = http.get("/status").json()
    assert body["memory"] is not None
    assert body["memory"]["messages"] == 0


def test_memory_search_finds_nothing_at_first(tmp_path, fake_client_cls):
    settings = make_settings(tmp_path)
    with make_app(settings, fake_client_cls()) as http:
        body = http.get("/memory", params={"q": "dog"}).json()
    assert body["count"] == 0
    assert body["results"] == []


def test_a_fact_can_be_stored_and_found(tmp_path, fake_client_cls):
    settings = make_settings(tmp_path)
    with make_app(settings, fake_client_cls()) as http:
        stored = http.post(
            "/memory/facts", json={"value": "My dog is called Rex", "key": "user.dog.name"}
        ).json()
        assert stored["id"] > 0
        assert stored["key"] == "user.dog.name"

        found = http.get("/memory", params={"q": "dog"}).json()
        assert found["count"] == 1
        assert "Rex" in found["results"][0]["content"]

        facts = http.get("/memory/facts").json()
        assert facts["count"] == 1


def test_a_fact_can_be_deleted(tmp_path, fake_client_cls):
    settings = make_settings(tmp_path)
    with make_app(settings, fake_client_cls()) as http:
        stored = http.post("/memory/facts", json={"value": "temporary"}).json()
        deleted = http.delete(f"/memory/{stored['id']}").json()
        assert deleted["deleted"] is True
        assert http.get("/memory/facts").json()["count"] == 0


def test_deleting_an_unknown_memory_says_so(tmp_path, fake_client_cls):
    settings = make_settings(tmp_path)
    with make_app(settings, fake_client_cls()) as http:
        assert http.delete("/memory/9999").json()["deleted"] is False


def test_the_conversation_can_be_cleared(tmp_path, fake_client_cls):
    settings = make_settings(tmp_path)
    client = fake_client_cls(["First.", "Second."])
    with make_app(settings, client) as http:
        http.post("/chat", json={"message": "hello"})
        body = http.post("/memory/clear").json()
        assert body["deleted"] == 2  # the user's message and the reply


def test_a_blank_fact_is_rejected(tmp_path, fake_client_cls):
    settings = make_settings(tmp_path)
    with make_app(settings, fake_client_cls()) as http:
        assert http.post("/memory/facts", json={"value": "   "}).status_code == 422


def test_memory_is_off_when_the_settings_say_so(tmp_path, fake_client_cls):
    settings = make_settings(tmp_path)
    settings.memory.enabled = False
    with make_app(settings, fake_client_cls()) as http:
        body = http.get("/memory", params={"q": "dog"}).json()
        assert body["count"] == 0
