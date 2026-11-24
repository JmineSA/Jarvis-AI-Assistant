from fastapi.testclient import TestClient

from jarvis.ai.ollama_client import (
    HealthStatus,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from jarvis.api.app import create_app


def make_app(settings, client):
    return TestClient(create_app(settings, client=client))


def test_health_ok(settings, fake_client_cls):
    client = fake_client_cls(health_status=HealthStatus(True, "m", True))
    with make_app(settings, client) as http:
        body = http.get("/health").json()
    assert body["status"] == "ok"
    assert body["ollama_reachable"] is True


def test_health_degraded_when_ollama_down(settings, fake_client_cls):
    client = fake_client_cls(health_status=HealthStatus(False, "m", False, "unreachable"))
    with make_app(settings, client) as http:
        response = http.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_chat_multi_turn_keeps_context(settings, fake_client_cls):
    client = fake_client_cls(["First.", "Second."])
    with make_app(settings, client) as http:
        one = http.post("/chat", json={"message": "hello"}).json()
        two = http.post("/chat", json={"message": "again"}).json()
    assert one == {"reply": "First.", "history_messages": 2}
    assert two == {"reply": "Second.", "history_messages": 4}
    assert len(client.calls[1]) == 4  # system + user + assistant + user


def test_chat_rejects_blank_message(settings, fake_client_cls):
    with make_app(settings, fake_client_cls()) as http:
        assert http.post("/chat", json={"message": "   "}).status_code == 422
        assert http.post("/chat", json={}).status_code == 422


def test_reset_clears_context(settings, fake_client_cls):
    client = fake_client_cls()
    with make_app(settings, client) as http:
        http.post("/chat", json={"message": "hello"})
        assert http.post("/reset").status_code == 204
        body = http.post("/chat", json={"message": "hello"}).json()
    assert body["history_messages"] == 2


def test_ollama_down_returns_503_not_crash(settings, fake_client_cls):
    client = fake_client_cls(error=OllamaUnavailableError("Cannot reach Ollama."))
    with make_app(settings, client) as http:
        response = http.post("/chat", json={"message": "hello"})
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def test_timeout_returns_504(settings, fake_client_cls):
    client = fake_client_cls(error=OllamaTimeoutError("slow"))
    with make_app(settings, client) as http:
        assert http.post("/chat", json={"message": "hello"}).status_code == 504


def test_bad_model_reply_returns_502(settings, fake_client_cls):
    client = fake_client_cls(error=OllamaResponseError("bad"))
    with make_app(settings, client) as http:
        assert http.post("/chat", json={"message": "hello"}).status_code == 502


def test_client_closed_on_shutdown(settings, fake_client_cls):
    client = fake_client_cls()
    with make_app(settings, client):
        pass
    assert client.closed


def test_chat_can_use_a_tool_and_answers_from_its_result(settings, fake_client_cls):
    from jarvis.ai.ollama_client import ToolCall

    client = fake_client_cls(tool_calls=[ToolCall("current_time", {})], tool_content="I made it up.")
    with make_app(settings, client) as http:
        body = http.post("/chat", json={"message": "what time is it"}).json()
    assert body["reply"].startswith("It's ")
    assert "made it up" not in body["reply"]
