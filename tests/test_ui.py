"""Tests for the web interface helpers (Phase 5)."""

from jarvis.ui.theme import CSS, pill, usage_bar, welcome_panel


def test_the_css_is_present():
    assert ".jarvis-brand" in CSS
    assert "--jarvis-cyan" in CSS


def test_a_pill_shows_the_status():
    html = pill("ok", "Tools")
    assert "pill-ok" in html
    assert "Tools" in html
    assert "dot-ok" in html


def test_an_unknown_status_falls_back():
    html = pill("something-else", "Odd")
    assert "pill-unknown" in html
    assert "dot-unknown" in html


def test_a_usage_bar_carries_the_percentage():
    html = usage_bar(42.5)
    assert "width:42.5%" in html
    assert "jarvis-bar-fill" in html


def test_a_usage_bar_is_clamped():
    assert "width:100.0%" in usage_bar(150)
    assert "width:0.0%" in usage_bar(-10)


def test_a_busy_bar_is_marked():
    assert "hot" in usage_bar(95)
    assert "warn" in usage_bar(80)
    assert "hot" not in usage_bar(10)


def test_the_welcome_panel_names_the_model():
    html = welcome_panel("qwen2.5:3b-instruct")
    assert "qwen2.5:3b-instruct" in html
    assert "At your service" in html


def test_the_backend_answers_a_health_check(settings, fake_client_cls):
    from jarvis.ai.ollama_client import HealthStatus
    from jarvis.ui.backend import ApiBackend

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

    class FakeHttp:
        def get(self, url, timeout=None):
            return FakeResponse()

    backend = ApiBackend("http://127.0.0.1:8000")
    backend._client = FakeHttp()
    assert backend.available() is True
    assert "127.0.0.1:8000" in backend.label


def test_the_local_backend_reports_a_working_system(settings, fake_client_cls):
    from jarvis.ai.ollama_client import HealthStatus
    from jarvis.ui.backend import LocalBackend

    client = fake_client_cls(health_status=HealthStatus(True, "m", True))
    brain = _brain_with(client, settings)
    backend = LocalBackend(settings, brain)
    status = backend.status()
    assert status["status"] in {"ok", "warn", "error", "unknown"}
    assert status["checks"]


def _brain_with(client, settings):
    from jarvis.wiring import build_brain

    return build_brain(settings, client)
