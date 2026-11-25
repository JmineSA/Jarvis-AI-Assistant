"""Dashboard tests. They run Qt without a screen and never touch the real data folder or Ollama."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

from jarvis.ai.ollama_client import HealthStatus
from jarvis.ui import theme
from jarvis.ui.engine import ButtonTrigger, Engine
from jarvis.ui.metrics import (
    GpuReading,
    parse_nvidia_smi,
    read_gpu,
    read_metrics,
)
from jarvis.ui.pages import ago, until
from jarvis.ui.widgets import ChatView, RingGauge, set_animations
from jarvis.ui.window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(theme.stylesheet())
    set_animations(False)
    return app


def wait_until(app, condition, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def engine(qapp, settings, fake_client_cls):
    client = fake_client_cls(
        replies=["Good evening."], health_status=HealthStatus(True, "qwen2.5:3b-instruct", True, "ok")
    )
    eng = Engine(settings, use_voice=False, client=client)
    yield eng
    eng.stop()


@pytest.fixture
def window(qapp, engine):
    win = MainWindow(engine)
    win.show()
    yield win
    win.metrics_timer.stop()
    win.clock_timer.stop()
    win.close()


# ------------------------------------------------------------------ pure helpers


def test_parse_nvidia_smi():
    reading = parse_nvidia_smi("42, 1024, 4096\n")
    assert reading == GpuReading(42.0, 1024.0, 4096.0)
    assert parse_nvidia_smi("") is None
    assert parse_nvidia_smi("garbage") is None


def test_read_gpu_without_nvidia_smi_is_none():
    def missing(*_a, **_k):
        raise FileNotFoundError

    assert read_gpu(missing) is None


def test_read_metrics_without_gpu():
    metrics = read_metrics(gpu=lambda: None)
    assert 0 <= metrics.cpu <= 100
    assert metrics.gpu is None


def test_time_words():
    now = 1000.0
    assert ago(now - 5, now) == "just now"
    assert ago(now - 300, now) == "5 min ago"
    assert until(now + 1200, now) == "in 20 min"
    assert until(now - 1, now) == "due now"


def test_ring_gauge_handles_missing_value(qapp):
    gauge = RingGauge("GPU", theme.PURPLE)
    gauge.set_value(None)
    assert gauge.text == "n/a"
    gauge.set_value(55)
    assert gauge.percent == 55


def test_chat_view_keeps_order_and_limits(qapp):
    chat = ChatView(owner="Lesiba")
    chat.add_message("user", "hello")
    chat.add_message("assistant", "hi")
    assert chat.texts() == [("user", "hello"), ("assistant", "hi")]
    for i in range(ChatView.MAX_MESSAGES + 10):
        chat.add_message("assistant", str(i))
    assert chat.count == ChatView.MAX_MESSAGES
    chat.clear_messages()
    assert chat.count == 0


def test_button_trigger_toggles():
    trigger = ButtonTrigger()
    assert not trigger.recording
    trigger.button_down()
    assert trigger.recording
    trigger.button_up()
    assert not trigger.recording


# ------------------------------------------------------------------ engine


def test_text_turn_shows_both_sides_and_stores_them(qapp, engine):
    seen = []
    engine.messageAdded.connect(lambda role, text: seen.append((role, text)))
    engine.start()
    engine.submit_text("hello")
    assert wait_until(qapp, lambda: len(seen) >= 2)
    assert seen[0] == ("user", "hello")
    assert seen[1] == ("assistant", "Good evening.")
    assert engine.store.counts().messages == 2


def test_health_is_reported(qapp, engine):
    seen = []
    engine.healthChanged.connect(lambda ok, detail: seen.append(ok))
    engine.start()
    assert wait_until(qapp, lambda: bool(seen))
    assert seen[0] is True


def test_quick_command_runs_the_tool_and_reports_it(qapp, engine):
    tools = []
    engine.toolFinished.connect(lambda name, status, msg: tools.append((name, status)))
    engine.start()
    engine.run_quick("What time is it", "current_time", {})
    assert wait_until(qapp, lambda: bool(tools))
    assert tools[0][0] == "current_time"


def test_confirmation_declined_stops_the_action(qapp, engine):
    requests = []
    results = []
    engine.confirmRequested.connect(lambda i, t: (requests.append(t), engine.answer_confirmation(i, False)))
    engine.toolFinished.connect(lambda name, status, msg: results.append((name, status)))
    engine.store.add_fact("my bank is Capitec")
    engine.start()
    engine.run_quick("Forget", "forget", {"about": "bank"})
    assert wait_until(qapp, lambda: bool(results))
    assert requests, "a confirmation must be asked first"
    assert results[0][1] != "done"
    assert engine.store.counts().facts == 1


def test_talk_without_voice_explains_itself(qapp, engine):
    seen = []
    engine.messageAdded.connect(lambda role, text: seen.append(text))
    assert engine.talk_pressed() is False
    assert any("Voice" in text for text in seen)


# ------------------------------------------------------------------ window


def test_window_has_all_pages_and_switches(window):
    assert set(window.pages) == {"command", "conversations", "memory", "tools", "apps", "settings"}
    for key in window.pages:
        window.show_page(key)
        assert window.current_page() == key
        assert window.nav_buttons[key].isChecked()


def test_tools_page_lists_every_tool(window):
    table = window.pages["tools"].table
    assert table.rowCount() == len(window.engine.tool_names)


def test_typing_a_command_sends_it(qapp, window):
    window.engine.start()
    window.input.setText("hello")
    window.send_text()
    assert window.input.text() == ""
    assert wait_until(qapp, lambda: ("assistant", "Good evening.") in window.command.chat.texts())


def test_quick_commands_come_from_the_profile(window):
    labels = [b.text() for b in window.command.quick_buttons]
    assert "Open VS Code" in labels
    assert "Open Gmail" in labels


def test_confirmation_dialog_answer_reaches_the_engine(qapp, window, monkeypatch):
    answers = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(window.engine, "answer_confirmation", lambda i, ok: answers.append((i, ok)))
    window.ask_confirmation(7, "Forget something")
    assert answers == [(7, True)]
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    window.ask_confirmation(8, "Forget something")
    assert answers[-1] == (8, False)


def test_state_changes_update_the_sidebar(window):
    window.set_state("listening")
    assert window.voice_state.text() == "LISTENING"
    window.set_state("idle")
    assert window.voice_state.text() == "IDLE"


# ------------------------------------------------------------------ memory page


def test_memory_page_add_and_forget_fact(window, monkeypatch):
    page = window.memory_page
    page.new_fact.setText("My favourite editor is VS Code")
    page.add_fact()
    assert window.engine.store.counts().facts == 1
    assert window.command._counts["facts"].text() == "1"

    fact = window.engine.store.list_facts()[0]
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    page.forget_fact(fact.id, fact.text)
    assert window.engine.store.counts().facts == 1  # said no: still there
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    page.forget_fact(fact.id, fact.text)
    assert window.engine.store.counts().facts == 0


def test_memory_page_cancel_reminder_needs_a_yes(window, monkeypatch):
    store = window.engine.store
    reminder = store.add_reminder("call the bank", (datetime.now() + timedelta(minutes=20)).timestamp())
    window.refresh_memory()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    window.memory_page.cancel_reminder(reminder.id, reminder.text)
    assert store.counts().pending_reminders == 1
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    window.memory_page.cancel_reminder(reminder.id, reminder.text)
    assert store.counts().pending_reminders == 0


def test_erase_everything_needs_the_word_delete(window, monkeypatch):
    store = window.engine.store
    store.add_fact("something")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("nope", True))
    window.memory_page.erase_everything()
    assert store.counts().facts == 1
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("DELETE", True))
    window.memory_page.erase_everything()
    assert store.counts().facts == 0


def test_clear_conversations_keeps_facts(window, monkeypatch):
    store = window.engine.store
    store.add_fact("keep me")
    store.add_message("s1", "user", "hello")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    window.memory_page.clear_conversations()
    counts = store.counts()
    assert counts.messages == 0
    assert counts.facts == 1


def test_conversations_search(window):
    store = window.engine.store
    store.add_message("s1", "user", "play some jazz")
    store.add_message("s1", "assistant", "Playing jazz")
    page = window.conversations
    page.run_search("jazz")
    assert page.list.count >= 1
    page.run_search("zzzz")
    assert "Nothing found." in [t for _r, t in page.list.texts()]


def test_pages_work_with_memory_off(qapp, settings, fake_client_cls):
    settings.memory.enabled = False
    eng = Engine(settings, use_voice=False, client=fake_client_cls(), store=None)
    win = MainWindow(eng)
    for key in win.pages:
        win.show_page(key)
    assert win.command._overview_values["memory"].text() == "Memory is off"
    win.metrics_timer.stop()
    win.clock_timer.stop()
    win.close()
