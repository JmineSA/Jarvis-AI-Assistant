"""The screens of the dashboard. Each page reads from the engine and shows what is really there."""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from jarvis.audio_devices import (
    AudioDevice,
    list_devices,
    loudest,
    measure_levels,
    play_test_tone,
)
from jarvis.config import resolve_config_path
from jarvis.diagnostics import build_diagnostics
from jarvis.geocode import Place, find_place
from jarvis.profile import expand_path
from jarvis.saving import save_audio_device, save_greeting_name, save_home_address
from jarvis.timeparse import format_due
from jarvis.tools.apps import find_executable
from jarvis.ui import theme
from jarvis.ui.engine import Engine
from jarvis.ui.metrics import Metrics
from jarvis.ui.widgets import (
    Card,
    ChatView,
    CoreWidget,
    Panel,
    RingGauge,
    dot,
    icon,
    icon_chip,
    muted,
    set_dot,
)

STATUS_COLORS = {"done": theme.TEAL, "launched": theme.TEAL, "failed": theme.RED, "denied": theme.AMBER}
STATUS_WORDS = {"done": "Done", "launched": "Launched", "failed": "Failed", "denied": "Declined"}


def clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


def ago(timestamp: float, now: float | None = None) -> str:
    seconds = max(0, int((now or time.time()) - timestamp))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86400} d ago"


def until(timestamp: float, now: float | None = None) -> str:
    seconds = int(timestamp - (now or time.time()))
    if seconds <= 0:
        return "due now"
    if seconds < 3600:
        return f"in {max(1, seconds // 60)} min"
    if seconds < 86400:
        return f"in {seconds // 3600} h"
    return f"in {seconds // 86400} d"


def open_local(path: Path) -> bool:
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


# =========================================================================== command center


class CommandCenterPage(QWidget):
    navigate = Signal(str)

    def __init__(self, engine: Engine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self._recent: list[tuple[str, str, float]] = []  # tool name, status, when
        self._model_info = ""
        self._health = (False, "Checking...")
        self._voice = (False, "Starting...")

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        # ---- row 1
        row1 = QHBoxLayout()
        row1.setSpacing(14)
        self.overview = Panel("Core overview")
        self.overview.setFixedWidth(260)
        self._overview_values: dict[str, QLabel] = {}
        self._overview_dots: dict[str, QLabel] = {}
        for key, name, icon_name, color in (
            ("llm", "Language model", "hex", theme.ACCENT_GLOW),
            ("voice", "Voice", "mic", theme.TEAL),
            ("memory", "Memory", "memory", theme.PURPLE),
            ("reminders", "Reminders", "clock", theme.AMBER),
            ("tools", "Tools", "tools", theme.ACCENT_GLOW),
        ):
            card = Card()
            card.row.addWidget(icon_chip(icon_name, color))
            text = QVBoxLayout()
            text.setSpacing(0)
            title = QLabel(name)
            title.setStyleSheet("font-weight:600;")
            value = QLabel("-")
            value.setStyleSheet(f"color:{color};font-size:11px;")
            value.setWordWrap(True)
            text.addWidget(title)
            text.addWidget(value)
            card.row.addLayout(text, 1)
            self._overview_values[key] = value
            self.overview.body.addWidget(card)
        self.overview.body.addStretch(1)
        row1.addWidget(self.overview)

        self.core = CoreWidget()
        row1.addWidget(self.core, 1)

        self.conversation = Panel("Live conversation", "LIVE")
        self.conversation.setFixedWidth(340)
        self.chat = ChatView(owner=engine.owner_name)
        self.conversation.body.addWidget(self.chat)
        row1.addWidget(self.conversation)
        root.addLayout(row1, 340)

        # ---- row 2
        row2 = QHBoxLayout()
        row2.setSpacing(14)
        self.tools_panel = Panel("Tool activity", f"{len(engine.tool_names)} tools · every action is logged")
        self.tools_grid = QGridLayout()
        self.tools_grid.setSpacing(10)
        self.tools_panel.body.addLayout(self.tools_grid)
        self.tools_panel.body.addStretch(1)
        row2.addWidget(self.tools_panel, 3)

        self.reminders_panel = Panel("Reminders", "spoken while JARVIS runs")
        self.reminders_panel.setFixedWidth(300)
        self.reminders_list = QVBoxLayout()
        self.reminders_list.setSpacing(8)
        self.reminders_panel.body.addLayout(self.reminders_list)
        self.reminders_panel.body.addStretch(1)
        row2.addWidget(self.reminders_panel)

        self.quick_panel = Panel("Quick commands")
        self.quick_panel.setFixedWidth(250)
        self._build_quick_commands()
        row2.addWidget(self.quick_panel)
        root.addLayout(row2, 190)

        # ---- row 3
        row3 = QHBoxLayout()
        row3.setSpacing(14)
        self.monitor = Panel("System monitor")
        self.monitor.setFixedWidth(360)
        gauges = QHBoxLayout()
        self.cpu = RingGauge("CPU", theme.ACCENT_GLOW)
        self.ram = RingGauge("RAM", theme.TEAL)
        self.gpu = RingGauge("GPU", theme.PURPLE)
        for gauge in (self.cpu, self.ram, self.gpu):
            gauges.addWidget(gauge, 0, Qt.AlignmentFlag.AlignCenter)
        self.monitor.body.addLayout(gauges)
        self.monitor_note = muted("", "dim")
        self.monitor.body.addWidget(self.monitor_note)
        row3.addWidget(self.monitor)

        self.memory_panel = Panel("Memory")
        self.memory_panel.setFixedWidth(330)
        link = QPushButton("Open memory  ›")
        link.setFlat(True)
        link.setStyleSheet(f"color:{theme.ACCENT_GLOW};border:none;text-align:right;background:transparent;")
        link.clicked.connect(lambda: self.navigate.emit("memory"))
        self.memory_panel.body.addWidget(link)
        counts = QHBoxLayout()
        self._counts: dict[str, QLabel] = {}
        for key, label in (("facts", "Facts"), ("messages", "Messages"), ("reminders", "Reminders")):
            column = QVBoxLayout()
            column.setSpacing(0)
            number = QLabel("0")
            number.setObjectName("bigNumber")
            column.addWidget(number)
            column.addWidget(muted(label, "dim"))
            counts.addLayout(column)
            self._counts[key] = number
        counts.addStretch(1)
        self.memory_panel.body.addLayout(counts)
        self.latest_fact = muted("")
        self.memory_panel.body.addWidget(self.latest_fact)
        self.memory_panel.body.addStretch(1)
        row3.addWidget(self.memory_panel)

        self.models_panel = Panel("Model and voice", "All local")
        grid = QGridLayout()
        grid.setSpacing(8)
        self._model_cards: dict[str, tuple[QLabel, QLabel]] = {}
        for index, (key, name) in enumerate(
            (("llm", "Ollama"), ("stt", "Speech to text"), ("tts", "Text to speech"), ("mic", "Microphone"))
        ):
            card = Card()
            status_dot = dot(theme.TEXT_DIM)
            card.row.addWidget(status_dot)
            text = QVBoxLayout()
            text.setSpacing(0)
            title = QLabel(name)
            title.setStyleSheet("font-weight:600;font-size:12px;")
            detail = QLabel("-")
            detail.setObjectName("muted")
            detail.setStyleSheet(f"color:{theme.TEXT_MUTED};font-size:11px;")
            text.addWidget(title)
            text.addWidget(detail)
            card.row.addLayout(text, 1)
            grid.addWidget(card, index // 2, index % 2)
            self._model_cards[key] = (status_dot, detail)
        self.models_panel.body.addLayout(grid)
        self.models_panel.body.addStretch(1)
        row3.addWidget(self.models_panel, 1)
        root.addLayout(row3, 170)

        voice = engine.settings.voice
        self.core.set_details(
            f"{engine.settings.ollama.model} · runs on this PC",
            f"Whisper {voice.stt.model} · Piper {voice.tts.voice.split('-')[1] if '-' in voice.tts.voice else voice.tts.voice}",
        )
        self._set_model_card("llm", None, engine.settings.ollama.model)
        self._set_model_card("stt", True, f"Whisper {voice.stt.model} · CPU")
        self._set_model_card("tts", True, f"Piper {voice.tts.voice}")
        self._set_model_card(
            "mic", None, "Waiting for voice..." if engine.settings.voice.input_device is None
            else f"Input device {engine.settings.voice.input_device}",
        )
        self.refresh_memory()
        self._render_tools()

    # ------------------------------------------------------------------ quick commands

    def _build_quick_commands(self) -> None:
        profile = self.engine.profile
        wanted: list[tuple[str, str, dict, str]] = []
        for spoken, label in (("vs code", "Open VS Code"), ("chrome", "Open Chrome")):
            key = profile.resolve("apps", spoken)
            if key:
                wanted.append((label, "open_app", {"name": key}, "code" if spoken == "vs code" else "globe"))
        if profile.location is not None:
            wanted.append(("Weather today", "weather", {}, "cloud"))
        if profile.music.folder:
            wanted.append(("Play music", "play_media", {"name": ""}, "music"))
        gmail = profile.resolve("websites", "gmail")
        if gmail:
            wanted.append(("Open Gmail", "open_website", {"name": gmail}, "mail"))
        self.quick_buttons: list[QPushButton] = []
        for label, tool, arguments, icon_name in wanted[:5]:
            button = QPushButton(label)
            button.setObjectName("quick")
            button.setIcon(icon(icon_name))
            button.setIconSize(QSize(16, 16))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, l=label, t=tool, a=arguments: self.engine.run_quick(l, t, a))
            self.quick_panel.body.addWidget(button)
            self.quick_buttons.append(button)
        if not wanted:
            self.quick_panel.body.addWidget(muted("Add apps and a location in profile.yaml to get shortcuts."))
        self.quick_panel.body.addStretch(1)

    # ------------------------------------------------------------------ updates

    def set_state(self, state: str) -> None:
        self.core.set_state(state)

    def add_message(self, role: str, text: str) -> None:
        self.chat.add_message(role, text)

    def add_tool_event(self, name: str, status: str, message: str) -> None:
        self.chat.add_tool_tag(f"{name} · {STATUS_WORDS.get(status, status).lower()}")
        self._recent = [r for r in self._recent if r[0] != name]
        self._recent.insert(0, (name, status, time.time()))
        del self._recent[6:]
        self._render_tools()

    def _render_tools(self) -> None:
        clear_layout(self.tools_grid)
        if not self._recent:
            self.tools_grid.addWidget(muted("Nothing yet. Ask JARVIS to do something and it will show up here."), 0, 0)
            return
        for index, (name, status, when) in enumerate(self._recent):
            color = STATUS_COLORS.get(status, theme.TEXT_MUTED)
            card = Card()
            card.row.addWidget(icon_chip(_tool_icon(name), color, 34))
            text = QVBoxLayout()
            text.setSpacing(0)
            title = QLabel(name)
            title.setStyleSheet("font-weight:600;")
            state = QLabel(STATUS_WORDS.get(status, status))
            state.setStyleSheet(f"color:{color};font-size:11px;")
            text.addWidget(title)
            text.addWidget(state)
            text.addWidget(muted(ago(when), "dim"))
            card.row.addLayout(text, 1)
            self.tools_grid.addWidget(card, index // 3, index % 3)

    def update_metrics(self, metrics: Metrics) -> None:
        self.cpu.set_value(metrics.cpu)
        self.ram.set_value(metrics.ram)
        self.monitor_note.setText(f"RAM {metrics.ram_used_gb:.1f} of {metrics.ram_total_gb:.0f} GB · disk {metrics.disk:.0f}% full")
        if metrics.gpu is None:
            self.gpu.set_value(None)
        else:
            self.gpu.set_value(metrics.gpu.percent)
            self.monitor_note.setText(
                self.monitor_note.text()
                + f" · VRAM {metrics.gpu.memory_used_mb / 1024:.1f} of {metrics.gpu.memory_total_mb / 1024:.0f} GB"
            )

    def set_health(self, ready: bool, detail: str) -> None:
        self._health = (ready, detail)
        self._set_model_card("llm", ready, self.engine.settings.ollama.model if ready else detail)
        self._refresh_overview()

    def set_voice(self, ready: bool, detail: str) -> None:
        self._voice = (ready, detail)
        self._set_model_card("mic", ready, "Ready" if ready else detail)
        self._refresh_overview()

    def _set_model_card(self, key: str, ok: bool | None, text: str) -> None:
        status_dot, detail = self._model_cards[key]
        set_dot(status_dot, theme.TEXT_DIM if ok is None else theme.TEAL if ok else theme.RED)
        detail.setText(text)
        detail.setToolTip(text)

    def set_model_info(self, text: str) -> None:
        self._model_info = text
        self._refresh_overview()

    def _refresh_overview(self) -> None:
        ready, detail = self._health
        info = f" · {self._model_info}" if self._model_info else ""
        self._overview_values["llm"].setText(f"{self.engine.settings.ollama.model}{info}" if ready else detail)
        self._overview_values["llm"].setToolTip(self._model_info)
        voice_ready, voice_detail = self._voice
        self._overview_values["voice"].setText("Whisper + Piper ready" if voice_ready else voice_detail)
        self._overview_values["tools"].setText(f"{len(self.engine.tool_names)} loaded")

    def refresh_memory(self) -> None:
        store = self.engine.store
        self._refresh_overview()
        if store is None:
            self._overview_values["memory"].setText("Memory is off")
            self._overview_values["reminders"].setText("Reminders are off")
            self.latest_fact.setText("Memory is turned off in settings.yaml.")
            clear_layout(self.reminders_list)
            self.reminders_list.addWidget(muted("Reminders need memory to be on."))
            return
        counts = store.counts()
        self._overview_values["memory"].setText(f"{counts.facts} facts saved")
        self._overview_values["reminders"].setText(f"{counts.pending_reminders} pending")
        self._counts["facts"].setText(str(counts.facts))
        self._counts["messages"].setText(str(counts.messages))
        self._counts["reminders"].setText(str(counts.pending_reminders))
        facts = store.list_facts(1)
        self.latest_fact.setText(f"Latest fact: {facts[0].text}" if facts else 'Nothing saved yet. Say "remember that..."')

        clear_layout(self.reminders_list)
        pending = store.pending_reminders()[:3]
        if not pending:
            self.reminders_list.addWidget(muted('No reminders. Say "remind me to... in 20 minutes".'))
        now = datetime.now()
        for reminder in pending:
            row = QHBoxLayout()
            row.setSpacing(10)
            row.addWidget(dot(theme.AMBER, 9), 0, Qt.AlignmentFlag.AlignTop)
            column = QVBoxLayout()
            column.setSpacing(0)
            column.addWidget(QLabel(reminder.text))
            column.addWidget(muted(format_due(datetime.fromtimestamp(reminder.due_at), now).capitalize(), "dim"))
            row.addLayout(column, 1)
            soon = QLabel(until(reminder.due_at))
            soon.setStyleSheet(f"color:{theme.AMBER};font-size:11px;")
            row.addWidget(soon, 0, Qt.AlignmentFlag.AlignTop)
            holder = QWidget()
            holder.setLayout(row)
            self.reminders_list.addWidget(holder)


def _tool_icon(name: str) -> str:
    return {
        "play_media": "music", "weather": "cloud", "web_search": "globe", "open_app": "folder",
        "open_folder": "folder", "open_website": "globe", "compose_email": "mail", "remember": "check",
        "recall": "memory", "forget": "trash", "set_reminder": "clock", "list_reminders": "clock",
        "cancel_reminder": "clock", "system_status": "grid", "current_time": "clock", "list_files": "folder",
    }.get(name, "tools")


# =========================================================================== conversations


class ConversationsPage(QWidget):
    def __init__(self, engine: Engine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(12)
        title = QLabel("CONVERSATIONS")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search everything you have said to JARVIS")
        self.search.setAccessibleName("Search conversations")
        self.search.returnPressed.connect(self.run_search)
        go = QPushButton("Search")
        go.setObjectName("primary")
        go.clicked.connect(self.run_search)
        clear = QPushButton("Show latest")
        clear.clicked.connect(self.show_latest)
        bar.addWidget(self.search, 1)
        bar.addWidget(go)
        bar.addWidget(clear)
        root.addLayout(bar)

        self.panel = Panel("Latest messages")
        self.list = ChatView(owner=engine.owner_name)
        self.panel.body.addWidget(self.list)
        root.addWidget(self.panel, 1)
        self.show_latest()

    def show_latest(self) -> None:
        self.search.clear()
        self.panel.set_title("Latest messages")
        self.list.clear_messages()
        store = self.engine.store
        if store is None:
            self.list.add_message("system", "Memory is turned off, so conversations are not stored.")
            return
        messages = store.recent_messages(limit=100)
        if not messages:
            self.list.add_message("system", "No stored conversations yet.")
        for message in messages:
            stamp = datetime.fromtimestamp(message.created_at).strftime("%d %b %H:%M")
            self.list.add_message(message.role, f"{message.content}\n{stamp}")

    def run_search(self, query: str | None = None) -> None:
        if isinstance(query, str):
            self.search.setText(query)
        words = self.search.text().strip()
        if not words:
            self.show_latest()
            return
        self.panel.set_title(f"Search: {words}")
        self.list.clear_messages()
        store = self.engine.store
        if store is None:
            self.list.add_message("system", "Memory is turned off, so there is nothing to search.")
            return
        found = store.search_messages(words, limit=50)
        if not found:
            self.list.add_message("system", "Nothing found.")
        for message in found:
            stamp = datetime.fromtimestamp(message.created_at).strftime("%d %b %H:%M")
            self.list.add_message(message.role, f"{message.content}\n{stamp}")

    def refresh(self) -> None:
        if not self.search.text().strip():
            self.show_latest()


# =========================================================================== memory


class MemoryPage(QWidget):
    changed = Signal()

    def __init__(self, engine: Engine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(12)
        heading = QVBoxLayout()
        heading.setSpacing(0)
        title = QLabel("MEMORY AND REMINDERS")
        title.setObjectName("pageTitle")
        self.where = muted("", "muted")
        heading.addWidget(title)
        heading.addWidget(self.where)
        root.addLayout(heading)

        body = QHBoxLayout()
        body.setSpacing(16)

        self.facts_panel = Panel("Saved facts", 'Saved only when you say "remember that..."')
        add = QHBoxLayout()
        self.new_fact = QLineEdit()
        self.new_fact.setPlaceholderText("Add a fact, e.g. My favourite editor is VS Code")
        self.new_fact.setAccessibleName("New fact")
        self.new_fact.returnPressed.connect(self.add_fact)
        save = QPushButton("Save")
        save.setObjectName("primary")
        save.clicked.connect(self.add_fact)
        add.addWidget(self.new_fact, 1)
        add.addWidget(save)
        self.facts_panel.body.addLayout(add)
        self.facts_host = QVBoxLayout()
        self.facts_host.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        holder.setLayout(self.facts_host)
        scroll.setWidget(holder)
        self.facts_panel.body.addWidget(scroll, 1)
        self.facts_panel.body.addWidget(muted('Deleting a fact asks you to confirm first, the same as when you say "forget".', "dim"))
        body.addWidget(self.facts_panel, 11)

        right = QVBoxLayout()
        right.setSpacing(16)
        self.reminders_panel = Panel("Reminders", "Spoken only while JARVIS is running")
        self.reminders_host = QVBoxLayout()
        self.reminders_host.setSpacing(8)
        self.reminders_panel.body.addLayout(self.reminders_host)
        right.addWidget(self.reminders_panel)

        self.history_panel = Panel("Past conversations")
        self.history_host = QVBoxLayout()
        self.history_host.setSpacing(8)
        self.history_panel.body.addLayout(self.history_host)
        self.history_panel.body.addStretch(1)
        right.addWidget(self.history_panel, 1)

        self.privacy = Panel("Privacy controls", name="panelDanger")
        self.privacy.title_label.hide()
        row = QHBoxLayout()
        text = QVBoxLayout()
        text.setSpacing(0)
        heading_label = QLabel("Privacy controls")
        heading_label.setStyleSheet("font-weight:600;")
        text.addWidget(heading_label)
        text.addWidget(muted("Clearing conversations keeps your facts and reminders. Erasing everything deletes all three.", "dim"))
        row.addLayout(text, 1)
        self.clear_button = QPushButton("Clear conversations")
        self.clear_button.setObjectName("danger")
        self.clear_button.clicked.connect(self.clear_conversations)
        self.erase_button = QPushButton("Erase everything...")
        self.erase_button.setObjectName("danger")
        self.erase_button.clicked.connect(self.erase_everything)
        row.addWidget(self.clear_button)
        row.addWidget(self.erase_button)
        self.privacy.body.addLayout(row)
        right.addWidget(self.privacy)
        body.addLayout(right, 10)
        root.addLayout(body, 1)
        self.refresh()

    # ------------------------------------------------------------------ data

    def refresh(self) -> None:
        store = self.engine.store
        self.where.setText(
            f"Everything JARVIS remembers lives on this PC in {self.engine.settings.resolve_database_path()}. "
            "You can see it, search it and delete it."
            if store is not None else "Memory is turned off in settings.yaml."
        )
        clear_layout(self.facts_host)
        clear_layout(self.reminders_host)
        clear_layout(self.history_host)
        for button in (self.clear_button, self.erase_button):
            button.setEnabled(store is not None)
        if store is None:
            self.facts_host.addWidget(muted("Memory is off."))
            return

        counts = store.counts()
        self.facts_panel.set_title(f"Saved facts · {counts.facts}")
        self.reminders_panel.set_title(f"Reminders · {counts.pending_reminders} pending")
        self.history_panel.set_title(f"Past conversations · {counts.messages} messages")

        facts = store.list_facts()
        if not facts:
            self.facts_host.addWidget(muted('Nothing saved yet. Say "remember that..." or add one above.'))
        for fact in facts:
            card = Card()
            card.row.addWidget(muted(f"#{fact.id}", "dim"))
            label = QLabel(fact.text)
            label.setWordWrap(True)
            card.row.addWidget(label, 1)
            card.row.addWidget(muted(datetime.fromtimestamp(fact.created_at).strftime("%d %b"), "dim"))
            delete = QPushButton()
            delete.setIcon(icon("trash", theme.RED, 15))
            delete.setObjectName("iconButton")
            delete.setToolTip("Forget this fact")
            delete.setAccessibleName("Forget this fact")
            delete.clicked.connect(lambda _=False, f=fact: self.forget_fact(f.id, f.text))
            card.row.addWidget(delete)
            self.facts_host.addWidget(card)
        self.facts_host.addStretch(1)

        now = datetime.now()
        pending = store.pending_reminders()
        if not pending:
            self.reminders_host.addWidget(muted("No pending reminders."))
        for reminder in pending:
            card = Card()
            card.row.addWidget(dot(theme.AMBER, 9))
            column = QVBoxLayout()
            column.setSpacing(0)
            column.addWidget(QLabel(reminder.text))
            column.addWidget(muted(format_due(datetime.fromtimestamp(reminder.due_at), now).capitalize(), "dim"))
            card.row.addLayout(column, 1)
            card.row.addWidget(muted(until(reminder.due_at), "dim"))
            cancel = QPushButton("Cancel")
            cancel.clicked.connect(lambda _=False, r=reminder: self.cancel_reminder(r.id, r.text))
            card.row.addWidget(cancel)
            self.reminders_host.addWidget(card)

        recent = store.recent_messages(limit=4)
        if not recent:
            self.history_host.addWidget(muted("No stored conversations yet."))
        for message in reversed(recent):
            card = QFrame()
            card.setObjectName("card")
            column = QVBoxLayout(card)
            column.setContentsMargins(12, 8, 12, 8)
            column.setSpacing(0)
            who = "You" if message.role == "user" else "JARVIS"
            stamp = datetime.fromtimestamp(message.created_at).strftime("%d %b %H:%M")
            column.addWidget(muted(f"{stamp} · {who}", "dim"))
            body = QLabel(message.content)
            body.setWordWrap(True)
            column.addWidget(body)
            self.history_host.addWidget(card)

    # ------------------------------------------------------------------ actions (each is a deliberate click)

    def add_fact(self) -> None:
        store = self.engine.store
        text = self.new_fact.text().strip()
        if store is None or not text:
            return
        try:
            store.add_fact(text)
        except ValueError:
            return
        self.new_fact.clear()
        self.refresh()
        self.changed.emit()

    def _ask(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(
            self, title, text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def forget_fact(self, fact_id: int, text: str) -> None:
        if self.engine.store is None or not self._ask("Forget this fact?", f'Forget: "{text}"?'):
            return
        self.engine.store.delete_fact(fact_id)
        self.refresh()
        self.changed.emit()

    def cancel_reminder(self, reminder_id: int, text: str) -> None:
        if self.engine.store is None or not self._ask("Cancel reminder?", f'Cancel the reminder: "{text}"?'):
            return
        self.engine.store.cancel_reminder(reminder_id)
        self.refresh()
        self.changed.emit()

    def clear_conversations(self) -> None:
        if self.engine.store is None:
            return
        if not self._ask("Clear conversations?", "Delete every stored conversation? Facts and reminders are kept."):
            return
        self.engine.store.clear_messages()
        self.refresh()
        self.changed.emit()

    def erase_everything(self) -> None:
        if self.engine.store is None:
            return
        typed, ok = QInputDialog.getText(
            self, "Erase everything",
            "This deletes every fact, reminder and conversation, and cannot be undone.\nType DELETE to confirm:",
        )
        if not ok or typed.strip() != "DELETE":
            return
        self.engine.store.wipe()
        self.refresh()
        self.changed.emit()


# =========================================================================== tools


class ToolsPage(QWidget):
    def __init__(self, engine: Engine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(12)
        title = QLabel("TOOLS")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(muted(
            "These are the only things JARVIS can do. It cannot run commands, and anything marked "
            '"asks first" needs your yes in a dialog.'
        ))
        panel = Panel("Available tools", f"{len(engine.tool_names)} loaded")
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Tool", "What it does", "Safety"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(0, 170)
        self.table.setColumnWidth(2, 120)
        self.table.horizontalHeader().setSectionResizeMode(1, self.table.horizontalHeader().ResizeMode.Stretch)
        executor = engine.brain.executor
        for name in engine.tool_names:
            tool = executor.registry.get(name) if executor else None
            if tool is None:
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(tool.description))
            safety = QTableWidgetItem("Asks first" if tool.risk == "confirm" else "Safe")
            safety.setForeground(Qt.GlobalColor.yellow if tool.risk == "confirm" else Qt.GlobalColor.green)
            self.table.setItem(row, 2, safety)
        panel.body.addWidget(self.table)
        root.addWidget(panel, 1)


# =========================================================================== apps and music


class AppsPage(QWidget):
    def __init__(self, engine: Engine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(12)
        title = QLabel("APPS AND MUSIC")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(muted(
            "What JARVIS is allowed to open. A red dot means it can't find the program or folder on this PC: "
            'fix the path in profile.yaml, or run  python -m jarvis add-app "Name".'
        ))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        self.host = QVBoxLayout(host)
        self.host.setSpacing(12)
        scroll.setWidget(host)
        root.addWidget(scroll, 1)
        buttons = QHBoxLayout()
        self.edit_button = QPushButton("Edit profile.yaml")
        self.edit_button.setObjectName("primary")
        self.edit_button.clicked.connect(lambda: open_local(self.engine.settings.resolve_profile_path()))
        recheck = QPushButton("Check again")
        recheck.clicked.connect(self.refresh)
        buttons.addWidget(self.edit_button)
        buttons.addWidget(recheck)
        buttons.addStretch(1)
        root.addLayout(buttons)
        self.refresh()

    def refresh(self) -> None:
        clear_layout(self.host)
        profile = self.engine.profile

        apps = Panel("Apps", f"{len(profile.apps)}")
        for key, entry in profile.apps.items():
            found = find_executable(entry.candidates, shutil.which, os.path.isfile)
            apps.body.addWidget(self._row(entry.label or key, found or "Not found on this PC", found is not None))
        if not profile.apps:
            apps.body.addWidget(muted("No apps yet."))
        self.host.addWidget(apps)

        folders = Panel("Folders", f"{len(profile.folders)}")
        for key, entry in profile.folders.items():
            path = expand_path(entry.path)
            folders.body.addWidget(self._row(entry.label or key, str(path), path.is_dir()))
        if not profile.folders:
            folders.body.addWidget(muted("No folders yet."))
        self.host.addWidget(folders)

        music = Panel("Music", "play_media")
        if profile.music.folder:
            path = expand_path(profile.music.folder)
            player_key = profile.resolve("apps", profile.music.player)
            player = profile.label("apps", player_key) if player_key else profile.music.player
            music.body.addWidget(self._row("Music folder", str(path), path.is_dir()))
            music.body.addWidget(self._row("Player", player, player_key is not None))
        else:
            music.body.addWidget(muted("No music folder set. Add one under music in profile.yaml."))
        self.host.addWidget(music)

        sites = Panel("Websites", f"{len(profile.websites)}")
        for key, entry in profile.websites.items():
            sites.body.addWidget(self._row(entry.label or key, entry.url, True))
        self.host.addWidget(sites)
        self.host.addStretch(1)

    @staticmethod
    def _row(name: str, detail: str, ok: bool) -> Card:
        card = Card()
        card.row.addWidget(dot(theme.TEAL if ok else theme.RED))
        label = QLabel(name)
        label.setStyleSheet("font-weight:600;")
        label.setFixedWidth(150)
        card.row.addWidget(label)
        text = QLabel(detail)
        text.setObjectName("muted")
        text.setStyleSheet(f"color:{theme.TEXT_MUTED};")
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        card.row.addWidget(text, 1)
        return card


# =========================================================================== settings


class SettingsPage(QWidget):
    """Your name, your home, your microphone and speakers, and where the files live."""

    _finished = Signal(str, object)   # (what finished, its result) from a background thread

    def __init__(
        self,
        engine: Engine,
        parent: QWidget | None = None,
        *,
        devices: Callable[[], tuple[list[AudioDevice], list[AudioDevice]]] = list_devices,
        measure: Callable[[list[AudioDevice]], dict[int, float]] = measure_levels,
        tone: Callable[[int | str | None], None] = play_test_tone,
        geocode: Callable[[str], Place | None] = find_place,
        config_path: Path | None = None,
        build_report: Callable[[], str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self._devices = devices
        self._measure = measure
        self._tone = tone
        self._geocode = geocode
        self._config_path = config_path or resolve_config_path()
        self._build_report = build_report or (lambda: build_diagnostics(engine.settings, engine))
        self._inputs: list[AudioDevice] = []
        self._outputs: list[AudioDevice] = []
        self._finished.connect(self._on_finished)
        settings = engine.settings

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)
        host = QWidget()
        scroll.setWidget(host)
        root = QVBoxLayout(host)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(12)
        title = QLabel("SETTINGS")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(muted("Settings live in plain files, so you can see exactly what they say."))

        # ---- you and your home
        you = Panel("You and your home", "saved to your settings and profile files")
        self.name_input = QLineEdit(settings.ui.greeting_name)
        self.name_input.setPlaceholderText("What should JARVIS call you?")
        self.name_input.setAccessibleName("Name JARVIS calls you")
        self.name_input.returnPressed.connect(self.save_name)
        save_name = QPushButton("Save name")
        save_name.setObjectName("primary")
        save_name.clicked.connect(self.save_name)
        you.body.addLayout(self._form_row("Call me", self.name_input, save_name))

        location = engine.profile.location
        self.address_input = QLineEdit(location.address if location else "")
        self.address_input.setPlaceholderText("e.g. 6365 Nellmapius Drive, Nellmapius, Pretoria")
        self.address_input.setAccessibleName("Home address")
        self.address_input.returnPressed.connect(self.save_address)
        save_address = QPushButton("Save address")
        save_address.setObjectName("primary")
        save_address.clicked.connect(self.save_address)
        you.body.addLayout(self._form_row("Home address", self.address_input, save_address))
        you.body.addWidget(muted(
            'Used for "find a pharmacy near me" and "directions to ...". It stays in profile.yaml on this PC; '
            "only the suburb name is looked up for the weather, never the street.", "dim",
        ))
        self.you_status = muted("", "muted")
        you.body.addWidget(self.you_status)
        root.addWidget(you)

        # ---- audio
        audio = Panel("Microphone and speakers", "applies right away")
        self.input_combo = QComboBox()
        self.input_combo.setAccessibleName("Microphone")
        self.output_combo = QComboBox()
        self.output_combo.setAccessibleName("Speakers")
        self.detect_button = QPushButton("Detect microphone")
        self.detect_button.clicked.connect(self.detect_microphone)
        self.tone_button = QPushButton("Test speakers")
        self.tone_button.clicked.connect(self.test_speakers)
        audio.body.addLayout(self._form_row("Microphone", self.input_combo, self.detect_button))
        audio.body.addLayout(self._form_row("Speakers", self.output_combo, self.tone_button))
        self.levels_host = QVBoxLayout()
        self.levels_host.setSpacing(4)
        audio.body.addLayout(self.levels_host)
        buttons = QHBoxLayout()
        self.apply_audio = QPushButton("Use these devices")
        self.apply_audio.setObjectName("primary")
        self.apply_audio.clicked.connect(self.save_devices)
        refresh = QPushButton("Refresh list")
        refresh.clicked.connect(self.load_devices)
        buttons.addWidget(self.apply_audio)
        buttons.addWidget(refresh)
        buttons.addStretch(1)
        audio.body.addLayout(buttons)
        audio.body.addWidget(muted(
            "Detect: make a sound into the microphone you want (tap it or speak) and press the button. "
            "JARVIS listens on every microphone for a moment and picks the one that heard you.", "dim",
        ))
        self.audio_status = muted("", "muted")
        audio.body.addWidget(self.audio_status)
        root.addWidget(audio)

        # ---- current settings
        panel = Panel("Current settings")
        voice = settings.voice
        rows = [
            ("Language model", f"{settings.ollama.model} (Ollama, on this PC)"),
            ("Speech recognition", f"Whisper {voice.stt.model} on the CPU"),
            ("Voice", f"Piper {voice.tts.voice}"),
            ("Push to talk", voice.push_to_talk_key),
            ("Memory", "on" if settings.memory.enabled else "off"),
            ("Speak typed replies", "yes" if settings.ui.speak_typed_replies else "no"),
            ("Quick commands (skip the model)", "on" if settings.brain.fast_commands else "off"),
            ("Memory database", str(settings.resolve_database_path())),
            ("Profile file", str(settings.resolve_profile_path())),
            ("Settings file", str(self._config_path)),
            ("Logs", str(settings.resolve_log_dir())),
        ]
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(8)
        for index, (name, value) in enumerate(rows):
            grid.addWidget(muted(name, "muted"), index, 0)
            label = QLabel(value)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setWordWrap(True)
            grid.addWidget(label, index, 1)
        grid.setColumnStretch(1, 1)
        panel.body.addLayout(grid)
        root.addWidget(panel)

        files = QHBoxLayout()
        profile_path = settings.resolve_profile_path()
        for text, path in (
            ("Open profile.yaml", profile_path),
            ("Open settings.yaml", self._config_path),
            ("Open logs folder", settings.resolve_log_dir()),
            ("Open data folder", settings.resolve_database_path().parent),
        ):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, p=path: open_local(p))
            files.addWidget(button)
        self.copy_button = QPushButton("Copy diagnostics")
        self.copy_button.setObjectName("primary")
        self.copy_button.setToolTip("Copies a report you can paste to get help. It includes recent log lines.")
        self.copy_button.clicked.connect(self.copy_diagnostics)
        files.addWidget(self.copy_button)
        files.addStretch(1)
        root.addLayout(files)
        self.diag_status = muted("", "muted")
        root.addWidget(self.diag_status)

        safety = Panel("What JARVIS will never do")
        for line in (
            "Run arbitrary commands or scripts. It can only use the tools listed on the Tools page.",
            "Open anything that is not in your profile (apps, folders, websites, music folder).",
            "Send an email. It only opens a Gmail draft for you to read and send.",
            "Delete anything without asking you first. Every destructive action shows a confirmation dialog.",
        ):
            safety.body.addWidget(muted(line))
        root.addWidget(safety)
        root.addStretch(1)
        self.load_devices()

    @staticmethod
    def _form_row(label: str, field: QWidget, button: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        name = muted(label, "muted")
        name.setFixedWidth(110)
        row.addWidget(name)
        row.addWidget(field, 1)
        row.addWidget(button)
        return row

    # ------------------------------------------------------------------ background work

    def _background(self, key: str, work: Callable[[], object]) -> None:
        def run() -> None:
            try:
                result: object = work()
            except Exception as exc:
                result = exc
            self._finished.emit(key, result)

        threading.Thread(target=run, name=f"jarvis-settings-{key}", daemon=True).start()

    def _on_finished(self, key: str, result: object) -> None:
        if key == "address":
            self._address_done(result)
        elif key == "detect":
            self._detect_done(result)
        elif key == "tone":
            self.tone_button.setEnabled(True)
            self.audio_status.setText(f"Couldn't play a sound there: {result}" if isinstance(result, Exception) else "Did you hear it?")

    # ------------------------------------------------------------------ name and address

    def save_name(self) -> None:
        try:
            name = save_greeting_name(self._config_path, self.name_input.text())
        except (ValueError, OSError) as exc:
            self.you_status.setText(str(exc))
            return
        self.engine.settings.ui.greeting_name = name
        self.name_input.setText(name)
        self.you_status.setText(f'Saved. I\'ll call you {name} when the dashboard opens.')

    def save_address(self) -> None:
        text = self.address_input.text()
        self.you_status.setText("Saving...")
        profile_path = self.engine.settings.resolve_profile_path()
        self._background("address", lambda: save_home_address(profile_path, text, self._geocode))

    def _address_done(self, result: object) -> None:
        if isinstance(result, Exception):
            self.you_status.setText(str(result))
            return
        message, place = result  # type: ignore[misc]
        location = self.engine.profile.location
        text = " ".join(self.address_input.text().split())
        if location is not None:
            location.address = text
            if place is not None:
                location.name = place.name.split(",")[0]
                location.latitude = round(place.latitude, 4)
                location.longitude = round(place.longitude, 4)
        self.you_status.setText(message)

    # ------------------------------------------------------------------ audio

    def load_devices(self) -> None:
        self._inputs, self._outputs = self._devices()
        voice = self.engine.settings.voice
        for combo, items, current in (
            (self.input_combo, self._inputs, voice.input_device),
            (self.output_combo, self._outputs, voice.output_device),
        ):
            combo.clear()
            combo.addItem("System default", None)
            for device in items:
                combo.addItem(device.label, device.index)
            position = combo.findData(current if isinstance(current, int) else None)
            combo.setCurrentIndex(max(position, 0))
        if not self._inputs and not self._outputs:
            self.audio_status.setText("I couldn't list your audio devices. Is the audio system available?")
        else:
            self.audio_status.setText(f"{len(self._inputs)} microphones and {len(self._outputs)} outputs found.")

    def detect_microphone(self) -> None:
        if not self._inputs:
            self.audio_status.setText("No microphones to listen on.")
            return
        self.detect_button.setEnabled(False)
        self.audio_status.setText("Listening on each microphone... make a sound now.")
        inputs = list(self._inputs)
        self._background("detect", lambda: self._measure(inputs))

    def _detect_done(self, result: object) -> None:
        self.detect_button.setEnabled(True)
        clear_layout(self.levels_host)
        if isinstance(result, Exception):
            self.audio_status.setText(f"I couldn't listen: {result}")
            return
        levels: dict[int, float] = result  # type: ignore[assignment]
        names = {d.index: d.name for d in self._inputs}
        for index, level in sorted(levels.items(), key=lambda item: item[1], reverse=True):
            row = QHBoxLayout()
            label = QLabel(f"{index} · {names.get(index, '?')}")
            label.setFixedWidth(340)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(min(100, int(level * 200)))
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            row.addWidget(label)
            row.addWidget(bar, 1)
            holder = QWidget()
            holder.setLayout(row)
            self.levels_host.addWidget(holder)
        winner = loudest(levels)
        if winner is None:
            self.audio_status.setText("I couldn't tell which microphone has sound. Try again a little louder, one microphone at a time.")
            return
        self.input_combo.setCurrentIndex(max(0, self.input_combo.findData(winner)))
        self.audio_status.setText(f"Heard you on {names.get(winner, winner)}. Press \"Use these devices\" to switch to it.")

    def test_speakers(self) -> None:
        device = self.output_combo.currentData()
        self.tone_button.setEnabled(False)
        self.audio_status.setText("Playing a short tone...")
        self._background("tone", lambda: self._tone(device))

    def save_devices(self) -> None:
        mic = self.input_combo.currentData()
        speakers = self.output_combo.currentData()
        try:
            save_audio_device(self._config_path, "input", mic)
            save_audio_device(self._config_path, "output", speakers)
        except (ValueError, OSError) as exc:
            self.audio_status.setText(str(exc))
            return
        voice = self.engine.settings.voice
        voice.input_device = mic
        voice.output_device = speakers
        self.engine.restart_voice()
        self.audio_status.setText("Saved. Restarting the voice with these devices (a few seconds).")

    # ------------------------------------------------------------------ diagnostics

    def copy_diagnostics(self) -> None:
        report = self._build_report()
        QGuiApplication.clipboard().setText(report)
        self.diag_status.setText(f"Copied {len(report.splitlines())} lines. Paste them into the chat.")
