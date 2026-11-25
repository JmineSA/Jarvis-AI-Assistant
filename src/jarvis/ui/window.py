"""The main window: sidebar, top bar, the pages, and the bottom bar with the talk button."""

from __future__ import annotations

import logging
import threading
from datetime import datetime

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from jarvis.ui import theme
from jarvis.ui.engine import Engine
from jarvis.ui.metrics import Metrics, read_metrics
from jarvis.ui.pages import (
    AppsPage,
    CommandCenterPage,
    ConversationsPage,
    MemoryPage,
    SettingsPage,
    ToolsPage,
)
from jarvis.ui.widgets import Waveform, dot, icon, muted, set_dot

logger = logging.getLogger(__name__)

NAV_ITEMS = (
    ("command", "Command Center", "grid"),
    ("conversations", "Conversations", "chat"),
    ("memory", "Memory", "memory"),
    ("tools", "Tools", "tools"),
    ("apps", "Apps and Music", "music"),
    ("settings", "Settings", "gear"),
)

PAGE_TITLES = {
    "command": "COMMAND CENTER",
    "conversations": "CONVERSATIONS",
    "memory": "MEMORY",
    "tools": "TOOLS",
    "apps": "APPS AND MUSIC",
    "settings": "SETTINGS",
}


class MainWindow(QMainWindow):
    _metricsReady = Signal(object)

    def __init__(self, engine: Engine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.setWindowTitle("JARVIS")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 760)
        self._metrics_busy = threading.Event()
        self._confirm_open = False

        central = QWidget()
        central.setObjectName("root")
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_sidebar())

        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self._build_topbar())
        self.stack = QStackedWidget()
        self.pages = {
            "command": CommandCenterPage(engine),
            "conversations": ConversationsPage(engine),
            "memory": MemoryPage(engine),
            "tools": ToolsPage(engine),
            "apps": AppsPage(engine),
            "settings": SettingsPage(engine),
        }
        for page in self.pages.values():
            self.stack.addWidget(page)
        column.addWidget(self.stack, 1)
        column.addWidget(self._build_footer())
        outer.addLayout(column, 1)

        self.command: CommandCenterPage = self.pages["command"]  # type: ignore[assignment]
        self.memory_page: MemoryPage = self.pages["memory"]  # type: ignore[assignment]
        self.conversations: ConversationsPage = self.pages["conversations"]  # type: ignore[assignment]

        self.command.navigate.connect(self.show_page)
        self.memory_page.changed.connect(self.refresh_memory)

        engine.stateChanged.connect(self.set_state)
        engine.messageAdded.connect(self.command.add_message)
        engine.toolFinished.connect(self.command.add_tool_event)
        engine.confirmRequested.connect(self.ask_confirmation)
        engine.memoryChanged.connect(self.refresh_memory)
        engine.healthChanged.connect(self.set_health)
        engine.voiceChanged.connect(self.set_voice)
        engine.weatherChanged.connect(self.weather_chip.setText)
        self._metricsReady.connect(self.command.update_metrics)

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        self.metrics_timer = QTimer(self)
        self.metrics_timer.timeout.connect(self.poll_metrics)
        self.metrics_timer.start(1000)
        self.update_clock()
        self.show_page("command")

    # ------------------------------------------------------------------ building

    def _build_sidebar(self) -> QWidget:
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(230)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(6)

        brand = QLabel("JARVIS")
        brand.setObjectName("brand")
        sub = QLabel("PERSONAL AI ASSISTANT")
        sub.setObjectName("brandSub")
        layout.addWidget(brand)
        layout.addWidget(sub)
        layout.addSpacing(18)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons: dict[str, QPushButton] = {}
        for key, label, icon_name in NAV_ITEMS:
            button = QPushButton(label)
            button.setObjectName("nav")
            button.setCheckable(True)
            button.setIcon(icon(icon_name, theme.TEXT_SOFT, 18))
            button.setIconSize(QSize(18, 18))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, k=key: self.show_page(k))
            self.nav_group.addButton(button)
            self.nav_buttons[key] = button
            layout.addWidget(button)
        layout.addStretch(1)

        voice = QFrame()
        voice.setObjectName("voicePanel")
        inner = QVBoxLayout(voice)
        inner.setContentsMargins(14, 12, 14, 12)
        inner.setSpacing(4)
        inner.addWidget(muted("VOICE", "panelTitle"))
        self.voice_state = QLabel("IDLE")
        self.voice_state.setStyleSheet(f"color:{theme.ACCENT_GLOW};font-weight:600;")
        inner.addWidget(self.voice_state)
        self.voice_hint = muted(theme.STATE_HINTS["idle"], "dim")
        self.voice_hint.setWordWrap(True)
        inner.addWidget(self.voice_hint)
        self.waveform = Waveform()
        self.waveform.setFixedHeight(38)
        inner.addWidget(self.waveform)
        layout.addWidget(voice)
        return side

    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("topbar")
        bar.setFixedHeight(64)
        row = QHBoxLayout(bar)
        row.setContentsMargins(24, 0, 24, 0)
        row.setSpacing(12)
        self.page_title = QLabel(PAGE_TITLES["command"])
        self.page_title.setObjectName("pageTitle")
        row.addWidget(self.page_title)
        row.addSpacing(12)
        self.status_dot = dot(theme.TEXT_DIM, 9)
        self.status_text = QLabel("Starting...")
        self.status_text.setObjectName("muted")
        row.addWidget(self.status_dot)
        row.addWidget(self.status_text)
        row.addStretch(1)

        self.clock = QLabel("")
        self.clock.setObjectName("clock")
        row.addWidget(self.clock)
        row.addSpacing(8)
        for name, tip, target in (
            ("search", "Search conversations", "conversations"),
            ("bell", "Reminders and memory", "memory"),
            ("gear", "Settings", "settings"),
        ):
            button = QPushButton()
            button.setObjectName("iconButton")
            button.setIcon(icon(name, theme.TEXT_SOFT, 18))
            button.setToolTip(tip)
            button.setAccessibleName(tip)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, t=target: self.show_page(t))
            row.addWidget(button)
        self.operator = QLabel(self.engine.owner_name)
        self.operator.setObjectName("chip")
        self.operator.setFixedHeight(32)
        row.addWidget(self.operator)
        return bar

    def _build_footer(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("footer")
        bar.setFixedHeight(72)
        row = QHBoxLayout(bar)
        row.setContentsMargins(24, 10, 24, 10)
        row.setSpacing(12)

        profile = self.engine.profile
        place = profile.location.name if profile.location else "No location set"
        self.location_chip = self._chip("pin", place)
        self.weather_chip = self._chip("cloud", "Weather: waiting")
        privacy = self._chip("shield", "100% local · no cloud")
        for chip in (self.location_chip, self.weather_chip, privacy):
            row.addWidget(chip)
        row.addStretch(1)

        self.talk = QPushButton("  TALK TO JARVIS")
        self.talk.setObjectName("talkWide")
        self.talk.setIcon(icon("mic", "#04121E", 20))
        self.talk.setIconSize(QSize(20, 20))
        self.talk.setCursor(Qt.CursorShape.PointingHandCursor)
        self.talk.setProperty("recording", False)
        self.talk.clicked.connect(self.toggle_talk)
        row.addWidget(self.talk)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Type a command...")
        self.input.setAccessibleName("Type a command")
        self.input.setMinimumWidth(320)
        self.input.returnPressed.connect(self.send_text)
        row.addWidget(self.input, 1)
        send = QPushButton()
        send.setObjectName("primary")
        send.setIcon(icon("send", "#04121E", 18))
        send.setAccessibleName("Send")
        send.setToolTip("Send")
        send.clicked.connect(self.send_text)
        row.addWidget(send)
        return bar

    @staticmethod
    def _chip(icon_name: str, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("chip")
        return label

    # ------------------------------------------------------------------ navigation

    def show_page(self, key: str) -> None:
        if key not in self.pages:
            return
        self.stack.setCurrentWidget(self.pages[key])
        self.page_title.setText(PAGE_TITLES[key])
        self.nav_buttons[key].setChecked(True)
        page = self.pages[key]
        refresh = getattr(page, "refresh", None)
        if callable(refresh):
            refresh()

    def current_page(self) -> str:
        current = self.stack.currentWidget()
        return next((k for k, v in self.pages.items() if v is current), "")

    # ------------------------------------------------------------------ actions

    def send_text(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.show_page("command")
        self.engine.submit_text(text)

    def toggle_talk(self) -> None:
        recording = self.engine.talk_pressed()
        self.talk.setProperty("recording", bool(recording))
        self.talk.setText("  LISTENING... TAP TO SEND" if recording else "  TALK TO JARVIS")
        self.talk.style().unpolish(self.talk)
        self.talk.style().polish(self.talk)

    def set_state(self, state: str) -> None:
        self.command.set_state(state)
        self.waveform.set_state(state)
        self.voice_state.setText(theme.STATE_WORDS.get(state, state.upper()))
        self.voice_state.setStyleSheet(f"color:{theme.STATE_COLORS.get(state, theme.ACCENT_GLOW)};font-weight:600;")
        self.voice_hint.setText(theme.STATE_HINTS.get(state, ""))
        if state != "listening" and self.talk.property("recording"):
            self.talk.setProperty("recording", False)
            self.talk.setText("  TALK TO JARVIS")
            self.talk.style().unpolish(self.talk)
            self.talk.style().polish(self.talk)

    def set_health(self, ready: bool, detail: str) -> None:
        self.command.set_health(ready, detail)
        set_dot(self.status_dot, theme.TEAL if ready else theme.RED)
        self.status_text.setText("All systems online" if ready else detail)

    def set_voice(self, ready: bool, detail: str) -> None:
        self.command.set_voice(ready, detail)

    def refresh_memory(self) -> None:
        self.command.refresh_memory()
        page = self.current_page()
        if page == "memory":
            self.memory_page.refresh()
        elif page == "conversations":
            self.conversations.refresh()

    def ask_confirmation(self, request_id: int, text: str) -> None:
        """JARVIS wants to do something risky: nothing happens unless you press Yes."""
        self.activateWindow()
        answer = QMessageBox.question(
            self, "JARVIS needs your OK", f"{text}\n\nAllow this?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        self.engine.answer_confirmation(request_id, answer == QMessageBox.StandardButton.Yes)

    # ------------------------------------------------------------------ timers

    def update_clock(self) -> None:
        self.clock.setText(datetime.now().strftime("%H:%M:%S  ·  %a %d %b"))

    def poll_metrics(self) -> None:
        """Read the numbers on a worker thread (nvidia-smi can take a moment)."""
        if self._metrics_busy.is_set() or not self.isVisible():
            return
        self._metrics_busy.set()

        def work() -> None:
            try:
                metrics: Metrics = read_metrics()
                self._metricsReady.emit(metrics)
            except Exception:
                logger.debug("Could not read system metrics", exc_info=True)
            finally:
                self._metrics_busy.clear()

        threading.Thread(target=work, name="jarvis-metrics", daemon=True).start()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.clock_timer.stop()
        self.metrics_timer.stop()
        self.engine.stop()
        super().closeEvent(event)
