"""Starts the desktop app: python -m jarvis app"""

from __future__ import annotations

import logging
import os
import sys

from PySide6.QtWidgets import QApplication

from jarvis.config import Settings
from jarvis.ui import theme, widgets
from jarvis.ui.engine import Engine
from jarvis.ui.window import MainWindow

logger = logging.getLogger(__name__)


def build_window(settings: Settings, *, use_voice: bool = True, engine: Engine | None = None) -> MainWindow:
    """Create the engine and window (the QApplication must already exist)."""
    engine = engine or Engine(settings, use_voice=use_voice)
    return MainWindow(engine)


def run_app(settings: Settings, *, use_voice: bool = True) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("JARVIS")
    theme.load_bundled_fonts()
    app.setStyleSheet(theme.stylesheet())
    # Respect "reduce motion": JARVIS_REDUCE_MOTION=1 stops the spinning rings and waveform.
    if os.environ.get("JARVIS_REDUCE_MOTION") == "1":
        widgets.set_animations(False)

    window = build_window(settings, use_voice=use_voice)
    window.engine.start()
    window.show()
    code = app.exec()
    window.engine.stop()
    return code
