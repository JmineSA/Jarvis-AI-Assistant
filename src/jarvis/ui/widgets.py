"""Reusable pieces: panels, the glowing core, ring gauges, the voice waveform, the chat list."""

from __future__ import annotations

import math
from typing import ClassVar

from PySide6.QtCore import QByteArray, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from jarvis.ui import theme

# Stroke icons (24x24 grid), the same shapes as the design.
ICONS = {
    "grid": "M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z",
    "chat": "M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z",
    "memory": "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6",
    "clock": "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 7v5l3 2",
    "tools": "M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.5 2.5-2.4-.6-.6-2.4z",
    "music": "M9 18V5l12-2v13M9 18a3 3 0 1 1-6 0 3 3 0 0 1 6 0zM21 16a3 3 0 1 1-6 0 3 3 0 0 1 6 0z",
    "gear": "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z",
    "mic": "M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3zM19 10v2a7 7 0 0 1-14 0v-2M12 19v3",
    "search": "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM21 21l-4.3-4.3",
    "bell": "M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9M13.7 21a2 2 0 0 1-3.4 0",
    "send": "M22 2L11 13M22 2l-7 20-4-9-9-4z",
    "cloud": "M7 16a5 5 0 1 1 9.6-2H17a3.5 3.5 0 0 1 0 7H8a3 3 0 0 1-1-5.8",
    "pin": "M12 21s-7-6.2-7-11a7 7 0 0 1 14 0c0 4.8-7 11-7 11zM12 12a2 2 0 1 0 0-4 2 2 0 0 0 0 4z",
    "shield": "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z",
    "globe": "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18",
    "mail": "M4 4h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zM22 6l-10 7L2 6",
    "folder": "M4 4h6l2 3h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z",
    "check": "M5 12l5 5L20 7",
    "code": "M16 18l6-6-6-6M8 6l-6 6 6 6",
    "hex": "M12 2l9 5v10l-9 5-9-5V7z",
    "trash": "M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 11v6M14 11v6",
}


def icon(name: str, color: str = theme.ACCENT_GLOW, size: int = 18) -> QIcon:
    """A crisp icon drawn from the shared stroke paths."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="{ICONS[name]}"/></svg>'
    )
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    scale = 2
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(scale)
    return QIcon(pixmap)


def icon_label(name: str, color: str = theme.ACCENT_GLOW, size: int = 16) -> QLabel:
    label = QLabel()
    label.setPixmap(icon(name, color, size).pixmap(size, size))
    label.setFixedSize(size, size)
    return label


# --------------------------------------------------------------------------- panels and cards


class Panel(QFrame):
    """A titled box. Put content in `body`."""

    def __init__(self, title: str, note: str = "", name: str = "panel", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(name)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)
        header = QHBoxLayout()
        self.title_label = QLabel(title.upper())
        self.title_label.setObjectName("panelTitle")
        self.note_label = QLabel(note)
        self.note_label.setObjectName("panelNote")
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.note_label)
        outer.addLayout(header)
        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        self.body.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self.body, 1)

    def set_title(self, text: str) -> None:
        self.title_label.setText(text.upper())

    def set_note(self, text: str) -> None:
        self.note_label.setText(text)


class Card(QFrame):
    """A small rounded row (an overview line, a tool, a fact)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(10, 8, 10, 8)
        self.row.setSpacing(10)


def icon_chip(name: str, color: str, size: int = 30) -> QLabel:
    """A rounded square holding an icon, like the ones in the overview."""
    label = QLabel()
    label.setFixedSize(size, size)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet(f"background:{theme.CHIP};border-radius:8px;border:none;")
    label.setPixmap(icon(name, color, 16).pixmap(16, 16))
    return label


def dot(color: str, size: int = 8) -> QLabel:
    label = QLabel()
    label.setFixedSize(size, size)
    label.setStyleSheet(f"background:{color};border-radius:{size // 2}px;border:none;")
    return label


def set_dot(label: QLabel, color: str) -> None:
    size = label.width()
    label.setStyleSheet(f"background:{color};border-radius:{size // 2}px;border:none;")


def muted(text: str, name: str = "muted") -> QLabel:
    label = QLabel(text)
    label.setObjectName(name)
    label.setWordWrap(True)
    return label


# --------------------------------------------------------------------------- animation switch

_animations = True


def animations_enabled() -> bool:
    return _animations


def set_animations(enabled: bool) -> None:
    """Turn the moving parts (spinning rings, waveform) on or off."""
    global _animations
    _animations = enabled


# --------------------------------------------------------------------------- the core


class CoreWidget(QWidget):
    """The glowing JARVIS sphere. Colour and caption follow the assistant's state."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = "idle"
        self._angle = 0.0
        self._breath = 0.0
        self._ripple = 0.0
        self._detail_left = ""
        self._detail_right = ""
        self.setMinimumSize(420, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._advance)

    @property
    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        if state not in theme.STATE_COLORS:
            state = "idle"
        if state != self._state:
            self._state = state
            self.update()

    def set_details(self, left: str, right: str) -> None:
        self._detail_left, self._detail_right = left, right
        self.update()

    def showEvent(self, event) -> None:
        if animations_enabled():
            self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    # per state: (ring spin speed, breathing speed, breathing depth, ripple speed; 0 = no ripples)
    MOTION: ClassVar[dict[str, tuple[float, float, float, float]]] = {
        "idle": (0.5, 0.05, 0.012, 0.0),
        "listening": (1.4, 0.16, 0.045, 0.012),
        "thinking": (3.2, 0.30, 0.030, 0.0),
        "speaking": (1.8, 0.24, 0.060, 0.016),
    }

    def _motion(self) -> tuple[float, float, float, float]:
        return self.MOTION.get(self._state, self.MOTION["idle"])

    def _advance(self) -> None:
        spin, breath, _depth, ripple = self._motion()
        self._angle = (self._angle + spin * 1.2) % 3600
        self._breath += breath
        self._ripple = (self._ripple + ripple) % 1.0 if ripple else 0.0
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        color = QColor(theme.STATE_COLORS[self._state])

        # backdrop
        backdrop = QRadialGradient(rect.center(), max(rect.width(), rect.height()) * 0.6)
        backdrop.setColorAt(0.0, QColor("#0C2C4A"))
        backdrop.setColorAt(0.6, QColor("#08182A"))
        backdrop.setColorAt(1.0, QColor("#071220"))
        path_pen = QPen(QColor(theme.BORDER), 1)
        painter.setPen(path_pen)
        painter.setBrush(QBrush(backdrop))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 12, 12)

        centre = rect.center() + QPointF(0, -10)
        radius = min(rect.width() * 0.3, rect.height() * 0.38)
        depth = self._motion()[2] if animations_enabled() else 0.0
        radius *= 1.0 + depth * math.sin(self._breath)  # the orb breathes; faster and deeper while it talks

        # ripples: a ring leaves the orb and fades while it is listening or speaking
        if animations_enabled() and self._motion()[3]:
            for offset in (0.0, 0.5):
                t = (self._ripple + offset) % 1.0
                painter.setPen(QPen(theme.qcolor(theme.STATE_COLORS[self._state], 0.45 * (1.0 - t)), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(centre, radius * (1.05 + 0.9 * t), radius * (1.05 + 0.9 * t))

        # orbit rings
        for width_f, height_f, tilt, spin, tint, alpha in (
            (1.9, 0.6, -18.0, 1.0, theme.ACCENT_GLOW, 0.35),
            (1.6, 0.92, 24.0, -0.6, theme.TEAL, 0.28),
        ):
            painter.save()
            painter.translate(centre)
            painter.rotate(tilt + self._angle * spin * 0.1)
            painter.setPen(QPen(theme.qcolor(tint, alpha), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(0, 0), radius * width_f, radius * height_f)
            painter.restore()

        # glow
        glow = QRadialGradient(centre, radius * 1.7)
        glow.setColorAt(0.0, theme.qcolor(theme.STATE_COLORS[self._state], 0.30 + 0.25 * depth * 10 * (0.5 + 0.5 * math.sin(self._breath))))
        glow.setColorAt(1.0, theme.qcolor(theme.STATE_COLORS[self._state], 0.0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(centre, radius * 1.7, radius * 1.7)

        # sphere
        sphere = QRadialGradient(centre + QPointF(-radius * 0.25, -radius * 0.3), radius * 1.3)
        sphere.setColorAt(0.0, theme.qcolor(theme.ACCENT_BRIGHT, 0.55))
        sphere.setColorAt(0.45, QColor(14, 84, 120, 140))
        sphere.setColorAt(1.0, QColor(6, 24, 42, 230))
        painter.setPen(QPen(theme.qcolor(theme.ACCENT_GLOW, 0.55), 1))
        painter.setBrush(QBrush(sphere))
        painter.drawEllipse(centre, radius, radius)

        # wireframe
        painter.setPen(QPen(QColor(150, 235, 250, 70), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for rx, ry in ((1.0, 0.4), (1.0, 0.76), (0.4, 1.0), (0.76, 1.0)):
            painter.drawEllipse(centre, radius * rx, radius * ry)
        painter.drawLine(QPointF(centre.x() - radius, centre.y()), QPointF(centre.x() + radius, centre.y()))
        painter.drawLine(QPointF(centre.x(), centre.y() - radius), QPointF(centre.x(), centre.y() + radius))

        # wordmark
        painter.setPen(QColor("#C9F6FF"))
        painter.setFont(theme.display_font(max(26, int(radius * 0.34)), QFont.Weight.Bold, spacing=9))
        title_rect = QRectF(rect.left(), centre.y() - radius * 0.42, rect.width(), radius * 0.5)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, "JARVIS")
        painter.setPen(QColor(theme.ACCENT_BRIGHT))
        painter.setFont(theme.body_font(12))
        font = painter.font()
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 4)
        painter.setFont(font)
        painter.drawText(
            QRectF(rect.left(), centre.y() + radius * 0.06, rect.width(), 20),
            Qt.AlignmentFlag.AlignCenter,
            "LOCAL AI CORE",
        )

        # state pill
        word = theme.STATE_WORDS[self._state]
        pill_font = theme.body_font(12, QFont.Weight.Medium)
        pill_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        painter.setFont(pill_font)
        text_width = painter.fontMetrics().horizontalAdvance(word)
        pill = QRectF(0, 0, text_width + 46, 26)
        pill.moveCenter(QPointF(centre.x(), centre.y() + radius * 0.42))
        painter.setPen(QPen(color, 1))
        painter.setBrush(QColor(4, 14, 26, 190))
        painter.drawRoundedRect(pill, 13, 13)
        pulse = 0.65 + 0.35 * math.sin(self._angle / 20.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(theme.qcolor(theme.STATE_COLORS[self._state], pulse))
        painter.drawEllipse(QPointF(pill.left() + 15, pill.center().y()), 4, 4)
        painter.setPen(color)
        painter.drawText(
            QRectF(pill.left() + 26, pill.top(), pill.width() - 30, pill.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            word,
        )

        # footer captions
        painter.setPen(QColor(theme.TEXT_DIM))
        painter.setFont(theme.body_font(11))
        painter.drawText(
            QRectF(rect.left() + 16, rect.bottom() - 26, rect.width() / 2, 18),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._detail_left,
        )
        painter.drawText(
            QRectF(rect.center().x(), rect.bottom() - 26, rect.width() / 2 - 16, 18),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            self._detail_right,
        )
        painter.end()


# --------------------------------------------------------------------------- gauges and waveform


class RingGauge(QWidget):
    """A circular percentage gauge with a name and value in the middle."""

    def __init__(self, name: str, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._name = name
        self._color = color
        self._percent: float | None = 0.0
        self._text = "0%"
        self.setFixedSize(84, 84)

    @property
    def text(self) -> str:
        return self._text

    @property
    def percent(self) -> float | None:
        return self._percent

    def set_value(self, percent: float | None, text: str | None = None) -> None:
        self._percent = None if percent is None else max(0.0, min(100.0, percent))
        self._text = text if text is not None else ("n/a" if percent is None else f"{percent:.0f}%")
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        box = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        track = QPen(QColor("#12324A"), 7)
        painter.setPen(track)
        painter.drawEllipse(box)
        if self._percent:
            arc = QPen(QColor(self._color), 7)
            arc.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(arc)
            painter.drawArc(box, 90 * 16, int(-self._percent / 100.0 * 360 * 16))
        painter.setPen(QColor(theme.TEXT_MUTED))
        painter.setFont(theme.body_font(11))
        painter.drawText(QRectF(0, 22, self.width(), 16), Qt.AlignmentFlag.AlignCenter, self._name)
        painter.setPen(QColor(theme.TEXT))
        painter.setFont(theme.body_font(14, QFont.Weight.DemiBold))
        painter.drawText(QRectF(0, 38, self.width(), 20), Qt.AlignmentFlag.AlignCenter, self._text)
        painter.end()


class Waveform(QWidget):
    """Little animated bars that show JARVIS is listening or speaking."""

    HEIGHTS = (8, 14, 22, 12, 28, 18, 10, 24, 30, 16, 9, 20, 26, 12, 18, 28, 14, 8, 22, 16, 10, 24, 12, 18, 8, 14)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._active = False
        self._phase = 0.0
        self._color = theme.ACCENT_GLOW
        self.setFixedHeight(36)
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._advance)

    def set_state(self, state: str) -> None:
        self._active = state in ("listening", "speaking", "thinking")
        self._color = theme.STATE_COLORS.get(state, theme.ACCENT_GLOW) if self._active else "#2A4A63"
        if self._active and animations_enabled() and self.isVisible():
            self._timer.start()
        elif not self._active:
            self._timer.stop()
        self.update()

    def showEvent(self, event) -> None:
        if self._active and animations_enabled():
            self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def _advance(self) -> None:
        self._phase += 0.35
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        count = len(self.HEIGHTS)
        step = 6
        left = (self.width() - count * step) / 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self._color))
        for i, base in enumerate(self.HEIGHTS):
            scale = 0.35 + 0.65 * abs(math.sin(self._phase + i * 0.55)) if self._active else 1.0
            h = max(4.0, base * scale)
            painter.drawRoundedRect(QRectF(left + i * step, (self.height() - h) / 2, 3, h), 1.5, 1.5)
        painter.end()


# --------------------------------------------------------------------------- chat list

_BUBBLE_STYLES = {
    "user": "background:#0E3550;border:1px solid #1B4F73;border-radius:10px;border-bottom-right-radius:2px;",
    "assistant": f"background:{theme.CARD};border:1px solid {theme.BORDER_SOFT};border-radius:10px;border-bottom-left-radius:2px;",
    "system": f"background:transparent;border:1px dashed {theme.BORDER_SOFT};border-radius:10px;color:{theme.TEXT_MUTED};",
}
_WHO = {"user": "You", "assistant": "JARVIS", "system": "System"}


class ChatView(QScrollArea):
    """A scrolling list of chat bubbles. Newest at the bottom."""

    MAX_MESSAGES = 200

    def __init__(self, owner: str = "You", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._owner = owner
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        self._layout = QVBoxLayout(host)
        self._layout.setContentsMargins(0, 0, 6, 0)
        self._layout.setSpacing(8)
        self._layout.addStretch(1)
        self.setWidget(host)
        self._rows: list[QWidget] = []
        self.verticalScrollBar().rangeChanged.connect(self._stick_to_bottom)

    @property
    def count(self) -> int:
        return len(self._rows)

    def texts(self) -> list[tuple[str, str]]:
        return [(row.property("role"), row.property("text")) for row in self._rows if row.property("role")]

    def add_message(self, role: str, text: str) -> None:
        if role not in _BUBBLE_STYLES:
            role = "system"
        row = QWidget()
        row.setProperty("role", role)
        row.setProperty("text", text)
        box = QVBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        who = QLabel(self._owner if role == "user" else _WHO[role])
        who.setObjectName("dim")
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bubble.setStyleSheet(_BUBBLE_STYLES[role] + "padding:8px 10px;")
        bubble.setMaximumWidth(520)
        align = Qt.AlignmentFlag.AlignRight if role == "user" else Qt.AlignmentFlag.AlignLeft
        box.addWidget(who, 0, align)
        # A plain row (not an aligned label) so long replies wrap instead of being cut off.
        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        if role == "user":
            line.addStretch(1)
            line.addWidget(bubble)
        else:
            line.addWidget(bubble)
            line.addStretch(1)
        box.addLayout(line)
        self._layout.insertWidget(self._layout.count() - 1, row)
        self._rows.append(row)
        while len(self._rows) > self.MAX_MESSAGES:
            old = self._rows.pop(0)
            self._layout.removeWidget(old)
            old.deleteLater()

    def add_tool_tag(self, text: str) -> None:
        """A small 'weather · done' tag under the last message."""
        tag = QLabel(text)
        tag.setStyleSheet(
            f"color:{theme.TEAL};background:#08201F;border:1px solid #14504F;border-radius:6px;padding:2px 8px;font-size:11px;"
        )
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(tag)
        row.addStretch(1)
        self._layout.insertWidget(self._layout.count() - 1, holder)
        self._rows.append(holder)

    def clear_messages(self) -> None:
        for row in self._rows:
            self._layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

    def _stick_to_bottom(self, _low: int, high: int) -> None:
        self.verticalScrollBar().setValue(high)
