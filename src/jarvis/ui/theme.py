"""Colours, fonts and the stylesheet. One place, so the whole app can be re-skinned."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase

# --- palette (taken from the design) ---------------------------------------------------
BG = "#050D18"
BG_DEEP = "#040A13"
PANEL_TOP = "#0B1A2D"
PANEL_BOTTOM = "#08121F"
CARD = "#0A1626"
BORDER = "#14405C"
BORDER_SOFT = "#123049"
ACCENT = "#22D3EE"
ACCENT_BRIGHT = "#67E8F9"
ACCENT_GLOW = "#38E1FF"
TEAL = "#2DD4BF"
PURPLE = "#A78BFA"
AMBER = "#F5B942"
RED = "#F87171"
TEXT = "#E6F4FF"
TEXT_SOFT = "#A9C4D8"
TEXT_MUTED = "#8FB0C8"
TEXT_DIM = "#6F93AD"
CHIP = "#0E2A44"
SELECTED_TOP = "#0C3150"
SELECTED_BOTTOM = "#08203A"

STATE_COLORS = {
    "idle": TEXT_MUTED,
    "listening": ACCENT_GLOW,
    "thinking": PURPLE,
    "speaking": TEAL,
}
STATE_WORDS = {"idle": "IDLE", "listening": "LISTENING", "thinking": "THINKING", "speaking": "SPEAKING"}
STATE_HINTS = {
    "idle": "Waiting for you",
    "listening": "Listening...",
    "thinking": "Thinking...",
    "speaking": "Speaking...",
}

DISPLAY_FAMILIES = ["Orbitron", "Bahnschrift", "Segoe UI"]
BODY_FAMILIES = ["Inter", "Segoe UI", "Arial"]

FONT_DIR = Path(__file__).with_name("fonts")


def load_bundled_fonts() -> int:
    """Load any .ttf/.otf dropped into ui/fonts (for example Orbitron and Inter)."""
    loaded = 0
    if FONT_DIR.is_dir():
        for path in sorted(FONT_DIR.iterdir()):
            if path.suffix.lower() in {".ttf", ".otf"} and QFontDatabase.addApplicationFont(str(path)) >= 0:
                loaded += 1
    return loaded


def display_font(size: int, weight: QFont.Weight = QFont.Weight.Medium, spacing: float = 0.0) -> QFont:
    font = QFont()
    font.setFamilies(DISPLAY_FAMILIES)
    font.setPixelSize(size)
    font.setWeight(weight)
    if spacing:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return font


def body_font(size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont()
    font.setFamilies(BODY_FAMILIES)
    font.setPixelSize(size)
    font.setWeight(weight)
    return font


def qcolor(value: str, alpha: float = 1.0) -> QColor:
    color = QColor(value)
    color.setAlphaF(alpha)
    return color


def stylesheet() -> str:
    families = ", ".join(f'"{f}"' for f in BODY_FAMILIES)
    display = ", ".join(f'"{f}"' for f in DISPLAY_FAMILIES)
    return f"""
* {{ font-family: {families}; color: {TEXT}; font-size: 13px; }}
QMainWindow, QWidget#root {{ background: {BG_DEEP}; }}
QWidget {{ background: transparent; }}
QToolTip {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 8px; }}

QWidget#sidebar {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #071422, stop:1 #050D18);
                  border-right: 1px solid {BORDER_SOFT}; }}
QWidget#topbar {{ border-bottom: 1px solid {BORDER_SOFT}; }}
QWidget#footer {{ background: {BG}; border-top: 1px solid {BORDER_SOFT}; }}

QFrame#panel {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {PANEL_TOP}, stop:1 {PANEL_BOTTOM});
               border: 1px solid {BORDER}; border-radius: 12px; }}
QFrame#panelDanger {{ background: #12090E; border: 1px solid #5B2A33; border-radius: 12px; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER_SOFT}; border-radius: 8px; }}
QFrame#voicePanel {{ background: {PANEL_BOTTOM}; border: 1px solid {BORDER}; border-radius: 12px; }}

QLabel#panelTitle {{ color: {ACCENT_BRIGHT}; font-size: 11px; font-weight: 600; letter-spacing: 2px; }}
QLabel#panelNote {{ color: {TEXT_DIM}; font-size: 11px; }}
QLabel#muted {{ color: {TEXT_MUTED}; }}
QLabel#dim {{ color: {TEXT_DIM}; font-size: 11px; }}
QLabel#pageTitle {{ font-family: {display}; font-size: 20px; letter-spacing: 2px; }}
QLabel#bigNumber {{ font-family: {display}; font-size: 22px; }}
QLabel#clock {{ font-family: {display}; font-size: 26px; color: {ACCENT_GLOW}; }}
QLabel#chip {{ background: {CHIP}; border: 1px solid {BORDER_SOFT}; border-radius: 14px; padding: 5px 12px; color: {TEXT_SOFT}; }}
QLabel#brand {{ font-family: {display}; font-size: 20px; font-weight: 700; letter-spacing: 3px; }}
QLabel#brandSub {{ color: {TEXT_DIM}; font-size: 11px; letter-spacing: 3px; }}

QPushButton {{ background: {CARD}; border: 1px solid {BORDER_SOFT}; border-radius: 8px; padding: 6px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:focus {{ border-color: {ACCENT_BRIGHT}; }}
QPushButton:disabled {{ color: {TEXT_DIM}; }}
QPushButton#nav {{ background: transparent; border: 1px solid transparent; text-align: left;
                   padding: 0 12px; height: 40px; font-size: 14px; color: {TEXT_SOFT}; }}
QPushButton#nav:hover {{ background: #0A1A2C; }}
QPushButton#nav:checked {{ border: 1px solid {ACCENT}; color: {TEXT};
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {SELECTED_TOP}, stop:1 {SELECTED_BOTTOM}); }}
QPushButton#primary {{ background: {SELECTED_TOP}; border: 1px solid {ACCENT}; font-weight: 500; }}
QPushButton#quick {{ text-align: left; height: 20px; padding: 4px 10px; }}
QPushButton#danger {{ background: #1A0E14; border: 1px solid #7A3A45; color: #F8A5A5; }}
QPushButton#danger:hover {{ border-color: {RED}; }}
QPushButton#iconButton {{ min-width: 36px; max-width: 36px; min-height: 36px; max-height: 36px; padding: 0; }}
QPushButton#talkWide {{ border: 1.5px solid {ACCENT}; border-radius: 24px; height: 44px;
        background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {SELECTED_TOP}, stop:1 {SELECTED_BOTTOM}); }}
QPushButton#talkWide[recording="true"] {{ border-color: {RED}; }}

QLineEdit {{ background: {PANEL_BOTTOM}; border: 1px solid {BORDER}; border-radius: 10px; padding: 0 12px;
            height: 38px; selection-background-color: {SELECTED_TOP}; }}
QLineEdit:focus {{ border-color: {ACCENT_BRIGHT}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QTableWidget {{ background: transparent; border: none; gridline-color: {BORDER_SOFT};
               selection-background-color: {SELECTED_TOP}; }}
QHeaderView::section {{ background: {CARD}; color: {TEXT_MUTED}; border: none;
               border-bottom: 1px solid {BORDER_SOFT}; padding: 6px 8px; font-size: 11px; }}

QMessageBox, QInputDialog, QDialog {{ background: {PANEL_BOTTOM}; }}
"""
