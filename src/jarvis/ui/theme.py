"""The look of the JARVIS web interface.

A dark, cyan-accented theme that matches the rest of the project. It is plain
CSS in a string, so the interface has no build step and no extra dependency.
"""

from __future__ import annotations

CSS = """
<style>
    :root {
        --jarvis-cyan: #35e0ff;
        --jarvis-dim: #7d95a6;
        --jarvis-ink: #0a1016;
        --jarvis-panel: #111a23;
        --jarvis-line: #1d2b38;
    }

    .stApp {
        background:
            radial-gradient(1100px 620px at 12% -8%, rgba(53, 224, 255, 0.10), transparent 60%),
            radial-gradient(900px 520px at 92% 4%, rgba(53, 224, 255, 0.06), transparent 62%),
            var(--jarvis-ink);
    }

    .block-container {
        padding-top: 2.1rem;
        padding-bottom: 2.4rem;
        max-width: 1400px;
    }

    /* ---------------------------------------------------------- headings */
    .jarvis-brand {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        letter-spacing: 0.34em;
        text-transform: uppercase;
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--jarvis-cyan);
        text-shadow: 0 0 22px rgba(53, 224, 255, 0.45);
    }
    .jarvis-sub {
        color: var(--jarvis-dim);
        font-size: 0.76rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin: 0.15rem 0 1.15rem;
    }
    .jarvis-section {
        color: var(--jarvis-dim);
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin: 0.2rem 0 0.55rem;
    }

    /* ------------------------------------------------------------ panels */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--jarvis-line);
        border-radius: 12px;
        background: rgba(17, 26, 35, 0.72);
        padding: 0.85rem 1rem;
    }

    /* --------------------------------------------------------- chat log */
    .stChatMessage {
        background: rgba(17, 26, 35, 0.55);
        border: 1px solid var(--jarvis-line);
        border-radius: 12px;
        padding: 0.2rem 0.4rem;
        margin-bottom: 0.55rem;
    }
    [data-testid="stChatMessageContent"] p {
        line-height: 1.55;
    }

    /* --------------------------------------------------------- indicators */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.16rem 0.6rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        border: 1px solid var(--jarvis-line);
        background: rgba(17, 26, 35, 0.9);
        color: var(--jarvis-dim);
        margin: 0.15rem 0.3rem 0.15rem 0;
    }
    .pill-ok      { color: #46e08a; border-color: rgba(70, 224, 138, 0.45); }
    .pill-warn    { color: #ffc857; border-color: rgba(255, 200, 87, 0.45); }
    .pill-error   { color: #ff6b6b; border-color: rgba(255, 107, 107, 0.45); }
    .pill-unknown { color: var(--jarvis-dim); }

    .jarvis-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-ok      { background: #46e08a; box-shadow: 0 0 8px #46e08a; }
    .dot-warn    { background: #ffc857; box-shadow: 0 0 8px #ffc857; }
    .dot-error   { background: #ff6b6b; box-shadow: 0 0 8px #ff6b6b; }
    .dot-unknown { background: var(--jarvis-dim); }

    /* ----------------------------------------------------------- metrics */
    .jarvis-metric {
        border: 1px solid var(--jarvis-line);
        border-radius: 10px;
        padding: 0.55rem 0.75rem;
        background: rgba(10, 16, 22, 0.65);
    }
    .jarvis-metric-label {
        color: var(--jarvis-dim);
        font-size: 0.66rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    .jarvis-metric-value {
        color: var(--jarvis-cyan);
        font-size: 1.32rem;
        font-weight: 700;
        line-height: 1.25;
    }
    .jarvis-metric-note {
        color: var(--jarvis-dim);
        font-size: 0.72rem;
    }

    /* ------------------------------------------------- usage bars (own) */
    .jarvis-bar {
        height: 6px;
        border-radius: 999px;
        background: rgba(53, 224, 255, 0.13);
        overflow: hidden;
        margin: 0.35rem 0 0.75rem;
    }
    .jarvis-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #35e0ff, #46e08a);
        box-shadow: 0 0 10px rgba(53, 224, 255, 0.45);
        transition: width 0.4s ease;
    }
    .jarvis-bar-fill.warn {
        background: linear-gradient(90deg, #ffc857, #ff9f43);
        box-shadow: 0 0 10px rgba(255, 200, 87, 0.45);
    }
    .jarvis-bar-fill.hot {
        background: linear-gradient(90deg, #ff9f43, #ff6b6b);
        box-shadow: 0 0 10px rgba(255, 107, 107, 0.45);
    }

    /* --------------------------------------------------- welcome state */
    .jarvis-welcome {
        border: 1px solid var(--jarvis-line);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        background: rgba(17, 26, 35, 0.55);
        margin-top: 1.2rem;
    }
    .jarvis-welcome h3 {
        margin: 0 0 0.5rem;
        color: var(--jarvis-cyan);
        font-size: 1.05rem;
        letter-spacing: 0.06em;
    }
    .jarvis-welcome ul {
        margin: 0.4rem 0 0;
        padding-left: 1.1rem;
        color: var(--jarvis-dim);
        font-size: 0.88rem;
        line-height: 1.75;
    }
    .jarvis-welcome li b {
        color: #b9c9d6;
        font-weight: 600;
    }

    /* ---------------------------------------------------------- mem rows */
    .mem-row {
        border: 1px solid var(--jarvis-line);
        border-left: 2px solid var(--jarvis-cyan);
        border-radius: 8px;
        padding: 0.45rem 0.6rem;
        margin-bottom: 0.4rem;
        background: rgba(10, 16, 22, 0.6);
        font-size: 0.82rem;
    }
    .mem-meta {
        color: var(--jarvis-dim);
        font-size: 0.68rem;
        letter-spacing: 0.05em;
    }
    .mem-fact { border-left-color: #46e08a; }
    .mem-fact .mem-key { color: #46e08a; font-weight: 600; }

    /* --------------------------------------------------------- side text */
    section[data-testid="stSidebar"] {
        background: rgba(8, 13, 18, 0.75);
        border-right: 1px solid var(--jarvis-line);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem;
    }
    .jarvis-foot {
        color: var(--jarvis-dim);
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-align: center;
        padding-top: 0.8rem;
    }
</style>
"""


def pill(status: str, label: str) -> str:
    """A small coloured status badge."""
    tone = status if status in {"ok", "warn", "error"} else "unknown"
    return (
        f'<span class="pill pill-{tone}">'
        f'<span class="jarvis-dot dot-{tone}"></span>{label}</span>'
    )


def usage_bar(percent: float) -> str:
    """A usage bar drawn in HTML, so it matches the theme.

    Streamlit's own progress bar cannot be recoloured reliably, because its
    inner markup changes between releases.
    """
    value = max(0.0, min(float(percent), 100.0))
    tone = "hot" if value >= 90 else ("warn" if value >= 75 else "")
    return (
        f'<div class="jarvis-bar"><div class="jarvis-bar-fill {tone}" '
        f'style="width:{value:.1f}%"></div></div>'
    )


def welcome_panel(model: str) -> str:
    """What the interface shows before the first question."""
    return (
        '<div class="jarvis-welcome">'
        "<h3>At your service</h3>"
        "<ul>"
        f"<li><b>Talk to me.</b> I run on <b>{model}</b> on your own machine, "
        "so nothing leaves this computer.</li>"
        "<li><b>Ask me to act.</b> Open your apps and folders, list what is in "
        "them, check the CPU and the time.</li>"
        "<li><b>I remember.</b> I keep our conversations and the facts you ask me "
        "to keep, and bring the relevant ones back into every question.</li>"
        "<li><b>Look around.</b> The panel on the left shows the live state of "
        "your machine and everything I remember.</li>"
        "</ul></div>"
    )
