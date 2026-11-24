"""The JARVIS web interface.

Run it with::

    python -m jarvis ui

or directly::

    streamlit run src/jarvis/ui/streamlit_app.py

The interface has three parts:

* the **chat**, which talks to the same brain as the terminal and voice modes,
* the **system panel**, which shows the live state of the machine, and
* the **memory browser**, which shows what JARVIS remembers and lets you
  search it, add to it and delete from it.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from jarvis.config import load_settings
from jarvis.ui.backend import Backend, pick_backend
from jarvis.ui.theme import CSS, pill, usage_bar, welcome_panel

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="JARVIS",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------- setup


@st.cache_resource(show_spinner=False)
def get_settings():
    return load_settings()


@st.cache_resource(show_spinner=False)
def get_backend() -> Backend:
    """One backend for the life of the interface.

    ``ui.backend: auto`` uses the running API server when it answers, and the
    in-process brain when it does not.
    """
    return pick_backend(get_settings())


@st.cache_data(ttl=3.0, show_spinner=False)
def read_status(backend_label: str) -> dict[str, Any]:
    """A fresh reading of the system. Cached for a few seconds."""
    backend = get_backend()
    try:
        return backend.status()
    except Exception as exc:  # the panel must never take the page down
        logger.exception("Could not read the status.")
        return {"status": "error", "checks": [], "snapshot": None, "detail": str(exc)}


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit in {"B", "KB", "MB"} else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"  # pragma: no cover


def metric_box(label: str, value: str, note: str = "") -> str:
    note_html = f'<div class="jarvis-metric-note">{note}</div>' if note else ""
    return (
        f'<div class="jarvis-metric">'
        f'<div class="jarvis-metric-label">{label}</div>'
        f'<div class="jarvis-metric-value">{value}</div>'
        f"{note_html}</div>"
    )


# -------------------------------------------------------------------- sidebar


@st.fragment(run_every=3.0)
def system_panel() -> None:
    """The live machine panel. Refreshes itself every few seconds."""
    backend = get_backend()
    status = read_status(backend.label)
    snapshot = status.get("snapshot") or {}

    st.markdown('<div class="jarvis-section">System</div>', unsafe_allow_html=True)

    if not snapshot:
        st.warning("The system state is not available right now.")
        return

    cpu = float(snapshot.get("cpu_percent", 0.0))
    memory_percent = float(snapshot.get("memory_percent", 0.0))
    memory_used = float(snapshot.get("memory_used_gb", 0.0))
    memory_total = float(snapshot.get("memory_total_gb", 0.0))
    cores = int(snapshot.get("cpu_count", 1) or 1)

    st.markdown(
        metric_box(
            "CPU",
            f"{cpu:.0f}%",
            f"{cores} cores · up {snapshot.get('uptime_human', '?')}",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(usage_bar(cpu), unsafe_allow_html=True)

    st.markdown(
        metric_box(
            "Memory",
            f"{memory_percent:.0f}%",
            f"{memory_used:.1f} / {memory_total:.1f} GB",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(usage_bar(memory_percent), unsafe_allow_html=True)

    gpus = snapshot.get("gpu") or []
    if gpus:
        gpu = gpus[0]
        gpu_percent = float(gpu.get("utilization_percent", 0))
        st.markdown(
            metric_box(
                "GPU",
                f"{gpu_percent:.0f}%",
                f"{gpu.get('name', 'GPU')} · "
                f"{float(gpu.get('memory_used_mb', 0)):.0f}/"
                f"{float(gpu.get('memory_total_mb', 0)):.0f} MB",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(usage_bar(gpu_percent), unsafe_allow_html=True)

    net_sent = float(snapshot.get("net_sent_mb_per_sec", 0.0))
    net_recv = float(snapshot.get("net_recv_mb_per_sec", 0.0))
    st.markdown(
        metric_box(
            "Network",
            f"↓ {net_recv:.1f} MB/s",
            f"↑ {net_sent:.1f} MB/s · {snapshot.get('process_count', 0)} processes",
        ),
        unsafe_allow_html=True,
    )

    disks = snapshot.get("disks") or []
    if disks:
        with st.expander("Disks", expanded=False):
            for disk in disks[:4]:
                st.markdown(
                    f"**{disk.get('mount', '?')}** · {disk.get('percent', 0):.0f}% "
                    f"({disk.get('used_gb', 0):.1f} / {disk.get('total_gb', 0):.1f} GB)",
                    unsafe_allow_html=True,
                )


@st.fragment(run_every=5.0)
def health_panel() -> None:
    """The health of every part of JARVIS."""
    backend = get_backend()
    status = read_status(backend.label)

    st.markdown('<div class="jarvis-section">Health</div>', unsafe_allow_html=True)
    checks = status.get("checks") or []
    if not checks:
        st.caption("No health information available.")
        return

    badges = []
    for check in checks:
        badges.append(pill(str(check.get("status", "unknown")), str(check.get("name", "?"))))
    st.markdown("".join(badges), unsafe_allow_html=True)

    for check in checks:
        detail = str(check.get("detail", ""))
        if detail:
            st.caption(f"{check.get('name', '')}: {detail}")


def memory_browser() -> None:
    """Search, read and edit what JARVIS remembers."""
    backend = get_backend()
    st.markdown('<div class="jarvis-section">Memory</div>', unsafe_allow_html=True)

    query = st.text_input("Search memory", key="mem_query", placeholder="dog, LightsOn, …")
    if st.button("Search", key="mem_search_btn", use_container_width=True):
        st.session_state["mem_results"] = backend.memory_search(query, limit=25)

    results = st.session_state.get("mem_results")
    if results is None:
        results = backend.memory_search("", limit=10)
        st.session_state["mem_results"] = results

    for hit in results:
        fact_class = "mem-fact" if hit.get("kind") == "fact" else ""
        key_part = (
            f'<span class="mem-key">{hit.get("key")}</span>: ' if hit.get("key") else ""
        )
        st.markdown(
            f'<div class="mem-row {fact_class}">{key_part}{hit.get("content", "")}'
            f'<div class="mem-meta">#{hit.get("id")} · {hit.get("date_label", "")} · '
            f'{hit.get("kind", "")}</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Delete", key=f"del_{hit.get('id')}", help="Forget this memory"):
            backend.forget(int(hit.get("id", 0)))
            st.session_state["mem_results"] = backend.memory_search(query, limit=25)
            st.rerun()

    with st.expander("Add a fact", expanded=False):
        fact = st.text_input("Fact", key="mem_new_fact", placeholder="My dog is called Rex")
        fact_key = st.text_input("Key (optional)", key="mem_new_key", placeholder="user.dog.name")
        if st.button("Remember", key="mem_add_btn"):
            if fact.strip():
                backend.remember(fact.strip(), fact_key.strip() or None)
                st.session_state["mem_results"] = backend.memory_search(query, limit=25)
                st.rerun()

    facts = backend.memory_facts()
    if facts:
        with st.expander(f"Facts ({len(facts)})", expanded=False):
            for fact in facts:
                st.markdown(
                    f'<div class="mem-row mem-fact">'
                    f'<span class="mem-key">{fact.get("key")}</span>: {fact.get("content", "")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    if st.button("Clear this conversation's memory", key="mem_clear_btn"):
        backend.clear_memory()
        st.session_state["mem_results"] = []
        st.rerun()


with st.sidebar:
    st.markdown('<div class="jarvis-brand">◈ Jarvis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="jarvis-sub">Local assistant console</div>', unsafe_allow_html=True
    )

    backend = get_backend()
    st.caption(f"Backend: {backend.label}")
    st.markdown("")

    health_panel()
    st.markdown("")
    system_panel()
    st.markdown("")

    memory_browser()

    st.markdown(
        '<div class="jarvis-foot">Phase 4 + 5 · memory &amp; monitoring</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------- main

settings = get_settings()

st.markdown('<div class="jarvis-brand">◈ Jarvis</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="jarvis-sub">{settings.ui.title} · {settings.ollama.model} · '
    f"local, private, yours</div>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

messages: list[dict[str, str]] = st.session_state["messages"]

if not messages:
    st.markdown(welcome_panel(settings.ollama.model), unsafe_allow_html=True)

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask JARVIS anything, or tell it to do something on your computer")
if prompt:
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                reply = backend.chat(prompt)
        except Exception as exc:
            logger.exception("The chat failed.")
            reply = f"I could not answer that: {exc}"
        st.markdown(reply)

    messages.append({"role": "assistant", "content": reply})

if messages:
    left, right = st.columns([1, 5])
    with left:
        if st.button("Clear chat", use_container_width=True):
            backend.reset()
            st.session_state["messages"] = []
            st.rerun()
    with right:
        st.caption(
            "JARVIS answers from what its tools really reported, and remembers the "
            "conversation between sessions."
        )
