"""The JARVIS web interface (Phase 5).

``streamlit_app`` holds the interface itself. Importing this package does not
start Streamlit, so the other modules can use the backends without it.
"""

from jarvis.ui.backend import ApiBackend, Backend, LocalBackend, build_local_backend, pick_backend

__all__ = [
    "ApiBackend",
    "Backend",
    "LocalBackend",
    "build_local_backend",
    "pick_backend",
]
