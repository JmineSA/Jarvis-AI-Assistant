"""The only code that starts programs or opens paths and addresses.

Programs are started from an argument list with no shell, so spoken text can
never be interpreted as a command.
"""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Protocol


class Launcher(Protocol):
    def start(self, argv: list[str], cwd: Path | None = None) -> None:
        """Start a program, detached from JARVIS. Raises OSError if it cannot start."""

    def open_path(self, path: Path) -> None:
        """Open a folder in the file manager. Raises OSError on failure."""

    def open_url(self, url: str) -> None:
        """Open an address in the default browser. Raises OSError on failure."""


class SystemLauncher:
    def start(self, argv: list[str], cwd: Path | None = None) -> None:
        extra: dict = {}
        if os.name == "nt":
            extra["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            extra["start_new_session"] = True
        subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            **extra,
        )

    def open_path(self, path: Path) -> None:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            self.start([opener, str(path)])

    def open_url(self, url: str) -> None:
        if not webbrowser.open(url):
            raise OSError("no web browser could be started")
