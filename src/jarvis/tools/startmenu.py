"""Find installed programs by name from the Windows Start menu, so you need not list each one.

Two sources, both just lists of what is installed:
- Start menu shortcuts (.lnk files) in the all-users and your own "Programs" folders.
- Microsoft Store apps, read with one fixed Windows command (`Get-StartApps`).

JARVIS only ever starts an entry that appears in these lists, by opening the shortcut itself
or by Windows' own app launcher with the app's registered ID. A spoken name is only used to
choose between entries, never as part of a command.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_SECONDS = 600
_WORD = re.compile(r"[a-z0-9]+")
_FILLER = {"the", "a", "an", "my", "app", "application", "program", "please", "microsoft", "windows"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9._!\-]+(?:_[A-Za-z0-9]+![A-Za-z0-9._\-]+)?$|^[A-Za-z0-9._!\-]+$")
_SKIP = re.compile(r"\b(?:uninstall|readme|help|documentation|license|website|release notes|manual)\b", re.IGNORECASE)


@dataclass(frozen=True)
class StartApp:
    name: str
    kind: str          # "shortcut" or "store"
    target: str        # the .lnk path, or the app's registered ID
    tokens: tuple[str, ...]


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(w for w in dict.fromkeys(_WORD.findall(text.lower())) if w not in _FILLER)


def shortcut_folders(env: dict[str, str] | None = None) -> list[Path]:
    env = os.environ if env is None else env
    folders = []
    if env.get("PROGRAMDATA"):
        folders.append(Path(env["PROGRAMDATA"]) / "Microsoft/Windows/Start Menu/Programs")
    if env.get("APPDATA"):
        folders.append(Path(env["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs")
    return folders


def scan_shortcuts(folders: list[Path]) -> list[StartApp]:
    found: list[StartApp] = []
    for folder in folders:
        if not folder.is_dir():
            continue
        for current, _dirs, files in os.walk(folder):
            for name in files:
                if name.lower().endswith(".lnk") and not _SKIP.search(name):
                    stem = Path(name).stem
                    found.append(StartApp(stem, "shortcut", str(Path(current) / name), _tokens(stem)))
    return found


def _store_apps_via_powershell() -> list[dict]:
    """Windows only. One fixed command; nothing from the user is put into it."""
    if os.name != "nt":
        return []
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", "Get-StartApps | ConvertTo-Json -Compress"],
        capture_output=True, text=True, timeout=12, check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    data = json.loads(result.stdout or "[]")
    return data if isinstance(data, list) else [data]


def scan_store_apps(reader: Callable[[], list[dict]] = _store_apps_via_powershell) -> list[StartApp]:
    try:
        rows = reader()
    except (OSError, ValueError, subprocess.SubprocessError):
        logger.info("Could not read Microsoft Store apps", exc_info=True)
        return []
    apps = []
    for row in rows:
        name, app_id = str(row.get("Name", "")).strip(), str(row.get("AppID", "")).strip()
        # Only entries with a plain app ID are launchable by Windows' own launcher.
        if name and app_id and "\\" not in app_id and "/" not in app_id and _SAFE_ID.match(app_id) and not _SKIP.search(name):
            apps.append(StartApp(name, "store", app_id, _tokens(name)))
    return apps


class StartMenu:
    """The list of installed programs, read once and reused for a while."""

    def __init__(
        self,
        folders: Callable[[], list[Path]] | None = None,
        store_reader: Callable[[], list[dict]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # Looked up when used (not when defined) so tests can switch the real Windows lookup off.
        self._folders = folders or (lambda: shortcut_folders())
        self._store_reader = store_reader or (lambda: _store_apps_via_powershell())
        self._scan_lock = threading.Lock()
        self._clock = clock
        self._lock = threading.Lock()
        self._apps: list[StartApp] = []
        self._at: float | None = None

    def apps(self) -> list[StartApp]:
        with self._lock:
            if self._at is not None and self._clock() - self._at < CACHE_SECONDS:
                return self._apps
        with self._scan_lock:  # one scan at a time: others wait for it instead of starting their own
            with self._lock:
                if self._at is not None and self._clock() - self._at < CACHE_SECONDS:
                    return self._apps
            shortcuts = scan_shortcuts(self._folders())
            store = scan_store_apps(self._store_reader)
            merged: dict[str, StartApp] = {}
            for app in (*shortcuts, *store):  # shortcuts win over a store entry of the same name
                merged.setdefault(app.name.lower(), app)
            with self._lock:
                self._apps, self._at = sorted(merged.values(), key=lambda a: a.name.lower()), self._clock()
                return self._apps

    def prewarm(self) -> None:
        threading.Thread(target=self.apps, name="jarvis-startmenu", daemon=True).start()

    def find(self, spoken: str, cutoff: float = 0.8) -> list[StartApp]:
        """Best matches for a spoken name: exact name, then all words, then close spelling."""
        wanted = _tokens(spoken)
        if not wanted:
            return []
        apps = self.apps()
        squashed = "".join(wanted)
        exact = [a for a in apps if "".join(a.tokens) == squashed]
        if exact:
            return exact[:1]
        every = [a for a in apps if all(w in a.tokens for w in wanted)]
        if every:
            fewest = min(len(a.tokens) for a in every)  # "Chrome" beats "Chrome Remote Desktop"
            return [a for a in every if len(a.tokens) == fewest]
        names = {"".join(a.tokens): a for a in apps if a.tokens}
        close = difflib.get_close_matches(squashed, list(names), n=3, cutoff=cutoff)
        return [names[c] for c in close[:1]]
