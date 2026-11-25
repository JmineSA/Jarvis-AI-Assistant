"""Your personal profile: the apps, folders and websites JARVIS is allowed to open.

Everything JARVIS can open comes from this list (config/profile.yaml). Nothing
outside it can be opened, so a misheard or invented name can never launch
something unexpected.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from jarvis.config import ConfigError

logger = logging.getLogger(__name__)

Kind = Literal["apps", "folders", "websites", "contacts"]

# Words people add when speaking ("open the VS Code app", "my portfolio folder").
_FILLER = {"the", "a", "my", "please", "app", "application", "program", "folder", "directory",
           "website", "site", "page"}


def normalize(text: str) -> str:
    """'Power BI', 'power-bi' and 'PowerBI app' all become 'powerbi'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    kept = [w for w in words if w not in _FILLER] or words
    return "".join(kept)


def expand_path(value: str) -> Path:
    """Expand ~ and environment variables such as %USERPROFILE%."""
    return Path(os.path.expandvars(os.path.expanduser(value)))


class AppEntry(BaseModel):
    label: str | None = None
    names: list[str] = Field(default_factory=list)       # other ways to say it
    candidates: list[str] = Field(default_factory=list)  # program names or paths, tried in order
    args: list[str] = Field(default_factory=list)
    # How a folder is handed to the app: "arg" = as an argument (VS Code),
    # "cwd" = the app starts inside it (Jupyter), "none" = not supported.
    folder_mode: Literal["none", "arg", "cwd"] = "none"


class FolderEntry(BaseModel):
    label: str | None = None
    names: list[str] = Field(default_factory=list)
    path: str


class SiteEntry(BaseModel):
    label: str | None = None
    names: list[str] = Field(default_factory=list)
    url: str = ""
    app: str | None = None  # key of an app to open it with, e.g. chrome


class ContactEntry(BaseModel):
    label: str | None = None
    names: list[str] = Field(default_factory=list)
    email: str


class Location(BaseModel):
    """Where you are, for the weather. Coordinates are sent to the weather service, nothing else."""

    name: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class MusicEntry(BaseModel):
    folder: str = ""         # the folder that holds your albums
    player: str = "vlc"      # key of the app (under `apps`) that plays them


class Profile(BaseModel):
    owner_name: str | None = None
    location: Location | None = None
    music: MusicEntry = Field(default_factory=MusicEntry)
    apps: dict[str, AppEntry] = Field(default_factory=dict)
    folders: dict[str, FolderEntry] = Field(default_factory=dict)
    websites: dict[str, SiteEntry] = Field(default_factory=dict)
    contacts: dict[str, ContactEntry] = Field(default_factory=dict)

    def _entries(self, kind: Kind) -> dict:
        return getattr(self, kind)

    def label(self, kind: Kind, key: str) -> str:
        return self._entries(kind)[key].label or key

    def labels(self, kind: Kind) -> list[str]:
        return [self.label(kind, key) for key in self._entries(kind)]

    def resolve(self, kind: Kind, query: str) -> str | None:
        """Find the entry a spoken name refers to, tolerating small speech errors."""
        index: dict[str, str] = {}
        for key, entry in self._entries(kind).items():
            for name in (key, entry.label or key, *entry.names):
                index.setdefault(normalize(name), key)
        wanted = normalize(query)
        if not wanted:
            return None
        if wanted in index:
            return index[wanted]
        close = difflib.get_close_matches(wanted, list(index), n=1, cutoff=0.8)
        return index[close[0]] if close else None

    def spoken_names(self) -> list[str]:
        """Names to hint to the speech recogniser so it hears them correctly."""
        names: list[str] = []
        for kind in ("apps", "folders", "websites", "contacts"):
            for key, entry in self._entries(kind).items():  # type: ignore[arg-type]
                names.append(entry.label or key)
        return list(dict.fromkeys(names))


def load_profile(path: Path) -> Profile:
    """Load the profile. A missing file gives an empty profile; a broken one is an error."""
    if not path.exists():
        logger.warning("Profile %s not found; no apps, folders or websites are set up.", path)
        return Profile()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read profile {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Profile {path} must contain a YAML mapping.")
    try:
        return Profile.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid profile {path}: {exc}") from exc
