"""Play an album (or a single song) from your music folder in your media player.

Only the music folder from config/profile.yaml is searched, and the player is started
from an argument list (no shell), so spoken words can never become a command.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path

from pydantic import Field

from jarvis.profile import Profile, expand_path
from jarvis.tools.apps import find_executable
from jarvis.tools.launcher import Launcher
from jarvis.tools.registry import ToolArgs, ToolResult

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
MAX_DEPTH = 2          # Music/Artist/Album: albums may sit one or two levels down
MAX_ENTRIES = 20000    # never walk forever on a huge drive
_FILLER = {"the", "a", "an", "my", "album", "song", "track", "music", "by", "from", "play", "please",
           "in", "on", "vlc", "folder", "some", "of", "for"}
_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _FILLER]


class PlayMediaArgs(ToolArgs):
    name: str = ""


def _candidates(root: Path) -> list[Path]:
    """Folders and audio files up to MAX_DEPTH below the music folder."""
    found: list[Path] = []
    seen = 0
    for current, dirs, files in os.walk(root):
        depth = len(Path(current).relative_to(root).parts)
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        if depth >= MAX_DEPTH:
            dirs[:] = []
        for name in dirs:
            found.append(Path(current) / name)
        for name in files:
            if Path(name).suffix.lower() in AUDIO_EXTENSIONS:
                found.append(Path(current) / name)
        seen += len(dirs) + len(files)
        if seen > MAX_ENTRIES:
            logger.warning("Music folder is very large; only part of it was searched.")
            break
    return found


def _score(wanted: list[str], root: Path, path: Path) -> int:
    """How many of the spoken words appear in the path below the music folder."""
    relative = " ".join(path.relative_to(root).with_suffix("").parts if path.is_file() else path.relative_to(root).parts)
    haystack = " ".join(_words(relative))
    return sum(1 for w in wanted if w in haystack)


def _title(root: Path, path: Path) -> str:
    return " - ".join(path.relative_to(root).with_suffix("").parts) if path.is_file() else path.name


def make_play_media(
    profile: Profile,
    launcher: Launcher,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.isfile,
) -> Callable[[PlayMediaArgs], ToolResult]:
    def run(args: PlayMediaArgs) -> ToolResult:
        if not profile.music.folder:
            return ToolResult.failed("No music folder is set. Add it under music in config/profile.yaml.")
        root = expand_path(profile.music.folder)
        if not root.is_dir():
            return ToolResult.failed(
                f"I can't find your music folder at {root}. Check the path under music in config/profile.yaml."
            )

        player_key = profile.resolve("apps", profile.music.player)
        if player_key is None:
            return ToolResult.failed(f"I don't have a media player called {profile.music.player} set up.")
        player_label = profile.label("apps", player_key)
        exe = find_executable(profile.apps[player_key].candidates, which, exists)
        if exe is None:
            return ToolResult.failed(
                f"I couldn't find {player_label} on this computer. Add its location to config/profile.yaml."
            )

        wanted = _words(args.name)
        if not wanted:
            target, title = root, "your music folder"
        else:
            scored = []
            for path in _candidates(root):
                score = _score(wanted, root, path)
                if score:
                    scored.append((score, path.is_file(), path))
            if not scored:
                return ToolResult.failed(f"I couldn't find anything matching {args.name} in your music folder.")
            enough = math.ceil(len(wanted) / 2)
            best = max(score for score, _, _ in scored)
            if best < enough:
                return ToolResult.failed(f"I couldn't find anything matching {args.name} in your music folder.")
            top = sorted((p for score, _, p in scored if score == best), key=lambda p: (p.is_file(), str(p)))
            # An album folder wins over a single file that merely shares its words.
            folders = [p for p in top if p.is_dir()]
            pool = folders or top
            # "Kendrick Lamar" matches the artist folder and every album inside it: play the artist.
            parents = [p for p in pool if all(p == q or p in q.parents for q in pool)]
            if len(pool) > 1 and parents:
                pool = parents[:1]
            if len(pool) > 1:
                names = "; ".join(_title(root, p) for p in pool[:3])
                return ToolResult.failed(
                    f"That matches {len(pool)} things: {names}. Please say which one, in more detail."
                )
            target = pool[0]
            title = _title(root, target)

        try:
            launcher.start([exe, str(target)])
        except OSError as exc:
            return ToolResult.failed(f"I couldn't start {player_label}: {exc.strerror or exc}.")
        return ToolResult.launched(f"Playing {title} in {player_label}.")

    return run
