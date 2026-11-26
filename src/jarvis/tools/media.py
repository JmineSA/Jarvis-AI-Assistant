"""Play music: from your music folder in VLC, or on YouTube Music / YouTube.

- Local: only the music folder from config/profile.yaml is searched, and the player is
  started from an argument list (no shell), so spoken words can never become a command.
- Online: JARVIS looks up the top result on YouTube's public search page and opens that
  video in your browser. It never reads anything else and never signs in to anything.
- "auto" (the default) uses YouTube Music when you are online and your own music folder when you are
  offline (`music.online` in profile.yaml can switch this to youtube or local).
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import socket
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote_plus

import httpx
from pydantic import field_validator

from jarvis.profile import Profile, expand_path
from jarvis.tools.apps import find_executable
from jarvis.tools.launcher import Launcher
from jarvis.tools.registry import ToolArgs, ToolResult, ToolStatus
from jarvis.tools.web import SEARCH_SITES, open_in_browser

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
MAX_DEPTH = 6          # Music/Artist/Year/Album/Disc 1/track: deep libraries work too
MAX_ENTRIES = 60000    # never walk forever on a huge drive
INDEX_SECONDS = 600    # how long the folder listing is reused before it is read again
_FILLER = {"the", "a", "an", "my", "album", "song", "track", "music", "by", "from", "play", "please",
           "in", "on", "vlc", "folder", "some", "of", "for"}
_WORD = re.compile(r"[a-z0-9]+")

Source = Literal["auto", "local", "youtube_music", "youtube"]
_SOURCE_SYNONYMS = {
    "vlc": "local", "folder": "local", "music folder": "local", "my music": "local", "offline": "local",
    "yt music": "youtube_music", "youtube music": "youtube_music", "youtubemusic": "youtube_music",
    "youtube_music": "youtube_music", "you tube": "youtube", "": "auto", "any": "auto",
}
_VIDEO = re.compile(r'"videoRenderer":\{"videoId":"([A-Za-z0-9_-]{11})"')
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def _words(text: str) -> list[str]:
    return [w for w in dict.fromkeys(_WORD.findall(text.lower())) if w not in _FILLER]


class PlayMediaArgs(ToolArgs):
    name: str = ""
    source: Source = "auto"

    @field_validator("source", mode="before")
    @classmethod
    def _friendly_source(cls, value: Any) -> Any:
        if isinstance(value, str):
            key = " ".join(value.lower().replace("-", " ").split())
            return _SOURCE_SYNONYMS.get(key, key)
        return value


# ----------------------------------------------------------------------------- the music folder


@dataclass(frozen=True)
class Entry:
    path: Path
    is_file: bool
    tokens: frozenset[str]


_index_lock = threading.Lock()
_index_cache: dict[str, tuple[float, list[Entry]]] = {}


def _scan(root: Path) -> list[Entry]:
    """Every folder and audio file up to MAX_DEPTH below the music folder."""
    found: list[Entry] = []
    seen = 0
    for current, dirs, files in os.walk(root):
        here = Path(current)
        depth = len(here.relative_to(root).parts)
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        if depth >= MAX_DEPTH:
            dirs[:] = []
        for name in dirs:
            path = here / name
            found.append(Entry(path, False, frozenset(_words(" ".join(path.relative_to(root).parts)))))
        for name in files:
            if Path(name).suffix.lower() in AUDIO_EXTENSIONS:
                path = here / name
                parts = path.relative_to(root).with_suffix("").parts
                found.append(Entry(path, True, frozenset(_words(" ".join(parts)))))
        seen += len(dirs) + len(files)
        if seen > MAX_ENTRIES:
            logger.warning("Music folder is very large; only part of it was searched.")
            break
    return found


def get_index(root: Path, now: Callable[[], float] = time.monotonic) -> list[Entry]:
    """The listing of your music folder, read once and reused for a few minutes."""
    key = str(root)
    with _index_lock:
        cached = _index_cache.get(key)
        if cached and now() - cached[0] < INDEX_SECONDS:
            return cached[1]
    entries = _scan(root)
    with _index_lock:
        _index_cache[key] = (now(), entries)
    return entries


def forget_index() -> None:
    with _index_lock:
        _index_cache.clear()


def prewarm(profile: Profile) -> None:
    """Read the music folder in the background so the first "play ..." is quick."""
    if not profile.music.folder:
        return
    root = expand_path(profile.music.folder)

    def work() -> None:
        try:
            if root.is_dir():
                entries = get_index(root)
                logger.info("Music folder indexed: %d items in %s", len(entries), root)
        except OSError:
            logger.info("Could not index the music folder", exc_info=True)

    threading.Thread(target=work, name="jarvis-music-index", daemon=True).start()


_NOISE = re.compile(r"^\s*(?:\d{1,3}\s*[-._)]\s*|\(?\d{4}\)?\s*[-._]\s*)|\s*[\[(]\s*(?:\d{4}|deluxe|remaster\w*|explicit)[^\])]*[\])]\s*$", re.IGNORECASE)


def hint_names(profile: Profile, limit: int = 30) -> list[str]:
    """Artist and album names from your music folder, so the speech recogniser expects them.

    Reads only the folder names (never the songs) and only what is already in the cached listing.
    """
    if not profile.music.folder:
        return []
    root = expand_path(profile.music.folder)
    try:
        if not root.is_dir():
            return []
        entries = get_index(root)
    except OSError:
        return []
    names: list[str] = []
    for depth in (1, 2, 3):
        for entry in entries:
            if entry.is_file or len(entry.path.relative_to(root).parts) != depth:
                continue
            name = _NOISE.sub("", entry.path.name).strip()
            if re.fullmatch(r"\d+|(?:disc|cd)\s*\d+", name, re.IGNORECASE):
                continue  # a year or disc folder, not a name anyone says
            if 2 < len(name) <= 40 and name.lower() not in {n.lower() for n in names}:
                names.append(name)
            if len(names) >= limit:
                return names
    return names


def describe_folder(root: Path) -> list[str]:
    """Plain-text report for `python -m jarvis music-check`."""
    lines = [f"Music folder: {root}"]
    if not root.exists():
        lines.append("  NOT FOUND. Is the drive connected? Check the path under music: in config/profile.yaml.")
        return lines
    if not root.is_dir():
        lines.append("  That path is not a folder.")
        return lines
    entries = _scan(root)
    files = [e for e in entries if e.is_file]
    kinds = Counter(e.path.suffix.lower() for e in files)
    deepest = max((len(e.path.relative_to(root).parts) for e in files), default=0)
    lines.append(f"  Found {len(files)} audio files in {len(entries) - len(files)} folders (searched {MAX_DEPTH} levels deep).")
    if kinds:
        lines.append("  Types: " + ", ".join(f"{ext} x{n}" for ext, n in kinds.most_common()))
        lines.append(f"  Deepest song is {deepest} levels below the music folder.")
        lines.append("  Examples:")
        lines.extend(f"    {e.path.relative_to(root)}" for e in files[:5])
    else:
        lines.append("  No audio files found. JARVIS plays " + ", ".join(sorted(AUDIO_EXTENSIONS)))
    return lines


def _score(wanted: list[str], entry: Entry) -> int:
    return sum(1 for word in wanted if word in entry.tokens)


def _title(root: Path, path: Path) -> str:
    return " - ".join(path.relative_to(root).with_suffix("").parts) if path.is_file() else path.name


# ----------------------------------------------------------------------------- YouTube


def fetch_video_id(query: str, transport: httpx.BaseTransport | None = None, timeout: float = 6.0) -> str | None:
    """The first real video on YouTube's search page for these words, or None if the page has none.

    Raises httpx.HTTPError when YouTube cannot be reached (for example when you are offline).
    """
    with httpx.Client(timeout=timeout, headers=_BROWSER_HEADERS, follow_redirects=True, transport=transport) as client:
        response = client.get("https://www.youtube.com/results", params={"search_query": query})
        response.raise_for_status()
    match = _VIDEO.search(response.text)
    return match.group(1) if match else None


def is_online(timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=timeout):
            return True
    except OSError:
        return False


# ----------------------------------------------------------------------------- the tool


def make_play_media(
    profile: Profile,
    launcher: Launcher,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.isfile,
    resolver: Callable[[str], str | None] = fetch_video_id,
    online: Callable[[], bool] = is_online,
) -> Callable[[PlayMediaArgs], ToolResult]:
    # ------------------------------------------------------------- local
    def play_local(query: str) -> tuple[ToolResult, str]:
        """Returns the result and why: played, not_found, ambiguous or setup."""
        if not profile.music.folder:
            return ToolResult.failed("No music folder is set. Add it under music in config/profile.yaml."), "setup"
        root = expand_path(profile.music.folder)
        if not root.is_dir():
            return ToolResult.failed(
                f"I can't find your music folder at {root}. Is the drive connected? "
                "Check the path under music in config/profile.yaml."
            ), "setup"

        player_key = profile.resolve("apps", profile.music.player)
        if player_key is None:
            return ToolResult.failed(f"I don't have a media player called {profile.music.player} set up."), "setup"
        player_label = profile.label("apps", player_key)
        exe = find_executable(profile.apps[player_key].candidates, which, exists)
        if exe is None:
            return ToolResult.failed(
                f"I couldn't find {player_label} on this computer. Add its location to config/profile.yaml."
            ), "setup"

        wanted = _words(query)
        argv = [exe]
        if not wanted:
            target, title = root, "your music folder"
            if player_key == "vlc":
                argv.append("--random")  # "play music" shuffles the whole library
        else:
            not_found = ToolResult.failed(f"I couldn't find anything matching {query} in your music folder.")
            entries = get_index(root)
            scored = [(score, e) for e in entries if (score := _score(wanted, e))]
            if not scored:
                return not_found, "not_found"
            need = 1 if len(wanted) == 1 else math.ceil(len(wanted) * 0.75)
            best = max(score for score, _ in scored)
            if best < need:
                return not_found, "not_found"
            top = [e for score, e in scored if score == best]
            # An album folder wins over a single file that merely shares its words.
            folders = [e for e in top if not e.is_file]
            pool = folders or top
            # "Kendrick Lamar" matches the artist folder and every album inside it: play the artist.
            parents = [e for e in pool if all(e is o or e.path in o.path.parents for o in pool)]
            if len(pool) > 1 and parents:
                pool = parents[:1]
            if len(pool) > 1:
                # Several songs share the words: the one with the fewest extra words is the closest title.
                tidy = sorted(pool, key=lambda e: (len(e.tokens - set(wanted)), str(e.path)))
                if len(tidy[0].tokens - set(wanted)) < len(tidy[1].tokens - set(wanted)):
                    pool = tidy[:1]
            if len(pool) > 1:
                shown = pool[:3]
                names = "; ".join(f"{i}. {_title(root, e.path)}" for i, e in enumerate(shown, 1))
                options = tuple((_title(root, e.path), {"name": _title(root, e.path), "source": "local"}) for e in shown)
                return ToolResult(
                    ToolStatus.FAILED,
                    f"That matches {len(pool)} things: {names}. Which one? Say the number or part of the name.",
                    options,
                ), "ambiguous"
            target = pool[0].path
            title = _title(root, target)
            argv.append(str(target))

        if not wanted:
            argv.append(str(target))
        try:
            launcher.start(argv)
        except OSError as exc:
            return ToolResult.failed(f"I couldn't start {player_label}: {exc.strerror or exc}."), "setup"
        return ToolResult.launched(f"Playing {title} in {player_label}."), "played"

    # ------------------------------------------------------------- online
    def play_web(query: str, site: str) -> ToolResult:
        name, search_url = SEARCH_SITES[site]
        base = "https://music.youtube.com" if site == "youtube_music" else "https://www.youtube.com"
        if not query:
            url, message = base, f"Opening {name}."
        else:
            try:
                video = resolver(query)
            except (httpx.HTTPError, OSError):
                return ToolResult.failed(f"I can't reach {name} right now. Are you online?")
            if video:
                url, message = f"{base}/watch?v={video}", f"Playing the top {name} result for {query}."
            else:
                url = search_url.format(q=quote_plus(query[:200]))
                message = f"I opened {name} for {query}, but couldn't pick a result. Choose one to play."
        try:
            open_in_browser(profile, launcher, url, which, exists)
        except OSError as exc:
            return ToolResult.failed(f"I couldn't open the browser: {exc.strerror or exc}.")
        return ToolResult.launched(message)

    # ------------------------------------------------------------- choose
    def run(args: PlayMediaArgs) -> ToolResult:
        query = " ".join(args.name.split())
        if args.source == "local":
            return play_local(query)[0]
        if args.source in ("youtube_music", "youtube"):
            return play_web(query, args.source)

        # auto: online means your usual streaming site (YouTube Music), offline means your own folder
        if not query:
            return play_local(query)[0]
        preferred = profile.music.online
        if online():
            if preferred != "local":
                streamed = play_web(query, preferred)
                if streamed.status is ToolStatus.LAUNCHED:
                    return streamed
            result, why = play_local(query)
            if why in ("played", "ambiguous"):
                return result
            if preferred == "local":
                return play_web(query, "youtube_music")
            return streamed
        result, why = play_local(query)
        if why in ("played", "ambiguous"):
            return result
        return ToolResult.failed(f"{result.message} I'm offline, so I can't try YouTube Music.")

    return run
