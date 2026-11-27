"""Browse your drive by name and play videos, without listing every folder in the profile.

You name the drive(s) once under `drives:` in config/profile.yaml (for example `G:\\`). After that:

- "open the main folder" / "open the G drive" opens the drive itself.
- "open the movies folder" finds a folder of that name anywhere inside the drive.
- "play the movie Dune" (or "open the movie folder and play Dune") finds the video and plays it in VLC.

Safety: only folders and video files *inside the listed drives* are ever touched, everything is
started from an argument list (no shell), nothing is deleted, moved or changed, and the name
you speak is only ever used to look things up in a listing of the drive.
"""

from __future__ import annotations

import difflib
import logging
import math
import os
import re
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from jarvis.profile import Profile, expand_path
from jarvis.tools.apps import find_executable
from jarvis.tools.launcher import Launcher
from jarvis.tools.registry import ToolArgs, ToolResult, ToolStatus

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".m4v", ".webm", ".flv", ".mpg", ".mpeg", ".ts"}
MAX_DEPTH = 6
MAX_ENTRIES = 150000
INDEX_SECONDS = 600
_SKIP_DIRS = {"$recycle.bin", "system volume information", "node_modules", "__pycache__", "venv", ".venv", "windows"}
_WORD = re.compile(r"[a-z0-9]+")
_SPOKEN_FILLER = {"the", "a", "an", "my", "folder", "directory", "please", "in", "on",
                  "with", "using", "vlc", "play", "watch", "of", "some", "for"}
_ROOT_WORDS = {"main", "root", "drive", "g", "all", "everything", "disk", "disc"}


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _tokens(text: str, drop: set[str] = frozenset()) -> list[str]:  # type: ignore[assignment]
    return [w for w in dict.fromkeys(_stem(t) for t in _WORD.findall(text.lower())) if w not in drop]


@dataclass(frozen=True)
class Entry:
    path: Path
    is_file: bool
    own: frozenset[str]        # words in the file/folder's own name
    path_words: frozenset[str]  # words in its whole path below the drive


_lock = threading.Lock()
_cache: dict[str, tuple[float, list[Entry]]] = {}


def _scan(root: Path) -> list[Entry]:
    found: list[Entry] = []
    seen = 0
    for current, dirs, files in os.walk(root):
        here = Path(current)
        depth = len(here.relative_to(root).parts)
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "$")) and d.lower() not in _SKIP_DIRS)
        if depth >= MAX_DEPTH:
            dirs[:] = []
        for name in dirs:
            path = here / name
            found.append(Entry(path, False, frozenset(_tokens(name)), frozenset(_tokens(" ".join(path.relative_to(root).parts)))))
        for name in files:
            if Path(name).suffix.lower() in VIDEO_EXTENSIONS:
                path = here / name
                parts = path.relative_to(root).with_suffix("").parts
                found.append(Entry(path, True, frozenset(_tokens(Path(name).stem)), frozenset(_tokens(" ".join(parts)))))
        seen += len(dirs) + len(files)
        if seen > MAX_ENTRIES:
            logger.warning("Drive %s is very large; only part of it was searched.", root)
            break
    return found


def get_index(root: Path, now: Callable[[], float] = time.monotonic) -> list[Entry]:
    key = str(root)
    with _lock:
        cached = _cache.get(key)
        if cached and now() - cached[0] < INDEX_SECONDS:
            return cached[1]
    entries = _scan(root)
    with _lock:
        _cache[key] = (now(), entries)
    return entries


def forget_index() -> None:
    with _lock:
        _cache.clear()


def roots_of(profile: Profile) -> list[Path]:
    return [expand_path(d) for d in profile.drives if d.strip()]


def prewarm(profile: Profile) -> None:
    """Read the drive listing in the background so the first request is quick."""
    roots = roots_of(profile)
    if not roots:
        return

    def work() -> None:
        for root in roots:
            try:
                if root.is_dir():
                    logger.info("Drive indexed: %d items in %s", len(get_index(root)), root)
            except OSError:
                logger.info("Could not index %s", root, exc_info=True)

    threading.Thread(target=work, name="jarvis-drive-index", daemon=True).start()


def hint_names(profile: Profile, limit: int = 25) -> list[str]:
    """Top-level folder names of the drive, so the speech recogniser expects them."""
    names: list[str] = []
    for root in roots_of(profile):
        try:
            if not root.is_dir():
                continue
            for entry in get_index(root):
                shallow = not entry.is_file and len(entry.path.relative_to(root).parts) <= 2 and 2 < len(entry.path.name) <= 30
                if shallow and entry.path.name.lower() not in {n.lower() for n in names}:
                    names.append(entry.path.name)
                if len(names) >= limit:
                    return names
        except OSError:
            continue
    return names


_JUNK = re.compile(r"^(?:(?:19|20)\d{2}|\d{3,4}p|4k|x26[45]|h26[45]|hevc|bluray|brrip|webrip|web|dl|hdrip|dvdrip|aac|ac3|yify|yts|mx|mp4|mkv)$")


def _title_words(tokens: frozenset[str]) -> frozenset[str]:
    """A title's words without the year, resolution and release tags ("The Run (2026) 1080p" -> {run})."""
    return frozenset(t for t in tokens if t not in _SPOKEN_FILLER and not _JUNK.match(t))


class State:
    """Where you last were, so "play X" after "open the movies folder" looks there first."""

    def __init__(self) -> None:
        self.folder: Path | None = None


def _inside(path: Path, roots: list[Path]) -> Path | None:
    """The root that contains `path`, or None. Resolves .. and links so nothing can escape."""
    try:
        real = path.resolve()
    except OSError:
        return None
    for root in roots:
        try:
            real.relative_to(root.resolve())
            return root
        except (ValueError, OSError):
            continue
    return None


def _close_words(word: str, vocab: set[str]) -> frozenset[str]:
    """Words in your library that `word` could be a mishearing or typo of ("godfather" -> "godfather")."""
    if word in vocab or len(word) < 4:
        return frozenset({word})
    near = [v for v in vocab if abs(len(v) - len(word)) <= 2 and v[0] == word[0]]
    close = difflib.get_close_matches(word, near, n=3, cutoff=0.75)
    prefixed = [v for v in near if v.startswith(word)][:3]
    return frozenset({word, *close, *prefixed})


def _pick(wanted: list[str], entries: list[Entry], files: bool, by: str) -> list[Entry]:
    """Best matches: whole words, at least 75% of them, closest title first.

    If nothing matches, words that do not exist in your library are retried as their nearest
    spellings, so "the godfather part too" or "interstellar" misheard as "interstaler" still finds it.
    """
    found = _pick_words([frozenset({w}) for w in wanted], wanted, entries, files, by)
    if found:
        return found
    kind = [e for e in entries if e.is_file == files]
    vocab: set[str] = set()
    for e in kind:
        vocab |= e.own if by == "own" else e.path_words
    alts = [_close_words(w, vocab) for w in wanted]
    if all(len(a) == 1 for a in alts):
        return []
    return _pick_words(alts, wanted, entries, files, by)


def _pick_words(alts: list[frozenset[str]], wanted: list[str], entries: list[Entry], files: bool, by: str) -> list[Entry]:
    scored = []
    spoken = set().union(*alts)
    for entry in entries:
        if entry.is_file != files:
            continue
        words = entry.own if by == "own" else entry.path_words
        score = sum(1 for options in alts if options & words)
        if score:
            scored.append((score, len(words - spoken), len(entry.path.parts), str(entry.path), entry))
    if not scored:
        return []
    need = 1 if len(wanted) == 1 else math.ceil(len(wanted) * 0.75)
    best = max(s[0] for s in scored)
    if best < need:
        return []
    top = [s for s in scored if s[0] == best]
    fewest = min(s[1] for s in top)
    top = [s for s in top if s[1] == fewest]
    shallow = min(s[2] for s in top)
    return [s[4] for s in sorted((s for s in top if s[2] == shallow), key=lambda s: s[3])]


def _ask(options: list[Path], root: Path, tool_extra: dict, what: str) -> ToolResult:
    shown = options[:3]
    labels = [str(p.relative_to(root)) if p.is_relative_to(root) else str(p) for p in shown]
    spoken = "; ".join(f"{i}. {Path(l).stem if p.is_file() else Path(l).name}" for i, (l, p) in enumerate(zip(labels, shown), 1))
    return ToolResult(
        ToolStatus.FAILED,
        f"That matches {len(options)} {what}: {spoken}. Which one? Say the number or part of the name.",
        tuple((Path(l).name, {"name": str(p), **tool_extra}) for l, p in zip(labels, shown)),
    )


# ----------------------------------------------------------------------------- open a folder


class OpenDriveFolderArgs(ToolArgs):
    name: str = Field(min_length=1)


def make_open_drive_folder(profile: Profile, launcher: Launcher, state: State) -> Callable[[OpenDriveFolderArgs], ToolResult]:
    def run(args: OpenDriveFolderArgs) -> ToolResult:
        roots = roots_of(profile)
        if not roots:
            return ToolResult.failed("No drive is set up. Add one under drives in config/profile.yaml.")
        usable = [r for r in roots if r.is_dir()]
        if not usable:
            return ToolResult.failed(f"I can't find {roots[0]}. Is the drive connected?")

        spoken = args.name.strip()
        target: Path | None = None
        # An exact path (from a "which one?" answer) is accepted only if it is inside a listed drive.
        if re.search(r"[\\/]", spoken):
            candidate = Path(spoken)
            if candidate.is_dir() and _inside(candidate, usable):
                target = candidate
        if target is None:
            words = _tokens(spoken, _SPOKEN_FILLER)
            letters = {r.drive.rstrip(":").lower() for r in usable if r.drive}
            if not words or all(w in _ROOT_WORDS or w in letters for w in words):
                target = usable[0]
            else:
                matches: list[Entry] = []
                for root in usable:
                    matches.extend(_pick(words, get_index(root), files=False, by="own"))
                if not matches:
                    return ToolResult.failed(f"I couldn't find a folder called {spoken} on your drive.")
                if len(matches) > 1:
                    return _ask([m.path for m in matches], usable[0], {}, "folders")
                target = matches[0].path
        try:
            launcher.open_path(target)
        except OSError as exc:
            return ToolResult.failed(f"I couldn't open that folder: {exc.strerror or exc}.")
        state.folder = target
        name = target.name or str(target)
        return ToolResult.launched(f"Opening {name}.")

    return run


# ----------------------------------------------------------------------------- play a video


class PlayVideoArgs(ToolArgs):
    name: str = ""
    folder: str = ""   # optional: look only inside the folder with this name
    exact: bool = False  # only accept a video whose whole title is what was said (used before streaming)


def make_play_video(
    profile: Profile,
    launcher: Launcher,
    state: State,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.isfile,
) -> Callable[[PlayVideoArgs], ToolResult]:
    def run(args: PlayVideoArgs) -> ToolResult:
        roots = [r for r in roots_of(profile) if r.is_dir()]
        if not roots:
            return ToolResult.failed("No drive is set up, or it isn't connected. Check drives in config/profile.yaml.")

        key = profile.resolve("apps", profile.music.player) or profile.resolve("apps", "vlc")
        if key is None:
            return ToolResult.failed("I don't have VLC set up. Add it under apps in config/profile.yaml.")
        label = profile.label("apps", key)
        exe = find_executable(profile.apps[key].candidates, which, exists)
        if exe is None:
            return ToolResult.failed(f"I couldn't find {label} on this computer.")

        # Where to look: an exact path from an earlier answer, a named folder, the folder you last opened, or everywhere.
        spoken = args.name.strip()
        exact: Path | None = None
        if re.search(r"[\\/]", spoken):
            candidate = Path(spoken)
            if candidate.exists() and _inside(candidate, roots):
                exact = candidate
        base: list[tuple[Path, Path]] = [(r, r) for r in roots]  # (root, search base)
        if args.folder.strip():
            found = []
            words = _tokens(args.folder, _SPOKEN_FILLER)
            for root in roots:
                found.extend((root, m.path) for m in _pick(words, get_index(root), files=False, by="own"))
            if not found:
                return ToolResult.failed(f"I couldn't find a folder called {args.folder} on your drive.")
            base = found[:1]
        elif state.folder is not None and state.folder.is_dir() and exact is None:
            owner = _inside(state.folder, roots)
            if owner is not None:
                base = [(owner, state.folder)]

        def start(target: Path, title: str) -> ToolResult:
            try:
                launcher.start([exe, str(target)])
            except OSError as exc:
                return ToolResult.failed(f"I couldn't start {label}: {exc.strerror or exc}.")
            state.folder = target if target.is_dir() else target.parent
            return ToolResult.launched(f"Playing {title} in {label}.")

        if exact is not None:
            return start(exact, exact.stem if exact.is_file() else exact.name)

        wanted = _tokens(spoken, _SPOKEN_FILLER)
        if not wanted:
            if args.folder.strip() or state.folder is not None:
                return start(base[0][1], base[0][1].name)
            return ToolResult.failed("Which movie? Tell me the name, or open the folder first.")

        results: list[Entry] = []
        same = set(wanted)
        for root, where in base:
            inside = [e for e in get_index(root) if where == root or where in e.path.parents]
            if args.exact:
                # Whole-title matches only, decided before any "closest match" narrowing.
                results.extend(
                    e for e in inside
                    if e.is_file and (
                        _title_words(e.own) == same or _title_words(frozenset(_tokens(e.path.parent.name))) == same
                    )
                )
            else:
                results.extend(_pick(wanted, inside, files=True, by="path"))
        if not results:
            where = "that folder" if args.folder.strip() or (state.folder and base[0][1] == state.folder) else "your drive"
            return ToolResult.failed(f"I couldn't find a video matching {spoken} in {where}.")
        if len(results) == 1:
            return start(results[0].path, results[0].path.stem)
        # Several episodes of one show sit in one folder named after it: play that folder.
        parents = {e.path.parent for e in results}
        if len(parents) == 1:
            folder = next(iter(parents))
            named = set(_tokens(folder.name))
            if all(w in named or len(_close_words(w, named)) > 1 for w in wanted):
                return start(folder, folder.name)
        return _ask([e.path for e in results], base[0][0], {}, "videos")

    return run
