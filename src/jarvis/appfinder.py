"""`python -m jarvis add-app "FL Studio"`: find a program on this PC and add it to your profile.

Only well-known install locations and the Windows list of registered apps are searched,
nothing is ever run, and nothing is written until you pick a match. The profile file is
backed up first (profile.yaml.bak).
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

from jarvis.profile import load_profile, normalize

logger = logging.getLogger(__name__)

MAX_DEPTH = 3
MAX_VISITED = 30000
MAX_RESULTS = 8
_SKIP_WORDS = ("unins", "uninstall", "update", "updater", "crash", "setup", "install", "helper",
               "service", "redist", "vcredist", "elevate", "report", "repair")

RegistryReader = Callable[[], list[Path]]


def default_roots(env: dict[str, str] | None = None) -> list[Path]:
    """The folders programs normally install into."""
    env = dict(os.environ) if env is None else env
    roots: list[Path] = []
    for var, sub in (
        ("ProgramFiles", ""), ("ProgramFiles(x86)", ""), ("ProgramW6432", ""),
        ("LOCALAPPDATA", "Programs"), ("LOCALAPPDATA", ""), ("APPDATA", ""),
    ):
        base = env.get(var)
        if base:
            path = Path(base) / sub if sub else Path(base)
            if path not in roots:
                roots.append(path)
    return roots


def registered_programs() -> list[Path]:
    """Programs Windows knows about (the 'App Paths' registry list). Empty off Windows."""
    if os.name != "nt":
        return []
    try:
        import winreg
    except ImportError:  # pragma: no cover
        return []
    found: list[Path] = []
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(hive, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths") as root:
                index = 0
                while True:
                    try:
                        name = winreg.EnumKey(root, index)
                    except OSError:
                        break
                    index += 1
                    try:
                        with winreg.OpenKey(root, name) as sub:
                            value, _ = winreg.QueryValueEx(sub, "")
                    except OSError:
                        continue
                    if isinstance(value, str):
                        found.append(Path(value.strip('"')))
        except OSError:
            continue
    return found


def _is_wanted_exe(path: Path) -> bool:
    return path.suffix.lower() == ".exe" and not any(w in path.stem.lower() for w in _SKIP_WORDS)


def _score(query: str, path: Path) -> float:
    """0 to 1: how well the program name, or its folder name, matches what you asked for."""
    wanted = normalize(query)
    if not wanted:
        return 0.0
    best = 0.0
    for text in (path.stem, path.parent.name, path.parent.parent.name):
        name = normalize(text)
        if not name:
            continue
        if wanted == name:
            score = 1.0
        elif wanted in name or (len(name) >= 3 and name in wanted):
            score = 0.9
        else:
            score = difflib.SequenceMatcher(None, wanted, name).ratio()
        # The exe's own name counts a little more than its folders.
        weight = 1.0 if text == path.stem else 0.95
        best = max(best, score * weight)
    return best


def find_programs(
    query: str,
    roots: Iterable[Path] | None = None,
    registry: RegistryReader = registered_programs,
    threshold: float = 0.7,
) -> list[Path]:
    """Matching .exe files, best first (at most MAX_RESULTS)."""
    scored: dict[Path, float] = {}

    def consider(path: Path) -> None:
        if _is_wanted_exe(path) and path not in scored:
            score = _score(query, path)
            if score >= threshold:
                scored[path] = score

    try:
        for path in registry():
            if path.is_file():
                consider(path)
    except Exception:  # a broken registry entry must never stop the search
        logger.exception("Could not read the registered programs")

    visited = 0
    for root in (default_roots() if roots is None else roots):
        if not root.is_dir():
            continue
        base_depth = len(root.parts)
        for current, dirs, files in os.walk(root):
            depth = len(Path(current).parts) - base_depth
            dirs[:] = [d for d in dirs if not d.startswith(".")] if depth < MAX_DEPTH else []
            for name in files:
                consider(Path(current) / name)
            visited += len(dirs) + len(files)
            if visited > MAX_VISITED:
                logger.warning("App search stopped early: too many files.")
                break
        if visited > MAX_VISITED:
            break

    ranked = sorted(scored, key=lambda p: (-scored[p], len(str(p)), str(p)))
    return ranked[:MAX_RESULTS]


# --------------------------------------------------------------------------- writing the profile


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def entry_key(label: str) -> str:
    return normalize(label) or "app"


def entry_text(key: str, label: str, names: list[str], exe: str) -> str:
    lines = [
        f"  {key}:",
        f"    label: {_quote(label)}",
    ]
    if names:
        lines.append("    names: [" + ", ".join(_quote(n) for n in names) + "]")
    lines.append("    candidates:")
    lines.append(f"      - {_quote(exe)}")
    return "\n".join(lines) + "\n"


def add_to_profile(path: Path, label: str, names: list[str], exe: str) -> str:
    """Add the app under `apps:` keeping the file's comments. Returns the key used.

    Raises ValueError if an app with that key already exists, OSError if the file can't be written.
    """
    key = entry_key(label)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if path.exists():
        existing = load_profile(path)
        if key in existing.apps:
            raise ValueError(f"There is already an app called '{key}' in {path.name}.")
        shutil.copyfile(path, path.with_name(path.name + ".bak"))

    block = entry_text(key, label, names, exe)
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if re.match(r"^apps:\s*(#.*)?$", line):
            lines.insert(i + 1, block)
            new_text = "".join(lines)
            break
    else:
        if re.search(r"^apps:", text, re.MULTILINE):
            raise ValueError("Your profile writes `apps:` in a style I can't edit safely. Add it by hand.")
        new_text = text + ("" if text.endswith("\n") or not text else "\n") + "\napps:\n" + block
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    tmp.replace(path)
    return key


def run_add_app(
    query_words: list[str],
    profile_path: Path,
    out: Callable[[str], None] = print,
    ask: Callable[[str], str] = input,
    find: Callable[[str], list[Path]] = find_programs,
) -> int:
    """Interactive flow. Returns a process exit code."""
    query = " ".join(query_words).strip()
    if not query:
        out('Usage: python -m jarvis add-app "FL Studio"')
        return 2

    matches = find(query)
    if not matches:
        out(f'I could not find "{query}" in the usual install folders.')
        out("You can add it by hand: copy an entry in config/profile.yaml and put the full path to its .exe")
        out("under `candidates`. (Microsoft Store apps often have no normal .exe path.)")
        return 1

    out(f'Found {len(matches)} match{"es" if len(matches) != 1 else ""} for "{query}":')
    for number, match in enumerate(matches, start=1):
        out(f"  {number}. {match}")
    choice = ask("Which one? (number, or press Enter to cancel): ").strip()
    if not choice:
        out("Cancelled. Nothing was changed.")
        return 1
    if not choice.isdigit() or not 1 <= int(choice) <= len(matches):
        out("That isn't one of the numbers. Nothing was changed.")
        return 2
    exe = matches[int(choice) - 1]

    suggested = query.title() if query == query.lower() else query
    label = ask(f"What should I call it? (Enter for '{suggested}'): ").strip() or suggested
    extra = ask("Other names you might say, separated by commas (or Enter for none): ")
    names = [n.strip() for n in extra.split(",") if n.strip()]

    try:
        key = add_to_profile(profile_path, label, names, str(exe))
    except (ValueError, OSError) as exc:
        out(f"Could not add it: {exc}")
        return 1
    out(f"Added '{label}' ({key}) to {profile_path.name}. A backup is in {profile_path.name}.bak.")
    out("Restart JARVIS, then say: open " + label)
    return 0
