"""Change one setting in a YAML file without disturbing its comments or layout.

Used by the Settings page. Only `section: key: value` (two levels) is supported, which is all
the settings and profile files need. The file is backed up (name.bak) before every change, the
result is re-read to check it says what was asked, and a bad result is rolled back.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import yaml


def format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\r", " ").replace("\n", " ")
    return "'" + text.replace("'", "''") + "'"


def patch_text(text: str, section: str, key: str, value: Any) -> str:
    """The text with `section.key` set to `value` (added if it is missing)."""
    lines = text.splitlines(keepends=True)
    rendered = format_scalar(value)
    head = re.compile(rf"^{re.escape(section)}:\s*(#.*)?$")
    for i, line in enumerate(lines):
        if not head.match(line.rstrip("\r\n")):
            continue
        # The section's block runs until the next line that starts in column 0 (not a comment).
        end = len(lines)
        for j in range(i + 1, len(lines)):
            stripped = lines[j].rstrip("\r\n")
            if stripped and not stripped.startswith((" ", "\t", "#")):
                end = j
                break
        pattern = re.compile(rf"^( {{2}}{re.escape(key)}:[ \t]*)([^#\r\n]*?)([ \t]*#.*)?(\r?\n?)$")
        for j in range(i + 1, end):
            match = pattern.match(lines[j])
            if match:
                comment = match.group(3) or ""
                lines[j] = f"{match.group(1).rstrip() } {rendered}{'  ' + comment.strip() if comment else ''}{match.group(4) or chr(10)}"
                return "".join(lines)
        lines.insert(i + 1, f"  {key}: {rendered}\n")
        return "".join(lines)
    if re.search(rf"^{re.escape(section)}:", text, re.MULTILINE):
        raise ValueError(f"`{section}:` is written in a style I can't edit safely. Change it by hand.")
    sep = "" if text.endswith("\n") or not text else "\n"
    return f"{text}{sep}\n{section}:\n  {key}: {rendered}\n"


def set_value(path: Path, section: str, key: str, value: Any) -> None:
    """Write `section.key: value` into the file. Raises ValueError/OSError and leaves the file as it was."""
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    new_text = patch_text(original, section, key, value)
    try:
        data = yaml.safe_load(new_text) or {}
        written = data.get(section, {}).get(key) if isinstance(data, dict) else None
    except yaml.YAMLError as exc:
        raise ValueError(f"That value would break {path.name}: {exc}") from exc
    if written != value and not (written is None and value is None):
        raise ValueError(f"I couldn't safely save that into {path.name}.")
    if path.exists():
        shutil.copyfile(path, path.with_name(path.name + ".bak"))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    tmp.replace(path)
