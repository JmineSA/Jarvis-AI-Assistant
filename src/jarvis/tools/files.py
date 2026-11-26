"""Read-only file listing, limited to folders in your profile."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import Field

from jarvis.profile import Profile, expand_path
from jarvis.tools.registry import ToolArgs, ToolResult

MAX_SPOKEN = 8


class ListFilesArgs(ToolArgs):
    folder: str = Field(min_length=1)
    subfolder: str | None = None


def make_list_files(profile: Profile) -> Callable[[ListFilesArgs], ToolResult]:
    def run(args: ListFilesArgs) -> ToolResult:
        key = profile.resolve("folders", args.folder)
        if key is None:
            options = ", ".join(profile.labels("folders"))
            return ToolResult.failed(f"I can only look inside these folders: {options}.")
        label = profile.label("folders", key)
        root = expand_path(profile.folders[key].path)
        if not root.is_dir():
            return ToolResult.failed(f"The {label} folder doesn't exist at the saved location.")

        target = root
        if args.subfolder and args.subfolder.strip():
            target = (root / args.subfolder.strip()).resolve()
            if not target.is_relative_to(root.resolve()):
                return ToolResult.failed("That is outside the allowed folder.")
            if not target.is_dir():
                return ToolResult.failed(f"I couldn't find {args.subfolder} inside {label}.")

        try:
            entries = [p for p in target.iterdir() if not p.name.startswith(".")]
        except OSError:
            return ToolResult.failed(f"I'm not allowed to read that part of {label}.")

        entries.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
        where = label if target == root else f"{label}, in {args.subfolder.strip()}"
        if not entries:
            return ToolResult.done(f"{where} is empty.")

        names = [p.name for p in entries[:MAX_SPOKEN]]
        more = len(entries) - len(names)
        noun = "item" if len(entries) == 1 else "items"
        message = f"{where} has {len(entries)} {noun}: {', '.join(names)}"
        message += f", and {more} more." if more > 0 else "."
        return ToolResult.done(message)

    return run
