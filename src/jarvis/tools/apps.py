"""Tools that open applications, folders and websites from your profile."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from urllib.parse import urlparse

from pydantic import Field

from jarvis.profile import Profile, expand_path
from jarvis.tools.launcher import Launcher
from jarvis.tools.registry import ToolArgs, ToolResult


def find_executable(
    candidates: list[str],
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.isfile,
) -> str | None:
    """First candidate that exists: an explicit path, or a program name found on PATH."""
    for candidate in candidates:
        expanded = os.path.expandvars(os.path.expanduser(candidate))
        if os.path.isabs(expanded) or any(sep in expanded for sep in ("/", "\\")):
            if exists(expanded):
                return expanded
        else:
            found = which(expanded)
            if found:
                return found
    return None


def _unknown(profile: Profile, kind: str, wanted: str, noun: str) -> ToolResult:
    options = ", ".join(profile.labels(kind))  # type: ignore[arg-type]
    return ToolResult.failed(f"I don't have {noun} called {wanted}. I can open: {options}.")


class OpenAppArgs(ToolArgs):
    name: str = Field(min_length=1)
    folder: str | None = None


def make_open_app(
    profile: Profile,
    launcher: Launcher,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.isfile,
) -> Callable[[OpenAppArgs], ToolResult]:
    def run(args: OpenAppArgs) -> ToolResult:
        key = profile.resolve("apps", args.name)
        if key is None:
            return _unknown(profile, "apps", args.name, "an app")
        app = profile.apps[key]
        label = profile.label("apps", key)

        exe = find_executable(app.candidates, which, exists)
        if exe is None:
            return ToolResult.failed(
                f"I couldn't find {label} on this computer. "
                "Add its location to config/profile.yaml."
            )

        argv = [exe, *app.args]
        cwd = None
        where = ""
        if args.folder:
            fkey = profile.resolve("folders", args.folder)
            if fkey is None:
                return _unknown(profile, "folders", args.folder, "a folder")
            folder = expand_path(profile.folders[fkey].path)
            flabel = profile.label("folders", fkey)
            if not folder.is_dir():
                return ToolResult.failed(f"The {flabel} folder doesn't exist at the saved location.")
            if app.folder_mode == "arg":
                argv.append(str(folder))
            elif app.folder_mode == "cwd":
                cwd = folder
            else:
                return ToolResult.failed(f"{label} can't be opened inside a folder.")
            where = f" in {flabel}"

        try:
            launcher.start(argv, cwd)
        except OSError as exc:
            return ToolResult.failed(f"I couldn't start {label}: {exc.strerror or exc}.")
        return ToolResult.launched(f"Opening {label}{where}.")

    return run


class OpenFolderArgs(ToolArgs):
    name: str = Field(min_length=1)


def make_open_folder(profile: Profile, launcher: Launcher) -> Callable[[OpenFolderArgs], ToolResult]:
    def run(args: OpenFolderArgs) -> ToolResult:
        key = profile.resolve("folders", args.name)
        if key is None:
            return _unknown(profile, "folders", args.name, "a folder")
        label = profile.label("folders", key)
        path = expand_path(profile.folders[key].path)
        if not path.is_dir():
            return ToolResult.failed(f"The {label} folder doesn't exist at the saved location.")
        try:
            launcher.open_path(path)
        except OSError as exc:
            return ToolResult.failed(f"I couldn't open the {label} folder: {exc.strerror or exc}.")
        return ToolResult.launched(f"Opening the {label} folder.")

    return run


class OpenWebsiteArgs(ToolArgs):
    name: str = Field(min_length=1)


def make_open_website(
    profile: Profile,
    launcher: Launcher,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.isfile,
) -> Callable[[OpenWebsiteArgs], ToolResult]:
    def run(args: OpenWebsiteArgs) -> ToolResult:
        key = profile.resolve("websites", args.name)
        if key is None:
            return _unknown(profile, "websites", args.name, "a website")
        site = profile.websites[key]
        label = profile.label("websites", key)

        url = site.url.strip()
        if not url:
            return ToolResult.failed(
                f"No address is saved for {label}. Add it under websites in config/profile.yaml."
            )
        if urlparse(url).scheme not in ("http", "https"):
            return ToolResult.failed(f"The saved address for {label} must start with http or https.")

        try:
            exe = None
            if site.app:
                akey = profile.resolve("apps", site.app)
                if akey is not None:
                    exe = find_executable(profile.apps[akey].candidates, which, exists)
            if exe:
                launcher.start([exe, url])
            else:
                launcher.open_url(url)  # fall back to the default browser
        except OSError as exc:
            return ToolResult.failed(f"I couldn't open {label}: {exc.strerror or exc}.")
        return ToolResult.launched(f"Opening {label}.")

    return run
