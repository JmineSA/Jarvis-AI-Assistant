"""Builds the registry of tools JARVIS has, based on your profile."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from jarvis.profile import Kind, Profile
from jarvis.tools.apps import (
    OpenAppArgs,
    OpenFolderArgs,
    OpenWebsiteArgs,
    make_open_app,
    make_open_folder,
    make_open_website,
)
from jarvis.tools.files import ListFilesArgs, make_list_files
from jarvis.tools.launcher import Launcher, SystemLauncher
from jarvis.tools.registry import Tool, ToolRegistry
from jarvis.tools.system import NoArgs, make_current_time, make_system_status


def _name_property(profile: Profile, kind: Kind, what: str) -> dict[str, Any]:
    keys = list(getattr(profile, kind))
    spoken = ", ".join(profile.labels(kind))
    return {"type": "string", "enum": keys, "description": f"{what}. Options: {spoken}."}


def build_registry(
    profile: Profile,
    launcher: Launcher | None = None,
    clock: Callable[[], datetime] = datetime.now,
) -> ToolRegistry:
    launcher = launcher or SystemLauncher()
    registry = ToolRegistry()

    if profile.apps:
        folder_prop = (
            {"folder": _name_property(profile, "folders", "Optional project folder to open it in")}
            if profile.folders
            else {}
        )
        registry.register(
            Tool(
                name="open_app",
                description="Open an application on the computer, optionally inside a project folder.",
                parameters={
                    "type": "object",
                    "properties": {"name": _name_property(profile, "apps", "The application"), **folder_prop},
                    "required": ["name"],
                },
                args_model=OpenAppArgs,
                run=make_open_app(profile, launcher),
            )
        )

    if profile.folders:
        registry.register(
            Tool(
                name="open_folder",
                description="Open a folder in Windows File Explorer.",
                parameters={
                    "type": "object",
                    "properties": {"name": _name_property(profile, "folders", "The folder")},
                    "required": ["name"],
                },
                args_model=OpenFolderArgs,
                run=make_open_folder(profile, launcher),
            )
        )
        registry.register(
            Tool(
                name="list_files",
                description="List the files and folders inside one of the user's folders.",
                parameters={
                    "type": "object",
                    "properties": {
                        "folder": _name_property(profile, "folders", "The folder to look inside"),
                        "subfolder": {"type": "string", "description": "Optional folder inside it"},
                    },
                    "required": ["folder"],
                },
                args_model=ListFilesArgs,
                run=make_list_files(profile),
            )
        )

    if profile.websites:
        registry.register(
            Tool(
                name="open_website",
                description="Open one of the user's saved websites in the browser.",
                parameters={
                    "type": "object",
                    "properties": {"name": _name_property(profile, "websites", "The website")},
                    "required": ["name"],
                },
                args_model=OpenWebsiteArgs,
                run=make_open_website(profile, launcher),
            )
        )

    registry.register(
        Tool(
            name="system_status",
            description="Check the current CPU and memory usage of the computer.",
            parameters={"type": "object", "properties": {}},
            args_model=NoArgs,
            run=make_system_status(),
        )
    )
    registry.register(
        Tool(
            name="current_time",
            description="Get the current time and date.",
            parameters={"type": "object", "properties": {}},
            args_model=NoArgs,
            run=make_current_time(clock),
        )
    )
    return registry
