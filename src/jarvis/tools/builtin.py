"""Builds the registry of tools JARVIS has, based on your profile."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from jarvis.memory.store import MemoryStore
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
from jarvis.tools.memory_tools import (
    CancelReminderArgs,
    ForgetArgs,
    NoReminderArgs,
    RecallArgs,
    RememberArgs,
    SetReminderArgs,
    make_cancel_reminder,
    make_forget,
    make_list_reminders,
    make_recall,
    make_remember,
    make_set_reminder,
)
from jarvis.tools.media import PlayMediaArgs, make_play_media
from jarvis.tools.registry import Tool, ToolRegistry
from jarvis.tools.system import NoArgs, make_current_time, make_system_status
from jarvis.tools.weather import WeatherArgs, make_weather
from jarvis.tools.web import (
    ComposeEmailArgs,
    WebSearchArgs,
    make_compose_email,
    make_web_search,
)


def _name_property(profile: Profile, kind: Kind, what: str) -> dict[str, Any]:
    keys = list(getattr(profile, kind))
    spoken = ", ".join(profile.labels(kind))
    return {"type": "string", "enum": keys, "description": f"{what}. Options: {spoken}."}


def _text_property(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _register_memory_tools(
    registry: ToolRegistry,
    store: MemoryStore,
    session_id: str | None,
    clock: Callable[[], datetime],
) -> None:
    forget_run, forget_describe = make_forget(store)
    no_params = {"type": "object", "properties": {}}
    registry.register(
        Tool(
            name="remember",
            description="Save a fact or preference the user asks you to remember.",
            parameters={
                "type": "object",
                "properties": {
                    "text": _text_property("The fact, written as a full sentence, e.g. 'The LightsOn project is due on 1 December.'")
                },
                "required": ["text"],
            },
            args_model=RememberArgs,
            run=make_remember(store),
        )
    )
    registry.register(
        Tool(
            name="recall",
            description="Look up what the user told you to remember, or what was said in earlier conversations.",
            parameters={
                "type": "object",
                "properties": {"topic": _text_property("What to look up. Leave empty for everything saved.")},
            },
            args_model=RecallArgs,
            run=make_recall(store, session_id),
        )
    )
    registry.register(
        Tool(
            name="forget",
            description="Delete a saved fact. The user is asked to confirm.",
            parameters={
                "type": "object",
                "properties": {"about": _text_property("Words identifying the saved fact to delete")},
                "required": ["about"],
            },
            args_model=ForgetArgs,
            run=forget_run,
            risk="confirm",
            describe=forget_describe,
        )
    )
    registry.register(
        Tool(
            name="set_reminder",
            description="Set a reminder that will be spoken at the given time.",
            parameters={
                "type": "object",
                "properties": {
                    "text": _text_property("What to be reminded of, e.g. 'call the bank'"),
                    "when": _text_property(
                        "When, exactly as the user said it, e.g. 'in 20 minutes', 'tomorrow at 9am', '1 December'"
                    ),
                },
                "required": ["text", "when"],
            },
            args_model=SetReminderArgs,
            run=make_set_reminder(store, clock),
        )
    )
    registry.register(
        Tool(
            name="list_reminders",
            description="List the user's pending reminders.",
            parameters=no_params,
            args_model=NoReminderArgs,
            run=make_list_reminders(store, clock),
        )
    )
    registry.register(
        Tool(
            name="cancel_reminder",
            description="Cancel a pending reminder.",
            parameters={
                "type": "object",
                "properties": {"about": _text_property("Words identifying the reminder to cancel")},
                "required": ["about"],
            },
            args_model=CancelReminderArgs,
            run=make_cancel_reminder(store),
        )
    )


def build_registry(
    profile: Profile,
    launcher: Launcher | None = None,
    clock: Callable[[], datetime] = datetime.now,
    store: MemoryStore | None = None,
    session_id: str | None = None,
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
            name="web_search",
            description="Search Google, YouTube or YouTube Music. Opens the results in the browser.",
            parameters={
                "type": "object",
                "properties": {
                    "query": _text_property("What to search for, e.g. 'best current action movies'"),
                    "site": {
                        "type": "string",
                        "enum": ["google", "youtube", "youtube_music"],
                        "description": "Where to search. Use google unless the user names YouTube or YouTube Music.",
                    },
                },
                "required": ["query"],
            },
            args_model=WebSearchArgs,
            run=make_web_search(profile, launcher),
        )
    )
    registry.register(
        Tool(
            name="compose_email",
            description="Open a Gmail draft for the user to read and send. It never sends anything.",
            parameters={
                "type": "object",
                "properties": {
                    "to": _text_property("Who it is for: a saved contact name or an email address"),
                    "subject": _text_property("The subject line"),
                    "body": _text_property("The message text"),
                },
            },
            args_model=ComposeEmailArgs,
            run=make_compose_email(profile, launcher),
        )
    )
    if profile.music.folder:
        registry.register(
            Tool(
                name="play_media",
                description="Play an album or song from the user's music folder in the media player.",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": _text_property(
                            "The album, artist or song, e.g. 'Kendrick Lamar GNX'. Leave empty to play the whole music folder."
                        )
                    },
                },
                args_model=PlayMediaArgs,
                run=make_play_media(profile, launcher),
            )
        )
    if profile.location is not None:
        registry.register(
            Tool(
                name="weather",
                description="Get the weather forecast for the user's home city, now or at a time.",
                parameters={
                    "type": "object",
                    "properties": {
                        "when": _text_property(
                            "When, exactly as the user said it: 'today', 'tomorrow', 'at 2pm', 'Friday'. Leave empty for right now."
                        )
                    },
                },
                args_model=WeatherArgs,
                run=make_weather(profile.location, clock),
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
    if store is not None:
        _register_memory_tools(registry, store, session_id, clock)
    return registry
