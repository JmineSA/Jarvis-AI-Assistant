"""Builds the registry of tools JARVIS has, based on your profile."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from jarvis.config import ScreenSettings
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
from jarvis.tools.drives import (
    OpenDriveFolderArgs,
    PlayVideoArgs,
    State,
    make_open_drive_folder,
    make_play_video,
)
from jarvis.tools.files import ListFilesArgs, make_list_files
from jarvis.tools.gmail import CheckEmailArgs, GmailClient, make_check_email
from jarvis.tools.launcher import Launcher, SystemLauncher
from jarvis.tools.maps import MapsArgs, make_maps
from jarvis.tools.media import PlayMediaArgs, make_play_media
from jarvis.tools.mediakeys import KeySender, MediaControlArgs, make_media_control
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
from jarvis.tools.registry import Tool, ToolRegistry
from jarvis.tools.screen import SCREEN_TIMEOUT, ScreenArgs, ScreenBackend, ScreenshotArgs, make_look_at_screen, make_take_screenshot
from jarvis.tools.startmenu import StartMenu
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
    media_keys: KeySender | None = None,
    startmenu: StartMenu | None = None,
    gmail: GmailClient | None = None,
    screen: ScreenBackend | None = None,
    screen_settings: ScreenSettings | None = None,
    screenshot_dir: Path | None = None,
) -> ToolRegistry:
    launcher = launcher or SystemLauncher()
    registry = ToolRegistry()

    folder_prop = (
        {"folder": _name_property(profile, "folders", "Optional project folder to open it in")}
        if profile.folders
        else {}
    )
    known = ", ".join(profile.labels("apps"))
    menu = startmenu or StartMenu()
    registry.register(
        Tool(
            name="open_app",
            description=(
                "Open an application on the computer by its name, optionally inside a project folder. "
                "Any installed program can be opened, not only the saved ones."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The application's name as the user said it"
                        + (f". Saved apps: {known}." if known else "."),
                    },
                    **folder_prop,
                },
                "required": ["name"],
            },
            args_model=OpenAppArgs,
            run=make_open_app(profile, launcher, startmenu=menu),
        )
    )

    if profile.folders:
        registry.register(
            Tool(
                name="open_folder",
                description="Open a folder as a window in Windows File Explorer. Only when asked to open it, not to list what is inside.",
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
                description="Read out what is inside one of the user's folders ('what is in my X folder', 'show me the files in X'). Does not open a window.",
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
            name="maps",
            description=(
                "Open Google Maps to find a place near the user's home (search) or to get directions "
                "from their home to a place (directions)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": _text_property("The place or thing, e.g. 'police station' or 'Menlyn Park'"),
                    "mode": {
                        "type": "string",
                        "enum": ["search", "directions"],
                        "description": "search = find places near home; directions = route from home to the place.",
                    },
                },
                "required": ["query"],
            },
            args_model=MapsArgs,
            run=make_maps(profile, launcher),
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
    if screen is not None:
        limit = (screen_settings or ScreenSettings()).max_chars
        registry.register(
            Tool(
                name="look_at_screen",
                description=(
                    "Read the words in the window the user is working in and explain them. Use for 'what's on my screen', "
                    "'explain this error / query / function / chart / article', 'summarise what I'm reading'. "
                    "Pass the user's own words as question."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "question": _text_property("What the user wants to know about the screen, in their words"),
                        "whole_screen": {"type": "boolean", "description": "true only if they say the whole or entire screen"},
                    },
                },
                args_model=ScreenArgs,
                run=make_look_at_screen(screen, limit),
                timeout=SCREEN_TIMEOUT,
            )
        )
        registry.register(
            Tool(
                name="take_screenshot",
                description="Save a picture of the screen to the screenshots folder.",
                parameters={
                    "type": "object",
                    "properties": {"whole_screen": {"type": "boolean", "description": "false = only the window in use"}},
                },
                args_model=ScreenshotArgs,
                run=make_take_screenshot(screen, screenshot_dir or Path("data/screenshots")),
                timeout=SCREEN_TIMEOUT,
            )
        )
    if gmail is not None:
        registry.register(
            Tool(
                name="check_email",
                description="Read-only: say how many unread emails there are and who the newest are from, or show the latest sent mail or what is in spam. Never sends or changes mail.",
                parameters={
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer", "description": "How many emails to list (1-10, default 3)"},
                        "unread_only": {"type": "boolean", "description": "Only unread mail (default true)"},
                        "sender": _text_property("Only mail from this person (to this person for sent mail)"),
                        "folder": {"type": "string", "enum": ["inbox", "sent", "spam"], "description": "inbox (default), sent for mail I sent, or spam"},
                    },
                },
                args_model=CheckEmailArgs,
                run=make_check_email(gmail),
            )
        )
    where = State()  # the folder you last opened, shared by the drive tools
    play_video = make_play_video(profile, launcher, where) if profile.drives else None
    registry.register(
        Tool(
            name="play_media",
            description=(
                "Play music. Local songs play in the media player from the user's music folder; "
                "YouTube Music and YouTube play in the browser. With no source it tries the music folder first."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": _text_property(
                        "The song, album or artist, e.g. 'Love Me by Lil Wayne'. Leave empty to play the whole music folder."
                    ),
                    "source": {
                        "type": "string",
                        "enum": ["auto", "local", "youtube_music", "youtube"],
                        "description": "Where to play it. Use auto unless the user names YouTube Music, YouTube, VLC or their music folder.",
                    },
                },
            },
            args_model=PlayMediaArgs,
            run=make_play_media(
                profile,
                launcher,
                video_fallback=(lambda q: play_video(PlayVideoArgs(name=q))) if play_video else None,
                video_exact=(lambda q: play_video(PlayVideoArgs(name=q, exact=True))) if play_video else None,
            ),
        )
    )
    if profile.drives:
        registry.register(
            Tool(
                name="open_drive_folder",
                description=(
                    "Open the main drive, or any folder on it by name (movies, games, documentaries...), "
                    "in File Explorer. Use for folders that are not in the saved folder list."
                ),
                parameters={
                    "type": "object",
                    "properties": {"name": _text_property("Folder name, or 'main' for the whole drive")},
                    "required": ["name"],
                },
                args_model=OpenDriveFolderArgs,
                run=make_open_drive_folder(profile, launcher, where),
            )
        )
        registry.register(
            Tool(
                name="play_video",
                description="Find a movie, series or video on the user's drive and play it in VLC.",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": _text_property("The movie, series or video title. Empty plays the folder last opened."),
                        "folder": _text_property("Optional: only look inside the folder with this name, e.g. movies"),
                    },
                },
                args_model=PlayVideoArgs,
                run=play_video,
            )
        )
    registry.register(
        Tool(
            name="media_control",
            description="Pause, resume, skip to the next or previous track, stop, mute or change the volume of the music.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["play_pause", "next", "previous", "stop", "volume_up", "volume_down", "mute"],
                        "description": "pause and resume are both play_pause",
                    }
                },
                "required": ["action"],
            },
            args_model=MediaControlArgs,
            run=make_media_control(media_keys),
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
