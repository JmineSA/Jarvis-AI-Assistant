from datetime import datetime

import pytest

from jarvis.ai.ollama_client import ToolCall
from jarvis.ai.router import CommandRouter, clean
from jarvis.config import PROJECT_ROOT
from jarvis.profile import load_profile

NOW = datetime(2026, 11, 25, 10, 0)
ALL_TOOLS = [
    "open_app", "open_folder", "open_website", "web_search", "play_media", "weather", "current_time",
    "system_status", "remember", "set_reminder", "list_reminders", "compose_email", "maps", "media_control",
]


@pytest.fixture(scope="module")
def route():
    profile = load_profile(PROJECT_ROOT / "config" / "profile.yaml")
    return CommandRouter(profile, ALL_TOOLS, clock=lambda: NOW, aliases=["Travis"])


def test_clean_strips_greeting_politeness_and_punctuation():
    assert clean("Hi Jarvis, could you please open VS Code?") == "open VS Code"
    assert clean("Hey Travis, what's the weather?", ["Travis"]) == "what's the weather"
    assert clean("  JARVIS open chrome.  ") == "open chrome"


@pytest.mark.parametrize(
    ("text", "tool", "arguments"),
    [
        ("open VS Code", "open_app", {"name": "vscode"}),
        ("Hi Jarvis, open VLC", "open_app", {"name": "vlc"}),
        ("open my music folder", "open_folder", {"name": "music"}),
        ("open gmail", "open_website", {"name": "gmail"}),
        ("open YouTube Music", "open_website", {"name": "youtubemusic"}),
        ("what time is it", "current_time", {}),
        ("what's the time?", "current_time", {}),
        ("system status", "system_status", {}),
        ("how's my computer doing", "system_status", {}),
        ("what are my reminders", "list_reminders", {}),
    ],
)
def test_simple_commands(route, text, tool, arguments):
    assert route(text) == ToolCall(tool, arguments)


@pytest.mark.parametrize(
    ("text", "when"),
    [
        ("what's the weather", None),
        ("what's the weather today", "today"),
        ("what is the weather like tomorrow", "tomorrow"),
        ("weather at 2pm", "at 2pm"),
        ("what's the weather in Pretoria tomorrow", "tomorrow"),
    ],
)
def test_weather(route, text, when):
    call = route(text)
    assert call.name == "weather"
    assert call.arguments == ({"when": when} if when else {})


def test_weather_for_another_city_goes_to_the_model(route):
    assert route("what's the weather in Cape Town") is None


@pytest.mark.parametrize(
    ("text", "name", "source"),
    [
        ("play Love Me by Lil Wayne", "Love Me by Lil Wayne", "auto"),
        ("play Love Me by Lil Wayne on YouTube Music", "Love Me by Lil Wayne", "youtube_music"),
        ("open YouTube Music and play Love Me by Lil Wayne", "Love Me by Lil Wayne", "youtube_music"),
        ("open my music folder and play Love Me by Lil Wayne in VLC", "Love Me by Lil Wayne", "local"),
        ("play Love Me by Lil Wayne locally", "Love Me by Lil Wayne", "local"),
        ("play the album Tha Carter V from my music folder", "Tha Carter V", "local"),
        ("play some Kendrick Lamar on youtube", "Kendrick Lamar", "youtube"),
        ("could you please play Discovery by Daft Punk", "Discovery by Daft Punk", "auto"),
    ],
)
def test_play_commands(route, text, name, source):
    assert route(text) == ToolCall("play_media", {"name": name, "source": source})


@pytest.mark.parametrize("text", ["play music", "play some music", "play my music", "put on some songs"])
def test_bare_play_music_means_the_local_library(route, text):
    assert route(text) == ToolCall("play_media", {"name": "", "source": "auto"})


def test_bare_music_on_youtube_music_opens_the_site(route):
    assert route("play music on YouTube Music") == ToolCall("play_media", {"name": "", "source": "youtube_music"})


@pytest.mark.parametrize("text", ["play it again", "play that", "play"])
def test_vague_play_goes_to_the_model(route, text):
    assert route(text) is None


@pytest.mark.parametrize(
    ("text", "query", "site"),
    [
        ("search Google for best current action movies", "best current action movies", "google"),
        ("search for best action movies on YouTube", "best action movies", "youtube"),
        ("google python dataclasses", "python dataclasses", "google"),
        ("search youtube music for Daft Punk", "Daft Punk", "youtube_music"),
        ("look up the capital of Peru", "the capital of Peru", "google"),
    ],
)
def test_search(route, text, query, site):
    assert route(text) == ToolCall("web_search", {"query": query, "site": site})


def test_remember_keeps_the_users_own_words(route):
    assert route("remember that my favourite editor is VS Code") == ToolCall(
        "remember", {"text": "my favourite editor is VS Code"}
    )
    assert route("remember to call mom") is None  # sounds like a reminder: the model decides


@pytest.mark.parametrize(
    ("text", "reminder", "when"),
    [
        ("remind me to stretch in 2 minutes", "stretch", "in 2 minutes"),
        ("remind me in 20 minutes to call the bank", "call the bank", "in 20 minutes"),
        ("remind me to call mom at the bank in 20 minutes", "call mom at the bank", "in 20 minutes"),
        ("remind me to send the report tomorrow at 9am", "send the report", "tomorrow at 9am"),
    ],
)
def test_reminders_need_a_time_that_really_parses(route, text, reminder, when):
    assert route(text) == ToolCall("set_reminder", {"text": reminder, "when": when})


def test_a_reminder_without_a_time_goes_to_the_model(route):
    assert route("remind me to be nicer") is None


@pytest.mark.parametrize(
    "text",
    [
        "tell me a joke",
        "what do you think about open source",
        "who is the president of France",
        "write an email to my boss",
        "forget my favourite editor",       # needs confirmation, so the model path handles it
        "",
    ],
)
def test_anything_else_goes_to_the_model(route, text):
    assert route(text) is None


def test_open_app_in_a_project_folder(route):
    call = route("open VS Code in LightsOn")
    assert call is not None and call.name == "open_app" and call.arguments["name"] == "vscode"
    assert call.arguments["folder"]


def test_only_tools_that_exist_are_routed():
    profile = load_profile(PROJECT_ROOT / "config" / "profile.yaml")
    router = CommandRouter(profile, ["current_time"])
    assert router("what time is it") is not None
    assert router("open VS Code") is None
    assert router("play music") is None


@pytest.mark.parametrize(
    ("text", "query", "mode"),
    [
        ("Hi Jarvis, can you check any police stations around my area?", "police stations", "search"),
        ("find the nearest pharmacy", "pharmacy", "search"),
        ("where is the closest ATM", "ATM", "search"),
        ("pizza near me", "pizza", "search"),
        ("show me restaurants nearby", "restaurants", "search"),
        ("directions to Menlyn Park", "Menlyn Park", "directions"),
        ("how do I get to the Union Buildings", "the Union Buildings", "directions"),
        ("take me to the airport", "the airport", "directions"),
    ],
)
def test_maps_commands(route, text, query, mode):
    assert route(text) == ToolCall("maps", {"query": query, "mode": mode})


def test_weather_near_me_is_weather_not_a_map_search(route):
    assert route("what's the weather near me").name == "weather"


@pytest.mark.parametrize(
    ("text", "action"),
    [
        ("pause", "play_pause"), ("pause the music", "play_pause"), ("resume", "play_pause"),
        ("resume the music", "play_pause"), ("next song", "next"), ("skip", "next"), ("skip this song", "next"),
        ("previous track", "previous"), ("go back", "previous"), ("stop the music", "stop"),
        ("volume up", "volume_up"), ("louder", "volume_up"), ("turn it up", "volume_up"),
        ("turn the volume down", "volume_down"), ("quieter", "volume_down"), ("mute", "mute"), ("unmute", "mute"),
    ],
)
def test_media_controls(route, text, action):
    assert route(text) == ToolCall("media_control", {"action": action})


def test_media_words_inside_other_sentences_are_left_alone(route):
    assert route("what is the next prime number") is None
    assert route("tell me to stop procrastinating") is None


# ------------------------------------------------------------------ video and drive
def _drive_router():
    from jarvis.profile import Profile

    profile = Profile.model_validate({"drives": ["G:\\"], "music": {"folder": "G:\\Music"}})
    tools = ["play_video", "open_drive_folder", "play_media", "open_app", "open_folder", "media_control"]
    return CommandRouter(profile, tools)


@pytest.mark.parametrize(
    "text,name,args",
    [
        ("open the main folder", "open_drive_folder", {"name": "main"}),
        ("open the G drive", "open_drive_folder", {"name": "G"}),
        ("open the movies folder", "open_drive_folder", {"name": "movies"}),
        ("play the movie Dune", "play_video", {"name": "Dune"}),
        ("play a movie", "play_video", {"name": ""}),
        ("watch Dune Part Two", "play_video", {"name": "Dune Part Two"}),
        ("play the movie Dune in VLC", "play_video", {"name": "Dune"}),
        ("open the movie folder and play Dune", "play_video", {"name": "Dune", "folder": "movie"}),
        ("open the series folder and then play Breaking Bad", "play_video", {"name": "Breaking Bad", "folder": "series"}),
    ],
)
def test_drive_and_video_commands(text, name, args):
    call = _drive_router()(text)
    assert call is not None and call.name == name and call.arguments == args


def test_music_commands_stay_music_with_drives_on():
    router = _drive_router()
    assert router("play Love Me by Lil Wayne").name == "play_media"
    assert router("open my music folder and play Kendrick").name == "play_media"
    assert router("play music").name == "play_media"


def test_unknown_app_names_go_to_the_start_menu_lookup():
    router = _drive_router()
    assert router("open Pod Player").arguments == {"name": "Pod Player"}
    assert router("open the Pod Player app").arguments == {"name": "Pod Player"}
    assert router("open it") is None
    assert router("open the file and then print it") is None


# ------------------------------------------------------------------ several steps in one sentence
def _names(call):
    calls = call if isinstance(call, list) else [call]
    return [(c.name, c.arguments) for c in calls]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("in the music folder search Love Me by Lil Wayne and play it in VLC", [("play_media", {"name": "Love Me by Lil Wayne", "source": "local"})]),
        ("in the main folder search a Drake song", [("play_media", {"name": "Drake", "source": "local"})]),
        ("search the music folder for Heartless by The Weeknd", [("play_media", {"name": "Heartless by The Weeknd", "source": "local"})]),
        ("search Love Me and play it", [("play_media", {"name": "Love Me", "source": "local"})]),
        (
            "open the main folder and then search the music folder for Heartless and play it in VLC",
            [("open_drive_folder", {"name": "main"}), ("play_media", {"name": "Heartless", "source": "local"})],
        ),
        ("open the main folder, then open the movies folder", [("open_drive_folder", {"name": "main"}), ("open_drive_folder", {"name": "movies"})]),
        ("open Pod Player and open the main folder", [("open_app", {"name": "Pod Player"}), ("open_drive_folder", {"name": "main"})]),
    ],
)
def test_several_steps_in_one_sentence(text, expected):
    assert _names(_drive_router()(text)) == expected


def test_steps_that_do_not_both_make_sense_go_to_the_model():
    router = _drive_router()
    assert router("open the main folder and tell me a joke") is None
    assert router("how do I make tea and then coffee") is None
