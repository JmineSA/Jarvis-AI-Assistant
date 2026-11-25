"""The shipped profile and the new tools, driven through the real brain and executor."""

from datetime import datetime
from pathlib import Path

import pytest

from jarvis.ai.brain import NO_TOOL_CHOSEN
from jarvis.ai.ollama_client import ToolCall
from jarvis.config import PROJECT_ROOT
from jarvis.profile import Profile, load_profile
from jarvis.tools.builtin import build_registry
from jarvis.wiring import build_brain


class FakeLauncher:
    def __init__(self):
        self.started, self.urls = [], []

    def start(self, argv, cwd=None):
        self.started.append(argv)

    def open_url(self, url):
        self.urls.append(url)

    def open_path(self, path):
        raise AssertionError


@pytest.fixture
def shipped():
    return load_profile(PROJECT_ROOT / "config" / "profile.yaml")


def test_shipped_profile_has_the_new_apps_sites_and_location(shipped):
    assert shipped.location.name == "Pretoria" and -27 < shipped.location.latitude < -25
    assert {"vlc", "flstudio"} <= set(shipped.apps)
    assert {"gmail", "youtube", "youtubemusic"} <= set(shipped.websites)
    assert shipped.music.player in shipped.apps
    assert shipped.resolve("websites", "my email") == "gmail"
    assert shipped.resolve("websites", "youtube music") == "youtubemusic"
    assert shipped.resolve("apps", "fruity loops") == "flstudio"
    assert shipped.resolve("apps", "VLC media player") == "vlc"


def test_shipped_registry_offers_every_tool_once(shipped, tmp_path):
    from jarvis.memory.store import MemoryStore

    registry = build_registry(shipped, FakeLauncher(), store=MemoryStore(tmp_path / "j.db"))
    names = registry.names()
    assert len(names) == len(set(names))
    assert {"weather", "web_search", "compose_email", "play_media", "open_website"} <= set(names)


def test_music_and_weather_tools_only_appear_when_configured():
    names = build_registry(Profile(), FakeLauncher()).names()
    assert "weather" not in names and "play_media" not in names


async def test_search_by_voice_end_to_end(settings, fake_client_cls, shipped, monkeypatch):
    launcher = FakeLauncher()
    monkeypatch.setattr("jarvis.wiring.build_registry",
                        lambda profile, **kw: build_registry(profile, launcher, **kw))
    client = fake_client_cls(tool_calls=[ToolCall("web_search", {"query": "best current action movies"})])
    reply = await build_brain(settings, client, profile=shipped).respond("search google for best current action movies")
    assert reply == "Searching Google for best current action movies."
    opened = launcher.started[0][-1] if launcher.started else launcher.urls[0]
    assert opened == "https://www.google.com/search?q=best+current+action+movies"


async def test_weather_by_voice_end_to_end(settings, fake_client_cls, shipped, monkeypatch):
    import jarvis.tools.builtin as builtin

    forecast = {
        "current": {"temperature_2m": 25.0, "weather_code": 0},
        "hourly": {"time": [], "temperature_2m": [], "precipitation_probability": [], "weather_code": []},
        "daily": {"time": [datetime.now().date().isoformat()], "temperature_2m_max": [30], "temperature_2m_min": [18],
                  "precipitation_probability_max": [10], "weather_code": [0]},
    }
    original = builtin.make_weather
    monkeypatch.setattr(builtin, "make_weather", lambda loc, clock: original(loc, clock, lambda l: forecast))
    client = fake_client_cls(tool_calls=[ToolCall("weather", {"when": ""})])
    reply = await build_brain(settings, client, profile=shipped).respond("what's the weather")
    assert reply.startswith("Right now in Pretoria it's 25 degrees and clear.")


@pytest.mark.parametrize(
    "claim", ["Searching Google for that now.", "Playing the album in VLC.", "Composing your email.", "I'm drafting it now."]
)
async def test_claiming_these_actions_without_a_tool_is_caught(settings, fake_client_cls, shipped, claim):
    brain = build_brain(settings, fake_client_cls([claim]), profile=shipped)
    assert await brain.respond("do it") == NO_TOOL_CHOSEN


def test_the_prompt_tells_the_model_about_the_new_abilities():
    from jarvis.ai.prompts import build_tool_system_prompt

    prompt = build_tool_system_prompt("Lesiba", memory=True)
    assert "weather" in prompt and "play music" in prompt and "write an email" in prompt
    assert "You cannot play music" not in prompt
