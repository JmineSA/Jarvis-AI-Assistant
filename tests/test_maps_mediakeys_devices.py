import pytest

from jarvis.audio_devices import (
    AudioDevice,
    list_devices,
    loudest,
    measure_levels,
    play_test_tone,
)
from jarvis.profile import Profile
from jarvis.tools.maps import MapsArgs, make_maps
from jarvis.tools.mediakeys import KEYS, MediaControlArgs, make_media_control
from jarvis.tools.registry import ToolStatus


class FakeLauncher:
    def __init__(self):
        self.urls = []
        self.started = []

    def open_url(self, url):
        self.urls.append(url)

    def start(self, argv, cwd=None):
        self.started.append(argv)


def profile(address="", location=True):
    data = {"apps": {"chrome": {"label": "Chrome", "candidates": ["chrome"]}}, "websites": {}}
    if location:
        data["location"] = {"name": "Pretoria", "latitude": -25.7479, "longitude": 28.2293, "address": address}
    return Profile.model_validate(data)


def run_maps(p, query, mode="search"):
    launcher = FakeLauncher()
    result = make_maps(p, launcher, lambda n: f"/bin/{n}", lambda x: False)(MapsArgs(query=query, mode=mode))
    return result, launcher


def test_search_near_the_typed_home_address():
    result, launcher = run_maps(profile("6365 Nellmapius Drive, Pretoria"), "police station")
    assert result.status is ToolStatus.LAUNCHED and "near your home" in result.message
    url = (launcher.urls + [a[-1] for a in launcher.started])[0]
    assert url.startswith("https://www.google.com/maps/search/?api=1&query=police+station+near+6365+Nellmapius+Drive")


def test_search_without_an_address_centres_on_the_weather_location():
    _, launcher = run_maps(profile(""), "pharmacy")
    url = (launcher.urls + [a[-1] for a in launcher.started])[0]
    assert url == "https://www.google.com/maps/search/pharmacy/@-25.74790,28.22930,14z"


def test_directions_start_from_home():
    _, launcher = run_maps(profile("6365 Nellmapius Drive, Pretoria"), "Menlyn Park", "directions")
    url = (launcher.urls + [a[-1] for a in launcher.started])[0]
    assert "destination=Menlyn+Park" in url and "origin=6365+Nellmapius+Drive%2C+Pretoria" in url


def test_maps_still_work_with_no_home_at_all():
    result, _launcher = run_maps(profile(location=False), "ATM")
    assert result.status is ToolStatus.LAUNCHED and "near your home" not in result.message


def test_odd_characters_cannot_break_out_of_the_address():
    _, launcher = run_maps(profile("1 Road"), "pizza & beer #1 ?x=1")
    url = (launcher.urls + [a[-1] for a in launcher.started])[0]
    assert " " not in url and "&query=pizza+%26+beer+%231+%3Fx%3D1+near+1+Road" in url


# ------------------------------------------------------------------ media keys


class FakeKeys:
    def __init__(self):
        self.pressed = []

    def press(self, key):
        self.pressed.append(key)


@pytest.mark.parametrize(
    ("said", "action", "presses"),
    [("pause", "play_pause", 1), ("resume", "play_pause", 1), ("skip", "next", 1), ("next track", "next", 1),
     ("previous", "previous", 1), ("stop", "stop", 1), ("louder", "volume_up", 5), ("quieter", "volume_down", 5),
     ("mute", "mute", 1), ("volume_up", "volume_up", 5)],
)
def test_media_actions_press_only_the_fixed_keys(said, action, presses):
    keys = FakeKeys()
    result = make_media_control(keys)(MediaControlArgs(action=said))
    assert result.status is ToolStatus.DONE
    assert keys.pressed == [KEYS[action]] * presses


def test_unknown_actions_are_rejected():
    with pytest.raises(ValueError):
        MediaControlArgs(action="format c:")


def test_media_keys_say_so_when_not_on_windows():
    result = make_media_control(None, platform=lambda: "linux")(MediaControlArgs(action="next"))
    assert result.status is ToolStatus.FAILED and "Windows" in result.message


# ------------------------------------------------------------------ audio devices


class FakeSd:
    def __init__(self, loud=None):
        self.default = type("D", (), {"hostapi": 0})()
        self.devices = [
            {"name": "Speakers", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100},
            {"name": "Mic 1", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000},
            {"name": "Mic 2", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 16000},
            {"name": "Mic 1 (WASAPI)", "hostapi": 1, "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 48000},
        ]
        self.loud = loud or {}
        self.last = None
        self.played = None

    def query_devices(self, index=None):
        return self.devices if index is None else self.devices[index]

    def rec(self, frames, samplerate, channels, device, dtype):
        import numpy as np

        self.last = device
        return np.full((frames, 1), self.loud.get(device, 0.001), dtype="float32")

    def wait(self):
        pass

    def play(self, tone, rate, device=None):
        self.played = (len(tone), rate, device)


def test_devices_listed_from_the_default_audio_system_only():
    inputs, outputs = list_devices(FakeSd())
    assert [d.label for d in inputs] == ["1 · Mic 1", "2 · Mic 2"]
    assert [d.label for d in outputs] == ["0 · Speakers"]


def test_listing_never_raises_when_audio_is_unavailable():
    class Broken:
        @property
        def default(self):
            raise OSError("no portaudio")

    assert list_devices(Broken()) == ([], [])


def test_detect_picks_the_microphone_with_sound():
    sd = FakeSd(loud={1: 0.02, 2: 0.4})
    levels = measure_levels(list_devices(sd)[0], sd, seconds=0.01)
    assert loudest(levels) == 2


def test_detect_says_nothing_when_quiet_or_unclear():
    assert loudest({1: 0.001, 2: 0.002}) is None
    assert loudest({1: 0.30, 2: 0.28}) is None        # both loud: can't tell
    assert loudest({}) is None


def test_a_microphone_that_cannot_be_opened_scores_zero():
    class Failing(FakeSd):
        def rec(self, *a, **k):
            raise OSError("busy")

    sd = Failing()
    assert measure_levels(list_devices(sd)[0], sd, seconds=0.01) == {1: 0.0, 2: 0.0}


def test_test_tone_plays_on_the_chosen_output():
    sd = FakeSd()
    play_test_tone(0, sd, seconds=0.1)
    assert sd.played[2] == 0 and sd.played[0] > 0


def test_audio_device_label():
    assert AudioDevice(3, "Focusrite").label == "3 · Focusrite"
