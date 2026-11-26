import pytest

from jarvis.profile import Profile
from jarvis.tools.apps import OpenAppArgs, make_open_app
from jarvis.tools.registry import ToolStatus
from jarvis.tools.startmenu import StartMenu, scan_store_apps


class FakeLauncher:
    def __init__(self):
        self.started, self.opened = [], []

    def start(self, argv, cwd=None):
        self.started.append(argv)

    def open_path(self, path):
        self.opened.append(path)


@pytest.fixture
def menu(tmp_path):
    programs = tmp_path / "Programs"
    for rel in [
        "Pod Player.lnk", "VideoLAN/VLC media player.lnk", "Google Chrome.lnk", "Chrome Remote Desktop.lnk",
        "Adobe/Uninstall Adobe.lnk", "FL Studio 21.lnk", "FL Studio 20.lnk",
    ]:
        path = programs / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    store = [{"Name": "Spotify", "AppID": "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify"}, {"Name": "Bad", "AppID": "C:\\evil.exe"}]
    return StartMenu(folders=lambda: [programs], store_reader=lambda: store)


def test_finds_apps_by_spoken_name(menu):
    assert [a.name for a in menu.find("pod player")] == ["Pod Player"]
    assert [a.name for a in menu.find("the VLC")] == ["VLC media player"]
    assert [a.name for a in menu.find("chrome")] == ["Google Chrome"]  # not "Chrome Remote Desktop"
    assert [a.name for a in menu.find("podplayer")] == ["Pod Player"]
    assert [a.name for a in menu.find("pod playa")] == ["Pod Player"]  # close spelling
    assert menu.find("nothing like this") == []


def test_uninstallers_and_unsafe_ids_are_left_out(menu):
    names = [a.name for a in menu.apps()]
    assert "Uninstall Adobe" not in names and "Bad" not in names and "Spotify" in names
    assert scan_store_apps(lambda: [{"Name": "X", "AppID": "a b; calc"}]) == []


def test_open_app_falls_back_to_the_start_menu(menu):
    launcher = FakeLauncher()
    run = make_open_app(Profile(), launcher, startmenu=menu)
    result = run(OpenAppArgs(name="pod player"))
    assert result.status is ToolStatus.LAUNCHED and result.message == "Opening Pod Player."
    assert launcher.opened[0].name == "Pod Player.lnk"
    run(OpenAppArgs(name="spotify"))
    assert launcher.started == [["explorer.exe", "shell:AppsFolder\\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify"]]


def test_saved_apps_still_win_and_unknown_apps_say_so(menu):
    profile = Profile.model_validate({"apps": {"vlc": {"label": "VLC", "candidates": ["/bin/vlc"]}}})
    launcher = FakeLauncher()
    run = make_open_app(profile, launcher, which=lambda n: None, exists=lambda p: p == "/bin/vlc", startmenu=menu)
    run(OpenAppArgs(name="vlc"))
    assert launcher.started == [["/bin/vlc"]] and launcher.opened == []
    assert "couldn't find an app called zzz" in run(OpenAppArgs(name="zzz")).message


def test_equal_matches_ask_which_one(menu):
    run = make_open_app(Profile(), FakeLauncher(), startmenu=menu)
    result = run(OpenAppArgs(name="fl studio"))
    assert result.status is ToolStatus.FAILED and len(result.choices) == 2
    chosen = run(OpenAppArgs(**result.choices[0][1]))
    assert chosen.status is ToolStatus.LAUNCHED
