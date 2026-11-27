import pytest

from jarvis.profile import Profile
from jarvis.tools import drives
from jarvis.tools.drives import (
    OpenDriveFolderArgs,
    PlayVideoArgs,
    State,
    make_open_drive_folder,
    make_play_video,
)
from jarvis.tools.registry import ToolStatus


class FakeLauncher:
    def __init__(self):
        self.started = []
        self.opened = []

    def start(self, argv, cwd=None):
        self.started.append(argv)

    def open_path(self, path):
        self.opened.append(path)

    def open_url(self, url):
        pass


@pytest.fixture(autouse=True)
def _fresh_index():
    drives.forget_index()
    yield
    drives.forget_index()


@pytest.fixture
def drive(tmp_path):
    root = tmp_path / "G"
    for rel in [
        "Movies/Dune (2021)/Dune.2021.1080p.mkv",
        "Movies/Heat (1995)/Heat.mkv",
        "Movies/Gym Night One.mp4",
        "Movies/Gym Night Two.mp4",
        "Series/Breaking Bad/S01E01.mkv",
        "Series/Breaking Bad/S01E02.mkv",
        "Games/Elden Ring/readme.txt",
        "Study/notes.pdf",
        "Music/Artist/song.mp3",
    ]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    return root


@pytest.fixture
def profile(drive):
    return Profile.model_validate(
        {"drives": [str(drive)], "music": {"player": "vlc"}, "apps": {"vlc": {"label": "VLC", "candidates": ["/bin/vlc"]}}}
    )


def opener(profile):
    launcher, state = FakeLauncher(), State()
    return make_open_drive_folder(profile, launcher, state), launcher, state


def player(profile, state=None, launcher=None):
    launcher = launcher or FakeLauncher()
    state = state or State()
    return make_play_video(profile, launcher, state, which=lambda n: None, exists=lambda p: p == "/bin/vlc"), launcher, state


def test_main_folder_opens_the_drive(profile, drive):
    run, launcher, _ = opener(profile)
    for phrase in ["main", "the main folder", "root"]:
        assert run(OpenDriveFolderArgs(name=phrase)).status is ToolStatus.LAUNCHED
    assert launcher.opened == [drive] * 3


def test_folder_found_by_name_even_plural(profile, drive):
    run, launcher, state = opener(profile)
    for phrase, expected in [("movie", "Movies"), ("movies folder", "Movies"), ("series", "Series"), ("games", "Games")]:
        assert run(OpenDriveFolderArgs(name=phrase)).status is ToolStatus.LAUNCHED, phrase
        assert launcher.opened[-1] == drive / expected
    assert state.folder == drive / "Games"


def test_unknown_folder_and_no_drive(profile):
    run, launcher, _ = opener(profile)
    assert "couldn't find a folder" in run(OpenDriveFolderArgs(name="holidays")).message
    assert launcher.opened == []
    empty, _, _ = opener(Profile())
    assert "No drive is set up" in empty(OpenDriveFolderArgs(name="movies")).message


def test_an_exact_path_must_be_inside_the_drive(profile, drive, tmp_path):
    run, launcher, _ = opener(profile)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    run(OpenDriveFolderArgs(name=str(outside)))
    sneaky = drive / "Movies" / ".." / ".." / "elsewhere"
    run(OpenDriveFolderArgs(name=str(sneaky)))
    assert launcher.opened == []
    assert run(OpenDriveFolderArgs(name=str(drive / "Series"))).status is ToolStatus.LAUNCHED


def test_play_a_movie_by_name(profile, drive):
    run, launcher, _ = player(profile)
    result = run(PlayVideoArgs(name="Heat"))
    assert result.status is ToolStatus.LAUNCHED and result.message == "Playing Heat in VLC."
    assert launcher.started == [["/bin/vlc", str(drive / "Movies/Heat (1995)/Heat.mkv")]]


def test_series_name_plays_the_whole_folder(profile, drive):
    run, launcher, _ = player(profile)
    assert run(PlayVideoArgs(name="Breaking Bad")).status is ToolStatus.LAUNCHED
    assert launcher.started == [["/bin/vlc", str(drive / "Series" / "Breaking Bad")]]


def test_ambiguous_title_asks_which_one_with_choices(profile, drive):
    run, launcher, _ = player(profile)
    result = run(PlayVideoArgs(name="gym night"))
    assert result.status is ToolStatus.FAILED and "Which one?" in result.message
    assert len(result.choices) == 2 and launcher.started == []
    chosen = run(PlayVideoArgs(**result.choices[1][1]))
    assert chosen.status is ToolStatus.LAUNCHED and launcher.started[0][1].endswith("Gym Night Two.mp4")


def test_folder_argument_and_last_opened_folder_narrow_the_search(profile, drive):
    run, launcher, state = player(profile)
    assert run(PlayVideoArgs(name="S01E02", folder="series")).status is ToolStatus.LAUNCHED
    assert launcher.started[-1][1].endswith("S01E02.mkv")
    assert state.folder == drive / "Series" / "Breaking Bad"
    assert "couldn't find a video" in run(PlayVideoArgs(name="Heat")).message  # not inside that folder


def test_empty_name_plays_the_last_folder_or_asks(profile, drive):
    run, launcher, state = player(profile)
    assert "Which movie?" in run(PlayVideoArgs()).message
    state.folder = drive / "Movies"
    assert run(PlayVideoArgs()).status is ToolStatus.LAUNCHED
    assert launcher.started[-1] == ["/bin/vlc", str(drive / "Movies")]


def test_nothing_found_and_missing_vlc(profile):
    run, launcher, _ = player(profile)
    assert "couldn't find a video matching" in run(PlayVideoArgs(name="Titanic")).message
    no_vlc = make_play_video(profile, launcher, State(), which=lambda n: None, exists=lambda p: False)
    assert "couldn't find VLC" in no_vlc(PlayVideoArgs(name="Heat")).message


def test_only_video_files_are_playable(profile):
    run, _, _ = player(profile)
    assert run(PlayVideoArgs(name="readme")).status is ToolStatus.FAILED
    assert run(PlayVideoArgs(name="notes")).status is ToolStatus.FAILED


def test_hint_names_and_registration(profile):
    assert {"Movies", "Series", "Games"} <= set(drives.hint_names(profile))
    from jarvis.tools.builtin import build_registry

    assert "play_video" in build_registry(profile).names()
    assert "play_video" not in build_registry(Profile()).names()


def test_play_media_falls_back_to_a_video_when_the_music_folder_has_no_match(profile, drive, tmp_path):
    from jarvis.tools.media import PlayMediaArgs, make_play_media

    music = tmp_path / "Music"
    music.mkdir()
    full = Profile.model_validate(
        {"drives": [str(drive)], "music": {"folder": str(music), "player": "vlc"},
         "apps": {"vlc": {"label": "VLC", "candidates": ["/bin/vlc"]}}}
    )
    video, launcher, _ = player(full)
    run = make_play_media(full, launcher, which=lambda n: None, exists=lambda p: p == "/bin/vlc",
                          resolver=lambda q: None, online=lambda: False,
                          video_fallback=lambda q: video(PlayVideoArgs(name=q)))
    result = run(PlayMediaArgs(name="Breaking Bad"))
    assert result.status is ToolStatus.LAUNCHED
    assert launcher.started == [["/bin/vlc", str(drive / "Series" / "Breaking Bad")]]
    assert run(PlayMediaArgs(name="Titanic")).status is ToolStatus.FAILED


def test_exact_title_only_matches_the_whole_title(tmp_path):
    from jarvis.tools.media import PlayMediaArgs, make_play_media

    root = tmp_path / "G"
    for rel in ["Movies & Series/Movies/2026/The Run (2026)/The.Run.2026.1080p.mp4", "Movies & Series/Movies/2024/Run Lola Run/Run.Lola.Run.mkv"]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    prof = Profile.model_validate(
        {"drives": [str(root)], "music": {"folder": str(tmp_path / "none"), "online": "youtube_music", "player": "vlc"},
         "apps": {"vlc": {"label": "VLC", "candidates": ["/bin/vlc"]}}}
    )
    video, launcher, _ = player(prof)
    web = []
    run = make_play_media(prof, launcher, which=lambda n: None, exists=lambda p: p == "/bin/vlc",
                          resolver=lambda q: web.append(q) or "abcdefghijk", online=lambda: True,
                          video_exact=lambda q: video(PlayVideoArgs(name=q, exact=True)))
    result = run(PlayMediaArgs(name="the run"))
    assert result.status is ToolStatus.LAUNCHED and web == []  # your own "The Run" wins, YouTube is not asked
    assert launcher.started[0][1].endswith("The.Run.2026.1080p.mp4")
    run(PlayMediaArgs(name="lola"))  # only part of a title: goes to YouTube Music as before
    assert web == ["lola"]


def test_misheard_movie_title_still_finds_the_file(profile, drive):
    run, launcher, _ = player(profile)
    assert run(PlayVideoArgs(name="Xyzzy")).status is ToolStatus.FAILED  # nothing close: not guessed
    assert run(PlayVideoArgs(name="Heats")).status is ToolStatus.LAUNCHED
    assert launcher.started[-1][1].endswith("Heat.mkv")


def test_misheard_multi_word_series_name(profile, drive):
    run, launcher, _ = player(profile)
    assert run(PlayVideoArgs(name="Braking Bad")).status is ToolStatus.LAUNCHED
    assert launcher.started[-1] == ["/bin/vlc", str(drive / "Series" / "Breaking Bad")]


def test_exact_mode_does_not_guess(profile, drive):
    run, launcher, _ = player(profile)
    assert run(PlayVideoArgs(name="Heet", exact=True)).status is ToolStatus.FAILED
    assert launcher.started == []
