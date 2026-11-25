import pytest

from jarvis.profile import Profile
from jarvis.tools.media import PlayMediaArgs, make_play_media
from jarvis.tools.registry import ToolStatus


class FakeLauncher:
    def __init__(self, fail=None):
        self.fail = fail
        self.started = []

    def start(self, argv, cwd=None):
        if self.fail:
            raise self.fail
        self.started.append(argv)


@pytest.fixture
def music(tmp_path):
    root = tmp_path / "Music"
    for folder, files in {
        "Kendrick Lamar/GNX": ["01 wacced out murals.mp3"],
        "Kendrick Lamar/DAMN": ["01 DNA.mp3"],
        "Daft Punk/Discovery": ["01 One More Time.flac"],
        "Daft Punk/Random Access Memories": ["01 Give Life Back to Music.flac"],
        "Various - Gym Mix 2024": ["01 a.mp3"],
        "Various - Gym Mix 2023": ["01 b.mp3"],
        "Lo-fi Beats": ["study.mp3"],
    }.items():
        (root / folder).mkdir(parents=True)
        for name in files:
            (root / folder / name).write_bytes(b"")
    (root / "Kendrick Lamar" / "notes.txt").write_text("not audio")
    return root


@pytest.fixture
def profile(music):
    return Profile.model_validate(
        {
            "music": {"folder": str(music), "player": "vlc"},
            "apps": {"vlc": {"label": "VLC", "names": ["media player"], "candidates": ["vlc"]}},
        }
    )


def play(profile, launcher, name="", which=lambda n: f"/bin/{n}"):
    return make_play_media(profile, launcher, which, lambda p: False)(PlayMediaArgs(name=name))


def test_plays_the_best_matching_album_folder(profile, music):
    launcher = FakeLauncher()
    result = play(profile, launcher, "Kendrick Lamar GNX")
    assert result.status is ToolStatus.LAUNCHED
    assert launcher.started == [["/bin/vlc", str(music / "Kendrick Lamar" / "GNX")]]
    assert result.message == "Playing GNX in VLC."


def test_filler_words_and_case_do_not_matter(profile, music):
    launcher = FakeLauncher()
    play(profile, launcher, "the Discovery album by daft punk")
    assert launcher.started[0][1] == str(music / "Daft Punk" / "Discovery")


def test_artist_alone_plays_the_whole_artist_folder(profile, music):
    launcher = FakeLauncher()
    result = play(profile, launcher, "Kendrick Lamar")
    assert launcher.started[0][1] == str(music / "Kendrick Lamar")
    assert result.message == "Playing Kendrick Lamar in VLC."


def test_a_single_song_file_can_be_named(profile, music):
    launcher = FakeLauncher()
    play(profile, launcher, "give life back to music")
    assert launcher.started[0][1].endswith("01 Give Life Back to Music.flac")


def test_empty_name_plays_the_whole_music_folder(profile, music):
    launcher = FakeLauncher()
    result = play(profile, launcher, "")
    assert launcher.started[0][1] == str(music)
    assert result.message == "Playing your music folder in VLC."


def test_equal_matches_ask_which_one(profile):
    launcher = FakeLauncher()
    result = play(profile, launcher, "gym mix")
    assert result.status is ToolStatus.FAILED and "matches 2 things" in result.message
    assert launcher.started == []


def test_no_match_says_so_and_starts_nothing(profile):
    launcher = FakeLauncher()
    result = play(profile, launcher, "Metallica Black Album")
    assert result.status is ToolStatus.FAILED and "couldn't find anything matching" in result.message
    assert launcher.started == []


def test_one_word_of_several_is_not_enough(profile):
    # "Lamar" alone would match, but "Lamar Metallica Reload" matches only 1 of 3 words.
    assert play(profile, FakeLauncher(), "Lamar Metallica Reload").status is ToolStatus.FAILED


def test_no_music_folder_configured():
    result = play(Profile(), FakeLauncher())
    assert result.status is ToolStatus.FAILED and "music" in result.message


def test_missing_music_folder_names_the_path(profile, tmp_path):
    bad = Profile.model_validate(
        {"music": {"folder": str(tmp_path / "nope")}, "apps": {"vlc": {"candidates": ["vlc"]}}}
    )
    result = play(bad, FakeLauncher())
    assert result.status is ToolStatus.FAILED and str(tmp_path / "nope") in result.message


def test_player_not_installed(profile):
    result = play(profile, FakeLauncher(), "GNX", which=lambda n: None)
    assert result.status is ToolStatus.FAILED and "couldn't find VLC" in result.message


def test_player_not_in_profile(music):
    result = play(Profile.model_validate({"music": {"folder": str(music)}}), FakeLauncher())
    assert result.status is ToolStatus.FAILED


def test_start_failure_is_reported(profile):
    result = play(profile, FakeLauncher(fail=OSError("denied")), "GNX")
    assert result.status is ToolStatus.FAILED and "couldn't start VLC" in result.message


def test_spoken_words_never_become_extra_arguments(profile):
    launcher = FakeLauncher()
    play(profile, launcher, "GNX --fullscreen ; calc.exe")
    assert all(len(argv) == 2 for argv in launcher.started)
