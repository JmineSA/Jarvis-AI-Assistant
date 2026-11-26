import pytest

from jarvis.profile import Profile
from jarvis.tools.media import PlayMediaArgs, make_play_media
from jarvis.tools.registry import ToolStatus


class FakeLauncher:
    def __init__(self, fail=None):
        self.fail = fail
        self.started = []
        self.urls = []

    def start(self, argv, cwd=None):
        if self.fail:
            raise self.fail
        self.started.append(argv)

    def open_url(self, url):
        self.urls.append(url)


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


def play(profile, launcher, name="", which=lambda n: f"/bin/{n}", source="auto"):
    def no_network(_query):
        raise AssertionError("these tests must not touch the network")

    return make_play_media(profile, launcher, which, lambda p: False, no_network, lambda: False)(
        PlayMediaArgs(name=name, source=source)
    )


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
    assert launcher.started[0] == ["/bin/vlc", "--random", str(music)]  # shuffled
    assert result.message == "Playing your music folder in VLC."


def test_equal_matches_ask_which_one(profile):
    launcher = FakeLauncher()
    result = play(profile, launcher, "gym mix")
    assert result.status is ToolStatus.FAILED and "matches 2 things" in result.message
    assert launcher.started == []
    assert len(result.choices) == 2 and result.choices[0][1]["source"] == "local"


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


# ------------------------------------------------------------------ deeper folders and sources


@pytest.fixture(autouse=True)
def _fresh_index():
    from jarvis.tools.media import forget_index

    forget_index()


@pytest.fixture
def deep_music(tmp_path):
    root = tmp_path / "Library"
    album = root / "Lil Wayne" / "2008" / "Tha Carter III" / "Disc 1"
    album.mkdir(parents=True)
    (album / "05 Lollipop.mp3").write_bytes(b"")
    (album / "09 Love Me.mp3").write_bytes(b"")
    (root / "Lil Wayne" / "2008" / "Tha Carter III" / "Disc 1" / "09 Love Me (Remix).mp3").write_bytes(b"")
    (root / "Drake").mkdir()
    (root / "Drake" / "Passionfruit.mp3").write_bytes(b"")
    return root


@pytest.fixture
def deep_profile(deep_music):
    return Profile.model_validate(
        {"music": {"folder": str(deep_music)}, "apps": {"vlc": {"label": "VLC", "candidates": ["vlc"]}}}
    )


def build(profile, launcher, resolver=None, online=lambda: True):
    def no_resolver(_q):
        raise AssertionError("no network expected")

    return make_play_media(profile, launcher, lambda n: f"/bin/{n}", lambda p: False, resolver or no_resolver, online)


def test_songs_several_folders_deep_are_found(deep_profile, deep_music):
    launcher = FakeLauncher()
    result = build(deep_profile, launcher)(PlayMediaArgs(name="Love Me by Lil Wayne", source="local"))
    assert result.status is ToolStatus.LAUNCHED
    assert launcher.started[0][1].endswith("09 Love Me.mp3")  # the closest title, not the remix


def test_whole_words_only(deep_profile):
    # "me" must not match inside "Passionfruit"-style substrings
    result = build(deep_profile, FakeLauncher())(PlayMediaArgs(name="ionfr", source="local"))
    assert result.status is ToolStatus.FAILED


def test_auto_online_uses_youtube_music_first(deep_profile):
    launcher = FakeLauncher()
    result = build(deep_profile, launcher, lambda q: "abcdefghijk")(PlayMediaArgs(name="Love Me by Lil Wayne"))
    assert "YouTube Music" in result.message
    assert launcher.urls == ["https://music.youtube.com/watch?v=abcdefghijk"] and launcher.started == []


def test_auto_offline_plays_from_the_music_folder_without_touching_the_network(deep_profile):
    launcher = FakeLauncher()
    result = build(deep_profile, launcher, online=lambda: False)(PlayMediaArgs(name="Love Me by Lil Wayne"))
    assert "VLC" in result.message and launcher.started[0][1].endswith("09 Love Me.mp3")
    assert launcher.urls == []


def test_auto_online_but_youtube_unreachable_falls_back_to_the_folder(deep_profile):
    import httpx

    def down(_q):
        raise httpx.ConnectError("blocked")

    launcher = FakeLauncher()
    result = build(deep_profile, launcher, down)(PlayMediaArgs(name="Love Me by Lil Wayne"))
    assert "VLC" in result.message and launcher.urls == []


def test_the_online_source_can_be_switched_to_local(deep_music):
    profile = Profile.model_validate(
        {"music": {"folder": str(deep_music), "online": "local"}, "apps": {"vlc": {"label": "VLC", "candidates": ["vlc"]}}}
    )
    launcher = FakeLauncher()
    result = build(profile, launcher)(PlayMediaArgs(name="Love Me by Lil Wayne"))  # online, but prefers local
    assert "VLC" in result.message and launcher.urls == []
    result = build(profile, launcher, lambda q: "abcdefghijk")(PlayMediaArgs(name="Some Song Nobody Owns"))
    assert "YouTube Music" in result.message  # not in the folder, so it still streams


def test_the_online_source_can_be_plain_youtube(deep_music):
    profile = Profile.model_validate(
        {"music": {"folder": str(deep_music), "online": "youtube"}, "apps": {"vlc": {"candidates": ["vlc"]}}}
    )
    launcher = FakeLauncher()
    build(profile, launcher, lambda q: "abcdefghijk")(PlayMediaArgs(name="Love Me"))
    assert launcher.urls == ["https://www.youtube.com/watch?v=abcdefghijk"]


def test_auto_falls_back_to_youtube_music_when_not_local_and_online(deep_profile):
    launcher = FakeLauncher()
    seen = []

    def resolver(query):
        seen.append(query)
        return "dQw4w9WgXcQ"

    result = build(deep_profile, launcher, resolver)(PlayMediaArgs(name="Some Song Nobody Owns"))
    assert seen == ["Some Song Nobody Owns"]
    assert result.status is ToolStatus.LAUNCHED and "YouTube Music" in result.message
    assert launcher.urls == ["https://music.youtube.com/watch?v=dQw4w9WgXcQ"]


def test_auto_offline_and_not_local_says_so(deep_profile):
    launcher = FakeLauncher()
    result = build(deep_profile, launcher, online=lambda: False)(PlayMediaArgs(name="Some Song Nobody Owns"))
    assert result.status is ToolStatus.FAILED and "offline" in result.message
    assert launcher.started == [] and launcher.urls == []


def test_missing_music_folder_does_not_stop_streaming_when_online(tmp_path):
    profile = Profile.model_validate(
        {"music": {"folder": str(tmp_path / "unplugged")}, "apps": {"vlc": {"candidates": ["vlc"]}}}
    )
    launcher = FakeLauncher()
    result = build(profile, launcher, lambda q: "abcdefghijk")(PlayMediaArgs(name="Love Me"))
    assert result.status is ToolStatus.LAUNCHED and "YouTube Music" in result.message


def test_missing_music_folder_offline_explains_the_problem(tmp_path):
    profile = Profile.model_validate(
        {"music": {"folder": str(tmp_path / "unplugged")}, "apps": {"vlc": {"candidates": ["vlc"]}}}
    )
    result = build(profile, FakeLauncher(), online=lambda: False)(PlayMediaArgs(name="Love Me"))
    assert result.status is ToolStatus.FAILED and "can't find your music folder" in result.message and "offline" in result.message


def test_explicit_youtube_music_goes_straight_to_the_video(deep_profile):
    launcher = FakeLauncher()
    result = build(deep_profile, launcher, lambda q: "abcdefghijk")(PlayMediaArgs(name="Love Me", source="youtube_music"))
    assert result.message == "Playing the top YouTube Music result for Love Me."
    assert launcher.urls == ["https://music.youtube.com/watch?v=abcdefghijk"]


def test_plain_youtube_uses_the_youtube_address(deep_profile):
    launcher = FakeLauncher()
    build(deep_profile, launcher, lambda q: "abcdefghijk")(PlayMediaArgs(name="Love Me", source="youtube"))
    assert launcher.urls == ["https://www.youtube.com/watch?v=abcdefghijk"]


def test_no_result_found_opens_the_search_page_instead(deep_profile):
    launcher = FakeLauncher()
    result = build(deep_profile, launcher, lambda q: None)(PlayMediaArgs(name="Love Me & more", source="youtube_music"))
    assert result.status is ToolStatus.LAUNCHED and "couldn't pick a result" in result.message
    assert launcher.urls == ["https://music.youtube.com/search?q=Love+Me+%26+more"]


def test_unreachable_youtube_is_reported_and_nothing_opens(deep_profile):
    import httpx

    def down(_q):
        raise httpx.ConnectError("offline")

    launcher = FakeLauncher()
    result = build(deep_profile, launcher, down)(PlayMediaArgs(name="Love Me", source="youtube_music"))
    assert result.status is ToolStatus.FAILED and "Are you online" in result.message
    assert launcher.urls == [] and launcher.started == []


def test_youtube_music_with_no_song_opens_the_site(deep_profile):
    launcher = FakeLauncher()
    build(deep_profile, launcher)(PlayMediaArgs(name="", source="youtube_music"))
    assert launcher.urls == ["https://music.youtube.com"]


def test_a_video_id_from_the_page_is_extracted_safely():
    import httpx

    from jarvis.tools.media import fetch_video_id

    page = '{"adSlotRenderer":{"videoId":"AAAAAAAAAAA"}} {"videoRenderer":{"videoId":"abcDEF_-123","title":1}}'
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=page))
    assert fetch_video_id("love me", transport=transport) == "abcDEF_-123"
    assert fetch_video_id("x", transport=httpx.MockTransport(lambda r: httpx.Response(200, text="nothing"))) is None
    with pytest.raises(httpx.HTTPError):
        fetch_video_id("x", transport=httpx.MockTransport(lambda r: httpx.Response(500)))


def test_friendly_source_words_are_understood():
    assert PlayMediaArgs(name="x", source="VLC").source == "local"
    assert PlayMediaArgs(name="x", source="YouTube Music").source == "youtube_music"
    assert PlayMediaArgs(name="x", source="yt music").source == "youtube_music"
    assert PlayMediaArgs(name="x").source == "auto"


def test_music_check_reports_and_dry_runs(deep_profile, deep_music, settings, monkeypatch, capsys):
    import functools

    from jarvis import diagnostics

    monkeypatch.setattr(diagnostics, "load_user_profile", lambda s: deep_profile)
    monkeypatch.setattr(
        diagnostics, "make_play_media",
        functools.partial(make_play_media, which=lambda n: f"/bin/{n}", exists=lambda p: False),
    )
    code = diagnostics.run_music_check(settings, ["love", "me", "lil", "wayne"])
    out = capsys.readouterr().out
    assert code == 0 and "Found 4 audio files" in out and "would start" in out and "09 Love Me.mp3" in out


def test_artist_and_album_names_become_recogniser_hints(deep_profile):
    from jarvis.tools.media import hint_names

    names = hint_names(deep_profile)
    assert names[:2] == ["Lil Wayne", "Drake"] or set(names[:2]) == {"Lil Wayne", "Drake"}
    assert "Tha Carter III" in hint_names(deep_profile)
    assert hint_names(Profile()) == []
