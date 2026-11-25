import pytest

from jarvis.appfinder import add_to_profile, entry_key, find_programs, run_add_app
from jarvis.profile import load_profile

PROFILE = """# my profile
owner_name: Lesiba

apps:
  # comment kept
  chrome:
    label: Chrome
    candidates: [chrome]

folders:
  docs:
    path: 'C:\\Docs'
"""


@pytest.fixture
def programs(tmp_path):
    root = tmp_path / "Program Files"
    for rel in [
        "Image-Line/FL Studio 21/FL64.exe",
        "Image-Line/FL Studio 21/unins000.exe",
        "VideoLAN/VLC/vlc.exe",
        "Other/notepad2.exe",
        "Deep/a/b/c/d/FL Studio hidden.exe",
    ]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return root


def find(query, programs, registry=lambda: []):
    return find_programs(query, roots=[programs], registry=registry)


def test_finds_a_program_by_its_folder_name(programs):
    found = find("FL Studio", programs)
    assert found and found[0].name == "FL64.exe"


def test_finds_by_exe_name(programs):
    assert [p.name for p in find("vlc", programs)] == ["vlc.exe"]


def test_uninstallers_are_skipped(programs):
    assert all("unins" not in p.name for p in find("FL Studio", programs))


def test_search_is_depth_limited_and_never_leaves_the_roots(programs):
    assert not any("hidden" in p.name for p in find("FL Studio hidden", programs))


def test_nothing_found(programs):
    assert find("Photoshop", programs) == []


def test_registry_entries_are_included(programs, tmp_path):
    exe = tmp_path / "Elsewhere" / "Spotify.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    assert find("Spotify", programs, registry=lambda: [exe]) == [exe]


def test_a_broken_registry_does_not_stop_the_search(programs):
    def boom():
        raise RuntimeError("registry unreadable")

    assert find("vlc", programs, registry=boom)


def test_add_to_profile_keeps_comments_backs_up_and_loads(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text(PROFILE, encoding="utf-8")
    key = add_to_profile(path, "FL Studio", ["fl", "fruity loops"], r"C:\Program Files\Image-Line\FL Studio 21\FL64.exe")
    assert key == "flstudio"
    text = path.read_text(encoding="utf-8")
    assert "# my profile" in text and "# comment kept" in text
    assert (tmp_path / "profile.yaml.bak").read_text(encoding="utf-8") == PROFILE
    profile = load_profile(path)
    assert profile.apps["flstudio"].candidates == [r"C:\Program Files\Image-Line\FL Studio 21\FL64.exe"]
    assert profile.resolve("apps", "fruity loops") == "flstudio"
    assert "chrome" in profile.apps and "docs" in profile.folders


def test_quotes_in_names_and_paths_stay_valid_yaml(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text(PROFILE, encoding="utf-8")
    add_to_profile(path, "Bob's App", ["it's bob"], r"C:\Bob's Apps\bob.exe")
    profile = load_profile(path)
    assert profile.apps["bobs"].candidates == [r"C:\Bob's Apps\bob.exe"]


def test_adding_a_duplicate_is_refused_and_changes_nothing(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text(PROFILE, encoding="utf-8")
    with pytest.raises(ValueError, match="already"):
        add_to_profile(path, "Chrome", [], "x.exe")
    assert path.read_text(encoding="utf-8") == PROFILE


def test_profile_without_apps_section_gets_one(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text("owner_name: Lesiba\n", encoding="utf-8")
    add_to_profile(path, "VLC", [], "vlc.exe")
    assert load_profile(path).apps["vlc"].candidates == ["vlc.exe"]


def test_missing_profile_file_is_created(tmp_path):
    path = tmp_path / "new.yaml"
    add_to_profile(path, "VLC", [], "vlc.exe")
    assert "vlc" in load_profile(path).apps


def test_entry_key():
    assert entry_key("FL Studio") == "flstudio" and entry_key("???") == "app"


# --------------------------------------------------------------------------- the interactive flow


def run(tmp_path, words, answers, found):
    path = tmp_path / "profile.yaml"
    path.write_text(PROFILE, encoding="utf-8")
    lines, replies = [], iter(answers)
    code = run_add_app(words, path, out=lines.append, ask=lambda prompt: next(replies), find=lambda q: found)
    return code, "\n".join(lines), path


def test_flow_adds_the_chosen_match(tmp_path):
    from pathlib import Path

    exe = Path("C:/x/FL64.exe")
    code, text, path = run(tmp_path, ["FL", "Studio"], ["1", "", "fl, fruity loops"], [exe])
    assert code == 0 and "Added 'FL Studio'" in text and "open FL Studio" in text
    assert load_profile(path).resolve("apps", "fruity loops") == "flstudio"


def test_flow_cancel_changes_nothing(tmp_path):
    from pathlib import Path

    code, text, path = run(tmp_path, ["FL"], [""], [Path("a.exe")])
    assert code == 1 and "Nothing was changed" in text and path.read_text(encoding="utf-8") == PROFILE


def test_flow_rejects_a_bad_number(tmp_path):
    from pathlib import Path

    code, _, path = run(tmp_path, ["FL"], ["9"], [Path("a.exe")])
    assert code == 2 and path.read_text(encoding="utf-8") == PROFILE


def test_flow_nothing_found_explains_the_manual_route(tmp_path):
    code, text, _ = run(tmp_path, ["Photoshop"], [], [])
    assert code == 1 and "candidates" in text


def test_flow_needs_a_name(tmp_path):
    code, text, _ = run(tmp_path, [], [], [])
    assert code == 2 and "Usage" in text
