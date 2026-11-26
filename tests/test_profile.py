import pytest

from jarvis.config import PROJECT_ROOT, ConfigError
from jarvis.profile import Profile, expand_path, load_profile, normalize


@pytest.fixture
def profile() -> Profile:
    return Profile.model_validate(
        {
            "apps": {
                "vscode": {"label": "VS Code", "names": ["visual studio code", "code"]},
                "powerbi": {"label": "Power BI", "names": ["power bi desktop"]},
            },
            "folders": {
                "lightson": {"label": "LightsOn", "names": ["solar project"], "path": "/x/lightson"},
                "j_mine": {"label": "Dune", "path": "/x/dune"},
            },
            "websites": {"github": {"label": "GitHub", "url": "https://github.com/x"}},
        }
    )


def test_normalize_ignores_case_spacing_and_filler():
    assert normalize("Power BI") == normalize("power-bi") == normalize("the PowerBI app") == "powerbi"
    assert normalize("my LightsOn folder") == "lightson"


def test_normalize_keeps_text_made_only_of_filler_words():
    assert normalize("the app") == "theapp"


def test_resolve_by_key_label_and_alias(profile):
    assert profile.resolve("apps", "vscode") == "vscode"
    assert profile.resolve("apps", "VS Code") == "vscode"
    assert profile.resolve("apps", "Visual Studio Code") == "vscode"
    assert profile.resolve("folders", "solar project") == "lightson"


def test_resolve_ignores_filler_words(profile):
    assert profile.resolve("folders", "the LightsOn folder") == "lightson"
    assert profile.resolve("apps", "the Power BI app") == "powerbi"


def test_resolve_tolerates_small_speech_errors(profile):
    assert profile.resolve("apps", "visual studio cod") == "vscode"
    assert profile.resolve("folders", "lights on") == "lightson"
    assert profile.resolve("folders", "j mine") == "j_mine"


def test_resolve_unknown_or_empty_is_none(profile):
    assert profile.resolve("apps", "firefox") is None
    assert profile.resolve("apps", "") is None
    assert profile.resolve("apps", "   ") is None


def test_labels_and_spoken_names(profile):
    assert profile.labels("apps") == ["VS Code", "Power BI"]
    assert profile.spoken_names() == ["VS Code", "Power BI", "LightsOn", "Dune", "GitHub"]


def test_expand_path_expands_environment_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_TEST_DIR", str(tmp_path))
    assert expand_path("$JARVIS_TEST_DIR/sub") == tmp_path / "sub"


def test_missing_profile_file_gives_empty_profile(tmp_path):
    profile = load_profile(tmp_path / "nope.yaml")
    assert profile.apps == {} and profile.folders == {} and profile.websites == {}


def test_malformed_yaml_raises(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text("apps: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_profile(path)


def test_profile_must_be_a_mapping(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_profile(path)


def test_folder_without_path_is_rejected(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text("folders:\n  x:\n    label: X\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_profile(path)


def test_shipped_profile_is_valid_and_complete():
    profile = load_profile(PROJECT_ROOT / "config" / "profile.yaml")
    assert profile.owner_name == "Lesiba"
    for app in ("vscode", "chrome", "powerbi", "jupyter"):
        assert app in profile.apps and profile.apps[app].candidates
    assert profile.resolve("apps", "Power BI") == "powerbi"
    assert profile.resolve("folders", "LightsOn") == "lightson"
    assert profile.resolve("websites", "Databricks") == "databricks"
    assert profile.apps["vscode"].folder_mode == "arg"
    assert profile.apps["jupyter"].folder_mode == "cwd"
