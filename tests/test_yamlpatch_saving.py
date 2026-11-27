import httpx
import pytest
import yaml

from jarvis.geocode import Place, find_place, place_names
from jarvis.saving import save_audio_device, save_greeting_name, save_home_address
from jarvis.yamlpatch import patch_text, set_value

SETTINGS = """# top comment
voice:
  input_device: 2              # the mic
  output_device: null
  stt:
    model: small.en

logging:
  level: INFO
"""


def test_changes_one_value_and_keeps_comments_and_layout():
    out = patch_text(SETTINGS, "voice", "input_device", 3)
    assert "# top comment" in out and "# the mic" in out
    assert yaml.safe_load(out)["voice"]["input_device"] == 3
    assert yaml.safe_load(out)["voice"]["stt"]["model"] == "small.en"   # nested block untouched
    assert yaml.safe_load(out)["logging"]["level"] == "INFO"


def test_adds_a_missing_key_to_an_existing_section():
    out = patch_text(SETTINGS, "voice", "extra", "x")
    assert yaml.safe_load(out)["voice"]["extra"] == "x"


def test_adds_a_missing_section():
    out = patch_text(SETTINGS, "ui", "greeting_name", "Jimmy")
    assert yaml.safe_load(out)["ui"]["greeting_name"] == "Jimmy"


def test_a_nested_key_with_the_same_name_is_not_mistaken_for_the_section_key():
    text = "voice:\n  stt:\n    input_device: 9\n"
    out = patch_text(text, "voice", "input_device", 1)
    data = yaml.safe_load(out)
    assert data["voice"]["input_device"] == 1 and data["voice"]["stt"]["input_device"] == 9


@pytest.mark.parametrize("value", ["It's 'quoted': yes # no", "a: b", "6365 Nellmapius Drive, Pretoria", "", None, 0, -25.7479])
def test_awkward_values_survive_a_round_trip(tmp_path, value):
    path = tmp_path / "s.yaml"
    path.write_text(SETTINGS, encoding="utf-8")
    set_value(path, "ui", "thing", value)
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["ui"]["thing"] == value


def test_a_backup_is_made_and_the_file_is_replaced_atomically(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(SETTINGS, encoding="utf-8")
    set_value(path, "voice", "input_device", 5)
    assert (tmp_path / "s.yaml.bak").read_text(encoding="utf-8") == SETTINGS
    assert not (tmp_path / "s.yaml.tmp").exists()


def test_a_section_written_in_flow_style_is_refused_not_mangled(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("voice: {input_device: 2}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        set_value(path, "voice", "input_device", 3)
    assert path.read_text(encoding="utf-8") == "voice: {input_device: 2}\n"


def test_creates_the_file_if_missing(tmp_path):
    path = tmp_path / "new.yaml"
    set_value(path, "ui", "greeting_name", "boss")
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"ui": {"greeting_name": "boss"}}


# ------------------------------------------------------------------ saving


def test_greeting_name_is_cleaned_and_saved(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(SETTINGS, encoding="utf-8")
    assert save_greeting_name(path, "  Jimmy   Boss  ") == "Jimmy Boss"
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["ui"]["greeting_name"] == "Jimmy Boss"
    with pytest.raises(ValueError):
        save_greeting_name(path, "   ")
    assert len(save_greeting_name(path, "x" * 200)) == 40


def test_audio_devices_are_saved_and_default_is_null(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(SETTINGS, encoding="utf-8")
    save_audio_device(path, "input", 4)
    save_audio_device(path, "output", None)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))["voice"]
    assert data["input_device"] == 4 and data["output_device"] is None
    with pytest.raises(ValueError):
        save_audio_device(path, "bogus", 1)


PROFILE = """owner_name: Lesiba
location:
  name: Pretoria
  latitude: -25.7479
  longitude: 28.2293
apps: {}
"""


def test_address_is_saved_and_the_weather_moves_to_the_suburb(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(PROFILE, encoding="utf-8")
    message, _place = save_home_address(path, "6365 Nellmapius Drive, Pretoria", lambda a: Place("Nellmapius, Gauteng", -25.7, 28.4))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))["location"]
    assert data["address"] == "6365 Nellmapius Drive, Pretoria" and data["name"] == "Nellmapius"
    assert data["latitude"] == -25.7 and "Nellmapius, Gauteng" in message


def test_address_is_still_saved_when_the_suburb_is_not_found(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(PROFILE, encoding="utf-8")
    message, place = save_home_address(path, "1 Nowhere Road", lambda a: None)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))["location"]
    assert place is None and data["address"] == "1 Nowhere Road" and data["name"] == "Pretoria"
    assert "unchanged" in message


def test_address_is_still_saved_when_offline(tmp_path):
    def offline(_a):
        raise httpx.ConnectError("no network")

    path = tmp_path / "p.yaml"
    path.write_text(PROFILE, encoding="utf-8")
    message, _ = save_home_address(path, "1 Nowhere Road", offline)
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["location"]["address"] == "1 Nowhere Road"
    assert "online" in message


def test_clearing_the_address(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(PROFILE + "", encoding="utf-8")
    save_home_address(path, "1 Road", lambda a: None)
    message, _ = save_home_address(path, "  ", lambda a: None)
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["location"]["address"] == ""
    assert "cleared" in message


def test_no_location_and_no_lookup_is_refused_with_a_reason(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text("owner_name: X\napps: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no location"):
        save_home_address(path, "1 Road", lambda a: None)


# ------------------------------------------------------------------ geocode


def test_only_place_names_are_ever_looked_up_never_street_lines():
    assert place_names("6365 Nellmapius Drive, Nellmapius, Pretoria") == ["Nellmapius", "Pretoria"]
    assert place_names("12 Main Street") == []
    assert place_names("6365 Nelmefias") == ["Nelmefias"]


def test_find_place_picks_a_south_african_match():
    body = {"results": [
        {"name": "Nellmapius", "country_code": "US", "latitude": 1.0, "longitude": 2.0},
        {"name": "Nellmapius", "admin1": "Gauteng", "country_code": "ZA", "latitude": -25.7, "longitude": 28.4},
    ]}
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, json=body)

    place = find_place("6365 Nellmapius Drive, Nellmapius", transport=httpx.MockTransport(handler))
    assert place == Place("Nellmapius, Gauteng", -25.7, 28.4)
    assert seen[0]["name"] == "Nellmapius"   # the street line was never sent


def test_find_place_returns_none_when_nothing_matches_and_raises_when_offline():
    empty = httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    assert find_place("Nowhere", transport=empty) is None

    def down(request):
        raise httpx.ConnectError("no")

    with pytest.raises(httpx.HTTPError):
        find_place("Nowhere", transport=httpx.MockTransport(down))
