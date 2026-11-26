import pytest

from jarvis.config import DEFAULT_CONFIG_PATH, ConfigError, Settings, load_settings


def test_defaults_are_valid():
    settings = Settings()
    assert settings.ollama.model == "qwen2.5:3b-instruct"
    assert settings.api.host == "127.0.0.1"


def test_shipped_settings_file_loads():
    assert DEFAULT_CONFIG_PATH.exists()
    settings = load_settings(env={})
    assert settings.brain.max_history_turns >= 1


def test_yaml_values_are_used(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("ollama:\n  model: llama3.2:3b\napi:\n  port: 9000\n", encoding="utf-8")
    settings = load_settings(path, env={})
    assert settings.ollama.model == "llama3.2:3b"
    assert settings.api.port == 9000


def test_env_overrides_yaml(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("api:\n  port: 9000\n", encoding="utf-8")
    env = {"JARVIS_API_PORT": "9100", "JARVIS_OLLAMA_MODEL": "phi3"}
    settings = load_settings(path, env=env)
    assert settings.api.port == 9100
    assert settings.ollama.model == "phi3"


def test_invalid_value_raises(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("api:\n  port: 99999\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(path, env={})


def test_missing_explicit_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "nope.yaml", env={})


def test_malformed_yaml_raises(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("ollama: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(path, env={})


def test_voice_defaults():
    voice = Settings().voice
    assert voice.push_to_talk_key == "right_ctrl"
    assert voice.stt.model == "small.en"
    assert voice.tts.voice == "en_GB-alan-medium"
    assert voice.tts.length_scale is None
    assert "Power BI" in voice.stt.hint_words


def test_voice_env_overrides_reach_nested_keys():
    env = {
        "JARVIS_PTT_KEY": "f8",
        "JARVIS_STT_MODEL": "base.en",
        "JARVIS_TTS_VOICE": "en_US-lessac-medium",
        "JARVIS_INPUT_DEVICE": "3",
    }
    voice = load_settings(env=env).voice
    assert voice.push_to_talk_key == "f8"
    assert voice.stt.model == "base.en"
    assert voice.tts.voice == "en_US-lessac-medium"
    assert str(voice.input_device) == "3"


def test_voice_section_in_yaml(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("voice:\n  input_device: 2\n  stt:\n    cpu_threads: 6\n", encoding="utf-8")
    voice = load_settings(path, env={}).voice
    assert voice.input_device == 2
    assert voice.stt.cpu_threads == 6
    assert voice.stt.model == "small.en"  # untouched defaults survive


def test_invalid_voice_value_raises(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("voice:\n  tts:\n    length_scale: -1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(path, env={})


def test_models_dir_is_project_relative_by_default():
    assert Settings().resolve_models_dir().is_absolute()


def test_relative_log_dir_is_project_relative():
    settings = Settings()
    assert settings.resolve_log_dir().is_absolute()


def test_tools_defaults():
    tools = Settings().tools
    assert tools.enabled is True
    assert tools.always_confirm == []
    assert tools.profile_path == "config/profile.yaml"
    assert Settings().resolve_profile_path().is_absolute()


def test_profile_path_env_override():
    settings = load_settings(env={"JARVIS_PROFILE": "/tmp/other.yaml"})
    assert str(settings.resolve_profile_path()) == "/tmp/other.yaml"


def test_tools_section_in_yaml(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("tools:\n  enabled: false\n  always_confirm: [open_app]\n", encoding="utf-8")
    tools = load_settings(path, env={}).tools
    assert tools.enabled is False and tools.always_confirm == ["open_app"]


def test_shipped_settings_target_the_focusrite_mic_and_name_fix():
    settings = load_settings(env={})
    assert settings.voice.input_device == 2
    assert "Travis" in settings.voice.stt.name_aliases


def test_memory_defaults():
    memory = Settings().memory
    assert memory.enabled is True and memory.save_conversations is True
    assert memory.database_path == "data/jarvis.db"
    assert Settings().resolve_database_path().is_absolute()


def test_database_path_env_override():
    settings = load_settings(env={"JARVIS_DB": "/tmp/elsewhere.db"})
    assert str(settings.resolve_database_path()) == "/tmp/elsewhere.db"
