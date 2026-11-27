"""Typed application settings.

Settings come from config/settings.yaml, then environment variables (optionally
loaded from a git-ignored .env file) override individual values. Secrets never
belong in settings.yaml.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# environment variable -> path of keys inside the settings mapping
ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "JARVIS_OLLAMA_HOST": ("ollama", "host"),
    "JARVIS_OLLAMA_MODEL": ("ollama", "model"),
    "JARVIS_OLLAMA_TIMEOUT": ("ollama", "timeout_seconds"),
    "JARVIS_API_HOST": ("api", "host"),
    "JARVIS_API_PORT": ("api", "port"),
    "JARVIS_LOG_LEVEL": ("logging", "level"),
    "JARVIS_LOG_DIR": ("logging", "directory"),
    "JARVIS_PTT_KEY": ("voice", "push_to_talk_key"),
    "JARVIS_INPUT_DEVICE": ("voice", "input_device"),
    "JARVIS_OUTPUT_DEVICE": ("voice", "output_device"),
    "JARVIS_STT_MODEL": ("voice", "stt", "model"),
    "JARVIS_TTS_VOICE": ("voice", "tts", "voice"),
    "JARVIS_PROFILE": ("tools", "profile_path"),
    "JARVIS_DB": ("memory", "database_path"),
}


class ConfigError(Exception):
    """Raised when settings cannot be loaded or are invalid."""


class OllamaSettings(BaseModel):
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:3b-instruct"
    timeout_seconds: float = Field(120.0, gt=0)
    temperature: float = Field(0.4, ge=0, le=2)
    num_ctx: int = Field(4096, ge=512)
    keep_alive: str = "30m"            # how long Ollama keeps the model loaded after a reply
    num_predict: int = Field(256, ge=16)  # longest reply in tokens: bounds the GPU work per turn


class BrainSettings(BaseModel):
    max_history_turns: int = Field(10, ge=1, le=100)
    fast_commands: bool = True   # answer obvious commands (open, play, weather...) without the model
    tool_subset: bool = True     # send the model only the tools a sentence could need (shorter prompt = faster)


class ApiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)


class LoggingSettings(BaseModel):
    level: str = "INFO"
    directory: str = "logs"
    max_bytes: int = Field(1_048_576, ge=1024)
    backup_count: int = Field(3, ge=0)
    record_turns: bool = True               # keep a local list of turns and timings (see `route-report`)
    turns_file: str = "data/turns.jsonl"


class SttSettings(BaseModel):
    """Speech-to-text (Faster-Whisper, runs on the CPU)."""

    model: str = "small.en"
    language: str | None = "en"
    compute_type: str = "int8"
    cpu_threads: int = Field(4, ge=0)  # 0 = let the library decide
    beam_size: int = Field(1, ge=1, le=8)  # 1 = fastest; 3 is a little more accurate and ~0.5 s slower
    auto_gain: bool = True  # raise quiet recordings so you can speak calmly
    correct_words: bool = True  # fix near-misses of your app/folder names ("power be eye" -> Power BI)
    # Words Whisper is nudged towards, so "Power BI" is not heard as "power be eye".
    hint_words: list[str] = Field(
        default_factory=lambda: [
            "JARVIS", "Power BI", "VS Code", "Jupyter", "Databricks", "Chrome", "Ollama",
        ]
    )
    # Whisper often mishears the assistant's name at the start of a sentence.
    name_aliases: list[str] = Field(
        default_factory=lambda: [
            "Travis", "Jervis", "Jarvas", "Javis", "Jarvus", "Jarvice", "Gervais",
        ]
    )


class TtsSettings(BaseModel):
    """Text-to-speech (Piper, runs on the CPU)."""

    voice: str = "en_GB-alan-medium"
    length_scale: float | None = Field(None, gt=0, le=3)  # < 1 faster, > 1 slower; None = voice default


class VoiceSettings(BaseModel):
    push_to_talk_key: str = "right_ctrl"
    wake_word: bool = False                 # say "Hey JARVIS" too (run `python -m jarvis wake-setup` first)
    wake_threshold: float = Field(0.5, gt=0, le=1)   # lower = hears you more easily, but more false alarms
    wake_silence_seconds: float = Field(1.2, gt=0.2, le=5)  # quiet for this long = you have finished
    ack_sound: bool = True                  # a soft tick when you stop talking, so you know it heard you
    wake_reply: str = "I'm listening."      # said after "Hey JARVIS" (empty = just a chime)
    wake_listen_seconds: float = Field(10.0, ge=2.0, le=30.0)   # how long it waits for you to start talking
    wake_follow_up: bool = True             # after a question from JARVIS ("What would you like to play?"), listen again
    wake_chime: bool = True                 # a short beep when it hears the wake word
    input_device: int | str | None = None   # index or part of the device name
    output_device: int | str | None = None
    min_speech_seconds: float = Field(0.3, ge=0)
    max_record_seconds: float = Field(30.0, gt=1)
    silence_peak: float = Field(0.004, ge=0, le=1)  # quieter than this counts as "no sound"
    models_dir: str = "data/models"
    stt: SttSettings = Field(default_factory=SttSettings)
    tts: TtsSettings = Field(default_factory=TtsSettings)


class ToolsSettings(BaseModel):
    enabled: bool = True
    profile_path: str = "config/profile.yaml"
    # Tools that must always ask you first, by name (e.g. [open_app]).
    always_confirm: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(10.0, gt=0)


class MemorySettings(BaseModel):
    enabled: bool = True
    database_path: str = "data/jarvis.db"
    save_conversations: bool = True       # false = remember facts and reminders only
    max_facts_in_prompt: int = Field(20, ge=0, le=100)
    reminder_check_seconds: float = Field(5.0, gt=0)


class GmailSettings(BaseModel):
    """Read-only Gmail ("check my email"). Run `python -m jarvis gmail-setup` once."""

    enabled: bool = False
    client_secret_file: str = "data/gmail/client_secret.json"   # the JSON you download from Google
    token_file: str = "data/gmail/token.json"                   # your saved sign-in (never leaves this PC)


class ScreenSettings(BaseModel):
    """"What's on my screen?": reads the window you are using, only when you ask. Nothing is stored."""

    enabled: bool = True
    max_chars: int = Field(5000, ge=500, le=12000)       # how much screen text the model gets (it has a small memory)
    screenshot_dir: str = "data/screenshots"            # where "take a screenshot" saves pictures


class UiSettings(BaseModel):
    greeting_name: str = Field("boss", max_length=40)   # what the dashboard calls you
    greet_on_start: bool = True  # say hello when the dashboard opens
    briefing_max: int = Field(2, ge=0, le=5)  # reminders read out at start (0 = none); the rest are counted
    speak_typed_replies: bool = True   # read replies aloud even when you typed the command
    weather_refresh_minutes: int = Field(15, ge=1, le=240)
    health_check_seconds: float = Field(10.0, gt=0)


class Settings(BaseModel):
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    brain: BrainSettings = Field(default_factory=BrainSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    tools: ToolsSettings = Field(default_factory=ToolsSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    ui: UiSettings = Field(default_factory=UiSettings)
    gmail: GmailSettings = Field(default_factory=GmailSettings)
    screen: ScreenSettings = Field(default_factory=ScreenSettings)

    def resolve_database_path(self) -> Path:
        """Memory database as an absolute path (relative paths are project-relative)."""
        path = Path(self.memory.database_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def resolve_profile_path(self) -> Path:
        """Profile file as an absolute path (relative paths are project-relative)."""
        path = Path(self.tools.profile_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def resolve_log_dir(self) -> Path:
        """Log directory as an absolute path (relative paths are project-relative)."""
        path = Path(self.logging.directory)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def resolve_turns_path(self) -> Path:
        path = Path(self.logging.turns_file)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def resolve_screenshot_dir(self) -> Path:
        path = Path(self.screen.screenshot_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def resolve_gmail_paths(self) -> tuple[Path, Path]:
        """(client file, saved sign-in) as absolute paths."""
        paths = []
        for text in (self.gmail.client_secret_file, self.gmail.token_file):
            path = Path(text)
            paths.append(path if path.is_absolute() else PROJECT_ROOT / path)
        return paths[0], paths[1]

    def resolve_models_dir(self) -> Path:
        """Where downloaded voice and speech models are kept."""
        path = Path(self.voice.models_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_config_path(env: Mapping[str, str] | None = None) -> Path:
    """The settings.yaml JARVIS is using (JARVIS_CONFIG if set, else config/settings.yaml)."""
    explicit = (env if env is not None else os.environ).get("JARVIS_CONFIG")
    return Path(explicit) if explicit else DEFAULT_CONFIG_PATH


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read settings file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Settings file {path} must contain a YAML mapping.")
    return data


def load_settings(
    config_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Load settings from YAML, then apply environment overrides.

    Pass `env` explicitly (for example in tests) to avoid reading the real
    environment or the .env file.
    """
    if env is None:
        load_dotenv(PROJECT_ROOT / ".env")
        env = os.environ

    explicit = config_path or env.get("JARVIS_CONFIG")
    path = Path(explicit) if explicit else DEFAULT_CONFIG_PATH

    if path.exists():
        data = _read_yaml(path)
    elif explicit:
        raise ConfigError(f"Settings file not found: {path}")
    else:
        data = {}

    for var, key_path in ENV_OVERRIDES.items():
        value = env.get(var)
        if value in (None, ""):
            continue
        node: dict[str, Any] = data
        for key in key_path[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ConfigError(f"Settings section '{key}' must be a mapping.")
        node[key_path[-1]] = value

    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid settings: {exc}") from exc
