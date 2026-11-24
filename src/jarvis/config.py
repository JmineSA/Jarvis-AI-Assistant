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
}


class ConfigError(Exception):
    """Raised when settings cannot be loaded or are invalid."""


class OllamaSettings(BaseModel):
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:3b-instruct"
    timeout_seconds: float = Field(120.0, gt=0)
    temperature: float = Field(0.4, ge=0, le=2)
    num_ctx: int = Field(4096, ge=512)


class BrainSettings(BaseModel):
    max_history_turns: int = Field(10, ge=1, le=100)


class ApiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)


class LoggingSettings(BaseModel):
    level: str = "INFO"
    directory: str = "logs"
    max_bytes: int = Field(1_048_576, ge=1024)
    backup_count: int = Field(3, ge=0)


class SttSettings(BaseModel):
    """Speech-to-text (Faster-Whisper, runs on the CPU)."""

    model: str = "small.en"
    language: str | None = "en"
    compute_type: str = "int8"
    cpu_threads: int = Field(4, ge=0)  # 0 = let the library decide
    # Words Whisper is nudged towards, so "Power BI" is not heard as "power be eye".
    hint_words: list[str] = Field(
        default_factory=lambda: [
            "JARVIS", "Power BI", "VS Code", "Jupyter", "Databricks", "Chrome", "Ollama",
        ]
    )


class TtsSettings(BaseModel):
    """Text-to-speech (Piper, runs on the CPU)."""

    voice: str = "en_GB-alan-medium"
    length_scale: float | None = Field(None, gt=0, le=3)  # < 1 faster, > 1 slower; None = voice default


class VoiceSettings(BaseModel):
    push_to_talk_key: str = "right_ctrl"
    input_device: int | str | None = None   # index or part of the device name
    output_device: int | str | None = None
    min_speech_seconds: float = Field(0.3, ge=0)
    max_record_seconds: float = Field(30.0, gt=1)
    silence_peak: float = Field(0.01, ge=0, le=1)  # quieter than this counts as "no sound"
    models_dir: str = "data/models"
    stt: SttSettings = Field(default_factory=SttSettings)
    tts: TtsSettings = Field(default_factory=TtsSettings)


class Settings(BaseModel):
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    brain: BrainSettings = Field(default_factory=BrainSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)

    def resolve_log_dir(self) -> Path:
        """Log directory as an absolute path (relative paths are project-relative)."""
        path = Path(self.logging.directory)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def resolve_models_dir(self) -> Path:
        """Where downloaded voice and speech models are kept."""
        path = Path(self.voice.models_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path


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
