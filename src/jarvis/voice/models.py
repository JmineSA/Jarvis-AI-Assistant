"""Locating and downloading speech models.

Models live under data/models (git-ignored). They are downloaded on first use.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from jarvis.voice.interfaces import VoiceError

logger = logging.getLogger(__name__)

Downloader = Callable[[str, Path], None]


def piper_dir(models_dir: Path) -> Path:
    return models_dir / "piper"


def whisper_dir(models_dir: Path) -> Path:
    return models_dir / "whisper"


def piper_voice_path(models_dir: Path, voice: str) -> Path:
    return piper_dir(models_dir) / f"{voice}.onnx"


def _default_piper_downloader(voice: str, target: Path) -> None:
    from piper.download_voices import download_voice

    download_voice(voice, target)


def ensure_piper_voice(
    models_dir: Path, voice: str, downloader: Downloader | None = None
) -> Path:
    """Return the path of the voice model, downloading it first if needed."""
    target = piper_dir(models_dir)
    onnx = piper_voice_path(models_dir, voice)
    config = onnx.with_name(onnx.name + ".json")
    if onnx.exists() and config.exists():
        return onnx

    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading voice '{voice}' (about 60 MB, first run only)...", flush=True)
    try:
        (downloader or _default_piper_downloader)(voice, target)
    except BaseException as exc:
        # Never leave half-downloaded files behind: they would be loaded as corrupt.
        for path in (onnx, config):
            path.unlink(missing_ok=True)
        if isinstance(exc, Exception):
            raise VoiceError(
                f"Could not download the voice '{voice}': {exc}. "
                "Check your internet connection and the voice name."
            ) from exc
        raise
    if not (onnx.exists() and config.exists()):
        raise VoiceError(f"The voice '{voice}' was not downloaded correctly.")
    return onnx


def ensure_whisper_model(models_dir: Path, name: str) -> None:
    """Download the Whisper model into data/models/whisper if it is not there yet."""
    root = whisper_dir(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    print(f"Checking speech model '{name}' (first run downloads about 250 MB)...", flush=True)
    try:
        from faster_whisper.utils import download_model

        download_model(name, cache_dir=str(root))
    except Exception as exc:
        raise VoiceError(
            f"Could not download the speech model '{name}': {exc}. "
            "Check your internet connection."
        ) from exc
