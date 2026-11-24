"""Speech-to-text with Faster-Whisper (CPU, int8)."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.voice.interfaces import VoiceError

logger = logging.getLogger(__name__)


def build_prompt(hint_words: list[str]) -> str | None:
    """Vocabulary hint that nudges Whisper towards names it would otherwise mishear."""
    return f"Vocabulary: {', '.join(hint_words)}." if hint_words else None


def is_prompt_echo(text: str, prompt: str | None) -> bool:
    """Whisper sometimes parrots its prompt on near-silence; treat that as 'nothing heard'."""
    if not prompt:
        return False
    cleaned = text.strip().lower().rstrip(".")
    return cleaned == prompt.strip().lower().rstrip(".") or cleaned.startswith("vocabulary:")


def fix_assistant_name(text: str, aliases: list[str]) -> str:
    """Turn a misheard name at the start of a sentence into JARVIS ("Hey Travis, ..." -> "Hey JARVIS, ...").

    Only the start of the sentence is touched, so a real "Travis" elsewhere stays as it is.
    """
    if not aliases:
        return text
    names = "|".join(re.escape(a) for a in aliases)
    pattern = rf"^(\W*(?:(?:hey|hi|hello|ok|okay)\b[\s,]*)?)(?:{names})\b"
    return re.sub(pattern, r"\1JARVIS", text, count=1, flags=re.IGNORECASE)


class FasterWhisperTranscriber:
    def __init__(
        self,
        model: str = "small.en",
        language: str | None = "en",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        download_root: Path | None = None,
        hint_words: list[str] | None = None,
        name_aliases: list[str] | None = None,
        model_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._name_aliases = name_aliases or []
        self._model_name = model
        self._language = language
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        self._download_root = download_root
        self._prompt = build_prompt(hint_words or [])
        self._model_factory = model_factory
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            logger.info("Loading Whisper model '%s' (first run downloads it)...", self._model_name)
            try:
                if self._model_factory is not None:
                    self._model = self._model_factory()
                else:
                    from faster_whisper import WhisperModel

                    self._model = WhisperModel(
                        self._model_name,
                        device="cpu",
                        compute_type=self._compute_type,
                        cpu_threads=self._cpu_threads,
                        download_root=str(self._download_root) if self._download_root else None,
                    )
            except Exception as exc:
                raise VoiceError(
                    f"Could not load the speech model '{self._model_name}': {exc}. "
                    "The first run needs an internet connection to download it."
                ) from exc
        return self._model

    def warm_up(self) -> None:
        self._load()

    def transcribe(self, audio: np.ndarray) -> str:
        model = self._load()
        segments, _info = model.transcribe(
            audio.astype(np.float32, copy=False),
            language=self._language,
            beam_size=1,                      # greedy decoding: fastest on a CPU
            vad_filter=True,                  # drop silence so Whisper does not invent words
            condition_on_previous_text=False,
            initial_prompt=self._prompt,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if is_prompt_echo(text, self._prompt):
            return ""
        return fix_assistant_name(text, self._name_aliases)
