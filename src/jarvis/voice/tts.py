"""Text-to-speech with Piper (CPU, ONNX)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.voice.interfaces import AudioClip, VoiceError

logger = logging.getLogger(__name__)


class PiperSynthesizer:
    def __init__(
        self,
        model_path: Path,
        length_scale: float | None = None,
        voice_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._model_path = model_path
        self._length_scale = length_scale
        self._voice_factory = voice_factory
        self._voice: Any = None

    def _load(self) -> Any:
        if self._voice is None:
            logger.info("Loading Piper voice %s", self._model_path.name)
            try:
                if self._voice_factory is not None:
                    self._voice = self._voice_factory()
                else:
                    from piper import PiperVoice

                    # The matching "<name>.onnx.json" config is found automatically.
                    self._voice = PiperVoice.load(self._model_path)
            except Exception as exc:
                raise VoiceError(f"Could not load the voice {self._model_path.name}: {exc}") from exc
        return self._voice

    def warm_up(self) -> None:
        self._load()

    def synthesize(self, text: str) -> Iterator[AudioClip]:
        voice = self._load()
        kwargs: dict[str, Any] = {}
        if self._length_scale is not None:
            from piper import SynthesisConfig

            kwargs["syn_config"] = SynthesisConfig(length_scale=self._length_scale)
        for chunk in voice.synthesize(text, **kwargs):  # one chunk per sentence
            yield AudioClip(
                samples=np.asarray(chunk.audio_float_array, dtype=np.float32),
                sample_rate=int(chunk.sample_rate),
            )
