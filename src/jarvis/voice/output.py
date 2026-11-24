"""Audio playback."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np

from jarvis.voice.input import resolve_device
from jarvis.voice.interfaces import AudioClip, VoiceError

logger = logging.getLogger(__name__)

_TAIL_PAD_SECONDS = 0.08  # some sound cards clip the last few milliseconds


class SoundDevicePlayer:
    def __init__(self, device: int | str | None = None) -> None:
        self._device = resolve_device(device)

    def play(self, clip: AudioClip, should_stop: Callable[[], bool]) -> bool:
        import sounddevice as sd

        pad = np.zeros(int(clip.sample_rate * _TAIL_PAD_SECONDS), dtype=np.float32)
        samples = np.concatenate([clip.samples.astype(np.float32), pad])
        try:
            sd.play(samples, clip.sample_rate, device=self._device)
            stream = sd.get_stream()
            while stream.active:
                if should_stop():
                    sd.stop()
                    return False
                time.sleep(0.02)
        except Exception as exc:
            sd.stop()
            raise VoiceError(f"Cannot play audio: {exc}") from exc
        return True
