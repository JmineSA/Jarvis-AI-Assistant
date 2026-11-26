"""Choosing a microphone and speakers: list them, hear which mic has sound, play a test tone."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

LISTEN_SECONDS = 0.7        # per microphone while detecting
SOUND_THRESHOLD = 0.02      # quieter than this counts as silence
WINNER_MARGIN = 1.5         # the loudest mic must be this much louder than the next


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str

    @property
    def label(self) -> str:
        return f"{self.index} · {self.name}"


def list_devices(sd: Any | None = None) -> tuple[list[AudioDevice], list[AudioDevice]]:
    """Inputs and outputs of the default audio system. Empty lists if audio isn't available."""
    try:
        if sd is None:
            import sounddevice as sd
        api = sd.default.hostapi
        inputs: list[AudioDevice] = []
        outputs: list[AudioDevice] = []
        for index, device in enumerate(sd.query_devices()):
            if device["hostapi"] != api:
                continue
            if device["max_input_channels"] > 0:
                inputs.append(AudioDevice(index, str(device["name"])))
            if device["max_output_channels"] > 0:
                outputs.append(AudioDevice(index, str(device["name"])))
        return inputs, outputs
    except Exception:
        logger.info("Could not list audio devices", exc_info=True)
        return [], []


def measure_levels(inputs: list[AudioDevice], sd: Any | None = None, seconds: float = LISTEN_SECONDS) -> dict[int, float]:
    """How loud each microphone is right now (0 to 1). A microphone that can't be opened scores 0."""
    if sd is None:
        import sounddevice as sd
    levels: dict[int, float] = {}
    for device in inputs:
        try:
            rate = int(sd.query_devices(device.index)["default_samplerate"])
            recording = sd.rec(int(rate * seconds), samplerate=rate, channels=1, device=device.index, dtype="float32")
            sd.wait()
            levels[device.index] = float(np.abs(recording).max()) if len(recording) else 0.0
        except Exception:
            logger.info("Could not listen on device %s", device.index, exc_info=True)
            levels[device.index] = 0.0
    return levels


def loudest(levels: dict[int, float]) -> int | None:
    """The device that clearly has sound on it, or None if all are quiet or it isn't clear."""
    ranked = sorted(levels.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] < SOUND_THRESHOLD:
        return None
    if len(ranked) > 1 and ranked[0][1] < ranked[1][1] * WINNER_MARGIN:
        return None
    return ranked[0][0]


def play_test_tone(device: int | str | None, sd: Any | None = None, seconds: float = 0.6) -> None:
    """A short, gentle tone on the chosen speakers."""
    if sd is None:
        import sounddevice as sd
    rate = 22050
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    fade = np.minimum(1.0, np.minimum(t, seconds - t) / 0.05)
    tone = (0.25 * np.sin(2 * np.pi * 440 * t) * fade).astype(np.float32)
    sd.play(tone, rate, device=device)
    sd.wait()
    time.sleep(0.05)
