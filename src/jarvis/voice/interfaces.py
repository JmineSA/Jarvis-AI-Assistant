"""Small interfaces so voice parts can be swapped (or faked in tests)."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16_000  # what Whisper expects: 16 kHz, mono, float32 in [-1, 1]


class VoiceError(Exception):
    """A voice component failed in a way the user should be told about."""


@dataclass(frozen=True)
class AudioClip:
    samples: np.ndarray  # mono float32 in [-1, 1]
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.samples.size / self.sample_rate if self.sample_rate else 0.0


class Recorder(Protocol):
    def start(self) -> None:
        """Begin capturing from the microphone."""

    def stop(self) -> np.ndarray:
        """Stop capturing; return mono float32 audio at SAMPLE_RATE."""


class Transcriber(Protocol):
    def warm_up(self) -> None: ...

    def transcribe(self, audio: np.ndarray) -> str:
        """Return the recognised text ('' if nothing was understood)."""


class Synthesizer(Protocol):
    def warm_up(self) -> None: ...

    def synthesize(self, text: str) -> Iterator[AudioClip]:
        """Yield audio roughly one sentence at a time."""


class Player(Protocol):
    def play(self, clip: AudioClip, should_stop: Callable[[], bool]) -> bool:
        """Play a clip. Return False if playback was cut short by should_stop()."""


class Trigger(Protocol):
    """Decides when the user is talking (push-to-talk)."""

    def start(self) -> None: ...

    def close(self) -> None: ...

    def wait_for_press(self, stop: threading.Event) -> bool:
        """Block until the user starts talking. Return False if `stop` was set."""

    def wait_for_release(self, stop: threading.Event, timeout: float | None = None) -> None:
        """Block until the user stops talking, `stop` is set, or `timeout` passes."""

    def interrupt_requested(self) -> bool:
        """True if the user pressed talk again (used to cut off speech)."""

    def clear(self) -> None: ...
