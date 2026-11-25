"""Microphone capture and push-to-talk triggers."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from jarvis.voice.audio_utils import resample
from jarvis.voice.interfaces import SAMPLE_RATE, VoiceError

logger = logging.getLogger(__name__)


def resolve_device(device: int | str | None) -> int | str | None:
    """'3' (from an environment variable) means device number 3."""
    if isinstance(device, str) and device.strip().isdigit():
        return int(device.strip())
    return device


# --------------------------------------------------------------------------- recorder


class SoundDeviceRecorder:
    """Records mono audio from the microphone while push-to-talk is held."""

    def __init__(self, device: int | str | None = None, max_seconds: float = 30.0) -> None:
        self._device = resolve_device(device)
        self._max_samples = int(max_seconds * SAMPLE_RATE)
        self._frames: list[np.ndarray] = []
        self._stream: Any = None
        self._rate = SAMPLE_RATE

    def _open(self, sd: Any, rate: int) -> None:
        def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            if status:
                logger.debug("Input stream status: %s", status)
            self._frames.append(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            device=self._device,
            callback=callback,
        )
        self._stream.start()
        self._rate = rate

    def start(self) -> None:
        import sounddevice as sd

        self._frames = []
        try:
            try:
                self._open(sd, SAMPLE_RATE)
            except sd.PortAudioError:
                # Some devices refuse 16 kHz. Use the device's own rate and convert afterwards.
                info = sd.query_devices(self._device, "input")
                rate = int(info["default_samplerate"])
                logger.info("Microphone does not support 16 kHz; recording at %d Hz.", rate)
                self._open(sd, rate)
        except Exception as exc:
            raise VoiceError(f"Cannot open the microphone: {exc}") from exc

    def stop(self) -> np.ndarray:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._frames).astype(np.float32)
        self._frames = []
        audio = resample(audio, self._rate, SAMPLE_RATE)
        return audio[: self._max_samples]


# --------------------------------------------------------------------------- hotkey logic

_MODIFIERS = ("ctrl", "shift", "alt", "cmd")
_ALIASES = {
    "right_ctrl": "ctrl_r", "rctrl": "ctrl_r", "left_ctrl": "ctrl_l", "lctrl": "ctrl_l",
    "control": "ctrl", "right_shift": "shift_r", "left_shift": "shift_l",
    "right_alt": "alt_r", "left_alt": "alt_l", "altgr": "alt_gr",
    "win": "cmd", "windows": "cmd", "super": "cmd",
    "esc": "esc", "return": "enter", "spacebar": "space",
}


def parse_hotkey(spec: str) -> frozenset[str]:
    """'ctrl+space' -> {'ctrl', 'space'}. Raises ValueError for an empty spec."""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError("push_to_talk_key must not be empty")
    return frozenset(_ALIASES.get(p, p) for p in parts)


def _tokens(key_name: str) -> set[str]:
    """Names a physical key satisfies: ctrl_r satisfies both 'ctrl_r' and 'ctrl'."""
    tokens = {key_name}
    for mod in _MODIFIERS:
        if key_name in (f"{mod}_l", f"{mod}_r"):
            tokens.add(mod)
    return tokens


class HotkeyTracker:
    """Turns key press/release events into 'start' and 'stop' of a held combination."""

    def __init__(self, spec: str) -> None:
        self.required = parse_hotkey(spec)
        self._down: set[str] = set()
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def press(self, name: str) -> str | None:
        self._down.add(name.lower())
        return self._update()

    def release(self, name: str) -> str | None:
        self._down.discard(name.lower())
        return self._update()

    def _update(self) -> str | None:
        satisfied = all(
            any(token in _tokens(key) for key in self._down) for token in self.required
        )
        if satisfied and not self._active:
            self._active = True
            return "start"
        if not satisfied and self._active:
            self._active = False
            return "stop"
        return None


def key_name(key: Any) -> str | None:
    """Name of a pynput key object, or None if it cannot be identified."""
    name = getattr(key, "name", None)
    if name:
        return str(name).lower()
    char = getattr(key, "char", None)
    if char:
        # With Ctrl held, pynput reports letters as control codes (Ctrl+J -> '\n').
        if len(char) == 1 and 1 <= ord(char) <= 26:
            return chr(ord(char) + 96)
        return char.lower()
    return None


# --------------------------------------------------------------------------- triggers


class HotkeyTrigger:
    """Global push-to-talk: hold the key combination anywhere in Windows."""

    def __init__(self, spec: str) -> None:
        self._spec = spec
        self._tracker = HotkeyTracker(spec)
        self._lock = threading.Lock()
        self._held = threading.Event()
        self._pressed = threading.Event()
        self._listener: Any = None

    @property
    def description(self) -> str:
        return self._spec

    def start(self) -> None:
        from pynput import keyboard

        def on_press(key: Any) -> None:
            name = key_name(key)
            if name is None:
                return
            with self._lock:
                if self._tracker.press(name) == "start":
                    self._held.set()
                    self._pressed.set()

        def on_release(key: Any) -> None:
            name = key_name(key)
            if name is None:
                return
            with self._lock:
                if self._tracker.release(name) == "stop":
                    self._held.clear()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()
        logger.info("Push-to-talk listening for: %s", self._spec)

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def wait_for_press(self, stop: threading.Event, timeout: float | None = None) -> bool | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not stop.is_set():
            pause = 0.1 if deadline is None else min(0.1, max(0.0, deadline - time.monotonic()))
            if self._pressed.wait(pause):
                self._pressed.clear()
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return None
        return False

    def wait_for_release(self, stop: threading.Event, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._held.is_set() and not stop.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                logger.warning("Push-to-talk held past the time limit; stopping the recording.")
                return
            time.sleep(0.02)

    def interrupt_requested(self) -> bool:
        return self._pressed.is_set()

    def clear(self) -> None:
        self._pressed.clear()


class EnterTrigger:
    """Fallback for testing without a global hotkey: press Enter to start and stop."""

    description = "Enter"

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def wait_for_press(self, stop: threading.Event, timeout: float | None = None) -> bool | None:
        # Blocks on the keyboard, so reminders are only spoken between turns in this test mode.
        try:
            input("\n[Press Enter, then speak] ")
        except EOFError:
            return False
        return not stop.is_set()

    def wait_for_release(self, stop: threading.Event, timeout: float | None = None) -> None:
        try:
            input("[Listening... press Enter when done] ")
        except EOFError:
            pass

    def interrupt_requested(self) -> bool:
        return False

    def clear(self) -> None:
        pass
