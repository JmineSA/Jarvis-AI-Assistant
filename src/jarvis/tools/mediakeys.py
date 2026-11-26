"""Pause, skip and volume, by pressing the same media keys your keyboard has.

Only this fixed list of keys can ever be pressed. Nothing spoken is typed anywhere. The keys
reach whatever last played sound (VLC, or YouTube Music in Chrome).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import field_validator

from jarvis.tools.registry import ToolArgs, ToolResult

Action = Literal["play_pause", "next", "previous", "stop", "volume_up", "volume_down", "mute"]

# Windows virtual-key codes for the media keys.
KEYS: dict[str, int] = {
    "play_pause": 0xB3, "next": 0xB0, "previous": 0xB1, "stop": 0xB2,
    "volume_up": 0xAF, "volume_down": 0xAE, "mute": 0xAD,
}
VOLUME_PRESSES = 5   # each press is about 2%, so "volume up" is about 10%
_SAYS = {
    "play_pause": "Toggled play and pause.", "next": "Skipping to the next track.",
    "previous": "Going back a track.", "stop": "Stopped the music.",
    "volume_up": "Turning it up.", "volume_down": "Turning it down.", "mute": "Toggled mute.",
}
_SYNONYMS = {
    "pause": "play_pause", "resume": "play_pause", "play": "play_pause", "continue": "play_pause",
    "play pause": "play_pause", "skip": "next", "next track": "next", "next song": "next",
    "previous track": "previous", "previous song": "previous", "back": "previous", "last": "previous",
    "louder": "volume_up", "up": "volume_up", "volume up": "volume_up", "turn up": "volume_up",
    "quieter": "volume_down", "down": "volume_down", "volume down": "volume_down", "turn down": "volume_down",
    "unmute": "mute",
}


class KeySender(Protocol):
    def press(self, virtual_key: int) -> None: ...


class WindowsKeySender:
    def press(self, virtual_key: int) -> None:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.keybd_event(virtual_key, 0, 0, 0)       # key down
        user32.keybd_event(virtual_key, 0, 0x0002, 0)  # key up


class MediaControlArgs(ToolArgs):
    action: Action

    @field_validator("action", mode="before")
    @classmethod
    def _friendly(cls, value: object) -> object:
        if isinstance(value, str):
            key = " ".join(value.lower().replace("_", " ").replace("-", " ").split())
            return _SYNONYMS.get(key, key.replace(" ", "_"))
        return value


def make_media_control(
    sender: KeySender | None = None, platform: Callable[[], str] = lambda: sys.platform
) -> Callable[[MediaControlArgs], ToolResult]:
    def run(args: MediaControlArgs) -> ToolResult:
        if sender is None and platform() != "win32":
            return ToolResult.failed("Media controls only work on Windows.")
        keys = sender or WindowsKeySender()
        presses = VOLUME_PRESSES if args.action in ("volume_up", "volume_down") else 1
        try:
            for _ in range(presses):
                keys.press(KEYS[args.action])
        except OSError as exc:
            return ToolResult.failed(f"I couldn't press the media key: {exc.strerror or exc}.")
        return ToolResult.done(_SAYS[args.action])

    return run
