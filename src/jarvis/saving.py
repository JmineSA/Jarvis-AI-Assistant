"""What the Settings page does when you press Save. Kept out of the UI so it can be tested."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import httpx

from jarvis.geocode import Place, find_place
from jarvis.profile import load_profile
from jarvis.yamlpatch import set_value

logger = logging.getLogger(__name__)

MAX_NAME = 40


def save_greeting_name(config_path: Path, name: str) -> str:
    """Save what the dashboard calls you. Returns the cleaned name."""
    cleaned = " ".join(name.split())[:MAX_NAME]
    if not cleaned:
        raise ValueError("Type a name first.")
    set_value(config_path, "ui", "greeting_name", cleaned)
    return cleaned


def save_audio_device(config_path: Path, kind: str, device: int | str | None) -> None:
    """kind is "input" or "output". None means the system default."""
    if kind not in ("input", "output"):
        raise ValueError("kind must be input or output")
    set_value(config_path, "voice", f"{kind}_device", device)


def save_home_address(
    profile_path: Path,
    address: str,
    find: Callable[[str], Place | None] = find_place,
) -> tuple[str, Place | None]:
    """Save the address, and (best effort) move the weather location to its suburb.

    Returns a plain message for the user and the place found, if any. Raises ValueError/OSError
    if the file could not be changed; the file is then left as it was.
    """
    cleaned = " ".join(address.replace("\n", " ").split())[:200]
    current = load_profile(profile_path).location

    place: Place | None = None
    note = ""
    if cleaned:
        try:
            place = find(cleaned)
        except (httpx.HTTPError, ValueError):
            note = " I couldn't look up the suburb for the weather (are you online?), so that is unchanged."
        else:
            if place is None:
                note = " I couldn't find that suburb for the weather, so the weather location is unchanged."

    if current is None and place is None:
        raise ValueError(
            "There is no location in your profile yet, and I couldn't look this one up. "
            "Add a location (name, latitude, longitude) to profile.yaml first."
        )

    if place is not None:
        set_value(profile_path, "location", "name", place.name.split(",")[0])
        set_value(profile_path, "location", "latitude", round(place.latitude, 4))
        set_value(profile_path, "location", "longitude", round(place.longitude, 4))
        note = f" The weather location is now {place.name}."
    set_value(profile_path, "location", "address", cleaned)
    if not cleaned:
        return "Home address cleared. Maps will use your weather location instead.", place
    return "Home address saved." + note, place
