"""Google Maps: find places near your home, or get directions from it.

JARVIS only builds a Maps address and opens it in your browser. Your home address stays in
config/profile.yaml and is sent to Google only when a Maps page opens (that is what Maps needs).
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from typing import Literal
from urllib.parse import quote, quote_plus

from pydantic import Field

from jarvis.profile import Profile
from jarvis.tools.launcher import Launcher
from jarvis.tools.registry import ToolArgs, ToolResult
from jarvis.tools.web import open_in_browser

MAX_QUERY = 200


class MapsArgs(ToolArgs):
    query: str = Field(min_length=1, max_length=400)
    mode: Literal["search", "directions"] = "search"


def _home(profile: Profile) -> tuple[str, str]:
    """(text Maps can use for your home, short coordinates form or ''). Empty when not set."""
    location = profile.location
    if location is None:
        return "", ""
    coords = f"{location.latitude:.5f},{location.longitude:.5f}"
    return (location.address.strip() or coords), coords


def search_url(query: str, home: str, coords: str) -> str:
    if home and home != coords:
        return "https://www.google.com/maps/search/?api=1&query=" + quote_plus(f"{query} near {home}")
    if coords:
        return f"https://www.google.com/maps/search/{quote(query, safe='')}/@{coords},14z"
    return "https://www.google.com/maps/search/?api=1&query=" + quote_plus(query)


def directions_url(destination: str, home: str) -> str:
    url = "https://www.google.com/maps/dir/?api=1&destination=" + quote_plus(destination)
    return url + ("&origin=" + quote_plus(home) if home else "")


def make_maps(
    profile: Profile,
    launcher: Launcher,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.isfile,
) -> Callable[[MapsArgs], ToolResult]:
    def run(args: MapsArgs) -> ToolResult:
        query = " ".join(args.query.split())[:MAX_QUERY]
        if not query:
            return ToolResult.failed("Tell me what to look for.")
        home, coords = _home(profile)
        if args.mode == "directions":
            url = directions_url(query, home)
            message = f"Getting directions to {query}" + (" from your home." if home else ".")
        else:
            url = search_url(query, home, coords)
            message = f"Searching Google Maps for {query}" + (" near your home." if home else ".")
        try:
            open_in_browser(profile, launcher, url, which, exists)
        except OSError as exc:
            return ToolResult.failed(f"I couldn't open the browser: {exc.strerror or exc}.")
        return ToolResult.launched(message)

    return run
