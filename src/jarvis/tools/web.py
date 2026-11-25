"""Searching the web and drafting email, by opening pages in your browser.

JARVIS only builds an address and opens it: it never reads the results and never
sends anything. For email it opens a Gmail draft that you read and send yourself.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Callable
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field

from jarvis.profile import Profile
from jarvis.tools.apps import find_executable
from jarvis.tools.launcher import Launcher
from jarvis.tools.registry import ToolArgs, ToolResult

MAX_QUERY = 200
MAX_SUBJECT = 200
MAX_BODY = 2000

SEARCH_SITES: dict[str, tuple[str, str]] = {
    # key: (spoken name, address with {q} where the search words go)
    "google": ("Google", "https://www.google.com/search?q={q}"),
    "youtube": ("YouTube", "https://www.youtube.com/results?search_query={q}"),
    "youtube_music": ("YouTube Music", "https://music.youtube.com/search?q={q}"),
}
GMAIL_COMPOSE = "https://mail.google.com/mail/?view=cm&fs=1&to={to}&su={su}&body={body}"

_EMAIL = re.compile(r"^[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]+$")


def open_in_browser(
    profile: Profile,
    launcher: Launcher,
    url: str,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.isfile,
) -> None:
    """Open an address in Chrome if the profile has it, else the default browser."""
    exe = None
    key = profile.resolve("apps", "chrome")
    if key is not None:
        exe = find_executable(profile.apps[key].candidates, which, exists)
    if exe:
        launcher.start([exe, url])
    else:
        launcher.open_url(url)


# --------------------------------------------------------------------------- search


class WebSearchArgs(ToolArgs):
    query: str = Field(min_length=1)
    site: Literal["google", "youtube", "youtube_music"] = "google"


def make_web_search(
    profile: Profile,
    launcher: Launcher,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.isfile,
) -> Callable[[WebSearchArgs], ToolResult]:
    def run(args: WebSearchArgs) -> ToolResult:
        query = " ".join(args.query.split())
        if not query:
            return ToolResult.failed("Tell me what to search for.")
        name, template = SEARCH_SITES[args.site]
        url = template.format(q=quote_plus(query[:MAX_QUERY]))
        try:
            open_in_browser(profile, launcher, url, which, exists)
        except OSError as exc:
            return ToolResult.failed(f"I couldn't open the browser: {exc.strerror or exc}.")
        return ToolResult.launched(f"Searching {name} for {query[:MAX_QUERY]}.")

    return run


# --------------------------------------------------------------------------- email


class ComposeEmailArgs(ToolArgs):
    to: str = ""
    subject: str = ""
    body: str = ""


def make_compose_email(
    profile: Profile,
    launcher: Launcher,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.isfile,
) -> Callable[[ComposeEmailArgs], ToolResult]:
    def run(args: ComposeEmailArgs) -> ToolResult:
        recipient = ""
        who = ""
        note = ""
        wanted = args.to.strip()
        if wanted:
            if _EMAIL.match(wanted):
                recipient, who = wanted, wanted
            else:
                key = profile.resolve("contacts", wanted)
                if key is not None:
                    recipient = profile.contacts[key].email
                    who = profile.label("contacts", key)
                else:
                    note = f" I don't have an address for {wanted}, so the recipient is blank."

        url = GMAIL_COMPOSE.format(
            to=quote_plus(recipient),
            su=quote_plus(args.subject.strip()[:MAX_SUBJECT]),
            body=quote_plus(args.body.strip()[:MAX_BODY]),
        )
        try:
            open_in_browser(profile, launcher, url, which, exists)
        except OSError as exc:
            return ToolResult.failed(f"I couldn't open the browser: {exc.strerror or exc}.")
        to_part = f" to {who}" if who else ""
        return ToolResult.launched(f"I've opened a Gmail draft{to_part}. Nothing is sent until you press send.{note}")

    return run
