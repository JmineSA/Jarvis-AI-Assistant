"""Read-only Gmail: "check my email" tells you who wrote, never sends, deletes or changes anything.

You create a free Google "desktop app" OAuth client once and save its JSON file (see the README).
`python -m jarvis gmail-setup` then opens your browser so you can allow read-only access. Nothing is
hard-coded: the client file and the saved sign-in live under data/gmail/ on your PC (git-ignored), and
only the gmail.readonly permission is ever requested.

Only the sender, subject and date are read. The message text is never fetched, so an email cannot
slip instructions to the assistant.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import re
import secrets
import threading
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from pydantic import Field

from jarvis.tools.registry import ToolArgs, ToolResult

logger = logging.getLogger(__name__)

SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
API = "https://gmail.googleapis.com/gmail/v1/users/me"
MAX_SHOWN = 10
FOLDERS = {"inbox": "INBOX", "sent": "SENT", "spam": "SPAM"}  # what you can ask about; all read-only


class GmailError(Exception):
    """Something went wrong talking to Gmail; the message is safe to say aloud."""


@dataclass(frozen=True)
class MailSummary:
    sender: str
    subject: str
    date: str


def _clean(text: str, limit: int) -> str:
    """Spoken-safe text: no control characters, no markup-looking runs, bounded length."""
    text = re.sub(r"[\x00-\x1f\x7f<>]+", " ", text)
    return " ".join(text.split())[:limit]


def display_name(sender: str) -> str:
    """'Alice Smith <alice@x.com>' -> 'Alice Smith'; a bare address stays as it is."""
    match = re.match(r'^\s*"?([^"<]+?)"?\s*<[^>]*>\s*$', sender)
    return _clean(match.group(1) if match else sender, 60)


# --------------------------------------------------------------------------- sign-in


def _load_client(client_secret_file: Path) -> dict[str, Any]:
    try:
        data = json.loads(client_secret_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GmailError(f"I can't find the Google client file at {client_secret_file}.") from exc
    except (OSError, ValueError) as exc:
        raise GmailError(f"The Google client file at {client_secret_file} could not be read.") from exc
    block = data.get("installed") or data.get("web")
    if not isinstance(block, dict) or not block.get("client_id") or not block.get("token_uri"):
        raise GmailError("That file is not a Google OAuth client file. Create a 'Desktop app' client and download its JSON.")
    return block


def _save_token(token_file: Path, token: dict[str, Any]) -> None:
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(json.dumps(token), encoding="utf-8")
    with contextlib.suppress(OSError):
        token_file.chmod(0o600)


class _Catcher(BaseHTTPRequestHandler):
    result: dict[str, str] = {}  # noqa: RUF012  (set per server below)

    def do_GET(self) -> None:
        query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        if "code" in query or "error" in query:
            self.server.caught = query  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"<html><body style='font-family:sans-serif'><h3>JARVIS is connected to Gmail (read-only).</h3>You can close this tab.</body></html>")

    def log_message(self, *_args: Any) -> None:  # keep the console quiet
        pass


def authorize(
    client_secret_file: Path,
    token_file: Path,
    say: Callable[[str], None] = print,
    open_browser: Callable[[str], Any] = webbrowser.open,
    transport: httpx.BaseTransport | None = None,
    timeout: float = 180.0,
) -> None:
    """Let the user allow read-only access in the browser, then save the sign-in. Raises GmailError."""
    client = _load_client(client_secret_file)
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(16)

    server = HTTPServer(("127.0.0.1", 0), _Catcher)
    server.caught = {}  # type: ignore[attr-defined]
    server.timeout = 1.0
    redirect = f"http://127.0.0.1:{server.server_port}"
    url = client.get("auth_uri", "https://accounts.google.com/o/oauth2/v2/auth") + "?" + urlencode(
        {
            "client_id": client["client_id"],
            "redirect_uri": redirect,
            "response_type": "code",
            "scope": SCOPE,
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    say("Opening your browser so you can allow JARVIS to READ your Gmail (it can never send or delete).")
    say("If nothing opens, copy this address into your browser:\n" + url)
    open_browser(url)

    deadline = time.monotonic() + timeout
    try:
        while not server.caught and time.monotonic() < deadline:  # type: ignore[attr-defined]
            server.handle_request()
    finally:
        server.server_close()
    caught: dict[str, str] = server.caught  # type: ignore[attr-defined]
    if not caught:
        raise GmailError("I didn't hear back from Google in time. Run gmail-setup again.")
    if caught.get("error"):
        raise GmailError(f"Google said no: {caught['error']}.")
    if caught.get("state") != state:
        raise GmailError("That answer did not match my request, so I ignored it. Run gmail-setup again.")

    payload = {
        "client_id": client["client_id"],
        "code": caught["code"],
        "code_verifier": verifier,
        "grant_type": "authorization_code",
        "redirect_uri": redirect,
    }
    if client.get("client_secret"):
        payload["client_secret"] = client["client_secret"]
    try:
        with httpx.Client(timeout=15, transport=transport) as http:
            response = http.post(client["token_uri"], data=payload)
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise GmailError(f"Google would not finish the sign-in: {exc}") from exc
    if not data.get("refresh_token"):
        raise GmailError("Google did not give a lasting sign-in. Remove JARVIS at myaccount.google.com/permissions and try again.")
    _save_token(token_file, {"refresh_token": data["refresh_token"], "access_token": data.get("access_token", ""), "expires_at": time.time() + int(data.get("expires_in", 0))})


# --------------------------------------------------------------------------- client


class GmailClient:
    def __init__(
        self,
        client_secret_file: Path,
        token_file: Path,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._secret_file = client_secret_file
        self._token_file = token_file
        self._transport = transport
        self._clock = clock
        self._lock = threading.Lock()

    @property
    def signed_in(self) -> bool:
        return self._token_file.is_file()

    def _token(self, http: httpx.Client) -> str:
        with self._lock:
            try:
                saved = json.loads(self._token_file.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise GmailError("Gmail is not set up yet. Run: python -m jarvis gmail-setup") from exc
            if saved.get("access_token") and saved.get("expires_at", 0) - 60 > self._clock():
                return str(saved["access_token"])
            client = _load_client(self._secret_file)
            payload = {"client_id": client["client_id"], "refresh_token": saved.get("refresh_token", ""), "grant_type": "refresh_token"}
            if client.get("client_secret"):
                payload["client_secret"] = client["client_secret"]
            try:
                response = http.post(client["token_uri"], data=payload)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as exc:
                raise GmailError("Google refused my sign-in. Run: python -m jarvis gmail-setup") from exc
            except (httpx.HTTPError, ValueError) as exc:
                raise GmailError("I can't reach Google right now. Are you online?") from exc
            saved["access_token"] = data["access_token"]
            saved["expires_at"] = self._clock() + int(data.get("expires_in", 3600))
            _save_token(self._token_file, saved)
            return str(saved["access_token"])

    def _get(self, http: httpx.Client, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = http.get(f"{API}/{path}", params=params, headers={"Authorization": f"Bearer {self._token(http)}"})
            response.raise_for_status()
            return dict(response.json())
        except httpx.HTTPStatusError as exc:
            raise GmailError(f"Gmail answered with an error ({exc.response.status_code}).") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise GmailError("I can't reach Gmail right now. Are you online?") from exc

    def check(
        self, count: int = 3, unread_only: bool = True, sender: str = "", folder: str = "inbox"
    ) -> tuple[int | None, list[MailSummary]]:
        """(how many in that folder, the newest matching messages: who, subject, date only).

        For the inbox the number is the unread count; for spam and sent it is the total. In the sent folder
        "who" is the person you wrote to.
        """
        folder = folder if folder in FOLDERS else "inbox"
        count = max(1, min(count, MAX_SHOWN))
        query = [f"in:{folder}"]
        if unread_only and folder == "inbox":
            query.append("is:unread")
        if sender.strip():
            key = "to" if folder == "sent" else "from"
            query.append(f"{key}:" + re.sub(r"[^\w@.\- ]", "", sender).strip())
        with httpx.Client(timeout=10, transport=self._transport) as http:
            label = self._get(http, f"labels/{FOLDERS[folder]}")
            unread = label.get("messagesUnread") if folder == "inbox" else label.get("messagesTotal")
            listing = self._get(http, "messages", {"q": " ".join(query), "maxResults": count, "includeSpamTrash": folder == "spam"})
            mails = []
            for item in listing.get("messages", [])[:count]:
                meta = self._get(
                    http, f"messages/{item['id']}", {"format": "metadata", "metadataHeaders": ["From", "To", "Subject", "Date"]}
                )
                headers = {h["name"].lower(): h["value"] for h in meta.get("payload", {}).get("headers", [])}
                who = headers.get("to", "someone") if folder == "sent" else headers.get("from", "someone")
                mails.append(
                    MailSummary(
                        display_name(who.split(",")[0]),
                        _clean(headers.get("subject", ""), 90) or "no subject",
                        _clean(headers.get("date", ""), 40),
                    )
                )
        return (int(unread) if unread is not None else None), mails


# --------------------------------------------------------------------------- the tool


class CheckEmailArgs(ToolArgs):
    count: int = Field(3, ge=1, le=MAX_SHOWN)
    unread_only: bool = True
    sender: str = ""  # only mail from this person (to this person, for sent mail)
    folder: Literal["inbox", "sent", "spam"] = "inbox"


def make_check_email(gmail: GmailClient) -> Callable[[CheckEmailArgs], ToolResult]:
    def run(args: CheckEmailArgs) -> ToolResult:
        folder = args.folder
        unread_only = args.unread_only and folder == "inbox"
        try:
            total, mails = gmail.check(args.count, unread_only, args.sender, folder)
        except GmailError as exc:
            return ToolResult.failed(str(exc))
        person = args.sender.strip()
        if folder == "sent":
            if not mails:
                return ToolResult.done(f"You have not sent anything{' to ' + person if person else ''} that I can find.")
            listed = "; ".join(f"{i}. to {m.sender}: {m.subject}" for i, m in enumerate(mails, 1))
            return ToolResult.done(f"Your latest sent: {listed}.")
        if folder == "spam":
            if not total and not mails:
                return ToolResult.done("Your spam folder is empty.")
            head = f"You have {total} email{'s' if total != 1 else ''} in spam." if total is not None and not person else ""
            if not mails:
                return ToolResult.done(f"{head} Nothing from {person} in spam.".strip())
            listed = "; ".join(f"{i}. {m.sender}: {m.subject}" for i, m in enumerate(mails, 1))
            return ToolResult.done(f"{head} Newest in spam{' from ' + person if person else ''}: {listed}.".strip())
        who = f" from {person}" if person else ""
        head = f"You have {total} unread email{'s' if total != 1 else ''} in your inbox." if total is not None and not who else ""
        if not mails:
            return ToolResult.done(head + (f" Nothing{' unread' if unread_only else ''}{who}." if who or not head else ""))
        listed = "; ".join(f"{i}. {m.sender}: {m.subject}" for i, m in enumerate(mails, 1))
        label = "Newest unread" if unread_only else "Newest"
        return ToolResult.done(f"{head} {label}{who}: {listed}.".strip())

    return run
