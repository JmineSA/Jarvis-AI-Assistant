import json
import threading
import time
import urllib.request
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from jarvis.tools.gmail import (
    CheckEmailArgs,
    GmailClient,
    GmailError,
    authorize,
    display_name,
    make_check_email,
)

SECRET = {"installed": {"client_id": "cid", "client_secret": "sec", "token_uri": "https://oauth.test/token",
                        "auth_uri": "https://accounts.test/auth"}}


def files(tmp_path, token=True, expires=9e12):
    secret = tmp_path / "client.json"
    secret.write_text(json.dumps(SECRET))
    tok = tmp_path / "token.json"
    if token:
        tok.write_text(json.dumps({"refresh_token": "r", "access_token": "a", "expires_at": expires}))
    return secret, tok


def api(requests):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if request.url.host == "oauth.test":
            return httpx.Response(200, json={"access_token": "fresh", "expires_in": 3600})
        if path.endswith("/labels/INBOX"):
            return httpx.Response(200, json={"messagesUnread": 7})
        if path.endswith("/messages"):
            return httpx.Response(200, json={"messages": [{"id": "m1"}, {"id": "m2"}]})
        n = path.rsplit("/", 1)[1]
        headers = [{"name": "From", "value": '"Alice Smith" <alice@x.com>' if n == "m1" else "bob@y.com"},
                   {"name": "Subject", "value": "Invoice <script>\x00 due" if n == "m1" else ""},
                   {"name": "Date", "value": "Fri, 27 Nov 2026"}]
        return httpx.Response(200, json={"payload": {"headers": headers}})
    return httpx.MockTransport(handler)


def test_display_name():
    assert display_name('"Alice Smith" <alice@x.com>') == "Alice Smith"
    assert display_name("Bob <b@y.com>") == "Bob"
    assert display_name("bob@y.com") == "bob@y.com"


def test_check_reads_only_headers_and_cleans_them(tmp_path):
    secret, token = files(tmp_path)
    seen = []
    client = GmailClient(secret, token, transport=api(seen))
    unread, mails = client.check(2)
    assert unread == 7
    assert [m.sender for m in mails] == ["Alice Smith", "bob@y.com"]
    assert mails[0].subject == "Invoice script due" and mails[1].subject == "no subject"
    # only metadata was requested, never the body; and it is a query for unread inbox mail
    msg_calls = [r for r in seen if "/messages/m" in r.url.path]
    assert msg_calls and all(r.url.params["format"] == "metadata" for r in msg_calls)
    listing = next(r for r in seen if r.url.path.endswith("/messages"))
    assert listing.url.params["q"] == "in:inbox is:unread"
    assert all(r.method == "GET" for r in seen)


def test_expired_token_is_refreshed_and_saved(tmp_path):
    secret, token = files(tmp_path, expires=1.0)
    seen = []
    GmailClient(secret, token, transport=api(seen), clock=lambda: 1000.0).check(1)
    assert any(r.url.host == "oauth.test" for r in seen)
    assert json.loads(token.read_text())["access_token"] == "fresh"


def test_not_set_up_says_so(tmp_path):
    secret, token = files(tmp_path, token=False)
    with pytest.raises(GmailError, match="gmail-setup"):
        GmailClient(secret, token, transport=api([])).check(1)


def test_sender_filter_is_sanitised(tmp_path):
    secret, token = files(tmp_path)
    seen = []
    GmailClient(secret, token, transport=api(seen)).check(1, sender='sam" OR in:trash')
    q = next(r for r in seen if r.url.path.endswith("/messages")).url.params["q"]
    assert '"' not in q and ":" not in q.split("from:")[1]


def test_tool_message(tmp_path):
    secret, token = files(tmp_path)
    run = make_check_email(GmailClient(secret, token, transport=api([])))
    result = run(CheckEmailArgs(count=2))
    assert result.message == "You have 7 unread emails in your inbox. Newest unread: 1. Alice Smith: Invoice script due; 2. bob@y.com: no subject."


def test_tool_reports_trouble_plainly(tmp_path):
    secret, token = files(tmp_path, token=False)
    result = make_check_email(GmailClient(secret, token))(CheckEmailArgs())
    assert result.status.value == "failed" and "gmail-setup" in result.message


def test_authorize_saves_a_refresh_token_and_checks_state(tmp_path):
    secret, token = files(tmp_path, token=False)
    posted = []

    def token_endpoint(request):
        posted.append(parse_qs(request.content.decode()))
        return httpx.Response(200, json={"access_token": "a1", "refresh_token": "r1", "expires_in": 3600})

    def fake_browser(url):
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        assert q["scope"].endswith("gmail.readonly") and q["code_challenge_method"] == "S256"

        def reply():
            time.sleep(0.2)
            urllib.request.urlopen(f"{q['redirect_uri']}/?code=abc&state={q['state']}", timeout=5).read()

        threading.Thread(target=reply, daemon=True).start()

    authorize(secret, token, say=lambda m: None, open_browser=fake_browser, transport=httpx.MockTransport(token_endpoint))
    assert json.loads(token.read_text())["refresh_token"] == "r1"
    assert posted[0]["code"] == ["abc"] and "code_verifier" in posted[0]


def test_authorize_rejects_a_wrong_state(tmp_path):
    secret, token = files(tmp_path, token=False)

    def fake_browser(url):
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        threading.Thread(target=lambda: (time.sleep(0.2), urllib.request.urlopen(f"{q['redirect_uri']}/?code=abc&state=nope", timeout=5).read()), daemon=True).start()

    with pytest.raises(GmailError, match="did not match"):
        authorize(secret, token, say=lambda m: None, open_browser=fake_browser)
    assert not token.exists()


def test_bad_client_file(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text("{}")
    with pytest.raises(GmailError, match="OAuth client"):
        authorize(bad, tmp_path / "t.json", say=lambda m: None, open_browser=lambda u: None)


def test_gmail_setup_command_switches_gmail_on(tmp_path):
    from jarvis.config import Settings
    from jarvis.diagnostics import run_gmail_setup

    config = tmp_path / "settings.yaml"
    config.write_text("gmail:\n  enabled: false\n", encoding="utf-8")
    settings = Settings.model_validate({"gmail": {"client_secret_file": str(tmp_path / "c.json"), "token_file": str(tmp_path / "t.json")}})
    lines = []
    assert run_gmail_setup(settings, config, say=lines.append, authorize=lambda *a, **k: None) == 0
    assert "enabled: true" in config.read_text(encoding="utf-8")

    def refuse(*a, **k):
        raise GmailError("Google said no: access_denied.")

    assert run_gmail_setup(settings, config, say=lines.append, authorize=refuse) == 1
    assert any("c.json" in line for line in lines)  # tells him where to save the client file


def test_spam_and_sent_folders_are_queried_read_only(tmp_path):
    secret, token = files(tmp_path)
    seen = []
    client = GmailClient(secret, token, transport=api(seen))
    client.check(2, unread_only=True, folder="spam")
    client.check(1, unread_only=True, sender="Alice", folder="sent")
    queries = [r.url.params["q"] for r in seen if r.url.path.endswith("/messages")]
    assert queries == ["in:spam", "in:sent to:Alice"]  # "unread only" only applies to the inbox
    assert any(r.url.path.endswith("/labels/SPAM") for r in seen) and any(r.url.path.endswith("/labels/SENT") for r in seen)
    assert next(r for r in seen if r.url.path.endswith("/messages")).url.params["includeSpamTrash"] == "true"
    assert all(r.method == "GET" for r in seen)


def test_sent_and_spam_messages_read_well(tmp_path):
    secret, token = files(tmp_path)
    run = make_check_email(GmailClient(secret, token, transport=api([])))
    sent = run(CheckEmailArgs(count=1, folder="sent")).message
    assert sent.startswith("Your latest sent: 1. to ") and "unread" not in sent
    spam = run(CheckEmailArgs(count=1, folder="spam")).message
    assert "in spam" in spam and "inbox" not in spam


@pytest.mark.parametrize(
    ("phrase", "args"),
    [
        ("what was my last email sent", {"folder": "sent", "count": 1}),
        ("check my last email I sent", {"folder": "sent", "count": 1}),
        ("show my sent emails", {"folder": "sent", "count": 3}),
        ("check my last email sent to Alice", {"folder": "sent", "count": 1, "sender": "Alice"}),
        ("check my spam folder", {"folder": "spam"}),
        ("check spam", {"folder": "spam"}),
        ("what's in my junk", {"folder": "spam"}),
    ],
)
def test_router_sends_sent_and_spam_to_the_email_tool(phrase, args):
    from jarvis.ai.router import CommandRouter
    from jarvis.profile import load_profile
    from tests.test_router import NOW, PROJECT_ROOT

    router = CommandRouter(load_profile(PROJECT_ROOT / "config" / "profile.yaml"), ["check_email"], clock=lambda: NOW)
    call = router(phrase)
    assert call is not None and call.name == "check_email"
    for key, value in args.items():
        assert call.arguments[key] == value, (phrase, call)
