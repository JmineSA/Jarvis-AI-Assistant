from urllib.parse import parse_qs, urlparse

import pytest

from jarvis.profile import Profile
from jarvis.tools.registry import ToolStatus
from jarvis.tools.web import ComposeEmailArgs, WebSearchArgs, make_compose_email, make_web_search


class FakeLauncher:
    def __init__(self, fail=None):
        self.fail = fail
        self.started, self.urls = [], []

    def start(self, argv, cwd=None):
        if self.fail:
            raise self.fail
        self.started.append(argv)

    def open_url(self, url):
        if self.fail:
            raise self.fail
        self.urls.append(url)

    def open_path(self, path):  # not used here
        raise AssertionError


WHICH = lambda name: f"/bin/{name}"  # noqa: E731


@pytest.fixture
def profile():
    return Profile.model_validate(
        {
            "apps": {"chrome": {"label": "Chrome", "candidates": ["chrome"]}},
            "contacts": {"mum": {"label": "Mum", "names": ["mother"], "email": "mum@example.com"}},
        }
    )


def search(profile, launcher, **kw):
    return make_web_search(profile, launcher, WHICH, lambda p: False)(WebSearchArgs(**kw))


def compose(profile, launcher, **kw):
    return make_compose_email(profile, launcher, WHICH, lambda p: False)(ComposeEmailArgs(**kw))


def test_google_search_opens_chrome_with_the_query_encoded(profile):
    launcher = FakeLauncher()
    result = search(profile, launcher, query="best current action movies")
    assert result.status is ToolStatus.LAUNCHED
    assert result.message == "Searching Google for best current action movies."
    argv = launcher.started[0]
    assert argv[0] == "/bin/chrome"
    assert argv[1] == "https://www.google.com/search?q=best+current+action+movies"


@pytest.mark.parametrize(
    "site, host",
    [("youtube", "www.youtube.com"), ("youtube_music", "music.youtube.com")],
)
def test_other_search_sites(profile, site, host):
    launcher = FakeLauncher()
    search(profile, launcher, query="lofi beats", site=site)
    assert urlparse(launcher.started[0][1]).netloc == host


def test_search_text_cannot_escape_the_address(profile):
    launcher = FakeLauncher()
    search(profile, launcher, query="a&b=c #x\" ; calc.exe")
    url = launcher.started[0][1]
    assert urlparse(url).netloc == "www.google.com"
    assert parse_qs(urlparse(url).query) == {"q": ['a&b=c #x" ; calc.exe']}
    assert len(launcher.started[0]) == 2  # exactly program + address, nothing extra


def test_unknown_site_is_rejected_by_validation():
    with pytest.raises(ValueError):
        WebSearchArgs(query="x", site="evil.example")


def test_search_without_chrome_uses_the_default_browser():
    launcher = FakeLauncher()
    make_web_search(Profile(), launcher)(WebSearchArgs(query="weather"))
    assert launcher.urls and launcher.urls[0].startswith("https://www.google.com/search?q=")


def test_search_browser_failure_is_reported(profile):
    result = search(profile, FakeLauncher(fail=OSError("no browser")), query="x")
    assert result.status is ToolStatus.FAILED


def test_blank_search_is_refused(profile):
    assert search(profile, FakeLauncher(), query="   ").status is ToolStatus.FAILED


def parsed(url):
    return urlparse(url).netloc, parse_qs(urlparse(url).query)


def test_compose_to_a_saved_contact(profile):
    launcher = FakeLauncher()
    result = compose(profile, launcher, to="mother", subject="Sunday lunch", body="See you at 1.")
    host, query = parsed(launcher.started[0][1])
    assert host == "mail.google.com"
    assert query["to"] == ["mum@example.com"] and query["su"] == ["Sunday lunch"] and query["body"] == ["See you at 1."]
    assert "to Mum" in result.message and "Nothing is sent" in result.message


def test_compose_to_a_typed_address(profile):
    launcher = FakeLauncher()
    compose(profile, launcher, to="boss@work.co.za", subject="Hi")
    assert parsed(launcher.started[0][1])[1]["to"] == ["boss@work.co.za"]


def test_compose_to_unknown_name_still_opens_a_draft_and_says_so(profile):
    launcher = FakeLauncher()
    result = compose(profile, launcher, to="Thabo", subject="Hi")
    assert "to" not in parsed(launcher.started[0][1])[1]
    assert "I don't have an address for Thabo" in result.message


def test_compose_cannot_smuggle_extra_recipients_or_fields(profile):
    launcher = FakeLauncher()
    compose(profile, launcher, to="a@b.com,evil@x.com", subject="s&bcc=evil@x.com")
    query = parsed(launcher.started[0][1])[1]
    assert "to" not in query            # not a single valid address, so not used
    assert "bcc" not in query and query["su"] == ["s&bcc=evil@x.com"]


def test_compose_with_nothing_still_opens_a_blank_draft(profile):
    result = compose(profile, FakeLauncher())
    assert result.status is ToolStatus.LAUNCHED


def test_compose_browser_failure(profile):
    assert compose(profile, FakeLauncher(fail=OSError("x"))).status is ToolStatus.FAILED
