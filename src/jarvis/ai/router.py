"""Quick routing of obvious commands, so they never need the language model.

"Open VS Code", "play Love Me by Lil Wayne on YouTube Music", "what's the weather tomorrow"
are matched with fixed patterns and turned straight into a tool call. This is instant and puts
no load on the GPU. Anything that is not clearly one of these returns None and goes to the model
as before. The router only *chooses* a tool and its arguments: the executor still validates
them, asks for confirmation when needed, and is the only thing that acts.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from datetime import datetime

from jarvis.ai.ollama_client import ToolCall
from jarvis.profile import Profile
from jarvis.timeparse import parse_when

_GREETING = re.compile(
    r"^\s*(?:(?:hi|hey|hello|ok|okay|yo)\s+)?(?:jarvis|j\.a\.r\.v\.i\.s\.?)\b[\s,.:!?-]*", re.IGNORECASE
)
_POLITE_START = re.compile(
    r"^(?:(?:can|could|would|will)\s+you\s+|i\s+(?:want|would like|'d like)\s+you\s+to\s+|go\s+ahead\s+and\s+)"
    r"(?:please\s+)?",
    re.IGNORECASE,
)
_PLEASE = re.compile(r"^please\s+|[\s,]+please$", re.IGNORECASE)

_SOURCES = {
    "youtube music": "youtube_music", "you tube music": "youtube_music", "yt music": "youtube_music",
    "youtube": "youtube", "you tube": "youtube",
    "vlc": "local", "vlc media player": "local", "local": "local", "locally": "local",
    "offline": "local", "my computer": "local", "my pc": "local",
    "my music folder": "local", "the music folder": "local", "music folder": "local",
    "my music library": "local", "my library": "local", "my folder": "local", "my music": "local",
}
_SOURCE_WORDS = "|".join(sorted((re.escape(k) for k in _SOURCES), key=len, reverse=True))
_TRAILING_SOURCE = re.compile(rf"\s+(?:on|in|using|with|from|via|through)\s+(?P<src>{_SOURCE_WORDS})$", re.IGNORECASE)
_LOCALLY = re.compile(r"\s+(?:locally|offline)$", re.IGNORECASE)

_PLAY = re.compile(
    r"^(?:(?:open|launch|start)\s+(?P<opened>.+?)\s+(?:and|then)\s+)?(?:play|put on|listen to)\s+(?P<rest>.+)$",
    re.IGNORECASE,
)
_BARE_MUSIC = {
    "music", "some music", "my music", "songs", "some songs", "my songs", "something", "anything",
    "my library", "my music library", "random music", "some random music", "a song", "some tunes", "tunes",
}
_VAGUE = {"it", "it again", "that", "this", "that again", "something else", "the next one", "next", "again"}
_QUERY_LEAD = re.compile(r"^(?:some\s+|the\s+|a\s+)?(?:song\s+|track\s+|album\s+|music\s+by\s+|songs\s+by\s+)?", re.IGNORECASE)

_OPEN = re.compile(r"^(?:open|launch|start|run)\s+(?:up\s+)?(?P<name>.+)$", re.IGNORECASE)
_OPEN_APP_IN = re.compile(r"^(?P<app>.+?)\s+(?:in|with|at)\s+(?P<folder>.+)$", re.IGNORECASE)
_NAME_LEAD = re.compile(r"^(?:my\s+|the\s+)", re.IGNORECASE)
_NAME_TRAIL = re.compile(r"\s+(?:folder|app|application|program|website|site|page)$", re.IGNORECASE)

_WEATHER = re.compile(
    r"^(?:what(?:'s|\s+is)\s+(?:the\s+)?weather|how(?:'s|\s+is)\s+the\s+weather|"
    r"(?:give me|tell me)\s+(?:the\s+)?weather|weather)(?:\s+(?:like|going to be|forecast))*"
    r"(?:\s+(?P<rest>.+))?$",
    re.IGNORECASE,
)
_TIME = re.compile(
    r"^(?:what(?:'s|\s+is)\s+(?:the\s+)?(?:current\s+)?(?:time|date|day)(?:\s+(?:is it|today|now))?|"
    r"what\s+(?:time|day)\s+is\s+it(?:\s+now)?|what(?:'s|\s+is)\s+today'?s?\s+date|tell me the time)$",
    re.IGNORECASE,
)
_STATUS = re.compile(
    r"^(?:system\s+status|(?:check|show)\s+(?:my\s+|the\s+)?(?:system|pc|computer)(?:\s+status)?|"
    r"how(?:'s|\s+is)\s+(?:my|the)\s+(?:pc|computer|system)(?:\s+doing)?)$",
    re.IGNORECASE,
)
_SEARCH = re.compile(r"^(?:search|google|look\s+up|find)\s+(?P<rest>.+)$", re.IGNORECASE)
_SITE_PREFIX = re.compile(r"^(?:(?:on|in)\s+)?(?P<site>youtube\s+music|yt\s+music|youtube|google)\s+(?:for\s+)?(?P<q>.+)$", re.IGNORECASE)
_SITE_SUFFIX = re.compile(r"^(?:for\s+)?(?P<q>.+?)\s+(?:on|in|using)\s+(?P<site>youtube\s+music|yt\s+music|youtube|google)$", re.IGNORECASE)
_DIRECTIONS = re.compile(
    r"^(?:(?:give|get|show)\s+me\s+)?(?:directions?\s+to|navigate\s+to|take\s+me\s+to|route\s+to|"
    r"how\s+(?:do|can)\s+i\s+get\s+to)\s+(?P<dest>.+)$",
    re.IGNORECASE,
)
_NEAR_ME = (
    r"(?:near|around|close\s+to|nearby|near\s+to)\s+(?:me|my\s+(?:area|place|home|house|location))|nearby|around\s+here|near\s+here"
)
_PLACES = re.compile(
    rf"^(?:(?:can\s+you\s+)?(?:find|check(?:\s+for)?|search(?:\s+for)?|look\s+for|show(?:\s+me)?|locate|where(?:'s|\s+is|\s+are))\s+)?"
    rf"(?:(?:any|the|a|some)\s+)?(?P<what>.+?)\s+(?:{_NEAR_ME})$",
    re.IGNORECASE,
)
_NEAREST = re.compile(
    r"^(?:(?:find|show(?:\s+me)?|where(?:'s|\s+is)|take\s+me\s+to)\s+)?(?:the\s+)?(?:nearest|closest)\s+(?P<what>.+)$",
    re.IGNORECASE,
)
_MEDIA_WORDS = {
    r"pause(?:\s+(?:the\s+)?(?:music|song|track|it|playback))?": "play_pause",
    r"resume(?:\s+(?:the\s+)?(?:music|song|track|playback))?": "play_pause",
    r"(?:un)?pause(?:\s+(?:the\s+)?music)?": "play_pause",
    r"continue\s+(?:the\s+)?(?:music|song|track)": "play_pause",
    r"(?:next|skip)(?:\s+(?:the\s+)?(?:song|track|one))?": "next",
    r"skip\s+(?:this|that)(?:\s+(?:song|track))?": "next",
    r"(?:previous|last)\s+(?:song|track|one)|go\s+back(?:\s+a\s+(?:song|track))?|play\s+(?:the\s+)?previous(?:\s+(?:song|track))?": "previous",
    r"stop(?:\s+(?:the\s+)?(?:music|song|track|playing|playback))": "stop",
    r"(?:turn|make)\s+(?:it|the\s+(?:music|volume|sound))\s+(?:up|louder)|volume\s+up|louder|turn\s+up(?:\s+the\s+(?:volume|music))?|raise\s+the\s+volume": "volume_up",
    r"(?:turn|make)\s+(?:it|the\s+(?:music|volume|sound))\s+(?:down|quieter|softer)|volume\s+down|quieter|softer|turn\s+down(?:\s+the\s+(?:volume|music))?|lower\s+the\s+volume": "volume_down",
    r"(?:un)?mute(?:\s+(?:the\s+)?(?:music|sound|volume))?": "mute",
}
_MEDIA = [(re.compile(rf"^(?:{pattern})$", re.IGNORECASE), action) for pattern, action in _MEDIA_WORDS.items()]
_REMEMBER = re.compile(r"^remember(?:\s+that)?\s+(?!to\b)(?P<text>.+)$", re.IGNORECASE)
_REMIND = re.compile(r"^remind\s+me\s+(?P<rest>.+)$", re.IGNORECASE)
_LIST_REMINDERS = re.compile(
    r"^(?:what(?:'s|\s+are)\s+my\s+reminders|list\s+(?:my\s+)?reminders|do\s+i\s+have\s+any\s+reminders|show\s+(?:my\s+)?reminders)$",
    re.IGNORECASE,
)
_WHEN_START = re.compile(r"\s(?=(?:in|at|on|tomorrow|tonight|next|this)\b)", re.IGNORECASE)

_SITE_KEYS = {"google": "google", "youtube": "youtube", "youtube music": "youtube_music", "yt music": "youtube_music"}


_ALLOWED_FILLER = re.compile(r"\bthe\s+(?:morning|afternoon|evening|weekend)\b|\bin\s+an?\s+(?:hour|minute|few)\b|\ban?\s+(?:hour|half)\b", re.IGNORECASE)
_NOT_A_TIME = re.compile(r"\b(?:the|my|our|a|an|that|this place|home|work|bank)\b", re.IGNORECASE)


def _plausible_when(when: str) -> bool:
    """A time phrase, not part of what to be reminded about ("at the bank")."""
    return not _NOT_A_TIME.search(_ALLOWED_FILLER.sub("", when))


def clean(text: str, aliases: Iterable[str] = ()) -> str:
    """The command with the greeting, politeness and end punctuation removed."""
    out = text.strip().replace("’", "'")
    names = [re.escape(a) for a in aliases]
    if names:
        out = re.sub(rf"^\s*(?:(?:hi|hey|hello|ok|okay|yo)\s+)?(?:{'|'.join(names)})\b[\s,.:!?-]*", "", out, flags=re.IGNORECASE)
    out = _GREETING.sub("", out)
    out = _POLITE_START.sub("", out.strip())
    out = _PLEASE.sub("", out.strip())
    return out.strip(" \t\n.,!?;:")


def _source_of(phrase: str) -> str | None:
    return _SOURCES.get(" ".join(phrase.lower().split()))


def _site_key(phrase: str) -> str:
    return _SITE_KEYS[" ".join(phrase.lower().split())]


class CommandRouter:
    def __init__(
        self,
        profile: Profile,
        tools: Iterable[str],
        clock: Callable[[], datetime] = datetime.now,
        aliases: Iterable[str] = (),
    ) -> None:
        self._profile = profile
        self._tools = set(tools)
        self._clock = clock
        self._aliases = list(aliases)

    def __call__(self, text: str) -> ToolCall | None:
        command = clean(text, self._aliases)
        if not command or len(command) > 200:
            return None
        for handler in (
            self._media, self._play, self._open, self._weather, self._maps, self._time, self._status,
            self._search, self._remember, self._remind, self._list_reminders,
        ):
            call = handler(command)
            if call is not None and call.name in self._tools:
                return call
        return None

    # ------------------------------------------------------------------ music

    def _play(self, command: str) -> ToolCall | None:
        match = _PLAY.match(command)
        if not match:
            return None
        rest = match.group("rest").strip()
        source = None
        opened = match.group("opened")
        if opened:
            source = _source_of(_NAME_LEAD.sub("", opened.strip()).strip()) or _source_of(opened.strip())
            if source is None and "music" in opened.lower() and "youtube" not in opened.lower():
                source = "local"
        trailing = _TRAILING_SOURCE.search(rest)
        if trailing:
            source = _source_of(trailing.group("src")) or source
            rest = rest[: trailing.start()].strip()
        else:
            locally = _LOCALLY.search(rest)
            if locally:
                source = "local"
                rest = rest[: locally.start()].strip()
        query = _QUERY_LEAD.sub("", rest).strip()
        if query.lower() in _BARE_MUSIC or rest.lower() in _BARE_MUSIC:
            query = ""
        elif not query or query.lower() in _VAGUE:
            return None
        arguments = {"name": query, "source": source or "auto"}
        return ToolCall("play_media", arguments)

    # ------------------------------------------------------------------ open

    def _kind_of(self, spoken: str) -> tuple[str, str] | None:
        """Which kind of thing a spoken name is (app, folder or website), with its key."""
        base = _NAME_LEAD.sub("", spoken.strip())
        for candidate in dict.fromkeys((spoken.strip(), base, _NAME_TRAIL.sub("", base))):
            if not candidate:
                continue
            for kind in ("apps", "folders", "websites"):
                key = self._profile.resolve(kind, candidate)  # type: ignore[arg-type]
                if key is not None:
                    return kind, key
        return None

    def _open(self, command: str) -> ToolCall | None:
        match = _OPEN.match(command)
        if not match:
            return None
        name = match.group("name").strip()
        # "open VS Code in LightsOn" opens the app inside that project folder.
        inside = _OPEN_APP_IN.match(name)
        if inside:
            app = self._kind_of(inside.group("app"))
            folder = self._kind_of(inside.group("folder"))
            if app and folder and app[0] == "apps" and folder[0] == "folders":
                return ToolCall("open_app", {"name": app[1], "folder": folder[1]})
            return None
        found = self._kind_of(name)
        if found is None:
            return None
        kind, key = found
        tool = {"apps": "open_app", "folders": "open_folder", "websites": "open_website"}[kind]
        return ToolCall(tool, {"name": key})

    # ------------------------------------------------------------------ media keys

    def _media(self, command: str) -> ToolCall | None:
        for pattern, action in _MEDIA:
            if pattern.match(command):
                return ToolCall("media_control", {"action": action})
        return None

    # ------------------------------------------------------------------ maps

    def _maps(self, command: str) -> ToolCall | None:
        directions = _DIRECTIONS.match(command)
        if directions:
            return ToolCall("maps", {"query": directions.group("dest").strip(), "mode": "directions"})
        found = _PLACES.match(command) or _NEAREST.match(command)
        if not found:
            return None
        what = found.group("what").strip()
        if re.match(r"^(?:google|youtube|the web|online|play|open|remind|remember|what|how|who|when|why|is|are|do|does)\b", what, re.IGNORECASE):
            return None
        if not what or what.lower() in {"it", "that", "this", "something", "anything"}:
            return None
        return ToolCall("maps", {"query": what, "mode": "search"})

    # ------------------------------------------------------------------ small ones

    def _weather(self, command: str) -> ToolCall | None:
        match = _WEATHER.match(command)
        if not match:
            return None
        rest = (match.group("rest") or "").strip()
        place = self._profile.location.name.lower() if self._profile.location else ""
        rest = re.sub(r"^(?:like|for|outside|there)\s+", "", rest, flags=re.IGNORECASE)
        rest = re.sub(rf"\s*(?:{_NEAR_ME}|around here|here|outside)\b", "", rest, flags=re.IGNORECASE).strip()
        if place:
            rest = re.sub(rf"\b(?:in|for|at)\s+{re.escape(place)}\b", "", rest, flags=re.IGNORECASE).strip()
        if re.match(r"^(?:in|for)\s+[a-z]", rest, flags=re.IGNORECASE) and not re.match(
            r"^in\s+(?:\d+|an?|half|one|two|three|four|five|six|a few)\b", rest, flags=re.IGNORECASE
        ):
            return None  # another city: let the model explain that only home is known
        return ToolCall("weather", {"when": rest} if rest else {})

    def _time(self, command: str) -> ToolCall | None:
        return ToolCall("current_time", {}) if _TIME.match(command) else None

    def _status(self, command: str) -> ToolCall | None:
        return ToolCall("system_status", {}) if _STATUS.match(command) else None

    def _search(self, command: str) -> ToolCall | None:
        match = _SEARCH.match(command)
        if not match:
            return None
        rest = match.group("rest").strip()
        site, query = "google", rest
        prefix = _SITE_PREFIX.match(rest)
        suffix = _SITE_SUFFIX.match(rest)
        if prefix:
            site, query = _site_key(prefix.group("site")), prefix.group("q")
        elif suffix:
            site, query = _site_key(suffix.group("site")), suffix.group("q")
        else:
            query = re.sub(r"^for\s+", "", rest, flags=re.IGNORECASE)
        query = query.strip()
        if not query or query.lower() in {"it", "that", "this", "something"}:
            return None
        # "find my notes" is not a web search, so only plain search words count for "find".
        if command.lower().startswith("find") and not (prefix or suffix):
            return None
        return ToolCall("web_search", {"query": query, "site": site})

    # ------------------------------------------------------------------ memory

    def _remember(self, command: str) -> ToolCall | None:
        match = _REMEMBER.match(command)
        return ToolCall("remember", {"text": match.group("text").strip()}) if match else None

    def _list_reminders(self, command: str) -> ToolCall | None:
        return ToolCall("list_reminders", {}) if _LIST_REMINDERS.match(command) else None

    def _remind(self, command: str) -> ToolCall | None:
        match = _REMIND.match(command)
        if not match:
            return None
        rest = match.group("rest").strip()
        now = self._clock()
        # "remind me in 20 minutes to call mom"
        lead = re.match(r"^(?P<when>(?:in|at|on|tomorrow|tonight|next|this)\b.+?)\s+(?:to|about|that)\s+(?P<text>.+)$", rest, re.IGNORECASE)
        if lead and _plausible_when(lead.group("when")) and parse_when(lead.group("when"), now) is not None:
            return ToolCall("set_reminder", {"text": lead.group("text").strip(), "when": lead.group("when").strip()})
        # "remind me to call mom in 20 minutes": the earliest split whose time part really parses
        body = re.sub(r"^(?:to|about|that)\s+", "", rest, flags=re.IGNORECASE)
        starts = [m.start() for m in _WHEN_START.finditer(body)]
        for start in starts:
            text, when = body[:start].strip(), body[start:].strip()
            if text and _plausible_when(when) and parse_when(when, now) is not None:
                return ToolCall("set_reminder", {"text": text, "when": when})
        return None
