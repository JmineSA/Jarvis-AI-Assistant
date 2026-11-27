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
_VIDEO = re.compile(
    r"^(?:(?:open|go to)\s+(?:the\s+|my\s+)?(?P<folder>.+?)\s+(?:folder\s+)?(?:and\s+then|and|then)\s+)?"
    r"(?:play|watch|put on)\s+(?:the\s+|a\s+|some\s+)?(?:(?P<kind>(?:music\s+)?videos?|movies?|films?|series|shows?|episodes?|documentar(?:y|ies)|live\s+performances?|concerts?)\s*)?(?P<name>.*?)"
    r"(?:\s+(?:in|on|with|using)\s+(?:vlc|(?:the\s+)?media player))?$",
    re.IGNORECASE,
)
# Words that mark a title as a series/movie file rather than a song: S01E02, season 2, 1080p, .mkv
_VIDEOISH = re.compile(r"\bs\d{1,2}(?:e\d{1,2})?\b|\bseason\s*\d|\bepisode\b|\b(?:480|720|1080|2160)p\b|\.(?:mkv|mp4|avi)\b", re.IGNORECASE)
# "... in the main folder" / "... in my movies folder" at the end of a play request
_SCOPE_TAIL = re.compile(r"\s+(?:in|inside|from)\s+(?:the\s+|my\s+)?(?P<scope>[\w &'-]+?)\s+(?:folder|drive)$", re.IGNORECASE)
_WHOLE_DRIVE = {"main", "root", "g", "everything", "all"}
_STEP_BREAK = re.compile(r"\s*,?\s+(?:and\s+then|and\s+after\s+that|after\s+that|then|and)\s+|\s*,\s+", re.IGNORECASE)
_TAIL_VLC = r"(?:\s+(?:in|on|with|using)\s+(?:vlc|(?:the\s+)?media player))?"
_PLAY_IT = r"(?:\s+(?:and\s+)?(?:then\s+)?play(?:\s+it)?)?"
_SCOPED_MUSIC = (
    re.compile(
        r"^(?:in|inside|from)\s+(?:the\s+|my\s+)?(?:main|music|my music)(?:\s+folder)?\s*,?\s*"
        r"(?:(?:search|find|look)(?:\s+for)?|play)?\s*(?P<q>.+?)" + _PLAY_IT + _TAIL_VLC + "$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:search|find|look\s+for)\s+(?:the\s+|my\s+)?(?:music|main)\s+folder\s+for\s+(?P<q>.+?)" + _PLAY_IT + _TAIL_VLC + "$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:search|find)\s+(?:for\s+)?(?P<q>.+?)\s+(?:and\s+)?(?:then\s+)?play\s+it" + _TAIL_VLC + "$",
        re.IGNORECASE,
    ),
)
_DRIVE_FOLDER = re.compile(
    r"^(?:(?:the|my)\s+)?(?P<name>.+?)\s+(?:folder|drive|directory)$|^(?:the\s+)?(?P<root>main|root)$", re.IGNORECASE
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
_IMPORTANT_LEAD = re.compile(
    r"^(?:(?:set|add|make|create)\s+(?:me\s+)?(?:an?\s+)?)?important\s+reminder\s+(?:for\s+me\s+)?(?P<rest>(?:to|about|that|in|at|on|tomorrow)\b.+)$",
    re.IGNORECASE,
)
_IMPORTANT_TAIL = re.compile(r"[,.]?\s*(?:(?:and\s+)?(?:it(?:'s|\s+is)|that(?:'s|\s+is)|this\s+is)\s+(?:very\s+)?important|mark\s+it\s+(?:as\s+)?important)\s*$", re.IGNORECASE)
_SCREENSHOT = re.compile(
    r"^(?:please\s+)?(?:take|grab|capture|save|make)\s+(?:me\s+)?(?:a\s+)?screen\s?shot(?:\s+of\s+(?:my\s+|the\s+)?(?:window|screen|whole\s+screen|entire\s+screen))?\s*[.!]?$",
    re.IGNORECASE,
)
_LOOK = re.compile(
    r"^(?:please\s+|jarvis[, ]+)?(?:"
    r"what(?:'s|\s+is|\s+are)\s+(?:on|in)\s+(?:my\s+|the\s+)?(?:(?:whole|entire|full)\s+)?(?:screen|window)"
    r"|what\s+(?:do|can)\s+you\s+see(?:\s+on\s+(?:my\s+|the\s+)?screen)?"
    r"|(?:look\s+at|read|check|scan)\s+(?:my\s+|the\s+)?(?:(?:whole|entire|full)\s+)?(?:screen|window)"
    r"|(?:explain|summari[sz]e|describe|analy[sz]e|review|debug|read)\s+(?:this|that|these|those|the\s+(?:error|query|code|function|chart|graph|article|page|screen|window|terminal|output|message|text|table|script|log))\b"
    r"|what\s+does\s+(?:this|that)\s+(?:\w+\s+)?(?:function|code|query|script|line|error|command|class|method|sql|program)\b"
    r"|what(?:'s|\s+is)\s+(?:this|that)\s+(?:error|chart|graph|page|code|query|application|app|window|program|table)\b"
    r"|why\s+am\s+i\s+(?:getting|seeing)\s+(?:this|that|an?|the)\s+(?:error|warning|exception|message)"
    r"|what\s+(?:is|does)\s+this\s+(?:application|app|page|window|program)\s+(?:showing|saying|telling|display)"
    r"|summari[sz]e\s+(?:the\s+)?(?:article|page|post|email|document|thread)\s+(?:i(?:'m|\s+am)\s+reading|on\s+(?:my\s+)?screen)"
    r")",
    re.IGNORECASE,
)
_EMAIL = re.compile(
    r"^(?:(?:please\s+)?(?:check|read|show|open|see)\s+(?:me\s+)?(?:my\s+)?(?:(?:last|latest|recent|new|unread)\s+)?(?:(?P<n>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?"
    r"(?:(?:new|unread|latest|last|recent)\s+)?(?:e-?mails?|inbox|mail|gmail)"
    r"|(?:do\s+i\s+have|are\s+there|have\s+i\s+got|any)\s+(?:any\s+)?(?:new\s+|unread\s+)?(?:e-?mails?|mail|messages?)"
    r"|what(?:'s|\s+is)\s+in\s+my\s+(?:inbox|mail|gmail))"
    r"(?:\s+(?:from|by)\s+(?P<sender>[\w .'-]{2,40}?))?\s*[?.!]?$",
    re.IGNORECASE,
)
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

    def __call__(self, text: str) -> ToolCall | list[ToolCall] | None:
        """One call, or a short list of calls for "do this, then that"; None leaves it to the model."""
        command = clean(text, self._aliases)
        if not command or len(command) > 200:
            return None
        return self._plan(command, depth=0)

    def _plan(self, command: str, depth: int) -> ToolCall | list[ToolCall] | None:
        call = self._single(command)
        if call is not None or depth >= 2:
            return call
        for match in _STEP_BREAK.finditer(command):
            left, right = command[: match.start()].strip(), command[match.end():].strip()
            first = self._single(left)
            if first is None or not right:
                continue
            rest = self._plan(right, depth + 1)
            if rest is not None:
                return [first, *(rest if isinstance(rest, list) else [rest])]
        return None

    def _single(self, command: str) -> ToolCall | None:
        for handler in (
            self._media, self._scoped_music, self._video, self._play, self._open, self._weather, self._maps, self._time, self._status,
            self._search, self._remember, self._remind, self._list_reminders, self._email, self._screen,
        ):
            call = handler(command)
            if call is not None and call.name in self._tools:
                return call
        return None

    # ------------------------------------------------------------------ video and drive

    def _scoped_music(self, command: str) -> ToolCall | None:
        """"In the music folder search Love Me by Lil Wayne and play it": the folder is only a hint."""
        if "play_media" not in self._tools:
            return None
        for pattern in _SCOPED_MUSIC:
            match = pattern.match(command)
            if match:
                query = re.sub(r"^(?:a|an|the|some|any)\s+", "", match.group("q").strip(), flags=re.IGNORECASE)
                query = re.sub(r"\s+(?:songs?|tracks?|music)$", "", query, flags=re.IGNORECASE).strip()
                if not query or re.match(r"^(?:open|launch|go|start)\b", query, re.IGNORECASE):
                    return None
                if re.search(r"\b(?:movie|film|series|video|google|youtube)\b", query, re.IGNORECASE):
                    return None
                return ToolCall("play_media", {"name": query, "source": "local"})
        return None

    def _video(self, command: str) -> ToolCall | None:
        if re.search(r"\byoutube\b", command, re.IGNORECASE):
            return None  # you asked for the streaming site: that is the music/YouTube route
        command = re.sub(r"\s+(?:locally|from\s+(?:my\s+)?(?:drive|computer|pc|laptop)|(?:on|from)\s+(?:the\s+)?(?:g\s+)?drive)$", "", command, flags=re.IGNORECASE)
        # "play Love Me by Lil Wayne music video" -> the kind word at the end moves to the front
        trailing = re.match(r"^(?P<verb>play|watch|put on)\s+(?P<name>.+?)\s+(?P<kind>(?:music\s+)?videos?)$", command, re.IGNORECASE)
        if trailing:
            command = f"{trailing.group('verb')} the {trailing.group('kind')} {trailing.group('name')}"
        match = _VIDEO.match(command)
        if not match or "play_video" not in self._tools:
            return None
        folder = (match.group("folder") or "").strip()
        kind = match.group("kind")
        name = match.group("name").strip()
        verb = command.split()[0].lower()
        tail = _SCOPE_TAIL.search(name)
        if tail and not folder:
            scope = tail.group("scope").strip()
            if re.search(r"music|song|album", scope, re.IGNORECASE):
                return None
            name = name[: tail.start()].strip()
            folder = "" if scope.lower() in _WHOLE_DRIVE else scope
            if not kind and verb != "watch" and not _VIDEOISH.search(name) and not folder:
                return None  # "play X in the main folder" with no sign it is a video: let the music search try
        if not folder and not kind and verb != "watch" and not _VIDEOISH.search(name):
            return None
        if folder and re.search(r"music|youtube|song|album", folder, re.IGNORECASE):
            return None  # "open my music folder and play ..." stays music
        if name.lower() in {"it", "that", "this", "something", "anything", "one"}:
            name = ""
        arguments = {"name": name}
        if folder:
            arguments["folder"] = folder
        return ToolCall("play_video", arguments)

    def _installed_app(self, name: str) -> ToolCall | None:
        """"open Pod Player": not in the profile, so let open_app look in the Start menu."""
        if "open_app" not in self._tools:
            return None
        words = name.split()
        if not 1 <= len(words) <= 4 or re.match(r"^(?:it|that|this|them|something|anything|up|my\s+(?:stuff|things))\b", name, re.IGNORECASE):
            return None
        if re.search(r"\b(?:and|then|to|for|in|on|with)\b", name, re.IGNORECASE):
            return None  # a longer request: leave it to the model
        return ToolCall("open_app", {"name": _NAME_TRAIL.sub("", _NAME_LEAD.sub("", name.strip()))})

    def _drive_folder(self, name: str) -> ToolCall | None:
        if "open_drive_folder" not in self._tools:
            return None
        if re.search(r"\b(?:and|then)\b|,", name, re.IGNORECASE):
            return None  # two requests in one: split into steps instead
        match = _DRIVE_FOLDER.match(name.strip())
        if not match:
            return None
        spoken = match.group("name") or match.group("root")
        return ToolCall("open_drive_folder", {"name": spoken.strip()})

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
        tail = _SCOPE_TAIL.search(rest)
        if tail:  # "play Love Me in the music folder" / "... in the main folder": only a hint
            rest = rest[: tail.start()].strip()
            if re.search(r"music|song|album", tail.group("scope"), re.IGNORECASE):
                source = source or "local"
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
            return self._drive_folder(name) or self._installed_app(name)
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

    def _screen(self, command: str) -> ToolCall | None:
        """"What's on my screen?", "explain this error", "take a screenshot": only ever on request."""
        text = command.strip()
        whole = bool(re.search(r"\b(?:whole|entire|full|all)\b.*\bscreens?\b", text, re.IGNORECASE))
        shot = _SCREENSHOT.match(text)
        if shot and "take_screenshot" in self._tools:
            window = bool(re.search(r"\bwindow\b", text, re.IGNORECASE))
            return ToolCall("take_screenshot", {"whole_screen": not window})
        if "look_at_screen" in self._tools and _LOOK.match(text):
            args: dict[str, object] = {"question": text}
            if whole:
                args["whole_screen"] = True
            return ToolCall("look_at_screen", args)
        return None

    def _email(self, command: str) -> ToolCall | None:
        """"Check my email", "any new emails from Sam", "read my last 5 emails" (read-only)."""
        if "check_email" not in self._tools:
            return None
        match = _EMAIL.match(command)
        if not match:
            return None
        args: dict[str, object] = {}
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        number = (match.group("n") or "").lower()
        if number:
            args["count"] = int(number) if number.isdigit() else words[number]
        if match.group("sender"):
            args["sender"] = match.group("sender").strip()
        if re.search(r"\b(?:last|latest|recent)\b", command, re.IGNORECASE):
            args["unread_only"] = False
        return ToolCall("check_email", args)

    def _list_reminders(self, command: str) -> ToolCall | None:
        return ToolCall("list_reminders", {}) if _LIST_REMINDERS.match(command) else None

    def _remind(self, command: str) -> ToolCall | None:
        important = False
        # "set an important reminder to ...", "important: remind me ...", "... , it's important"
        lead = _IMPORTANT_LEAD.match(command)
        if lead:
            command, important = "remind me " + lead.group("rest"), True
        elif _IMPORTANT_TAIL.search(command):
            command, important = _IMPORTANT_TAIL.sub("", command), True
        elif re.match(r"^important\b[:,]?\s+remind\s+me\b", command, re.IGNORECASE):
            command, important = re.sub(r"^important\b[:,]?\s+", "", command, flags=re.IGNORECASE), True
        call = self._remind_plain(command)
        if call is not None and important:
            call.arguments["important"] = True
        return call

    def _remind_plain(self, command: str) -> ToolCall | None:
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
