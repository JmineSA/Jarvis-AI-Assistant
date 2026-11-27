"""Send the model only the tools a sentence could need.

Each tool description costs the model time to read (about 1.2 s for 2,000 tokens on a GTX 970, and Ollama
cannot reuse that work when the sentence changes). Most sentences need one or two tools, so we send those.

Safety nets so a missing tool never blocks you:
  * a tool this table does not know about is always sent;
  * the tools used in your previous turn are sent again ("what about tomorrow?" after the weather);
  * if no tool matches, the model answers as plain conversation, and if it then says it can't do something,
    the request is retried once with every tool.
"""

from __future__ import annotations

import re
from typing import Any

from jarvis.profile import Profile

_RULES: dict[str, str] = {
    "open_app": r"\b(open|launch|start|run|fire up|pull up|bring up|load|switch to|go to)\b",
    "open_folder": r"\b(folders?|director(?:y|ies)|explorer|open)\b",
    "list_files": r"\b(files?|folders?|inside|contents?|list|what'?s in|what is in|show me|look in)\b",
    "open_website": r"\b(web ?sites?|sites?|web ?pages?|go to|browse|open|visit)\b|\.com\b",
    "web_search": r"\b(search|google|look up|lookup|find out|find|youtube|web|news|latest)\b",
    "maps": r"\b(maps?|directions?|routes?|navigate|near|nearby|nearest|closest|how far|how do i get|where)\b",
    "compose_email": r"\b(e-?mails?|mail|gmail|draft|compose|write (?:to|an?)|message)\b",
    "check_email": r"\b(e-?mails?|mail|inbox|gmail|unread|messages?|spam|junk|sent)\b",
    "play_media": r"\b(play|put on|listen|music|songs?|albums?|artists?|tracks?|playlist|shuffle|tunes?|by)\b",
    "open_drive_folder": r"\b(drive|main folder|movies?|games|documentar(?:y|ies)|videos?|series|folders?|software|study)\b",
    "play_video": r"\b(play|watch|movies?|films?|series|episodes?|season|videos?|documentar(?:y|ies)|vlc|concert)\b",
    "media_control": r"\b(pause|resume|skip|next|previous|back|stop|mute|unmute|volume|louder|quieter|turn (?:it )?(?:up|down)|play)\b",
    "weather": r"\b(weather|rain(?:ing|y)?|temperature|hot|cold|forecast|sunny|cloud(?:y|s)|umbrella|wind(?:y)?|degrees|humid(?:ity)?|storms?|warm|chilly)\b",
    "system_status": r"\b(cpu|memory|ram|processor|usage|status|performance|disk|battery|health|how is my (?:computer|pc|machine)|slow|lag(?:ging)?|running)\b",
    "current_time": r"\b(time|date|day|today|o'?clock|tonight|now)\b",
    "remember": r"\b(remember|keep in mind|note that|don'?t forget|save|my (?:name|favou?rite|birthday|number)|i (?:like|prefer|am|work|live|use))\b",
    "recall": r"\b(remember|recall|what do you know|what did i|did i (?:tell|say)|know about me|told you|what(?:'s| is) my)\b",
    "forget": r"\b(forget|delete|remove|erase|clear|wipe)\b",
    "set_reminder": r"\b(remind(?:er)?s?|alarm|alert me|don'?t let me forget|wake me|notify)\b",
    "list_reminders": r"\b(reminders?|schedule|agenda|what do i have|to-?do|coming up)\b",
    "cancel_reminder": r"\b(cancel|delete|remove|clear|reminders?)\b",
    "look_at_screen": r"\b(screen|window|errors?|explain|summari[sz]e|what does this|this (?:code|query|function|page|chart|article|error|script)|read this|why am i (?:getting|seeing))\b",
    "take_screenshot": r"\b(screen ?shots?|capture)\b",
}


class ToolSelector:
    def __init__(self, profile: Profile | None = None) -> None:
        self._rules = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in _RULES.items()}
        # Your own app, folder and website names also point to the matching tool.
        self._names: dict[str, re.Pattern[str]] = {}
        if profile is not None:
            for tool, kind in (("open_app", "apps"), ("open_folder", "folders"), ("list_files", "folders"), ("open_website", "websites")):
                found: set[str] = set()
                for key, entry in profile._entries(kind).items():
                    found.update(w.lower() for w in (key, entry.label or key, *entry.names))
                words = sorted((w for w in found if len(w) >= 2), key=len, reverse=True)
                if words:
                    self._names[tool] = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(w) for w in words) + r")(?!\w)", re.IGNORECASE)

    def select(self, text: str, schemas: list[dict[str, Any]], recent_tools: list[str] | None = None) -> list[dict[str, Any]]:
        recent = set(recent_tools or [])
        chosen = []
        for schema in schemas:
            name = schema.get("function", {}).get("name", "")
            rule = self._rules.get(name)
            if (
                rule is None  # a tool we know nothing about is always offered
                or name in recent
                or rule.search(text)
                or (name in self._names and self._names[name].search(text))
            ):
                chosen.append(schema)
        return chosen


REFUSAL = re.compile(
    r"^\W*(?:i(?:'m| am)?\s+(?:sorry|unable|not able)|sorry|unfortunately|i\s+can(?:'|no)t|i\s+cannot|i\s+(?:don't|do not)\s+have)",
    re.IGNORECASE,
)
