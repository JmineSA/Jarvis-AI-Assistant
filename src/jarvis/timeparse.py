"""Understands spoken times for reminders ("in 20 minutes", "tomorrow at 9am", "1 December").

Deliberately small and predictable. Everything works on naive local time, and
`parse_when` returns None whenever it is not sure, so JARVIS asks instead of guessing.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty five": 45, "forty": 40,
    "fifty": 50, "sixty": 60,
}
_NUMBER = r"(?:\d+(?:\.\d+)?|" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True)) + r")"

_UNITS = {
    "second": 1, "sec": 1,
    "minute": 60, "min": 60,
    "hour": 3600, "hr": 3600,
    "day": 86400,
    "week": 604800,
}
_RELATIVE = re.compile(
    rf"\bin\s+(?:about\s+|around\s+)?(?P<n>{_NUMBER})\s*(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)\b"
)
_HALF_HOUR = re.compile(r"\bin\s+half\s+an\s+hour\b")

_MONTHS = {
    name: index
    for index, names in enumerate(
        [
            ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
            ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
            ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"),
            ("december", "dec"),
        ],
        start=1,
    )
    for name in names
}
_MONTH = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + ")"
_DAY_MONTH = re.compile(rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m>{_MONTH})\b(?:\s+(?P<y>\d{{4}}))?")
_MONTH_DAY = re.compile(rf"\b(?P<m>{_MONTH})\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(?P<y>\d{{4}}))?")

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WEEKDAY = re.compile(r"\b(?:next\s+|on\s+)?(?P<wd>" + "|".join(_WEEKDAYS) + r")\b")

_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(?:[t ](\d{1,2}):(\d{2}))?")
_CLOCK = re.compile(r"\b(?:at\s+)?(?P<h>\d{1,2})[:.](?P<m>\d{2})\s*(?P<ap>am|pm)?\b")
_HOUR_AMPM = re.compile(r"\b(?:at\s+)?(?P<h>\d{1,2})\s*(?P<ap>am|pm)\b")
_BARE_HOUR = re.compile(r"\bat\s+(?P<h>\d{1,2})\b(?!\s*(?:st|nd|rd|th|:|\.))")

_PERIODS = {"morning": 8, "afternoon": 15, "evening": 18, "tonight": 20, "night": 20}
DEFAULT_HOUR = 9  # a date with no time means 09:00


def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b([ap])\.m\.?", r"\1m", text)  # "p.m." -> "pm"
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)  # "forty-five" -> "forty five"; ISO dates keep theirs
    return re.sub(r"\s+", " ", text)


def _number(token: str) -> float:
    return float(token) if token[0].isdigit() else float(_WORD_NUMBERS[token])


def _guess_ampm(hour: int) -> str | None:
    """For a time said without am or pm: 1 to 7 and 12 mean the afternoon, 8 to 11 the morning."""
    if 1 <= hour <= 7 or hour == 12:  # people rarely mean 3 in the morning
        return "pm"
    if 8 <= hour <= 11:
        return "am"
    return None  # 0 or 13 to 23: already unambiguous


def _clock_time(s: str) -> tuple[int, int] | None:
    """Hour and minute spoken in the text, or None."""
    if re.search(r"\b(noon|midday)\b", s):
        return 12, 0
    if re.search(r"\bmidnight\b", s):
        return 0, 0
    m = _CLOCK.search(s)
    if m:
        hour, minute, ap = int(m["h"]), int(m["m"]), m["ap"]
        if ap is None and not m["h"].startswith("0"):  # "09:30" is 24-hour; "3:30" is not
            ap = _guess_ampm(hour)
    else:
        m = _HOUR_AMPM.search(s)
        if m:
            hour, minute, ap = int(m["h"]), 0, m["ap"]
        else:
            m = _BARE_HOUR.search(s)
            if not m:
                return None
            hour, minute, ap = int(m["h"]), 0, _guess_ampm(int(m["h"]))
    if ap == "pm" and hour < 12:
        hour += 12
    elif ap == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def _spoken_date(s: str, today: date) -> date | None:
    m = _DAY_MONTH.search(s) or _MONTH_DAY.search(s)
    if not m:
        return None
    day, month = int(m["d"]), _MONTHS[m["m"]]
    year = int(m["y"]) if m["y"] else today.year
    try:
        found = date(year, month, day)
    except ValueError:
        return None
    if not m["y"] and found < today:  # "1 March" said in November means next March
        try:
            found = date(year + 1, month, day)
        except ValueError:
            return None
    return found


def parse_when(text: str, now: datetime) -> datetime | None:
    """Turn a spoken time into a datetime, or None if it cannot be understood confidently."""
    s = _normalise(text)
    if not s:
        return None

    iso = _ISO.search(s)
    if iso:
        try:
            day = date(int(iso[1]), int(iso[2]), int(iso[3]))
            hour, minute = (int(iso[4]), int(iso[5])) if iso[4] else (DEFAULT_HOUR, 0)
            return datetime(day.year, day.month, day.day, hour, minute)
        except ValueError:
            return None

    if _HALF_HOUR.search(s):
        return (now + timedelta(minutes=30)).replace(microsecond=0)
    rel = _RELATIVE.search(s)
    if rel:
        seconds = _number(rel["n"]) * _UNITS[rel["unit"].rstrip("s")]
        return (now + timedelta(seconds=seconds)).replace(microsecond=0)

    today = now.date()
    day: date | None = None
    if "day after tomorrow" in s:
        day = today + timedelta(days=2)
    elif "tomorrow" in s:
        day = today + timedelta(days=1)
    elif re.search(r"\b(today|tonight|this (?:morning|afternoon|evening))\b", s):
        day = today
    else:
        day = _spoken_date(s, today)

    weekday = None if day else _WEEKDAY.search(s)
    if weekday:
        ahead = (_WEEKDAYS.index(weekday["wd"]) - today.weekday()) % 7
        day = today + timedelta(days=ahead)  # same weekday = today, rolled forward below if passed

    time = _clock_time(s)
    if time is None:
        for word, hour in _PERIODS.items():
            if re.search(rf"\b{word}\b", s):
                time = (hour, 0)
                break
    if time is None and day is not None:
        time = (DEFAULT_HOUR, 0)
    if time is None:
        return None

    explicit_day = day is not None
    result = datetime.combine(day or today, datetime.min.time()).replace(hour=time[0], minute=time[1])
    if result <= now:
        if weekday:
            result += timedelta(days=7)
        elif not explicit_day:
            result += timedelta(days=1)  # "at 9am" said at 10am means tomorrow
    return result


def format_due(due: datetime, now: datetime) -> str:
    """How a due time is spoken: "today at 15:30", "tomorrow at 09:00", "1 December at 09:00"."""
    clock = f"{due:%H:%M}"
    delta = (due.date() - now.date()).days
    if delta == 0:
        return f"today at {clock}"
    if delta == 1:
        return f"tomorrow at {clock}"
    if 1 < delta < 7:
        return f"{due:%A} at {clock}"
    year = f" {due.year}" if due.year != now.year else ""
    return f"{due.day} {due:%B}{year} at {clock}"
