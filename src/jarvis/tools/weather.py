"""Weather for your home city, from Open-Meteo (free, no account or key).

Only your city's coordinates are sent. The spoken answer is built from the real forecast.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

import httpx
from pydantic import Field

from jarvis.profile import Location
from jarvis.timeparse import parse_when
from jarvis.tools.registry import ToolArgs, ToolResult

logger = logging.getLogger(__name__)

API_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT = 8.0
FORECAST_DAYS = 7

Fetch = Callable[[Location], dict[str, Any]]

# WMO weather codes, as words that read well aloud.
_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy",
    51: "drizzly", 53: "drizzly", 55: "drizzly", 56: "drizzly", 57: "drizzly",
    61: "rainy", 63: "rainy", 65: "rainy heavily", 66: "rainy", 67: "rainy",
    71: "snowy", 73: "snowy", 75: "snowy", 77: "snowy", 85: "snowy", 86: "snowy",
    80: "showery", 81: "showery", 82: "showery heavily",
    95: "stormy", 96: "stormy with hail", 99: "stormy with hail",
}

# Phrases that mean a particular hour rather than the whole day.
_HOUR_SPOKEN = re.compile(
    r"\d\s*(?:am|pm|a\.m\.|p\.m\.)|\d[:.]\d{2}|\bat\s+\d|noon|midday|midnight|"
    r"morning|afternoon|evening|tonight|\bnight\b|\bin\s+\S+\s+(?:hours?|hrs?|minutes?|mins?)\b",
    re.IGNORECASE,
)


def fetch_forecast(location: Location, transport: httpx.BaseTransport | None = None) -> dict[str, Any]:
    """Ask Open-Meteo. Raises httpx.HTTPError or ValueError on any problem."""
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "current": "temperature_2m,weather_code",
        "hourly": "temperature_2m,precipitation_probability,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
        "timezone": "auto",
        "forecast_days": FORECAST_DAYS,
    }
    with httpx.Client(timeout=REQUEST_TIMEOUT, transport=transport) as client:
        response = client.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
    if not isinstance(data, dict) or "daily" not in data or "hourly" not in data:
        raise ValueError("unexpected weather response")
    return data


def _describe(code: Any) -> str:
    return _CODES.get(int(code), "unsettled") if code is not None else "unsettled"


def _degrees(value: Any) -> str:
    return f"{round(float(value))} degrees"


def _rain(percent: Any) -> str:
    if percent is None:
        return ""
    chance = int(percent)
    return "no real chance of rain" if chance < 5 else f"a {chance} percent chance of rain"


def _day_word(day: date, today: date) -> str:
    delta = (day - today).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    return f"on {day:%A}"


def _index(times: list[str], wanted: str) -> int | None:
    return times.index(wanted) if wanted in times else None


def _daily_sentence(data: dict[str, Any], day: date, today: date, place: str) -> str | None:
    daily = data["daily"]
    i = _index(daily["time"], day.isoformat())
    if i is None:
        return None
    word = _day_word(day, today)
    start = (
        f"{word[:1].upper()}{word[1:]} in {place}: {_describe(daily['weather_code'][i])}, "
        f"between {round(daily['temperature_2m_min'][i])} and {round(daily['temperature_2m_max'][i])} degrees"
    )
    rain = _rain(daily["precipitation_probability_max"][i])
    return start + (f", with {rain}." if rain else ".")


def _hour_sentence(data: dict[str, Any], due: datetime, today: date, place: str) -> str | None:
    hourly = data["hourly"]
    i = _index(hourly["time"], due.strftime("%Y-%m-%dT%H:00"))
    if i is None:
        return None
    rain = _rain(hourly["precipitation_probability"][i])
    return (
        f"At {due:%H}:00 {_day_word(due.date(), today)} in {place} it should be "
        f"{_degrees(hourly['temperature_2m'][i])} and {_describe(hourly['weather_code'][i])}"
        + (f", with {rain}." if rain else ".")
    )


def _now_sentence(data: dict[str, Any], today: date, place: str) -> str:
    current = data["current"]
    text = (
        f"Right now in {place} it's {_degrees(current['temperature_2m'])} and "
        f"{_describe(current['weather_code'])}."
    )
    day = _daily_sentence(data, today, today, place)
    if day:
        daily = data["daily"]
        i = _index(daily["time"], today.isoformat())
        if i is not None:
            rain = _rain(daily["precipitation_probability_max"][i])
            text += (
                f" Today's high is {round(daily['temperature_2m_max'][i])} and the low is "
                f"{round(daily['temperature_2m_min'][i])}" + (f", with {rain}." if rain else ".")
            )
    return text


class WeatherArgs(ToolArgs):
    when: str = Field(default="", max_length=100)


def make_weather(
    location: Location | None,
    clock: Callable[[], datetime] = datetime.now,
    fetch: Fetch = fetch_forecast,
) -> Callable[[WeatherArgs], ToolResult]:
    def run(args: WeatherArgs) -> ToolResult:
        if location is None:
            return ToolResult.failed("I don't know where you are. Add your city under location in config/profile.yaml.")

        now = clock()
        wanted = args.when.strip()
        due = parse_when(wanted, now) if wanted else None
        if wanted and due is None and wanted.lower() not in {"now", "today", "currently", "right now"}:
            return ToolResult.failed(
                "I couldn't understand when. Try: today, tomorrow, at 2pm, or Friday."
            )
        if due is not None and (due.date() - now.date()).days >= FORECAST_DAYS:
            return ToolResult.failed("I can only forecast the next week.")

        try:
            data = fetch(location)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.warning("Weather lookup failed: %s", exc)
            return ToolResult.failed("I couldn't reach the weather service just now.")

        try:
            today = now.date()
            if due is None:
                sentence = _now_sentence(data, today, location.name)
            elif _HOUR_SPOKEN.search(wanted):
                sentence = _hour_sentence(data, due, today, location.name)
            else:
                sentence = _daily_sentence(data, due.date(), today, location.name)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("Weather response was not as expected: %s", exc)
            return ToolResult.failed("The weather service gave me something I couldn't read.")

        if sentence is None:
            return ToolResult.failed("I don't have a forecast for that time.")
        return ToolResult.done(sentence)

    return run


def current_chip(data: dict[str, Any]) -> str:
    """Short text for the dashboard, e.g. "24° Partly cloudy"."""
    current = data["current"]
    return f"{round(float(current['temperature_2m']))}° {_describe(current['weather_code']).capitalize()}"
