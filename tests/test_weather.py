from datetime import datetime

import httpx
import pytest

from jarvis.profile import Location
from jarvis.tools.registry import ToolStatus
from jarvis.tools.weather import WeatherArgs, fetch_forecast, make_weather

NOW = datetime(2026, 11, 25, 10, 15)  # a Wednesday
PRETORIA = Location(name="Pretoria", latitude=-25.7479, longitude=28.2293)


def forecast():
    hours = [f"2026-11-{25 + h // 24:02d}T{h % 24:02d}:00" for h in range(0, 72)]
    return {
        "current": {"temperature_2m": 24.4, "weather_code": 2},
        "hourly": {
            "time": hours,
            "temperature_2m": [10 + (h % 24) for h in range(72)],  # 14:00 -> 24 degrees
            "precipitation_probability": [h % 24 * 4 for h in range(72)],
            "weather_code": [3 if h % 24 >= 12 else 0 for h in range(72)],
        },
        "daily": {
            "time": ["2026-11-25", "2026-11-26", "2026-11-27"],
            "temperature_2m_max": [29.2, 27.0, 31.0],
            "temperature_2m_min": [16.6, 15.0, 17.0],
            "precipitation_probability_max": [40, 0, 90],
            "weather_code": [2, 0, 95],
        },
    }


def run(when="", location=PRETORIA, fetch=lambda loc: forecast()):
    return make_weather(location, lambda: NOW, fetch)(WeatherArgs(when=when))


def test_right_now_gives_current_conditions_and_the_days_range():
    result = run()
    assert result.status is ToolStatus.DONE
    assert result.message == (
        "Right now in Pretoria it's 24 degrees and partly cloudy. "
        "Today's high is 29 and the low is 17, with a 40 percent chance of rain."
    )


def test_a_specific_hour_uses_the_hourly_forecast():
    result = run("at 2pm")
    assert result.message == "At 14:00 today in Pretoria it should be 24 degrees and overcast, with a 56 percent chance of rain."


def test_tomorrow_gives_the_daily_summary():
    result = run("tomorrow")
    assert result.message == "Tomorrow in Pretoria: clear, between 15 and 27 degrees, with no real chance of rain."


def test_an_hour_tomorrow_morning():
    result = run("tomorrow morning")
    assert result.message.startswith("At 08:00 tomorrow in Pretoria it should be 18 degrees and clear")


def test_a_later_day_is_named_by_weekday():
    result = run("friday")
    assert result.message.startswith("On Friday in Pretoria: stormy")


def test_today_means_the_daily_summary():
    assert run("today").message.startswith("Today in Pretoria: partly cloudy, between 17 and 29")


def test_without_a_location_it_says_how_to_fix_it():
    result = run(location=None)
    assert result.status is ToolStatus.FAILED and "location" in result.message


@pytest.mark.parametrize("when", ["banana", "someday"])
def test_unclear_time_is_not_guessed(when):
    assert run(when).status is ToolStatus.FAILED


def test_too_far_ahead_is_refused():
    assert run("in 30 days").message == "I can only forecast the next week."


def test_a_time_with_no_forecast_row_says_so():
    result = run("in 100 hours")  # inside the week, but beyond the rows we hold
    assert result.status is ToolStatus.FAILED and "forecast" in result.message


def test_network_failure_is_reported_plainly():
    def broken(loc):
        raise httpx.ConnectError("offline")

    result = run(fetch=broken)
    assert result.status is ToolStatus.FAILED and result.message == "I couldn't reach the weather service just now."


def test_malformed_response_is_reported_plainly():
    result = run(fetch=lambda loc: {"current": {}, "hourly": {"time": []}, "daily": {"time": ["2026-11-25"]}})
    assert result.status is ToolStatus.FAILED and "couldn't read" in result.message


def test_fetch_sends_only_coordinates_and_parses():
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=forecast())

    data = fetch_forecast(PRETORIA, transport=httpx.MockTransport(handler))
    assert "daily" in data
    assert seen["params"]["latitude"] == "-25.7479" and seen["params"]["longitude"] == "28.2293"
    assert set(seen["params"]) == {
        "latitude", "longitude", "current", "hourly", "daily", "timezone", "forecast_days",
    }


def test_fetch_raises_on_http_error_and_junk():
    with pytest.raises(httpx.HTTPError):
        fetch_forecast(PRETORIA, transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(ValueError):
        fetch_forecast(PRETORIA, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"x": 1})))
