from datetime import datetime

import pytest

from jarvis.greeting import make_greeting, part_of_day


@pytest.mark.parametrize("hour,part", [(5, "morning"), (11, "morning"), (12, "afternoon"), (16, "afternoon"), (17, "evening"), (21, "evening"), (22, "night"), (3, "night")])
def test_part_of_day(hour, part):
    assert part_of_day(datetime(2026, 11, 26, hour)) == part


def test_morning_and_afternoon_wording():
    assert make_greeting("boss", datetime(2026, 11, 26, 10), True) == (
        "Good morning, boss. All systems are running. How can I help you this morning?"
    )
    assert make_greeting("boss", datetime(2026, 11, 26, 13), True).startswith("Good afternoon, boss. All systems")


def test_system_claim_only_when_checked():
    assert "All systems" not in make_greeting("boss", datetime(2026, 11, 26, 9), None)
    assert "isn't responding" in make_greeting("boss", datetime(2026, 11, 26, 9), False)


def test_night_and_blank_name():
    assert make_greeting(" ", datetime(2026, 11, 26, 2)).startswith("Working late, boss?")
    assert make_greeting("boss", datetime(2026, 11, 26, 23)).startswith("Good evening, boss.")
