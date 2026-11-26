from datetime import datetime

import pytest

from jarvis.timeparse import format_due, parse_when

NOW = datetime(2026, 11, 24, 16, 45)  # a Tuesday


def at(*args):
    return datetime(*args)


@pytest.mark.parametrize(
    "text, expected",
    [
        # relative
        ("in 20 minutes", at(2026, 11, 24, 17, 5)),
        ("in an hour", at(2026, 11, 24, 17, 45)),
        ("in two hours", at(2026, 11, 24, 18, 45)),
        ("in 1.5 hours", at(2026, 11, 24, 18, 15)),
        ("in half an hour", at(2026, 11, 24, 17, 15)),
        ("in forty-five minutes", at(2026, 11, 24, 17, 30)),
        ("in 3 days", at(2026, 11, 27, 16, 45)),
        ("remind me in 20 minutes please", at(2026, 11, 24, 17, 5)),
        # today / tomorrow
        ("tomorrow at 9am", at(2026, 11, 25, 9, 0)),
        ("tomorrow", at(2026, 11, 25, 9, 0)),
        ("tomorrow morning", at(2026, 11, 25, 8, 0)),
        ("day after tomorrow at 8", at(2026, 11, 26, 8, 0)),
        ("tonight", at(2026, 11, 24, 20, 0)),
        ("this evening", at(2026, 11, 24, 18, 0)),
        ("today at 5:15 pm", at(2026, 11, 24, 17, 15)),
        ("at 5:15 p.m.", at(2026, 11, 24, 17, 15)),
        # a time with no day: today if still ahead, otherwise tomorrow
        ("at 5", at(2026, 11, 24, 17, 0)),
        ("at 3pm", at(2026, 11, 25, 15, 0)),
        ("at 9", at(2026, 11, 25, 9, 0)),
        ("at 3:30", at(2026, 11, 25, 15, 30)),
        ("at 09:30", at(2026, 11, 25, 9, 30)),
        ("at 18:00", at(2026, 11, 24, 18, 0)),
        ("at noon", at(2026, 11, 25, 12, 0)),
        ("midnight", at(2026, 11, 25, 0, 0)),
        # dates
        ("1 December", at(2026, 12, 1, 9, 0)),
        ("1st of December at 10:30", at(2026, 12, 1, 10, 30)),
        ("December 1", at(2026, 12, 1, 9, 0)),
        ("december 1st 2027", at(2027, 12, 1, 9, 0)),
        ("3 March", at(2027, 3, 3, 9, 0)),  # already passed this year: next year
        # weekdays
        ("monday", at(2026, 11, 30, 9, 0)),
        ("next monday at 10:30", at(2026, 11, 30, 10, 30)),
        ("friday at 2pm", at(2026, 11, 27, 14, 0)),
        ("tuesday at 5pm", at(2026, 11, 24, 17, 0)),   # later today
        ("tuesday at 9am", at(2026, 12, 1, 9, 0)),     # already passed today: next week
        # ISO
        ("2026-12-01T09:30", at(2026, 12, 1, 9, 30)),
        ("2026-12-01", at(2026, 12, 1, 9, 0)),
    ],
)
def test_understood_times(text, expected):
    assert parse_when(text, NOW) == expected


@pytest.mark.parametrize(
    "text",
    ["", "   ", "banana", "soon", "someday", "30 February", "at 25:00", "2026-13-45", "remind me"],
)
def test_unclear_times_are_not_guessed(text):
    assert parse_when(text, NOW) is None


def test_a_past_date_is_returned_as_is_so_the_caller_can_refuse_it():
    assert parse_when("December 1 2020", NOW) == at(2020, 12, 1, 9, 0)


@pytest.mark.parametrize(
    "due, expected",
    [
        (at(2026, 11, 24, 17, 5), "today at 17:05"),
        (at(2026, 11, 25, 9, 0), "tomorrow at 09:00"),
        (at(2026, 11, 27, 14, 0), "Friday at 14:00"),
        (at(2026, 12, 1, 9, 0), "1 December at 09:00"),
        (at(2027, 3, 3, 9, 0), "3 March 2027 at 09:00"),
    ],
)
def test_format_due(due, expected):
    assert format_due(due, NOW) == expected
