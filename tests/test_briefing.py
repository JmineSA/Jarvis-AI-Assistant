from datetime import datetime

from jarvis.briefing import reminder_briefing
from jarvis.memory.store import Reminder

NOW = datetime(2026, 11, 26, 9, 0)


def rem(i, text, hour, minute=0, day=26):
    return Reminder(i, text, datetime(2026, 11, day, hour, minute).timestamp(), 0.0, "pending")


def test_no_reminders_today_says_nothing():
    assert reminder_briefing([], NOW) == ""
    assert reminder_briefing([rem(1, "dentist", 10, day=27)], NOW) == ""


def test_reads_the_earliest_and_counts_the_rest():
    items = [rem(3, "call Sam", 16), rem(1, "the team meeting", 10), rem(2, "submit the report", 14, 30)]
    text = reminder_briefing(items, NOW, limit=2)
    assert text == "You have the team meeting at 10 AM. Then submit the report at 2:30 PM. And 1 more later today."


def test_single_reminder_and_limit_zero():
    assert reminder_briefing([rem(1, "gym", 18)], NOW) == "You have gym at 6 PM."
    assert reminder_briefing([rem(1, "gym", 18)], NOW, limit=0) == ""


def test_past_reminders_are_left_to_the_scheduler():
    assert reminder_briefing([rem(1, "early thing", 7)], NOW) == ""


def test_important_reminders_are_read_first():
    early = rem(1, "stretch", 10)
    late = Reminder(2, "pay rent", datetime(2026, 11, 26, 17).timestamp(), 0.0, "pending", True)
    text = reminder_briefing([early, late], NOW, limit=1)
    assert text == "You have an important reminder: pay rent at 5 PM. And 1 more later today."
