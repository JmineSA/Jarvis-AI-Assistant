"""A short spoken summary of today's reminders, added to the start-up greeting."""

from __future__ import annotations

from datetime import datetime, time, timedelta

from jarvis.memory.store import Reminder


def _spoken_time(moment: datetime) -> str:
    hour = moment.hour % 12 or 12
    suffix = "AM" if moment.hour < 12 else "PM"
    return f"{hour} {suffix}" if moment.minute == 0 else f"{hour}:{moment.minute:02d} {suffix}"


def reminder_briefing(reminders: list[Reminder], now: datetime, limit: int = 2) -> str:
    """Upcoming reminders for the rest of today: important ones first, then earliest. Only `limit` are read out.

    Reminders that are already late are announced separately by the reminder scheduler.
    """
    if limit <= 0:
        return ""
    end_of_day = datetime.combine(now.date() + timedelta(days=1), time.min).timestamp()
    today = sorted(
        (r for r in reminders if now.timestamp() <= r.due_at < end_of_day), key=lambda r: (not r.important, r.due_at, r.id)
    )
    if not today:
        return ""
    parts = [f"{'an important reminder: ' if r.important else ''}{r.text} at {_spoken_time(datetime.fromtimestamp(r.due_at))}" for r in today[:limit]]
    spoken = " Then ".join(f"You have {parts[0]}." if i == 0 else f"{p}." for i, p in enumerate(parts))
    if len(today) > limit:
        more = len(today) - limit
        spoken += f" And {more} more later today."
    return spoken
