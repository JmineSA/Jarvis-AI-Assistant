"""Delivers reminders when they come due.

A background thread checks the database every few seconds and hands each due
reminder to `announce` (in voice mode that is spoken between conversations).
A reminder is marked delivered *before* it is announced, so it can never repeat.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime

from jarvis.memory.store import MemoryStore, Reminder
from jarvis.timeparse import format_due

logger = logging.getLogger(__name__)

LATE_AFTER_SECONDS = 120  # later than this counts as "missed while JARVIS was off"


def reminder_message(reminder: Reminder, now: float) -> str:
    if now - reminder.due_at > LATE_AFTER_SECONDS:
        due = datetime.fromtimestamp(reminder.due_at)
        when = format_due(due, datetime.fromtimestamp(now))
        return f"I missed a reminder from {when}: {reminder.text}."
    return f"Reminder: {reminder.text}."


class ReminderScheduler:
    def __init__(
        self,
        store: MemoryStore,
        announce: Callable[[str], None],
        clock: Callable[[], float] = time.time,
        interval: float = 5.0,
    ) -> None:
        self._store = store
        self._announce = announce
        self._clock = clock
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def tick(self) -> int:
        """Deliver everything that is due now. Returns how many reminders were delivered."""
        now = self._clock()
        delivered = 0
        for reminder in self._store.due_reminders(now):
            if not self._store.mark_fired(reminder.id):
                continue  # something else handled it first
            delivered += 1
            logger.info("Reminder %d due: %s", reminder.id, reminder.text)
            try:
                self._announce(reminder_message(reminder, now))
            except Exception:
                logger.exception("Could not announce reminder %d", reminder.id)
        return delivered

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="reminders", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _run(self) -> None:
        while True:
            try:
                self.tick()
            except Exception:
                logger.exception("Reminder check failed")
            if self._stop.wait(self._interval):
                return
