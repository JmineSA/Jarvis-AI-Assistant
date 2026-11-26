import threading
import time

from jarvis.memory.store import MemoryStore
from jarvis.reminders import ReminderScheduler, reminder_message


def make(tmp_path, now=10_000.0, announce=None):
    store = MemoryStore(tmp_path / "j.db")
    spoken = []
    scheduler = ReminderScheduler(
        store, announce or spoken.append, clock=lambda: now, interval=0.01
    )
    return store, scheduler, spoken


def test_due_reminder_is_announced_once(tmp_path):
    store, scheduler, spoken = make(tmp_path)
    store.add_reminder("call the bank", 9_990)
    assert scheduler.tick() == 1
    assert scheduler.tick() == 0
    assert spoken == ["Reminder: call the bank."]


def test_future_reminders_wait(tmp_path):
    store, scheduler, spoken = make(tmp_path)
    store.add_reminder("later", 20_000)
    assert scheduler.tick() == 0 and spoken == []
    assert len(store.pending_reminders()) == 1


def test_cancelled_reminder_is_not_announced(tmp_path):
    store, scheduler, spoken = make(tmp_path)
    reminder = store.add_reminder("nope", 9_990)
    store.cancel_reminder(reminder.id)
    scheduler.tick()
    assert spoken == []


def test_late_reminder_says_it_was_missed(tmp_path):
    store, scheduler, spoken = make(tmp_path, now=1_000_000.0)
    store.add_reminder("call the bank", 1_000_000.0 - 3 * 3600)
    scheduler.tick()
    assert spoken[0].startswith("I missed a reminder from") and spoken[0].endswith("call the bank.")


def test_just_late_reminder_is_still_a_normal_reminder(tmp_path):
    store, _, _ = make(tmp_path)
    reminder = store.add_reminder("x", 9_950)
    assert reminder_message(reminder, 10_000.0) == "Reminder: x."


def test_failed_announcement_is_not_repeated(tmp_path):
    def broken(text):
        raise RuntimeError("speaker unplugged")

    store, scheduler, _ = make(tmp_path, announce=broken)
    store.add_reminder("call the bank", 9_990)
    assert scheduler.tick() == 1  # marked delivered before announcing: at most once
    assert scheduler.tick() == 0


def test_background_thread_delivers_and_stops(tmp_path):
    store, scheduler, spoken = make(tmp_path)
    store.add_reminder("call the bank", 9_990)
    scheduler.start()
    scheduler.start()  # a second start is harmless
    deadline = time.time() + 2
    while not spoken and time.time() < deadline:
        time.sleep(0.01)
    scheduler.stop()
    assert spoken == ["Reminder: call the bank."]
    assert not any(t.name == "reminders" for t in threading.enumerate())
