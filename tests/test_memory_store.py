import sqlite3
import threading

import pytest

from jarvis.memory.database import MIGRATIONS, Database
from jarvis.memory.store import (
    MAX_FACT_LENGTH,
    MAX_MESSAGE_LENGTH,
    MemoryStore,
    keywords,
)


class Clock:
    """Each call moves one second forward, so ordering is predictable."""

    def __init__(self, start=1_000_000.0):
        self.now = start

    def __call__(self):
        self.now += 1
        return self.now


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "data" / "jarvis.db", clock=Clock())


# --------------------------------------------------------------------------- database


def test_creates_file_and_parent_folder_at_the_current_schema(tmp_path):
    path = tmp_path / "deep" / "dir" / "j.db"
    Database(path)
    assert path.exists()
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)


def test_reopening_keeps_data_and_does_not_redo_migrations(tmp_path):
    path = tmp_path / "j.db"
    MemoryStore(path).add_fact("The sky is blue")
    reopened = MemoryStore(path)
    assert [f.text for f in reopened.list_facts()] == ["The sky is blue"]


def test_database_from_a_newer_jarvis_is_refused(tmp_path):
    path = tmp_path / "j.db"
    Database(path)
    with sqlite3.connect(path) as conn:
        conn.execute(f"PRAGMA user_version = {len(MIGRATIONS) + 5}")
    with pytest.raises(sqlite3.DatabaseError, match="newer JARVIS"):
        Database(path)


def test_unusable_location_raises_an_os_or_sqlite_error(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises((OSError, sqlite3.Error)):
        MemoryStore(blocker / "j.db")


# --------------------------------------------------------------------------- keywords


def test_keywords_drop_filler_short_words_and_duplicates():
    assert keywords("What did I say about the Solar solar battery?") == ["solar", "battery"]
    assert keywords("what do you know about me") == []
    assert keywords("") == []


# --------------------------------------------------------------------------- messages


def test_messages_come_back_oldest_first_and_respect_limit(store):
    for i in range(5):
        store.add_message("s1", "user", f"message {i}")
    recent = store.recent_messages(limit=3)
    assert [m.content for m in recent] == ["message 2", "message 3", "message 4"]


def test_recent_messages_can_be_limited_to_one_session(store):
    store.add_message("a", "user", "from a")
    store.add_message("b", "user", "from b")
    assert [m.content for m in store.recent_messages(session_id="b")] == ["from b"]


def test_very_long_messages_are_cut(store):
    store.add_message("s", "user", "x" * (MAX_MESSAGE_LENGTH + 500))
    assert len(store.recent_messages()[0].content) == MAX_MESSAGE_LENGTH


def test_search_finds_messages_by_meaningful_words(store):
    store.add_message("old", "user", "My solar battery is 10 kWh")
    store.add_message("old", "assistant", "Noted the battery size")
    store.add_message("old", "user", "Open Chrome")
    found = store.search_messages("what is my battery size")
    assert {m.content for m in found} == {"My solar battery is 10 kWh", "Noted the battery size"}


def test_search_ranks_more_matching_words_first_then_newer(store):
    store.add_message("s", "user", "battery")
    store.add_message("s", "user", "solar battery sizing")
    store.add_message("s", "user", "battery again")
    found = store.search_messages("solar battery")
    assert found[0].content == "solar battery sizing"
    assert found[1].content == "battery again"  # newer of the single-word matches


def test_search_can_skip_the_current_session(store):
    store.add_message("now", "user", "battery question")
    store.add_message("old", "user", "battery answer")
    found = store.search_messages("battery", exclude_session="now")
    assert [m.content for m in found] == ["battery answer"]


def test_search_with_only_filler_words_finds_nothing(store):
    store.add_message("s", "user", "what do you know about me")
    assert store.search_messages("what about you") == []


def test_clear_messages_reports_how_many_were_deleted(store):
    store.add_message("s", "user", "a")
    store.add_message("s", "user", "b")
    assert store.clear_messages() == 2
    assert store.recent_messages() == []


# --------------------------------------------------------------------------- facts


def test_add_and_list_facts_newest_first(store):
    first, created = store.add_fact("The LightsOn project is due on 1 December.")
    store.add_fact("I prefer short answers")
    assert created is True and first.id == 1
    assert [f.text for f in store.list_facts()] == [
        "I prefer short answers",
        "The LightsOn project is due on 1 December.",
    ]
    assert len(store.list_facts(limit=1)) == 1


def test_duplicate_fact_is_not_saved_twice_even_if_worded_with_different_case_or_punctuation(store):
    original, _ = store.add_fact("My GPU is a GTX 970")
    again, created = store.add_fact("my gpu is a gtx 970!!")
    assert created is False and again.id == original.id
    assert len(store.list_facts()) == 1


def test_fact_without_words_is_rejected(store):
    with pytest.raises(ValueError):
        store.add_fact("  ...  ")


def test_long_fact_is_cut_and_whitespace_tidied(store):
    fact, _ = store.add_fact("word   " * 200)
    assert len(fact.text) <= MAX_FACT_LENGTH
    assert "  " not in fact.text


def test_match_facts_ranks_by_shared_words(store):
    store.add_fact("The solar project is due Monday")
    store.add_fact("The solar battery is 10 kWh")
    store.add_fact("I like tea")
    scored = store.match_facts("solar project")
    assert [(score, f.text) for score, f in scored] == [
        (2, "The solar project is due Monday"),
        (1, "The solar battery is 10 kWh"),
    ]


def test_delete_fact(store):
    fact, _ = store.add_fact("Temporary")
    assert store.delete_fact(fact.id) is True
    assert store.delete_fact(fact.id) is False
    assert store.list_facts() == []


# --------------------------------------------------------------------------- reminders


def test_pending_reminders_are_ordered_by_due_time(store):
    store.add_reminder("later", 2000)
    store.add_reminder("sooner", 1000)
    assert [r.text for r in store.pending_reminders()] == ["sooner", "later"]


def test_due_reminders_only_include_those_already_due(store):
    store.add_reminder("past", 1000)
    store.add_reminder("future", 5000)
    assert [r.text for r in store.due_reminders(now=2000)] == ["past"]


def test_a_reminder_can_only_be_marked_delivered_once(store):
    reminder = store.add_reminder("call the bank", 1000)
    assert store.mark_fired(reminder.id) is True
    assert store.mark_fired(reminder.id) is False
    assert store.due_reminders(now=9999) == []
    assert store.pending_reminders() == []


def test_cancelled_reminders_never_come_due(store):
    reminder = store.add_reminder("call the bank", 1000)
    assert store.cancel_reminder(reminder.id) is True
    assert store.cancel_reminder(reminder.id) is False
    assert store.due_reminders(now=9999) == []


def test_reminder_needs_text(store):
    with pytest.raises(ValueError):
        store.add_reminder("   ", 1000)


def test_match_reminders_only_looks_at_pending_ones(store):
    done = store.add_reminder("call the bank", 1000)
    store.add_reminder("call mom", 2000)
    store.mark_fired(done.id)
    assert [r.text for _, r in store.match_reminders("bank")] == []  # the delivered one is not offered
    assert [r.text for _, r in store.match_reminders("call mom")] == ["call mom"]


# --------------------------------------------------------------------------- housekeeping


def test_counts_oldest_message_and_wipe(store):
    assert store.oldest_message_time() is None
    store.add_message("s", "user", "hello")
    store.add_fact("A fact")
    store.add_reminder("r", 1000)
    done = store.add_reminder("done", 500)
    store.mark_fired(done.id)

    counts = store.counts()
    assert (counts.messages, counts.facts, counts.pending_reminders) == (1, 1, 1)
    assert store.oldest_message_time() is not None

    store.wipe()
    counts = store.counts()
    assert (counts.messages, counts.facts, counts.pending_reminders) == (0, 0, 0)
    store.add_fact("works again after a wipe")


def test_safe_to_use_from_many_threads_at_once(store):
    def worker(n):
        for i in range(25):
            store.add_message(f"t{n}", "user", f"message {n}-{i}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.counts().messages == 200


def test_reminders_keep_the_important_flag_and_old_databases_upgrade(tmp_path):
    import sqlite3

    from jarvis.memory.database import MIGRATIONS
    from jarvis.memory.store import MemoryStore

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(MIGRATIONS[0])
    conn.execute("PRAGMA user_version = 1")
    conn.execute("INSERT INTO reminders (text, due_at, created_at) VALUES ('old one', 1.0, 0.0)")
    conn.commit()
    conn.close()
    store = MemoryStore(path)
    store.add_reminder("big", 2.0, important=True)
    flags = {r.text: r.important for r in store.pending_reminders()}
    assert flags == {"old one": False, "big": True}
