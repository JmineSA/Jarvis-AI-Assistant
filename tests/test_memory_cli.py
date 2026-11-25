import pytest

from jarvis.memory.cli import run_memory_cli
from jarvis.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "j.db")


def run(store, *args, answer="no"):
    lines = []
    code = run_memory_cli(store, list(args), out=lines.append, ask=lambda prompt: answer)
    return code, "\n".join(lines)


def test_default_shows_everything(store):
    store.add_fact("I like tea")
    store.add_reminder("call the bank", 4_102_444_800)  # year 2100
    store.add_message("s", "user", "hi")
    code, text = run(store)
    assert code == 0
    assert "I like tea" in text and "call the bank" in text and "1 messages stored" in text
    assert str(store.path) in text


def test_empty_store_is_explained(store):
    _, text = run(store, "list")
    assert "(nothing saved)" in text and "(none)" in text


def test_search_history(store):
    store.add_message("s", "user", "my solar battery")
    assert "solar battery" in run(store, "search", "battery")[1]
    assert run(store, "search", "dragons")[1] == "Nothing found."
    assert run(store, "search")[0] == 2


def test_forget_and_cancel_by_id(store):
    fact, _ = store.add_fact("temporary")
    reminder = store.add_reminder("r", 4_102_444_800)
    assert run(store, "forget", str(fact.id))[0] == 0
    assert run(store, "cancel", str(reminder.id))[0] == 0
    assert store.list_facts() == [] and store.pending_reminders() == []
    assert run(store, "forget", "99")[0] == 1
    assert run(store, "forget", "abc")[0] == 2


def test_wipe_history_keeps_facts(store):
    store.add_fact("keep me")
    store.add_message("s", "user", "x")
    assert run(store, "wipe-history")[0] == 0
    assert store.counts().messages == 0 and len(store.list_facts()) == 1


def test_wipe_requires_typing_delete(store):
    store.add_fact("keep me")
    code, text = run(store, "wipe", answer="yes")
    assert code == 1 and "Nothing was deleted" in text and len(store.list_facts()) == 1
    code, _ = run(store, "wipe", answer="DELETE")
    assert code == 0 and store.list_facts() == []


def test_unknown_command(store):
    code, text = run(store, "dance")
    assert code == 2 and "Unknown memory command" in text
