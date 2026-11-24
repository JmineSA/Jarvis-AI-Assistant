"""Tests for the remember / recall / forget tools (Phase 4)."""

import pytest

from jarvis.memory.db import MemoryDatabase
from jarvis.memory.store import MemoryStore
from jarvis.tools.memory import (
    ForgetArgs,
    RecallArgs,
    RememberArgs,
    make_forget,
    make_recall,
    make_remember,
)


@pytest.fixture
def store(tmp_path):
    database = MemoryDatabase(tmp_path / "memory.db")
    memory = MemoryStore(database, recent_turns=0)
    yield memory
    database.close()


def test_remember_stores_a_fact(store):
    result = make_remember(store)(RememberArgs(fact="My dog is called Rex"))
    assert result.status.value == "done"
    assert store.stats().facts == 1


def test_remember_with_a_key_updates_the_same_fact(store):
    remember = make_remember(store)
    remember(RememberArgs(fact="My dog is called Rex", key="user.dog.name"))
    remember(RememberArgs(fact="My dog is called Rexy", key="user.dog.name"))
    assert store.stats().facts == 1
    assert store.all_facts()[0].content == "My dog is called Rexy"


def test_recall_finds_a_stored_fact(store):
    make_remember(store)(RememberArgs(fact="The LightsOn project lives in D:\\Projects"))
    result = make_recall(store)(RecallArgs(query="LightsOn"))
    assert "LightsOn" in result.message


def test_recall_says_so_when_nothing_matches(store):
    result = make_recall(store)(RecallArgs(query="zebra giraffe"))
    assert "don't have anything" in result.message


def test_forget_deletes_the_memory(store):
    memory_id, _ = store.remember_fact("temporary")
    result = make_forget(store)(ForgetArgs(id=memory_id))
    assert result.status.value == "done"
    assert store.get(memory_id) is None


def test_forget_reports_an_unknown_id(store):
    result = make_forget(store)(ForgetArgs(id=4242))
    assert result.status.value == "failed"


def test_the_tools_are_registered_when_memory_is_on(settings):
    from jarvis.profile import Profile
    from jarvis.wiring import build_registry_for

    store = MemoryStore(MemoryDatabase(":memory:"))
    registry = build_registry_for(settings, Profile(), store)
    names = registry.names()
    assert "remember" in names
    assert "recall" in names
    assert "forget" in names
    # The tool schemas the model sees must be well formed.
    schema = next(s for s in registry.schemas() if s["function"]["name"] == "remember")
    assert schema["function"]["parameters"]["required"] == ["fact"]


def test_the_memory_tools_are_absent_without_memory(settings):
    from jarvis.profile import Profile
    from jarvis.wiring import build_registry_for

    registry = build_registry_for(settings, Profile(), None)
    assert "remember" not in registry.names()
