"""Tests for the SQLite memory and its retrieval (Phase 4)."""

import pytest

from jarvis.memory.db import MemoryDatabase
from jarvis.memory.store import MemoryStore, slugify, tokenize


@pytest.fixture
def db(tmp_path):
    database = MemoryDatabase(tmp_path / "memory.db")
    yield database
    database.close()


@pytest.fixture
def store(db):
    return MemoryStore(db, session_id="test", recent_turns=0)


# --------------------------------------------------------------------- basics


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("Open the VS Code app, please!") == ["open", "vs", "code", "app"]


def test_tokenize_keeps_numbers():
    assert "970" in tokenize("the GTX 970 has 4 GB")


def test_slugify_makes_a_short_key():
    assert slugify("My dog is called Rex") == "my-dog-is-called-rex"


def test_database_creates_its_file(tmp_path):
    path = tmp_path / "nested" / "memory.db"
    database = MemoryDatabase(path)
    try:
        assert path.exists()
        assert database.is_writable()
    finally:
        database.close()


def test_messages_are_stored(db, store):
    store.remember_turn("hello there", "Good to meet you.")
    stats = store.stats()
    assert stats.messages == 2
    assert stats.facts == 0


def test_facts_are_stored_and_updated(store):
    memory_id, created = store.remember_fact("My dog is called Rex", "user.dog.name")
    assert created is True
    again, created_again = store.remember_fact("My dog is called Rexy", "user.dog.name")
    assert created_again is False
    assert again == memory_id
    assert store.get(memory_id).content == "My dog is called Rexy"


def test_facts_get_a_key_when_none_is_given(store):
    store.remember_fact("I work on the LightsOn project")
    fact = store.all_facts()[0]
    assert fact.key == "i-work-on-the-lightson-project"


def test_empty_fact_is_rejected(store):
    with pytest.raises(ValueError):
        store.remember_fact("   ")


def test_recent_messages_are_newest_first(store):
    store.remember_message("user", "first")
    store.remember_message("user", "second")
    recent = store.recent_messages(limit=5)
    assert recent[0].content == "second"
    assert recent[1].content == "first"


# ------------------------------------------------------------------ retrieval


def test_search_finds_a_matching_message(store):
    store.remember_message("user", "where is the LightsOn project folder")
    hits = store.search("LightsOn")
    assert hits
    assert "LightsOn" in hits[0].content


def test_search_ignores_unrelated_words(store):
    store.remember_message("user", "the dog is called Rex")
    assert store.search("quantum chromodynamics") == []


def test_search_prefers_the_more_relevant_message(store):
    store.remember_message("user", "python programming code editor")
    store.remember_message("user", "python is a programming language")
    hits = store.search("python programming")
    assert hits
    assert hits[0].score >= hits[-1].score


def test_facts_outrank_messages(store):
    store.remember_message("user", "a long chat about many unrelated things here")
    store.remember_fact("The LightsOn project lives in D:\\Projects\\LightsOn", "lightson.path")
    hits = store.search("LightsOn project")
    assert hits[0].kind == "fact"


def test_forget_deletes_a_fact(store):
    memory_id, _ = store.remember_fact("something to forget")
    assert store.forget(memory_id) is True
    assert store.get(memory_id) is None


def test_forget_deletes_a_message(store):
    memory_id = store.remember_message("user", "delete me")
    assert store.forget(memory_id) is True
    assert store.get(memory_id) is None


def test_forget_of_an_unknown_id_does_nothing(store):
    assert store.forget(9999) is False


def test_clear_session_keeps_the_facts(store):
    store.remember_fact("a fact worth keeping")
    store.remember_message("user", "a message from this session")
    deleted = store.clear_session()
    assert deleted == 1
    assert store.stats().messages == 0
    assert store.stats().facts == 1


def test_clear_all_removes_everything(store):
    store.remember_fact("a fact")
    store.remember_message("user", "a message")
    assert store.clear_all() == 2
    assert store.stats().messages == 0
    assert store.stats().facts == 0


# -------------------------------------------------------------------- context


def test_context_includes_the_matching_memory(store):
    store.remember_message("user", "the deployment runs on Friday")
    context = store.context_for("when does the deployment run")
    assert "deployment" in context


def test_context_is_empty_when_nothing_matches(store):
    store.remember_message("user", "the deployment runs on Friday")
    assert store.context_for("zebra giraffe elephant") == ""


def test_context_respects_the_character_budget(db):
    store = MemoryStore(db, recent_turns=0, max_context_chars=200)
    for index in range(20):
        store.remember_message("user", f"a long memory number {index} about deployments")
    context = store.context_for("deployment")
    assert len(context) <= 200


def test_recent_turns_are_added_even_without_a_match(db):
    store = MemoryStore(db, recent_turns=2, min_score=99.0)
    store.remember_message("user", "something I said a moment ago")
    context = store.context_for("completely unrelated topic")
    assert "something I said a moment ago" in context


def test_context_marks_facts_and_messages_differently(store):
    store.remember_fact("The timezone is SAST", "user.timezone")
    context = store.context_for("timezone")
    assert "[fact]" in context
    assert "user.timezone" in context


def test_stats_report_the_sizes(store):
    store.remember_turn("hello", "hi there")
    store.remember_fact("a fact")
    stats = store.stats()
    assert stats.messages == 2
    assert stats.facts == 1
    assert stats.oldest is not None
    assert stats.newest is not None
