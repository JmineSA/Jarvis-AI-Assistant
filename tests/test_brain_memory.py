import sqlite3
from datetime import datetime

import pytest

from jarvis.ai.brain import NO_TOOL_CHOSEN, Brain
from jarvis.ai.ollama_client import ToolCall
from jarvis.ai.prompts import build_tool_system_prompt
from jarvis.config import Settings
from jarvis.memory.store import MemoryStore
from jarvis.profile import Profile
from jarvis.wiring import build_brain, make_context, open_store


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "j.db")


# --------------------------------------------------------------------------- context


def test_context_has_saved_facts_oldest_first_and_no_clock(store):
    store.add_fact("First fact")
    store.add_fact("Second fact")
    text = make_context(store, "Lesiba", 20)()
    assert "16:45" not in text and "Current date" not in text  # keeps the prompt start unchanging
    assert "What you know about Lesiba" in text
    assert text.index("First fact") < text.index("Second fact")


def test_context_without_facts_is_empty(store):
    assert make_context(store, None, 20)() == ""


def test_clock_note_is_the_date_and_time():
    from jarvis.wiring import make_clock_note

    note = make_clock_note(clock=lambda: datetime(2026, 11, 24, 16, 45))()
    assert note == "It is Tuesday, 24 November 2026, 16:45."


async def test_the_time_goes_after_the_users_words_not_in_the_system_prompt(fake_client_cls, settings, store):
    client = fake_client_cls(["Hello."])
    brain = _full_brain(settings, client, store)
    await brain.respond("hello there")
    messages = client.calls[-1]
    assert "It is " not in messages[0]["content"]
    assert messages[-1]["content"].startswith("hello there") and "It is " in messages[-1]["content"]
    assert brain.history[0]["content"] == "hello there"  # history keeps only what was said


def test_context_respects_the_fact_limit(store):
    for i in range(5):
        store.add_fact(f"fact number {i}")
    text = make_context(store, None, 2)()
    assert "fact number 4" in text and "fact number 3" in text and "fact number 0" not in text
    assert "What you know" not in make_context(store, None, 0)()


# --------------------------------------------------------------------------- open_store


def test_open_store_disabled_gives_none(tmp_path):
    settings = Settings.model_validate({"memory": {"enabled": False, "database_path": str(tmp_path / "x.db")}})
    assert open_store(settings) is None
    assert not (tmp_path / "x.db").exists()


def test_open_store_creates_the_database(settings):
    assert open_store(settings) is not None
    assert settings.resolve_database_path().exists()


def test_open_store_unopenable_path_gives_none(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    settings = Settings.model_validate({"memory": {"database_path": str(blocker / "j.db")}})
    assert open_store(settings) is None


# --------------------------------------------------------------------------- the brain


def test_prompt_mentions_memory_only_when_it_exists():
    assert "cannot yet remember" in build_tool_system_prompt("Lesiba")
    assert "cannot yet remember" not in build_tool_system_prompt("Lesiba", memory=True)


async def test_context_reaches_the_model_every_turn(fake_client_cls, store):
    store.add_fact("My GPU is a GTX 970")
    client = fake_client_cls(["Noted."])
    brain = Brain(client, "BASE", context=make_context(store, "Lesiba", 20))
    await brain.respond("hello")
    system = client.calls[0][0]
    assert system["role"] == "system" and system["content"].startswith("BASE")
    assert "GTX 970" in system["content"]
    assert all("GTX 970" not in m["content"] for m in brain.history)  # not stored in the history


async def test_a_broken_context_does_not_break_the_turn(fake_client_cls):
    def boom():
        raise RuntimeError("nope")

    client = fake_client_cls(["Fine."])
    assert await Brain(client, "BASE", context=boom).respond("hi") == "Fine."
    assert client.calls[0][0]["content"] == "BASE"


async def test_turns_are_saved_to_long_term_memory(fake_client_cls, store):
    brain = Brain(fake_client_cls(["Good evening."]), "BASE", memory=store, session_id="s1")
    await brain.respond("hello there")
    saved = store.recent_messages(session_id="s1")
    assert [(m.role, m.content) for m in saved] == [("user", "hello there"), ("assistant", "Good evening.")]


async def test_a_failing_database_never_breaks_a_conversation(fake_client_cls, store, monkeypatch):
    def broken(*a, **k):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, "add_message", broken)
    brain = Brain(fake_client_cls(["Still here."]), "BASE", memory=store, session_id="s1")
    assert await brain.respond("hello") == "Still here."


async def test_a_failed_model_call_saves_nothing(fake_client_cls, store):
    from jarvis.ai.ollama_client import OllamaUnavailableError

    brain = Brain(fake_client_cls(error=OllamaUnavailableError("down")), "BASE", memory=store, session_id="s")
    with pytest.raises(OllamaUnavailableError):
        await brain.respond("hello")
    assert store.counts().messages == 0


def _full_brain(settings, client, store):
    settings.brain.fast_commands = False  # these tests exercise the model path
    return build_brain(settings, client, profile=Profile(owner_name="Lesiba"), store=store)


async def test_remember_end_to_end_through_the_brain(fake_client_cls, settings, store):
    client = fake_client_cls(tool_calls=[ToolCall("remember", {"text": "The project is due on Monday."})])
    brain = _full_brain(settings, client, store)
    reply = await brain.respond("remember that the project is due on Monday")
    assert reply == "I'll remember that: The project is due on Monday."
    assert [f.text for f in store.list_facts()] == ["The project is due on Monday."]


async def test_forget_end_to_end_asks_first(fake_client_cls, settings, store):
    store.add_fact("The project is due on Monday.")
    client = fake_client_cls(tool_calls=[ToolCall("forget", {"about": "project"})])
    brain = _full_brain(settings, client, store)
    prompts = []

    async def refuse(prompt):
        prompts.append(prompt)
        return False

    reply = await brain.respond("forget the project", confirm=refuse)
    assert "The project is due on Monday." in prompts[0]
    assert "won't" in reply and len(store.list_facts()) == 1


@pytest.mark.parametrize(
    "claim", ["I'll remember that.", "Noted.", "Sure, I've saved that.", "Reminder set for 5pm."]
)
async def test_claiming_to_remember_without_a_tool_is_caught(fake_client_cls, settings, store, claim):
    brain = _full_brain(settings, fake_client_cls([claim]), store)
    assert await brain.respond("remember my cat is called Tom") == NO_TOOL_CHOSEN
    assert store.list_facts() == []


async def test_ordinary_sentences_about_remembering_are_not_flagged(fake_client_cls, settings, store):
    reply = "Your cat is called Tom."
    assert await _full_brain(settings, fake_client_cls([reply]), store).respond("cat?") == reply


async def test_save_conversations_off_keeps_facts_but_not_chat(fake_client_cls, tmp_path, store):
    settings = Settings.model_validate(
        {"memory": {"database_path": str(tmp_path / "j.db"), "save_conversations": False}}
    )
    brain = _full_brain(settings, fake_client_cls(["Hello."]), store)
    await brain.respond("hello")
    assert store.counts().messages == 0

    remember = fake_client_cls(tool_calls=[ToolCall("remember", {"text": "I like tea"})])
    await _full_brain(settings, remember, store).respond("remember I like tea")
    assert len(store.list_facts()) == 1


def test_memory_tools_are_offered_only_with_a_store(fake_client_cls, settings, store):
    with_store = _full_brain(settings, fake_client_cls(), store)
    without = _full_brain(settings, fake_client_cls(), None)
    assert "remember" in with_store._executor.registry
    assert "remember" not in without._executor.registry
