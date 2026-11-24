"""Tests for the brain's use of memory (Phase 4)."""

import pytest

from jarvis.ai.brain import Brain
from jarvis.ai.prompts import MEMORY_CONTEXT_HEADER


class FakeMemory:
    """Stands in for the memory store."""

    def __init__(self, context="", fail=False):
        self.context = context
        self.fail = fail
        self.saved: list[tuple[str, str]] = []
        self.queries: list[str] = []

    def context_for(self, query: str) -> str:
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("the database is locked")
        return self.context

    def remember_turn(self, user_text: str, assistant_text: str) -> None:
        self.saved.append((user_text, assistant_text))


async def test_memory_is_injected_into_the_context(fake_client_cls):
    client = fake_client_cls(["Hello."])
    memory = FakeMemory("- [fact] user.timezone: SAST")
    brain = Brain(client, memory=memory)
    await brain.respond("what time is it")
    sent = client.calls[0]
    assert sent[1]["role"] == "system"
    assert MEMORY_CONTEXT_HEADER in sent[1]["content"]
    assert "user.timezone: SAST" in sent[1]["content"]


async def test_the_query_is_used_for_the_search(fake_client_cls):
    memory = FakeMemory("- [fact] something")
    brain = Brain(fake_client_cls(), memory=memory)
    await brain.respond("where is the project")
    assert memory.queries == ["where is the project"]


async def test_no_memory_block_when_nothing_was_found(fake_client_cls):
    client = fake_client_cls(["Hello."])
    brain = Brain(client, memory=FakeMemory(""))
    await brain.respond("hi")
    assert [message["role"] for message in client.calls[0]] == ["system", "user"]


async def test_a_failing_memory_does_not_stop_the_conversation(fake_client_cls):
    client = fake_client_cls(["Hello."])
    brain = Brain(client, memory=FakeMemory(fail=True))
    reply = await brain.respond("hi")
    assert reply == "Hello."
    assert [message["role"] for message in client.calls[0]] == ["system", "user"]


async def test_the_turn_is_saved_to_memory(fake_client_cls):
    memory = FakeMemory()
    brain = Brain(fake_client_cls(["Hi."]), memory=memory)
    await brain.respond("hello there")
    assert memory.saved == [("hello there", "Hi.")]


async def test_a_failed_turn_is_not_saved(fake_client_cls):
    from jarvis.ai.ollama_client import OllamaUnavailableError

    memory = FakeMemory()
    brain = Brain(fake_client_cls(error=OllamaUnavailableError("down")), memory=memory)
    with pytest.raises(OllamaUnavailableError):
        await brain.respond("hello")
    assert memory.saved == []


async def test_the_memory_block_sits_between_the_prompt_and_the_history(fake_client_cls):
    client = fake_client_cls(["One.", "Two."])
    brain = Brain(client, memory=FakeMemory("- [fact] a fact"))
    await brain.respond("first")
    await brain.respond("second")
    roles = [message["role"] for message in client.calls[1]]
    assert roles == ["system", "system", "user", "assistant", "user"]


async def test_has_memory_reports_the_state(fake_client_cls):
    assert Brain(fake_client_cls()).has_memory is False
    assert Brain(fake_client_cls(), memory=FakeMemory()).has_memory is True
