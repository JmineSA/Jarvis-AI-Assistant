import pytest

from jarvis.ai.brain import Brain
from jarvis.ai.ollama_client import OllamaUnavailableError
from jarvis.ai.prompts import SYSTEM_PROMPT


async def test_system_prompt_is_sent_first(fake_client_cls):
    client = fake_client_cls(["Hello."])
    brain = Brain(client)
    await brain.respond("Hi")
    first_call = client.calls[0]
    assert first_call[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert first_call[-1] == {"role": "user", "content": "Hi"}


async def test_context_is_carried_between_turns(fake_client_cls):
    client = fake_client_cls(["One.", "Two."])
    brain = Brain(client)
    await brain.respond("first")
    await brain.respond("second")
    roles = [m["role"] for m in client.calls[1]]
    assert roles == ["system", "user", "assistant", "user"]
    assert brain.history_length == 4


async def test_history_is_trimmed_to_max_turns(fake_client_cls):
    client = fake_client_cls(["ok"])
    brain = Brain(client, max_history_turns=2)
    for i in range(5):
        await brain.respond(f"message {i}")
    history = brain.history
    assert len(history) == 4
    assert history[0] == {"role": "user", "content": "message 3"}
    assert history[-1]["role"] == "assistant"


async def test_failed_turn_does_not_pollute_history(fake_client_cls):
    client = fake_client_cls(error=OllamaUnavailableError("down"))
    brain = Brain(client)
    with pytest.raises(OllamaUnavailableError):
        await brain.respond("hello")
    assert brain.history_length == 0


async def test_blank_input_is_rejected(fake_client_cls):
    brain = Brain(fake_client_cls())
    with pytest.raises(ValueError):
        await brain.respond("   ")


async def test_reset_clears_history(fake_client_cls):
    brain = Brain(fake_client_cls())
    await brain.respond("hello")
    brain.reset()
    assert brain.history_length == 0


def test_invalid_history_size_rejected(fake_client_cls):
    with pytest.raises(ValueError):
        Brain(fake_client_cls(), max_history_turns=0)
