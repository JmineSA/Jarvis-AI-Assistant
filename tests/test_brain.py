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


def test_a_reply_cut_off_by_the_length_cap_ends_at_a_full_sentence():
    from jarvis.ai.brain import finish_sentence

    assert finish_sentence("Pretoria is warm today. Expect around 28 degrees, and a chance of") == "Pretoria is warm today."
    assert finish_sentence("All good, thanks!") == "All good, thanks!"
    assert finish_sentence("Short cut off") == "Short cut off"  # nothing complete to fall back to
    assert finish_sentence("") == ""


def test_persona_sounds_like_a_person_and_knows_about_email_only_when_set_up():
    from jarvis.ai.prompts import build_tool_system_prompt

    plain = build_tool_system_prompt("Lesiba")
    assert "contractions" in plain and "check_email" not in plain
    with_mail = build_tool_system_prompt("Lesiba", email=True)
    assert "check_email" in with_mail and "read web pages or emails" not in with_mail


async def test_past_messages_are_resent_exactly_as_they_were_sent(fake_client_cls):
    """Same clock note on past turns keeps the start of each request identical, so Ollama can reuse its cache."""
    client = fake_client_cls()
    notes = iter(["It is 10:01.", "It is 10:02.", "It is 10:03."])
    brain = Brain(client, "sys", 4, turn_note=lambda: next(notes))
    await brain.respond("first")
    await brain.respond("second")
    await brain.respond("third")
    sent = client.calls[-1]
    users = [m["content"] for m in sent if m["role"] == "user"]
    assert users == ["first\n\n(It is 10:01.)", "second\n\n(It is 10:02.)", "third\n\n(It is 10:03.)"]
    assert brain.history[0] == {"role": "user", "content": "first"}  # stored plainly
    for _ in range(5):
        await brain.respond("more")
    assert len(brain._sent) <= 4
    brain.reset()
    assert brain._sent == {}
