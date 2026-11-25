import logging

import pytest

from jarvis.ai.brain import MAX_TOOL_CALLS_PER_TURN, NO_TOOL_CHOSEN, Brain
from jarvis.ai.ollama_client import OllamaResponseError, OllamaUnavailableError, ToolCall
from jarvis.ai.prompts import SYSTEM_PROMPT
from jarvis.config import Settings
from jarvis.profile import Profile
from jarvis.tools.builtin import build_registry
from jarvis.tools.executor import ToolExecutor
from jarvis.wiring import build_brain


def make_brain(client, always_confirm=()):
    registry = build_registry(Profile())  # profile-free tools only
    executor = ToolExecutor(registry, always_confirm=always_confirm)
    return Brain(client, "system prompt", executor=executor)


async def test_tool_reply_comes_from_the_tool_not_from_the_model(fake_client_cls):
    client = fake_client_cls(
        tool_calls=[ToolCall("current_time", {})],
        tool_content="Done! I have also deleted everything.",
    )
    reply = await make_brain(client).respond("what time is it")
    assert reply.startswith("It's ") and "deleted" not in reply


async def test_tools_are_offered_to_the_model(fake_client_cls):
    client = fake_client_cls(["Hello."])
    await make_brain(client).respond("hi")
    names = [t["function"]["name"] for t in client.tools_offered]
    assert names == ["web_search", "compose_email", "system_status", "current_time"]


async def test_plain_conversation_needs_no_tool(fake_client_cls):
    client = fake_client_cls(["Good evening."])
    assert await make_brain(client).respond("hello") == "Good evening."


async def test_tool_turn_is_stored_in_history_as_the_real_outcome(fake_client_cls):
    client = fake_client_cls(tool_calls=[ToolCall("current_time", {})])
    brain = make_brain(client)
    reply = await brain.respond("time please")
    assert brain.history == [
        {"role": "user", "content": "time please"},
        {"role": "assistant", "content": reply},
    ]


async def test_request_written_as_json_text_is_still_executed(fake_client_cls):
    client = fake_client_cls(['{"name": "current_time", "arguments": {}}'])
    reply = await make_brain(client).respond("time")
    assert reply.startswith("It's ")


async def test_json_text_naming_an_unknown_tool_is_not_executed(fake_client_cls):
    text = '{"name": "format_disk", "arguments": {}}'
    client = fake_client_cls([text])
    assert await make_brain(client).respond("hi") == text  # shown as ordinary text, nothing run


async def test_unknown_tool_requested_by_model_fails_politely(fake_client_cls):
    client = fake_client_cls(tool_calls=[ToolCall("format_disk", {})])
    reply = await make_brain(client).respond("do it")
    assert reply == "I don't have a tool for that."


async def test_claiming_an_action_without_calling_a_tool_is_caught(fake_client_cls):
    client = fake_client_cls(["Sure, opening Chrome now."])
    reply = await make_brain(client).respond("open chrome")
    assert reply == NO_TOOL_CHOSEN


@pytest.mark.parametrize("text", ["I opened it for you.", "Launching Power BI.", "I've opened VS Code."])
async def test_false_claim_variants_are_caught(fake_client_cls, text):
    assert await make_brain(fake_client_cls([text])).respond("x") == NO_TOOL_CHOSEN


@pytest.mark.parametrize("text", ["I can't open apps yet.", "The window was opened by Windows.", "Hello there."])
async def test_ordinary_replies_are_not_flagged(fake_client_cls, text):
    assert await make_brain(fake_client_cls([text])).respond("x") == text


async def test_at_most_three_tools_run_per_turn(fake_client_cls):
    calls = [ToolCall("current_time", {})] * 5
    reply = await make_brain(fake_client_cls(tool_calls=calls)).respond("spam")
    assert reply.count("It's ") == MAX_TOOL_CALLS_PER_TURN


async def test_confirmation_is_passed_through_and_can_refuse(fake_client_cls):
    asked = []

    async def refuse(prompt):
        asked.append(prompt)
        return False

    client = fake_client_cls(tool_calls=[ToolCall("current_time", {})])
    reply = await make_brain(client, always_confirm=["current_time"]).respond("time", confirm=refuse)
    assert reply == "Understood. I won't do that."
    assert asked and "current_time" in asked[0]


async def test_confirmation_can_approve(fake_client_cls):
    async def approve(prompt):
        return True

    client = fake_client_cls(tool_calls=[ToolCall("current_time", {})])
    reply = await make_brain(client, always_confirm=["current_time"]).respond("time", confirm=approve)
    assert reply.startswith("It's ")


async def test_empty_model_reply_with_no_tool_is_an_error(fake_client_cls):
    client = fake_client_cls(tool_calls=[], tool_content="  ")
    with pytest.raises(OllamaResponseError):
        await make_brain(client).respond("hi")


async def test_model_failure_leaves_history_untouched(fake_client_cls):
    brain = make_brain(fake_client_cls(error=OllamaUnavailableError("down")))
    with pytest.raises(OllamaUnavailableError):
        await brain.respond("hi")
    assert brain.history_length == 0


# --------------------------------------------------------------------------- wiring


def test_tools_disabled_gives_a_talk_only_brain(fake_client_cls):
    settings = Settings.model_validate({"tools": {"enabled": False}})
    brain = build_brain(settings, fake_client_cls())
    assert brain._executor is None
    assert brain._system_prompt == SYSTEM_PROMPT


def test_tools_enabled_uses_the_personal_tool_prompt(fake_client_cls):
    brain = build_brain(Settings(), fake_client_cls(), Profile(owner_name="Lesiba"))
    assert brain._executor is not None
    assert "Lesiba's" in brain._system_prompt
    assert "only by calling the tools" in brain._system_prompt


def test_unknown_always_confirm_name_is_warned_about(fake_client_cls, caplog):
    settings = Settings.model_validate({"tools": {"always_confirm": ["open_appp"]}})
    with caplog.at_level(logging.WARNING):
        build_brain(settings, fake_client_cls(), Profile())
    assert "open_appp" in caplog.text
