import logging

import pytest

from jarvis.ai.brain import MAX_TOOL_CALLS_PER_TURN, NO_TOOL_CHOSEN, Brain
from jarvis.ai.ollama_client import (
    OllamaResponseError,
    OllamaUnavailableError,
    ToolCall,
)
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
    assert names == ["web_search", "maps", "compose_email", "play_media", "media_control", "system_status", "current_time"]


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


# ------------------------------------------------------------------ fast path


class RecordingRouter:
    def __init__(self, call):
        self.call = call
        self.seen = []

    def __call__(self, text):
        self.seen.append(text)
        return self.call


async def test_a_routed_command_never_calls_the_model(fake_client_cls):
    client = fake_client_cls(["should not be used"])
    registry = build_registry(Profile())
    brain = Brain(client, "system", executor=ToolExecutor(registry), router=RecordingRouter(ToolCall("current_time", {})))
    reply = await brain.respond("what time is it")
    assert reply.startswith("It's ") and client.calls == []
    assert brain.last_route == "fast"
    assert [m["role"] for m in brain.history] == ["user", "assistant"]


async def test_a_command_that_is_not_routed_goes_to_the_model(fake_client_cls):
    client = fake_client_cls(["Hello."])
    brain = Brain(client, "system", executor=ToolExecutor(build_registry(Profile())), router=RecordingRouter(None))
    assert await brain.respond("hi") == "Hello."
    assert brain.last_route == "model" and len(client.calls) == 1


async def test_a_routed_risky_tool_still_asks_for_confirmation(fake_client_cls):
    client = fake_client_cls(["unused"])
    brain = Brain(
        client, "system",
        executor=ToolExecutor(build_registry(Profile()), always_confirm=["current_time"]),
        router=RecordingRouter(ToolCall("current_time", {})),
    )

    async def refuse(prompt):
        return False

    reply = await brain.respond("time", confirm=refuse)
    assert "won't" in reply or "not" in reply.lower()


async def test_warm_up_uses_the_fixed_prompt_and_tools():
    calls = []

    class Warmable:
        async def warm_up(self, messages, tools=None):
            calls.append((messages, tools))

        async def chat(self, messages):
            return "x"

    brain = Brain(Warmable(), "SYSTEM", executor=ToolExecutor(build_registry(Profile())))
    await brain.warm_up()
    messages, tools = calls[0]
    assert messages[0] == {"role": "system", "content": "SYSTEM"} and tools


async def test_warm_up_is_harmless_with_a_client_that_cannot_warm_up(fake_client_cls):
    await Brain(fake_client_cls(), "SYSTEM").warm_up()  # no error


# ------------------------------------------------------------------ "which one?" follow-ups
from dataclasses import dataclass  # noqa: E402

from jarvis.ai.brain import pick_choice  # noqa: E402
from jarvis.tools.registry import Tool, ToolArgs, ToolRegistry, ToolResult, ToolStatus  # noqa: E402


class _PickArgs(ToolArgs):
    name: str = ""


def _asking_brain(fake_client_cls, played):
    def run(args):
        if args.name == "gym":
            return ToolResult(
                ToolStatus.FAILED,
                "That matches 2 things: 1. Gym Mix One; 2. Gym Mix Two. Which one?",
                (("Gym Mix One", {"name": "Gym Mix One"}), ("Gym Mix Two", {"name": "Gym Mix Two"})),
            )
        played.append(args.name)
        return ToolResult.launched(f"Playing {args.name}.")

    registry = ToolRegistry()
    registry.register(Tool("play_media", "play", {"type": "object", "properties": {}}, _PickArgs, run))
    client = fake_client_cls(["model reply"])
    return Brain(client, "system", executor=ToolExecutor(registry), router=RecordingRouter(ToolCall("play_media", {"name": "gym"}))), client


@pytest.mark.parametrize("answer,expected", [("the second one", "Gym Mix Two"), ("2", "Gym Mix Two"), ("first", "Gym Mix One"), ("mix one", "Gym Mix One")])
async def test_which_one_can_be_answered_next_turn(fake_client_cls, answer, expected):
    played = []
    brain, client = _asking_brain(fake_client_cls, played)
    assert "Which one?" in await brain.respond("play gym")
    brain._router = RecordingRouter(None)
    assert await brain.respond(answer) == f"Playing {expected}."
    assert played == [expected] and client.calls == []


async def test_which_one_can_be_cancelled_or_ignored(fake_client_cls):
    played = []
    brain, client = _asking_brain(fake_client_cls, played)
    await brain.respond("play gym")
    assert await brain.respond("never mind") == "Okay, cancelled."
    await brain.respond("play gym")
    brain._router = RecordingRouter(None)
    assert await brain.respond("how are you") == "model reply"  # unrelated: normal path
    assert played == []
    assert await brain.respond("the second one") == "model reply"  # question is no longer pending


def test_pick_choice_rules():
    opts = (("A", {"name": "Alpha Song"}), ("B", {"name": "Beta Song"}))
    assert pick_choice("number two", opts) == 1
    assert pick_choice("alpha", opts) == 0
    assert pick_choice("song", opts) is None  # matches both
    assert pick_choice("something else entirely", opts) is None
