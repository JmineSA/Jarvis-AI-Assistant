"""End to end: real OllamaClient (mock server), real database, real tools and scheduler."""

import json
from datetime import datetime

import httpx

from jarvis.ai.ollama_client import OllamaClient
from jarvis.memory.store import MemoryStore
from jarvis.profile import Profile
from jarvis.reminders import ReminderScheduler
from jarvis.wiring import build_brain


def scripted_server(*tool_calls):
    """An Ollama that answers each request with the next scripted tool call."""
    queue = list(tool_calls)
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        name, args = queue.pop(0)
        call = {"function": {"name": name, "arguments": args}}
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "", "tool_calls": [call]}})

    client = OllamaClient("http://ollama.test", "qwen2.5:3b-instruct", transport=httpx.MockTransport(handler))
    return client, seen


async def test_remember_remind_recall_forget_and_fire(settings, tmp_path):
    store = MemoryStore(tmp_path / "smoke.db")
    client, seen = scripted_server(
        ("remember", {"text": "The LightsOn project is due on 1 December."}),
        ("set_reminder", {"text": "call the bank", "when": "in 20 minutes"}),
        ("recall", {"topic": "LightsOn"}),
        ("forget", {"about": "LightsOn"}),
    )
    settings.brain.fast_commands = False  # this test drives the model path
    brain = build_brain(settings, client, profile=Profile(owner_name="Lesiba"), store=store)

    assert (await brain.respond("remember the LightsOn project is due on 1 December")).startswith("I'll remember")
    assert "Reminder set for" in await brain.respond("remind me to call the bank in 20 minutes")
    assert "1 December" in await brain.respond("what do you know about LightsOn")

    async def approve(prompt):
        return True

    assert (await brain.respond("forget LightsOn", confirm=approve)).startswith("Forgotten")
    assert store.list_facts() == []

    # The saved fact reached the model on the following request (before it was forgotten).
    system_texts = [r["messages"][0]["content"] for r in seen]
    assert "LightsOn project is due" not in system_texts[0]
    assert "LightsOn project is due" in system_texts[2]

    # Time passes: the reminder is spoken exactly once.
    spoken = []
    later = datetime.now().timestamp() + 20 * 60 + 30
    scheduler = ReminderScheduler(store, spoken.append, clock=lambda: later)
    scheduler.tick()
    scheduler.tick()
    assert spoken == ["Reminder: call the bank."]

    # Conversations were stored too.
    assert store.counts().messages == 8
    await client.aclose()


def test_tests_never_touch_the_real_database(real_db_unchanged):
    """Your own data/jarvis.db may exist; the tests just must not create or change it."""
    same, before, now = real_db_unchanged()
    assert same, f"a test created or changed data/jarvis.db (before {before}, now {now})"
