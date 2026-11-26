from datetime import datetime

import pytest

from jarvis.memory.store import MemoryStore
from jarvis.profile import Profile
from jarvis.tools.builtin import build_registry
from jarvis.tools.executor import ToolExecutor
from jarvis.tools.registry import ToolStatus

NOW = datetime(2026, 11, 24, 16, 45)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "j.db")


@pytest.fixture
def executor(store):
    registry = build_registry(Profile(), clock=lambda: NOW, store=store, session_id="now")
    return ToolExecutor(registry)


async def yes(prompt):
    yes.prompts.append(prompt)
    return True


yes.prompts = []


async def no(prompt):
    return False


def test_memory_tools_exist_only_with_a_store(store):
    without = build_registry(Profile(), clock=lambda: NOW)
    with_store = build_registry(Profile(), clock=lambda: NOW, store=store)
    for name in ("remember", "forget", "recall", "set_reminder", "list_reminders", "cancel_reminder"):
        assert name not in without
        assert name in with_store


async def test_remember_saves_and_is_idempotent(executor, store):
    first = await executor.run("remember", {"text": "The LightsOn project is due on 1 December."})
    assert first.status is ToolStatus.DONE and first.message.startswith("I'll remember that")
    again = await executor.run("remember", {"text": "the lightson project is due on 1 december"})
    assert again.message.startswith("I already knew that")
    assert len(store.list_facts()) == 1


async def test_remember_nothing_fails(executor):
    result = await executor.run("remember", {"text": "  ... "})
    assert result.status is ToolStatus.FAILED


async def test_forget_needs_confirmation_and_names_the_fact(executor, store):
    store.add_fact("My favourite editor is VS Code")
    yes.prompts.clear()
    result = await executor.run("forget", {"about": "editor"}, confirm=yes)
    assert result.status is ToolStatus.DONE and store.list_facts() == []
    assert "My favourite editor is VS Code" in yes.prompts[0]


async def test_forget_refused_deletes_nothing(executor, store):
    store.add_fact("My favourite editor is VS Code")
    result = await executor.run("forget", {"about": "editor"}, confirm=no)
    assert result.status is ToolStatus.DENIED and len(store.list_facts()) == 1


async def test_forget_without_confirm_callback_is_denied(executor, store):
    store.add_fact("My favourite editor is VS Code")
    result = await executor.run("forget", {"about": "editor"})
    assert result.status is ToolStatus.DENIED and len(store.list_facts()) == 1


async def test_forget_unknown_topic_fails(executor):
    result = await executor.run("forget", {"about": "unicorns"}, confirm=yes)
    assert result.status is ToolStatus.FAILED


async def test_forget_ambiguous_topic_asks_for_detail(executor, store):
    store.add_fact("The solar project is due Monday")
    store.add_fact("The solar battery is 10 kWh")
    result = await executor.run("forget", {"about": "solar"}, confirm=yes)
    assert result.status is ToolStatus.FAILED and "2 saved facts" in result.message
    assert len(store.list_facts()) == 2


async def test_recall_finds_matching_facts(executor, store):
    store.add_fact("The solar battery is 10 kWh")
    store.add_fact("I like tea")
    result = await executor.run("recall", {"topic": "solar battery"})
    assert "10 kWh" in result.message and "tea" not in result.message


async def test_recall_with_no_topic_lists_latest_facts(executor, store):
    store.add_fact("I like tea")
    result = await executor.run("recall", {})
    assert "tea" in result.message


async def test_recall_nothing_says_so(executor):
    result = await executor.run("recall", {"topic": "dragons"})
    assert result.message == "I don't have anything saved about dragons."


async def test_recall_includes_past_conversations_but_not_the_current_one(executor, store):
    store.add_message("earlier", "user", "The inverter model is SUN-8K")
    store.add_message("now", "user", "inverter question from this session")
    result = await executor.run("recall", {"topic": "inverter"})
    assert "SUN-8K" in result.message and "this session" not in result.message


async def test_set_reminder_stores_a_due_time(executor, store):
    result = await executor.run("set_reminder", {"text": "to call the bank", "when": "in 20 minutes"})
    assert result.message == "Reminder set for today at 17:05: call the bank."
    (reminder,) = store.pending_reminders()
    assert reminder.text == "call the bank"
    assert reminder.due_at == datetime(2026, 11, 24, 17, 5).timestamp()


async def test_set_reminder_time_left_inside_the_text_still_works(executor, store):
    result = await executor.run("set_reminder", {"text": "call mum tomorrow at 9am", "when": "later"})
    assert result.status is ToolStatus.DONE
    assert store.pending_reminders()[0].due_at == datetime(2026, 11, 25, 9, 0).timestamp()


@pytest.mark.parametrize("when", ["banana", "December 1 2020"])
async def test_set_reminder_with_bad_or_past_time_fails_and_saves_nothing(executor, store, when):
    result = await executor.run("set_reminder", {"text": "call the bank", "when": when})
    assert result.status is ToolStatus.FAILED
    assert store.pending_reminders() == []


async def test_list_reminders(executor):
    empty = await executor.run("list_reminders", {})
    assert empty.message == "You have no reminders."
    await executor.run("set_reminder", {"text": "call the bank", "when": "tomorrow at 9am"})
    one = await executor.run("list_reminders", {})
    assert one.message == "You have 1 reminder: call the bank, tomorrow at 09:00."


async def test_list_reminders_caps_what_is_read_out(executor):
    for i in range(7):
        await executor.run("set_reminder", {"text": f"task {i}", "when": f"in {i + 1} hours"})
    result = await executor.run("list_reminders", {})
    assert "You have 7 reminders" in result.message and result.message.endswith("and 2 more.")


async def test_cancel_reminder(executor, store):
    await executor.run("set_reminder", {"text": "call the bank", "when": "in 1 hour"})
    result = await executor.run("cancel_reminder", {"about": "bank"})
    assert result.message == "Cancelled the reminder: call the bank."
    assert store.pending_reminders() == []


async def test_cancel_reminder_unknown_or_ambiguous(executor):
    assert (await executor.run("cancel_reminder", {"about": "bank"})).status is ToolStatus.FAILED
    await executor.run("set_reminder", {"text": "call the bank", "when": "in 1 hour"})
    await executor.run("set_reminder", {"text": "call the bank again", "when": "in 2 hours"})
    # "call bank" ties one reminder against the other
    ambiguous = await executor.run("cancel_reminder", {"about": "call"})
    assert ambiguous.status is ToolStatus.FAILED and "2 reminders" in ambiguous.message
