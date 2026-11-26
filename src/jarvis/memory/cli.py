"""`python -m jarvis memory ...`: see and delete everything JARVIS remembers.

    memory                 show saved facts, reminders and conversation totals
    memory search WORDS    search past conversations
    memory forget ID       delete one saved fact
    memory cancel ID       cancel one reminder
    memory wipe-history    delete all stored conversations
    memory wipe            delete EVERYTHING (asks you to type DELETE)
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from jarvis.memory.store import MemoryStore

USAGE = __doc__.split("\n", 1)[1].strip("\n") if __doc__ else ""


def _day(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")


def _moment(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def show_everything(store: MemoryStore, out: Callable[[str], None]) -> None:
    facts = store.list_facts()
    reminders = store.pending_reminders()
    counts = store.counts()

    out(f"Saved facts ({len(facts)}):")
    for fact in reversed(facts):
        out(f"  [{fact.id}] {_day(fact.created_at)}  {fact.text}")
    if not facts:
        out("  (nothing saved)")

    out(f"\nPending reminders ({len(reminders)}):")
    for reminder in reminders:
        out(f"  [{reminder.id}] {_moment(reminder.due_at)}  {reminder.text}")
    if not reminders:
        out("  (none)")

    oldest = store.oldest_message_time()
    since = f", oldest from {_day(oldest)}" if oldest else ""
    out(f"\nConversation history: {counts.messages} messages stored{since}")
    out(f"Database file: {store.path}")


def run_memory_cli(
    store: MemoryStore,
    args: list[str],
    out: Callable[[str], None] = print,
    ask: Callable[[str], str] = input,
) -> int:
    """Returns a process exit code."""
    action = args[0].lower() if args else "list"
    rest = args[1:]

    if action in {"list", "show"}:
        show_everything(store, out)
        return 0

    if action == "search":
        if not rest:
            out("Usage: memory search WORDS")
            return 2
        found = store.search_messages(" ".join(rest), limit=10)
        for message in found:
            who = "You" if message.role == "user" else "JARVIS"
            out(f"{_moment(message.created_at)}  {who}: {message.content}")
        if not found:
            out("Nothing found.")
        return 0

    if action in {"forget", "cancel"}:
        if len(rest) != 1 or not rest[0].isdigit():
            out(f"Usage: memory {action} ID   (the number in square brackets from `memory`)")
            return 2
        item_id = int(rest[0])
        done = store.delete_fact(item_id) if action == "forget" else store.cancel_reminder(item_id)
        noun = "fact" if action == "forget" else "reminder"
        out(f"Deleted {noun} {item_id}." if done and action == "forget" else
            f"Cancelled reminder {item_id}." if done else f"There is no {noun} {item_id}.")
        return 0 if done else 1

    if action == "wipe-history":
        out(f"Deleted {store.clear_messages()} stored messages.")
        return 0

    if action == "wipe":
        if ask("This erases ALL facts, reminders and conversations. Type DELETE to confirm: ").strip() != "DELETE":
            out("Cancelled. Nothing was deleted.")
            return 1
        store.wipe()
        out("Everything JARVIS remembered has been erased.")
        return 0

    out(f"Unknown memory command '{action}'.\n\n{USAGE}")
    return 2
