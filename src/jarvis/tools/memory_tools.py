"""Tools for remembering things, recalling them, and reminders."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from pydantic import Field

from jarvis.memory.store import Fact, MemoryStore, keywords
from jarvis.timeparse import format_due, parse_when
from jarvis.tools.registry import ToolArgs, ToolResult

SNIPPET_LENGTH = 140
MAX_RECALLED_MESSAGES = 2
MAX_RECALLED_FACTS = 5
MAX_LISTED_REMINDERS = 5


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIPPET_LENGTH else text[: SNIPPET_LENGTH - 1].rstrip() + "…"


def _strip_leading_to(text: str) -> str:
    text = text.strip()
    return text[3:].strip() if text.lower().startswith("to ") else text


# --------------------------------------------------------------------------- remember


class RememberArgs(ToolArgs):
    text: str = Field(min_length=1)


def make_remember(store: MemoryStore) -> Callable[[RememberArgs], ToolResult]:
    def run(args: RememberArgs) -> ToolResult:
        try:
            fact, created = store.add_fact(args.text)
        except ValueError:
            return ToolResult.failed("There was nothing to remember in that.")
        if not created:
            return ToolResult.done(f"I already knew that: {fact.text}")
        return ToolResult.done(f"I'll remember that: {fact.text}")

    return run


# --------------------------------------------------------------------------- forget


class ForgetArgs(ToolArgs):
    about: str = Field(min_length=1)


def _unique_best(matches: list[tuple[int, Fact]]) -> Fact | None:
    """The best-matching fact, or None if there is none or several tie for best."""
    if not matches:
        return None
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        return None
    return matches[0][1]


def make_forget(store: MemoryStore) -> tuple[Callable[[ForgetArgs], ToolResult], Callable[[ForgetArgs], str]]:
    def run(args: ForgetArgs) -> ToolResult:
        matches = store.match_facts(args.about)
        if not matches:
            return ToolResult.failed(f"I don't have anything saved about {args.about}.")
        fact = _unique_best(matches)
        if fact is None:
            return ToolResult.failed(
                f"That matches {len(matches)} saved facts. Please say which one, in more detail."
            )
        store.delete_fact(fact.id)
        return ToolResult.done(f"Forgotten: {fact.text}")

    def describe(args: ForgetArgs) -> str:
        fact = _unique_best(store.match_facts(args.about))
        if fact is None:
            return f"forget saved facts about {args.about}."
        return f'forget this saved fact: "{fact.text}".'

    return run, describe


# --------------------------------------------------------------------------- recall


class RecallArgs(ToolArgs):
    topic: str = ""


def make_recall(store: MemoryStore, session_id: str | None) -> Callable[[RecallArgs], ToolResult]:
    def run(args: RecallArgs) -> ToolResult:
        topic = args.topic.strip()
        if keywords(topic):
            facts = [f for _, f in store.match_facts(topic)[:MAX_RECALLED_FACTS]]
        else:  # "what do you know about me?": nothing specific to look for, so list the latest
            facts = store.list_facts(MAX_RECALLED_FACTS)

        parts: list[str] = []
        if facts:
            parts.append("You told me: " + "; ".join(f.text for f in facts) + ".")

        for message in store.search_messages(topic, MAX_RECALLED_MESSAGES, exclude_session=session_id):
            when = datetime.fromtimestamp(message.created_at)
            who = "you said" if message.role == "user" else "I said"
            parts.append(f'On {when.day} {when:%B}, {who}: "{_snippet(message.content)}"')

        if not parts:
            about = f" about {topic}" if topic else ""
            return ToolResult.done(f"I don't have anything saved{about}.")
        return ToolResult.done(" ".join(parts))

    return run


# --------------------------------------------------------------------------- reminders


class SetReminderArgs(ToolArgs):
    text: str = Field(min_length=1)
    when: str = Field(min_length=1)
    important: bool = False  # important reminders are read first in the morning briefing


def make_set_reminder(
    store: MemoryStore, clock: Callable[[], datetime]
) -> Callable[[SetReminderArgs], ToolResult]:
    def run(args: SetReminderArgs) -> ToolResult:
        now = clock()
        text = _strip_leading_to(args.text)
        due = parse_when(args.when, now) or parse_when(text, now)  # the model may leave the time in the text
        if due is None:
            return ToolResult.failed(
                "I couldn't understand when. Try something like: in 20 minutes, "
                "tomorrow at 9am, or 1 December."
            )
        if due <= now:
            return ToolResult.failed("That time has already passed.")
        try:
            store.add_reminder(text, due.timestamp(), args.important)
        except ValueError:
            return ToolResult.failed("Tell me what to remind you about.")
        kind = "Important reminder" if args.important else "Reminder"
        return ToolResult.done(f"{kind} set for {format_due(due, now)}: {text}.")

    return run


class NoReminderArgs(ToolArgs):
    pass


def make_list_reminders(
    store: MemoryStore, clock: Callable[[], datetime]
) -> Callable[[NoReminderArgs], ToolResult]:
    def run(_: NoReminderArgs) -> ToolResult:
        pending = store.pending_reminders()
        if not pending:
            return ToolResult.done("You have no reminders.")
        now = clock()
        shown = [
            f"{'important: ' if r.important else ''}{r.text}, {format_due(datetime.fromtimestamp(r.due_at), now)}"
            for r in pending[:MAX_LISTED_REMINDERS]
        ]
        noun = "reminder" if len(pending) == 1 else "reminders"
        message = f"You have {len(pending)} {noun}: " + "; ".join(shown)
        more = len(pending) - len(shown)
        return ToolResult.done(message + (f"; and {more} more." if more > 0 else "."))

    return run


class CancelReminderArgs(ToolArgs):
    about: str = Field(min_length=1)


def make_cancel_reminder(store: MemoryStore) -> Callable[[CancelReminderArgs], ToolResult]:
    def run(args: CancelReminderArgs) -> ToolResult:
        matches = store.match_reminders(args.about)
        if not matches:
            return ToolResult.failed(f"I don't have a reminder about {args.about}.")
        if len(matches) > 1 and matches[0][0] == matches[1][0]:
            return ToolResult.failed(
                f"That matches {len(matches)} reminders. Please say which one, in more detail."
            )
        reminder = matches[0][1]
        store.cancel_reminder(reminder.id)
        return ToolResult.done(f"Cancelled the reminder: {reminder.text}.")

    return run
