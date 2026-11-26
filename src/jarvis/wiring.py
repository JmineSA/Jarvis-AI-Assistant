"""Assembles the brain, tools, profile and memory from settings (shared by every entry point)."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from datetime import datetime

from jarvis.ai.brain import Brain, ChatClient
from jarvis.ai.prompts import SYSTEM_PROMPT, build_tool_system_prompt
from jarvis.ai.router import CommandRouter
from jarvis.config import Settings
from jarvis.memory.store import MemoryStore
from jarvis.profile import Profile, load_profile
from jarvis.tools.builtin import build_registry
from jarvis.tools.executor import ToolExecutor
from jarvis.tools.drives import prewarm as prewarm_drives
from jarvis.tools.media import prewarm as prewarm_music

logger = logging.getLogger(__name__)


def load_user_profile(settings: Settings) -> Profile:
    """Your profile, or an empty one when tools are switched off."""
    if not settings.tools.enabled:
        return Profile()
    return load_profile(settings.resolve_profile_path())


def open_store(settings: Settings) -> MemoryStore | None:
    """The memory database, or None if memory is off or the file cannot be opened.

    JARVIS keeps working without memory; it just forgets when it stops.
    """
    if not settings.memory.enabled:
        return None
    path = settings.resolve_database_path()
    try:
        return MemoryStore(path)
    except (sqlite3.Error, OSError) as exc:
        logger.error("Memory is unavailable (%s): %s", path, exc)
        return None


def make_context(
    store: MemoryStore,
    owner_name: str | None,
    max_facts: int,
    clock: Callable[[], datetime] = datetime.now,
) -> Callable[[], str]:
    """Text added to the system prompt every turn: what you asked it to remember.

    The date and time are NOT here (see make_clock_note): text that changes every minute would
    force the model to re-read the whole prompt on every request.
    """

    def context() -> str:
        lines: list[str] = []
        facts = store.list_facts(max_facts) if max_facts > 0 else []
        if facts:
            who = owner_name or "the user"
            lines.append(f"What you know about {who} (saved at their request):")
            lines.extend(f"- {fact.text}" for fact in reversed(facts))  # oldest first
        return "\n".join(lines)

    return context


def make_clock_note(clock: Callable[[], datetime] = datetime.now) -> Callable[[], str]:
    def note() -> str:
        now = clock()
        return f"It is {now:%A}, {now.day} {now:%B %Y}, {now:%H:%M}."

    return note


def build_brain(
    settings: Settings,
    client: ChatClient,
    profile: Profile | None = None,
    store: MemoryStore | None = None,
) -> Brain:
    """The brain, with tools and memory attached when they are enabled."""
    if not settings.tools.enabled:
        return Brain(client, SYSTEM_PROMPT, settings.brain.max_history_turns)

    profile = profile if profile is not None else load_user_profile(settings)
    session_id = MemoryStore.new_session_id()
    registry = build_registry(profile, store=store, session_id=session_id)
    executor = ToolExecutor(
        registry,
        always_confirm=settings.tools.always_confirm,
        timeout=settings.tools.timeout_seconds,
    )
    unknown = [n for n in settings.tools.always_confirm if n not in registry]
    if unknown:
        logger.warning("always_confirm names tools that do not exist: %s", ", ".join(unknown))
    logger.info("Tools ready: %s", ", ".join(registry.names()))
    prewarm_music(profile)
    prewarm_drives(profile)
    router = (
        CommandRouter(profile, registry.names(), aliases=settings.voice.stt.name_aliases)
        if settings.brain.fast_commands
        else None
    )

    return Brain(
        client,
        build_tool_system_prompt(profile.owner_name, memory=store is not None),
        settings.brain.max_history_turns,
        executor=executor,
        context=make_context(store, profile.owner_name, settings.memory.max_facts_in_prompt)
        if store is not None
        else None,
        memory=store if settings.memory.save_conversations else None,
        session_id=session_id,
        router=router,
        turn_note=make_clock_note(),
    )
