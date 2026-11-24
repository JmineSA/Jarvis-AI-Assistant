"""Assembles the brain, tools, memory and profile from settings.

Every entry point (terminal chat, voice, the HTTP API and the web interface)
builds the same objects through this module, so they all behave identically.
"""

from __future__ import annotations

import logging

from jarvis.ai.brain import Brain, ChatClient
from jarvis.ai.prompts import SYSTEM_PROMPT, build_tool_system_prompt
from jarvis.config import Settings
from jarvis.memory.db import MemoryDatabase
from jarvis.memory.store import MemoryStore
from jarvis.profile import Profile, load_profile
from jarvis.tools.builtin import build_registry
from jarvis.tools.executor import ToolExecutor
from jarvis.tools.memory import ForgetArgs, RecallArgs, RememberArgs, make_forget, make_recall, make_remember
from jarvis.tools.registry import Tool

logger = logging.getLogger(__name__)


def load_user_profile(settings: Settings) -> Profile:
    """Your profile, or an empty one when tools are switched off."""
    if not settings.tools.enabled:
        return Profile()
    return load_profile(settings.resolve_profile_path())


def build_memory(settings: Settings, session_id: str = "default") -> MemoryStore | None:
    """The memory store, or None when memory is switched off or cannot open."""
    if not settings.memory.enabled:
        logger.info("Memory is disabled.")
        return None
    try:
        db = MemoryDatabase(settings.resolve_memory_path())
        store = MemoryStore(
            db,
            session_id=session_id,
            max_results=settings.memory.max_results,
            max_context_chars=settings.memory.max_context_chars,
            recent_turns=settings.memory.recent_turns,
            min_score=settings.memory.min_score,
            retention_days=settings.memory.retention_days,
            save_conversations=settings.memory.save_conversations,
        )
        store.prune()
        logger.info("Memory ready: %s", settings.resolve_memory_path())
        return store
    except Exception:
        logger.exception("Could not open the memory database; continuing without memory.")
        return None


def build_registry_for(
    settings: Settings,
    profile: Profile,
    memory: MemoryStore | None = None,
):
    """The tool registry, with the memory tools added when memory is on."""
    registry = build_registry(profile)
    if memory is not None:
        registry.register(
            Tool(
                name="remember",
                description="Store a fact worth keeping for later conversations.",
                parameters={
                    "type": "object",
                    "properties": {
                        "fact": {
                            "type": "string",
                            "description": "The fact, written as one short sentence.",
                        },
                        "key": {
                            "type": "string",
                            "description": (
                                "Optional short name for the fact, e.g. 'user.dog.name'. "
                                "Reuse a key to update it."
                            ),
                        },
                    },
                    "required": ["fact"],
                },
                args_model=RememberArgs,
                run=make_remember(memory),
            )
        )
        registry.register(
            Tool(
                name="recall",
                description="Search the memory of earlier conversations.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to look for.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "How many memories to return (1-20).",
                        },
                    },
                    "required": ["query"],
                },
                args_model=RecallArgs,
                run=make_recall(memory),
            )
        )
        registry.register(
            Tool(
                name="forget",
                description="Delete one stored memory by its id.",
                parameters={
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "description": "The id of the memory to delete.",
                        }
                    },
                    "required": ["id"],
                },
                args_model=ForgetArgs,
                run=make_forget(memory),
                risk="confirm",
            )
        )
    return registry


def build_brain(
    settings: Settings,
    client: ChatClient,
    profile: Profile | None = None,
    memory: MemoryStore | None = None,
    session_id: str = "default",
) -> Brain:
    """The brain, with tools and memory attached unless they are switched off."""
    if memory is None:
        memory = build_memory(settings, session_id=session_id)

    if not settings.tools.enabled:
        return Brain(
            client,
            SYSTEM_PROMPT,
            settings.brain.max_history_turns,
            memory=memory,
        )

    profile = profile if profile is not None else load_user_profile(settings)
    registry = build_registry_for(settings, profile, memory)
    executor = ToolExecutor(
        registry,
        always_confirm=settings.tools.always_confirm,
        timeout=settings.tools.timeout_seconds,
    )
    unknown = [n for n in settings.tools.always_confirm if n not in registry]
    if unknown:
        logger.warning("always_confirm names tools that do not exist: %s", ", ".join(unknown))
    logger.info("Tools ready: %s", ", ".join(registry.names()))
    return Brain(
        client,
        build_tool_system_prompt(profile.owner_name, has_memory=memory is not None),
        settings.brain.max_history_turns,
        executor=executor,
        memory=memory,
    )
