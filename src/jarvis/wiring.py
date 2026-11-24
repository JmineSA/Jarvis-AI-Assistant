"""Assembles the brain, tools and profile from settings (shared by every entry point)."""

from __future__ import annotations

import logging

from jarvis.ai.brain import Brain, ChatClient
from jarvis.ai.prompts import SYSTEM_PROMPT, build_tool_system_prompt
from jarvis.config import Settings
from jarvis.profile import Profile, load_profile
from jarvis.tools.builtin import build_registry
from jarvis.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


def load_user_profile(settings: Settings) -> Profile:
    """Your profile, or an empty one when tools are switched off."""
    if not settings.tools.enabled:
        return Profile()
    return load_profile(settings.resolve_profile_path())


def build_brain(
    settings: Settings, client: ChatClient, profile: Profile | None = None
) -> Brain:
    """The brain, with tools attached unless they are disabled in settings."""
    if not settings.tools.enabled:
        return Brain(client, SYSTEM_PROMPT, settings.brain.max_history_turns)

    profile = profile if profile is not None else load_user_profile(settings)
    registry = build_registry(profile)
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
        build_tool_system_prompt(profile.owner_name),
        settings.brain.max_history_turns,
        executor=executor,
    )
