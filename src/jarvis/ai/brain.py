"""The conversation brain.

Phase 1: keeps an in-memory, trimmed conversation history and asks the model
for a reply. Later phases add tool selection and SQLite-backed memory here.
The brain only *reasons*; it never touches the operating system.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from jarvis.ai.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

Message = dict[str, str]


class ChatClient(Protocol):
    async def chat(self, messages: list[Message]) -> str: ...


class Brain:
    def __init__(
        self,
        client: ChatClient,
        system_prompt: str = SYSTEM_PROMPT,
        max_history_turns: int = 10,
    ) -> None:
        if max_history_turns < 1:
            raise ValueError("max_history_turns must be at least 1")
        self._client = client
        self._system_prompt = system_prompt
        self._max_messages = max_history_turns * 2
        self._history: list[Message] = []
        self._lock = asyncio.Lock()  # one turn at a time keeps history consistent

    @property
    def history(self) -> list[Message]:
        """A copy of the stored user/assistant messages."""
        return list(self._history)

    @property
    def history_length(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        self._history.clear()
        logger.info("Conversation history cleared.")

    async def respond(self, user_text: str) -> str:
        """Process one user message and return the assistant's reply.

        If the model call fails, the history is left untouched so a retry
        does not duplicate the user's message.
        """
        text = user_text.strip()
        if not text:
            raise ValueError("Message must not be empty.")

        async with self._lock:
            user_message: Message = {"role": "user", "content": text}
            messages = [
                {"role": "system", "content": self._system_prompt},
                *self._history,
                user_message,
            ]
            reply = await self._client.chat(messages)

            self._history.extend([user_message, {"role": "assistant", "content": reply}])
            self._trim()
            logger.info("Turn complete (history: %d messages).", len(self._history))
            return reply

    def _trim(self) -> None:
        excess = len(self._history) - self._max_messages
        if excess > 0:
            del self._history[:excess]
