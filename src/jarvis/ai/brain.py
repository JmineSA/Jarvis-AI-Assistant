"""The conversation brain.

Keeps a trimmed in-memory conversation history and asks the model for a reply.
When tools are available the model may ask for one; the brain hands the request
to the executor and answers from the tool's real result, never from the model's
own claim of what it did. The brain only *reasons*: it never touches the
operating system itself.

When a memory store is attached, the brain also does the Phase 4 job: before
each turn it searches the store for memories that match what was just said and
injects them into the model's context, then writes the finished exchange back
into the store.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Protocol

from jarvis.ai.ollama_client import ChatResult, OllamaResponseError, ToolCall
from jarvis.ai.prompts import MEMORY_CONTEXT_HEADER, SYSTEM_PROMPT
from jarvis.tools.executor import Confirm, ToolExecutor

logger = logging.getLogger(__name__)

Message = dict[str, str]

MAX_TOOL_CALLS_PER_TURN = 3

# A reply that starts like "Opening Chrome now" without any tool having run is a false claim.
_FALSE_ACTION_CLAIM = re.compile(
    r"^\W*(?:(?:sure|certainly|of course|absolutely|okay|ok)\W+)?"
    r"(?:(?:i(?:'ve| have|'m| am| will|'ll)|i)\s+)?(?:now\s+)?"
    r"(?:opened|opening|launched|launching)\b",
    re.IGNORECASE,
)
NO_TOOL_CHOSEN = (
    "I haven't actually done that, because I didn't select any of my tools. "
    "Please say it again, for example: open VS Code."
)


class ChatClient(Protocol):
    async def chat(self, messages: list[Message]) -> str: ...

    async def chat_with_tools(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> ChatResult: ...


class MemoryLike(Protocol):
    """The part of the memory store the brain needs.

    Keeping it a protocol means the brain can be tested with a small stub and
    the memory package stays free of any dependency on the AI layer.
    """

    def context_for(self, query: str) -> str: ...

    def remember_turn(self, user_text: str, assistant_text: str) -> None: ...


def _calls_from_text(content: str, executor: ToolExecutor) -> list[ToolCall]:
    """Small models sometimes write the tool request as JSON text instead of using the tool slot."""
    text = re.sub(r"</?tool_call>|```(?:json)?", "", content).strip()
    if not text.startswith(("{", "[")):
        return []
    try:
        data = json.loads(text)
    except ValueError:
        return []
    calls: list[ToolCall] = []
    for item in data if isinstance(data, list) else [data]:
        if isinstance(item, dict) and isinstance(item.get("name"), str) and executor.has(item["name"]):
            arguments = item.get("arguments", item.get("parameters", {}))
            calls.append(ToolCall(item["name"], arguments if isinstance(arguments, (dict, str)) else {}))
    return calls


class Brain:
    def __init__(
        self,
        client: ChatClient,
        system_prompt: str = SYSTEM_PROMPT,
        max_history_turns: int = 10,
        executor: ToolExecutor | None = None,
        memory: MemoryLike | None = None,
    ) -> None:
        if max_history_turns < 1:
            raise ValueError("max_history_turns must be at least 1")
        self._client = client
        self._system_prompt = system_prompt
        self._max_messages = max_history_turns * 2
        self._executor = executor
        self._memory = memory
        self._history: list[Message] = []
        self._lock = asyncio.Lock()  # one turn at a time keeps history consistent

    @property
    def history(self) -> list[Message]:
        """A copy of the stored user/assistant messages."""
        return list(self._history)

    @property
    def history_length(self) -> int:
        return len(self._history)

    @property
    def has_memory(self) -> bool:
        return self._memory is not None

    def reset(self) -> None:
        self._history.clear()
        logger.info("Conversation history cleared.")

    async def respond(self, user_text: str, confirm: Confirm | None = None) -> str:
        """Process one user message and return the assistant's reply.

        `confirm` is asked before any tool that needs your approval. If the model
        call fails, the history is left untouched so a retry does not duplicate
        the user's message.
        """
        text = user_text.strip()
        if not text:
            raise ValueError("Message must not be empty.")

        async with self._lock:
            user_message: Message = {"role": "user", "content": text}
            messages = [
                {"role": "system", "content": self._system_prompt},
                *self._memory_messages(text),
                *self._history,
                user_message,
            ]
            if self._executor is None:
                reply = await self._client.chat(messages)
            else:
                reply = await self._respond_with_tools(messages, self._executor, confirm)

            self._history.extend([user_message, {"role": "assistant", "content": reply}])
            self._trim()
            self._remember_turn(text, reply)
            logger.info("Turn complete (history: %d messages).", len(self._history))
            return reply

    def _memory_messages(self, query: str) -> list[Message]:
        """The retrieved memories, as a message for the model.

        A failure to read the memory must never stop the conversation, so any
        error here is logged and the turn continues without memories.
        """
        if self._memory is None:
            return []
        try:
            context = self._memory.context_for(query)
        except Exception:
            logger.exception("Memory retrieval failed; continuing without memories.")
            return []
        if not context or not context.strip():
            return []
        logger.info("Injected %d characters of memory into the context.", len(context))
        return [
            {
                "role": "system",
                "content": f"{MEMORY_CONTEXT_HEADER}\n{context.strip()}",
            }
        ]

    def _remember_turn(self, user_text: str, reply: str) -> None:
        """Write the finished exchange into the memory store."""
        if self._memory is None:
            return
        try:
            self._memory.remember_turn(user_text, reply)
        except Exception:
            logger.exception("Could not save this turn to memory.")

    async def _respond_with_tools(
        self, messages: list[Message], executor: ToolExecutor, confirm: Confirm | None
    ) -> str:
        result = await self._client.chat_with_tools(messages, executor.registry.schemas())
        calls = list(result.tool_calls) or _calls_from_text(result.content, executor)

        if not calls:
            reply = result.content.strip()
            if not reply:
                raise OllamaResponseError("Ollama returned an empty reply.")
            if _FALSE_ACTION_CLAIM.match(reply):
                logger.warning("Model claimed an action without calling a tool: %r", reply)
                return NO_TOOL_CHOSEN
            return reply

        outcomes: list[str] = []
        for call in calls[:MAX_TOOL_CALLS_PER_TURN]:
            outcome = await executor.run(call.name, call.arguments, confirm)
            outcomes.append(outcome.message)
        # The reply comes from what the tools really reported, not from the model's words.
        return " ".join(outcomes)

    def _trim(self) -> None:
        excess = len(self._history) - self._max_messages
        if excess > 0:
            del self._history[:excess]
