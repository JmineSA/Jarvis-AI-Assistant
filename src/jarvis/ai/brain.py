"""The conversation brain.

Keeps a trimmed in-memory conversation history and asks the model for a reply.
When tools are available the model may ask for one; the brain hands the request
to the executor and answers from the tool's real result, never from the model's
own claim of what it did. The brain only *reasons*: it never touches the
operating system itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
from collections.abc import Callable
from typing import Any, Protocol

from jarvis.ai.ollama_client import ChatResult, OllamaResponseError, ToolCall
from jarvis.ai.prompts import EXPLAIN_PROMPT, SYSTEM_PROMPT
from jarvis.ai.toolselect import REFUSAL, ToolSelector
from jarvis.memory.store import MemoryStore
from jarvis.tools.executor import Confirm, ToolExecutor
from jarvis.tools.registry import ToolResult, ToolStatus
from jarvis.turnlog import TurnLog

logger = logging.getLogger(__name__)

Message = dict[str, str]

MAX_TOOL_CALLS_PER_TURN = 3

_ORDINALS = {
    "1": 0, "one": 0, "first": 0, "1st": 0,
    "2": 1, "two": 1, "second": 1, "2nd": 1,
    "3": 2, "three": 2, "third": 2, "3rd": 2,
}
_CANCEL = re.compile(r"^\W*(?:cancel|never ?mind|forget it|none|stop|no)\W*$", re.IGNORECASE)
_FILLER_WORDS = {"the", "a", "an", "one", "number", "option", "play", "please", "that", "this", "it", "no", "yes"}


def pick_choice(text: str, choices: tuple[tuple[str, dict[str, Any]], ...]) -> int | None:
    """Which option the user meant ("the second one", "2", part of a title), or None."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    if not words:
        return None
    for word in words:
        if word in _ORDINALS and len(words) <= 4 and _ORDINALS[word] < len(choices):
            return _ORDINALS[word]
    wanted = [w for w in words if w not in _FILLER_WORDS]
    if not wanted:
        return None
    scores = []
    for _, args in choices:
        label = set(re.findall(r"[a-z0-9']+", str(args.get("name", "")).lower()))
        scores.append(sum(1 for w in wanted if w in label))
    best = max(scores)
    if best == 0 or best < len(wanted) * 0.6:
        return None
    winners = [i for i, sc in enumerate(scores) if sc == best]
    return winners[0] if len(winners) == 1 else None

# A reply that starts like "Opening Chrome now" without any tool having run is a false claim.
_FALSE_ACTION_CLAIM = re.compile(
    r"^\W*(?:(?:sure|certainly|of course|absolutely|okay|ok)\W+)?"
    r"(?:(?:i(?:'ve| have|'m| am| will|'ll)|i)\s+)?(?:now\s+)?"
    r"(?:opened|opening|launched|launching|searching|playing|composing|drafting)\b",
    re.IGNORECASE,
)
# Likewise "I'll remember that" or "Reminder set" when no memory tool ran.
_FALSE_MEMORY_CLAIM = re.compile(
    r"^\W*(?:(?:sure|certainly|of course|absolutely|okay|ok)\W+)?"
    r"(?:i(?:'ll| will|'ve| have)\s+(?:remember|note|noted|save|saved|store|stored|remind|set a reminder)\b"
    r"|noted\b|reminder (?:set|is set)\b)",
    re.IGNORECASE,
)
NO_TOOL_CHOSEN = (
    "I haven't actually done that, because I didn't select any of my tools. "
    "Please say it again, a little shorter, for example: open VS Code, "
    "or: remember that my project is due on Monday."
)


class ChatClient(Protocol):
    async def chat(self, messages: list[Message]) -> str: ...

    async def chat_with_tools(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> ChatResult: ...


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


_BARE_PLAY = re.compile(r"^(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?play(?:\s+please)?\s*[.?!]?$", re.IGNORECASE)


def finish_sentence(text: str) -> str:
    """A reply cut off by the length cap ends at its last full sentence, so speech never stops mid-word."""
    text = text.strip()
    if not text or text[-1] in ".!?\"')":
        return text
    cut = max(text.rfind(mark) for mark in (".", "!", "?"))
    return text[: cut + 1] if cut >= 20 else text


class Brain:
    def __init__(
        self,
        client: ChatClient,
        system_prompt: str = SYSTEM_PROMPT,
        max_history_turns: int = 10,
        executor: ToolExecutor | None = None,
        context: Callable[[], str] | None = None,
        memory: MemoryStore | None = None,
        session_id: str = "",
        router: Callable[[str], ToolCall | None] | None = None,
        turn_note: Callable[[], str] | None = None,
        turn_log: TurnLog | None = None,
        tool_selector: ToolSelector | None = None,
    ) -> None:
        if max_history_turns < 1:
            raise ValueError("max_history_turns must be at least 1")
        self._client = client
        self._system_prompt = system_prompt
        self._max_messages = max_history_turns * 2
        self._executor = executor
        self._context = context      # extra system text each turn (date, saved facts)
        self._memory = memory        # where finished turns are saved
        self._session_id = session_id
        # Obvious commands ("open VS Code") skip the model: instant, and no GPU work.
        self._router = router
        # Changing text (the time) goes after the user's words, so the start of the prompt
        # stays identical between turns and Ollama can reuse its cached copy of it.
        self._turn_note = turn_note
        self.last_route = ""         # "fast" or "model" for the most recent turn
        self.last_tools: list[str] = []
        self.turn_log = turn_log
        self._selector = tool_selector
        self._previous_tools: list[str] = []
        self._expect_play = False
        self._pending: tuple[str, tuple[tuple[str, dict[str, Any]], ...]] | None = None
        self._history: list[Message] = []
        # What was really sent for each past user message (with the clock note). Re-sending it exactly keeps the
        # start of every request identical to the last one, so Ollama can reuse its cached copy of the prompt.
        self._sent: dict[int, str] = {}
        self._last_sent: tuple[str, str] | None = None
        self._lock = asyncio.Lock()  # one turn at a time keeps history consistent

    @property
    def executor(self) -> ToolExecutor | None:
        return self._executor

    @property
    def history(self) -> list[Message]:
        """A copy of the stored user/assistant messages."""
        return list(self._history)

    @property
    def history_length(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        self._history.clear()
        self._sent.clear()
        self._pending = None
        logger.info("Conversation history cleared.")

    async def respond(self, user_text: str, confirm: Confirm | None = None) -> str:
        """Answer one message, and note in the turn log how it was handled and how long it took."""
        started = time.monotonic()
        self.last_tools = []
        ok = True
        try:
            return await self._respond(user_text, confirm)
        except BaseException:
            ok = False
            raise
        finally:
            self._previous_tools = list(self.last_tools)
            if self.turn_log is not None:
                self.turn_log.record(user_text, self.last_route, list(self.last_tools), time.monotonic() - started, ok)

    async def _respond(self, user_text: str, confirm: Confirm | None = None) -> str:
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
            pending, self._pending = self._pending, None
            if pending is not None and self._executor is not None:
                tool_name, choices = pending
                if _CANCEL.match(text):
                    reply = "Okay, cancelled."
                    self._remember_turn(user_message, reply, text)
                    return reply
                index = pick_choice(text, choices)
                if index is not None:
                    logger.info("Follow-up picked option %d for %s", index + 1, tool_name)
                    self.last_route = "fast"
                    self.last_tools.append(tool_name)
                    outcome = await self._executor.run(tool_name, choices[index][1], confirm)
                    self._remember_turn(user_message, outcome.message, text)
                    return outcome.message
                # Not an answer to the question: treat it as a fresh request.
            if self._router and self._executor:
                if _BARE_PLAY.match(text):
                    self._expect_play = True  # "play" on its own: ask what, and treat the next words as the title
                    reply = "Sure. What would you like to play?"
                    self.last_route = "fast"
                    self._remember_turn(user_message, reply, text)
                    return reply
                if self._expect_play:
                    self._expect_play = False
                    if self._router(text) is None:
                        text = f"play {text}"
                        logger.info("Answer to 'what would you like to play?': %s", text)
            planned = self._router(text) if (self._router and self._executor) else None
            if planned is not None and self._executor is not None:
                steps = planned if isinstance(planned, list) else [planned]
                self.last_route = "fast"
                messages_out: list[str] = []
                for step in steps:
                    logger.info("Fast path: %s %s", step.name, step.arguments)
                    self.last_tools.append(step.name)
                    outcome = await self._executor.run(step.name, step.arguments, confirm)
                    messages_out.append(await self._explained(outcome))
                    if outcome.choices:
                        self._pending = (step.name, outcome.choices)
                    if outcome.status is not ToolStatus.LAUNCHED and outcome.status is not ToolStatus.DONE:
                        break  # do not carry on with the next step after one that did not work
                reply = " ".join(messages_out)
                self._remember_turn(user_message, reply, text)
                return reply

            self.last_route = "model"
            messages = self._build_messages(text)
            if self._executor is None:
                reply = finish_sentence(await self._client.chat(messages))
            else:
                reply = await self._respond_with_tools(messages, self._executor, confirm, text)

            self._history.extend([user_message, {"role": "assistant", "content": reply}])
            self._keep_sent(user_message, text)
            self._trim()
            self._save_turn(text, reply)
            logger.info("Turn complete (history: %d messages).", len(self._history))
            return reply

    async def _explained(self, outcome: ToolResult) -> str:
        """What to say for a tool outcome. If the tool handed over material to explain (text from the
        screen), the model explains it in a fresh, tool-free call; nothing from it can trigger an action."""
        if not outcome.ask:
            return outcome.message
        messages: list[Message] = [
            {"role": "system", "content": EXPLAIN_PROMPT},
            {"role": "user", "content": outcome.ask},
        ]
        reply = finish_sentence(await self._client.chat(messages))
        return reply or outcome.message

    def _build_messages(self, text: str) -> list[Message]:
        note = self._note()
        sent = f"{text}\n\n({note})" if note else text
        self._last_sent = (text, sent)
        past = [
            {"role": m["role"], "content": self._sent.get(id(m), m["content"])} if m["role"] == "user" else m
            for m in self._history
        ]
        return [
            {"role": "system", "content": self._system_text()},
            *past,
            {"role": "user", "content": sent},
        ]

    def _note(self) -> str:
        if self._turn_note is None:
            return ""
        try:
            return self._turn_note()
        except Exception:
            logger.exception("Could not build the turn note")
            return ""

    async def warm_up(self) -> None:
        """Load the model and cache the prompt's fixed start. Safe to call any time; never raises."""
        warm = getattr(self._client, "warm_up", None)
        if warm is None:
            return
        try:
            tools = self._executor.registry.schemas() if self._executor else None
            await warm(self._build_messages("Hello."), tools)
        except Exception:
            logger.info("Warm-up failed", exc_info=True)

    def _system_text(self) -> str:
        """The system prompt plus live context (the date and what you asked it to remember)."""
        if self._context is None:
            return self._system_prompt
        try:
            extra = self._context()
        except Exception:
            logger.exception("Could not build the memory context")
            return self._system_prompt
        return f"{self._system_prompt}\n\n{extra}" if extra else self._system_prompt

    def _save_turn(self, user_text: str, reply: str) -> None:
        """Keep the exchange in long-term memory. Memory trouble never breaks a conversation."""
        if self._memory is None:
            return
        try:
            self._memory.add_message(self._session_id, "user", user_text)
            self._memory.add_message(self._session_id, "assistant", reply)
        except (sqlite3.Error, OSError):
            logger.exception("Could not save the conversation to memory")

    async def _respond_with_tools(
        self, messages: list[Message], executor: ToolExecutor, confirm: Confirm | None, text: str = ""
    ) -> str:
        result, calls = await self.decide(messages, executor, text)

        if not calls:
            reply = finish_sentence(result.content.strip())
            if not reply:
                raise OllamaResponseError("Ollama returned an empty reply.")
            if _FALSE_ACTION_CLAIM.match(reply) or _FALSE_MEMORY_CLAIM.match(reply):
                logger.warning("Model claimed an action without calling a tool: %r", reply)
                return NO_TOOL_CHOSEN
            return reply

        outcomes: list[str] = []
        for call in calls[:MAX_TOOL_CALLS_PER_TURN]:
            self.last_tools.append(call.name)
            outcome = await executor.run(call.name, call.arguments, confirm)
            if outcome.choices and len(calls) == 1:
                self._pending = (call.name, outcome.choices)
            outcomes.append(await self._explained(outcome))
        # The reply comes from what the tools really reported, not from the model's words.
        return " ".join(outcomes)

    def _remember_turn(self, user_message: Message, reply: str, text: str) -> None:
        self._history.extend([user_message, {"role": "assistant", "content": reply}])
        self._trim()
        self._save_turn(text, reply)

    async def decide(self, messages: list[Message], executor: ToolExecutor, text: str) -> tuple[ChatResult, list[ToolCall]]:
        """Ask the model what to do, offering only the tools the sentence could need (see toolselect.py)."""
        every = executor.registry.schemas()
        offered = self._selector.select(text, every, self._previous_tools) if self._selector and text else every
        result = await self._chat(messages, offered)
        calls = list(result.tool_calls) or _calls_from_text(result.content, executor)
        if not calls and len(offered) < len(every) and REFUSAL.match(result.content.strip()):
            logger.info("No tool matched and the model declined; retrying with all %d tools.", len(every))
            result = await self._chat(messages, every)
            calls = list(result.tool_calls) or _calls_from_text(result.content, executor)
        return result, calls

    async def _chat(self, messages: list[Message], schemas: list[dict[str, Any]]) -> ChatResult:
        if schemas:
            return await self._client.chat_with_tools(messages, schemas)
        return ChatResult(await self._client.chat(messages), ())  # nothing here needs a tool: plain conversation

    def _keep_sent(self, user_message: Message, text: str) -> None:
        if self._last_sent is not None and self._last_sent[0] == text:
            self._sent[id(user_message)] = self._last_sent[1]

    def _trim(self) -> None:
        excess = len(self._history) - self._max_messages
        if excess > 0:
            del self._history[:excess]
        alive = {id(m) for m in self._history}
        for key in [k for k in self._sent if k not in alive]:
            del self._sent[key]
