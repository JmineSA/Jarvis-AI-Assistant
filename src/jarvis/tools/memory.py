"""Tools that let JARVIS remember and recall things.

These are the only way the model touches the memory database. The user can
also drive them from the terminal or the web interface, so every tool works
without the model's help.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import Field

from jarvis.memory.store import MemoryStore
from jarvis.tools.registry import ToolArgs, ToolResult


class RememberArgs(ToolArgs):
    fact: str = Field(
        ...,
        min_length=1,
        max_length=600,
        description="The thing worth remembering, written as a complete short sentence.",
    )
    key: str | None = Field(
        None,
        max_length=80,
        description=(
            "A short stable name for the fact, such as 'user.dog.name'. "
            "Reusing a key updates that fact instead of adding a new one."
        ),
    )


class RecallArgs(ToolArgs):
    query: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="What to look for: words, a topic, or a question.",
    )
    limit: int = Field(5, ge=1, le=20, description="How many memories to return.")


class ForgetArgs(ToolArgs):
    id: int = Field(..., ge=1, description="The id of the memory to delete.")


def make_remember(store: MemoryStore) -> Callable[[RememberArgs], ToolResult]:
    def run(args: RememberArgs) -> ToolResult:
        try:
            _, created = store.remember_fact(args.fact, args.key, source="assistant")
        except ValueError:
            return ToolResult.failed("I didn't understand what to remember.")
        if created:
            return ToolResult.done("I've noted that for next time.")
        return ToolResult.done("I've updated what I had noted about that.")

    return run


def make_recall(store: MemoryStore) -> Callable[[RecallArgs], ToolResult]:
    def run(args: RecallArgs) -> ToolResult:
        hits = store.search(args.query, limit=args.limit)
        if not hits:
            return ToolResult.done("I don't have anything stored about that.")
        parts = []
        for hit in hits:
            if hit.kind == "fact":
                parts.append(hit.content)
            else:
                parts.append(f"{hit.role}: {hit.content}")
        return ToolResult.done("; ".join(parts))

    return run


def make_forget(store: MemoryStore) -> Callable[[ForgetArgs], ToolResult]:
    def run(args: ForgetArgs) -> ToolResult:
        hit = store.get(args.id)
        if hit is None:
            return ToolResult.failed("I couldn't find that memory to delete.")
        if store.forget(args.id):
            return ToolResult.done("Forgotten.")
        return ToolResult.failed("I couldn't delete that memory.")

    return run
