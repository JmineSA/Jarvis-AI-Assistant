"""Runs tool requests safely.

The only path from "the model asked for something" to "something happened":
  1. the tool must exist in the registry,
  2. its arguments must validate,
  3. risky tools need your confirmation (the AI cannot approve its own action),
  4. it runs with a time limit, and any crash becomes a plain failure message.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from pydantic import ValidationError

from jarvis.tools.registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

Confirm = Callable[[str], Awaitable[bool]]


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        always_confirm: Iterable[str] = (),
        timeout: float = 10.0,
    ) -> None:
        self.registry = registry
        self._always_confirm = set(always_confirm)
        self._timeout = timeout
        self._listeners: list[Callable[[str, ToolResult], None]] = []

    def has(self, name: str) -> bool:
        return name in self.registry

    def add_listener(self, listener: Callable[[str, ToolResult], None]) -> None:
        """Be told about every finished tool request (the dashboard shows these)."""
        self._listeners.append(listener)

    @staticmethod
    def _confirm_prompt(tool: Any, args: Any) -> str:
        if tool.describe is not None:
            try:
                return f"JARVIS wants to {tool.describe(args)}"
            except Exception:
                logger.exception("Could not describe %s", tool.name)
        summary = ", ".join(f"{k}={v}" for k, v in args.model_dump(exclude_none=True).items())
        return f"JARVIS wants to run '{tool.name}'" + (f" ({summary})." if summary else ".")

    async def run(self, name: str, arguments: Any, confirm: Confirm | None = None) -> ToolResult:
        result = await self._run(name, arguments, confirm)
        for listener in self._listeners:
            try:
                listener(name, result)
            except Exception:
                logger.exception("Tool listener failed")
        return result

    async def _run(self, name: str, arguments: Any, confirm: Confirm | None) -> ToolResult:
        tool = self.registry.get(name)
        if tool is None:
            logger.warning("Model asked for unknown tool %r", name)
            return ToolResult.failed("I don't have a tool for that.")

        if isinstance(arguments, str):  # some models send the arguments as JSON text
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = None
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return ToolResult.failed("I didn't understand the details of that request.")

        try:
            args = tool.args_model.model_validate(arguments)
        except ValidationError:
            logger.warning("Bad arguments for %s: %r", name, arguments)
            return ToolResult.failed("I didn't understand the details of that request.")

        if tool.risk == "confirm" or name in self._always_confirm:
            prompt = self._confirm_prompt(tool, args)
            approved = False
            if confirm is not None:
                try:
                    approved = bool(await confirm(prompt))
                except Exception:
                    logger.exception("Confirmation prompt failed")
            if not approved:
                logger.info("TOOL %s %r -> denied", name, arguments)
                return ToolResult.denied("Understood. I won't do that.")

        try:
            result = await asyncio.wait_for(asyncio.to_thread(tool.run, args), tool.timeout or self._timeout)
        except TimeoutError:
            logger.error("Tool %s timed out after %.0fs", name, self._timeout)
            return ToolResult.failed("That took too long, so I gave up.")
        except Exception:
            logger.exception("Tool %s crashed", name)
            return ToolResult.failed("Something went wrong while doing that.")

        logger.info("TOOL %s %r -> %s: %s", name, arguments, result.status.value, result.message)
        return result
