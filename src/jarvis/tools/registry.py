"""Tool definitions and the registry the AI brain chooses from."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ToolStatus(str, Enum):
    DONE = "done"          # finished and the outcome is known
    LAUNCHED = "launched"  # Windows was asked to start it; the window itself is not verified
    FAILED = "failed"
    DENIED = "denied"      # needed your confirmation and did not get it


@dataclass(frozen=True)
class ToolResult:
    status: ToolStatus
    message: str  # plain sentence, safe to speak aloud
    # Options the user can pick from next turn: (label, arguments for the same tool).
    choices: tuple[tuple[str, dict[str, Any]], ...] = ()

    @classmethod
    def done(cls, message: str) -> ToolResult:
        return cls(ToolStatus.DONE, message)

    @classmethod
    def launched(cls, message: str) -> ToolResult:
        return cls(ToolStatus.LAUNCHED, message)

    @classmethod
    def failed(cls, message: str) -> ToolResult:
        return cls(ToolStatus.FAILED, message)

    @classmethod
    def denied(cls, message: str) -> ToolResult:
        return cls(ToolStatus.DENIED, message)


class ToolArgs(BaseModel):
    """Base class for tool arguments. Unknown extra fields from the model are ignored."""

    model_config = ConfigDict(extra="ignore")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]          # JSON schema shown to the model
    args_model: type[ToolArgs]          # what we actually validate against
    run: Callable[[Any], ToolResult]
    risk: Literal["safe", "confirm"] = "safe"
    # Plain-words description of what this call will do, shown when asking you to confirm.
    describe: Callable[[Any], str] | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Tool list in the format Ollama expects."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]
