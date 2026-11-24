"""System information tools."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import psutil

from jarvis.tools.registry import ToolArgs, ToolResult


class NoArgs(ToolArgs):
    pass


def make_system_status() -> Callable[[NoArgs], ToolResult]:
    def run(_: NoArgs) -> ToolResult:
        cpu = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        gb = 1024**3
        return ToolResult.done(
            f"CPU is at {cpu:.0f} percent. Memory is at {memory.percent:.0f} percent, "
            f"{memory.used / gb:.1f} of {memory.total / gb:.0f} gigabytes."
        )

    return run


def make_current_time(clock: Callable[[], datetime] = datetime.now) -> Callable[[NoArgs], ToolResult]:
    def run(_: NoArgs) -> ToolResult:
        now = clock()
        return ToolResult.done(f"It's {now:%H:%M} on {now:%A}, {now.day} {now:%B %Y}.")

    return run
