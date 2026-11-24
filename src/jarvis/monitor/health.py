"""The health report: is every part of JARVIS actually working?

This is what the web interface shows at the top and what ``python -m jarvis
status`` prints. Each check is independent, so one broken part does not hide
the others.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from jarvis.config import Settings
from jarvis.monitor.system import SystemMonitor, SystemSnapshot

logger = logging.getLogger(__name__)

CheckStatus = Literal["ok", "warn", "error", "unknown"]


@dataclass(frozen=True)
class HealthCheck:
    """One part of the system and whether it is working."""

    name: str
    status: CheckStatus
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass
class HealthReport:
    """All the checks, plus the machine's current state."""

    status: CheckStatus = "unknown"
    checks: list[HealthCheck] = field(default_factory=list)
    snapshot: SystemSnapshot | None = None
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "checks": [check.as_dict() for check in self.checks],
            "snapshot": self.snapshot.as_dict() if self.snapshot else None,
            **self.extras,
        }

    def summary(self) -> str:
        """One line per check, for the terminal."""
        marks = {"ok": "[ ok ]", "warn": "[warn]", "error": "[fail]", "unknown": "[????]"}
        lines = []
        for check in self.checks:
            detail = f" - {check.detail}" if check.detail else ""
            lines.append(f"{marks.get(check.status, '[????]')} {check.name}{detail}")
        return "\n".join(lines)


def _worst(statuses: list[CheckStatus]) -> CheckStatus:
    for candidate in ("error", "warn", "unknown", "ok"):
        if candidate in statuses:
            return candidate
    return "unknown"


async def collect_health(
    settings: Settings,
    client,
    brain=None,
    monitor: SystemMonitor | None = None,
    memory=None,
) -> HealthReport:
    """Check every part of JARVIS and read the machine's state.

    `client` is an OllamaClient (or anything with a `health()` coroutine).
    `brain` and `memory` are optional; when given, they add their own checks.
    """
    monitor = monitor or SystemMonitor()
    checks: list[HealthCheck] = []

    # --- the language model -------------------------------------------------
    try:
        status = await client.health()
        if status.reachable and status.model_available:
            checks.append(
                HealthCheck("Language model", "ok", f"{status.model} is ready.")
            )
        elif status.reachable:
            checks.append(
                HealthCheck(
                    "Language model",
                    "warn",
                    status.detail or f"Model '{status.model}' is not installed.",
                )
            )
        else:
            checks.append(
                HealthCheck(
                    "Language model",
                    "error",
                    status.detail or "Ollama cannot be reached.",
                )
            )
    except Exception as exc:
        checks.append(HealthCheck("Language model", "error", f"Check failed: {exc}"))

    # --- the memory database ------------------------------------------------
    if memory is None and settings.memory.enabled:
        checks.append(HealthCheck("Memory", "warn", "Memory is enabled but not attached."))
    elif memory is not None:
        try:
            stats = memory.stats()
            writable = memory.db.is_writable()
            checks.append(
                HealthCheck(
                    "Memory",
                    "ok" if writable else "error",
                    f"{stats.messages} messages, {stats.facts} facts"
                    + ("" if writable else " - the database is not writable."),
                )
            )
        except Exception as exc:
            checks.append(HealthCheck("Memory", "error", f"Check failed: {exc}"))

    # --- the tools ----------------------------------------------------------
    if brain is not None:
        executor = getattr(brain, "_executor", None)
        registry = getattr(executor, "registry", None)
        if registry is not None:
            names = registry.names()
            checks.append(
                HealthCheck("Tools", "ok", f"{len(names)} available: {', '.join(names)}.")
            )
        else:
            checks.append(HealthCheck("Tools", "warn", "Tools are switched off."))

    # --- the voice models ---------------------------------------------------
    models_dir = settings.resolve_models_dir()
    if models_dir.exists() and any(models_dir.rglob("*")):
        files = sum(1 for path in models_dir.rglob("*") if path.is_file())
        checks.append(HealthCheck("Voice models", "ok", f"{files} files in {models_dir.name}."))
    else:
        checks.append(
            HealthCheck(
                "Voice models",
                "warn",
                "Not downloaded yet. Run: python -m jarvis download-models",
            )
        )

    # --- the machine --------------------------------------------------------
    try:
        snapshot = monitor.snapshot()
    except Exception as exc:
        logger.exception("Could not read the system state.")
        snapshot = None
        checks.append(HealthCheck("System", "error", f"Could not read the system state: {exc}"))

    extras: dict[str, object] = {}
    if memory is not None:
        extras["memory"] = memory.stats().as_dict()
    if brain is not None:
        extras["history_messages"] = brain.history_length
    extras["model"] = settings.ollama.model
    extras["api"] = f"http://{settings.api.host}:{settings.api.port}"

    return HealthReport(
        status=_worst([check.status for check in checks]),
        checks=checks,
        snapshot=snapshot,
        extras=extras,
    )
