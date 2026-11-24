"""Import every module of the package one at a time, with a timeout each.

A quick way to catch a module that fails to import on this machine, without the
voice modules (they need audio hardware) and without the Streamlit app script
(it is an application, not a library).

Run: python scripts/import_check.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"

MODULES = [
    "jarvis",
    "jarvis.config",
    "jarvis.logging_setup",
    "jarvis.profile",
    "jarvis.wiring",
    "jarvis.main",
    "jarvis.ai",
    "jarvis.ai.brain",
    "jarvis.ai.ollama_client",
    "jarvis.ai.prompts",
    "jarvis.api.app",
    "jarvis.api.routes",
    "jarvis.api.schemas",
    "jarvis.memory",
    "jarvis.memory.db",
    "jarvis.memory.store",
    "jarvis.monitor",
    "jarvis.monitor.health",
    "jarvis.monitor.system",
    "jarvis.tools",
    "jarvis.tools.builtin",
    "jarvis.tools.executor",
    "jarvis.tools.memory",
    "jarvis.tools.registry",
    "jarvis.ui",
    "jarvis.ui.backend",
    "jarvis.ui.theme",
]


def main() -> int:
    failures: list[tuple[str, str]] = []
    for name in MODULES:
        result = subprocess.run(
            [sys.executable, "-c", f"import {name}"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(ROOT.parent),
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        if result.returncode == 0:
            print(f"ok    {name}", flush=True)
        else:
            detail = (result.stderr or "").strip().splitlines()[-1] if result.stderr else ""
            print(f"FAIL  {name}: {detail}", flush=True)
            failures.append((name, detail))
    print(f"\n{len(MODULES) - len(failures)}/{len(MODULES)} modules import cleanly.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
