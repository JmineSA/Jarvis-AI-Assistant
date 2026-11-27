"""A small local record of each conversation turn, so we can see where time and mistakes go.

One line of JSON per turn in data/turns.jsonl: what was said, whether the fast path or the model handled it,
which tools ran, and how long each stage took. It stays on your PC, is trimmed when it gets big, and can be
switched off with logging.record_turns: false. `python -m jarvis route-report` reads it.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_BYTES = 512_000
MAX_TEXT = 300


class TurnLog:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time, max_bytes: int = MAX_BYTES) -> None:
        self._path = path
        self._clock = clock
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._last_id = 0

    def _append(self, record: dict[str, Any]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.exists() and self._path.stat().st_size > self._max_bytes:
                lines = self._path.read_text(encoding="utf-8").splitlines()
                self._path.write_text("\n".join(lines[len(lines) // 2:]) + "\n", encoding="utf-8")
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("Could not write the turn log")

    def record(self, text: str, route: str, tools: list[str], brain_s: float, ok: bool = True) -> None:
        with self._lock:
            self._last_id = int(self._clock() * 1000)
            self._append(
                {"id": self._last_id, "text": " ".join(text.split())[:MAX_TEXT], "route": route, "tools": tools,
                 "brain_s": round(brain_s, 2), "ok": ok}
            )

    def annotate(self, **fields: Any) -> None:
        """Add details to the most recent turn (for example how long speech recognition took)."""
        with self._lock:
            if self._last_id:
                self._append({"id": self._last_id, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in fields.items()}})


def read_turns(path: Path) -> list[dict[str, Any]]:
    """All turns, with later annotations merged in."""
    turns: dict[int, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and "id" in item:
            turns.setdefault(item["id"], {}).update(item)
    return [t for t in turns.values() if "route" in t]


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_report(turns: list[dict[str, Any]], top: int = 12) -> str:
    if not turns:
        return "No turns recorded yet. Use JARVIS for a while, then run this again."
    fast = [t for t in turns if t["route"] == "fast"]
    model = [t for t in turns if t["route"] == "model"]
    out = [
        f"{len(turns)} turns recorded: {len(fast)} handled instantly (fast path), {len(model)} needed the model.",
        f"Average time to answer: fast {_avg([t['brain_s'] for t in fast]):.1f} s, model {_avg([t['brain_s'] for t in model]):.1f} s.",
    ]
    stt = [t["stt_s"] for t in turns if "stt_s" in t]
    tts = [t["tts_first_s"] for t in turns if "tts_first_s" in t]
    if stt:
        out.append(f"Speech recognition averages {_avg(stt):.1f} s; the voice starts speaking {_avg(tts):.1f} s after the reply is ready.")
    failed = [t for t in turns if not t.get("ok", True)]
    if failed:
        out.append(f"{len(failed)} turns ended in an error.")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in model:
        groups[t["text"].lower().strip(" .!?")].append(t)

    def listing(rows: list[tuple[str, list[dict[str, Any]]]]) -> list[str]:
        return [
            f"  {len(items)}x  {text}   ({_avg([i['brain_s'] for i in items]):.1f} s; tools: {', '.join(sorted({n for i in items for n in i['tools']})) or 'none'})"
            for text, items in rows[:top]
        ]

    acted = sorted(((k, v) for k, v in groups.items() if any(i["tools"] for i in v)), key=lambda kv: (-len(kv[1]), kv[0]))
    chat = sorted(((k, v) for k, v in groups.items() if not any(i["tools"] for i in v)), key=lambda kv: (-len(kv[1]), kv[0]))
    if acted:
        out += ["", "Commands the model had to work out (good candidates for a new instant rule; send me these):", *listing(acted)]
    if chat:
        out += ["", "Plain conversation (these need the model, so their time is the model's speed):", *listing(chat)]
    return "\n".join(out)
