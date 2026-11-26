"""The time-aware hello: "Good morning, boss. All systems are running. How can I help you this morning?"."""

from __future__ import annotations

from datetime import datetime


def part_of_day(now: datetime) -> str:
    hour = now.hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def make_greeting(name: str, now: datetime, model_ready: bool | None = None) -> str:
    """`model_ready`: True = checked and fine, False = checked and not working, None = not checked yet.

    "All systems are running" is only said when it was really checked.
    """
    name = name.strip() or "boss"
    part = part_of_day(now)
    if part == "night":
        hello = f"Working late, {name}?" if now.hour < 5 else f"Good evening, {name}."
        ask = "How can I help you tonight?"
    else:
        hello = f"Good {part}, {name}."
        ask = f"How can I help you this {part}?"
    if model_ready is True:
        status = "All systems are running."
    elif model_ready is False:
        status = "My language model isn't responding yet, so I can only run simple commands."
    else:
        status = ""
    return " ".join(p for p in (hello, status, ask) if p)
