"""Everything JARVIS remembers: conversations, facts you told it, and reminders.

All data stays in one local SQLite file. This is the only module that runs SQL.
"""

from __future__ import annotations

import re
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from jarvis.memory.database import Database

MAX_FACT_LENGTH = 300
MAX_MESSAGE_LENGTH = 2000

_STOPWORDS = frozenset(
    ["the", "and", "for", "with", "that", "this", "what", "when", "where", "which", "who", "whom", "how", "did", "does", "about", "you", "your", "have", "has", "had", "was", "were", "are", "not", "but", "can", "could", "would", "should", "will", "just", "tell", "told", "said", "say", "know", "remember", "please", "anything", "something", "any", "all", "from", "into", "than", "then", "them", "they", "their", "there", "here", "its", "our", "out", "too", "very"]
)


def keywords(text: str) -> list[str]:
    """Meaningful lowercase words of a question, for matching against saved text."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return list(dict.fromkeys(w for w in words if len(w) >= 3 and w not in _STOPWORDS))


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _escape_like(word: str) -> str:
    return word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@dataclass(frozen=True)
class Fact:
    id: int
    text: str
    created_at: float


@dataclass(frozen=True)
class Reminder:
    id: int
    text: str
    due_at: float
    created_at: float
    status: str


@dataclass(frozen=True)
class StoredMessage:
    id: int
    session_id: str
    role: str
    content: str
    created_at: float


@dataclass(frozen=True)
class Counts:
    messages: int
    facts: int
    pending_reminders: int


def _fact(row: sqlite3.Row) -> Fact:
    return Fact(row["id"], row["text"], row["created_at"])


def _reminder(row: sqlite3.Row) -> Reminder:
    return Reminder(row["id"], row["text"], row["due_at"], row["created_at"], row["status"])


def _message(row: sqlite3.Row) -> StoredMessage:
    return StoredMessage(row["id"], row["session_id"], row["role"], row["content"], row["created_at"])


class MemoryStore:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self._db = Database(path)
        self._clock = clock

    @property
    def path(self) -> Path:
        return self._db.path

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex

    # ------------------------------------------------------------------ conversation

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._db.connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content[:MAX_MESSAGE_LENGTH], self._clock()),
            )

    def recent_messages(self, limit: int = 20, session_id: str | None = None) -> list[StoredMessage]:
        """Newest messages, returned oldest first."""
        sql = "SELECT * FROM messages"
        params: list = []
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._db.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_message(r) for r in reversed(rows)]

    def search_messages(
        self, query: str, limit: int = 5, exclude_session: str | None = None
    ) -> list[StoredMessage]:
        """Past messages that mention the query's words, best match first."""
        words = keywords(query)
        if not words:
            return []
        clause = " OR ".join("content LIKE ? ESCAPE '\\'" for _ in words)
        params: list = [f"%{_escape_like(w)}%" for w in words]
        sql = f"SELECT * FROM messages WHERE ({clause})"
        if exclude_session is not None:
            sql += " AND session_id != ?"
            params.append(exclude_session)
        sql += " ORDER BY id DESC LIMIT 300"
        with self._db.connect() as conn:
            rows = [_message(r) for r in conn.execute(sql, params).fetchall()]

        def score(message: StoredMessage) -> int:
            text = message.content.lower()
            return sum(w in text for w in words)

        rows.sort(key=lambda m: (-score(m), -m.id))
        return rows[:limit]

    def clear_messages(self) -> int:
        with self._db.connect() as conn:
            return conn.execute("DELETE FROM messages").rowcount

    # ------------------------------------------------------------------ facts

    def add_fact(self, text: str) -> tuple[Fact, bool]:
        """Save a fact. Returns (fact, created); created is False if it was already saved."""
        text = " ".join(text.split())[:MAX_FACT_LENGTH]
        norm = _normalise(text)
        if not norm:
            raise ValueError("A fact must contain some words.")
        with self._db.connect() as conn:
            try:
                cursor = conn.execute(
                    "INSERT INTO facts (text, norm, created_at) VALUES (?, ?, ?)",
                    (text, norm, self._clock()),
                )
            except sqlite3.IntegrityError:
                row = conn.execute("SELECT * FROM facts WHERE norm = ?", (norm,)).fetchone()
                return _fact(row), False
            row = conn.execute("SELECT * FROM facts WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return _fact(row), True

    def list_facts(self, limit: int | None = None) -> list[Fact]:
        """Facts, newest first."""
        sql = "SELECT * FROM facts ORDER BY id DESC"
        params: list = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._db.connect() as conn:
            return [_fact(r) for r in conn.execute(sql, params).fetchall()]

    def match_facts(self, query: str) -> list[tuple[int, Fact]]:
        """Facts that share words with the query as (score, fact), best first."""
        words = keywords(query)
        scored = []
        for fact in self.list_facts():
            text = fact.text.lower()
            score = sum(w in text for w in words)
            if score:
                scored.append((score, fact))
        scored.sort(key=lambda pair: (-pair[0], -pair[1].id))
        return scored

    def delete_fact(self, fact_id: int) -> bool:
        with self._db.connect() as conn:
            return conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,)).rowcount > 0

    # ------------------------------------------------------------------ reminders

    def add_reminder(self, text: str, due_at: float) -> Reminder:
        text = " ".join(text.split())[:MAX_FACT_LENGTH]
        if not text:
            raise ValueError("A reminder must say what to be reminded of.")
        with self._db.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO reminders (text, due_at, created_at) VALUES (?, ?, ?)",
                (text, due_at, self._clock()),
            )
            row = conn.execute("SELECT * FROM reminders WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return _reminder(row)

    def pending_reminders(self) -> list[Reminder]:
        with self._db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reminders WHERE status = 'pending' ORDER BY due_at, id"
            ).fetchall()
        return [_reminder(r) for r in rows]

    def due_reminders(self, now: float) -> list[Reminder]:
        with self._db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reminders WHERE status = 'pending' AND due_at <= ? ORDER BY due_at, id",
                (now,),
            ).fetchall()
        return [_reminder(r) for r in rows]

    def mark_fired(self, reminder_id: int) -> bool:
        """Mark a pending reminder as delivered. False if it was not pending (already handled)."""
        with self._db.connect() as conn:
            return (
                conn.execute(
                    "UPDATE reminders SET status = 'fired', fired_at = ? WHERE id = ? AND status = 'pending'",
                    (self._clock(), reminder_id),
                ).rowcount
                > 0
            )

    def cancel_reminder(self, reminder_id: int) -> bool:
        with self._db.connect() as conn:
            return (
                conn.execute(
                    "UPDATE reminders SET status = 'cancelled' WHERE id = ? AND status = 'pending'",
                    (reminder_id,),
                ).rowcount
                > 0
            )

    def match_reminders(self, query: str) -> list[tuple[int, Reminder]]:
        words = keywords(query)
        scored = []
        for reminder in self.pending_reminders():
            text = reminder.text.lower()
            score = sum(w in text for w in words)
            if score:
                scored.append((score, reminder))
        scored.sort(key=lambda pair: (-pair[0], pair[1].due_at))
        return scored

    # ------------------------------------------------------------------ housekeeping

    def counts(self) -> Counts:
        with self._db.connect() as conn:
            messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM reminders WHERE status = 'pending'"
            ).fetchone()[0]
        return Counts(messages, facts, pending)

    def oldest_message_time(self) -> float | None:
        with self._db.connect() as conn:
            row = conn.execute("SELECT MIN(created_at) FROM messages").fetchone()
        return row[0]

    def wipe(self) -> None:
        """Erase everything JARVIS remembers (the empty tables stay)."""
        with self._db.connect() as conn:
            for table in ("messages", "facts", "reminders"):
                conn.execute(f"DELETE FROM {table}")
