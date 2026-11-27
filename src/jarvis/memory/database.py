"""SQLite connection and schema migrations.

Each operation opens its own short-lived connection, so the voice loop, the
reminder thread and the tool threads can all use the database safely.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Index in this list + 1 is the schema version. Never edit an old entry: append a new one.
MIGRATIONS: list[str] = [
    """
    CREATE TABLE messages (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role       TEXT NOT NULL,
        content    TEXT NOT NULL,
        created_at REAL NOT NULL
    );
    CREATE INDEX idx_messages_session ON messages(session_id, id);

    CREATE TABLE facts (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        text       TEXT NOT NULL,
        norm       TEXT NOT NULL UNIQUE,
        created_at REAL NOT NULL
    );

    CREATE TABLE reminders (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        text       TEXT NOT NULL,
        due_at     REAL NOT NULL,
        created_at REAL NOT NULL,
        status     TEXT NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending', 'fired', 'cancelled')),
        fired_at   REAL
    );
    CREATE INDEX idx_reminders_due ON reminders(status, due_at);
    """,
    """
    ALTER TABLE reminders ADD COLUMN important INTEGER NOT NULL DEFAULT 0;
    """,
]


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """A committed-or-rolled-back connection that is always closed afterwards."""
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:  # commit on success, roll back on error
                yield conn
        finally:
            conn.close()

    def _migrate(self) -> None:
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > len(MIGRATIONS):
                raise sqlite3.DatabaseError(
                    f"{self.path} was made by a newer JARVIS (schema {version}); update JARVIS."
                )
            for number in range(version, len(MIGRATIONS)):
                logger.info("Upgrading memory database to schema %d", number + 1)
                conn.executescript(MIGRATIONS[number])
                conn.execute(f"PRAGMA user_version = {number + 1}")
