"""SQLite storage for JARVIS's memory.

This module is the only code in the project that touches SQLite. It owns the
schema and gives the rest of the program a small, thread-safe connection
handle. Everything here is synchronous on purpose: the calls are tiny and the
results are read from worker threads by the tool executor.

The database file lives outside the source tree (``data/jarvis_memory.db`` by
default) and is git-ignored.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Tokens are stored space-padded so a SQL LIKE can match a whole word:
#   tokens = " open vscode "
#   WHERE tokens LIKE '% vscode %'
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL DEFAULT 'default',
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    tokens     TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);

CREATE TABLE IF NOT EXISTS facts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    key              TEXT NOT NULL UNIQUE,
    value            TEXT NOT NULL,
    source           TEXT NOT NULL DEFAULT 'user',
    session_id       TEXT NOT NULL DEFAULT 'default',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    access_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TEXT
);

CREATE TABLE IF NOT EXISTS summaries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL DEFAULT 'default',
    summary    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id, id DESC);
"""


def utc_now() -> str:
    """Timestamp used everywhere in the database (sortable as text)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryDatabase:
    """A SQLite database with JARVIS's memory schema.

    The connection is created once and guarded by a lock, because the brain
    writes from the event loop while the tools read from worker threads.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._open()

    # ------------------------------------------------------------------ setup

    def _open(self) -> None:
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            timeout=15.0,
        )
        self._conn.row_factory = sqlite3.Row
        # WAL lets a reader (the web UI) work while a writer (the voice loop)
        # is busy, which matters as soon as two entry points share a file.
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:  # pragma: no cover - depends on the filesystem
            logger.debug("Could not enable WAL mode on %s", self.path)
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:  # pragma: no cover - only after close()
            raise RuntimeError("The memory database is closed.")
        return self._conn

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """Run a write transaction."""
        with self._lock:
            conn = self.connection
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Run a read query and return every row."""
        with self._lock:
            return list(self.connection.execute(sql, tuple(params)).fetchall())

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.connection.execute(sql, tuple(params)).fetchone()

    # -------------------------------------------------------------- utilities

    def size_bytes(self) -> int:
        """File size on disk (0 for an in-memory database)."""
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def is_writable(self) -> bool:
        """True when a write would succeed (used by the health check)."""
        try:
            with self.write() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES('write_probe', ?)",
                    (utc_now(),),
                )
            return True
        except sqlite3.Error:
            logger.exception("The memory database is not writable.")
            return False

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
