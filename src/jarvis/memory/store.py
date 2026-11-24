"""The memory store: what JARVIS remembers and how it finds it again.

Two kinds of memory live here:

* **Messages** - the conversation itself, kept so a later session can pick up
  where the last one left off.
* **Facts** - short, curated statements ("the LightsOn project is in
  D:\\Projects\\LightsOn"), either said by the user or written by the model
  through the ``remember`` tool.

Retrieval is a small BM25 ranker written in plain Python. It needs no model,
no network and no extra dependency, which matters because JARVIS has to keep
working when Ollama is down.

The public entry point for the brain is :meth:`MemoryStore.context_for`, which
returns the block of text that gets injected into the model's context.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from jarvis.memory.db import MemoryDatabase, utc_now

logger = logging.getLogger(__name__)

# BM25 parameters. k1 controls how fast a repeated word stops counting, b how
# much a long message is penalised for being long.
_K1 = 1.5
_B = 0.75

# Facts are curated by the user, so they get a small scoring bonus.
_FACT_BOOST = 1.35

# Words that carry no retrieval signal.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by did do does for from had has have he her him his how i if in
    into is it its me my of on or our out she so than that the their them then there these they this
    to too up us was we were what when where which who why will with would you your please
    """.split()
)

_WORD = re.compile(r"[a-z0-9']+")

MAX_MESSAGE_CHARS = 2000


def tokenize(text: str) -> list[str]:
    """Lowercase words without stopwords, keeping duplicates (term frequency)."""
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


def _pad(tokens: Sequence[str]) -> str:
    """Store tokens padded with spaces so SQL can match a whole word."""
    if not tokens:
        return ""
    return f" {' '.join(tokens)} "


def slugify(text: str, max_words: int = 8) -> str:
    """A short, stable key for a fact, e.g. 'lights-on-project-path'."""
    words = _WORD.findall(text.lower())
    if not words:
        return "fact"
    return "-".join(words[:max_words])[:120]


@dataclass(frozen=True)
class MemoryHit:
    """One retrieved piece of memory."""

    id: int
    kind: str  # "message" | "fact"
    role: str  # "user" | "assistant" | "fact"
    content: str
    score: float
    created_at: str
    key: str | None = None

    @property
    def date_label(self) -> str:
        try:
            stamp = datetime.fromisoformat(self.created_at)
        except ValueError:
            return self.created_at
        return stamp.astimezone().strftime("%Y-%m-%d %H:%M")

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "role": self.role,
            "content": self.content,
            "score": round(self.score, 4),
            "created_at": self.created_at,
            "key": self.key,
            "date_label": self.date_label,
        }


@dataclass
class MemoryStats:
    """Counts shown by the status endpoint and the web interface."""

    messages: int = 0
    facts: int = 0
    summaries: int = 0
    sessions: int = 0
    size_bytes: int = 0
    oldest: str | None = None
    newest: str | None = None
    extras: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "messages": self.messages,
            "facts": self.facts,
            "summaries": self.summaries,
            "sessions": self.sessions,
            "size_bytes": self.size_bytes,
            "oldest": self.oldest,
            "newest": self.newest,
            **self.extras,
        }


class MemoryStore:
    """Everything JARVIS remembers, and how it searches for it."""

    def __init__(
        self,
        db: MemoryDatabase,
        *,
        session_id: str = "default",
        max_results: int = 5,
        max_context_chars: int = 1800,
        recent_turns: int = 3,
        min_score: float = 0.35,
        retention_days: int = 0,
        save_conversations: bool = True,
    ) -> None:
        self.db = db
        self.session_id = session_id
        self.max_results = max_results
        self.max_context_chars = max_context_chars
        self.recent_turns = recent_turns
        self.min_score = min_score
        self.retention_days = retention_days
        self.save_conversations = save_conversations

    # ------------------------------------------------------------------ write

    def remember_message(self, role: str, content: str, session_id: str | None = None) -> int:
        """Store one message of the conversation."""
        text = content.strip()
        if not text:
            return 0
        with self.db.write() as conn:
            cursor = conn.execute(
                "INSERT INTO messages(session_id, role, content, tokens, created_at)"
                " VALUES(?,?,?,?,?)",
                (
                    session_id or self.session_id,
                    role,
                    text,
                    _pad(tokenize(text)),
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid or 0)

    def remember_turn(self, user_text: str, assistant_text: str, session_id: str | None = None) -> None:
        """Store a full exchange (the user's message and JARVIS's answer)."""
        if not self.save_conversations:
            return
        self.remember_message("user", user_text, session_id)
        self.remember_message("assistant", assistant_text, session_id)

    def remember_fact(
        self,
        value: str,
        key: str | None = None,
        source: str = "user",
        session_id: str | None = None,
    ) -> tuple[int, bool]:
        """Save a fact. Returns (id, created). An existing key is updated."""
        text = value.strip()
        if not text:
            raise ValueError("A fact must not be empty.")
        fact_key = (key or slugify(text)).strip().lower()[:120]
        if not fact_key:
            fact_key = slugify(text)
        now = utc_now()
        with self.db.write() as conn:
            row = conn.execute("SELECT id FROM facts WHERE key = ?", (fact_key,)).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE facts SET value = ?, source = ?, updated_at = ? WHERE id = ?",
                    (text, source, now, row["id"]),
                )
                return int(row["id"]), False
            cursor = conn.execute(
                "INSERT INTO facts(key, value, source, session_id, created_at, updated_at)"
                " VALUES(?,?,?,?,?,?)",
                (fact_key, text, source, session_id or self.session_id, now, now),
            )
            return int(cursor.lastrowid or 0), True

    def add_summary(self, summary: str, session_id: str | None = None) -> int:
        """Store a short summary of a stretch of conversation."""
        text = summary.strip()
        if not text:
            return 0
        with self.db.write() as conn:
            cursor = conn.execute(
                "INSERT INTO summaries(session_id, summary, created_at) VALUES(?,?,?)",
                (session_id or self.session_id, text, utc_now()),
            )
            return int(cursor.lastrowid or 0)

    # ------------------------------------------------------------------- read

    def recent_messages(self, limit: int = 10, session_id: str | None = None) -> list[MemoryHit]:
        """The last messages of this session, newest first."""
        if limit <= 0:
            return []
        rows = self.db.query(
            "SELECT id, role, content, created_at FROM messages"
            " WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id or self.session_id, limit),
        )
        return [
            MemoryHit(
                id=int(row["id"]),
                kind="message",
                role=str(row["role"]),
                content=str(row["content"]),
                score=0.0,
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def all_facts(self, limit: int = 200) -> list[MemoryHit]:
        """Every stored fact, most recently updated first."""
        rows = self.db.query(
            "SELECT id, key, value, source, created_at, updated_at FROM facts"
            " ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        return [
            MemoryHit(
                id=int(row["id"]),
                kind="fact",
                role="fact",
                content=str(row["value"]),
                score=0.0,
                created_at=str(row["updated_at"] or row["created_at"]),
                key=str(row["key"]),
            )
            for row in rows
        ]

    def get(self, memory_id: int) -> MemoryHit | None:
        """One memory by id (a fact or a message)."""
        row = self.db.query_one(
            "SELECT id, key, value, source, created_at, updated_at FROM facts WHERE id = ?",
            (memory_id,),
        )
        if row is not None:
            return MemoryHit(
                id=int(row["id"]),
                kind="fact",
                role="fact",
                content=str(row["value"]),
                score=0.0,
                created_at=str(row["updated_at"] or row["created_at"]),
                key=str(row["key"]),
            )
        row = self.db.query_one(
            "SELECT id, role, content, created_at FROM messages WHERE id = ?",
            (memory_id,),
        )
        if row is None:
            return None
        return MemoryHit(
            id=int(row["id"]),
            kind="message",
            role=str(row["role"]),
            content=str(row["content"]),
            score=0.0,
            created_at=str(row["created_at"]),
        )

    # -------------------------------------------------------------- retrieval

    def search(
        self,
        query: str,
        limit: int | None = None,
        session_id: str | None = None,
    ) -> list[MemoryHit]:
        """Rank memories against a query with BM25.

        Messages from this session compete with messages from earlier sessions
        and with the facts. The result is empty when nothing matches.
        """
        limit = limit or self.max_results
        if limit <= 0:
            return []

        terms = tokenize(query)
        docs = self._candidate_documents(terms, session_id)
        if not docs:
            return []

        if not terms:
            # Nothing searchable in the query: fall back to the newest memories.
            return sorted(docs, key=lambda d: d["created_at"], reverse=True)[:limit]

        total = len(docs)
        doc_freq: dict[str, int] = {}
        for doc in docs:
            for term in set(doc["terms"]):
                doc_freq[term] = doc_freq.get(term, 0) + 1

        avg_len = sum(len(doc["terms"]) for doc in docs) / total or 1.0
        scored: list[MemoryHit] = []
        for doc in docs:
            counts: dict[str, int] = {}
            for term in doc["terms"]:
                counts[term] = counts.get(term, 0) + 1
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = doc_freq.get(term, 0)
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                norm = 1 - _B + _B * (len(doc["terms"]) / avg_len)
                score += idf * (tf * (_K1 + 1)) / (tf + _K1 * norm)
            if score <= 0:
                continue
            score *= doc.get("boost", 1.0)
            score *= self._recency_factor(doc["created_at"])
            scored.append(
                MemoryHit(
                    id=doc["id"],
                    kind=doc["kind"],
                    role=doc["role"],
                    content=doc["content"],
                    score=score,
                    created_at=doc["created_at"],
                    key=doc.get("key"),
                )
            )

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:limit]

    def _candidate_documents(
        self, terms: Sequence[str], session_id: str | None
    ) -> list[dict[str, object]]:
        """Everything that could match, with its token list."""
        docs: list[dict[str, object]] = []
        seen: set[tuple[str, int]] = set()

        def add(kind: str, ident: int, role: str, content: str, created: str, boost: float, key: str | None) -> None:
            marker = (kind, ident)
            if marker in seen:
                return
            seen.add(marker)
            docs.append(
                {
                    "id": ident,
                    "kind": kind,
                    "role": role,
                    "content": content,
                    "created_at": created,
                    "terms": tokenize(content),
                    "boost": boost,
                    "key": key,
                }
            )

        # Facts: always a candidate, they are few and deliberately important.
        for row in self.db.query(
            "SELECT id, key, value, source, created_at, updated_at FROM facts"
        ):
            add(
                "fact",
                int(row["id"]),
                "fact",
                f"{row['key']} {row['value']}",
                str(row["updated_at"] or row["created_at"]),
                _FACT_BOOST,
                str(row["key"]),
            )

        # Messages that contain at least one of the search terms.
        message_rows: list = []
        if terms:
            clauses = " OR ".join(["tokens LIKE ?"] * len(terms))
            params = [f"% {term} %" for term in terms]
            message_rows = self.db.query(
                f"SELECT id, role, content, created_at, tokens FROM messages WHERE {clauses}",
                params,
            )
        else:
            message_rows = self.db.query(
                "SELECT id, role, content, created_at, tokens FROM messages"
                " WHERE session_id = ? ORDER BY id DESC LIMIT 50",
                (session_id or self.session_id,),
            )

        current = session_id or self.session_id
        for row in message_rows:
            add(
                "message",
                int(row["id"]),
                str(row["role"]),
                str(row["content"]),
                str(row["created_at"]),
                1.0,
                None,
            )
        if current:
            logger.debug("Memory search over %d candidate documents.", len(docs))
        return docs

    @staticmethod
    def _recency_factor(created_at: str) -> float:
        """Newer memories count a little more than old ones."""
        try:
            stamp = datetime.fromisoformat(created_at)
        except ValueError:
            return 1.0
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (datetime.now(timezone.utc) - stamp).total_seconds() / 86400)
        return 1.0 + 0.25 * math.exp(-age_days / 30.0)

    # ---------------------------------------------------------------- context

    def context_for(self, query: str) -> str:
        """The memory block that is injected into the model's context.

        Returns an empty string when there is nothing worth telling the model.
        """
        hits = self.search(query, limit=self.max_results)
        hits = [hit for hit in hits if hit.score >= self.min_score] or hits[:1]

        lines: list[str] = []
        used = 0
        budget = self.max_context_chars

        for hit in hits:
            if hit.kind == "fact":
                text = f"- [fact] {hit.key}: {hit.content}" if hit.key else f"- [fact] {hit.content}"
            else:
                text = f"- [{hit.date_label}] {hit.role}: {hit.content}"
            text = text.strip()
            if len(text) + 1 > budget - used:
                continue
            lines.append(text)
            used += len(text) + 1

        # A few recent messages keep the thread alive even when the query was
        # too vague to match anything.
        if self.recent_turns > 0:
            recent = self.recent_messages(limit=self.recent_turns)
            known = {hit.id for hit in hits}
            for hit in reversed(recent):
                if hit.id in known:
                    continue
                text = f"- [{hit.date_label}] {hit.role}: {hit.content}"
                if len(text) + 1 > budget - used:
                    continue
                lines.append(text)
                used += len(text) + 1

        return "\n".join(lines)

    # ----------------------------------------------------------------- delete

    def forget(self, memory_id: int) -> bool:
        """Delete one memory (a fact or a message). True when something went."""
        with self.db.write() as conn:
            cursor = conn.execute("DELETE FROM facts WHERE id = ?", (memory_id,))
            if cursor.rowcount:
                return True
            cursor = conn.execute("DELETE FROM messages WHERE id = ?", (memory_id,))
            return bool(cursor.rowcount)

    def clear_session(self, session_id: str | None = None) -> int:
        """Delete every message of this session. Facts are kept."""
        with self.db.write() as conn:
            cursor = conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id or self.session_id,)
            )
            return int(cursor.rowcount or 0)

    def clear_all(self) -> int:
        """Delete everything: messages, facts and summaries."""
        with self.db.write() as conn:
            total = 0
            for table in ("messages", "facts", "summaries"):
                cursor = conn.execute(f"DELETE FROM {table}")
                total += int(cursor.rowcount or 0)
            return total

    def prune(self, retention_days: int | None = None) -> int:
        """Delete messages older than the retention window. 0 keeps everything."""
        days = self.retention_days if retention_days is None else retention_days
        if days <= 0:
            return 0
        cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat(timespec="seconds")
        with self.db.write() as conn:
            cursor = conn.execute("DELETE FROM messages WHERE created_at < ?", (cutoff_iso,))
            return int(cursor.rowcount or 0)

    # ------------------------------------------------------------------ stats

    def stats(self) -> MemoryStats:
        """Counts for the status endpoint and the web interface."""
        messages = self.db.query_one("SELECT COUNT(*) AS n FROM messages")
        facts = self.db.query_one("SELECT COUNT(*) AS n FROM facts")
        summaries = self.db.query_one("SELECT COUNT(*) AS n FROM summaries")
        sessions = self.db.query_one("SELECT COUNT(DISTINCT session_id) AS n FROM messages")
        oldest = self.db.query_one("SELECT MIN(created_at) AS t FROM messages")
        newest = self.db.query_one("SELECT MAX(created_at) AS t FROM messages")
        return MemoryStats(
            messages=int(messages["n"]) if messages else 0,
            facts=int(facts["n"]) if facts else 0,
            summaries=int(summaries["n"]) if summaries else 0,
            sessions=int(sessions["n"]) if sessions else 0,
            size_bytes=self.db.size_bytes(),
            oldest=str(oldest["t"]) if oldest and oldest["t"] else None,
            newest=str(newest["t"]) if newest and newest["t"] else None,
        )

    def touch(self, memory_id: int) -> None:
        """Record that a memory was used (helps show what JARVIS relies on)."""
        with self.db.write() as conn:
            conn.execute(
                "UPDATE facts SET access_count = access_count + 1, last_accessed_at = ?"
                " WHERE id = ?",
                (utc_now(), memory_id),
            )

    def close(self) -> None:
        self.db.close()


def hits_to_dicts(hits: Iterable[MemoryHit]) -> list[dict[str, object]]:
    return [hit.as_dict() for hit in hits]
