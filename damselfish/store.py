from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TargetStats:
    target_id: str
    requests: int = 0
    successes: int = 0
    failures: int = 0
    rate_limits: int = 0
    consecutive_failures: int = 0
    ewma_latency_ms: float | None = None
    last_latency_ms: float | None = None
    last_success_at: float | None = None
    last_failure_at: float | None = None
    last_probe_at: float | None = None
    circuit_open_until: float = 0.0
    last_error: str | None = None
    cap_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["circuit_open"] = self.circuit_open_until > time.time()
        return data


_TARGET_STATS_FIELDS = frozenset(field.name for field in fields(TargetStats))


def _target_stats_from_row(row: sqlite3.Row) -> TargetStats:
    return TargetStats(
        **{key: value for key, value in dict(row).items() if key in _TARGET_STATS_FIELDS}
    )


class Store:
    def __init__(self, path: Path, target_ids: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._initialize(target_ids)

    def _initialize(self, target_ids: list[str]) -> None:
        with self._lock, self._connection:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS target_stats (
                    target_id TEXT PRIMARY KEY,
                    requests INTEGER NOT NULL DEFAULT 0,
                    successes INTEGER NOT NULL DEFAULT 0,
                    failures INTEGER NOT NULL DEFAULT 0,
                    rate_limits INTEGER NOT NULL DEFAULT 0,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    ewma_latency_ms REAL,
                    last_latency_ms REAL,
                    last_success_at REAL,
                    last_failure_at REAL,
                    last_probe_at REAL,
                    circuit_open_until REAL NOT NULL DEFAULT 0,
                    last_error TEXT,
                    cap_count INTEGER NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    messages_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS project_sessions (
                    project_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    source_device TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(project_id, session_id),
                    FOREIGN KEY(project_id) REFERENCES projects(project_id)
                );
                CREATE TABLE IF NOT EXISTS memory_events (
                    event_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    source_device TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    synced INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    session_id TEXT,
                    scenario TEXT NOT NULL,
                    persona TEXT,
                    target_id TEXT,
                    latency_ms REAL,
                    success INTEGER NOT NULL,
                    status INTEGER,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS decisions_created_idx
                    ON decisions(created_at DESC);
                CREATE INDEX IF NOT EXISTS project_sessions_updated_idx
                    ON project_sessions(project_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS memory_events_pending_idx
                    ON memory_events(synced, created_at);
                """
            )
            # Schema migration: add cap_count to existing databases
            try:
                self._connection.execute(
                    "ALTER TABLE target_stats ADD COLUMN cap_count INTEGER NOT NULL DEFAULT 0",
                )
            except Exception:
                pass  # Column already exists
            # Schema migration: add token usage columns to existing databases
            for col in ("prompt_tokens", "completion_tokens", "total_tokens"):
                try:
                    self._connection.execute(
                        f"ALTER TABLE target_stats ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0",
                    )
                except Exception:
                    pass  # Column already exists
            self._connection.executemany(
                "INSERT OR IGNORE INTO target_stats(target_id) VALUES (?)",
                [(target_id,) for target_id in target_ids],
            )

    def stats(self, target_id: str) -> TargetStats:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM target_stats WHERE target_id = ?", (target_id,)
            ).fetchone()
        if row is None:
            raise KeyError(target_id)
        return _target_stats_from_row(row)

    def ensure_targets(self, target_ids: list[str]) -> None:
        with self._lock, self._connection:
            self._connection.executemany(
                "INSERT OR IGNORE INTO target_stats(target_id) VALUES (?)",
                [(target_id,) for target_id in target_ids],
            )

    def all_stats(self) -> dict[str, TargetStats]:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM target_stats").fetchall()
        return {row["target_id"]: _target_stats_from_row(row) for row in rows}

    def record_success(
        self, target_id: str, latency_ms: float, alpha: float, probe: bool = False
    ) -> None:
        now = time.time()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT ewma_latency_ms FROM target_stats WHERE target_id = ?",
                (target_id,),
            ).fetchone()
            previous = row[0] if row else None
            ewma = latency_ms if previous is None else alpha * latency_ms + (1 - alpha) * previous
            self._connection.execute(
                """
                UPDATE target_stats SET
                    requests = requests + ?, successes = successes + ?,
                    consecutive_failures = 0, ewma_latency_ms = ?,
                    last_latency_ms = ?, last_success_at = ?,
                    last_probe_at = CASE WHEN ? THEN ? ELSE last_probe_at END,
                    circuit_open_until = 0, last_error = NULL
                WHERE target_id = ?
                """,
                (0 if probe else 1, 0 if probe else 1, ewma, latency_ms, now, probe, now, target_id),
            )

    def record_failure(
        self,
        target_id: str,
        status: int,
        error: str,
        circuit_open_until: float,
        probe: bool = False,
    ) -> None:
        now = time.time()
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE target_stats SET
                    requests = requests + ?, failures = failures + ?,
                    rate_limits = rate_limits + ?,
                    consecutive_failures = consecutive_failures + 1,
                    last_failure_at = ?,
                    last_probe_at = CASE WHEN ? THEN ? ELSE last_probe_at END,
                    circuit_open_until = ?, last_error = ?
                WHERE target_id = ?
                """,
                (
                    0 if probe else 1,
                    0 if probe else 1,
                    0 if probe or status != 429 else 1,
                    now,
                    probe,
                    now,
                    circuit_open_until,
                    error[:500],
                    target_id,
                ),
            )

    def record_cap(self, target_id: str) -> None:
        """Increment the cap_count counter when max_new_tokens is capped."""
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE target_stats SET cap_count = cap_count + 1 WHERE target_id = ?",
                (target_id,),
            )

    def record_usage(
        self,
        target_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        """Accumulate token usage from upstream ``usage`` fields.

        Called after a successful completion (streaming or non-streaming)
        when the upstream response includes a ``usage`` object.  Values are
        accumulated per-target so the stats endpoint can report totals.
        """
        if not (prompt_tokens or completion_tokens or total_tokens):
            return
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE target_stats SET
                    prompt_tokens = prompt_tokens + ?,
                    completion_tokens = completion_tokens + ?,
                    total_tokens = total_tokens + ?
                WHERE target_id = ?
                """,
                (prompt_tokens, completion_tokens, total_tokens, target_id),
            )

    def record_decision(
        self,
        session_id: str | None,
        scenario: str,
        persona: str | None,
        target_id: str | None,
        latency_ms: float | None,
        success: bool,
        status: int | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO decisions(
                    created_at, session_id, scenario, persona, target_id,
                    latency_ms, success, status, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(), session_id, scenario, persona, target_id,
                    latency_ms, int(success), status, error[:500] if error else None,
                ),
            )

    def recent_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_session(self, session_id: str, ttl_days: int) -> list[dict[str, Any]]:
        return self.get_project_session("default", session_id, ttl_days)

    def get_project_session(
        self, project_id: str, session_id: str, ttl_days: int
    ) -> list[dict[str, Any]]:
        cutoff = time.time() - ttl_days * 86400
        with self._lock:
            row = self._connection.execute(
                """
                SELECT messages_json, updated_at FROM project_sessions
                WHERE project_id = ? AND session_id = ?
                """,
                (project_id, session_id),
            ).fetchone()
            if row is None and project_id == "default":
                row = self._connection.execute(
                    "SELECT messages_json, updated_at FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
        if row is None or row["updated_at"] < cutoff:
            return []
        try:
            messages = json.loads(row["messages_json"])
            return messages if isinstance(messages, list) else []
        except json.JSONDecodeError:
            return []

    def save_session(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        max_messages: int,
        project_id: str = "default",
        project_title: str | None = None,
        session_title: str | None = None,
        source_device: str = "local",
    ) -> dict[str, Any]:
        trimmed = _trim_messages(messages, max_messages)
        now = time.time()
        project_title = project_title or _human_title(project_id)
        session_title = session_title or _session_title(trimmed, session_id)
        event = {
            "schema_version": 1,
            "event_id": uuid.uuid4().hex,
            "project_id": project_id,
            "project_title": project_title,
            "session_id": session_id,
            "session_title": session_title,
            "source_device": source_device,
            "created_at": now,
            "messages": trimmed,
        }
        serialized_messages = json.dumps(trimmed, ensure_ascii=False)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO projects(project_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    title = excluded.title, updated_at = excluded.updated_at
                """,
                (project_id, project_title, now, now),
            )
            self._connection.execute(
                """
                INSERT INTO project_sessions(
                    project_id, session_id, title, messages_json, source_device,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, session_id) DO UPDATE SET
                    title = excluded.title,
                    messages_json = excluded.messages_json,
                    source_device = excluded.source_device,
                    updated_at = excluded.updated_at
                """,
                (
                    project_id, session_id, session_title, serialized_messages,
                    source_device, now, now,
                ),
            )
            self._connection.execute(
                """
                INSERT INTO memory_events(
                    event_id, project_id, session_id, created_at,
                    source_device, snapshot_json, synced
                ) VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    event["event_id"], project_id, session_id, now, source_device,
                    json.dumps(event, ensure_ascii=False),
                ),
            )
            if project_id == "default":
                self._connection.execute(
                    """
                    INSERT INTO sessions(session_id, messages_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        messages_json = excluded.messages_json,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, serialized_messages, now),
                )
        return event


    def update_session_messages(
        self, session_id: str, messages: list[dict[str, Any]],
    ) -> None:
        with self._lock:
            serialized = json.dumps(messages, ensure_ascii=False)
            self._connection.execute(
                "UPDATE sessions SET messages_json = ?, updated_at = ? WHERE session_id = ?",
                (serialized, time.time(), session_id),
            )
            self._connection.execute(
                "UPDATE project_sessions SET messages_json = ?, updated_at = ?"
                " WHERE session_id = ?",
                (serialized, time.time(), session_id),
            )

    def list_projects(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT p.project_id, p.title, p.created_at, p.updated_at,
                       COUNT(s.session_id) AS session_count
                FROM projects p
                LEFT JOIN project_sessions s ON s.project_id = p.project_id
                GROUP BY p.project_id
                ORDER BY p.updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_project_sessions(self, project_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT project_id, session_id, title, source_device, created_at,
                       updated_at, json_array_length(messages_json) AS message_count
                FROM project_sessions WHERE project_id = ?
                ORDER BY updated_at DESC
                """,
                (project_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def project_context(
        self,
        project_id: str,
        exclude_session_id: str,
        session_limit: int = 3,
        message_limit: int = 6,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT session_id, title, messages_json FROM project_sessions
                WHERE project_id = ? AND session_id != ?
                ORDER BY updated_at DESC LIMIT ?
                """,
                (project_id, exclude_session_id, session_limit),
            ).fetchall()
        context = []
        for row in reversed(rows):
            messages = json.loads(row["messages_json"])
            context.append(
                {
                    "session_id": row["session_id"],
                    "title": row["title"],
                    "messages": messages[-message_limit:],
                }
            )
        return context

    def pending_memory_events(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT snapshot_json FROM memory_events WHERE synced = 0 ORDER BY created_at"
            ).fetchall()
        return [json.loads(row["snapshot_json"]) for row in rows]

    def pending_memory_event_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM memory_events WHERE synced = 0"
            ).fetchone()
        return int(row["count"])

    def mark_memory_events_synced(self, event_ids: list[str]) -> None:
        if not event_ids:
            return
        with self._lock, self._connection:
            self._connection.executemany(
                "UPDATE memory_events SET synced = 1 WHERE event_id = ?",
                [(event_id,) for event_id in event_ids],
            )

    def import_memory_event(self, event: dict[str, Any]) -> bool:
        required = {
            "event_id", "project_id", "session_id", "created_at",
            "source_device", "messages",
        }
        if not required.issubset(event) or not isinstance(event["messages"], list):
            raise ValueError("invalid memory event")
        event_id = str(event["event_id"])
        project_id = str(event["project_id"])
        session_id = str(event["session_id"])
        created_at = float(event["created_at"])
        messages = event["messages"]
        project_title = str(event.get("project_title") or _human_title(project_id))
        session_title = str(event.get("session_title") or _session_title(messages, session_id))
        with self._lock, self._connection:
            exists = self._connection.execute(
                "SELECT 1 FROM memory_events WHERE event_id = ?", (event_id,)
            ).fetchone()
            if exists:
                return False
            self._connection.execute(
                """
                INSERT INTO projects(project_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    title = excluded.title,
                    updated_at = MAX(projects.updated_at, excluded.updated_at)
                """,
                (project_id, project_title, created_at, created_at),
            )
            current = self._connection.execute(
                """
                SELECT messages_json, updated_at FROM project_sessions
                WHERE project_id = ? AND session_id = ?
                """,
                (project_id, session_id),
            ).fetchone()
            should_apply = current is None
            if current is not None:
                current_messages = json.loads(current["messages_json"])
                should_apply = len(messages) > len(current_messages) or (
                    len(messages) == len(current_messages)
                    and created_at > current["updated_at"]
                )
            if should_apply:
                self._connection.execute(
                    """
                    INSERT INTO project_sessions(
                        project_id, session_id, title, messages_json, source_device,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(project_id, session_id) DO UPDATE SET
                        title = excluded.title,
                        messages_json = excluded.messages_json,
                        source_device = excluded.source_device,
                        updated_at = excluded.updated_at
                    """,
                    (
                        project_id, session_id, session_title,
                        json.dumps(messages, ensure_ascii=False),
                        str(event["source_device"]), created_at, created_at,
                    ),
                )
            self._connection.execute(
                """
                INSERT INTO memory_events(
                    event_id, project_id, session_id, created_at,
                    source_device, snapshot_json, synced
                ) VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    event_id, project_id, session_id, created_at,
                    str(event["source_device"]), json.dumps(event, ensure_ascii=False),
                ),
            )
        return True

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def merge_messages(
    stored: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not stored:
        return list(incoming)
    if not incoming:
        return list(stored)
    maximum = min(len(stored), len(incoming))
    for overlap in range(maximum, 0, -1):
        if stored[-overlap:] == incoming[:overlap]:
            return stored + incoming[overlap:]
    if len(incoming) >= len(stored) and incoming[: len(stored)] == stored:
        return list(incoming)
    system = [message for message in incoming if message.get("role") == "system"]
    body = [message for message in incoming if message.get("role") != "system"]
    stored_body = [message for message in stored if message.get("role") != "system"]
    return system + stored_body + body


def project_context_message(
    project_id: str, sessions: list[dict[str, Any]], max_chars: int
) -> dict[str, str] | None:
    if not sessions:
        return None
    lines = [f"Damselfish shared memory for project {project_id}:"]
    for session in sessions:
        lines.append(f"\nSession {session['title']} ({session['session_id']}):")
        for message in session["messages"]:
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                lines.append(f"{message.get('role', 'unknown')}: {content.strip()}")
    content = "\n".join(lines)
    if len(content) > max_chars:
        content = content[-max_chars:]
        content = f"Damselfish shared memory for project {project_id} (truncated):\n{content}"
    return {"role": "system", "content": content}


def _trim_messages(messages: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if len(messages) <= maximum:
        return messages
    system = [message for message in messages if message.get("role") == "system"][:1]
    remaining = maximum - len(system)
    return system + [message for message in messages if message.get("role") != "system"][-remaining:]


def _human_title(identifier: str) -> str:
    return identifier.replace("-", " ").replace("_", " ").strip().title() or "Default"


def _session_title(messages: list[dict[str, Any]], fallback: str) -> str:
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            title = " ".join(message["content"].split())
            if title:
                return title[:80]
    return _human_title(fallback)
