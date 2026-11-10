"""Session persistence, shared rate limiting, and the live speech turn.

Sessions are stored in SQLite (WAL) instead of a module-level dict so that a
restart does not silently destroy a conversation, and so two workers do not
disagree about whether a session exists. The store is still deliberately
small: transcript text is capped, expires on a timer, and is never written
anywhere except this local database — there is no transcript archive.

Rate limiting has three independent budgets: per session, per client address,
and an optional global daily turn budget. The counters live in this database,
not in process memory: with two workers, in-memory counters meant the real
limit was double and a restarted worker handed out a fresh allowance.

The same applies to the speech turn a barge-in has to stop: it is a row here,
so a caller whose audio is streaming from worker B is still stopped when the
interrupt lands on worker A.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / ".runtime" / "sessions.sqlite3"

SESSION_TTL_SECONDS = 4 * 60 * 60
MAX_SESSIONS = 5000
HISTORY_MESSAGES = 12
HISTORY_CHARS = 6000
# Compaction: turns that fall out of the live window are folded into a short
# rolling digest instead of vanishing. Entries are bounded so a long call
# cannot grow context linearly; the fact ledger carries the structured state,
# so the digest only has to preserve the narrative thread.
EARLIER_MESSAGES = 40
EARLIER_CHARS = 240
ARTIFACT_LIMIT = 20
CONSENT_NOTICE_VERSION = "consent-2026-11"


_INITIALISED = False


def ensure() -> None:
    """Create the schema on first use, whichever module touches it first."""
    if not _INITIALISED:
        init()


def _connect() -> sqlite3.Connection:
    ensure()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def init() -> None:
    global _INITIALISED
    # Set before connecting: _connect() calls ensure(), and this is what stops
    # that from recursing on the very first connection.
    _INITIALISED = True
    with _connect() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                language TEXT NOT NULL,
                created REAL NOT NULL,
                consent_at REAL NOT NULL,
                consent_version TEXT NOT NULL,
                history TEXT NOT NULL DEFAULT '[]',
                artifacts TEXT NOT NULL DEFAULT '[]',
                reminders TEXT NOT NULL DEFAULT '[]',
                route_counts TEXT NOT NULL DEFAULT '{}',
                turns INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        db.execute("CREATE INDEX IF NOT EXISTS sessions_created ON sessions(created)")
        columns = {row["name"] for row in db.execute("PRAGMA table_info(sessions)")}
        # Fact ledger is additive: an existing database keeps its sessions.
        if "facts" not in columns:
            db.execute("ALTER TABLE sessions ADD COLUMN facts TEXT NOT NULL DEFAULT '{}'")
        if "last_state" not in columns:
            db.execute("ALTER TABLE sessions ADD COLUMN last_state TEXT NOT NULL DEFAULT '{}'")
        # A session id alone used to be the whole credential. It travels in a
        # URL path, a query string and the browser history, so it is not a
        # secret. The token below never enters a URL except the speech socket,
        # which cannot send headers, and only its hash is stored.
        if "token_hash" not in columns:
            db.execute("ALTER TABLE sessions ADD COLUMN token_hash TEXT NOT NULL DEFAULT ''")
        # Rolling digest of turns dropped from the live window (additive like
        # the facts column: an existing database keeps its sessions).
        if "earlier" not in columns:
            db.execute("ALTER TABLE sessions ADD COLUMN earlier TEXT NOT NULL DEFAULT '[]'")
        # Shared rate limiting. One row per admitted request, pruned by age.
        db.execute("CREATE TABLE IF NOT EXISTS rate (key TEXT NOT NULL, ts REAL NOT NULL)")
        db.execute("CREATE INDEX IF NOT EXISTS rate_key ON rate(key, ts)")
        # The utterance currently playing, so any worker can stop it.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS speech (
                session_id TEXT PRIMARY KEY,
                speech_id TEXT NOT NULL,
                text TEXT NOT NULL DEFAULT '',
                started_at REAL NOT NULL,
                cancelled_at REAL,
                reason TEXT,
                heard_ms INTEGER NOT NULL DEFAULT 0,
                spoken TEXT NOT NULL DEFAULT '',
                truncated INTEGER NOT NULL DEFAULT 0
            )
            """
        )


def _prune(db: sqlite3.Connection) -> None:
    cutoff = time.time() - SESSION_TTL_SECONDS
    db.execute("DELETE FROM sessions WHERE created < ?", (cutoff,))
    count = db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    if count > MAX_SESSIONS:
        db.execute(
            "DELETE FROM sessions WHERE id IN (SELECT id FROM sessions ORDER BY created ASC LIMIT ?)",
            (count - MAX_SESSIONS,),
        )


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create(language: str, consent_version: str = CONSENT_NOTICE_VERSION) -> tuple[str, str]:
    """Create a session. Returns (session_id, token); the token is shown once."""
    session_id = secrets.token_urlsafe(18)
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _connect() as db:
        _prune(db)
        db.execute(
            "INSERT INTO sessions(id, language, created, consent_at, consent_version, token_hash) "
            "VALUES(?,?,?,?,?,?)",
            (session_id, language, now, now, consent_version, _hash_token(token)),
        )
    return session_id, token


def verify_token(session_id: str, token: str) -> bool:
    """Constant-time check that `token` belongs to `session_id`.

    An empty stored hash (a session created before tokens existed) matches
    nothing, so those sessions must be restarted rather than being silently
    open.
    """
    if not session_id or not token:
        return False
    with _connect() as db:
        row = db.execute("SELECT token_hash FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not row or not row["token_hash"]:
        return False
    return hmac.compare_digest(str(row["token_hash"]), _hash_token(token))


def _row_to_session(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "language": row["language"],
        "created": row["created"],
        "consent_at": row["consent_at"],
        "consent_version": row["consent_version"],
        "history": json.loads(row["history"] or "[]"),
        "earlier": json.loads(row["earlier"] if "earlier" in row.keys() else "[]"),
        "artifacts": json.loads(row["artifacts"] or "[]"),
        "reminders": json.loads(row["reminders"] or "[]"),
        "route_counts": json.loads(row["route_counts"] or "{}"),
        "facts": json.loads(row["facts"] or "{}"),
        "last_state": json.loads(row["last_state"] or "{}"),
        "turns": row["turns"],
    }


def get(session_id: str) -> dict[str, Any] | None:
    with _connect() as db:
        row = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not row:
        return None
    session = _row_to_session(row)
    if time.time() - session["created"] > SESSION_TTL_SECONDS:
        delete(session_id)
        return None
    return session


def delete(session_id: str) -> None:
    with _connect() as db:
        db.execute("DELETE FROM sessions WHERE id=?", (session_id,))


def record_turn(session_id: str, user_text: str, assistant_text: str, route: str, language: str) -> None:
    session = get(session_id)
    if not session:
        return
    history = list(session["history"])
    history.append({"role": "user", "content": str(user_text)[:HISTORY_CHARS], "at": time.time()})
    history.append({"role": "assistant", "content": str(assistant_text)[:HISTORY_CHARS], "at": time.time()})
    # Compaction: turn the messages that fall out of the live window into a
    # bounded digest instead of dropping them, so turn 20 of a call still
    # knows what turn 3 was about. Each entry is one short labelled line;
    # privacy masking happens downstream at the same boundary as history.
    earlier = list(session.get("earlier") or [])
    if len(history) > HISTORY_MESSAGES:
        overflow = history[:-HISTORY_MESSAGES]
        history = history[-HISTORY_MESSAGES:]
        for item in overflow:
            label = "caller" if item.get("role") == "user" else "assistant"
            content = str(item.get("content") or "").strip()[: max(EARLIER_CHARS - len(label) - 2, 80)]
            if not content:
                continue
            earlier.append({"role": item.get("role"), "content": f"{label}: {content}"})
        earlier = earlier[-EARLIER_MESSAGES:]
    counts = dict(session["route_counts"])
    counts[route] = counts.get(route, 0) + 1
    with _connect() as db:
        db.execute(
            "UPDATE sessions SET history=?, earlier=?, route_counts=?, turns=turns+1, language=? WHERE id=?",
            (json.dumps(history, ensure_ascii=False), json.dumps(earlier, ensure_ascii=False),
             json.dumps(counts), language, session_id),
        )


def save_state(session_id: str, ledger: dict[str, Any], last_state: dict[str, Any] | None = None) -> None:
    """Persist the fact ledger so a restart does not forget what the caller said."""
    fields = ["facts=?"]
    values: list[Any] = [json.dumps(ledger, ensure_ascii=False)]
    if last_state is not None:
        fields.append("last_state=?")
        values.append(json.dumps(last_state, ensure_ascii=False))
    values.append(session_id)
    with _connect() as db:
        db.execute(f"UPDATE sessions SET {', '.join(fields)} WHERE id=?", values)


def state(session_id: str) -> dict[str, Any] | None:
    session = get(session_id)
    if not session:
        return None
    return {
        "session_id": session_id,
        "turns": session["turns"],
        "facts": session["facts"],
        "last_state": session["last_state"],
        "earlier_turns": len(session.get("earlier") or []),
        "retention": "local session store; expires after 4 hours or on deletion",
    }


def add_artifact(session_id: str, item: dict[str, Any]) -> dict[str, Any] | None:
    session = get(session_id)
    if not session:
        return None
    artifacts = (session["artifacts"] + [item])[-ARTIFACT_LIMIT:]
    with _connect() as db:
        db.execute("UPDATE sessions SET artifacts=? WHERE id=?", (json.dumps(artifacts, ensure_ascii=False), session_id))
    return item


def add_reminder(session_id: str, item: dict[str, Any]) -> dict[str, Any] | None:
    session = get(session_id)
    if not session:
        return None
    reminders = (session["reminders"] + [item])[-ARTIFACT_LIMIT:]
    with _connect() as db:
        db.execute("UPDATE sessions SET reminders=? WHERE id=?", (json.dumps(reminders, ensure_ascii=False), session_id))
    return item


def artifacts(session_id: str) -> dict[str, Any] | None:
    session = get(session_id)
    if not session:
        return None
    return {
        "artifacts": session["artifacts"],
        "reminders": session["reminders"],
        "retention": "local session store; expires after 4 hours or on deletion",
        "consent": {"given_at": session["consent_at"], "notice_version": session["consent_version"]},
    }


class Limiter:
    """Sliding-window counters in the shared store, not in this process.

    Each admission is one short `BEGIN IMMEDIATE` transaction, so two workers
    cannot both read `limit - 1` and both admit the last request. SQLite in WAL
    mode is the same file the sessions already use: one less thing to run.
    """

    def allow(self, key: str, limit: int, window_seconds: int = 60) -> tuple[bool, int]:
        if limit <= 0:
            return True, 0
        now = time.time()
        db = _connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM rate WHERE key=? AND ts <= ?", (key, now - window_seconds))
            used = db.execute("SELECT COUNT(*) FROM rate WHERE key=?", (key,)).fetchone()[0]
            if used >= limit:
                db.commit()
                return False, 0
            db.execute("INSERT INTO rate(key, ts) VALUES(?,?)", (key, now))
            db.commit()
            return True, limit - used - 1
        except sqlite3.Error:
            # Failing closed on a shared counter would take the whole app down
            # with the limiter. The per-IP budget is the backstop.
            return True, 0
        finally:
            db.close()

    def clear(self, prefix: str = "") -> None:
        """Drop counters, all of them or one key family.

        Counters live in the shared store, so they outlive a process: without
        this, a second test run inside the same minute inherits the first one's
        spent budget. Never raises; a limiter that fails is not worth a crash.
        """
        db = _connect()
        try:
            if prefix:
                db.execute("DELETE FROM rate WHERE key LIKE ?", (f"{prefix}%",))
            else:
                db.execute("DELETE FROM rate")
            db.commit()
        except sqlite3.Error:
            pass
        finally:
            db.close()

    def daily_allow(self, limit: int) -> bool:
        if limit <= 0:
            return True
        today = time.strftime("%Y-%m-%d")
        allowed, _ = self.allow(f"daily:{today}", limit, 86400)
        db = _connect()
        try:
            # One cheap sweep a day keeps the table from growing forever.
            db.execute("DELETE FROM rate WHERE ts < ?", (time.time() - 2 * 86400,))
        except sqlite3.Error:
            pass
        finally:
            db.close()
        return allowed

    def snapshot(self) -> dict[str, Any]:
        db = _connect()
        try:
            keys = db.execute("SELECT COUNT(DISTINCT key) FROM rate").fetchone()[0]
            today = db.execute(
                "SELECT COUNT(*) FROM rate WHERE key=?", (f"daily:{time.strftime('%Y-%m-%d')}",)
            ).fetchone()[0]
        finally:
            db.close()
        return {"tracked_keys": keys, "daily_turns": today}


def save_speech(session_id: str, speech_id: str, text: str, started_at: float) -> None:
    db = _connect()
    try:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT INTO speech(session_id, speech_id, text, started_at) VALUES(?,?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET speech_id=excluded.speech_id, "
            "text=excluded.text, started_at=excluded.started_at, cancelled_at=NULL, "
            "reason=NULL, heard_ms=0, spoken='', truncated=0",
            (session_id, speech_id, text[:4000], started_at),
        )
        db.commit()
    finally:
        db.close()


def load_speech(session_id: str) -> dict[str, Any] | None:
    db = _connect()
    try:
        row = db.execute("SELECT * FROM speech WHERE session_id=?", (session_id,)).fetchone()
    finally:
        db.close()
    if not row:
        return None
    return {
        "speech_id": row["speech_id"],
        "text": row["text"],
        "started_at": row["started_at"],
        "cancelled_at": row["cancelled_at"],
        "reason": row["reason"],
        "heard_ms": row["heard_ms"],
        "spoken": row["spoken"],
        "truncated": bool(row["truncated"]),
    }


def interrupt_speech(
    session_id: str, speech_id: str, *, reason: str, heard_ms: int, spoken: str, truncated: bool,
) -> bool:
    """Mark the current utterance cancelled. False when it already was."""
    db = _connect()
    try:
        db.execute("BEGIN IMMEDIATE")
        cursor = db.execute(
            "UPDATE speech SET cancelled_at=?, reason=?, heard_ms=?, spoken=?, truncated=? "
            "WHERE session_id=? AND speech_id=? AND cancelled_at IS NULL",
            (time.time(), reason, int(heard_ms), spoken, 1 if truncated else 0, session_id, speech_id),
        )
        db.commit()
        return cursor.rowcount > 0
    finally:
        db.close()


def clear_speech(session_id: str, speech_id: str | None = None) -> None:
    db = _connect()
    try:
        if speech_id:
            db.execute("DELETE FROM speech WHERE session_id=? AND speech_id=?", (session_id, speech_id))
        else:
            db.execute("DELETE FROM speech WHERE session_id=?", (session_id,))
    finally:
        db.close()


LIMITER = Limiter()
