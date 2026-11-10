"""Session store and limiter behaviour."""
from __future__ import annotations

import pytest

from app import sessions


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "DB_PATH", tmp_path / "sessions.sqlite3")
    sessions.init()
    return sessions


def test_session_roundtrip(store) -> None:
    session_id, _token = store.create("hi-IN")
    session = store.get(session_id)
    assert session is not None
    assert session["language"] == "hi-IN"
    assert session["consent_version"] == store.CONSENT_NOTICE_VERSION
    assert session["consent_at"] > 0


def test_history_is_capped(store) -> None:
    session_id, _token = store.create("en-IN")
    for index in range(12):
        store.record_turn(session_id, f"question {index}", f"answer {index}", "statute", "en-IN")
    session = store.get(session_id)
    assert len(session["history"]) == store.HISTORY_MESSAGES
    assert session["turns"] == 12
    assert session["route_counts"]["statute"] == 12


def test_artifacts_are_session_scoped(store) -> None:
    first, _ = store.create("en-IN")
    second, _ = store.create("en-IN")
    store.add_artifact(first, {"payload": "one"})
    assert store.artifacts(first)["artifacts"]
    assert store.artifacts(second)["artifacts"] == []


def test_delete_is_permanent(store) -> None:
    session_id, _token = store.create("en-IN")
    store.delete(session_id)
    assert store.get(session_id) is None


def test_limiter_enforces_a_window(store) -> None:
    # The counters are shared state now, so this test gets its own database
    # instead of inheriting rows from a previous run.
    limiter = store.Limiter()
    assert limiter.allow("k", 2, 60)[0] is True
    assert limiter.allow("k", 2, 60)[0] is True
    allowed, remaining = limiter.allow("k", 2, 60)
    assert allowed is False and remaining == 0


def test_two_workers_share_one_budget(store) -> None:
    worker_a = store.Limiter()
    worker_b = store.Limiter()
    assert worker_a.allow("shared", 2, 60)[0] is True
    assert worker_b.allow("shared", 2, 60)[0] is True
    # In-memory counters used to double the real limit to four.
    assert worker_a.allow("shared", 2, 60)[0] is False


def test_daily_budget(store) -> None:
    limiter = store.Limiter()
    assert limiter.daily_allow(1) is True
    assert limiter.daily_allow(1) is False
    assert limiter.daily_allow(0) is True
