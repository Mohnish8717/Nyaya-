"""A session id is not a credential.

The id appears in a URL path, a query string, a proxy log and the browser
history. Anyone holding it could read the transcript, delete the conversation
or spend the daily budget. Every authenticated route now needs the token that
was issued once, at creation, and only its hash is stored.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import sessions
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def new_session(client) -> tuple[str, str]:
    response = client.post("/api/session", json={"language": "en-IN", "consent_confirmed": True})
    assert response.status_code == 200, response.text
    data = response.json()
    return data["session_id"], data["session_token"]


def test_creation_returns_a_token_that_is_not_the_id(client) -> None:
    session_id, token = new_session(client)
    assert token and token != session_id
    assert len(token) >= 32


def test_the_token_is_not_stored_in_the_clear(client) -> None:
    session_id, token = new_session(client)
    stored = sessions.get(session_id)
    assert stored is not None
    # The row never carries the token itself, so a database read cannot mint a
    # working credential.
    assert token not in str(stored)


def test_a_turn_without_the_token_is_refused(client) -> None:
    session_id, _ = new_session(client)
    response = client.post("/api/turn", json={
        "session_id": session_id, "text": "hello", "language": "en-IN", "consent_confirmed": True,
    })
    assert response.status_code == 401


def test_a_wrong_token_is_refused(client) -> None:
    session_id, _ = new_session(client)
    response = client.post("/api/turn", json={
        "session_id": session_id, "text": "hello", "language": "en-IN", "consent_confirmed": True,
    }, headers={"Authorization": "Bearer not-the-token"})
    assert response.status_code == 401


def test_the_right_token_is_accepted(client) -> None:
    session_id, token = new_session(client)
    response = client.post("/api/turn", json={
        "session_id": session_id, "text": "how do I file an FIR at the police station",
        "language": "en-IN", "consent_confirmed": True,
    }, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200, response.text


def test_state_artifacts_and_deletion_all_need_the_token(client) -> None:
    session_id, token = new_session(client)
    for method, url, kwargs in (
        ("get", f"/api/session/{session_id}/state", {}),
        ("get", f"/api/session/{session_id}/artifacts", {}),
        ("delete", f"/api/session/{session_id}", {}),
        ("post", "/api/review/preview", {"json": {"session_id": session_id, "summary": "x"}}),
        ("post", "/api/voice/speech", {"json": {"session_id": session_id, "text": "hi"}}),
    ):
        response = getattr(client, method)(url, **kwargs)
        assert response.status_code == 401, f"{method} {url} was not protected"


def test_verify_token_rejects_an_empty_or_foreign_token(client) -> None:
    session_id, token = new_session(client)
    other_id, other_token = new_session(client)
    assert sessions.verify_token(session_id, "") is False
    assert sessions.verify_token(session_id, other_token) is False
    assert sessions.verify_token(other_id, token) is False
    assert sessions.verify_token(session_id, token) is True


def test_deleting_with_the_token_removes_the_session(client) -> None:
    session_id, token = new_session(client)
    response = client.delete(f"/api/session/{session_id}", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert sessions.get(session_id) is None
