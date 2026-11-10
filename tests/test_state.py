"""Multi-turn state: facts survive, identifiers stay masked, nothing is re-asked."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.support import AuthedClient, start


@pytest.fixture(scope="module")
def inner_client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client(inner_client):
    return AuthedClient(inner_client)


@pytest.fixture
def session(client):
    session_id, _ = start(client)
    return session_id


def turn(client, session_id, text):
    return client.post("/api/turn", json={
        "session_id": session_id, "text": text, "language": "en-IN", "consent_confirmed": True,
    })


def test_facts_persist_across_turns_and_can_be_inspected(client, session) -> None:
    first = turn(client, session, "I am in Kerala and my employer has not paid my salary of ₹40,000")
    assert first.status_code == 200, first.text
    data = first.json()
    slots = {item["slot"]: item["value"] for item in data["conversation_state"]["known_facts"]}
    assert slots["jurisdiction"] == "IN-KL"
    assert "40,000" in slots["amount"]

    state = client.get(f"/api/session/{session}/state").json()
    assert state["facts"]["facts"]["jurisdiction"]["value"] == "IN-KL"

    second = turn(client, session, "What can I do about it next?")
    assert second.status_code == 200, second.text
    kept = {item["slot"]: item["value"] for item in second.json()["conversation_state"]["known_facts"]}
    assert kept["jurisdiction"] == "IN-KL"


def test_the_same_question_is_not_asked_twice(client, session) -> None:
    first = turn(client, session, "Am I allowed to record a phone call?").json()
    assert (first.get("next_question") or "").lower().startswith("which state")
    second = turn(client, session, "I am in Tamil Nadu. Is recording allowed?").json()
    assert (second.get("next_question") or "").lower().find("which state") == -1


def test_review_preview_masks_identifiers_by_default(client, session) -> None:
    summary = "Applicant Aadhaar 4321 8765 2109, phone 9876543210"
    masked = client.post("/api/review/preview", json={"session_id": session, "summary": summary}).json()
    assert masked["masked"] is True
    assert "4321" not in masked["body"]
    assert "9876543210" not in masked["body"]

    included = client.post("/api/review/preview", json={
        "session_id": session, "summary": summary, "include_identifiers": True,
    }).json()
    assert included["masked"] is False
    assert "4321 8765 2109" in included["body"]


def test_identifier_in_a_turn_is_reported_to_the_caller(client, session) -> None:
    data = turn(client, session, "My Aadhaar is 4321 8765 2109 and my landlord in Delhi locked me out").json()
    kinds = {item["kind"] for item in data["identifiers_in_message"]}
    assert "aadhaar" in kinds
    assert "4321" not in data["response"]
    assert data["safety"]["level"] == "high"
