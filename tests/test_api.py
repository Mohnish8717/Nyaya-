"""End-to-end API behaviour, including the degraded mode with no model."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import dense, sessions
from app.main import app
from app.model_gateway import configured as llm_configured
from app.model_gateway import configured as model_configured
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


def turn(client, session_id, text, **extra):
    payload = {"session_id": session_id, "text": text, "language": "en-IN", "consent_confirmed": True}
    payload.update(extra)
    return client.post("/api/turn", json=payload)


def test_health_reports_honest_capabilities(client) -> None:
    data = client.get("/api/health").json()
    assert data["ok"] is True
    assert data["procedures"]["total"] >= 8
    # Health must report the capability that is actually present, not a
    # constant: the dense index is built on this machine, so it is true here.
    assert data["dense_retrieval"] == dense.available()
    assert data["semantic_reading"] == llm_configured()
    assert data["sessions_backend"] == "sqlite"


def test_session_requires_consent(client) -> None:
    assert client.post("/api/session", json={"language": "en-IN", "consent_confirmed": False}).status_code == 400


def test_turn_requires_consent(client, session) -> None:
    response = client.post("/api/turn", json={"session_id": session, "text": "hello", "consent_confirmed": False})
    assert response.status_code == 400


def test_procedure_turn_needs_no_model(client, session, monkeypatch) -> None:
    monkeypatch.setattr("app.legal_engine.model_configured", lambda: False)
    response = turn(client, session, "how do I file an FIR at the police station")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["route"]["name"] == "procedural"
    assert data["procedure"]["id"] == "police-fir"
    assert data["answer_status"] == "procedure_grounded"
    assert data["mode"] == "procedure"
    assert data["next_question"]
    assert len(data["procedure"]["steps"]) >= 3
    assert data["verification"]["status"] == "curated_procedure"
    # A draft procedure must never be presented as reviewed.
    assert data["procedure"]["review_status"] == "draft"
    assert data["procedure"]["reviewed"] is False


def test_emergency_turn_returns_safety_script(client, session) -> None:
    response = turn(client, session, "my husband is beating me and he is coming now")
    assert response.status_code == 200
    data = response.json()
    assert data["route"]["name"] == "emergency"
    assert data["answer_status"] == "emergency_safety"
    assert "112" in data["response"]
    assert "immediate_safety" in data["risk_flags"]
    assert "triage:immediate_danger" in data["risk_flags"]
    assert data["safety"]["level"] == "critical"
    assert data["safety"]["blocks_drafting"] is True
    assert data["claims"] == []


def test_handoff_turn_offers_a_person_but_connects_nobody(client, session) -> None:
    response = turn(client, session, "I want to talk to a lawyer")
    data = response.json()
    assert data["human_connection"]["offered"] is True
    assert data["human_connection"]["connected"] is False


@pytest.mark.skipif(model_configured(), reason="model configured: grounded-only mode is not active")
def test_statutory_turn_degrades_honestly_without_a_model(client, session) -> None:
    response = turn(client, session, "what does section 138 of the Negotiable Instruments Act say")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "grounded_only"
    assert data["answer_status"] == "no_supported_legal_claim"
    assert data["verification"]["status"] in {"no_supported_legal_claim", "retrieved_uninterpreted"}
    # It must not present an uninterpreted passage as a legal conclusion.
    assert data["claims"] == []


def test_stream_emits_stage_events_and_a_final( client, session) -> None:
    response = client.post("/api/turn/stream", json={
        "session_id": session, "text": "how do I file an FIR", "language": "en-IN", "consent_confirmed": True,
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: open" in body
    assert "event: final" in body


def test_delete_removes_the_session(client, session) -> None:
    assert client.delete(f"/api/session/{session}").status_code == 200
    assert client.get(f"/api/session/{session}/artifacts").status_code == 404


def test_procedure_catalog_endpoint(client) -> None:
    data = client.get("/api/procedures").json()
    assert data["procedures"]
    assert all("review_status" in item for item in data["procedures"])


def test_speech_is_registered_before_it_can_be_interrupted(client, session) -> None:
    started = client.post("/api/voice/speech", json={"session_id": session, "text": "Here are the three steps you need."})
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["speech_id"].startswith("sp_")
    assert body["speak_after_ms"] >= 0
    assert body["backoff_ms"] > 0
    assert body["stop_speaking"]["voice_seconds"] > 0

    assert client.post("/api/voice/speech", json={"session_id": "x" * 20, "text": "hi"}).status_code == 404


def test_backchannel_does_not_stop_the_assistant(client, session) -> None:
    speech = client.post("/api/voice/speech", json={"session_id": session, "text": "The first step is to write a complaint."}).json()
    result = client.post("/api/voice/interrupt", json={
        "session_id": session, "speech_id": speech["speech_id"], "partial_text": "haan", "vad_ms": 500,
    }).json()
    assert result["stop"] is False
    assert result["reason"] == "acknowledgement"


def test_stop_word_stops_and_truncates_the_record(client, session) -> None:
    text = "Under Section 138 of the Negotiable Instruments Act a dishonoured cheque can support a complaint if the notice is sent within thirty days of the return memo."
    speech = client.post("/api/voice/speech", json={"session_id": session, "text": text}).json()
    result = client.post("/api/voice/interrupt", json={
        "session_id": session, "speech_id": speech["speech_id"],
        "partial_text": "ruko ek minute", "vad_ms": 120, "heard_ms": 1200,
    }).json()
    assert result["stop"] is True
    assert result["reason"] == "interruption_phrase"
    assert result["speech"]["truncated"] is True
    assert result["speech"]["text"].endswith("…")
    assert len(result["speech"]["text"]) < len(text)
    assert result["resume_after_ms"] > 0


def test_voice_onset_stops_without_waiting_for_words(client, session) -> None:
    speech = client.post("/api/voice/speech", json={"session_id": session, "text": "You can file this at the district court."}).json()
    onset = client.post("/api/voice/interrupt", json={
        "session_id": session, "speech_id": speech["speech_id"], "partial_text": "",
        "vad_ms": 260, "reason": "caller_spoke",
    }).json()
    assert onset["stop"] is True
    assert onset["reason"] == "voice_activity"

    # An explicit Stop control always stops, even with silence on the line.
    second = client.post("/api/voice/speech", json={"session_id": session, "text": "Another answer."}).json()
    manual = client.post("/api/voice/interrupt", json={
        "session_id": session, "speech_id": second["speech_id"], "reason": "button",
    }).json()
    assert manual["stop"] is True
    assert manual["reason"] == "button"


def test_a_stale_speech_id_cannot_stop_the_current_utterance(client, session) -> None:
    first = client.post("/api/voice/speech", json={"session_id": session, "text": "An older answer."}).json()
    second = client.post("/api/voice/speech", json={"session_id": session, "text": "The current answer."}).json()
    stale = client.post("/api/voice/interrupt", json={
        "session_id": session, "speech_id": first["speech_id"], "partial_text": "stop",
    }).json()
    assert stale["stop"] is False
    assert stale["reason"] == "stale_speech"
    assert stale["speech_id"] == second["speech_id"]


def test_speech_metrics_distinguish_server_and_client_stages(client, session) -> None:
    speech = client.post("/api/voice/speech", json={"session_id": session, "text": "A measured answer."}).json()
    client.post("/api/voice/interrupt", json={
        "session_id": session, "speech_id": speech["speech_id"], "partial_text": "no",
    })
    client.post("/api/voice/metrics", json={"speech_id": speech["speech_id"], "first_audio_ms": 640, "stopped_ms": 90})
    data = client.get("/api/voice/metrics").json()
    assert "interrupt_decision_ms" in data["stages"]
    assert data["stages"]["first_audio_ms"]["p50"] == 640
    assert data["plans"]["stop_speaking"]["num_words"] == 0
    assert "browser" in data["note"]


def test_session_creation_has_its_own_budget(inner_client) -> None:
    """A session that never sends a turn never reaches the per-turn budget, so
    creation is metered on its own address key."""
    sessions.LIMITER.clear("session-create:")
    codes = [
        inner_client.post("/api/session", json={"language": "en-IN", "consent_confirmed": True}).status_code
        for _ in range(70)
    ]
    assert 429 in codes, "session creation is unmetered"
    assert codes.index(429) >= 60, f"budget engaged too early at {codes.index(429) + 1}"


def test_debug_analysis_is_metered(client, monkeypatch) -> None:
    """Unauthenticated is fine (it reads nothing stored); unmetered is not: it
    runs retrieval, so it carries an address budget."""
    from types import SimpleNamespace

    async def fake_analyze(*a, **k):
        return SimpleNamespace(
            route={}, risk={}, jurisdiction="IN", procedure=None, sources=[], cases=[],
            ledger=SimpleNamespace(as_dict=lambda: {}), trace=[]
        )

    monkeypatch.setattr("app.main.analyze", fake_analyze)
    codes = [client.get("/api/debug/analysis", params={"text": "cheque bounce"}).status_code
             for _ in range(70)]
    assert 429 in codes, "analysis endpoint has no budget"


def test_intake_workflow_guides_turns_and_persists(client, session) -> None:
    first = turn(client, session, "how do I file an FIR at the police station").json()
    assert first["workflow"]["id"] == "police-fir"
    assert first["workflow"]["step"] == 1
    assert "refuse in writing" in (first["next_question"] or "").lower()

    second = turn(client, session, "The police refused to register it, verbally").json()
    assert second["workflow"]["id"] == "police-fir"
    assert second["workflow"]["step"] == 2
    assert "written refusal" in (second["next_question"] or "").lower()


def test_export_pdf_endpoint(client) -> None:
    resp = client.post(
        "/api/export/pdf",
        json={
            "title": "Application under Section 156(3) CrPC",
            "content": "IN THE COURT OF THE LEARNED MAGISTRATE\n\nMOST RESPECTFULLY SHOWETH:\n1. That offence was committed.\n\nPRAYER:\nWherefore, direct FIR.\n\nAPPLICANT\nVERIFICATION:\nVerified at Delhi.\nDEPONENT",
            "document_type": "application_156_3",
            "warning": "Statutory advocate notice.",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "attachment" in resp.headers["content-disposition"]
    assert resp.content.startswith(b"%PDF")


def test_export_session_artifact_pdf_endpoint(client, session) -> None:
    session_id = session
    confirm = client.post(
        "/api/action/confirm",
        json={
            "session_id": session_id,
            "action": "save_artifact",
            "payload": {
                "type": "bail_application",
                "title": "Bail Application under Section 439 CrPC",
                "content": "IN THE COURT OF SESSIONS JUDGE\n\nAPPLICATION FOR REGULAR BAIL\n\nMOST RESPECTFULLY SHOWETH:\n1. That the applicant is innocent.\n\nPRAYER:\nRelease on bail.\n\nAPPLICANT\nVERIFICATION:\nVerified.\nDEPONENT",
            },
            "user_confirmed": True,
        },
    )
    assert confirm.status_code == 200

    pdf_resp = client.get(f"/api/session/{session_id}/artifact/0/pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert pdf_resp.content.startswith(b"%PDF")

    unauth = client._inner.get(f"/api/session/{session_id}/artifact/0/pdf")
    assert unauth.status_code == 401
