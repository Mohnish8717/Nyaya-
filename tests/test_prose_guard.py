"""The spoken sentence is checked, not only the structured claims.

`legal_claims` was validated and `reply` was spoken as-is, so a model could
state a section, a limitation period or a sum in prose and reach the caller
unverified. These tests pin the second gate: an unsupported number in the
reply gets one repair, then the deterministic grounded answer.
"""
from __future__ import annotations

import asyncio

import pytest

from app import legal_engine, model_gateway, understanding
from app.legal_engine import INDIAN_JURISDICTIONS


def interpretation(text, *, intent="statute", facts=None, procedure_id=None,
                   confidence=0.9, issue="", next_question=None):
    raw = {
        "intent": intent,
        "confidence": confidence,
        "language": "en-IN",
        "risk": [],
        "facts": facts or [],
        "procedure_id": procedure_id,
        "issue": issue,
        "next_question": next_question,
        "needs_precedents": False,
    }
    validated, _ = understanding.validate(raw, text, INDIAN_JURISDICTIONS)
    return understanding.combine(validated, text)


def empty_session():
    return {"turns": 0, "facts": {}, "last_state": {}}


def test_section_and_period_tokens_are_comparable() -> None:
    tokens = legal_engine._numeric_tokens("Under section 138 you may file within 30 days.")
    assert "s:138" in tokens
    assert "u:30d" in tokens


def test_a_number_the_caller_said_is_not_a_gap() -> None:
    assert legal_engine._prose_gaps(
        "You said 30 days have passed since the cheque.", [], "It has been 30 days"
    ) == []


def test_an_unsupported_section_in_prose_is_a_gap() -> None:
    gaps = legal_engine._prose_gaps("This is covered by section 138 of the Act.", [], "my cheque bounced")
    assert "s:138" in gaps


def test_an_emergency_short_code_is_never_a_gap() -> None:
    assert legal_engine._prose_gaps("Call 112 now.", [], "he is outside my door") == []


def test_a_quote_that_omits_the_number_does_not_verify_the_claim() -> None:
    source = {"id": "s1", "text": "The cheque shall be presented within six months."}
    claims, rejected = legal_engine._claims(
        [{"text": "Section 138 governs dishonour.", "source_id": "s1",
          "quote": "The cheque shall be presented within six months."}],
        [source],
    )
    assert claims == []
    assert any("quote_omits" in item for item in rejected)


def test_a_quote_with_the_number_still_verifies() -> None:
    source = {"id": "s1", "text": "A notice under section 138 must be given within 30 days."}
    claims, rejected = legal_engine._claims(
        [{"text": "Notice is required under section 138 within 30 days.", "source_id": "s1",
          "quote": "A notice under section 138 must be given within 30 days."}],
        [source],
    )
    assert len(claims) == 1
    assert rejected == []


def test_verify_claims_reads_verdicts_and_keeps_skipped_items() -> None:
    async def fake_completion(*args, **kwargs):
        return {"verdicts": [{"index": 0, "supported": True}, {"index": 1, "supported": False}]}

    calls = {"n": 0}

    def fake_available():
        calls["n"] += 1
        return True

    original = model_gateway.entailment_available
    original_completion = model_gateway._json_completion
    model_gateway.entailment_available = fake_available
    model_gateway._json_completion = fake_completion
    try:
        verdicts = asyncio.run(model_gateway.verify_claims([
            {"text": "a", "quote": "qa"}, {"text": "b", "quote": "qb"}, {"text": "c", "quote": "qc"},
        ]))
    finally:
        model_gateway.entailment_available = original
        model_gateway._json_completion = original_completion
    assert verdicts == {0: True, 1: False}
    assert calls["n"] == 1


def test_verify_claims_returns_none_when_it_cannot_run() -> None:
    original = model_gateway.entailment_available
    model_gateway.entailment_available = lambda: False
    try:
        assert asyncio.run(model_gateway.verify_claims([{"text": "a", "quote": "q"}])) is None
    finally:
        model_gateway.entailment_available = original


def test_a_section_in_prose_without_a_claim_is_not_spoken(monkeypatch) -> None:
    text = "my cheque bounced, what can I do"
    calls = []

    async def fake_generate(context, **kwargs):
        calls.append(dict(context))
        return {
            "intent": "statute",
            "reply": "Under section 138 of the Negotiable Instruments Act you can send a notice within 30 days.",
            "language": "en-IN",
            "legal_claims": [],
        }

    monkeypatch.setattr(legal_engine, "model_configured", lambda: True)
    monkeypatch.setattr(legal_engine, "generate_turn", fake_generate)

    result = asyncio.run(legal_engine.answer_turn(text, "en-IN", "auto", None, [], empty_session()))

    assert result["mode"] == "grounded_only"
    assert "138" not in result["response"]
    assert result["verification"]["status"] in {"retrieved_uninterpreted", "no_supported_legal_claim"}
    # One call, then the deterministic answer. The old design spent a second
    # full generation asking the model to take the number out; on a throttled
    # provider that was the turn's whole budget, and the caller still got
    # passages. The number simply is not spoken.
    assert len(calls) == 1
