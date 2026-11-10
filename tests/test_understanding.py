"""Contracts for the semantic reader.

These tests do not call a model. They pin what the application is allowed to
do with a model's reading: closed vocabularies, grounded evidence, monotone
safety, and an honest fallback when the provider is unavailable.
"""
from __future__ import annotations

import asyncio

import pytest

from app import facts, legal_engine, model_gateway, routing, understanding
from app.understanding import Interpretation, validate

JURISDICTIONS = {"IN", "IN-KA", "IN-MH", "IN-UP", "IN-DL", "IN-TN", "IN-BR", "IN-WB"}


def reading(**overrides):
    base = {
        "intent": "statute",
        "confidence": 0.8,
        "language": "hi-IN",
        "risk": [],
        "facts": [],
        "procedure_id": None,
        "issue": "unpaid wages",
        "next_question": None,
        "needs_precedents": False,
    }
    base.update(overrides)
    return base


def test_grounded_fact_is_kept_and_ungrounded_fact_is_dropped():
    text = "Mera maalik teen mahine se tankha nahi de raha, main Karnataka mein hoon"
    clean, rejected = validate(
        reading(facts=[
            {"slot": "jurisdiction", "value": "Karnataka", "evidence": "Karnataka mein hoon"},
            {"slot": "amount", "value": "50000", "evidence": "paanch hazaar rupaye"},
        ]),
        text,
        JURISDICTIONS,
    )
    assert [fact.value for fact in clean.facts] == ["IN-KA"]
    assert "fact_ungrounded:amount" in rejected


def test_invented_jurisdiction_is_dropped_even_when_quoted():
    text = "I live in Atlantis and my landlord kept my deposit"
    clean, rejected = validate(
        reading(facts=[{"slot": "jurisdiction", "value": "Atlantis", "evidence": "I live in Atlantis"}]),
        text,
        JURISDICTIONS,
    )
    assert clean.facts == []
    assert "jurisdiction_not_known" in rejected


def test_search_query_is_a_vocabulary_hint_and_is_bounded():
    """The rewrite is the one field that is deliberately not grounded: its whole
    job is to say the same thing in different words.

    It may contain a section number the caller never said. A guard against
    "invented digits" used to delete exactly the rewrite that fixes the hard
    cases — a caller reports a refused FIR, the reader answers "registration of
    FIR for a cognizable offence under section 154", and that number is the word
    the authorities use. A wrong number in a search query cannot become a false
    statement: it only selects passages that already exist, and every quoted
    claim is verified separately.
    """
    text = "he changed the lock and threw my things out"
    clean, _ = validate(reading(search_query="dispossession of a tenant without due process of law"), text)
    assert clean.search_query == "dispossession of a tenant without due process of law"

    cited, rejected = validate(reading(search_query="eviction under section 138 of the rent act"), text)
    assert cited.search_query == "eviction under section 138 of the rent act"
    assert "search_query_invented_numbers" not in rejected

    short, rejected_short = validate(reading(search_query="ok"), text)
    assert short.search_query == ""
    assert "search_query_unusable" in rejected_short

    # The bounds that still matter: one line, no markup, no URLs.
    assert validate(reading(search_query="see https://example.com/case law"), text)[0].search_query == ""

    assert validate(reading(search_query=""), text)[0].search_query == ""


def test_the_consultation_payload_carries_the_retrieval_rewrite():
    """The answer and the reading are one object, so the rewrite rides with it.

    There is no separate reading schema to drift out of sync: the same JSON
    that holds `reply` holds `search_query`, and `from_consultation` keeps it.
    """
    text = "he changed the lock and threw my things out"
    result = understanding.from_consultation(
        {
            **reading(search_query="dispossession of a tenant without due process of law"),
            "reply": "A landlord cannot remove you without due process.",
            "legal_claims": [],
        },
        text,
        jurisdictions=JURISDICTIONS,
    )
    assert result.search_query == "dispossession of a tenant without due process of law"
    assert result.intent == "statute"


def test_reported_facts_are_read_from_the_same_payload_as_the_reply():
    text = "my employer owes me around ten thousand for two months"
    result = understanding.from_consultation(
        {
            "intent": "procedural",
            "confidence": 0.8,
            "language": "en-IN",
            "reply": "You can claim the arrears.",
            "reported_facts": [{
                "slot": "amount", "value": "around ten thousand",
                "normalized": "INR 10000", "evidence": "around ten thousand",
                "status": "estimated", "confidence": "medium",
            }],
            "legal_claims": [],
        },
        text,
        jurisdictions=JURISDICTIONS,
    )
    assert result.intent == "procedural"
    assert result.facts[0]["slot"] == "amount"
    assert result.facts[0]["status"] == "estimated"
    assert result.facts[0]["normalized"] == "INR 10000"


def test_the_turn_searches_the_call_not_only_the_sentence():
    """With one model call, last turn's legal phrasing seeds this turn's search.

    The consultation writes `search_query` in the corpus's vocabulary and the
    session keeps it. When the caller only points at the matter, that phrasing
    is part of what gets searched -- otherwise the one sentence in the call
    written in the language of the judgments is thrown away exactly when it is
    needed.
    """
    ledger = facts.Ledger()
    search, carried, _focus = legal_engine._turn_search(
        {"search_query": "dispossession of a tenant otherwise than in due course of law"},
        ledger,
        "what about the money?",
        [],
        "",
    )
    assert carried == ""
    assert "dispossession of a tenant" in search
    assert "money" in search

    # A turn with its own subject leads alone: an old query cannot pull a new
    # matter back to itself.
    own = legal_engine._turn_search(
        {"search_query": "dispossession of a tenant"},
        ledger,
        "my employer has not paid salary for three months and I have the slips",
        [],
        "",
    )[0]
    assert "dispossession" not in own


def test_a_number_the_caller_did_say_survives_the_rewrite():
    text = "what does section 138 say about a bounced cheque"
    clean, rejected = validate(reading(search_query="dishonour of cheque under section 138"), text)
    assert clean.search_query == "dishonour of cheque under section 138"
    assert "search_query_invented_numbers" not in rejected


def test_ungrounded_critical_is_capped_but_grounded_critical_stands():
    text = "My husband is threatening me again, I am scared"
    clean, rejected = validate(
        reading(risk=[{"category": "immediate_danger", "severity": "critical", "evidence": "he will stab me"}]),
        text,
        JURISDICTIONS,
    )
    assert clean.risk[0].severity == "high"
    assert "critical_risk_ungrounded:immediate_danger" in rejected

    # Devanagari span, quoted exactly, is a real critical finding.
    text_hi = "मेरा पति मुझे मार रहा है, मैं घर पर अकेली हूँ"
    clean_hi, _ = validate(
        reading(risk=[{"category": "immediate_danger", "severity": "critical", "evidence": "मुझे मार रहा है"}]),
        text_hi,
        JURISDICTIONS,
    )
    assert clean_hi.risk[0].severity == "critical"


def test_unknown_vocabulary_cannot_enter_the_pipeline():
    text = "What is the punishment under section 138?"
    clean, rejected = validate(
        reading(
            intent="vibes",
            risk=[{"category": "cosmic_dread", "severity": "apocalyptic", "evidence": "punishment"}],
            facts=[{"slot": "mood", "value": "bad", "evidence": "punishment"}],
            procedure_id="invent-a-remedy",
        ),
        text,
        JURISDICTIONS,
    )
    assert clean.intent == "unclear"
    assert clean.procedure_id is None
    assert clean.facts == []
    assert clean.risk[0].category == "other" and clean.risk[0].severity == "normal"
    assert {"intent_not_allowed:vibes", "procedure_not_in_catalog:invent-a-remedy"} <= set(rejected)


def test_repeated_question_is_not_asked_again():
    text = "My landlord changed the locks last week"
    clean, _ = validate(reading(next_question="Which state are you in?"), text, JURISDICTIONS, asked=["Which state are you in"])
    assert clean.next_question is None

    kept, _ = validate(reading(next_question="did the landlord give you any written notice"), text, JURISDICTIONS)
    assert kept.next_question.endswith("?")


def test_combine_never_lowers_the_lexical_net():
    text = "He is outside my door and says he will kill me"
    # A model that reads this as an ordinary civil dispute cannot downgrade it.
    result = understanding.combine(validate(reading(intent="statute", risk=[]), text, JURISDICTIONS)[0], text)
    assert result.intent == "emergency"
    assert result.risk["level"] == "critical"


def test_combine_raises_severity_from_a_grounded_model_finding():
    text = "The agent took my OTP while I was on the call"
    result = understanding.combine(
        validate(
            reading(intent="procedural", risk=[
                {"category": "fraud_in_progress", "severity": "high", "evidence": "took my OTP"},
            ]),
            text,
            JURISDICTIONS,
        )[0],
        text,
    )
    assert result.risk["level"] == "high"
    assert "report_now" in result.risk["policies"]
    assert result.emergency is False


def test_no_consultation_degrades_to_the_net_and_says_so():
    result = understanding.from_consultation(None, "My landlord changed the locks", jurisdictions=JURISDICTIONS)
    assert result.source == "lexical"
    assert result.intent == "unclear"
    assert "consultation reply unavailable" in result.reasons


def test_navigation_uses_the_reading_but_procedures_stay_curated():
    text = "Police refused to write my FIR after the theft"
    result = understanding.combine(
        validate(reading(intent="procedural", procedure_id="police-fir", confidence=0.9), text, JURISDICTIONS)[0],
        text,
    )
    route = routing.route_turn(text, "IN", reading=result)
    assert route.name == "procedural"
    assert route.procedure is not None
    assert route.procedure["id"] == "police-fir"


def test_reading_cannot_downgrade_an_emergency_route():
    text = "My husband is beating me right now"
    result = understanding.combine(validate(reading(intent="statute"), text, JURISDICTIONS)[0], text)
    route = routing.route_turn(text, "IN", reading=result)
    assert route.name == "emergency"


def test_low_confidence_reading_asks_instead_of_acting():
    text = "Something about my case I am not sure"
    result = understanding.combine(
        validate(reading(intent="draft", confidence=0.2), text, JURISDICTIONS)[0], text
    )
    route = routing.route_turn(text, "IN", reading=result)
    assert route.name == "unclear"


def test_reader_cannot_overwrite_what_the_caller_stated():
    ledger = facts.Ledger()
    ledger.record("amount", "₹40,000", "caller", 1, "amount")
    written = facts.apply_reading(
        ledger,
        [{"slot": "amount", "value": "₹4,000", "evidence": "char hazaar"}],
        turn=2,
    )
    assert ledger.value("amount") == "₹40,000"
    assert written == []


def test_overlay_keeps_the_net_reasons_for_review():
    text = "There is a court hearing tomorrow and I have no notice"
    result = understanding.combine(validate(reading(intent="procedural", confidence=0.7), text, JURISDICTIONS)[0], text)
    route = routing.route_turn(text, "IN", reading=result)
    assert route.name == "procedural"
    assert any(reason.startswith("reader:") for reason in route.reasons)


def test_a_process_question_the_catalogue_does_not_cover_still_retrieves():
    """No curated card is not the same as no evidence.

    The lexical pass finds no procedure for a stuck passport application, and
    that must not stop the turn retrieving primary law for it. The old reader
    could label this `procedural` before retrieval; without one, the property
    is simply that the turn is searched.
    """
    context = asyncio.run(legal_engine.analyze("my passport application has been stuck for months"))
    assert context.procedure is None
    assert any(
        stage["stage"] == "evidence" and stage.get("mode") != "curated_procedure"
        for stage in context.trace
    )


def test_lexical_emergency_never_calls_the_model(monkeypatch):
    async def boom(*args, **kwargs):
        raise AssertionError("no model may run on the emergency fast path")

    monkeypatch.setattr(legal_engine, "generate_turn", boom)
    monkeypatch.setattr(legal_engine, "model_configured", lambda: True)
    context = asyncio.run(legal_engine.analyze("He is beating me right now"))
    assert context.route["name"] == "emergency"
    assert any(stage["stage"] == "understand" and stage.get("skipped") for stage in context.trace)


def test_interpretation_serialises_without_model_internals():
    empty = Interpretation()
    payload = empty.as_dict()
    assert payload["source"] == "lexical"
    assert payload["risk_level"] == "normal"
    assert "latency_ms" in payload
