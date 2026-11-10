"""Conversation-level intake behaviour, with the consultation stubbed.

These pin the difference between a form and a conversation: an intake keeps
running across a terse answer, a state named once is remembered, a second
matter is held rather than dropped, and a finished intake does not block the
next one. The single model call is faked, so the tests describe intent rather
than a provider -- and they exercise the same payload shape the real turn uses.
"""
from __future__ import annotations

import asyncio

from app import legal_engine


def consultation(text, *, intent="procedural", facts=None, procedure_id=None,
                 confidence=0.9, issue="", next_question=None, reply=None):
    """One model reply: the answer and its reading, in the same object."""
    return {
        "intent": intent,
        "confidence": confidence,
        "language": "en-IN",
        "risk": [],
        "reported_facts": facts or [],
        "procedure_id": procedure_id,
        "issue": issue,
        "next_question": next_question,
        "needs_precedents": False,
        "reply": reply or "Understood. What happened next?",
        "legal_claims": [],
    }


def fact(slot, value, evidence):
    return {"slot": slot, "value": value, "evidence": evidence}


def scripted(monkeypatch, by_text):
    async def fake_generate(context, **kwargs):
        return by_text[context["caller_message"]]

    monkeypatch.setattr(legal_engine, "generate_turn", fake_generate)
    monkeypatch.setattr(legal_engine, "model_configured", lambda: True)


def run(text, session):
    return asyncio.run(legal_engine.answer_turn(text, "en-IN", "auto", None, [], session))


def carry(session, result):
    """Persist one turn the way the HTTP layer does."""
    state = result["conversation_state"]
    return {
        "turns": session["turns"] + 1,
        "facts": state["ledger"],
        "last_state": {"workflow": state.get("workflow"), "route": result["route"]["name"]},
    }


def empty_session():
    return {"turns": 0, "facts": {}, "last_state": {}}


def test_a_terse_answer_does_not_drop_the_intake(monkeypatch):
    first = "how do I file an FIR at the police station"
    second = "he said it verbally"
    scripted(monkeypatch, {
        first: consultation(first, procedure_id="police-fir"),
        second: consultation(second, intent="unclear", confidence=0.3),
    })

    session = empty_session()
    opened = run(first, session)
    assert opened["workflow"]["id"] == "police-fir"
    assert opened["procedure"]["id"] == "police-fir"
    session = carry(session, opened)

    # The reader cannot classify a three-word answer. The conversation must not
    # fall out of the intake because of that.
    follow = run(second, session)
    assert follow["route"]["name"] == "procedural"
    assert follow["workflow"]["procedure_id"] == "police-fir"
    assert follow["workflow"]["id"] == "police-fir"
    assert follow["workflow"]["status"] == "active"
    assert "refuse in writing" in (follow["workflow"]["ask"] or "").lower()
    # The steps are already on screen from the opening turn; reprinting the
    # card under every answer is what made the intake read as a form.
    assert follow["procedure"] is None


def test_a_state_stated_once_is_remembered_for_later_turns(monkeypatch):
    first = "I was arrested in Karnataka yesterday, what are my rights under Article 22"
    second = "can they hold me for two days"
    scripted(monkeypatch, {
        first: consultation(
            first, intent="statute",
            facts=[fact("jurisdiction", "Karnataka", "Karnataka"), fact("when", "yesterday", "yesterday")],
        ),
        second: consultation(second, intent="statute"),
    })

    session = empty_session()
    opened = run(first, session)
    assert opened["jurisdiction"] == "IN-KA"
    assert opened["jurisdiction_basis"] == "caller_location_in_message"
    session = carry(session, opened)

    follow = run(second, session)
    assert follow["jurisdiction"] == "IN-KA"
    assert follow["jurisdiction_basis"] == "stated_earlier_in_this_call"
    assert "which state" not in (follow["next_question"] or "").lower()


def test_a_finished_intake_does_not_block_the_next_matter(monkeypatch):
    first = "The police refused to register my FIR in Karnataka yesterday and gave me a written refusal"
    second = "Now my company has not paid my salary of 40000 rupees for March"
    scripted(monkeypatch, {
        first: consultation(
            first, procedure_id="police-fir",
            facts=[
                fact("stage", "police_refused", "police refused"),
                fact("notice_or_order", "written refusal", "written refusal"),
                fact("jurisdiction", "Karnataka", "Karnataka"),
                fact("when", "yesterday", "yesterday"),
            ],
            issue="police refused to register the FIR",
        ),
        second: consultation(
            second, procedure_id="unpaid-wages",
            facts=[
                fact("amount", "40000 rupees", "40000 rupees"),
                fact("when", "March", "March"),
                fact("recipient", "company", "company"),
            ],
            issue="unpaid wages",
        ),
    })

    session = empty_session()
    finished = run(first, session)
    assert finished["workflow"]["id"] == "police-fir"
    assert finished["workflow"]["status"] == "completed"
    session = carry(session, finished)

    next_matter = run(second, session)
    assert next_matter["workflow"]["id"] == "unpaid-wages"
    assert next_matter["workflow"]["status"] == "active"
    assert next_matter["procedure"]["id"] == "unpaid-wages"


def test_a_second_matter_raised_mid_intake_is_held_not_dropped(monkeypatch):
    first = "how do I file an FIR at the police station"
    second = "also my employer has not paid my wages"
    scripted(monkeypatch, {
        first: consultation(first, procedure_id="police-fir"),
        second: consultation(second, procedure_id="unpaid-wages", issue="unpaid wages"),
    })

    session = empty_session()
    opened = run(first, session)
    session = carry(session, opened)

    follow = run(second, session)
    # The intake in progress keeps the floor ...
    assert follow["workflow"]["id"] == "police-fir"
    assert follow["workflow"]["status"] == "active"
    # ... and the other problem is on the record for when it ends.
    matters = follow["conversation_state"]["ledger"]["matters"]
    assert [item["procedure_id"] for item in matters] == ["unpaid-wages"]
    assert matters[0]["status"] == "noted"


def test_an_intake_turn_is_one_question_in_the_callers_language(monkeypatch):
    text = "how do I file an FIR at the police station"
    scripted(monkeypatch, {text: consultation(text, procedure_id="police-fir")})
    seen = {}

    async def fake_generate(context, **kwargs):
        seen["intake"] = context["intake"]
        return {
            "intent": "procedural",
            "confidence": 0.9,
            "reply": "समझ गया. पुलिस ने मना किया — लिखित में दिया या मौखिक रूप से?",
            "language": "hi-IN",
            "legal_claims": [],
        }

    monkeypatch.setattr(legal_engine, "generate_turn", fake_generate)
    monkeypatch.setattr(legal_engine, "model_configured", lambda: True)
    result = run(text, empty_session())

    assert seen["intake"]["needed_slot"] == "stage"
    assert seen["intake"]["procedure_label"]
    assert result["mode"] == "intake"
    assert result["answer_status"] == "intake_turn"
    assert result["language"] == "hi-IN"
    assert "मौखिक" in result["response"]
    # The reply already asks the question; a second follow-up chip would talk
    # over it.
    assert result["next_question"] is None
    assert result["procedure"]["id"] == "police-fir"


def test_an_empty_health_check_still_reports_no_workflow(monkeypatch):
    text = "hello"
    scripted(monkeypatch, {text: consultation(text, intent="unclear", confidence=0.2)})
    result = run(text, empty_session())
    assert result["workflow"] is None
