"""Latency-path behaviour: speak first, stream safely, report the budget."""
from __future__ import annotations

import asyncio

from app import legal_engine, model_gateway, understanding
from app.legal_engine import INDIAN_JURISDICTIONS


def interpretation(text: str, **overrides):
    raw = {
        "intent": "procedural", "confidence": 0.9, "language": "en-IN", "risk": [],
        "facts": [], "procedure_id": None, "issue": "", "next_question": None,
        "needs_precedents": False,
    }
    raw.update(overrides)
    validated, rejected = understanding.validate(raw, text, INDIAN_JURISDICTIONS)
    return understanding.combine(validated, text, rejected)


def collect(text: str) -> list[tuple[str, dict]]:
    async def run():
        return [event async for event in legal_engine.answer_stream(text, "en-IN")]

    return asyncio.run(run())


def test_partial_reply_reads_a_string_that_is_still_arriving():
    assert model_gateway.partial_reply('{"reply": "Understood. Which') == "Understood. Which"
    assert model_gateway.partial_reply('{"reply": "a\\"b') == 'a"b'
    assert model_gateway.partial_reply('{"reply": "line\\nnext') == "line\nnext"
    assert model_gateway.partial_reply('{"issue": "x"}') is None
    assert model_gateway.partial_reply('{"reply": "d' ) == "d"


def test_early_speech_allows_conversation_and_refuses_propositions():
    assert legal_engine._early_speak_ok("Understood. Which months are those for?")
    assert legal_engine._early_speak_ok("Go on, I am listening.")
    assert not legal_engine._early_speak_ok("Under section 138 of the Act you can file a complaint.")
    assert not legal_engine._early_speak_ok("They owe you 15,000 rupees.")
    assert not legal_engine._early_speak_ok("See https://indiacode.nic.in for the text.")


def test_acknowledgement_is_yielded_before_the_model_reads_the_turn(monkeypatch):
    called = {"analyze": False}

    async def fake_analyze(*args, **kwargs):
        called["analyze"] = True
        raise AssertionError("the reader must not run before the first event")

    monkeypatch.setattr(legal_engine, "analyze", fake_analyze)

    async def first_event():
        return await legal_engine.answer_stream("hello, can you help me").__anext__()

    kind, data = asyncio.run(first_event())
    assert kind == "ack"
    assert data["text"]
    assert called["analyze"] is False


def test_an_emergency_is_not_delayed_by_an_acknowledgement():
    events = collect("my husband is beating me and he is coming now")
    assert events
    assert events[0][0] != "ack"
    final = events[-1][1]
    assert final["route"]["name"] == "emergency"


def test_an_intake_reply_starts_speaking_before_the_turn_finishes(monkeypatch):
    text = "how do I file an FIR at the police station"

    async def fake_stream(context, tools=None):
        yield "sentence", "Understood."
        await asyncio.sleep(0.12)
        yield "final", {
            "intent": "procedural", "reply": "Understood. Which months are those?",
            "language": "en-IN", "legal_claims": [],
        }

    monkeypatch.setattr(legal_engine, "model_configured", lambda: True)
    monkeypatch.setattr(legal_engine, "generate_turn_stream", fake_stream)

    events = collect(text)
    kinds = [kind for kind, _ in events]
    assert "speak" in kinds
    assert kinds.index("speak") < kinds.index("final")
    assert [data["text"] for kind, data in events if kind == "speak"] == ["Understood."]
    final = events[-1][1]
    assert final["timing"]["first_speech_ms"] is not None
    assert final["timing"]["total_ms"] >= final["timing"]["route_ms"] >= 0


def test_a_legal_answer_does_not_stream_before_validation(monkeypatch):
    text = "what does the law say about unpaid wages in Karnataka"

    async def fake_stream(context, tools=None):
        raise AssertionError("a legal answer must not stream before claim validation")
        yield  # pragma: no cover

    monkeypatch.setattr(legal_engine, "model_configured", lambda: True)
    monkeypatch.setattr(legal_engine, "generate_turn_stream", fake_stream)
    monkeypatch.setattr(legal_engine, "generate_turn", lambda *a, **k: asyncio.sleep(0, result=None))

    events = collect(text)
    assert [kind for kind, _ in events if kind == "speak"] == []
    assert events[-1][0] == "final"


def test_tool_support_is_remembered_instead_of_reprobed(monkeypatch):
    calls: list[object] = []

    async def fake_json(*args, **kwargs):
        calls.append(kwargs.get("tools"))
        return {}

    monkeypatch.setattr(model_gateway, "configured", lambda: True)
    monkeypatch.setattr(model_gateway, "_json_completion", fake_json)
    monkeypatch.setattr(model_gateway, "_TOOLS_UNSUPPORTED", True)
    asyncio.run(model_gateway.generate_turn({}, tools=[{"type": "function"}]))
    assert calls == [None]
    assert model_gateway.tool_calling_available() is False
