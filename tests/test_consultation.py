"""The consultation is one model call, and its reading is still checked.

These are the acceptance tests for replacing the two-call turn (standalone
reader, then generation) with a single call whose JSON carries the answer and
the reading together. What must not change: a fact needs the caller's own
words, a correction keeps its history, the FSM cannot block a supported
answer, and bad output degrades visibly rather than being trusted.

No provider is called. `generate_turn` is scripted, so the tests measure the
engine's contract rather than a model's behaviour.
"""
from __future__ import annotations

import asyncio
import json
import time

import httpx

from app import facts, legal_engine, model_gateway


def empty_session():
    return {"turns": 0, "facts": {}, "last_state": {}}


def carry(session, result):
    state = result["conversation_state"]
    return {
        "turns": session["turns"] + 1,
        "facts": state["ledger"],
        "last_state": {
            "workflow": state.get("workflow"),
            "route": result["route"]["name"],
            "search_query": result.get("search_query"),
        },
    }


def scripted(monkeypatch, replies):
    """One reply per caller message, and a count of how many calls were made."""
    calls: list[dict] = []

    async def fake_generate(context, **kwargs):
        calls.append(context)
        reply = replies.get(context["caller_message"])
        return reply() if callable(reply) else reply

    monkeypatch.setattr(legal_engine, "generate_turn", fake_generate)
    monkeypatch.setattr(legal_engine, "model_configured", lambda: True)
    return calls


def run(text, session=None):
    return asyncio.run(
        legal_engine.answer_turn(text, "en-IN", "auto", None, [], session or empty_session())
    )


def answer(reply, **overrides):
    base = {
        "intent": "statute",
        "confidence": 0.9,
        "language": "en-IN",
        "reply": reply,
        "reported_facts": [],
        "legal_claims": [],
        "risk": [],
    }
    base.update(overrides)
    return base


def fact(slot, value, evidence, **overrides):
    return {"slot": slot, "value": value, "evidence": evidence, **overrides}


def test_a_legal_turn_makes_exactly_one_model_call(monkeypatch) -> None:
    text = "am I allowed to record a phone call without telling the other person"
    calls = scripted(monkeypatch, {text: answer("A notice is the first step. Do you still have the return memo?")})

    result = run(text)

    assert len(calls) == 1
    assert result["mode"] == "model"
    assert result["response"].startswith("A notice is the first step")
    assert result["verification"]["status"] == "no_legal_claims_emitted"


def test_a_failed_call_is_not_retried_with_a_second_full_request(monkeypatch) -> None:
    text = "am I allowed to record a phone call without telling the other person"
    calls = scripted(monkeypatch, {text: None})

    result = run(text)

    assert len(calls) == 1
    assert result["mode"] == "grounded_only"


def test_a_fact_without_a_verbatim_span_never_enters_the_ledger(monkeypatch) -> None:
    text = "the company owes me ten thousand rupees for March"
    calls = scripted(monkeypatch, {
        text: answer(
            "You can claim the arrears. Which months are unpaid?",
            reported_facts=[
                fact("amount", "ten thousand rupees", "ten thousand rupees",
                     normalized="INR 10000", status="estimated", confidence="medium"),
                fact("recipient", "a company", "my employer Limited"),
            ],
        ),
    })

    result = run(text)
    known = {item["slot"]: item for item in result["conversation_state"]["known_facts"]}

    assert known["amount"]["value"] == "ten thousand rupees"
    assert known["amount"]["normalized"] == "INR 10000"
    assert known["amount"]["status"] == "estimated"
    assert known["amount"]["evidence"] == "ten thousand rupees"
    # The second fact cited words the caller never said, so it was dropped.
    assert "recipient" not in known
    assert len(calls) == 1


def test_a_correction_replaces_the_value_and_keeps_the_earlier_one(monkeypatch) -> None:
    first = "the company owes me ten thousand rupees"
    second = "actually, it is twelve thousand"
    scripted(monkeypatch, {
        first: answer(
            "Noted. Which months does that cover?",
            reported_facts=[fact("amount", "ten thousand rupees", "ten thousand rupees")],
        ),
        second: answer(
            "I have corrected the figure. Do you have salary slips?",
            reported_facts=[fact(
                "amount", "twelve thousand", "twelve thousand",
                normalized="INR 12000", status="corrected",
            )],
        ),
    })

    session = empty_session()
    opened = run(first, session)
    session = carry(session, opened)
    corrected = run(second, session)

    known = {item["slot"]: item for item in corrected["conversation_state"]["known_facts"]}
    assert known["amount"]["value"] == "twelve thousand"
    assert known["amount"]["status"] == "corrected"
    assert known["amount"]["history"][-1]["value"] == "ten thousand rupees"
    conflicts = corrected["conversation_state"]["conflicts"]
    assert conflicts[-1]["slot"] == "amount"
    assert conflicts[-1]["earlier"] == "ten thousand rupees"


def test_an_amount_the_pattern_net_cannot_read_still_reaches_the_ledger(monkeypatch) -> None:
    """The reading is the extractor; the pattern net is a fallback, not a gate.

    "One and a half lakhs" and "three months of salary" defeat every amount
    regular expression in `facts.py` on purpose -- a bare number is not money.
    If the model is the intelligence layer, its grounded reading has to land.
    """
    text = "they owe me one and a half lakhs, it has been three months of salary"
    # Nothing in the deterministic pass reads this sentence. That is the point.
    assert facts.extract(text, 1)[1] == []

    scripted(monkeypatch, {
        text: answer(
            "A wage claim for that amount is possible. Which months are unpaid?",
            reported_facts=[
                fact("amount", "one and a half lakhs", "one and a half lakhs",
                     normalized="INR 150000", precision="approximate",
                     original_expression="one and a half lakhs", status="estimated"),
                fact("when", "three months of salary", "three months of salary",
                     normalized="relative: three months", precision="relative",
                     status="caller_confirmed"),
            ],
        ),
    })

    result = run(text)
    known = {item["slot"]: item for item in result["conversation_state"]["known_facts"]}

    assert known["amount"]["source"] == "reader"
    assert known["amount"]["normalized"] == "INR 150000"
    assert known["amount"]["precision"] == "approximate"
    assert known["amount"]["original_expression"] == "one and a half lakhs"
    assert known["when"]["normalized"] == "relative: three months"


def test_the_reading_replaces_the_pattern_net_for_the_same_sentence(monkeypatch) -> None:
    """A pattern match on the caller's words never outranks a reading of them."""
    text = "the company owes me 10000 rupees"
    net_fragment, net_slots = facts.extract(text, 1)
    assert "amount" in net_slots, "the net has to have written something to be replaced"

    scripted(monkeypatch, {
        text: answer(
            "That is arrears. Do you have a salary slip?",
            reported_facts=[fact(
                "amount", "10000 rupees", "10000 rupees",
                normalized="INR 10000", original_expression="10000 rupees",
            )],
        ),
    })

    result = run(text)
    known = {item["slot"]: item for item in result["conversation_state"]["known_facts"]}

    assert known["amount"]["source"] == "reader"
    assert known["amount"]["normalized"] == "INR 10000"
    # Re-reading one message more precisely is not the caller changing their
    # story, so it must not be reported to them as a conflict.
    assert result["conversation_state"]["conflicts"] == []


def test_the_reading_settles_the_route_after_the_call(monkeypatch) -> None:
    """The phrase net decides the route early; the model's reading decides it."""
    text = "my neighbour keeps parking in front of my gate, what can I do"
    scripted(monkeypatch, {
        text: answer(
            "A civil dispute over access can be raised before the local authority. "
            "Has the society been informed in writing?",
            intent="statute",
        ),
    })

    result = run(text)

    assert result["route"]["name"] == "statute"
    assert any("model: read this as statute" in reason for reason in result["route"]["reasons"])
    assert result["mode"] == "model"


def test_an_ungrounded_emergency_intent_does_not_move_the_route(monkeypatch) -> None:
    text = "my landlord will not return my deposit"
    scripted(monkeypatch, {
        text: answer(
            "A deposit is recoverable. Do you have a receipt?",
            intent="emergency",
        ),
    })

    result = run(text)

    assert result["route"]["name"] != "emergency"
    assert result["safety"]["level"] == "high"


def test_the_intake_is_advisory_and_cannot_replace_a_supported_answer(monkeypatch) -> None:
    first = "how do I file an FIR at the police station"
    second = "how do I recover the money he took"
    reply = (
        "You can pursue recovery through the route the supplied provision allows, and you "
        "should keep the receipt. Which month did you pay?"
    )
    seen: dict = {}

    def second_reply():
        return answer(reply, intent="procedural", confidence=0.8)

    async def fake_generate(context, **kwargs):
        seen["intake"] = context.get("intake")
        return second_reply()

    monkeypatch.setattr(legal_engine, "generate_turn", fake_generate)
    monkeypatch.setattr(legal_engine, "model_configured", lambda: True)

    session = empty_session()
    opened = run(first, session)
    session = carry(session, opened)
    result = run(second, session)

    assert seen["intake"]["advisory"] is True
    assert "Answer the caller's legal question first" in seen["intake"]["guidance"]
    # The model's answer is the turn; the deterministic template is only the
    # floor when generation failed.
    assert result["response"] == reply
    assert result["mode"] == "intake"


def test_a_procedure_named_only_by_the_model_opens_the_intake(monkeypatch) -> None:
    text = "they are not writing down my complaint against the shopkeeper"
    scripted(monkeypatch, {
        text: answer(
            "A written complaint is the first step. Was the refusal verbal?",
            intent="procedural",
            procedure_id="police-fir",
        ),
    })

    result = run(text)

    assert result["workflow"]["id"] == "police-fir"
    assert result["procedure"]["id"] == "police-fir"
    assert result["route"]["name"] == "procedural"


def test_a_grounded_emergency_from_the_model_overrides_the_generated_answer(monkeypatch) -> None:
    text = "my brother keeps saying he will hurt me tonight"
    scripted(monkeypatch, {
        text: answer(
            "You may file a complaint at the police station.",
            intent="emergency",
            risk=[{
                "category": "immediate_danger", "severity": "critical",
                "evidence": "he will hurt me tonight",
            }],
        ),
    })

    result = run(text)

    assert result["route"]["name"] == "emergency"
    assert result["mode"] == "emergency"
    assert "police station" not in result["response"]
    assert result["safety"]["level"] == "critical"


def test_an_ungrounded_emergency_cannot_take_the_safety_path(monkeypatch) -> None:
    text = "my landlord will not return my deposit"
    scripted(monkeypatch, {
        text: answer(
            "A deposit is recoverable. Do you have a receipt?",
            intent="emergency",
            risk=[{
                "category": "immediate_danger", "severity": "critical",
                "evidence": "he threatened me with a knife",
            }],
        ),
    })

    result = run(text)

    assert result["route"]["name"] != "emergency"
    assert result["response"].startswith("A deposit is recoverable")


def test_a_tool_request_costs_one_round_and_never_three(monkeypatch) -> None:
    text = "what does the law say about a refused FIR"
    calls: list[int] = []

    async def fake_generate(context, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return {
                "intent": "procedural",
                "reply": "",
                "tool_requests": [{"name": "search_law", "arguments": {"query": "FIR registration"}}],
            }
        return answer("A refusal can be escalated. Do you have it in writing?")

    monkeypatch.setattr(legal_engine, "generate_turn", fake_generate)
    monkeypatch.setattr(legal_engine, "model_configured", lambda: True)

    result = run(text)

    # One round trip to ask for the tool, one to answer. Not four.
    assert len(calls) == 2
    assert result["response"].startswith("A refusal can be escalated")


def test_the_model_search_phrasing_is_kept_for_the_next_turn(monkeypatch) -> None:
    text = "he changed the lock and threw my things out"
    scripted(monkeypatch, {
        text: answer(
            "A landlord cannot remove you without due process. Which state are you in?",
            search_query="dispossession of a tenant without due process of law",
        ),
    })

    result = run(text)

    assert result["search_query"] == "dispossession of a tenant without due process of law"


def test_the_ledger_serialises_its_provenance() -> None:
    ledger = facts.Ledger()
    ledger.record(
        "amount", "around ten thousand", "reader", 2, "amount",
        normalized="INR 10000", evidence="around ten thousand",
        status="estimated", confidence="medium",
    )
    restored = facts.Ledger.from_dict(ledger.as_dict())

    assert restored.value("amount") == "around ten thousand"
    assert restored.facts["amount"].normalized == "INR 10000"
    assert restored.facts["amount"].status == "estimated"
    assert restored.facts["amount"].confidence == "medium"


class _Response:
    def __init__(self, status: int, body: dict | None = None, headers: dict | None = None) -> None:
        self.status_code = status
        self._body = body or {}
        self.headers = headers or {}
        self.text = json.dumps(self._body)

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)  # type: ignore[arg-type]


class _Client:
    def __init__(self, responses: list[_Response]) -> None:
        self._responses = responses
        self.calls = 0

    async def post(self, *args, **kwargs) -> _Response:
        self.calls += 1
        return self._responses.pop(0)


def _gateway(monkeypatch, responses: list[_Response]) -> _Client:
    client = _Client(responses)
    monkeypatch.setattr(model_gateway, "configured", lambda: True)
    monkeypatch.setattr(model_gateway, "_client", lambda: client)
    monkeypatch.setattr(model_gateway, "_TOOLS_UNSUPPORTED", False)
    monkeypatch.setattr(model_gateway, "TPM_BUDGET", 0)
    return client


def test_a_429_gets_one_short_retry_and_then_degrades(monkeypatch) -> None:
    client = _gateway(monkeypatch, [
        _Response(429, headers={"retry-after": "0"}),
        _Response(429, headers={"retry-after": "0"}),
        _Response(200, {"choices": [{"message": {"content": "{}"}}]}),
    ])

    started = time.perf_counter()
    result = asyncio.run(model_gateway._json_completion("system", [], {"x": 1}, max_tokens=50))
    elapsed = time.perf_counter() - started

    assert result is None
    assert client.calls == 2, "a quota wall does not clear inside a voice turn"
    assert elapsed < 5.0


def test_a_429_followed_by_success_keeps_the_turn(monkeypatch) -> None:
    _gateway(monkeypatch, [
        _Response(429, headers={"retry-after": "0"}),
        _Response(200, {
            "choices": [{"message": {"content": '{"reply": "ok"}'}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
        }),
    ])

    result = asyncio.run(model_gateway._json_completion("system", [], {"x": 1}, max_tokens=50))

    assert result == {"reply": "ok"}
    usage = model_gateway.usage_snapshot()
    assert usage["last"]["total_tokens"] == 42
    assert usage["minute_tokens"] >= 42


def test_usage_is_counted_from_the_provider_not_guessed(monkeypatch) -> None:
    _gateway(monkeypatch, [
        _Response(200, {
            "choices": [{"message": {"content": '{"reply": "ok"}'}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
        }),
    ])

    asyncio.run(model_gateway._json_completion("system", [], {"x": 1}, max_tokens=50))

    assert model_gateway.usage_snapshot()["last"] == {
        "prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
        "at": model_gateway.usage_snapshot()["last"]["at"],
    }
    assert model_gateway.usage_snapshot()["last"]["at"] > 0


def test_gemini_live_configuration_and_headers(monkeypatch) -> None:
    monkeypatch.setattr(model_gateway, "MODEL", "gemini-3.8-live")
    monkeypatch.setattr(model_gateway, "API_KEY", "test-gemini-key")
    monkeypatch.setattr(model_gateway, "BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")

    assert model_gateway.is_gemini_live() is True
    assert model_gateway.configured() is True
    headers = model_gateway._auth_headers()
    assert headers["Authorization"] == "Bearer test-gemini-key"
    assert headers.get("x-goog-api-key") == "test-gemini-key"
    assert "chat/completions" in model_gateway.endpoint()


def test_gemini_live_generate_turn_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(model_gateway, "MODEL", "gemini-3.8-live")
    monkeypatch.setattr(model_gateway, "API_KEY", "test-gemini-key")

    dispatched = []

    async def fake_live_completion(context, tools=None, extra_messages=None, **kwargs):
        dispatched.append(context)
        return {
            "reply": "Live legal response.",
            "intent": "procedural",
            "confidence": 0.95,
            "issue": "FIR refusal",
            "legal_claims": [],
            "reported_facts": [],
        }

    monkeypatch.setattr(model_gateway, "_gemini_live_completion", fake_live_completion)

    result = asyncio.run(model_gateway.generate_turn({"caller_message": "police refused FIR"}))
    assert len(dispatched) == 1
    assert result is not None
    assert result["reply"] == "Live legal response."
    assert result["intent"] == "procedural"

