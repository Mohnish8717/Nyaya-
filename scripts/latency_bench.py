"""Latency bench for the turn pipeline, with a simulated provider.

The point is to measure *our* orchestration, not a vendor. The provider is a
parameterised fake: time-to-first-token, tokens per second, and a reply size
per route. Every number printed is wall-clock time through the real
`analyze` / `compose` code path.

    python -m scripts.latency_bench --ttft 350 --tps 120 --runs 7

Baseline assumptions are mid-range published provider figures, stated on the
command line so a reviewer can change them rather than trust them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

from app import legal_engine, model_gateway, understanding
from app.legal_engine import INDIAN_JURISDICTIONS

TURNS = [
    ("how do I file an FIR at the police station", {"intent": "procedural", "procedure_id": "police-fir"}, "intake"),
    ("They refused in writing yesterday, in Karnataka",
     {"intent": "procedural", "procedure_id": "police-fir", "facts": [
         {"slot": "stage", "value": "police_refused", "evidence": "refused"},
         {"slot": "when", "value": "yesterday", "evidence": "yesterday"},
         {"slot": "jurisdiction", "value": "Karnataka", "evidence": "Karnataka"},
     ]}, "intake"),
    ("what does the law say about unpaid wages in Karnataka",
     {"intent": "statute"}, "statute"),
    ("is a similar case where the supreme court held the employer liable",
     {"intent": "precedent", "needs_precedents": True}, "precedent"),
]

REPLY_SIZE = {"intake": 180, "statute": 950, "precedent": 800}
READING_SIZE = 320
# Provider parameters, set by `install_fake_provider`; the legacy emulation
# charges the removed reader call at the same rate.
_TTFT_MS = 350.0
_TPS = 120.0


def _sized(size: int) -> str:
    seed = "The passage explains what the source says about this situation. "
    return (seed * (size // len(seed) + 1))[:size].strip()


def reading(text: str, spec: dict[str, Any]) -> understanding.Interpretation:
    raw = {
        "intent": "unclear", "confidence": 0.0, "language": "en-IN", "risk": [],
        "facts": [], "procedure_id": None, "issue": "", "next_question": None,
        "needs_precedents": False,
    }
    raw.update(spec)
    validated, rejected = understanding.validate(raw, text, INDIAN_JURISDICTIONS)
    return understanding.combine(validated, text, rejected)


async def sleep_ms(ms: float) -> None:
    await asyncio.sleep(ms / 1000.0)


def _sentences(text: str, consumed: int = 0):
    """Split a reply into the sentences a streaming model would emit."""
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    for part in parts:
        if part.strip():
            yield part.strip(), consumed + len(part)


def install_fake_provider(ttft_ms: float, tps: float) -> None:
    """One timed fake of the consultation call, in the real payload shape.

    The reading fields ride in the same object as `reply`, which is the change
    this bench measures: the turn pays one time-to-first-token, not two.
    """
    global _TTFT_MS, _TPS
    _TTFT_MS, _TPS = ttft_ms, tps
    readings = {text: reading(text, spec) for text, spec, _ in TURNS}

    def reading_fields(context: dict[str, Any]) -> dict[str, Any]:
        result = readings.get(context["caller_message"])
        if result is None:
            return {"intent": "unclear", "confidence": 0.0, "risk": [], "reported_facts": []}
        return {
            "intent": result.intent, "confidence": result.confidence,
            "language": "en-IN", "risk": result.risk.get("categories", []),
            "reported_facts": result.facts, "procedure_id": result.procedure_id,
            "issue": result.issue, "next_question": result.next_question,
            "needs_precedents": result.needs_precedents,
            "search_query": result.search_query,
        }

    def body(context):
        if context.get("intake"):
            return _sized(REPLY_SIZE["intake"])
        route = (context.get("route") or {}).get("name")
        return _sized(REPLY_SIZE.get(route, REPLY_SIZE["statute"]))

    def envelope(context: dict[str, Any]) -> dict[str, Any]:
        text = body(context)
        return {
            "reply": text, "language": "en-IN", "issue": "",
            **reading_fields(context),
            "legal_claims": [], "case_context": [], "services_used": [],
            "follow_up": None, "artifact": None, "risk_flags": [], "source_gap": None,
        }

    async def generate_turn(context, tools=None, extra_messages=None):
        payload = envelope(context)
        await sleep_ms(ttft_ms + len(payload["reply"]) / 4 / max(tps, 1) * 1000)
        return payload

    async def generate_turn_stream(context, tools=None):
        text = body(context)
        await sleep_ms(ttft_ms)
        consumed = 0
        for split in _sentences(text, consumed):
            sentence, consumed = split
            await sleep_ms(len(sentence) / 4 / max(tps, 1) * 1000)
            yield "sentence", sentence
        yield "final", envelope(context)

    model_gateway.configured = lambda: True
    legal_engine.model_configured = lambda: True
    legal_engine.generate_turn = generate_turn
    legal_engine.generate_turn_stream = generate_turn_stream
    model_gateway.LAST_ERROR = None


async def legacy_answer_stream(text: str, session: dict[str, Any]):
    """The previous ordering: read, then acknowledge, then generate in full.

    Kept in the bench so the before/after numbers share one provider, one
    payload size and one process. It is an emulation of the old control flow,
    not a run of the old file: the removed reader call is charged here at the
    same provider rate the old reader paid.
    """
    await sleep_ms(_TTFT_MS + READING_SIZE / 4 / max(_TPS, 1) * 1000)
    context = await legal_engine.analyze(text, "en-IN", "auto", None, [], session)
    yield "stage", {"stage": "routed", "route": context.route}
    yield "stage", {"stage": "evidence", "sources": len(context.sources)}
    yield "ack", {"text": "Let me check the current sources for that."}
    yield "stage", {"stage": "composing"}
    yield "final", await legal_engine.compose(context)


async def one_turn(text: str, session: dict[str, Any], baseline: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    marks: dict[str, float] = {}
    final: dict[str, Any] | None = None
    stream = legacy_answer_stream(text, session) if baseline else legal_engine.answer_stream(text, "en-IN", "auto", None, [], session=session)
    async for kind, data in stream:
        elapsed = (time.perf_counter() - started) * 1000
        if kind not in marks:
            marks[kind] = round(elapsed, 1)
        if kind == "speak":
            marks.setdefault("first_speech", round(elapsed, 1))
        if kind == "final":
            final = data
    total = (time.perf_counter() - started) * 1000
    state = (final or {}).get("conversation_state", {})
    nxt = {
        "turns": session["turns"] + 1,
        "facts": state.get("ledger", {}),
        "last_state": {"workflow": state.get("workflow")},
    }
    return {"marks": marks, "total": round(total, 1), "session": nxt, "result": final}


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[index], 1)


async def measure(runs: int, baseline: bool) -> dict[str, Any]:
    ack: list[float] = []
    per_route: dict[str, dict[str, list[float]]] = {}
    for _ in range(runs):
        session = {"turns": 0, "facts": {}, "last_state": {}}
        for text, _spec, route in TURNS:
            out = await one_turn(text, session, baseline=baseline)
            session = out["session"]
            ack.append(out["marks"].get("ack", out["total"]))
            bucket = per_route.setdefault(route, {"speech": [], "final": []})
            bucket["speech"].append(out["marks"].get("first_speech", out["total"]))
            bucket["final"].append(out["total"])
    return {
        "ack_ms": {"p50": pct(ack, 0.50), "p95": pct(ack, 0.95)},
        "routes": {
            route: {
                "first_speech_p50": pct(bucket["speech"], 0.50),
                "final_p50": pct(bucket["final"], 0.50),
                "final_p95": pct(bucket["final"], 0.95),
            }
            for route, bucket in sorted(per_route.items())
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttft", type=float, default=350.0, help="provider time to first token, ms")
    parser.add_argument("--tps", type=float, default=120.0, help="provider decode speed, tokens/s")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--statute-chars", type=int, default=950,
                        help="simulated answer length in characters for a legal answer")
    args = parser.parse_args()
    REPLY_SIZE["statute"] = args.statute_chars

    install_fake_provider(args.ttft, args.tps)
    print(json.dumps({
        "provider": {"ttft_ms": args.ttft, "tokens_per_second": args.tps},
        "runs": args.runs,
        "previous_ordering": await measure(args.runs, baseline=True),
        "current": await measure(args.runs, baseline=False),
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
