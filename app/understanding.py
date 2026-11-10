"""Validation of the model's reading of a turn: the model judges, code checks.

The previous design decided what a turn *meant* with phrase lists. That broke
in exactly the way a caller notices: one unexpected word and a wage dispute is
filed as "unclear"; a paraphrase of "he is going to kill me" misses the
emergency list; a state named in Devanagari is not recognised at all.

So the model reads the turn. It does so inside the *same* call that answers
it: one consultation turn returns the reply and the reading together (intent,
risk, facts, the one next question, the search phrasing). A second extraction
call in front of every answer doubled the token spend -- the thing that turned
one throttled question into a 37-second turn -- and taught the model to answer
in two disconnected halves.

Code does four things the model cannot be trusted with:

  1. **Closed vocabularies.** An intent, risk category or slot outside the
     published set is discarded, so a new category cannot silently appear at
     runtime and bypass the policy table.
  2. **Grounding.** Every fact and every risk finding must carry a verbatim
     span from words the caller actually said -- this turn, or an earlier turn
     in the same call. Ungrounded facts are dropped: a model guess must never
     become session state. An ungrounded `critical` is capped at `high`,
     because inventing an emergency is its own harm.
  3. **Monotone safety.** Risk is the union of the model's reading and the
     lexical net. The model can raise severity; it can never lower it. If the
     provider is down, the net still runs the turn.
  4. **Fallback.** No model, invalid JSON, or exhausted retries degrades to the
     lexical route, and the trace says so. Behaviour never silently pretends a
     model read the turn.

That is the division of labour: judgement comes from the model, invariants
come from code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from . import facts, procedural, triage

INTENTS = ("emergency", "human", "procedural", "statute", "precedent", "draft", "unclear")
SLOTS = ("jurisdiction", "when", "amount", "notice_or_order", "stage", "recipient", "outcome", "situation")
SEVERITIES = ("normal", "high", "critical")
SEVERITY_ORDER = {"normal": 0, "high": 1, "critical": 2}
# Categories the policy table knows. `other` is accepted but carries no policy:
# a novel risk still raises severity, it just cannot claim a safety script.
KNOWN_CATEGORIES = frozenset(category.key for category in triage.CATEGORIES) | {"other"}
MAX_QUESTION = 240
MIN_CONFIDENCE = 0.45


class RiskFinding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: str = "other"
    severity: str = "normal"
    evidence: str = ""


class FactClaim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    slot: str = ""
    value: str = ""
    evidence: str = ""
    # How the value is read ("around ten thousand" -> "INR 10000"). It never
    # replaces `value`, which stays what the caller said.
    normalized: str = ""
    # How they said it, before normalisation, so the caller can hear their own
    # phrase back. Defaults to the evidence span when the model omits it.
    original_expression: str = ""
    # exact | approximate | relative | reported | unknown
    precision: str = "reported"
    # caller_confirmed | corrected | estimated | reported | unresolved | disputed
    status: str = "reported"
    confidence: str = "high"
    # Only meaningful on a correction: why the value changed. The ledger keeps
    # its own `previous_value` from what it actually held, because a model that
    # misremembers turn four must not rewrite the record of it.
    change_reason: str = ""


class Reading(BaseModel):
    """The model's typed answer. Everything outside this shape is ignored."""

    model_config = ConfigDict(extra="ignore")

    intent: str = "unclear"
    confidence: float = 0.0
    language: str | None = None
    risk: list[RiskFinding] = Field(default_factory=list)
    facts: list[FactClaim] = Field(default_factory=list)
    procedure_id: str | None = None
    issue: str = ""
    next_question: str | None = None
    needs_precedents: bool = False
    # The turn's question written in the vocabulary Indian judgments use. Not a
    # translation, and not a summary: a caller says "he changed the lock" and
    # the judgment says "dispossession", a caller asks in Hindi and the corpus
    # is English. Retrieval searches both phrasings; the caller's own words are
    # never replaced by this.
    search_query: str = ""


@dataclass
class Interpretation:
    """What the turn is, after validation, grounding and the safety net."""

    intent: str = "unclear"
    confidence: float = 0.0
    language: str | None = None
    procedure_id: str | None = None
    issue: str = ""
    next_question: str | None = None
    needs_precedents: bool = False
    search_query: str = ""
    facts: list[dict[str, str]] = field(default_factory=list)
    risk: dict[str, Any] = field(default_factory=lambda: {"level": "normal", "categories": [], "policies": [], "reasons": []})
    emergency: bool = False
    source: str = "lexical"  # model | lexical | model+net
    reasons: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    latency_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 2),
            "reachable_language": self.language,
            "procedure_id": self.procedure_id,
            "issue": self.issue,
            "next_question": self.next_question,
            "needs_precedents": self.needs_precedents,
            "search_query": self.search_query,
            "facts": self.facts,
            "risk_level": self.risk.get("level"),
            "risk_categories": [item.get("key") for item in self.risk.get("categories", [])],
            "emergency": self.emergency,
            "source": self.source,
            "reasons": self.reasons,
            "rejected": self.rejected,
            "latency_ms": self.latency_ms,
        }


def _norm(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _grounded(span: str, text: str, minimum: int = 3) -> bool:
    """True when `span` really occurs in the caller's message."""
    needle = _norm(span)
    return len(needle) >= minimum and needle in _norm(text)


def _norm_question(value: str) -> str:
    return _norm(value).rstrip("?!. ")[:MAX_QUESTION]


def _clean_question(raw: str | None, known: set[str]) -> str | None:
    if not isinstance(raw, str):
        return None
    text = " ".join(raw.split())[:MAX_QUESTION]
    if len(text) < 8:
        return None
    if not text.endswith("?"):
        text = text.rstrip(".!;: ") + "?"
    if _norm_question(text) in known:
        return None
    return text


def validate(
    raw: Any,
    text: str,
    jurisdictions: set[str] | None = None,
    asked: list[str] | None = None,
    transcript: str = "",
) -> tuple[Reading, list[str]]:
    """Turn arbitrary model output into a reading that obeys the contracts.

    `text` is the caller's message this turn; `transcript` is what they said in
    the earlier turns of the same call. A fact may be grounded in either -- a
    caller who says "still not paid" is restating the wages they described in
turn one, and dropping that because the words are old is how a conversation
    forgets. A *critical* risk is stricter: it must be grounded in the current
    message, because an emergency is about now.

    Returns the sanitised reading plus a list of rejection codes, which land in
    the trace so a reviewer can see what the model tried to say and why it was
    not allowed to say it.
    """
    rejected: list[str] = []
    if isinstance(raw, dict) and "reported_facts" in raw and "facts" not in raw:
        # The unified consultation contract calls this field `reported_facts`;
        # `facts` is the older reading schema. One validator, both contracts.
        raw = {**raw, "facts": raw.get("reported_facts")}
    try:
        reading = Reading.model_validate(raw if isinstance(raw, dict) else {})
    except Exception:
        return Reading(), ["reading_unparseable"]

    if reading.intent not in INTENTS:
        rejected.append(f"intent_not_allowed:{reading.intent[:24]}")
        reading.intent = "unclear"
    reading.confidence = max(0.0, min(1.0, float(reading.confidence or 0.0)))
    reading.issue = " ".join(str(reading.issue or "").split())[:200]

    kept_risk: list[RiskFinding] = []
    for finding in reading.risk[:8]:
        category = str(finding.category or "other").strip()
        severity = str(finding.severity or "normal").strip().lower()
        evidence = " ".join(str(finding.evidence or "").split())[:400]
        if category not in KNOWN_CATEGORIES:
            rejected.append(f"risk_category_not_allowed:{category[:24]}")
            category = "other"
        if severity not in SEVERITIES:
            rejected.append(f"risk_severity_not_allowed:{severity[:12]}")
            severity = "normal"
        grounded = _grounded(evidence, text)
        # An inferred emergency is capped: a model that cannot point at the
        # caller's words may still raise urgency, but not trigger the script.
        if severity == "critical" and not grounded:
            rejected.append(f"critical_risk_ungrounded:{category}")
            severity = "high"
        kept_risk.append(RiskFinding(category=category, severity=severity, evidence=evidence))
    reading.risk = kept_risk

    known_questions = {_norm_question(item) for item in (asked or [])}
    reading.next_question = _clean_question(reading.next_question, known_questions)

    kept_facts: list[FactClaim] = []
    seen_slots: set[str] = set()
    # Facts may cite either this message or an earlier one in the same call.
    grounding_text = text if not transcript else f"{text}\n{transcript}"
    for fact in reading.facts[:12]:
        slot = str(fact.slot or "").strip()
        value = " ".join(str(fact.value or "").split())[:200]
        evidence = " ".join(str(fact.evidence or "").split())[:400]
        if slot not in SLOTS:
            rejected.append(f"fact_slot_not_allowed:{slot[:24]}")
            continue
        if slot in seen_slots:
            continue
        if not value:
            continue
        if not _grounded(evidence, grounding_text):
            rejected.append(f"fact_ungrounded:{slot}")
            continue
        if slot == "jurisdiction":
            code = value.upper()
            if code not in (jurisdictions or set()):
                # The reader may answer "Karnataka" or the code; anything that
                # is neither is dropped rather than trusted.
                code = facts.STATE_NAMES.get(value.strip().lower(), "")
            if not code:
                rejected.append("jurisdiction_not_known")
                continue
            value = code
        seen_slots.add(slot)
        raw_status = str(fact.status or "reported").strip().lower()
        # A model may not assert a verification the application did not run.
        # "document_supported" and "externally_verified" describe work no stage
        # in this turn performed, so they are recorded as what they are -- a
        # report from the caller -- and the attempt is named in the trace.
        if raw_status in {"document_supported", "externally_verified"}:
            rejected.append(f"fact_verification_not_run:{slot}")
        status = facts.canonical_status(raw_status)
        confidence = str(fact.confidence or "high").strip().lower()
        kept_facts.append(FactClaim(
            slot=slot, value=value, evidence=evidence,
            normalized=" ".join(str(fact.normalized or "").split())[:200],
            original_expression=" ".join(str(fact.original_expression or "").split())[:400] or evidence,
            precision=facts.canonical_precision(fact.precision),
            status=status,
            confidence=confidence if confidence in {"high", "medium", "low"} else "medium",
            change_reason=" ".join(str(fact.change_reason or "").split())[:200],
        ))
    reading.facts = kept_facts

    if reading.procedure_id:
        procedure = procedural.BY_ID.get(str(reading.procedure_id))
        if not procedure or procedure.get("review_status") in {"rejected", "expired"}:
            rejected.append(f"procedure_not_in_catalog:{str(reading.procedure_id)[:32]}")
            reading.procedure_id = None
        else:
            reading.procedure_id = procedure["id"]
    if not isinstance(reading.needs_precedents, bool):
        reading.needs_precedents = bool(reading.needs_precedents)

    # A search query is a vocabulary hint, never a fact and never a conclusion,
    # so it is not grounded against the caller's words -- the whole point is
    # that it says something in different words.
    #
    # It *may* contain section numbers the caller never said. That rule looked
    # safe and was actively harmful: a caller reports a refused FIR, the reader
    # correctly rewrites the search as "registration of FIR for a cognizable
    # offence under section 154", and the guard deleted it for inventing "154"
    # — the one word that separates the leading authorities from income-tax
    # "refusal of registration" cases. A wrong number in a search query cannot
    # become a false statement to the caller: the query only selects passages
    # that already exist, every quoted passage is checked as an exact
    # substring, and the claims that reach the caller are verified separately.
    # The bounds that do matter are kept: one line, no markup, no URLs, a hard
    # length cap.
    search_query = " ".join(str(reading.search_query or "").split())[:240]
    if search_query:
        if len(search_query) < 8 or "http" in search_query.lower() or "[" in search_query:
            rejected.append("search_query_unusable")
            search_query = ""
    reading.search_query = search_query

    return reading, rejected


def _risk_from(reading: Reading, net: triage.Assessment) -> dict[str, Any]:
    """Union of model findings and the lexical net. Severity only ever rises."""
    level = net.level
    categories: list[dict[str, Any]] = list(net.categories)
    policies: list[str] = list(net.policies)
    reasons: list[str] = list(net.reasons)

    for finding in reading.risk:
        if SEVERITY_ORDER[finding.severity] > SEVERITY_ORDER[level]:
            level = finding.severity
        categories.append({
            "key": finding.category,
            "severity": finding.severity,
            "evidence": finding.evidence[:120],
            "source": "model",
        })
        for policy in triage.policies_for(finding.category):
            if policy not in policies:
                policies.append(policy)
        reasons.append(f"model:{finding.category}:{finding.severity}")
    return {"level": level, "categories": categories, "policies": policies, "reasons": reasons[:12]}


def combine(reading: Reading | None, text: str, rejected: list[str] | None = None) -> Interpretation:
    """Fold a validated reading into the lexical net that always runs."""
    net = triage.assess(text)
    result = Interpretation(risk=_risk_from(reading, net) if reading else net.as_dict())
    result.rejected = list(rejected or [])
    net_emergency = triage.is_emergency(net)

    if reading is None:
        result.intent = "emergency" if net_emergency else "unclear"
        result.confidence = 0.0
        result.emergency = net_emergency
        result.source = "lexical"
        result.reasons = ["no model reading; lexical net decided"]
        return result

    result.source = "model+net" if net.categories else "model"
    result.intent = reading.intent
    result.confidence = reading.confidence
    result.language = reading.language
    result.procedure_id = reading.procedure_id
    result.issue = reading.issue
    result.next_question = reading.next_question
    result.needs_precedents = reading.needs_precedents
    result.search_query = reading.search_query
    result.facts = [fact.model_dump() for fact in reading.facts]
    result.reasons = [f"model intent={reading.intent} p={reading.confidence:.2f}"] + [
        f"risk:{item.category}:{item.severity}" for item in reading.risk
    ]

    # `critical` here can only be a grounded model finding (validation caps an
    # ungrounded one at `high`) or a hit from the lexical net. An `emergency`
    # intent with no evidence behind it is not an emergency: it raises urgency
    # and is reported, but it cannot trigger the safety script on the model's
    # say-so. The old union did exactly that, which made a hallucinated
    # emergency as easy as one word in the JSON.
    critical = any(item.get("severity") == "critical" for item in result.risk["categories"])
    result.emergency = bool(net_emergency or critical)
    if reading.intent == "emergency" and not result.emergency:
        result.reasons.append("emergency intent without grounded evidence; urgency raised, script not triggered")
        if SEVERITY_ORDER[result.risk["level"]] < SEVERITY_ORDER["high"]:
            result.risk["level"] = "high"
        result.intent = "emergency"
    elif result.emergency:
        if reading.intent != "emergency":
            result.reasons.append("emergency raised by the safety net")
        result.intent = "emergency"
    if result.intent == "procedural" and not result.procedure_id:
        # A process question with no catalogue match is still a process
        # question; the engine will answer with reviewed steps or ask.
        result.reasons.append("procedural intent without a catalog match")
    if result.intent == "unclear" and result.confidence >= MIN_CONFIDENCE and result.next_question:
        result.reasons.append("model asked a clarifying question")
    return result


def from_consultation(
    payload: dict[str, Any] | None,
    text: str,
    *,
    jurisdictions: set[str] | None = None,
    asked: list[str] | None = None,
    transcript: str = "",
) -> Interpretation:
    """The reading carried by the single consultation reply.

    There is no second call to interpret the turn: the same object that holds
    `reply` holds `intent`, `risk` and `reported_facts`, and it goes through
    exactly the validation it would have gone through as a separate reading.
    A missing object is not an error -- it degrades to the lexical net, and the
    trace says the model did not read the turn.
    """
    if not isinstance(payload, dict) or not payload:
        result = combine(None, text)
        result.reasons.append("consultation reply unavailable")
        return result
    reading, rejected = validate(payload, text, jurisdictions, asked, transcript)
    return combine(reading, text, rejected)
