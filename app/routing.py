"""Intent routing: decide *what kind* of turn this is before any generation.

A legal assistant answers several different things, and they need different
machines:

  procedure  - a known process ("how do I file an FIR") -> curated, reviewed steps
  statute    - what the law says                          -> retrieved authority
  precedent  - what courts decided                        -> judgments, never authority
  draft      - prepare a document                         -> artifact + warning
  emergency  - immediate danger                           -> safety script, no legal analysis
  human      - the caller wants a person                  -> handoff, never automatic

Who decides: the semantic reader in `understanding` reads the turn with a
model and returns a validated intent. This module overlays that reading on a
lexical route that always runs underneath it. The lexical pass is a *net*, not
the decision-maker: it keeps the product working when the provider is down,
and it can raise an emergency the model missed. It can never lower one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import procedural, triage
from .understanding import MIN_CONFIDENCE, Interpretation

ROUTES = ("emergency", "human", "procedural", "statute", "precedent", "draft", "unclear")

# Safety beats everything else. These are patterns, not a keyword classifier
# for the caller's whole request: they only decide whether to interrupt.
EMERGENCY_PATTERNS = (
    r"\b(suicide|kill myself|end my life|self harm|khudkhushi)\b",
    r"\b(he|she|they|husband|father|brother|uncle|landlord)\b[^.]{0,40}\b(is coming|will kill|threaten|threatened|beating|hit me|attacked)\b",
    r"\b(locked (me )?in|not allowed to leave|held captive|abduct|kidnap)\b",
    r"\b(bleeding|unconscious|overdose|poison(ed)?)\b",
    r"\b(coming (right )?now|outside my (house|door)|right now he)\b",
    r"\b(maar raha|maar diya|jaan se|dhamki de raha)\b",
    r"\b(acid attack|molest|rape)\b",
)

HUMAN_PATTERNS = (
    r"\b(talk|speak|connect|refer)\b[^.]{0,20}\b(lawyer|advocate|person|human|someone|vakil|attorney|volunteer|paralegal)\b",
    r"\b(real|actual|qualified) (person|lawyer|advocate)\b",
    r"\b(vakil|lawyer) (se|chahiye|bulaiye)\b",
    r"\bhuman review\b",
)

PROCEDURE_CUES = (
    r"\bhow (do|can|should) i\b", r"\bhow to\b", r"\bwhere (do|can|should) i\b",
    r"\bwhat (documents|papers|forms|fees?)\b", r"\bwhat (do|should) i (need|bring|take)\b",
    r"\b(apply|file|register|submit|complain|report|appeal)\b",
    r"\b(kahan|kaise) (jaun|karun|file)\b", r"\bkaise\b",
)

PRECEDENT_CUES = (
    r"\b(similar|comparable|related) (case|judgment|decision|matter)\b",
    r"\b(precedent|case law|case-law|citation|held in|ruled in)\b",
    r"\b(supreme court|high court|district court)\b[^.]{0,40}\b(said|held|ruled|decided)\b",
    # Plurals matter: "are there judgments where ..." is the commonest way this
    # is asked, and `judgment\b` alone did not match it.
    r"\b(judgments?|judgements?|verdicts?|orders?)\b",
    # "I want the most relevant cases that cover X" is a request for case law
    # in a caller's own words. Without this the turn fell through to a procedure
    # card that happened to share the word "FIR", and a caller who asked to be
    # shown decisions was asked an intake question instead.
    r"\b(relevant|similar|comparable|related|any) cases?\b",
    r"\bcases? (on|about|covering|that cover|where|which|dealing)\b",
)

STATUTE_CUES = (
    r"\b(section|s\.|sec\.|article|clause|provision|amendment)\b[^.]{0,20}\d",
    r"\b(what does|what is|is it|are we|am i)\b[^.]{0,40}\b(law|legal|illegal|allowed|punish|penalty|rights?|liable)\b",
    r"\b(under the|as per|according to)\b[^.]{0,20}\b(act|code|rules?)\b",
    r"\b(kanoon|qanoon) (kya|kehta)\b",
)

DRAFT_CUES = (
    r"\b(draft|write|prepare|compose|make me|generate|create)\b[^.]{0,30}\b(notice|letter|application|complaint|affidavit|agreement|reply|petition|document|pdf)\b",
    r"\b(notice|application|letter|document|draft|petition|complaint|pdf)\b[^.]{0,30}\b(likh|draft|prepare|karo|banao|cheyyi|raayi|banao|ready)\b",
    r"\b(pdf|pdf format|in pdf|download pdf|save pdf|export pdf|format lo)\b",
    r"(పిడిఎఫ్|పీడీఎఫ్|అప్లికేషన్|దరఖాస్తు|తయారు|డ్రాఫ్ట్|రాయండి|చేసివు|చేసి|ఫార్మాట్)",
    r"(पीडीएफ|प्रारूप|ड्राफ्ट|आवेदन|तैयार|बनाओ|लिखो)",
    r"(மனு|விண்ணப்பம்|தயாரி)",
    r"(ಅರ್ಜಿ|ಕರಡು|ಸಿದ್ಧಪಡಿಸಿ)",
    r"(খসড়া|আবেদন|তৈরি)",
)

# Words that carry no routing signal on their own but appear in every request.
NOISE = re.compile(r"\b(please|kindly|help|me|my|our|the|a|an|i|we|you)\b")


@dataclass
class Route:
    name: str = "unclear"
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    procedure: dict[str, Any] | None = None
    wants_human: bool = False
    emergency: bool = False
    risk: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "confidence": round(self.confidence, 2),
            "reasons": self.reasons,
            "procedure_id": (self.procedure or {}).get("id"),
            "wants_human": self.wants_human,
            "emergency": self.emergency,
            "risk_level": self.risk.get("level", "normal"),
            "risk_categories": [item.get("key") for item in self.risk.get("categories", [])],
        }


def _matches(patterns: tuple[str, ...], text: str) -> list[str]:
    hits: list[str] = []
    for pattern in patterns:
        found = re.search(pattern, text)
        if found:
            hits.append(found.group(0).strip())
    return hits


def route_turn(
    text: str,
    jurisdiction: str = "auto",
    reading: Interpretation | None = None,
) -> Route:
    """Route the turn: the reading decides, the lexical pass is the floor."""
    base = _lexical_route(text, jurisdiction)
    return _overlay(base, reading) if reading is not None else base


def _overlay(base: Route, reading: Interpretation) -> Route:
    """Apply a validated model reading without ever weakening safety."""
    result = Route(
        name=base.name,
        confidence=base.confidence,
        reasons=list(base.reasons),
        procedure=base.procedure,
        wants_human=base.wants_human,
        emergency=base.emergency,
        risk=reading.risk or base.risk,
    )
    result.reasons = [f"reader: {reason}" for reason in reading.reasons[:4]] + result.reasons

    if reading.emergency or base.emergency:
        result.name = "emergency"
        result.emergency = True
        result.confidence = max(0.97, reading.confidence)
        return result

    if reading.intent == "human":
        result.name = "human"
        result.wants_human = True
        result.confidence = max(reading.confidence, 0.6)
        return result
    if base.wants_human and reading.intent not in {"procedural", "statute", "precedent", "draft"}:
        result.name = "human"
        result.confidence = max(reading.confidence, 0.7)
        return result

    if reading.confidence < MIN_CONFIDENCE and base.name == "unclear":
        # Not confident enough to act on a category; the engine asks a question.
        result.name = "unclear"
        result.confidence = max(base.confidence, reading.confidence)
        return result

    if reading.intent in ROUTES and reading.intent != "unclear":
        result.name = reading.intent
        result.confidence = max(reading.confidence, 0.5)
        if reading.intent == "procedural":
            chosen = procedural.BY_ID.get(reading.procedure_id or "")
            if chosen and chosen.get("review_status") not in {"rejected", "expired"}:
                result.procedure = {**chosen, "score": 0.0, "matched_signals": ["reader"]}
            elif base.procedure:
                result.procedure = base.procedure
    elif reading.intent == "unclear" and base.name != "unclear":
        # The reader could not tell, but the net found a settled route. Keep it
        # only when the net is specific; otherwise ask.
        result.name = base.name
    return result


def settle(route: dict[str, Any], reading: Interpretation) -> dict[str, Any]:
    """Let a validated reading settle the route *after* the consultation ran.

    `route_turn` has to run before any model, because retrieval and the safety
    net need a route then -- and because a person in danger must not wait for
    one. But that early verdict is a phrase net guessing from four words, and
    letting it outlive the consultation meant the interface reported a category
    the model had already contradicted.

    The rules that keep this safe:
      * an emergency is never lowered, only raised;
      * a caller who asked for a person keeps that route unless the model read
        an actual legal question in the same message;
      * an unconfident reading does not overwrite a settled route.
    """
    result = dict(route)
    base_name = str(route.get("name") or "unclear")
    if base_name == "emergency" or route.get("emergency"):
        return result
    wants_human = bool(route.get("wants_human"))
    reasons = [str(item) for item in route.get("reasons") or []][:6]

    def settle_to(name: str, confidence: float, reason: str) -> dict[str, Any]:
        result["name"] = name
        result["confidence"] = round(max(confidence, float(route.get("confidence") or 0.0)), 2)
        result["reasons"] = [reason, *reasons][:7]
        return result

    if reading.emergency:
        return settle_to("emergency", 0.97, "model: emergency grounded in the caller's words")
    if reading.intent == "emergency":
        # The model said "emergency" but could not point at the caller's words,
        # so validation capped it at `high`. That raises urgency; it does not
        # move the turn onto the safety script or relabel it for the caller.
        return result
    if reading.intent == "human":
        result["wants_human"] = True
        return settle_to("human", max(reading.confidence, 0.6), "model: caller asked for a person")
    if wants_human and reading.intent not in {"procedural", "statute", "precedent", "draft"}:
        return settle_to("human", max(reading.confidence, 0.7), "caller asked for a person")
    if reading.intent not in ROUTES or reading.intent == "unclear":
        # The model could not tell either. Whatever the net settled stands:
        # re-labeling the turn "unclear" would throw away a real match.
        return result
    if reading.confidence < MIN_CONFIDENCE and base_name == "unclear":
        return result
    if reading.intent == "procedural":
        chosen = procedural.BY_ID.get(reading.procedure_id or "")
        if chosen and chosen.get("review_status") not in {"rejected", "expired"}:
            result["procedure_id"] = chosen["id"]
    return settle_to(
        reading.intent, max(reading.confidence, 0.5),
        f"model: read this as {reading.intent} (p={reading.confidence:.2f})",
    )


def _lexical_route(text: str, jurisdiction: str = "auto") -> Route:
    """Phrase-level routing. Recall-oriented safety net and offline fallback."""
    lowered = (text or "").lower()
    result = Route()

    # Two independent detectors. The triage layer knows urgency and what it
    # forbids; the phrase list catches blunt statements the classifier may
    # treat as historical. Either one is enough to stop the turn.
    assessment = triage.assess(lowered)
    result.risk = assessment.as_dict()
    emergency_hits = _matches(EMERGENCY_PATTERNS, lowered)
    if emergency_hits or triage.is_emergency(assessment):
        result.name = "emergency"
        result.emergency = True
        result.confidence = 0.97
        result.reasons = [f"safety phrase: {hit}" for hit in emergency_hits[:2]] or [f"triage: {hit}" for hit in assessment.reasons[:2]]
        return result

    human_hits = _matches(HUMAN_PATTERNS, lowered)
    if human_hits:
        result.wants_human = True

    procedure = procedural.best_match(lowered, jurisdiction)
    procedure_hits = _matches(PROCEDURE_CUES, lowered)
    precedent_hits = _matches(PRECEDENT_CUES, lowered)
    statute_hits = _matches(STATUTE_CUES, lowered)
    draft_hits = _matches(DRAFT_CUES, lowered)

    # A strong procedural match with a procedural cue (or the procedure's own
    # trigger words alone, when they are specific enough) is settled offline.
    # An explicit request for a document outranks a topic match. "Draft a notice
    # asking my landlord to return my deposit" names a procedure and asks for a
    # document; the caller asked for the document, and a walkthrough would be an
    # answer to a question they did not ask. An explicit process cue still keeps
    # the walkthrough, because that is what "how do I ..." is asking for.
    if procedure and (procedure_hits or (procedure["score"] >= 3.0 and not draft_hits)):
        result.name = "procedural"
        result.procedure = procedure
        result.confidence = min(0.95, 0.55 + 0.1 * procedure["score"] + 0.1 * len(procedure_hits))
        result.reasons = [f"procedure match: {procedure['label']}"] + [
            f"cue: {hit}" for hit in procedure_hits[:2]
        ]
        if result.wants_human:
            result.reasons.append("caller also asked for a person")
        return result

    if draft_hits:
        result.name = "draft"
        result.confidence = 0.75
        result.reasons = [f"draft cue: {hit}" for hit in draft_hits[:2]]
        if assessment.has("block_drafting"):
            # Still a draft request, but the turn will not return a document.
            result.reasons.append("drafting blocked while an urgent risk is active")
        return result

    if precedent_hits:
        result.name = "precedent"
        result.confidence = 0.7
        result.reasons = [f"precedent cue: {hit}" for hit in precedent_hits[:2]]
        return result

    if statute_hits:
        result.name = "statute"
        result.confidence = 0.7
        result.reasons = [f"statute cue: {hit}" for hit in statute_hits[:2]]
        return result

    if result.wants_human:
        result.name = "human"
        result.confidence = 0.8
        result.reasons = [f"handoff cue: {hit}" for hit in human_hits[:2]]
        return result

    # Last stop for a curated process: a plain statement of the problem can name
    # one signal word without any "how do I" phrasing, which is how most people
    # actually talk ("company ne salary nahi di", "OTP and the money was gone").
    # This sits after every cue family, so it only decides turns nothing more
    # specific claimed, and the card it returns carries its own review status.
    if procedure:
        result.name = "procedural"
        result.procedure = procedure
        result.confidence = 0.6
        result.reasons = [f"procedure signal: {hit}" for hit in procedure["matched_signals"][:3]]
        return result

    if procedure_hits and procedure is None:
        result.name = "procedural"
        result.confidence = 0.45
        result.reasons = [f"procedure cue without a catalog match: {hit}" for hit in procedure_hits[:2]]
        return result

    result.name = "unclear"
    result.confidence = 0.3
    result.reasons = ["no route matched; ask one clarifying question"]
    return result
