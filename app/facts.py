"""Conversation state: a fact ledger instead of a chat transcript.

Best-in-class trust-critical agents (clinical triage, legal intake, enterprise
voice agents) do not re-derive what the caller said from the last N messages.
They keep an explicit, inspectable slot table: which facts are known, where
each came from, which are still missing, and which conflict.

That matters here for three concrete failures a transcript-only design has:

  * asking the same question twice because the answer was 13 messages ago;
  * letting the model "remember" a fact the caller never stated;
  * silently choosing one of two conflicting statements.

Everything in this module is deterministic. The model *proposes* facts in its
single consultation turn; this module validates, normalises and commits them,
and it keeps provenance on every one: the verbatim span, which turn it came
from, whether the caller confirmed it, corrected it or was estimating. A
correction replaces the value and keeps the earlier one in `history`, so the
record shows both what was said first and what was said last.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Slot order per route. Earlier slots are more material: for a statutory
# question, which state's law applies beats almost anything else.
ORDER: dict[str, list[str]] = {
    "statute": ["jurisdiction", "when", "notice_or_order"],
    "precedent": ["jurisdiction", "when", "outcome"],
    "procedural": ["jurisdiction", "stage", "amount", "when"],
    "draft": ["jurisdiction", "recipient", "amount", "when"],
    "unclear": ["jurisdiction", "situation"],
    "human": ["jurisdiction"],
    "emergency": [],
}

QUESTION = {
    "jurisdiction": "Which state or union territory are you in?",
    "when": "When did this happen, even roughly?",
    "amount": "How much money is involved?",
    "notice_or_order": "Have you received any notice, order or letter about this?",
    "outcome": "What outcome are you hoping for?",
    "stage": "What have you already tried, and what did they say?",
    "recipient": "Who is the notice or letter addressed to?",
    "situation": "What happened, in your own words?",
}

STATE_NAMES = {
    "andhra pradesh": "IN-AP", "arunachal pradesh": "IN-AR", "assam": "IN-AS", "bihar": "IN-BR",
    "chhattisgarh": "IN-CT", "delhi": "IN-DL", "goa": "IN-GA", "gujarat": "IN-GJ", "haryana": "IN-HR",
    "himachal pradesh": "IN-HP", "jharkhand": "IN-JH", "karnataka": "IN-KA", "kerala": "IN-KL",
    "madhya pradesh": "IN-MP", "maharashtra": "IN-MH", "odisha": "IN-OR", "orissa": "IN-OR",
    "punjab": "IN-PB", "rajasthan": "IN-RJ", "sikkim": "IN-SK", "tamil nadu": "IN-TN",
    "telangana": "IN-TS", "uttar pradesh": "IN-UP", "uttarakhand": "IN-UT", "west bengal": "IN-WB",
    "jammu and kashmir": "IN-JK", "jammu & kashmir": "IN-JK", "ladakh": "IN-LA", "puducherry": "IN-PY",
}

DATE_PATTERNS = (
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b",
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b",
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:\s+\d{4})?\b",
    # Month + year without a day is still the material fact for a deadline.
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b",
    r"\b(?:yesterday|last week|last month|last year|today|aaj|kal)\b",
)
# Currency either side of the number: speech often arrives as "40000 rupees"
# and typing as "₹40,000". A bare number is not treated as money.
AMOUNT_RE = re.compile(
    r"(?:(?:₹|rs\.?|inr)\s?[\d,]+(?:\.\d+)?(?:\s?(?:lakh|lakhs|crore|crores|k))?"
    r"|\b[\d,]{3,}(?:\.\d+)?\s?(?:rupees?|rupaye|rupaya|₹)\b)",
    re.I,
)
AMOUNT_WORDS_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:lakh|lakhs|crore|crores)\b", re.I)
BOOLEAN_PATTERNS = {
    "police_refused": (r"\bpolice\b[^.]{0,30}\b(refus|denied|not regist|nahi likh)", "The police refused to register it"),
    "notice_received": (r"\b(notice|summons|order)\b[^.]{0,20}\b(received|mila|aaya|came|got)\b", "A notice or order was received"),
    "notice_not_received": (r"\b(no|without|not received|nahi mila)\b[^.]{0,20}\b(notice|summons)\b", "No notice has been received"),
    "salary_unpaid": (r"\b(salary|wages|tankha)\b[^.]{0,30}\b(not paid|unpaid|nahi mili|pending)\b", "Wages are unpaid"),
    "cheque_returned": (r"\bcheque\b[^.]{0,30}\b(bounce|return|dishonou?r|insufficient)\b", "A cheque was dishonoured"),
    "fir_filed": (r"\b(fir|complaint)\b[^.]{0,20}\b(filed|registered|likh|darj)\b", "A complaint was already filed"),
}

INJECTION_RE = re.compile(
    r"(ignore (all )?(previous|prior|above) instructions|system prompt|you are now|"
    r"disregard (the )?(rules|policy)|reveal your (prompt|instructions))",
    re.I,
)


# What the ledger is allowed to say about a value. The first four are what the
# model proposes; the last two describe evidence the application cannot create
# by itself. `unresolved` is the ledger's own mark for "read by a pattern, not
# confirmed by anyone" -- the lexical floor's facts carry it so a grounded
# model reading can replace them instead of being blocked by them.
FACT_STATUS = (
    "caller_confirmed", "corrected", "estimated", "reported", "unresolved", "disputed",
)
# How the value was expressed, independent of how settled it is. "Approximately
# ten thousand" is an estimate that is precise about its own uncertainty, and a
# relative time ("two months ago") is a real fact that must not be turned into
# an invented date.
FACT_PRECISION = ("exact", "approximate", "relative", "reported", "unknown")
# A model that uses the longer vocabulary is understood, not obeyed: a claim of
# documentary or external verification is recorded as what it actually is --
# something the caller said -- because no verification ran. Reporting a review
# that did not happen is the one thing this product must never do.
STATUS_ALIASES = {
    "caller_reported": "reported",
    "approximate": "estimated",
    "document_supported": "reported",
    "externally_verified": "reported",
}


def canonical_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = STATUS_ALIASES.get(text, text)
    return text if text in FACT_STATUS else "reported"


def canonical_precision(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in FACT_PRECISION else "reported"


def _clean(value: Any, limit: int = 200) -> str:
    return " ".join(str(value or "").split())[:limit]


@dataclass
class Fact:
    slot: str
    value: str
    source: str  # caller | plan | ui | reader
    turn: int = 0
    # Which intake this fact belongs to. Empty means "said outside an intake",
    # which counts for whichever intake is open. Scoping matters because the
    # dates of a police complaint must not satisfy an employment intake: a
    # lawyer does not let one matter's facts answer another matter's questions.
    matter: str = ""
    # Provenance. `evidence` is the caller's own span, checked verbatim before
    # the fact is recorded; `normalized` is how the value is read ("around ten
    # thousand" -> "INR 10000") and never replaces what they said.
    normalized: str = ""
    evidence: str = ""
    status: str = "caller_confirmed"
    confidence: str = "high"
    updated_turn: int = 0
    # How the caller actually said it, before normalisation. A caller who says
    # "around ten grand" must be able to hear that phrase back, not only the
    # number the reader derived from it.
    original_expression: str = ""
    precision: str = "reported"
    # Set only on a correction: what the slot held before, and why it changed.
    # The history keeps every earlier value; these two make the newest change
    # legible without reading the list.
    previous_value: str = ""
    change_reason: str = ""
    # Which turn the value was first stated in, for a reviewer reconstructing
    # the call. `turn` is the first turn, `updated_turn` the latest.
    source_turn: int = 0
    # Earlier values this slot held, newest last: a correction is a change of
    # mind, not a mistake to erase.
    history: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot, "value": self.value, "source": self.source,
            "turn": self.turn, "matter": self.matter,
            "normalized": self.normalized, "evidence": self.evidence,
            "status": self.status, "confidence": self.confidence,
            "updated_turn": self.updated_turn,
            "original_expression": self.original_expression,
            "precision": self.precision,
            "previous_value": self.previous_value,
            "change_reason": self.change_reason,
            "source_turn": self.source_turn or self.turn,
            "history": self.history[-5:],
        }


@dataclass
class Ledger:
    facts: dict[str, Fact] = field(default_factory=dict)
    asked: list[str] = field(default_factory=list)
    contradictions: list[dict[str, str]] = field(default_factory=list)
    injections: int = 0
    # Matters raised while another intake is running. A person on a call
    # mentions the landlord, then the wages, then the landlord again; holding
    # the second thread is what a lawyer does instead of losing it.
    matters: list[dict[str, Any]] = field(default_factory=list)

    # --- serialisation -------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "facts": {slot: fact.as_dict() for slot, fact in self.facts.items()},
            "asked": self.asked,
            "contradictions": self.contradictions,
            "injections": self.injections,
            "matters": self.matters[-4:],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Ledger":
        if not isinstance(data, dict):
            return cls()
        ledger = cls()
        for slot, fact in (data.get("facts") or {}).items():
            if isinstance(fact, dict) and fact.get("value"):
                history = [
                    item for item in (fact.get("history") or [])
                    if isinstance(item, dict) and item.get("value")
                ][-5:]
                ledger.facts[slot] = Fact(
                    slot, str(fact["value"]), str(fact.get("source", "caller")),
                    int(fact.get("turn", 0)), str(fact.get("matter", "") or ""),
                    normalized=str(fact.get("normalized", "") or "")[:200],
                    evidence=str(fact.get("evidence", "") or "")[:400],
                    status=str(fact.get("status", "caller_confirmed") or "caller_confirmed"),
                    confidence=str(fact.get("confidence", "high") or "high"),
                    updated_turn=int(fact.get("updated_turn", fact.get("turn", 0)) or 0),
                    original_expression=str(fact.get("original_expression", "") or "")[:400],
                    precision=canonical_precision(fact.get("precision")),
                    previous_value=str(fact.get("previous_value", "") or "")[:200],
                    change_reason=str(fact.get("change_reason", "") or "")[:200],
                    source_turn=int(fact.get("source_turn", fact.get("turn", 0)) or 0),
                    history=history,
                )
        ledger.asked = [slot for slot in data.get("asked", []) if isinstance(slot, str)]
        ledger.contradictions = [c for c in data.get("contradictions", []) if isinstance(c, dict)]
        ledger.injections = int(data.get("injections", 0) or 0)
        ledger.matters = [
            {"procedure_id": str(item["procedure_id"]), "issue": str(item.get("issue", ""))[:200],
             "turn": int(item.get("turn", 0) or 0), "status": str(item.get("status", "noted"))}
            for item in (data.get("matters") or [])
            if isinstance(item, dict) and item.get("procedure_id")
        ][-4:]
        return ledger

    # --- mutation ------------------------------------------------------
    def record(
        self, slot: str, value: str, source: str, turn: int, kind: str = "text",
        matter: str = "", *, normalized: str = "", evidence: str = "",
        status: str = "caller_confirmed", confidence: str = "high",
        original_expression: str = "", precision: str = "reported",
        change_reason: str = "",
    ) -> None:
        value = _clean(value)
        if not value:
            return
        status = canonical_status(status)
        provenance = dict(
            normalized=_clean(normalized), evidence=_clean(evidence, 400),
            status=status,
            confidence=confidence if confidence in {"high", "medium", "low"} else "medium",
            updated_turn=turn,
            original_expression=_clean(original_expression, 400),
            precision=canonical_precision(precision),
            source_turn=turn,
        )
        existing = self.facts.get(slot)
        if existing and existing.value and existing.value.lower() != value.lower():
            if existing.matter and matter and existing.matter != matter:
                # A different matter: this is a new fact, not a contradiction.
                self.facts[slot] = Fact(slot, value, source, turn, matter, **provenance)
                return
            # A caller can change their account ("actually, it is twelve
            # thousand") or state a different figure the model read as one of
            # two values. Either way the newest statement wins; the earlier one
            # stays in `history` and in `contradictions`, so nothing is erased
            # and a reviewer can see the call changed its mind.
            corrected = (
                status == "corrected"
                or kind in {"date", "amount", "jurisdiction"}
                # A grounded model reading beats a pattern match on the same
                # sentence. The pattern match is recorded with source `net`
                # precisely so this can happen to it, and never to something
                # the caller typed or picked in the interface.
                or (existing.source == "net" and source == "reader")
            )
            if corrected:
                # The line is between "the same words read better" and "the
                # caller said something different". A pattern match on turn 3
                # being replaced by the model's grounded reading of turn 3 is
                # the machine re-reading one message; there is nothing for the
                # caller to resolve. Two separate messages that disagree are a
                # contradiction whoever read them, and suppressing it because
                # both happened to be pattern matches would hide a caller who
                # changed their story.
                same_utterance = (
                    source == "reader"
                    and existing.source == "net"
                    and turn <= (existing.updated_turn or existing.turn)
                )
                if not same_utterance:
                    self.contradictions.append({
                        "slot": slot, "earlier": existing.value, "later": value,
                        "turn": turn,
                        "note": "caller stated two different values; newest kept, both shown for review",
                    })
                    self.contradictions = self.contradictions[-5:]
                history = [*existing.history[-4:], {
                    "value": existing.value, "turn": existing.updated_turn or existing.turn,
                    "status": existing.status,
                }]
                provenance["previous_value"] = existing.value
                provenance["change_reason"] = _clean(
                    change_reason or ("caller correction" if status == "corrected" else "re-read more precisely"),
                    200,
                )
                # The first statement is still the source turn: a correction
                # does not restate when the matter began.
                provenance["source_turn"] = existing.source_turn or existing.turn or turn
                self.facts[slot] = Fact(
                    slot, value, source, existing.turn or turn,
                    matter or existing.matter, history=history, **provenance,
                )
                return
            return
        if existing:
            # Same value restated: keep the first provenance but let a later
            # turn strengthen it (an estimate becomes confirmed, a correction
            # status is not lost by a plain restatement).
            history = existing.history
            if status == "corrected" and existing.status != "corrected":
                history = [*history[-4:], {
                    "value": existing.value, "turn": existing.updated_turn or existing.turn,
                    "status": existing.status,
                }]
            self.facts[slot] = Fact(
                slot, value, source, existing.turn or turn, matter or existing.matter,
                normalized=provenance["normalized"] or existing.normalized,
                evidence=provenance["evidence"] or existing.evidence,
                # A restatement never weakens what is known: a plain "yes"
                # does not undo the caller's earlier certainty.
                status=status if status == "corrected" else existing.status,
                confidence=provenance["confidence"],
                updated_turn=turn,
                original_expression=provenance["original_expression"] or existing.original_expression,
                precision=(
                    provenance["precision"] if provenance["precision"] != "reported"
                    else existing.precision
                ),
                source_turn=existing.source_turn or existing.turn or turn,
                history=history,
            )
            return
        self.facts[slot] = Fact(slot, value, source, turn, matter, **provenance)

    def mark_asked(self, slot: str) -> None:
        if slot not in self.asked:
            self.asked.append(slot)
            self.asked = self.asked[-12:]

    def known(self, slot: str) -> bool:
        return bool(self.facts.get(slot))

    def value(self, slot: str) -> str | None:
        fact = self.facts.get(slot)
        return fact.value if fact else None

    def known_for(self, slot: str, matter: str = "") -> bool:
        """Whether a fact is usable by the given intake."""
        fact = self.facts.get(slot)
        if not fact:
            return False
        return not matter or not fact.matter or fact.matter == matter

    def value_for(self, slot: str, matter: str = "") -> str | None:
        fact = self.facts.get(slot)
        if not fact:
            return None
        return fact.value if (not matter or not fact.matter or fact.matter == matter) else None

    def tag_matter(self, turn: int, matter: str, exclude: tuple[str, ...] = ("jurisdiction",)) -> list[str]:
        """Attach this turn's new facts to the intake that is now open."""
        tagged: list[str] = []
        for slot, fact in self.facts.items():
            if fact.turn == turn and not fact.matter and slot not in exclude:
                fact.matter = matter
                tagged.append(slot)
        return tagged

    # --- parallel matters ----------------------------------------------
    def note_matter(self, procedure_id: str, issue: str, turn: int) -> bool:
        """Hold a second legal matter without dropping the first."""
        if not procedure_id:
            return False
        if any(item.get("procedure_id") == procedure_id and item.get("status") == "noted" for item in self.matters):
            return False
        self.matters.append({
            "procedure_id": procedure_id,
            "issue": " ".join(str(issue or "").split())[:200],
            "turn": turn,
            "status": "noted",
        })
        self.matters = self.matters[-4:]
        return True

    def pending_matter(self) -> dict[str, Any] | None:
        return next((item for item in self.matters if item.get("status") == "noted"), None)

    def resolve_matter(self, procedure_id: str) -> None:
        for item in self.matters:
            if item.get("procedure_id") == procedure_id:
                item["status"] = "taken_up"


def _find(patterns: tuple[str, ...], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(0).strip()
    return None


def extract(text: str, turn: int = 0) -> tuple[Ledger, list[str]]:
    """The lexical floor: what a pattern can read without a model.

    This is support, not comprehension. Every fact it writes is marked
    `source="net"` and `status="unresolved"`, which is the ledger's way of
    saying "a pattern matched, nobody has read this yet". A grounded model
    reading of the same caller message replaces it -- see `apply_reading` -- so
    a natural-language statement the patterns cannot parse is never blocked by
    one they can half-parse.

    It still runs on every turn, for two reasons that are worth the
    microseconds: with no provider configured it is the only fact extraction
    the app has, and without it a Karnataka question would be retrieved as a
    national one before the model ever speaks.
    """
    ledger = Ledger()
    slots: list[str] = []
    lowered = text.lower()

    for name, code in STATE_NAMES.items():
        match = re.search(r"(?<![a-z])" + re.escape(name) + r"(?![a-z])", lowered)
        if match:
            ledger.record(
                "jurisdiction", code, "net", turn, "jurisdiction",
                evidence=match.group(0), original_expression=match.group(0),
                status="reported", confidence="medium", precision="exact",
            )
            slots.append("jurisdiction")
            break

    when = _find(DATE_PATTERNS, text)
    if when:
        # A relative expression stays relative: "two months ago" is not a date
        # and converting it would invent one.
        relative = bool(re.fullmatch(r"(yesterday|last week|last month|last year|today|aaj|kal)", when, re.I))
        ledger.record(
            "when", when, "net", turn, "date",
            evidence=when, original_expression=when, status="reported",
            confidence="medium", precision="relative" if relative else "exact",
        )
        slots.append("when")

    amount = AMOUNT_RE.search(text) or AMOUNT_WORDS_RE.search(text)
    if amount:
        ledger.record(
            "amount", amount.group(0), "net", turn, "amount",
            evidence=amount.group(0), original_expression=amount.group(0),
            status="reported", confidence="medium",
            precision="approximate" if re.search(r"\b(about|around|approx|roughly|lagbhag|karib)\b", lowered) else "exact",
        )
        slots.append("amount")

    received = (
        re.search(r"\b(yes|haan|ha|i have|have)\b[^.]{0,40}\b(notice|summons|order|written refusal|refusal|receipt|diary|gd|acknowledg|copy)\b", lowered)
        or re.search(r"\b(notice|summons|order|written refusal|receipt|diary number|gd (?:number|entry)|complaint copy)\b[^.]{0,25}\b(mila|aaya|received|have|hai|got|exists)\b", lowered)
        or re.search(r"\b(have|got|holds?)\b[^.]{0,20}\b(written refusal|receipt|diary number|gd number|acknowledgment|copy)\b", lowered)
    )
    if received:
        ledger.record(
            "notice_or_order", "received", "net", turn,
            evidence=received.group(0), original_expression=received.group(0),
            status="reported", confidence="medium",
        )
        slots.append("notice_or_order")
    missing_notice = (
        re.search(r"\b(no|without|nahi|not|didn'?t|never)\b[^.]{0,25}\b(notice|summons|order|written|receipt|diary|gd)\b", lowered)
        or re.search(r"\b(?:only|just)\s+verbally\b|\bverbally\s+refused\b|\brudely sent (?:me )?out\b|\bsent (?:me )?out\b|\brefused (?:to give|verbally)\b", lowered)
    )
    if missing_notice:
        ledger.record(
            "notice_or_order", "not received", "net", turn,
            evidence=missing_notice.group(0), original_expression=missing_notice.group(0),
            status="reported", confidence="medium",
        )
        slots.append("notice_or_order")

    stage_hits: list[str] = []
    stage_evidence: list[str] = []
    for key, (pattern, _label) in BOOLEAN_PATTERNS.items():
        match = re.search(pattern, lowered)
        if match:
            stage_hits.append(key)
            stage_evidence.append(match.group(0))
    if stage_hits:
        ledger.record(
            "stage", ", ".join(stage_hits), "net", turn,
            evidence=stage_evidence[0], original_expression="; ".join(stage_evidence)[:400],
            status="reported", confidence="medium",
        )
        slots.append("stage")

    if INJECTION_RE.search(text):
        ledger.injections += 1

    return ledger, slots


def merge(previous: Ledger, fragment: Ledger, turn: int) -> Ledger:
    """Merge an extraction fragment into a stored ledger, keeping history."""
    previous.injections += fragment.injections
    for slot, fact in fragment.facts.items():
        previous.record(
            slot, fact.value, fact.source, fact.turn or turn,
            "text" if slot not in {"when", "amount", "jurisdiction"} else slot,
            normalized=fact.normalized, evidence=fact.evidence,
            status=fact.status, confidence=fact.confidence,
            original_expression=fact.original_expression, precision=fact.precision,
        )
    return previous


def apply_plan(ledger: Ledger, plan: dict[str, Any] | None, jurisdiction: str, turn: int) -> Ledger:
    """Facts the planner reports are recorded as model-derived, not caller-stated."""
    if jurisdiction and jurisdiction != "IN":
        ledger.record("jurisdiction", jurisdiction, "ui", turn, "jurisdiction")
    if not isinstance(plan, dict):
        return ledger
    issue = plan.get("issue")
    if isinstance(issue, str) and issue.strip() and issue.strip().lower() not in {"unclear", "unknown"}:
        ledger.record("situation", issue.strip(), "plan", turn)
    return ledger


def apply_reading(ledger: Ledger, claims: list[dict[str, Any]], turn: int, matter: str = "") -> list[str]:
    """Record facts the consultation turn grounded in the caller's own words.

    Three rules keep this honest. Each claim arrives already checked for a
    verbatim evidence span by `understanding.validate`, so no model guess can
    enter here. A typed or interface-stated value is never overwritten by a
    model reading of the same matter. And an explicit `corrected` claim *is*
    allowed to replace an earlier value — that is the caller changing their
    account, and refusing it would keep a figure they just disowned.
    """
    written: list[str] = []
    allowed = {item for slots in ORDER.values() for item in slots}
    for claim in claims or []:
        if not isinstance(claim, dict):
            continue
        slot = str(claim.get("slot", "")).strip()
        value = str(claim.get("value", "")).strip()
        status = canonical_status(claim.get("status", "reported"))
        if slot not in allowed or not value:
            continue
        existing = ledger.facts.get(slot)
        same_matter = not matter or not existing or not existing.matter or existing.matter == matter
        # A value the caller typed or picked in the interface is theirs, not
        # the model's to overwrite, unless they are correcting it. A value a
        # *pattern* wrote has no such protection: that is the whole point of
        # marking it `net`. This is the line that makes the model, not the
        # regex, the thing that understands the caller.
        if (
            existing and existing.source in {"caller", "ui"}
            and same_matter and status != "corrected"
        ):
            continue
        kind = slot if slot in {"when", "amount", "jurisdiction"} else "text"
        ledger.record(
            slot, value, "reader", turn, kind, matter,
            normalized=str(claim.get("normalized", "") or ""),
            evidence=str(claim.get("evidence", "") or ""),
            status=status, confidence=str(claim.get("confidence", "high") or "high"),
            original_expression=str(claim.get("original_expression", "") or ""),
            precision=str(claim.get("precision", "") or "reported"),
            change_reason=str(claim.get("change_reason", "") or ""),
        )
        written.append(slot)
    return written


def next_missing(ledger: Ledger, route: str) -> tuple[str | None, str | None]:
    """Return (slot, question) for the most material unasked gap."""
    for slot in ORDER.get(route, ORDER["unclear"]):
        if ledger.known(slot) or slot in ledger.asked:
            continue
        question = QUESTION.get(slot)
        if question:
            return slot, question
    return None, None


def conflict_note(ledger: Ledger) -> str | None:
    if not ledger.contradictions:
        return None
    latest = ledger.contradictions[-1]
    slot = str(latest.get("slot", "detail"))
    label = slot.replace("_", " ")
    fact = ledger.facts.get(slot)
    reason = (fact.change_reason if fact else "") or ""
    tail = f" ({reason})" if reason and reason != "caller correction" else ""
    return (
        f"You mentioned two different values for {label} "
        f"({latest.get('earlier')} and {latest.get('later')}); I am using the most recent one{tail}."
    )
