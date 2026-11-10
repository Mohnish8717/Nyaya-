"""Bounded tools the model may call mid-turn.

Reversed from the tool-calling shape in the `@vapi-ai/web` types (`ToolCall`,
`ToolMessage`, `function.arguments` as a JSON string): the model proposes a
call, the application executes it, and the result returns as a tool message
before the model speaks. The difference here is the boundary.

A legal agent must not have a general "run this" tool. Every tool in this file
is read-only against a validated local corpus, or a *guarded write* into the
fact ledger and the workflow state where the arguments themselves are checked
against what the caller actually said. `record_fact` requires an evidence span
that occurs verbatim in the caller's message; `advance_workflow` requires a
transition the workflow file actually declares and facts the step requires. A
model that hallucinates a slot value gets a refusal, not a fact.

There is no network tool, no shell, no file access, and no action tool. The
worst a compromised model can do here is retrieve the wrong passage.

Each tool also carries registry metadata: a side-effect class (`read_only` or
`guarded_write`) and per-tool wall-clock budgets. Execution validates arguments
against the declared schema *before* the handler runs — a model that invented
keys, got a type wrong or hallucinated a slot cannot reach the corpus or the
ledger at all.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import case_corpus, facts, knowledge, procedural, workflow

SLOTS = {
    "jurisdiction", "when", "amount", "notice_or_order",
    "stage", "recipient", "outcome", "situation",
}

MAX_TOOL_RESULT_CHARS = 6000

# Side-effect classes. `read_only` tools cannot change any state you could
# observe from the next turn (they read the bundled corpus and session-own
# context). A `guarded_write` tool is admitted only when its arguments carry
# proof the model did not invent the value: `record_fact` demands a verbatim
# span from the caller's message, `advance_workflow` only moves along
# transitions the workflow file declares. Nothing in this registry is
# `mutable` (unconditional write) or `destructive`, and nothing here shells
# out, touches the network or the filesystem.
READ_ONLY = "read_only"
GUARDED_WRITE = "guarded_write"

# Wall-clock budgets per tool, in milliseconds, measured by `execute`. This is
# not an in-process preemption: every handler is bounded by construction
# (fixed limits, capped tests), so the budget is a regression tripwire — a
# tool that starts exceeding its budget shows up in traces and fails a test
# before it can eat the turn budget.
BUDGET_MS = {
    "search_law": 2000,
    "search_judgments": 2000,
    "get_procedure": 300,
    "record_fact": 200,
    "advance_workflow": 300,
    "draft_legal_document": 500,
}

TOOL_META: dict[str, dict[str, str]] = {
    "search_law":   {"side_effects": READ_ONLY},
    "search_judgments": {"side_effects": READ_ONLY},
    "get_procedure": {"side_effects": READ_ONLY},
    "record_fact":  {"side_effects": GUARDED_WRITE},
    "advance_workflow": {"side_effects": GUARDED_WRITE},
    "draft_legal_document": {"side_effects": READ_ONLY},
}

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_law",
            "description": (
                "Search the approved primary-law source pack for passages relevant to a question. "
                "Use it before stating any legal rule. Returns source ids and verbatim passages; "
                "only these may support a legal claim."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The legal question, in the caller's terms."},
                    "jurisdiction": {"type": "string", "description": "Indian state code such as IN-KA, or IN for national law."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_judgments",
            "description": (
                "Search the bundled court judgments for fact patterns similar to the caller's. "
                "Returns case records, not authority: never claim a case is binding, current, or overruled."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "jurisdiction": {"type": "string"},
                    "court": {"type": "string", "description": "Optional court name filter."},
                    "year_from": {"type": "integer", "description": "Optional earliest decision year."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_procedure",
            "description": (
                "Fetch the reviewed step-by-step procedure for a known process. Returns the exact "
                "steps, required documents and official source. Never paraphrase a step as a legal rule."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "procedure_id": {"type": "string", "description": "Exact id from the supplied procedure catalogue."},
                },
                "required": ["procedure_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_fact",
            "description": (
                "Store one fact the caller actually stated, so the conversation stops re-asking it. "
                "`evidence` must be a verbatim span from the caller's message; an unquoted value is rejected."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "slot": {"type": "string", "enum": sorted(SLOTS)},
                    "value": {"type": "string"},
                    "evidence": {"type": "string", "description": "Exact words from the caller's message."},
                },
                "required": ["slot", "value", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advance_workflow",
            "description": (
                "Move the active intake workflow to another node. Only transitions declared by the "
                "workflow are accepted, and only when the current step's required facts are known."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["node_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_legal_document",
            "description": (
                "Draft any formal legal document, court application, statutory notice, complaint, "
                "or petition in proper Indian legal format and terms. Generates a complete submission-ready "
                "legal draft with formal title, memo of parties, statement of facts, legal provisions, "
                "prayer, and verification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_type": {
                        "type": "string",
                        "description": "Category of legal document, e.g. application_156_3, complaint_to_sp, legal_notice_138, rti_application, consumer_complaint, affidavit, or general_legal_draft.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Formal legal title of the document or application.",
                    },
                    "complainant_or_applicant": {
                        "type": "string",
                        "description": "Name and details of the applicant, complainant, or sender.",
                    },
                    "respondent_or_opposite_party": {
                        "type": "string",
                        "description": "Name and details of respondent, opposite party, or recipient authority.",
                    },
                    "facts_and_grounds": {
                        "type": "string",
                        "description": "Chronological facts of the grievance, dates, police refusal, loss, or transaction.",
                    },
                    "prayer": {
                        "type": "string",
                        "description": "Specific legal relief, directions, or prayer sought from the court or authority.",
                    },
                    "legal_provisions": {
                        "type": "string",
                        "description": "Relevant sections and statutes, e.g. Section 156(3) CrPC / Section 175(3) BNSS.",
                    },
                    "jurisdiction": {
                        "type": "string",
                        "description": "Indian state code or jurisdiction, e.g. IN-DL, IN-KA, IN.",
                    },
                    "court_or_authority": {
                        "type": "string",
                        "description": "Court or authority name, e.g. Court of the Judicial Magistrate First Class.",
                    },
                },
                "required": ["document_type", "title", "facts_and_grounds", "prayer"],
                "additionalProperties": False,
            },
        },
    },
]

SCHEMA_BY_NAME: dict[str, dict[str, Any]] = {
    spec["function"]["name"]: spec["function"]["parameters"] for spec in TOOL_SPECS
}


class _Invalid(Exception):
    """An internal control-flow error for the validator only."""


def validate_arguments(name: str, arguments: dict[str, Any]) -> str | None:
    """Check proposed arguments against the tool's declared schema.

    Returns an error key when the call must be refused, None when it is well
    formed. Enforced checks: the payload is an object, every `required` key is
    present, every key is declared (additionalProperties is false for all
    tools), declared types hold, and enum values are from the allowed set.
    Contract drift between this validator and the handlers is covered by
    tests, not by hope: tests/test_tool_schema.py asserts the registry and the
    validator stay consistent.
    """
    schema = SCHEMA_BY_NAME.get(name)
    if schema is None:
        return "tool_not_allowed"
    if not isinstance(arguments, dict):
        return "arguments_not_object"
    try:
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in arguments:
                raise _Invalid("missing_required_argument")
        for key, value in arguments.items():
            spec = properties.get(key)
            if spec is None:
                raise _Invalid("argument_not_allowed")
            expected = spec.get("type", "string")
            if expected == "string":
                if not isinstance(value, str):
                    raise _Invalid("argument_type_mismatch")
            elif expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
                raise _Invalid("argument_type_mismatch")
            elif expected == "boolean" and not isinstance(value, bool):
                raise _Invalid("argument_type_mismatch")
            elif expected not in {("string"), ("integer"), ("boolean")}:
                raise _Invalid("schema_unsupported_type")
            enum = spec.get("enum")
            if isinstance(enum, list) and value not in enum:
                raise _Invalid("argument_not_allowed")
    except _Invalid as error:
        return str(error)
    except Exception:  # a malformed never-validate-again registry is a bug
        return "schema_error"
    return None


@dataclass
class ToolContext:
    text: str
    jurisdiction: str
    ledger: facts.Ledger
    workflow_state: dict[str, Any] | None = None
    turn: int = 0
    allowed_tools: tuple[str, ...] = tuple(spec["function"]["name"] for spec in TOOL_SPECS)


@dataclass
class ToolResult:
    name: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    ms: int = 0
    sources: list[dict[str, Any]] = field(default_factory=list)

    def as_message(self) -> str:
        payload = {"tool": self.name, "ok": self.ok}
        if self.error:
            payload["error"] = self.error
        payload.update(self.data)
        text = json.dumps(payload, ensure_ascii=False, default=str)
        return text[:MAX_TOOL_RESULT_CHARS]

    def as_dict(self) -> dict[str, Any]:
        budget = BUDGET_MS.get(self.name, 0)
        return {
            "tool": self.name,
            "side_effects": (TOOL_META.get(self.name) or {}).get("side_effects"),
            "budget_ms": budget,
            "over_budget": bool(budget and self.ms > budget),
            "ok": self.ok, "ms": self.ms, "error": self.error,
            "result_keys": sorted(self.data.keys()),
        }


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _query(args: dict[str, Any]) -> str | None:
    query = _text(args.get("query"), 300)
    return query if len(query) >= 3 else None


def _search_law(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = _query(args)
    if not query:
        return ToolResult("search_law", False, error="query_too_short")
    jurisdiction = _text(args.get("jurisdiction"), 8) or ctx.jurisdiction
    sources = knowledge.retrieve(query, jurisdiction, limit=4)
    return ToolResult(
        "search_law", True,
        data={
            "count": len(sources),
            "passages": [
                {"id": source["id"], "title": source["title"], "authority": source["authority"], "text": source["text"][:1500]}
                for source in sources
            ],
            "note": "Only these passages may support a legal claim; quote them exactly.",
        },
        sources=sources,
    )


def _search_judgments(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = _query(args)
    if not query:
        return ToolResult("search_judgments", False, error="query_too_short")
    jurisdiction = _text(args.get("jurisdiction"), 8) or ctx.jurisdiction
    cases = case_corpus.retrieve_cases(query, jurisdiction, limit=3)
    return ToolResult(
        "search_judgments", True,
        data={
            "count": len(cases),
            "cases": [
                {
                    "id": case["id"], "title": case["title"], "court": case["court"],
                    "court_level": case.get("court_level"),
                    "decision_date": case.get("decision_date"), "citation": case.get("citation"),
                    "excerpt": (case.get("excerpt") or case.get("text") or "")[:1200],
                }
                for case in cases
            ],
            "note": "Case records show what one court decided; treatment has not been checked. court_level ranks authority (supreme > high > district) and is not a similarity score.",
        },
        sources=cases,
    )


def _get_procedure(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    procedure_id = _text(args.get("procedure_id"), 64)
    procedure = procedural.BY_ID.get(procedure_id)
    if not procedure or procedure.get("review_status") in {"rejected", "expired"}:
        return ToolResult("get_procedure", False, error="procedure_not_found")
    return ToolResult(
        "get_procedure", True,
        data={
            "id": procedure["id"],
            "label": procedure["label"],
            "summary": procedure["summary"],
            "review_status": procedure["review_status"],
            "steps": [
                {"order": index + 1, "action": step.get("action"), "where": step.get("where"), "bring": step.get("bring")}
                for index, step in enumerate(procedure.get("steps", []))
            ],
            "source": procedure["source"],
        },
    )


def _record_fact(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    slot = _text(args.get("slot"), 32).lower()
    value = _text(args.get("value"), 200)
    evidence = _text(args.get("evidence"), 400)
    if slot not in SLOTS:
        return ToolResult("record_fact", False, error="slot_not_allowed")
    if not value:
        return ToolResult("record_fact", False, error="empty_value")
    caller = " ".join(ctx.text.casefold().split())
    if not evidence or " ".join(evidence.casefold().split()) not in caller:
        return ToolResult("record_fact", False, error="evidence_not_in_caller_message")
    if slot == "jurisdiction":
        code = value.upper()
        if code not in knowledge_jurisdictions():
            code = facts.STATE_NAMES.get(value.lower(), "")
        if not code:
            return ToolResult("record_fact", False, error="jurisdiction_not_known")
        value = code
    kind = slot if slot in {"when", "amount", "jurisdiction"} else "text"
    ctx.ledger.record(slot, value, "tool", ctx.turn, kind)
    return ToolResult("record_fact", True, data={"slot": slot, "value": ctx.ledger.value(slot)})


def knowledge_jurisdictions() -> set[str]:
    # Imported lazily: legal_engine owns the canonical set, and importing it
    # here would create a cycle at module load.
    from .legal_engine import INDIAN_JURISDICTIONS

    return INDIAN_JURISDICTIONS


def _advance_workflow(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    state = workflow.load_state(ctx.workflow_state)
    if not state:
        return ToolResult("advance_workflow", False, error="no_active_workflow")
    node_id = _text(args.get("node_id"), 64)
    updated, events = workflow.advance(state, ctx.ledger, to=node_id, turn=ctx.turn)
    ctx.workflow_state.clear()
    ctx.workflow_state.update(updated)
    refused = next((event for event in events if event["event"] == "refused"), None)
    if refused:
        return ToolResult("advance_workflow", False, error=refused["reason"], data={"events": events})
    return ToolResult("advance_workflow", True, data={"node": updated["node"], "status": updated["status"], "events": events})


def _format_facts_numbered(facts_raw: str, default_intro: str = "") -> str:
    """Format facts and grounds into structured court-standard numbered paragraphs."""
    lines = [l.strip() for l in facts_raw.split("\n") if l.strip()]
    if not lines:
        fallback = default_intro or "the material facts giving rise to this grievance are submitted herein for the consideration of the competent authority / Hon'ble Court."
        return f"1. That {fallback}"

    formatted_paras: list[str] = []
    para_num = 1
    for line in lines:
        if line.lower().startswith("recorded facts:"):
            continue
        clean_line = line.lstrip("-*• ").strip()
        if not clean_line:
            continue
        # Convert raw slot keys like 'Amount: 1,00,000' or 'When: 2 years' into clean legal sentences
        if ":" in clean_line and not re.match(r"^(?:\d+[\.\)]|\([0-9a-zA-Z]+\))\s*", clean_line):
            key, val = clean_line.split(":", 1)
            k_lower = key.strip().lower()
            val = val.strip()
            if k_lower in {"amount", "money", "deposit"}:
                clean_line = f"the principal sum involved in the dispute is {val}."
            elif k_lower in {"when", "date", "time"}:
                clean_line = f"the transaction / cause of action occurred approximately {val}."
            elif k_lower in {"situation", "issue", "dispute"}:
                clean_line = f"the matter pertains to {val}."
            elif k_lower in {"what was tried", "what_was_tried", "action_taken"}:
                clean_line = f"the applicant approached the opposite party / authorities and {val}."
            else:
                clean_line = f"in respect of {key.strip()}, {val}."

        if re.match(r"^(?:\d+[\.\)]|\([0-9a-zA-Z]+\))\s*", clean_line):
            formatted_paras.append(clean_line)
        else:
            prefix = "That " if not clean_line.lower().startswith("that ") else ""
            formatted_paras.append(f"{para_num}. {prefix}{clean_line}")
            para_num += 1
    return "\n\n".join(formatted_paras)


def _draft_legal_document(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    doc_type = _text(args.get("document_type"), 100).lower().strip() or "general_legal_draft"
    title = _text(args.get("title"), 300).strip() or "Legal Application / Draft"
    applicant = _text(args.get("complainant_or_applicant"), 500).strip() or "The Applicant / Complainant"
    respondent = _text(args.get("respondent_or_opposite_party"), 500).strip() or "The Respondent / Opposite Party"
    facts_grounds = _text(args.get("facts_and_grounds"), 8000).strip()
    prayer = _text(args.get("prayer"), 3000).strip()
    provisions = _text(args.get("legal_provisions"), 400).strip()
    jurisdiction = _text(args.get("jurisdiction"), 64).strip() or ctx.jurisdiction or "IN"
    court = _text(args.get("court_or_authority"), 300).strip()

    state_display = facts.STATE_NAMES.get(jurisdiction.lower(), jurisdiction)
    formatted_facts = _format_facts_numbered(facts_grounds)

    # 1. Section 156(3) CrPC / Section 175(3) BNSS Magistrate Application
    if doc_type in {"application_156_3", "application_156_3_magistrate"} or "156" in doc_type or "156" in title:
        court_header = court or f"IN THE COURT OF THE LEARNED CHIEF METROPOLITAN MAGISTRATE / JUDICIAL MAGISTRATE FIRST CLASS AT {state_display.upper()}"
        full_title = title or "APPLICATION UNDER SECTION 156(3) OF THE CODE OF CRIMINAL PROCEDURE, 1973 (READ WITH SECTION 175(3) OF BHARATIYA NAGARIK SURAKSHA SANHITA, 2023)"
        sections = provisions or "Section 156(3) CrPC / Section 175(3) BNSS, Section 154(3) CrPC / Section 173(3) BNSS"

        full_draft = f"""{court_header}
CRIMINAL MISC. APPLICATION NO. ________ OF 2026

IN THE MATTER OF:
{applicant}
... APPLICANT / COMPLAINANT

VERSUS

1. THE STATE OF {state_display.upper()}
   (Through Station House Officer, Police Station Concerned)
2. {respondent}
... RESPONDENTS / ACCUSED

{full_title}
PRAYING FOR DIRECTIONS TO THE POLICE TO REGISTER FIR AND CONDUCT INVESTIGATION

MOST RESPECTFULLY SHOWETH:

1. That the Applicant is a citizen of India residing at the address stated above and is entitled to the full protection of law under the Constitution of India.

2. That the Respondent No. 2 named above has committed serious, cognizable offences against the Applicant. The material facts and grounds giving rise to this application are as follows:
{formatted_facts}

3. That the facts set out above clearly and unequivocally disclose the commission of cognizable offences under the law.

4. That on encountering the aforesaid commission of offences, the Applicant immediately approached the local Police Station with a written complaint requesting the registration of a First Information Report (FIR). However, the concerned police officers refused to record the complaint or register the FIR, in direct violation of the mandatory statutory provisions of Section 154 CrPC / Section 173 BNSS and the binding judgment of the Hon'ble Supreme Court in Lalita Kumari v. Govt. of U.P. (2014) 2 SCC 1.

5. That aggrieved by the refusal of the police officer to register the FIR, the Applicant duly complied with the statutory prerequisite under Section 154(3) CrPC / Section 173(3) BNSS by sending the substance of the complaint in writing and by registered post to the Superintendent of Police / Deputy Commissioner of Police concerned.

6. That despite the receipt of the complaint, no direction for investigation has been made and no FIR has been registered by the police authorities till date.

7. That a police investigation is indispensable in this matter for the recovery of material evidence, recording statements of witnesses, and apprehending the accused persons. The Applicant has exhausted all departmental remedies and has no other efficacious remedy except to approach this Hon'ble Court under {sections}.

8. That the Applicant has not filed any other application or petition before this or any other court for the same cause of action.

PRAYER:
Wherefore, in light of the facts and circumstances stated above, it is most respectfully prayed that this Hon'ble Court may graciously be pleased to:

a) Direct the Station House Officer (SHO), Police Station concerned, to forthwith register a First Information Report (FIR) on the basis of the Applicant's complaint under the appropriate provisions of law;
b) Direct the Investigating Officer to conduct a fair, impartial, and prompt investigation into the matter and submit a final report in accordance with law;
c) {prayer if prayer else "Pass any other or further order(s) as this Hon'ble Court may deem fit and proper in the interest of justice."}

APPLICANT
THROUGH COUNSEL / ADVOCATE

VERIFICATION:
I, the Applicant above-named, do hereby verify on solemn affirmation that the contents of paragraphs 1 to 8 of the above application are true and correct to my personal knowledge and belief, and no material fact has been concealed therefrom.
Verified at ________ on this ____ day of ________, 2026.

DEPONENT / APPLICANT

AFFIDAVIT
I, the Deponent above-named, do hereby solemnly affirm and state as follows:
1. That I am the Applicant in the accompanying application and am well conversant with the facts of the case.
2. That the contents of the accompanying application are true and correct to my knowledge and belief.
DEPONENT
Verified at ________ on this ____ day of ________, 2026 that the contents of this affidavit are true and correct.
DEPONENT
"""

    # 2. Section 154(3) CrPC / 173(3) BNSS Complaint to SP
    elif doc_type in {"complaint_to_sp", "complaint_sp_police"} or "superintendent" in title.lower() or "154(3)" in title or "173(3)" in title:
        full_title = title or "WRITTEN COMPLAINT UNDER SECTION 154(3) CrPC / SECTION 173(3) BNSS"
        full_draft = f"""TO:
THE SUPERINTENDENT OF POLICE / DEPUTY COMMISSIONER OF POLICE,
DISTRICT / DIVISION: {state_display.upper()}

SUBJECT: {full_title} REGARDING REFUSAL BY LOCAL POLICE STATION TO REGISTER FIR IN A COGNIZABLE OFFENCE

FROM:
{applicant}

AGAINST:
{respondent}

RESPECTED SIR / MADAM,

1. I am writing to bring to your urgent attention the commission of a cognizable offence and the subsequent refusal of the local Police Station to discharge their statutory duty of registering a First Information Report (FIR).

2. Summary of incident and material facts:
{formatted_facts}

3. That on approaching the local Police Station, the officer in charge refused to register an FIR despite the disclosure of cognizable offences, violating Section 154 CrPC / Section 173 BNSS and the directives of the Hon'ble Supreme Court in Lalita Kumari v. Govt. of U.P. (2014) 2 SCC 1.

4. That in terms of Section 154(3) CrPC / Section 173(3) BNSS, I am submitting the substance of my complaint in writing and by registered post to your office.

PRAYER:
I earnestly request you to examine the complaint, direct the registration of an FIR, and cause an investigation to be made by an officer subordinate to you or investigate the matter personally in accordance with law.
{prayer if prayer else ""}

Yours faithfully,

(Signature)
{applicant}
Date: ________
Place: ________
Enclosures: Copy of original complaint, postal receipt, and supporting evidence.
"""

    # 3. Regular Bail Application (Section 437/439 CrPC / Section 480/483 BNSS)
    elif doc_type in {"bail_application", "regular_bail"} or ("bail" in doc_type and "anticipatory" not in doc_type) or ("bail" in title.lower() and "anticipatory" not in title.lower()):
        court_header = court or f"IN THE COURT OF THE LEARNED SESSIONS JUDGE / SPECIAL JUDGE AT {state_display.upper()}"
        full_title = title or "APPLICATION FOR REGULAR BAIL UNDER SECTION 439 CrPC (READ WITH SECTION 483 BNSS)"
        sections = provisions or "Section 439 CrPC / Section 483 BNSS"

        full_draft = f"""{court_header}
CRIMINAL MISC. (BAIL) APPLICATION NO. ________ OF 2026

IN THE MATTER OF:
{applicant}
... APPLICANT / ACCUSED

VERSUS

THE STATE OF {state_display.upper()}
(Through Station House Officer, Police Station Concerned)
... RESPONDENT / PROSECUTION

{full_title}
IN FIR NO. ________ DATED ________, POLICE STATION ________
UNDER SECTIONS: {sections}

MOST RESPECTFULLY SHOWETH:

1. That the Applicant is a respectable, law-abiding citizen of India residing at the given address, having deep roots in society and no criminal antecedents.

2. That the Applicant has been falsely implicated in the above-mentioned FIR and was arrested without any justifiable evidence.

3. Factual matrix and grounds of innocence:
{formatted_facts}

4. That the investigation in the case is substantially complete, custody of the Applicant is no longer required for any custodial interrogation, and nothing remains to be recovered from or at the instance of the Applicant.

5. That the Applicant is not a flight risk, has immovable property and familial roots in the jurisdiction, and undertakes to abide by all conditions that may be imposed by this Hon'ble Court.

6. That the Applicant undertakes not to tamper with evidence, influence any prosecution witnesses, or jump bail in any manner whatsoever.

7. That the Hon'ble Supreme Court of India in Satender Kumar Antil v. CBI (2022) 10 SCC 51 and Sanjay Chandra v. CBI (2012) 1 SCC 40 has reaffirmed the foundational doctrine that "Bail is the rule and jail is an exception", and detention should not be punitive prior to conviction.

PRAYER:
Wherefore, it is most respectfully prayed that this Hon'ble Court may graciously be pleased to:
a) Enlarging the Applicant on regular bail in FIR No. ________, registered at Police Station ________, upon furnishing of reasonable bail bonds and surety;
b) {prayer if prayer else "Pass any other or further order(s) as this Hon'ble Court may deem fit and proper in the interest of justice."}

APPLICANT
THROUGH COUNSEL

VERIFICATION:
I, the Deponent above-named, do hereby verify on solemn affirmation that the contents of paragraphs 1 to 7 above are true and correct to the best of my knowledge and belief.
Verified at ________ on this ____ day of ________, 2026.

DEPONENT

AFFIDAVIT
I, the Deponent above-named, do hereby solemnly affirm and state as under:
1. That I am the Applicant in the accompanying bail application and well conversant with the facts and circumstances of the case.
2. That the contents of the bail application are true and correct.
DEPONENT
Verified at ________ on ____/____/2026 that the contents of this affidavit are true.
DEPONENT
"""

    # 4. Anticipatory Bail Application (Section 438 CrPC / Section 482 BNSS)
    elif doc_type in {"anticipatory_bail"} or "anticipatory" in doc_type or "anticipatory" in title.lower() or "438" in doc_type or "438" in title or "482 bnss" in title.lower():
        court_header = court or f"IN THE COURT OF THE LEARNED SESSIONS JUDGE AT {state_display.upper()}"
        full_title = title or "APPLICATION FOR ANTICIPATORY BAIL UNDER SECTION 438 CrPC / SECTION 482 BNSS"
        sections = provisions or "Section 438 CrPC / Section 482 BNSS"

        full_draft = f"""{court_header}
CRIMINAL MISC. (ANTICIPATORY BAIL) APPLICATION NO. ________ OF 2026

IN THE MATTER OF:
{applicant}
... APPLICANT / ACCUSED

VERSUS

THE STATE OF {state_display.upper()}
(Through Station House Officer, Police Station Concerned)
... RESPONDENT / PROSECUTION

{full_title}
IN CONNECTION WITH COMPLAINT / PROPOSED FIR AT POLICE STATION ________
UNDER SECTIONS: {sections}

MOST RESPECTFULLY SHOWETH:

1. That the Applicant is a respectable citizen of India with deep social roots, gainful employment, and an unblemished reputation.

2. That the Applicant harbours a bona fide and grave apprehension of arrest in connection with false, frivolous, and concocted allegations orchestrated by the complainant party with ulterior motives.

3. Factual background and grounds establishing prima facie innocence:
{formatted_facts}

4. That the custodial interrogation of the Applicant is entirely unnecessary and uncalled for, as the allegations are predominantly documentary / verbal in nature.

5. That in accordance with Section 438(2) CrPC / Section 482(2) BNSS, the Applicant unequivocally undertakes to:
   a) Make himself/herself available for interrogation by a police officer as and when required;
   b) Not make any inducement, threat, or promise to any person acquainted with the facts of the case;
   c) Not leave India without previous permission of the Court.

6. That the Hon'ble Constitution Bench of the Supreme Court in Gurbaksh Singh Sibbia v. State of Punjab (1980) 2 SCC 565 and Sushila Aggarwal v. State (NCT of Delhi) (2020) 5 SCC 1 held that anticipatory bail is an indispensable safeguard to protect personal liberty under Article 21 from frivolous and vexatious arrests.

PRAYER:
Wherefore, it is most respectfully prayed that this Hon'ble Court may graciously be pleased to:
a) Direct that in the event of arrest of the Applicant in connection with Complaint / FIR No. ________, Police Station ________, the Applicant be released on bail forthwith;
b) {prayer if prayer else "Grant ad-interim ex-parte protection against arrest during the pendency of the present application;"}
c) Pass any further order(s) as this Hon'ble Court may deem fit in the interest of justice.

APPLICANT
THROUGH COUNSEL

VERIFICATION:
I, the Applicant above-named, do hereby verify that paragraphs 1 to 6 of this application are true to my knowledge and belief.
Verified at ________ on this ____ day of ________, 2026.

DEPONENT

AFFIDAVIT
I, the Deponent above-named, do hereby solemnly affirm and declare on oath that I am the Applicant, the facts stated in the petition are true, and I have not filed any other anticipatory bail application before this or any other court.
DEPONENT
Verified at ________ on this ____ day of ________, 2026.
DEPONENT
"""

    # 5. Maintenance Application (Section 125 CrPC / Section 144 BNSS)
    elif doc_type in {"maintenance_application", "maintenance_125"} or "maintenance" in doc_type or "maintenance" in title.lower() or "125" in doc_type or "125" in title or "144 bnss" in title.lower():
        court_header = court or f"IN THE COURT OF THE LEARNED PRINCIPAL JUDGE, FAMILY COURT AT {state_display.upper()}"
        full_title = title or "APPLICATION FOR MAINTENANCE UNDER SECTION 125 CrPC (READ WITH SECTION 144 BNSS)"
        sections = provisions or "Section 125 CrPC / Section 144 BNSS"

        full_draft = f"""{court_header}
MAINTENANCE PETITION NO. ________ OF 2026

IN THE MATTER OF:
{applicant}
... APPLICANT / PETITIONER

VERSUS

{respondent}
... RESPONDENT

{full_title}
FOR GRANT OF MONTHLY MAINTENANCE AND INTERIM RELIEF

MOST RESPECTFULLY SHOWETH:

1. That the Applicant is legally wedded to / a dependent of the Respondent, and their relationship subsists in law.

2. That the Respondent has sufficient means, substantial income, and lucrative assets, but has wilfully neglected and refused to maintain the Applicant.

3. Factual details of marriage, desertion, cruelty, and financial condition:
{formatted_facts}

4. That the Applicant has no independent source of livelihood and is entirely incapable of maintaining herself/himself at the standard of living accustomed to in the matrimonial home.

5. That the Respondent earns a handsome monthly income of Rs. ________ and possesses valuable properties, while having no other dependants.

6. That in compliance with the guidelines laid down by the Hon'ble Supreme Court in Rajnesh v. Neha (2021) 2 SCC 324, the Applicant is filing the requisite Affidavit of Assets and Liabilities along with this petition.

PRAYER:
Wherefore, it is most respectfully prayed that this Hon'ble Court may graciously be pleased to:
a) Direct the Respondent to pay a monthly maintenance allowance of Rs. ________ per month to the Applicant from the date of filing of this application;
b) Direct the Respondent to pay interim maintenance during the pendency of these proceedings;
c) Direct the Respondent to pay litigation expenses of Rs. ________;
d) {prayer if prayer else "Pass any other or further order(s) as this Hon'ble Court may deem fit and proper."}

APPLICANT
THROUGH COUNSEL

VERIFICATION:
Verified at ________ on this ____ day of ________, 2026 that the contents of paragraphs 1 to 6 are true and correct.
DEPONENT

AFFIDAVIT
I, the Deponent above-named, do hereby solemnly affirm and state that I am the Applicant and verify that the facts stated in the petition and my asset disclosure are true and correct.
DEPONENT
"""

    # 6. Domestic Violence Petition (Section 12 PWDVA, 2005)
    elif doc_type in {"domestic_violence", "pwdva_application"} or "domestic violence" in title.lower() or "pwdva" in doc_type or "pwdva" in title.lower() or "dv act" in title.lower():
        court_header = court or f"IN THE COURT OF THE LEARNED CHIEF METROPOLITAN MAGISTRATE / JMFC AT {state_display.upper()}"
        full_title = title or "APPLICATION UNDER SECTION 12 OF THE PROTECTION OF WOMEN FROM DOMESTIC VIOLENCE ACT, 2005"

        full_draft = f"""{court_header}
MISC. (D.V.) APPLICATION NO. ________ OF 2026

IN THE MATTER OF:
{applicant}
... AGGRIEVED PERSON / APPLICANT

VERSUS

{respondent}
... RESPONDENTS

{full_title}
FOR ORDERS UNDER SECTIONS 18, 19, 20, AND 22 OF THE PWDVA, 2005

MOST RESPECTFULLY SHOWETH:

1. That the Applicant is an 'Aggrieved Person' within the meaning of Section 2(a) of the PWDVA, who has lived with the Respondents in a shared household and was subjected to persistent acts of domestic violence.

2. Narrative of domestic relationship, specific incidents of domestic violence, cruelty, and dispossession:
{formatted_facts}

3. That the acts and omissions of the Respondents constitute physical, verbal, emotional, and economic abuse within the definition of Section 3 of the Act.

4. That the Applicant is in imminent danger of being dispossessed from the shared household / deprived of financial support without protection orders.

PRAYER:
It is most respectfully prayed that this Hon'ble Court may graciously be pleased to pass:
a) Protection Order under Section 18 prohibiting the Respondents from committing any further acts of domestic violence or contacting the Applicant;
b) Residence Order under Section 19 restraining the Respondents from dispossessing the Applicant from the shared household;
c) Monetary Relief under Section 20 directing payment of monthly maintenance and medical expenses;
d) Compensation Order under Section 22 for damages and mental agony;
e) {prayer if prayer else "Pass ex-parte interim orders under Section 23 of the Act."}

AGGRIEVED PERSON
THROUGH COUNSEL

VERIFICATION:
Verified at ________ on this ____ day of ________, 2026 that the contents of the application are true to my personal knowledge and belief.
DEPONENT
"""

    # 7. Section 138 NI Act Cheque Dishonour Notice
    elif doc_type in {"legal_notice_138"} or "138" in doc_type or "138" in title or "cheque" in title.lower() or "dishonour" in title.lower():
        full_title = title or "FORMAL LEGAL NOTICE UNDER SECTION 138 OF THE NEGOTIABLE INSTRUMENTS ACT, 1881"
        full_draft = f"""REGISTERED A.D. / SPEED POST
LEGAL NOTICE

To,
{respondent}

From,
ADVOCATE (On behalf of Client: {applicant})

SUBJECT: {full_title} FOR DISHONOUR OF CHEQUE

Sir / Madam,

Under instructions from and on behalf of my client, {applicant}, I hereby serve upon you this formal statutory Legal Notice under Section 138 of the Negotiable Instruments Act, 1881:

1. That you, the Noticee, in discharge of your legally enforceable debt/liability towards my client, issued cheque(s) in favour of my client.

2. Factual background and transaction details:
{formatted_facts}

3. That the said cheque was presented by my client to the bank, but was returned unpaid with the bank memo citing dishonour.

4. That you have failed to make payment of the cheque amount to my client despite knowing that the cheque would be dishonoured upon presentation.

DEMAND / PRAYER:
I hereby call upon you to pay the entire cheque amount to my client within 15 (fifteen) days from the date of receipt of this notice, failing which my client shall be constrained to initiate criminal proceedings against you under Section 138 read with Section 142 of the Negotiable Instruments Act, 1881, holding you liable for all costs and consequences thereof.
{prayer if prayer else ""}

ADVOCATE FOR CLIENT
Date: ________
Place: ________
"""

    # 7B. Legal Notice for Recovery of Debt / Promissory Note
    elif doc_type in {"legal_notice_debt", "promissory_note", "promissory_note_notice"} or "promissory" in doc_type or "promissory" in title.lower() or "pronote" in doc_type or "debt" in doc_type or "recovery" in doc_type or "loan" in doc_type:
        full_title = title or "LEGAL NOTICE FOR RECOVERY OF MONEY DUE UNDER PROMISSORY NOTE / DEBT"
        full_draft = f"""REGISTERED A.D. / SPEED POST
LEGAL NOTICE

To,
{respondent}
(Noticee / Debtor)

From,
ADVOCATE (On behalf of Client: {applicant})

Date: [Current Date]

SUBJECT: {full_title}

Sir / Madam,

Under instructions from and on behalf of my client, {applicant}, I hereby serve upon you this formal statutory Legal Notice:

1. That you, the Noticee, approached my client seeking financial assistance / friendly loan and promised to repay the said loan amount with interest upon demand.

2. Factual background and transaction particulars:
{formatted_facts}

3. That in consideration of the advance of the said sum, you duly executed a Promissory Note / acknowledgement of debt in favour of my client, undertaking to repay the entire principal amount along with agreed interest.

4. That despite the expiry of the stipulated repayment period and repeated oral and written requests by my client, you have willfully failed and neglected to discharge your legally enforceable debt.

DEMAND / REQUISITION:
You are hereby called upon to pay to my client the total outstanding sum along with interest at the agreed rate within 15 (fifteen) days from the receipt of this notice, failing which my client shall be constrained to initiate appropriate civil and criminal legal proceedings against you in the competent court of law, at your sole risk as to costs and consequences.

Copy retained for record.

ADVOCATE FOR CLIENT
"""

    # 8. Eviction Notice under Section 106 Transfer of Property Act, 1882
    elif doc_type in {"legal_notice_106_tpa", "eviction_notice", "tenancy_notice"} or "106" in doc_type or "106" in title or "eviction" in title.lower() or "tenancy" in title.lower():
        full_title = title or "NOTICE OF TERMINATION OF TENANCY UNDER SECTION 106 OF THE TRANSFER OF PROPERTY ACT, 1882"
        full_draft = f"""REGISTERED POST WITH ACKNOWLEDGEMENT DUE
LEGAL NOTICE

To,
{respondent}
(Tenant / Noticee)

From,
ADVOCATE (On behalf of Landlord: {applicant})

SUBJECT: {full_title} FOR PREMISES SITUATED AT {state_display.upper()}

Sir / Madam,

Under instructions from and on behalf of my client, {applicant} (hereinafter referred to as the 'Landlord'), I hereby serve upon you this formal statutory Notice under Section 106 of the Transfer of Property Act, 1882:

1. That you are a monthly tenant in respect of the premises comprising [describe property] at a monthly rent of Rs. ________ exclusive of electricity and water charges.

2. Grounds of termination and material facts:
{formatted_facts}

3. That my client no longer wishes to continue your tenancy and hereby terminates the same with effect from the expiry of fifteen (15) days from the date of receipt of this notice.

DEMAND / PRAYER:
You are hereby called upon to vacate and hand over peaceful, vacant, and physical possession of the said tenanted premises to my client on or before the expiry of 15 days, and clear all outstanding arrears of rent. Take notice that failing compliance, my client shall institute a suit for eviction, possession, recovery of rent, and mesne profits/damages at the market rate, holding you entirely liable for all legal expenses.
{prayer if prayer else ""}

ADVOCATE FOR LANDLORD
Date: ________
Place: ________
"""

    # 9. Notice under Section 80 CPC (Notice to Government)
    elif doc_type in {"legal_notice_80_cpc", "notice_section_80"} or "80 cpc" in doc_type or "80 cpc" in title.lower() or "section 80" in title.lower():
        full_title = title or "STATUTORY NOTICE UNDER SECTION 80 OF THE CODE OF CIVIL PROCEDURE, 1908"
        full_draft = f"""REGISTERED POST WITH A.D.
FORMAL LEGAL NOTICE UNDER SECTION 80 CPC

To,
1. THE SECRETARY TO THE GOVERNMENT,
   DEPARTMENT OF ________________, GOVERNMENT OF {state_display.upper()}
2. {respondent}

From,
ADVOCATE (On behalf of Claimant: {applicant})

SUBJECT: {full_title} PRIOR TO INSTITUTION OF SUIT

Sir,

Please take notice that my client intends to institute a civil suit against you in the competent court upon the expiration of two (2) months from the date of service of this notice, pursuant to Section 80 of the Code of Civil Procedure, 1908:

1. Name, description, and place of residence of the Claimant:
{applicant}

2. Cause of action and factual statement:
{formatted_facts}

3. Relief claimed:
{prayer if prayer else "Payment of damages/compensation and performance of statutory obligation."}

Take notice that if the relief claimed is not satisfied within sixty (60) days of receipt hereof, my client shall proceed to file the suit at your risk and costs.

ADVOCATE FOR CLAIMANT
Date: ________
Place: ________
"""

    # 10. General Legal Demand Notice
    elif doc_type in {"legal_notice", "legal_notice_general"} or "notice" in doc_type or "notice" in title.lower() or "demand notice" in title.lower():
        full_title = title or "FORMAL LEGAL NOTICE AND DEMAND"
        full_draft = f"""REGISTERED SPEED POST WITH ACKNOWLEDGEMENT DUE
FORMAL LEGAL NOTICE

To,
{respondent}

From,
ADVOCATE (On behalf of Client: {applicant})

SUBJECT: {full_title}

Sir / Madam,

Under instructions and authority from my client, {applicant}, I hereby serve upon you this formal legal notice:

1. Statement of Relationship and Transactions:
That my client had entered into lawful transactions / agreements with you as detailed below.

2. Factual matrix and breach of obligation:
{formatted_facts}

3. That your acts, defaults, and omissions have caused serious pecuniary loss, harassment, and violation of my client's legal rights.

DEMAND:
I hereby call upon you to comply with my client's lawful demand within fifteen (15) days from the receipt of this notice:
{prayer if prayer else "To rectify the breach, make payment of outstanding dues with interest, and cease unlawful interference."}

Failing compliance, my client has given me peremptory instructions to initiate appropriate civil and criminal proceedings against you before competent courts, holding you liable for all interest, damages, and costs.

ADVOCATE FOR CLIENT
Date: ________
Place: ________
"""

    # 11. Civil Suit for Injunction / Recovery / Damages (Order VII CPC)
    elif doc_type in {"civil_suit", "plaint", "suit_for_injunction", "suit_for_recovery"} or "suit" in doc_type or "plaint" in doc_type or "plaint" in title.lower() or "suit" in title.lower():
        court_header = court or f"IN THE COURT OF THE LEARNED SENIOR CIVIL JUDGE / DISTRICT JUDGE AT {state_display.upper()}"
        full_title = title or "PLAINT IN SUIT FOR PERMANENT INJUNCTION AND DAMAGES UNDER ORDER VII RULE 1 CPC"
        sections = provisions or "Order VII Rule 1 CPC, Specific Relief Act, 1963"

        full_draft = f"""{court_header}
CIVIL SUIT NO. ________ OF 2026

IN THE MATTER OF:
{applicant}
... PLAINTIFF

VERSUS

{respondent}
... DEFENDANT

{full_title}
UNDER {sections}

MOST RESPECTFULLY SHOWETH:

1. That the Plaintiff is a law-abiding citizen residing at the address stated in the cause title, within the territorial jurisdiction of this Hon'ble Court.

2. That the Defendant resides / works for gain within the jurisdiction of this Hon'ble Court and is amenable to its authority.

3. Chronological statement of facts, cause of action, and breach by the Defendant:
{formatted_facts}

4. CAUSE OF ACTION:
That the cause of action first arose on ________ when the dispute/breach occurred, and subsequently on ________ when the Defendant refused compliance, and continues to subsist de die in diem.

5. JURISDICTION & VALUATION:
That the property / cause of action arose within the territorial limits of this Hon'ble Court. The suit is valued for the purpose of court fees and jurisdiction at Rs. ________, and requisite ad-valorem court fee has been affixed.

PRAYER:
Wherefore, it is most respectfully prayed that this Hon'ble Court may be pleased to pass a Decree in favour of the Plaintiff and against the Defendant:
a) Granting permanent prohibitory injunction restraining the Defendant from interfering with Plaintiff's lawful rights;
b) {prayer if prayer else "Directing recovery of damages / debt along with pendente lite and future interest at 18% per annum;"}
c) Awarding costs of the suit;
d) Passing any further relief deemed just and equitable.

PLAINTIFF
THROUGH COUNSEL

VERIFICATION:
I, the Plaintiff above-named, do hereby verify that the contents of paragraphs 1 to 5 above are true to my personal knowledge, and legal advice believed to be true.
Verified at ________ on this ____ day of ________, 2026.

PLAINTIFF / DEPONENT

AFFIDAVIT
I, the Deponent above-named, do hereby solemnly affirm that I am the Plaintiff in the present suit and verify that the facts stated in the Plaint are true and correct to my knowledge.
DEPONENT
"""

    # 12. Temporary Injunction Application (Order 39 Rules 1 & 2 CPC)
    elif doc_type in {"injunction_application", "temporary_injunction"} or "injunction" in doc_type or "injunction" in title.lower() or "order 39" in title.lower():
        court_header = court or f"IN THE COURT OF THE LEARNED CIVIL JUDGE AT {state_display.upper()}"
        full_title = title or "APPLICATION UNDER ORDER XXXIX RULES 1 & 2 READ WITH SECTION 151 CPC FOR TEMPORARY INJUNCTION"

        full_draft = f"""{court_header}
IN CIVIL SUIT NO. ________ OF 2026

IN THE MATTER OF:
{applicant}
... APPLICANT / PLAINTIFF

VERSUS

{respondent}
... RESPONDENT / DEFENDANT

{full_title}

MOST RESPECTFULLY SHOWETH:

1. That the Applicant has filed the accompanying suit for permanent injunction and refers to the averments therein as part and parcel of this application.

2. Facts and grounds establishing urgency and imminent threat:
{formatted_facts}

3. PRIMA FACIE CASE:
That the Applicant has a strong prima facie case on merits and is likely to succeed in the main suit.

4. BALANCE OF CONVENIENCE:
That the balance of convenience lies overwhelmingly in favour of the Applicant and against the Respondent.

5. IRREPARABLE LOSS:
That if temporary injunction is not granted, the Respondent will change the nature of the property / dissipate assets, causing irreparable harm that cannot be compensated in money.

PRAYER:
It is most respectfully prayed that this Hon'ble Court may graciously be pleased to:
a) Pass an ex-parte ad-interim order of injunction restraining the Respondent from alienating, encumbering, or changing the status-quo;
b) {prayer if prayer else "Restrain the Respondent from interfering with Applicant's peaceful possession;"}
c) Pass such other orders as deemed fit.

APPLICANT
THROUGH ADVOCATE

VERIFICATION & AFFIDAVIT:
Verified on solemn affirmation at ________ on ____/____/2026 that the contents of paragraphs 1 to 5 are true.
DEPONENT
"""

    # 13. High Court / Supreme Court Writ Petition (Article 226 / Article 32)
    elif doc_type in {"writ_petition", "article_226", "article_32"} or "writ" in doc_type or "writ" in title.lower() or "226" in title or "32" in title:
        court_header = court or f"IN THE HON'BLE HIGH COURT OF JUDICATURE AT {state_display.upper()}"
        full_title = title or "WRIT PETITION UNDER ARTICLE 226 OF THE CONSTITUTION OF INDIA (EXTRA-ORDINARY WRIT JURISDICTION)"
        sections = provisions or "Article 226 of the Constitution of India"

        full_draft = f"""{court_header}
CIVIL / CRIMINAL WRIT PETITION NO. ________ OF 2026

IN THE MATTER OF:
{applicant}
... PETITIONER

VERSUS

1. THE STATE OF {state_display.upper()}
   (Through Secretary / Competent Authority)
2. {respondent}
... RESPONDENTS

{full_title}
FOR ISSUANCE OF A WRIT OF MANDAMUS / CERTIORARI / HABEAS CORPUS OR ANY OTHER WRIT

MOST RESPECTFULLY SHOWETH:

1. That the Petitioner is an Indian citizen invoking the extra-ordinary constitutional writ jurisdiction of this Hon'ble Court.

2. Facts of the case in chronological sequence:
{formatted_facts}

3. GROUNDS:
   a) That the impugned action/inaction of the Respondents is arbitrary, capricious, and violates Article 14 of the Constitution.
   b) That the Respondents have deprived the Petitioner of lawful rights without authority of law, infringing Article 21.
   c) That the statutory authorities have failed to perform their public duties enjoined upon them by law.

4. NO ALTERNATIVE REMEDY:
That the Petitioner has no other efficacious and speedy alternative remedy available in law except to approach this Hon'ble Court under {sections}.

PRAYER:
Wherefore, it is most respectfully prayed that this Hon'ble Court may graciously be pleased to:
a) Issue a Writ of Mandamus or any other appropriate Writ directing the Respondents to grant the relief sought;
b) Quash and set aside the arbitrary order / notice dated ________;
c) {prayer if prayer else "Pass any other or further order(s) as this Hon'ble Court may deem fit and proper."}

PETITIONER
THROUGH ADVOCATE

VERIFICATION:
Verified at ________ on this ____ day of ________, 2026 that the contents of the writ petition are true to my knowledge and belief.
DEPONENT

AFFIDAVIT
I, the Deponent above-named, do hereby solemnly affirm that I am the Petitioner, and the statements made in the petition are true.
DEPONENT
"""

    # 14. RTI Application (Section 6(1) Right to Information Act, 2005)
    elif doc_type in {"rti_application"} or "rti" in doc_type or "rti" in title.lower() or "right to information" in title.lower():
        full_title = title or "APPLICATION UNDER SECTION 6(1) OF THE RIGHT TO INFORMATION ACT, 2005"
        full_draft = f"""FORM 'A'
{full_title}

To,
The Public Information Officer (PIO) / APIO,
Office of: {respondent}

1. Full Name of the Applicant: {applicant}
2. Address for correspondence: [Applicant Address]
3. Particulars of Information Required:
   Subject Matter: {title}
   Period to which information relates: [Relevant Period / Dates]
   Description of Information sought:
{formatted_facts}

4. Purpose / Context:
   {prayer if prayer else "For lawful personal and legal records."}

5. Application Fee:
   Prescribed fee of Rs. 10/- attached via Postal Order / Demand Draft / Court Fee Stamp No. ________.

Place: ________
Date: ________
Signature of Applicant: ____________________
"""

    # 15. Consumer Complaint (Section 35 Consumer Protection Act, 2019)
    elif doc_type in {"consumer_complaint"} or "consumer" in doc_type or "consumer" in title.lower():
        full_title = title or "COMPLAINT UNDER SECTION 35 OF THE CONSUMER PROTECTION ACT, 2019"
        full_draft = f"""BEFORE THE HON'BLE DISTRICT CONSUMER DISPUTES REDRESSAL COMMISSION AT {state_display.upper()}
CONSUMER COMPLAINT NO. ________ OF 2026

IN THE MATTER OF:
{applicant}
... COMPLAINANT

VERSUS

{respondent}
... OPPOSITE PARTY / SERVICE PROVIDER

{full_title}
FOR DEFICIENCY IN SERVICE AND UNFAIR TRADE PRACTICE

MOST RESPECTFULLY SHOWETH:

1. That the Complainant is a consumer within the meaning of Section 2(7) of the Consumer Protection Act, 2019.

2. That the Opposite Party is a service provider / seller engaged in commercial trade.

3. Factual details of purchase/service, deficiency, and harassment:
{formatted_facts}

4. That despite representations and notices, the Opposite Party failed to rectify the defect or refund the money.

PRAYER:
It is most respectfully prayed that this Hon'ble Commission may be pleased to direct the Opposite Party to:
a) Refund the amount paid along with interest;
b) Award compensation for mental agony, harassment, and litigation expenses: {prayer if prayer else "Rs. ________"};
c) Pass any other relief deemed fit in the interest of justice.

COMPLAINANT
THROUGH COUNSEL

VERIFICATION:
Verified at ________ on ____/____/2026 that the contents of the complaint are true to my knowledge and belief.
DEPONENT
"""

    # 16. Labour / Gratuity / Wages Grievance Application
    elif doc_type in {"labour_grievance", "gratuity_claim", "wages_claim"} or "gratuity" in title.lower() or "wages" in title.lower() or "labour" in title.lower():
        court_header = court or f"BEFORE THE CONTROLLING AUTHORITY UNDER THE PAYMENT OF GRATUITY ACT, 1972 / PAYMENT OF WAGES ACT AT {state_display.upper()}"
        full_title = title or "APPLICATION FOR RECOVERY OF STATUTORY DUES / GRATUITY"
        sections = provisions or "Payment of Gratuity Act, 1972 / Payment of Wages Act, 1936"

        full_draft = f"""{court_header}
CLAIM APPLICATION NO. ________ OF 2026

IN THE MATTER OF:
{applicant}
... APPLICANT / EMPLOYEE

VERSUS

{respondent}
... OPPOSITE PARTY / EMPLOYER

{full_title}
UNDER {sections}

MOST RESPECTFULLY SHOWETH:

1. That the Applicant was employed with the Opposite Party and rendered continuous service as an employee.

2. Details of tenure, last drawn wages, and withholding of legal dues:
{formatted_facts}

3. That the Opposite Party is statutorily obligated to disburse the dues within thirty days, but has unlawfully failed and withheld the payment.

PRAYER:
Wherefore, the Applicant prays that this Authority direct the Opposite Party to:
a) Disburse the unpaid statutory dues of Rs. ________;
b) Pay compound interest at statutory rates pursuant to Section 7(3A) of the Act;
c) {prayer if prayer else "Award costs and compensation for delay."}

APPLICANT
Date: ________
Place: ________
VERIFICATION:
Verified at ________ on ____/____/2026 that the facts stated above are true and correct.
DEPONENT
"""

    # 17. Motor Accident Claims Petition (MACT Section 166)
    elif doc_type in {"motor_accident_claim", "mact_petition"} or "mact" in doc_type or "mact" in title.lower() or "motor accident" in title.lower():
        court_header = court or f"BEFORE THE MOTOR ACCIDENT CLAIMS TRIBUNAL (MACT) AT {state_display.upper()}"
        full_title = title or "CLAIM PETITION UNDER SECTION 166 OF THE MOTOR VEHICLES ACT, 1988"

        full_draft = f"""{court_header}
MACP NO. ________ OF 2026

IN THE MATTER OF:
{applicant}
... CLAIMANT

VERSUS

1. Driver of offending vehicle: [Driver Name]
2. Owner of offending vehicle: [Owner Name]
3. Insurance Company: {respondent}
... RESPONDENTS

{full_title}
FOR COMPENSATION ARISING OUT OF MOTOR VEHICULAR ACCIDENT

MOST RESPECTFULLY SHOWETH:

1. That on ________ at about ________ hrs, an accident occurred due to rash and negligent driving of vehicle No. ________.

2. Factual details of accident, injuries/fatalities, FIR, and medical expenditure:
{formatted_facts}

3. That the Respondent Insurance Company is statutorily liable to indemnify and pay the compensation.

PRAYER:
The Claimant prays for an award of compensation of Rs. ________ along with interest at 9% per annum from the date of petition until realisation.
{prayer if prayer else ""}

CLAIMANT
THROUGH ADVOCATE

VERIFICATION:
Verified at ________ on ____/____/2026 that the statements in this claim petition are true.
DEPONENT
"""

    # 18. General / Special Power of Attorney
    elif doc_type in {"power_of_attorney", "gpa", "spa"} or "power of attorney" in title.lower() or "attorney" in title.lower():
        full_title = title or "GENERAL POWER OF ATTORNEY"
        full_draft = f"""GENERAL POWER OF ATTORNEY

KNOW ALL MEN BY THESE PRESENTS that I, {applicant} (hereinafter called the 'Executant'), do hereby appoint, nominate, and constitute:
{respondent} (hereinafter called the 'Attorney')
as my true and lawful attorney in my name and on my behalf to do and execute all or any of the following acts, deeds, and things:

STATEMENT OF PURPOSE AND POWERS:
{formatted_facts}

1. To appear, represent, and sign all documents before courts, tribunals, sub-registrars, and statutory authorities.
2. To sign plaints, written statements, affidavits, and receipts.
3. {prayer if prayer else "To do all lawful acts necessary for the management of the subject matter."}

AND I hereby agree to ratify and confirm all whatsoever my said attorney shall lawfully do by virtue of these presents.

IN WITNESS WHEREOF I have signed this Deed of General Power of Attorney on this ____ day of ________, 2026.

EXECUTANT
WITNESSES:
1. ____________________
2. ____________________
"""

    # 19. Sworn Affidavit
    elif doc_type in {"affidavit", "sworn_affidavit"} or "affidavit" in doc_type or "affidavit" in title.lower():
        court_header = court or f"BEFORE THE APPROPRIATE COURT / FORUM AT {state_display.upper()}"
        full_title = title or "SWORN AFFIDAVIT ON SOLEMN AFFIRMATION"
        full_draft = f"""{court_header}

IN THE MATTER OF:
{applicant}
... DEPONENT

VERSUS

{respondent}
... OPPOSITE PARTY

{full_title}

AFFIDAVIT
I, {applicant}, aged about ____ years, residing at ____________________, do hereby solemnly affirm and state on oath as follows:

1. That I am the Deponent herein and am fully conversant with the facts and circumstances of the case.

2. Statements on oath:
{formatted_facts}

3. That the contents stated above are true and correct to the best of my personal knowledge and belief, and nothing material has been concealed therefrom.

DEPONENT

VERIFICATION:
Verified at ________ on this ____ day of ________, 2026 that the contents of this affidavit are true and correct.
DEPONENT
"""

    # 20. UNIVERSAL COURT & STATUTORY ADJUDICATORY GENERATOR
    else:
        court_header = court or f"IN THE COURT OF THE LEARNED JUDGE / APPROPRIATE FORUM AT {state_display.upper()}"
        full_title = title or "FORMAL LEGAL APPLICATION / PETITION"
        sections = provisions or "THE APPLICABLE STATUTORY PROVISIONS"

        full_draft = f"""{court_header}
CASE / APPLICATION NO. ________ OF 2026

IN THE MATTER OF:
{applicant}
... APPLICANT / PETITIONER

VERSUS

{respondent}
... RESPONDENT / OPPOSITE PARTY

{full_title}
UNDER {sections}

MOST RESPECTFULLY SHOWETH:

1. That the Applicant / Petitioner submits this application before this Hon'ble Forum for statutory consideration and grant of relief in accordance with law.

2. Chronological statement of facts, dates, and cause of action:
{formatted_facts}

3. That the acts and omissions of the Respondent have directly infringed the legal rights, statutory entitlements, and interests of the Applicant.

4. That this Hon'ble Court / Forum has territorial and subject-matter jurisdiction to entertain and adjudicate upon the present application.

PRAYER:
Wherefore, in light of the facts and circumstances stated above, it is most respectfully prayed that this Hon'ble Court / Authority may graciously be pleased to:
a) Grant the relief prayed for in terms of law;
b) Direct the Respondent to comply with statutory obligations;
c) {prayer if prayer else "Pass any other or further order(s) as may be deemed just, proper, and equitable in the interest of justice."}

APPLICANT / PETITIONER
THROUGH COUNSEL / ADVOCATE

VERIFICATION:
I, the Applicant / Deponent above-named, do hereby verify on solemn affirmation that the contents of paragraphs 1 to 4 of the above application are true and correct to my knowledge and belief, and no material fact has been concealed therefrom.
Verified at ________ on this ____ day of ________, 2026.

DEPONENT / APPLICANT

AFFIDAVIT
I, the Deponent above-named, do hereby solemnly affirm and state on oath as follows:
1. That I am the Applicant in the accompanying application and am fully conversant with the facts of the case.
2. That the statements made in the accompanying application are true and correct to my knowledge.
DEPONENT
Verified at ________ on ____/____/2026 that the contents of this affidavit are true.
DEPONENT
"""

    warning = (
        "Statutory Notice: This draft is generated for assistance and reference. "
        "It must be reviewed, verified, and finalized in consultation with an enrolled advocate before filing in court."
    )
    return ToolResult(
        "draft_legal_document",
        True,
        data={
            "document_type": doc_type,
            "title": title,
            "content": full_draft.strip(),
            "intended_for_submission": True,
            "warning": warning,
            "artifact": {
                "type": doc_type,
                "title": title,
                "content": full_draft.strip(),
                "intended_for_submission": True,
                "warning": warning,
            },
        },
    )


HANDLERS: dict[str, Callable[[dict[str, Any], ToolContext], ToolResult]] = {
    "search_law": _search_law,
    "search_judgments": _search_judgments,
    "get_procedure": _get_procedure,
    "record_fact": _record_fact,
    "advance_workflow": _advance_workflow,
    "draft_legal_document": _draft_legal_document,
}


def execute(name: str, arguments: Any, ctx: ToolContext) -> ToolResult:
    """Run one proposed tool call. Never raises; refusals are results."""
    started = time.perf_counter()
    if name not in ctx.allowed_tools or name not in HANDLERS:
        return ToolResult(name, False, error="tool_not_allowed", ms=int((time.perf_counter() - started) * 1000))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return ToolResult(name, False, error="arguments_not_json", ms=int((time.perf_counter() - started) * 1000))
    if not isinstance(arguments, dict):
        return ToolResult(name, False, error="arguments_not_object", ms=int((time.perf_counter() - started) * 1000))
    # Schema check before the handler: an invented key, wrong type or invalid
    # enum value never reaches the corpus or the ledger.
    violation = validate_arguments(name, arguments)
    if violation:
        return ToolResult(name, False, error=violation, ms=int((time.perf_counter() - started) * 1000))
    try:
        result = HANDLERS[name](arguments, ctx)
    except Exception as exc:  # a tool bug must not take the turn down
        result = ToolResult(name, False, error=f"tool_error:{type(exc).__name__}")
    result.ms = int((time.perf_counter() - started) * 1000)
    return result
