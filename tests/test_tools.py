"""Tool boundary: read-only retrieval, guarded writes, refusals as results."""
from __future__ import annotations

from app import facts, tools, workflow


def context(text="Mera maalik teen mahine se ₹40,000 nahi de raha, Karnataka mein", **kwargs):
    return tools.ToolContext(
        text=text,
        jurisdiction="IN-KA",
        ledger=facts.Ledger(),
        workflow_state=kwargs.get("workflow_state"),
        turn=2,
    )


def test_unquoted_fact_is_refused_and_quoted_fact_is_stored():
    ctx = context()
    refused = tools.execute(
        "record_fact",
        {"slot": "amount", "value": "₹40,000", "evidence": "pachaas hazaar rupaye"},
        ctx,
    )
    assert refused.ok is False
    assert refused.error == "evidence_not_in_caller_message"
    assert ctx.ledger.known("amount") is False

    accepted = tools.execute(
        "record_fact",
        {"slot": "amount", "value": "₹40,000", "evidence": "₹40,000 nahi de raha"},
        ctx,
    )
    assert accepted.ok is True
    assert ctx.ledger.value("amount") == "₹40,000"
    assert ctx.ledger.facts["amount"].source == "tool"


def test_fact_slots_and_jurisdictions_are_closed_vocabularies():
    ctx = context()
    # The slot enum is now enforced twice over: the argument validator
    # refuses an undeclared slot before the handler ever runs, and the
    # handler keeps its own closed-vocabulary check as the backstop.
    refusal = tools.execute("record_fact", {"slot": "mood", "value": "bad", "evidence": "nahi de raha"}, ctx)
    assert refusal.ok is False
    assert refusal.error in {"argument_not_allowed", "slot_not_allowed"}
    unknown = tools.execute(
        "record_fact",
        {"slot": "jurisdiction", "value": "Atlantis", "evidence": "Karnataka mein"},
        ctx,
    )
    assert unknown.error == "jurisdiction_not_known"


def test_procedure_tool_returns_reviewed_steps_only():
    result = tools.execute("get_procedure", {"procedure_id": "police-fir"}, context())
    assert result.ok is True
    assert result.data["steps"]
    assert result.data["review_status"] in {"approved", "draft"}
    assert result.data["source"]["url"].startswith("https://")

    missing = tools.execute("get_procedure", {"procedure_id": "invented"}, context())
    assert missing.ok is False
    assert missing.error == "procedure_not_found"


def test_law_search_returns_quotable_passages():
    result = tools.execute("search_law", {"query": "arrested and produced before a magistrate"}, context())
    assert result.ok is True
    assert result.data["count"] >= 1
    first = result.data["passages"][0]
    assert first["id"] and first["text"]


def test_workflow_tool_cannot_jump_without_facts():
    state = workflow.start("unpaid-wages")
    ctx = context(workflow_state=state)
    refused = tools.execute("advance_workflow", {"node_id": "procedure"}, ctx)
    assert refused.ok is False
    assert refused.error.startswith("missing_")
    assert state["node"] == "amount"

    ctx.ledger.record("amount", "₹40,000", "caller", 2, "amount")
    ctx.ledger.record("when", "March", "caller", 2, "date")
    moved = tools.execute("advance_workflow", {"node_id": "employer"}, ctx)
    assert moved.ok is True
    assert state["node"] == "employer"


def test_tools_are_an_allowlist_and_arguments_are_validated():
    assert tools.execute("run_shell", {"cmd": "ls"}, context()).error == "tool_not_allowed"
    assert tools.execute("search_law", "{not json", context()).error == "arguments_not_json"
    assert tools.execute("search_law", ["list"], context()).error == "arguments_not_object"
    assert tools.execute("search_law", {"query": "a"}, context()).error == "query_too_short"


def test_tool_result_serialises_for_the_model_within_a_bound():
    result = tools.execute("get_procedure", {"procedure_id": "unpaid-wages"}, context())
    message = result.as_message()
    assert len(message) <= tools.MAX_TOOL_RESULT_CHARS
    assert '"procedure_id"' not in message  # shaped data, not raw internals
    assert result.as_dict()["tool"] == "get_procedure"
