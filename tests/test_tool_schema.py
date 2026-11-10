"""Tool registry contract: schema enforcement, side-effect metadata, budgets.

These tests hold the harness guideline explicit: the registry, the declared
schemas and the handlers cannot drift apart silently, and every proposed tool
call is validated before a handler can see it.
"""
from __future__ import annotations

import json

from app import facts, tools, workflow


def context(text="Mera maalik teen mahine se ₹40,000 nahi de raha, Karnataka mein", **kwargs):
    return tools.ToolContext(
        text=text,
        jurisdiction="IN-KA",
        ledger=facts.Ledger(),
        workflow_state=kwargs.get("workflow_state"),
        turn=2,
    )


def test_registry_and_handlers_and_meta_cannot_drift():
    names = {spec["function"]["name"] for spec in tools.TOOL_SPECS}
    assert names == set(tools.HANDLERS), "every declared tool needs a handler and vice versa"
    assert names == set(tools.TOOL_META), "every tool needs side-effect metadata"
    assert names == set(tools.BUDGET_MS), "every tool needs a wall-clock budget"
    for spec in tools.TOOL_SPECS:
        fn = spec["function"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert params.get("additionalProperties") is False
        assert set(params["required"]) <= set(params["properties"])
        for key, prop in params["properties"].items():
            assert prop.get("type") in {("string"), ("integer"), ("boolean")}
            enum = prop.get("enum")
            if enum:
                assert all(isinstance(value, str) for value in enum)
    assert set(tools.BUDGET_MS) <= set(tools.HANDLERS)


def test_side_effect_classes_run_from_read_only_to_guarded_write_and_nothing_further():
    classes = {name: meta["side_effects"] for name, meta in tools.TOOL_META.items()}
    assert set(classes.values()) == {tools.READ_ONLY, tools.GUARDED_WRITE}
    for name, cls in classes.items():
        if cls == tools.GUARDED_WRITE:
            assert name in {"record_fact", "advance_workflow"}


def test_validator_refuses_invented_keys_types_and_enums():
    assert tools.validate_arguments("search_law", {"query": "child maintenance"}) is None
    assert tools.validate_arguments("search_law", {"query": "child maintenance", "hack": 1}) == "argument_not_allowed"
    assert tools.validate_arguments("search_law", {"query": 42}) == "argument_type_mismatch"
    assert tools.validate_arguments("search_law", {}) == "missing_required_argument"
    assert tools.validate_arguments("search_judgments", {"query": "fir", "year_from": "2020"}) == "argument_type_mismatch"
    assert tools.validate_arguments("search_judgments", {"query": "fir", "year_from": 2012}) is None
    assert tools.validate_arguments("record_fact", {"slot": "mood", "value": "v", "evidence": "e"}) == "argument_not_allowed"
    assert tools.validate_arguments("advance_workflow", {"node_id": "employer", "reason": "caller named it"}) is None


def test_validator_neutralises_a_shell_tool_or_any_undeclared_name():
    assert tools.validate_arguments("run_shell", {"cmd": "ls"}) == "tool_not_allowed"
    assert tools.execute("run_shell", {"cmd": "ls"}, context()).error == "tool_not_allowed"
    assert tools.execute("search_law", {"query": "child maintenance", "hack": "drop table"}, context()).error == "argument_not_allowed"


def test_validation_happens_before_the_handler_sees_arguments():
    # An invalid call must not be able to reach corpus work: the refusal is
    # returned by the gate, not by the handler.
    forged = tools.execute("search_law", {"query": "fir rights", "text": "x" * 100000}, context())
    assert forged.ok is False
    assert forged.error == "argument_not_allowed"
    assert forged.data == {}


def test_workflow_refusal_stops_at_missing_facts():
    state = workflow.start("police-fir")
    ctx = context(workflow_state=state)
    refused = tools.execute("advance_workflow", {"node_id": "escalate"}, ctx)
    assert refused.ok is False
    assert refused.error.startswith("missing_")
    assert state["node"] == "refusal"


def test_result_trace_reports_metadata_and_budget():
    result = tools.execute("search_law", {"query": "wages withheld"}, context())
    trace = result.as_dict()
    assert trace["side_effects"] == tools.READ_ONLY
    assert trace["budget_ms"] == tools.BUDGET_MS["search_law"]
    assert trace["over_budget"] is False
    message = result.as_message()
    assert json.loads(message)  # still clean JSON for the model


def test_draft_legal_document_generates_valid_artifact():
    res = tools.execute(
        "draft_legal_document",
        {
            "document_type": "application_156_3",
            "title": "Application under Section 156(3) CrPC before the Magistrate",
            "facts_and_grounds": "The complainant went to the police station to file an FIR on theft, but the police refused to register it.",
            "prayer": "Direct the police station to register an FIR and submit an investigation report.",
            "jurisdiction": "IN-DL",
        },
        context(),
    )
    assert res.ok is True
    assert "artifact" in res.data
    artifact = res.data["artifact"]
    assert artifact["intended_for_submission"] is True
    assert "Section 156(3)" in artifact["content"]
    assert "Lalita Kumari" in artifact["content"]
    assert "PRAYER" in artifact["content"]
    assert "VERIFICATION" in artifact["content"]

