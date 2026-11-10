"""Finite-state intake: the model talks, the file decides the path."""
from __future__ import annotations

import copy

import pytest

from app import facts, workflow


def ledger_with(**slots):
    ledger = facts.Ledger()
    for slot, value in slots.items():
        ledger.record(slot, value, "caller", 1)
    return ledger


def test_a_workflow_starts_at_its_first_step_and_asks_one_question():
    state = workflow.start("unpaid-wages")
    assert state["node"] == "amount"
    slot, question = workflow.question(state, facts.Ledger())
    assert slot == "amount"
    assert question == "How much is still unpaid, and for which months?"


def test_it_refuses_to_advance_past_missing_facts():
    state = workflow.start("unpaid-wages")
    moved, events = workflow.advance(state, ledger_with(amount="₹40,000"))
    assert moved["node"] == "amount"
    assert events == []

    refused = workflow.advance(moved, ledger_with(amount="₹40,000"), to="employer")
    assert refused[0]["node"] == "amount"
    assert refused[1][0]["event"] == "refused"
    assert refused[1][0]["reason"].startswith("missing_")


def test_it_advances_when_the_step_is_satisfied_and_branches_on_facts():
    # A household employer takes the household branch and, with its fact known,
    # runs to the reviewed procedure.
    ledger = ledger_with(amount="₹40,000", when="March", recipient="a household", stage="salary_unpaid")
    moved, events = workflow.advance(workflow.start("unpaid-wages"), ledger)
    assert "household" in moved["history"]
    assert moved["status"] == "completed"
    assert any(event["to"] == "household" for event in events if event["event"] == "advanced")

    # Without the household fact it stops there and asks the step's question.
    partial = ledger_with(amount="₹40,000", when="March", recipient="a household")
    stopped, _ = workflow.advance(workflow.start("unpaid-wages"), partial)
    assert stopped["node"] == "household"
    slot, question = workflow.question(stopped, partial)
    assert slot == "stage"
    assert "more than one household" in question

    # A company employer takes the other branch instead.
    company = ledger_with(amount="₹40,000", when="March", recipient="a private company")
    branched, _ = workflow.advance(workflow.start("unpaid-wages"), company)
    assert branched["node"] == "employer-proof"
    assert "household" not in branched["history"]


def test_a_tool_node_returns_a_reviewed_procedure_and_completes():
    ledger = ledger_with(
        amount="₹40,000", when="March", recipient="a company",
        notice_or_order="received", stage="salary_unpaid",
    )
    state = workflow.start("unpaid-wages")
    moved, events = workflow.advance(state, ledger)
    tools = [event for event in events if event["event"] == "tool_call"]
    assert len(tools) == 1
    assert tools[0]["tool"] == "get_procedure"
    assert tools[0]["arguments"] == {"procedure_id": "unpaid-wages"}
    assert moved["status"] == "completed"

    assert workflow.public(moved, ledger)["status"] == "completed"
    assert workflow.public(moved, ledger)["needs"] is None


def test_an_illegal_transition_is_refused_without_moving():
    ledger = ledger_with(amount="₹40,000", when="March", recipient="a company")
    state = workflow.start("unpaid-wages")
    moved, events = workflow.advance(state, ledger, to="done")
    assert moved["node"] == state["node"]
    assert events[0]["reason"] == "transition_not_allowed"


def test_safety_aborts_an_active_workflow():
    state = workflow.start("police-fir")
    aborted = workflow.abort(state, "immediate_danger")
    assert aborted["status"] == "aborted"
    assert workflow.question(aborted, facts.Ledger()) == (None, None)
    assert workflow.advance(aborted, facts.Ledger())[1] == []


def test_safety_does_not_rewrite_an_intake_that_already_finished():
    ledger = ledger_with(
        amount="₹40,000", when="March", recipient="a company",
        notice_or_order="received", stage="salary_unpaid",
    )
    done, _ = workflow.advance(workflow.start("unpaid-wages"), ledger)
    assert done["status"] == "completed"
    # A later emergency must not claim the finished intake was interrupted.
    assert workflow.abort(done, "immediate_danger")["status"] == "completed"


def test_step_count_follows_the_questions_actually_asked():
    ledger = ledger_with(amount="₹40,000", when="March", recipient="a company")
    state, _ = workflow.advance(workflow.start("unpaid-wages"), ledger)
    assert state["node"] == "employer-proof"
    view = workflow.public(state, ledger)
    # Four conversation nodes exist in the file, but the household branch was
    # never asked, so "4 / 4" would be a false report.
    assert view["step"] == 3
    assert view["total"] == 4
    assert view["skipped"] == []

    completed, _ = workflow.advance(state, ledger_with(
        amount="₹40,000", when="March", recipient="a company", notice_or_order="received",
    ))
    done = workflow.public(completed, ledger)
    assert done["step"] == 3
    assert done["total"] == 3
    assert done["skipped"] == ["household"]


def test_facts_from_another_matter_do_not_satisfy_this_one():
    ledger = facts.Ledger()
    ledger.record("when", "yesterday", "caller", 1, "date", "police-fir")
    ledger.record("notice_or_order", "written refusal", "caller", 1, "text", "police-fir")
    # The same slots are still unknown to the wages intake.
    assert ledger.known_for("when", "unpaid-wages") is False
    assert ledger.value_for("notice_or_order", "unpaid-wages") is None
    assert ledger.known_for("when", "police-fir") is True


def test_a_fact_restated_in_a_new_matter_is_not_a_contradiction():
    ledger = facts.Ledger()
    ledger.record("when", "yesterday", "caller", 1, "date", "police-fir")
    ledger.record("when", "March", "caller", 2, "date", "unpaid-wages")
    assert ledger.value_for("when", "unpaid-wages") == "March"
    assert ledger.contradictions == []


def test_state_round_trips_and_rejects_unknown_nodes():
    state = workflow.start("police-fir")
    restored = workflow.load_state(copy.deepcopy(state))
    assert restored == state
    assert workflow.load_state({"id": "nope", "node": "x"}) is None
    assert workflow.load_state({"id": "police-fir", "node": "invented"}) is None


def test_procedure_lookup_for_starting_a_workflow():
    assert workflow.for_procedure("unpaid-wages")["id"] == "unpaid-wages"
    assert workflow.for_procedure("not-a-procedure") is None
    assert workflow.for_procedure(None) is None


def test_every_declared_workflow_links_to_a_real_procedure():
    from app import procedural

    for item in workflow.WORKFLOWS:
        procedure_id = item.get("procedure_id")
        if procedure_id:
            assert procedure_id in procedural.BY_ID, item["id"]
