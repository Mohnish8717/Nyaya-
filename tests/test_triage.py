"""Triage decides severity and, more importantly, what is forbidden."""
from __future__ import annotations

from app import triage


def test_active_threat_is_critical_and_blocks_drafting() -> None:
    result = triage.assess("my husband is beating me and he is coming now")
    assert result.level == "critical"
    assert result.has("block_drafting")
    assert result.has("safety_first")
    assert triage.is_emergency(result)


def test_devanagari_threat_is_detected() -> None:
    result = triage.assess("मेरे पति मुझे मार रहे हैं")
    assert result.level == "critical"
    assert triage.is_emergency(result)


def test_fraud_in_progress_is_high_and_tells_the_caller_to_report_now() -> None:
    result = triage.assess("an OTP call happened and ₹50,000 was debited from my bank account")
    assert result.level == "high"
    assert result.has("report_now")
    assert result.has("preserve_evidence")
    assert not triage.is_emergency(result)


def test_routine_question_is_normal() -> None:
    result = triage.assess("What documents do I need to file an FIR?")
    assert result.level == "normal"
    assert result.policies == []
    assert not triage.is_emergency(result)


def test_highest_severity_wins_and_policies_accumulate() -> None:
    result = triage.assess("My husband beats me and my child is missing")
    assert result.level == "critical"
    keys = {item["key"] for item in result.categories}
    assert {"domestic_violence", "child_at_risk"} <= keys
    assert result.has("safety_first")
    assert result.has("block_drafting")


def test_deadline_pressure_never_claims_a_computed_deadline() -> None:
    result = triage.assess("my court hearing is tomorrow and the notice is pending")
    assert result.level == "normal"
    assert result.has("mention_deadline_without_computing")


def test_safety_question_is_ordered_first() -> None:
    questions = ["What is your name?", "Are you somewhere safe right now?"]
    ordered = triage.sort_questions(questions, triage.assess("he is coming now with a knife"))
    assert ordered[0].startswith("Are you somewhere safe")
