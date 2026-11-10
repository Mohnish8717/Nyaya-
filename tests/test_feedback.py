"""Failure-to-rule loop: bounded store, PII-safe records, deduplicated candidates."""
from __future__ import annotations

import json

import pytest

from app import feedback


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "FAILURES_PATH", tmp_path / "failures.jsonl")
    return feedback


def test_record_and_recent_roundtrip(store):
    store.record("degraded_turn", note="generation degraded", route="statute")
    rows = store.recent()
    assert len(rows) == 1
    assert rows[0]["kind"] == "degraded_turn"
    assert rows[0]["signature"] == feedback.signature("generation degraded")


def test_failure_store_never_holds_raw_identifiers(store):
    store.record(
        "unhandled_error", note="KeyError",
        detail="mobile +919876543210 pan ABCDE1234F aadhaar 234567890123 account 12345678901234",
    )
    blob = json.dumps(store.recent(), ensure_ascii=False)
    for secret in ("919876543210", "ABCDE1234F", "234567890123", "12345678901234"):
        assert secret not in blob


def test_store_is_bounded_by_rotation(store, monkeypatch):
    monkeypatch.setattr(feedback, "MAX_LINES", 25)
    for index in range(40):
        store.record("claim_rejected", note=f"case {index}")
    rows = store.recent(limit=100)
    assert len(rows) == feedback.MAX_LINES
    assert rows[0]["note"].startswith("case 39")  # newest kept, oldest rotated out


def test_candidates_group_and_rank_repeats(store):
    for _ in range(5):
        store.record("degraded_turn", note="provider timeout", route="statute")
    for _ in range(2):
        store.record("claim_rejected", note="section invented", route="statute")
    store.record("tool_over_budget", note="search_law slow", ms=2000)
    ranked = store.candidates()
    assert ranked[0]["kind"] == "degraded_turn"
    assert ranked[0]["occurrences"] == 5
    kinds = {row["kind"] for row in ranked}
    assert kinds == {"degraded_turn", "claim_rejected", "tool_over_budget"}
    total = sum(row["occurrences"] for row in ranked)
    assert total == store.stats()["failures"] == 8


def test_unknown_kinds_are_still_recorded_but_named_in_the_registry(store):
    store.record("some_new_signal", note="future kind")
    assert [row["kind"] for row in store.recent()] == ["some_new_signal"]
    # The registry exists so a review of KNOWN_KINDS lists what the loop
    # watches; new kinds arrive here through code review, not silently.
    assert feedback.KNOWN_KINDS == {
        "unhandled_error", "degraded_turn", "claim_rejected", "tool_over_budget",
    }
