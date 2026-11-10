"""The post-retrieval relevance check.

Retrieval answers "which passages share the question's words". This stage
answers the different question a lawyer asks before citing anything: "is this
judgment about what the caller asked". The tests pin the two properties that
matter for safety — it never invents a verdict, and an unavailable judge must
leave the candidate list untouched rather than deleting decisions.
"""
from __future__ import annotations

import asyncio

import pytest

from app import model_gateway


CANDIDATES = [
    {"id": "a", "title": "Refusal of registration under the Income-tax Act", "excerpt": "refusal of registration"},
    {"id": "b", "title": "Lalita Kumari v State of U.P.", "excerpt": "information disclosing a cognizable offence"},
]


def test_no_model_means_no_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_gateway, "configured", lambda: False)
    assert asyncio.run(model_gateway.judge_cases("refusal of FIR", CANDIDATES)) is None
    assert model_gateway.case_judge_available() is False


def test_disabled_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_CASE_JUDGE", "off")
    monkeypatch.setattr(model_gateway, "configured", lambda: True)
    assert model_gateway.case_judge_available() is False


def test_verdicts_are_parsed_and_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_CASE_JUDGE", "on")
    monkeypatch.setattr(model_gateway, "configured", lambda: True)

    async def fake_json_completion(*args, **kwargs):
        return {"verdicts": [{"index": 0, "addresses": False}, {"index": 1, "addresses": True}]}

    monkeypatch.setattr(model_gateway, "_json_completion", fake_json_completion)
    verdicts = asyncio.run(model_gateway.judge_cases("refusal of FIR", CANDIDATES))
    assert verdicts == {0: False, 1: True}


def test_relevance_scores_are_parsed_and_can_reorder(monkeypatch: pytest.MonkeyPatch) -> None:
    """A keep/drop verdict cannot order four similar decisions.

    The retriever can put the leading authority fourth and a judgment that
    merely cites it first; a 0-3 score is the same small call with the
    information the caller's reading order needs. The boolean view stays for
    callers that only filter.
    """
    monkeypatch.setenv("LLM_CASE_JUDGE", "on")
    monkeypatch.setattr(model_gateway, "configured", lambda: True)

    async def fake_json_completion(*args, **kwargs):
        return {"verdicts": [
            {"index": 0, "addresses": False, "relevance": 1},
            {"index": 1, "addresses": True, "relevance": 3},
        ]}

    monkeypatch.setattr(model_gateway, "_json_completion", fake_json_completion)
    assert asyncio.run(model_gateway.case_relevance("refusal of FIR", CANDIDATES)) == {0: 1, 1: 3}
    assert asyncio.run(model_gateway.judge_cases("refusal of FIR", CANDIDATES)) == {0: False, 1: True}


def test_a_malformed_answer_is_not_a_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """A judge that returns prose, or an index out of range, must not be read
    as "these judgments are irrelevant" — that would silently delete evidence."""
    monkeypatch.setenv("LLM_CASE_JUDGE", "on")
    monkeypatch.setattr(model_gateway, "configured", lambda: True)

    async def bad(*args, **kwargs):
        return {"verdicts": "not a list"}

    monkeypatch.setattr(model_gateway, "_json_completion", bad)
    assert asyncio.run(model_gateway.judge_cases("q", CANDIDATES)) is None

    async def out_of_range(*args, **kwargs):
        return {"verdicts": [{"index": 99, "addresses": False}]}

    monkeypatch.setattr(model_gateway, "_json_completion", out_of_range)
    assert asyncio.run(model_gateway.judge_cases("q", CANDIDATES)) is None
