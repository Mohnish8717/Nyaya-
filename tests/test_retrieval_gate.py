"""The CI retrieval gate: a regression past the checked-in baseline fails."""
from __future__ import annotations

import json
from pathlib import Path

from evals.retrieval_eval import BASELINE, gate


def test_baseline_is_checked_in_and_well_formed():
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    for name in ("lexical_bm25", "dense_only", "hybrid_no_rerank", "hybrid_rerank"):
        row = metrics[name]
        assert isinstance(row["hit@5"], (int, float)) and 0.0 <= row["hit@5"] <= 1.0
        assert isinstance(row["mrr"], (int, float)) and 0.0 <= row["mrr"] <= 1.0
    assert 0.0 <= data["tolerance"] <= 0.25


def run_only(results, config="hybrid_rerank"):
    return [row for row in results if row["config"] == config]


def test_gate_passes_at_the_recorded_baseline():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    stored = baseline["metrics"]["hybrid_rerank"]
    run = [{"config": "hybrid_rerank", "hit@5": stored["hit@5"], "mrr": stored["mrr"]}]
    ok, failures = gate(run, baseline, tolerance=baseline["tolerance"])
    assert ok and failures == []


def test_gate_fails_when_hit_rate_regresses_beyond_tolerance():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    stored = baseline["metrics"]["hybrid_rerank"]
    run = [{"config": "hybrid_rerank", "hit@5": stored["hit@5"] - baseline["tolerance"] - 0.05, "mrr": stored["mrr"]}]
    ok, failures = gate(run, baseline, tolerance=baseline["tolerance"])
    assert not ok
    assert any("hit@5" in failure for failure in failures)
    assert not any("mrr" in failure for failure in failures)


def test_gate_ignores_latency_and_missing_configs_fail():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert gate([{"config": "unrelated", "hit@5": 1.0}], baseline, 0.1)[0] is False
