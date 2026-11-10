"""Retrieval eval: does hybrid actually beat lexical on this corpus?

Fifteen known-item cases: one judgment was read, then a query was written in
the words a citizen would use rather than the judgment's own title. Four are
Hindi, because the failure being fixed is precisely a caller speaking a
language the corpus does not share tokens with.

Metrics per configuration: hit@5, MRR, and median query latency. The cases are
few and the corpus is a 2,046-judgment sample; this measures a stage, not the
product's legal quality.

    python evals/retrieval_eval.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import case_corpus, dense, retrieval  # noqa: E402

CASES = ROOT / "evals" / "retrieval_cases.jsonl"
BASELINE = ROOT / "evals" / "retrieval_baseline.json"


def gate(results: list[dict], baseline: dict, tolerance: float, provider: str = "hybrid_rerank") -> tuple[bool, list[str]]:
    """Fail when the provider config regresses beyond `tolerance` on the baseline.

    Comparison is absolute, against checked-in numbers measured on the same
    bundled corpus. Latency is deliberately not gated: CI runners are noisy,
    and latency has its own budgets in tests/test_latency.py.
    """
    baseline_metrics = (baseline.get("metrics") or {}).get(provider)
    if not baseline_metrics:
        return False, [f"baseline has no metrics for '{provider}'"]
    run = next((row for row in results if row.get("config") == provider), None)
    if run is None:
        return False, [f"results have no '{provider}' row"]
    failures: list[str] = []
    for metric in ("hit@5", "mrr"):
        recorded = baseline_metrics.get(metric)
        observed = run.get(metric)
        if not isinstance(recorded, (int, float)) or not isinstance(observed, (int, float)):
            failures.append(f"{metric}: baseline={recorded!r} observed={observed!r}")
            continue
        if observed < recorded - tolerance:
            failures.append(
                f"{provider}.{metric} regressed: {observed} < baseline {recorded} (tolerance {tolerance})"
            )
    return not failures, failures


def evaluate(name: str, search, selected: list[dict] | None = None) -> dict:
    hits = 0
    reciprocal = 0.0
    latencies: list[float] = []
    misses: list[str] = []
    for case in (selected if selected is not None else cases):
        started = time.perf_counter()
        ids = search(case)
        latencies.append((time.perf_counter() - started) * 1000)
        if case["target"] in ids:
            hits += 1
            reciprocal += 1.0 / (ids.index(case["target"]) + 1)
        else:
            misses.append(case["id"])
    total = len(selected if selected is not None else cases) or 1
    return {
        "config": name,
        "hit@5": round(hits / total, 3),
        "mrr": round(reciprocal / total, 3),
        "p50_ms": round(statistics.median(latencies), 1),
        "misses": misses,
    }


cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]


USE_CORPUS: bool | None = False


def lexical(case: dict) -> list[str]:
    cases_result, _ = case_corpus.retrieve_cases_with_stats(
        case["query"], "IN", 5, use_dense=False, use_rerank=False, use_corpus=USE_CORPUS
    )
    return [item["id"] for item in cases_result]


def hybrid(case: dict) -> list[str]:
    cases_result, _ = case_corpus.retrieve_cases_with_stats(case["query"], "IN", 5, use_corpus=USE_CORPUS)
    return [item["id"] for item in cases_result]


def hybrid_no_rerank(case: dict) -> list[str]:
    cases_result, _ = case_corpus.retrieve_cases_with_stats(
        case["query"], "IN", 5, use_rerank=False, use_corpus=USE_CORPUS
    )
    return [item["id"] for item in cases_result]


def hybrid_rewrite(case: dict) -> list[str]:
    """Retrieval with the turn's second phrasing merged under the first.

    This is the production shape: the reader writes `search_query` in the
    vocabulary judgments use, retrieval searches both, and the interleave keeps
    the caller's own wording in front. Cases without a rewrite behave exactly
    like `hybrid`, so the row is only reported under `--rewrites`, where the
    difference it measures is real.
    """
    base = [item["id"] for item in case_corpus.retrieve_cases(case["query"], "IN", 5, use_corpus=USE_CORPUS)]
    rewrite = case.get("rewrite")
    if not rewrite:
        return base
    extra = [item["id"] for item in case_corpus.retrieve_cases(rewrite, "IN", 5, use_corpus=USE_CORPUS)]
    return retrieval.interleave(base, extra, 5)


def dense_only(case: dict) -> list[str]:
    return dense.search(case["query"], limit=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate", action="store_true",
        help="exit 1 when hybrid_rerank regresses beyond the checked-in baseline",
    )
    parser.add_argument(
        "--rewrites", action="store_true",
        help=(
            "also measure hybrid retrieval with the turn's rewritten search query "
            "merged under the caller's own wording, on the cases that carry one"
        ),
    )
    parser.add_argument(
        "--store", choices=("legacy", "hybrid"), default="legacy",
        help=(
            "which judgment store to measure. `legacy` (default) is the curated "
            "partition set the checked-in baseline was measured on, so the gate "
            "keeps meaning what it meant. `hybrid` adds the embedded passage "
            "corpus and reports without a baseline: its targets are not in that "
            "store, so hit@5 there measures rank position against other, often "
            "more authoritative judgments rather than relevance."
        ),
    )
    args = parser.parse_args()
    global USE_CORPUS
    USE_CORPUS = args.store == "hybrid"
    if not dense.available():
        print(json.dumps({
            "error": "dense index not available",
            "status": dense.backend_status(),
            "fix": "python scripts/build_dense_index.py",
        }, indent=2))
        return 1
    # Warm the model and matrix so latency reflects steady-state turns.
    dense.search("warm up query", limit=1)
    # A case whose target lives in the embedded corpus cannot be found by the
    # curated store, so counting it in a legacy run would read as a regression
    # in a store that was never asked. It is measured only where it exists.
    selected = [
        case for case in cases
        if USE_CORPUS or not str(case.get("target", "")).startswith("opennyai:")
    ]
    rows = [
        evaluate("lexical_bm25", lexical, selected),
        evaluate("dense_only", dense_only, selected),
        evaluate("hybrid_no_rerank", hybrid_no_rerank, selected),
        evaluate("hybrid_rerank", hybrid, selected),
    ]
    if args.rewrites:
        rows.append(evaluate("hybrid_rewrite", hybrid_rewrite, selected))
    report = {"cases": len(selected), "store": args.store, "results": rows}
    if args.rewrites:
        report["rewrite_cases"] = sum(1 for case in cases if case.get("rewrite"))
    if args.store == "hybrid" and args.gate:
        print(json.dumps({
            "gate": "not_applicable",
            "reason": "the checked-in baseline was measured on the curated partitions; "
                      "run without --store hybrid, or regenerate a hybrid baseline deliberately",
            **report,
        }, indent=2))
        return 0
    if not args.gate:
        print(json.dumps(report, indent=2))
        return 0
    if not BASELINE.exists():
        print(json.dumps({"gate": "failed", "error": f"baseline missing: {BASELINE.name}"}, indent=2))
        return 1
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    ok, failures = gate(rows, baseline, float(baseline.get("tolerance", 0.1)))
    print(json.dumps({"gate": "passed" if ok else "failed", "failures": failures, **report}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
