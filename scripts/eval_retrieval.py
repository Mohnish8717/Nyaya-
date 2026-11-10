#!/usr/bin/env python3
"""Measure the retrieval change instead of asserting it.

Per query:

  legacy     judgments the old OR-of-every-word query considered
  candidates judgments the new cascade actually considered end to end
  focus      share of the returned top 5 that contain the two rarest query
             terms (the terms the cascade treats as required)

`focus` is a cheap precision proxy, not a relevance judgement; there is no
labelled relevance set in this repo yet, and pretending otherwise would be
the kind of number this project is trying not to produce.

Run: python scripts/eval_retrieval.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import case_corpus, retrieval  # noqa: E402

QUERIES = [
    "my landlord is not returning my security deposit and he is threatening me",
    "my salary has not been paid for three months and I was terminated without notice",
    "the cheque bounced because of insufficient funds what can I do",
    "husband and in laws harassing me for dowry",
    "police refused to register my fir for theft",
    "defective product the seller will not refund it",
    "builder delayed possession of my flat",
    "employer did not deposit my provident fund",
]


def legacy_pool(db: sqlite3.Connection, query: str) -> int:
    expression = retrieval.legacy_match(query)
    if not expression:
        return 0
    return db.execute(
        "SELECT COUNT(DISTINCT source_id) FROM case_fts WHERE case_fts MATCH ?", (expression,)
    ).fetchone()[0]


def main() -> int:
    if not case_corpus.CASE_DB.exists():
        print("No judgment index. Run: python scripts/reindex_judgments.py")
        return 1

    with sqlite3.connect(case_corpus.CASE_DB) as db:
        total = db.execute("SELECT COUNT(DISTINCT source_id) FROM case_fts").fetchone()[0]
        print(f"index: {total} judgments\n")
        header = f"{'query':52} {'legacy':>7} {'candidates':>11} {'focus':>6}"
        print(header)
        print("-" * len(header))

        legacy_total = candidate_total = 0
        for query in QUERIES:
            cases, stats = case_corpus.retrieve_cases_with_stats(query, "IN", limit=5)
            legacy = legacy_pool(db, query)
            strict_terms = stats["terms"][:2]
            focused = 0
            if cases and strict_terms:
                focused = sum(
                    1 for case in cases if all(term in (case.get("excerpt") or "").lower() for term in strict_terms)
                )
            focus = focused / len(cases) if cases else 0.0
            legacy_total += legacy
            candidate_total += stats["candidates"]
            label = query if len(query) <= 50 else query[:47] + "..."
            print(f"{label:52} {legacy:>7} {stats['candidates']:>11} {focus:>5.0%}")

    reduction = 1 - (candidate_total / legacy_total) if legacy_total else 0
    print(f"\nmean candidates: {legacy_total / len(QUERIES):.0f} legacy -> {candidate_total / len(QUERIES):.0f} cascade ({reduction:.0%} smaller)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
