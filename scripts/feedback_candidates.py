"""Fold the failure store into candidate eval fixtures.

    python scripts/feedback_candidates.py         # summary to stdout
    python scripts/feedback_candidates.py --write # also refresh evals/fixture_candidates.jsonl

Closes the failure-to-rule loop: recurring `unhandled_error`,
`degraded_turn`, `claim_rejected` and `tool_over_budget` signals from
production become grouped candidates in `evals/fixture_candidates.jsonl`.
A maintainer reviews these, turns real ones into rules (code +
`evals/golden.jsonl` behavioural cases) and deletes them from the
candidates file. The script never invents fixtures by itself — a human
decision is the gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals" / "fixture_candidates.jsonl"

sys.path.insert(0, str(ROOT))

from app import feedback  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="refresh evals/fixture_candidates.jsonl")
    args = parser.parse_args()
    stats = feedback.stats()
    candidates = feedback.candidates()
    report = {"stats": stats, "candidates": len(candidates), "rows": candidates}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.write:
        OUT.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in candidates) + ("\n" if candidates else ""),
            encoding="utf-8",
        )
        print(f"wrote {len(candidates)} candidate(s) to {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
