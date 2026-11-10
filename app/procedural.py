"""Curated procedural navigation.

This is the layer that answers "how do I actually do this" without generating
anything. It is deterministic, reviewable, and versioned: the steps come from
a reviewed file, not from a model. The model may translate or explain a
procedure, but it cannot add, drop or reorder a step.

Records carry their own review status. A `draft` record is shown only with a
confirmation banner and the official link; the app never presents an
unreviewed procedure as settled.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "data" / "procedural_kb.json"

REQUIRED = {"id", "label", "summary", "jurisdiction", "urgency", "review_status", "source", "signals", "steps"}
REVIEW_STATES = {"approved", "draft", "expired", "rejected"}


def _validate(kb: Any) -> dict[str, Any]:
    if not isinstance(kb, dict) or not isinstance(kb.get("procedures"), list):
        raise ValueError("Procedural knowledge base must be an object with a procedures array.")
    seen: set[str] = set()
    for index, procedure in enumerate(kb["procedures"]):
        if not isinstance(procedure, dict):
            raise ValueError(f"Procedure {index} must be an object.")
        missing = REQUIRED - procedure.keys()
        if missing:
            raise ValueError(f"Procedure {index} is missing: {', '.join(sorted(missing))}")
        proc_id = str(procedure["id"])
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", proc_id):
            raise ValueError(f"Procedure {index} has an invalid id.")
        if proc_id in seen:
            raise ValueError(f"Duplicate procedure id: {proc_id}")
        seen.add(proc_id)
        if procedure["review_status"] not in REVIEW_STATES:
            raise ValueError(f"Procedure {proc_id} has an invalid review_status.")
        source = procedure.get("source")
        if not isinstance(source, dict) or not str(source.get("url", "")).startswith("https://"):
            raise ValueError(f"Procedure {proc_id} needs an https source url.")
        if not isinstance(procedure["signals"], list) or not procedure["signals"]:
            raise ValueError(f"Procedure {proc_id} needs trigger signals.")
        steps = procedure["steps"]
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"Procedure {proc_id} needs at least one step.")
        for step_index, step in enumerate(steps):
            if not isinstance(step, dict) or not str(step.get("action", "")).strip():
                raise ValueError(f"Procedure {proc_id} step {step_index} needs an action.")
    return kb


def load(path: Path = KB_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return _validate(json.load(handle))


KB = load()
PROCEDURES: list[dict[str, Any]] = KB["procedures"]
BY_ID = {procedure["id"]: procedure for procedure in PROCEDURES}


def _signal_score(signal: str, text: str) -> float:
    signal = signal.strip().lower()
    if not signal:
        return 0.0
    # Word-boundary match so "fir" does not fire inside "first".
    pattern = r"(?<![a-z0-9])" + re.escape(signal) + r"(?![a-z0-9])"
    if not re.search(pattern, text):
        return 0.0
    return 3.0 if " " in signal else 2.0


def best_match(text: str, jurisdiction: str = "auto") -> dict[str, Any] | None:
    """Return the best procedure for this text, or None.

    Threshold is deliberately conservative: a wrong procedure is worse than
    asking one more question, because the steps look authoritative.
    """
    lowered = (text or "").lower()
    best: dict[str, Any] | None = None
    best_score = 0.0
    for procedure in PROCEDURES:
        if procedure["review_status"] in {"rejected", "expired"}:
            continue
        scope = procedure.get("jurisdiction", "*")
        if scope != "*" and jurisdiction not in {"auto", scope}:
            continue
        hits = [signal for signal in procedure["signals"] if _signal_score(signal, lowered)]
        score = sum(_signal_score(signal, lowered) for signal in hits)
        if score > best_score:
            best_score = score
            best = {**procedure, "score": score, "matched_signals": hits[:6]}
    if best and best_score >= 2.0:
        return best
    return None


def public(procedure: dict[str, Any]) -> dict[str, Any]:
    """Shape a procedure for the API, keeping review status visible."""
    return {
        "id": procedure["id"],
        "label": procedure["label"],
        "summary": procedure["summary"],
        "jurisdiction": procedure["jurisdiction"],
        "urgency": procedure["urgency"],
        "review_status": procedure["review_status"],
        "checked_at": procedure.get("checked_at"),
        "source": procedure["source"],
        "steps": [
            {
                "order": index + 1,
                "action": step.get("action", ""),
                "detail": step.get("detail", ""),
                "where": step.get("where", ""),
                "bring": step.get("bring", []),
            }
            for index, step in enumerate(procedure["steps"])
        ],
        "if_refused": procedure.get("if_refused"),
        "handoff": procedure.get("handoff"),
        "verify_at_source": procedure.get("verify_at_source", []),
        "reviewed": procedure["review_status"] == "approved",
    }


def list_public() -> list[dict[str, Any]]:
    return [public(procedure) for procedure in PROCEDURES if procedure["review_status"] in {"approved", "draft"}]


def status() -> dict[str, Any]:
    counts: dict[str, int] = {}
    for procedure in PROCEDURES:
        counts[procedure["review_status"]] = counts.get(procedure["review_status"], 0) + 1
    return {"version": KB.get("version"), "total": len(PROCEDURES), "by_review_status": counts}
