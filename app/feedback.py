"""Failure-to-rule loop: production failures become candidate eval fixtures.

The loop the harness wants is: a *production* error is recorded with enough
context to act on (never raw PII — everything passes `privacy.redact_deep`
before it is written), survives restarts in a bounded local JSONL store, and
can be folded into candidate behavioural fixtures a reviewer can turn into
permanent rules or eval cases in `evals/golden.jsonl`.

The module deliberately does the simplest honest version of that:

* `record(kind, **fields)` — one redacted, timestamped line per failure.
* `recent()` / `candidates()` — grouped, deduplicated summary for review.
* `scripts/feedback_candidates.py` — writes `evals/fixture_candidates.jsonl`.

Rotation keeps the file bounded. The store never holds caller text at full
length, and `record` never raises: if the failure store itself is broken, the
failure being recorded still reaches the normal log via the caller.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from . import privacy

ROOT = Path(__file__).resolve().parents[1]
FAILURES_PATH = ROOT / ".runtime" / "failures.jsonl"

MAX_LINES = 400
MAX_FIELD_TEXT = 400
KNOWN_KINDS = frozenset({
    "unhandled_error",
    "degraded_turn",
    "claim_rejected",
    "tool_over_budget",
})

_WORD_RE = re.compile(r"[a-z][a-z]+[a-z_]*")


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())[:MAX_FIELD_TEXT]


def _bounded(fields: dict[str, Any]) -> dict[str, Any]:
    """Short, flat, shallow copy — long text and nests get truncated."""
    bounded: dict[str, Any] = {}
    for index, (key, value) in enumerate(fields.items()):
        if index >= 10:
            break
        if value is None:
            continue
        if isinstance(value, str):
            bounded[key] = _text(value)
        elif isinstance(value, (bool, int, float)):
            bounded[key] = value
        else:
            bounded[key] = _text(json.dumps(value, default=str, ensure_ascii=False))
    return bounded


def signature(text: str) -> str:
    """A stable, coarse fingerprint: the four longest words in order."""
    words = (match.group(0) for match in _WORD_RE.finditer(str(text or "").casefold()))
    ranked = sorted(dict.fromkeys(words), key=len, reverse=True)[:4]
    return "+".join(ranked) or "none"


def _read_lines() -> list[str]:
    if not FAILURES_PATH.exists():
        return []
    try:
        return FAILURES_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def record(kind: str, note: str = "", **fields: Any) -> None:
    """Append one redacted failure record. Never raises."""
    entry: dict[str, Any] = {
        "kind": str(kind)[:64],
        "signature": signature(note or " ".join(fields)),
        "note": _text(note),
        "at": round(time.time(), 3),
        "fields": privacy.redact_deep(_bounded(fields)),
    }
    lines = _read_lines()
    lines.append(json.dumps(entry, ensure_ascii=False))
    try:
        FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        FAILURES_PATH.write_text(
            "\n".join(lines[-MAX_LINES:]) + "\n", encoding="utf-8"
        )
    except OSError:
        pass  # from-master failure recording must not break the request


def recent(limit: int = 50) -> list[dict[str, Any]]:
    """Newest records first, up to `limit`."""
    rows: list[dict[str, Any]] = []
    for line in _read_lines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(rows[-limit:]))


def candidates() -> list[dict[str, Any]]:
    """Group the failure store into deduplicated candidates for review.

    A candidate is (kind, signature) with the most severe frequency and the
    fields of its latest occurrence. This is what a maintainer turns into a
    rule or an eval fixture — the script writes these to
    `evals/fixture_candidates.jsonl`.
    """
    rows = list(_read_lines())[-MAX_LINES:]
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for line in rows:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (str(row.get("kind")), str(row.get("signature") or "none"))
        group = groups.setdefault(key, {
            "kind": key[0],
            "signature": key[1],
            "occurrences": 0,
            "first_seen": row.get("at"),
            "last_seen": row.get("at"),
            "latest_fields": row.get("fields") or {},
        })
        group["occurrences"] += 1
        group["last_seen"] = row.get("at")
        group["latest_fields"] = row.get("fields") or {}
    ranked = sorted(groups.values(), key=lambda item: (-int(item["occurrences"]), str(item["last_seen"])))
    return ranked


def stats() -> dict[str, Any]:
    counts = Counter()
    for line in _read_lines():
        try:
            counts[str(json.loads(line).get("kind"))] += 1
        except (json.JSONDecodeError, AttributeError):
            continue
    return {"failures": sum(counts.values()), "by_kind": dict(counts)}
