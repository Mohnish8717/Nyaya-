"""Privacy boundary for a legal-aid product.

The caller's message routinely contains the most sensitive identifiers in
Indian life: Aadhaar, PAN, phone, bank account, UPI handle. Four rules follow:

  1. logs and traces never contain them;
  2. the review handoff shows what will be shared, with identifiers masked
     until the caller chooses otherwise;
  3. a draft warns when it contains an identifier;
  4. nothing is *silently* changed — masking is always labelled.

`redact` is deliberately conservative: it masks token shapes, not names or
addresses, because over-redaction breaks the very summary a reviewer needs.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

PATTERNS: tuple[tuple[str, str], ...] = (
    ("aadhaar", r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"),
    ("pan", r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    ("email", r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    ("phone", r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b"),
    ("account", r"\b\d{11,18}\b"),
    ("iframe", r"\b\d{2}\s?\d{4}\s?\d{7}\b"),  # driving licence
    ("passport", r"\b[A-PR-WY][1-9]\d{2}\s?\d{4}\b"),
)

_COMPILED = [(name, re.compile(pattern)) for name, pattern in PATTERNS]

# Order matters: an Aadhaar number also matches the generic long-digit rule,
# so the most specific identifiers are applied first.
_PRIORITY = ("aadhaar", "pan", "passport", "iframe", "email", "phone", "account")


def find(text: str) -> list[dict[str, str]]:
    """Return the identifier kinds present, with masked previews."""
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for name in _PRIORITY:
        pattern = dict(_COMPILED)[name]
        for match in pattern.finditer(text or ""):
            raw = match.group(0)
            key = f"{name}:{raw}"
            if key in seen:
                continue
            seen.add(key)
            found.append({"kind": name, "masked": mask(raw)})
    return found


def mask(value: str) -> str:
    cleaned = value.replace(" ", "")
    if len(cleaned) <= 4:
        return "•" * len(cleaned)
    return "•" * (len(cleaned) - 4) + cleaned[-4:]


def redact(text: str) -> str:
    """Mask identifiers, keeping the surrounding sentence intact."""
    out = text or ""
    for name in _PRIORITY:
        pattern = dict(_COMPILED)[name]
        out = pattern.sub(lambda m: f"[{name} {mask(m.group(0))}]", out)
    return out


def for_model(text: str) -> str:
    """What a language model is allowed to see of the caller's message.

    Redaction used to be a log-surface concern: the transcript was masked in
    the logs and sent to the model intact. A model does not need an Aadhaar
    number, an account number or a phone number to answer a legal question, so
    it is not given one. The last four digits stay, which is enough for the
    caller to recognise which document is being discussed.
    """
    return redact(text)


def redact_deep(value: Any) -> Any:
    """Redact every string in a nested structure (for logs and traces)."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_deep(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_deep(v) for v in value]
    return value


def fingerprint(text: str) -> str:
    """Stable, non-reversible turn fingerprint for latency/quality joins."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def share_preview(summary: str, include_identifiers: bool) -> dict[str, Any]:
    """What a human reviewer would actually receive."""
    identifiers = find(summary)
    body = summary if include_identifiers else redact(summary)
    return {
        "body": body,
        "identifiers": identifiers,
        "masked": bool(identifiers) and not include_identifiers,
        "notice": (
            "This summary contains identity numbers. They are masked unless you choose to share them."
            if identifiers and not include_identifiers
            else None
        ),
    }
