"""Identifiers must never survive into logs, a model prompt, or an unapproved handoff."""
from __future__ import annotations

import asyncio

from app import legal_engine, privacy, understanding


def test_aadhaar_pan_phone_and_email_are_masked() -> None:
    text = "My Aadhaar is 4321 8765 2109, PAN ABCDE1234F, phone 9876543210, mail ravi@example.com."
    masked = privacy.redact(text)
    assert "4321" not in masked
    assert "ABCDE1234F" not in masked
    assert "9876543210" not in masked
    assert "ravi@example.com" not in masked
    assert "[aadhaar" in masked and "[pan" in masked


def test_the_model_only_ever_receives_the_masked_message(monkeypatch) -> None:
    """Redaction used to be a log-surface concern; the prompt was intact."""
    seen: dict[str, str] = {}

    async def fake_generate(context, **kwargs):
        # The consultation is the only model-facing surface: it must carry the
        # masked message and masked history, never the digits the caller typed.
        seen["text"] = context["caller_message"]
        seen["history"] = "".join(
            item.get("content", "") for item in context.get("history") or []
        )
        return {"reply": "I have noted the wages. Which months are unpaid?", "language": "en-IN", "legal_claims": []}

    monkeypatch.setattr(legal_engine, "generate_turn", fake_generate)
    monkeypatch.setattr(legal_engine, "model_configured", lambda: True)
    asyncio.run(legal_engine.answer_turn(
        "My Aadhaar is 4321 8765 2109 and my employer has not paid 40000 rupees",
        "en-IN",
        "auto",
        None,
        [{"role": "user", "content": "earlier I gave account 123456789012"}],
        {"turns": 1, "facts": {}, "last_state": {}},
    ))

    assert "4321 8765 2109" not in seen["text"]
    assert "4321" not in seen["text"]
    assert "[aadhaar" in seen["text"]
    # The amount is not an identifier and must survive: the answer needs it.
    assert "40000" in seen["text"]
    assert "123456789012" not in seen["history"]


def test_last_four_digits_remain_for_identification() -> None:
    masked = privacy.redact("account 123456789012")
    assert masked.endswith("9012]")
    assert "12345678" not in masked


def test_find_reports_kinds_without_leaking_values() -> None:
    found = privacy.find("Call 9876543210 about PAN ABCDE1234F")
    kinds = {item["kind"] for item in found}
    assert kinds == {"phone", "pan"}
    assert all("9876543210" not in item["masked"] for item in found)


def test_nested_log_fields_are_redacted() -> None:
    payload = {"note": "aadhaar 4321 8765 2109", "list": ["phone 9876543210"], "ok": 1}
    clean = privacy.redact_deep(payload)
    assert "4321" not in str(clean)
    assert "9876543210" not in str(clean)
    assert clean["ok"] == 1


def test_share_preview_masks_by_default_and_can_include_on_request() -> None:
    summary = "Payee aadhaar 4321 8765 2109"
    masked = privacy.share_preview(summary, include_identifiers=False)
    included = privacy.share_preview(summary, include_identifiers=True)
    assert masked["masked"] is True
    assert "4321" not in masked["body"]
    assert masked["notice"]
    assert included["masked"] is False
    assert "4321 8765 2109" in included["body"]


def test_fingerprint_is_stable_and_not_the_text() -> None:
    first = privacy.fingerprint("hello")
    assert first == privacy.fingerprint("hello")
    assert first != privacy.fingerprint("hello!")
    assert "hello" not in first
