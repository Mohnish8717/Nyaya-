"""Calendar delivery for a saved reminder.

A reminder used to be stored and never delivered, which is the same as not
having one: there is no notification service in this build, and telling a
caller "I will remind you" without one would be a lie.

A calendar file is the one delivery channel that needs no account, no push
service and no third party holding the caller's case details: the file is
generated here, saved by the caller, and their own phone does the reminding.
It is a reminder to act — never a calculation that a legal deadline falls on
that date, which the app cannot verify.
"""
from __future__ import annotations

import re
import time
import uuid
from datetime import date as _date
from typing import Any

PRODID = "-//Nyaya Setu//Legal reminder//EN"
MAX_LINE_OCTETS = 75


def _escape(value: str) -> str:
    """RFC 5545 TEXT escaping. A comma or newline in the note breaks the file."""
    text = " ".join(str(value or "").split())
    for old, new in (("\\", "\\\\"), (";", "\\;"), (",", "\\,")):
        text = text.replace(old, new)
    return text


def _fold(line: str) -> str:
    """Fold a content line at 75 octets without splitting a UTF-8 character."""
    encoded = line.encode("utf-8")
    if len(encoded) <= MAX_LINE_OCTETS:
        return line
    parts: list[str] = []
    current = b""
    for char in line:
        chunk = char.encode("utf-8")
        # A continuation line starts with one space, which counts toward the
        # octet budget, so the first line gets 75 and the rest get 74.
        budget = MAX_LINE_OCTETS if not parts else MAX_LINE_OCTETS - 1
        if len(current) + len(chunk) > budget:
            parts.append(current.decode("utf-8"))
            current = b""
        current += chunk
    if current:
        parts.append(current.decode("utf-8"))
    return "\r\n ".join(parts)


def _stamp(moment: float | None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(moment if moment is not None else time.time()))


def to_ics(item: dict[str, Any], *, now: float | None = None) -> str:
    """One all-day VEVENT for a stored reminder.

    `item["date"]` has already been validated as YYYY-MM-DD by the endpoint, so
    this function formats rather than parses.
    """
    raw = str(item.get("date") or "").strip()
    # Strict: `fromisoformat` alone is lenient in newer Pythons, and "14-03-2026"
    # would otherwise become a plausible-looking 8-digit string.
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        raise ValueError("A reminder needs a YYYY-MM-DD date.")
    try:
        day = _date.fromisoformat(raw).strftime("%Y%m%d")
    except ValueError as exc:
        raise ValueError("A reminder needs a real calendar date.") from exc
    note = str(item.get("note") or "").strip() or "Follow up on your legal matter"
    summary = f"Legal reminder: {note}"[:180]
    description = (
        f"{note}\n\n"
        "Set by the legal information assistant at your request. This is a reminder to act, "
        "not a deadline the assistant calculated, and not legal advice. Confirm any actual "
        "deadline with the court, the office or a lawyer."
    )
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:" + PRODID,
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{uuid.uuid4().hex}@nyaya-setu",
        f"DTSTAMP:{_stamp(now)}",
        f"DTSTART;VALUE=DATE:{day}",
        f"DTEND;VALUE=DATE:{day}",
        f"SUMMARY:{_escape(summary)}",
        f"DESCRIPTION:{_escape(description)}",
        "TRANSP:TRANSPARENT",
        "BEGIN:VALARM",
        "TRIGGER:-PT9H",
        "ACTION:DISPLAY",
        f"DESCRIPTION:{_escape(summary)}",
        "END:VALARM",
        "END:VEVENT",
        "END:VCALENDAR",
    ]
    return "\r\n".join(_fold(line) for line in lines) + "\r\n"
