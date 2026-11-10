"""A reminder that is only stored is not a reminder.

There is no notification service here, so the reminder is delivered as a
calendar file the caller saves. These tests pin the file's validity, its
escaping and its honesty: it is a reminder to act, never a calculated deadline.
"""
from __future__ import annotations

import pytest

from app import reminders


def test_a_reminder_becomes_a_valid_all_day_event() -> None:
    body = reminders.to_ics({"date": "2026-03-14", "note": "File the written complaint"})
    assert body.startswith("BEGIN:VCALENDAR\r\n")
    assert body.rstrip().endswith("END:VCALENDAR")
    assert "DTSTART;VALUE=DATE:20260314" in body
    assert "SUMMARY:Legal reminder: File the written complaint" in body
    assert body.count("BEGIN:VEVENT") == 1
    assert body.count("END:VEVENT") == 1


def test_the_file_says_it_is_not_a_calculated_deadline() -> None:
    body = reminders.to_ics({"date": "2026-03-14", "note": "Hearing"})
    assert "not a deadline" in body.lower()


def test_commas_and_newlines_are_escaped() -> None:
    body = reminders.to_ics({"date": "2026-03-14", "note": "Call Mr Rao, then\nfile it"})
    assert "Call Mr Rao\\, then file it" in body
    # An unescaped comma would split the property value.
    assert "Call Mr Rao, then" not in body


def test_long_lines_are_folded_on_octets_not_characters() -> None:
    body = reminders.to_ics({"date": "2026-03-14", "note": "విచారణ " * 20})
    for line in body.split("\r\n"):
        assert len(line.encode("utf-8")) <= 75, line
    assert body.count("\r\n ") >= 1


def test_a_bad_date_is_refused_rather_than_guessed() -> None:
    for bad in ({"date": "14-03-2026"}, {"date": ""}, {"date": "2026-3-4"}):
        with pytest.raises(ValueError):
            reminders.to_ics({**bad, "note": "x"})


def test_the_endpoint_serves_the_file_and_requires_the_token(client_session) -> None:
    client, session_id = client_session
    saved = client.post("/api/action/confirm", json={
        "session_id": session_id, "action": "create_reminder",
        "payload": {"date": "2026-03-14", "note": "File the complaint"},
        "user_confirmed": True,
    })
    assert saved.status_code == 200, saved.text
    data = saved.json()
    assert data["delivery"].startswith("Save the calendar file")
    assert "BEGIN:VCALENDAR" in data["calendar"]

    response = client.get(f"/api/session/{session_id}/reminders/0.ics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/calendar")
    assert "attachment" in response.headers["content-disposition"]

    missing = client.get(f"/api/session/{session_id}/reminders/5.ics")
    assert missing.status_code == 404
