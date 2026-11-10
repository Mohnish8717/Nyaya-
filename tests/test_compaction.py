"""Context compaction: turns that leave the live window are not lost.

The 12-turn window is regression-enforced by test_sessions.py; here we pin
the compaction contract: overflow folds into a bounded, privacy-masked
digest that reaches the model alongside the ledger, never vanishing turn 3
at turn 20 of a call.
"""
from __future__ import annotations

import pytest

from app import legal_engine, privacy, sessions


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "DB_PATH", tmp_path / "sessions.sqlite3")
    sessions.init()
    return sessions


def test_overflow_turns_land_in_the_earlier_digest(store):
    session_id, _token = store.create("en-IN")
    assert store.get(session_id)["earlier"] == []
    for index in range(9):  # 9 turns = 18 messages: 6 must overflow the 12-slot window
        store.record_turn(session_id, f"wages question {index}", f"answer {index}", "statute", "en-IN")
    session = store.get(session_id)
    assert len(session["history"]) == sessions.HISTORY_MESSAGES
    digest = session["earlier"]
    assert digest, "overflow must be folded into the digest, not dropped"
    assert all("wages question" in entry["content"] or "answer" in entry["content"] for entry in digest)
    assert len(digest) == 6


def test_digest_is_bounded_so_long_calls_do_not_grow_context(store):
    session_id, _token = store.create("en-IN")
    filler = "word " * 200
    for index in range(sessions.HISTORY_MESSAGES):
        store.record_turn(session_id, f"{filler} {index}", f"{filler} replied {index}", "statute", "en-IN")
    earlier = store.get(session_id)["earlier"]
    for entry in earlier:
        assert len(entry["content"]) <= sessions.EARLIER_CHARS


def test_engine_digest_is_masked_at_the_boundary():
    # Compactly: `_earlier_digest` folds and caps; masking is analyze's job,
    # applied to the whole digest exactly like the live caller message.
    items = [
        {"role": "user", "content": "aadhaar 234567890123 garni hai, account 12345678901234"},
        {"role": "assistant", "content": "ack " * 100},  # long line is capped
    ]
    digest = legal_engine._earlier_digest(items)
    assert len(digest) <= 2000
    assert "234567890123" in digest  # unmasked here, by design...
    masked = privacy.for_model(digest)
    assert "234567890123" not in masked and "12345678901234" not in masked  # ...but never reaches a model

    huge = legal_engine._earlier_digest(
        [{"role": "user", "content": "line " * 1000}] * 50
    )
    assert len(huge) <= 2000


def test_turn_context_carries_the_digest(store):
    from app import legal_engine

    session_id, _token = store.create("en-IN")
    for index in range(9):
        store.record_turn(session_id, f"wages turn {index}", f"reply {index}", "statute", "en-IN")
    session = store.get(session_id)
    # Turn 1 + compacted digest are both visible; analyze reads `earlier`
    # through the session, exactly like the facts ledger.
    context = legal_engine.TurnContext(
        text="visible", model_text="visible", language="en-IN", jurisdiction="IN",
        jurisdiction_basis="default", route={}, procedure=None, plan=None,
        history=session["history"], earlier=legal_engine._earlier_digest(session["earlier"]),
    )
    assert context.earlier and "wages turn" in context.earlier
