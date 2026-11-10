"""Barge-in and endpointing contracts.

These are the behaviours a caller notices in the first two seconds: does it
stop when I speak, does it ignore my "haan", and does it wait for me to finish
a number instead of jumping in.
"""
from __future__ import annotations

import pytest

from app import sessions, voice


def test_acknowledgement_never_takes_the_floor():
    for phrase in ("okay", "haan", "theek hai", "hmm", "हाँ", "ji"):
        decision = voice.decide_interruption(phrase, vad_ms=600)
        assert decision.stop is False, phrase
        assert decision.reason == "acknowledgement"


def test_stop_words_interrupt_immediately_regardless_of_thresholds():
    plan = voice.StopSpeakingPlan(num_words=5)
    for phrase in ("no wait", "ruko", "नहीं", "stop"):
        decision = voice.decide_interruption(phrase, vad_ms=0, plan=plan)
        assert decision.stop is True, phrase
        assert decision.reason == "interruption_phrase"


def test_word_threshold_interrupts_after_enough_words():
    plan = voice.StopSpeakingPlan(num_words=2)
    assert voice.decide_interruption("my employer", plan=plan).reason == "words"
    assert voice.decide_interruption("my", plan=plan).stop is False


def test_vad_spike_interrupts_when_word_count_is_zero():
    plan = voice.StopSpeakingPlan(num_words=0, voice_seconds=0.2)
    assert voice.decide_interruption("", vad_ms=120, plan=plan).stop is False
    assert voice.decide_interruption("", vad_ms=240, plan=plan).reason == "voice_activity"


def test_a_sentence_ends_faster_than_an_unfinished_clause():
    fast = voice.endpoint_delay_ms("I was not paid for March.")
    devanagari = voice.endpoint_delay_ms("मुझे मार्च का वेतन नहीं मिला।")
    slow = voice.endpoint_delay_ms("I was not paid for")
    assert fast < slow
    assert devanagari == fast


def test_a_trailing_number_is_held_for_the_caller_to_finish():
    number = voice.endpoint_delay_ms("my account number is 4 5 6 7")
    assert number == int(voice.START_PLAN.on_number_seconds * 1000)
    assert number < voice.endpoint_delay_ms("my account number is")


def test_interrupted_answer_is_truncated_to_what_was_heard():
    text = "Under Section 138 of the Negotiable Instruments Act, a dishonoured cheque can support a complaint if the notice is sent within thirty days."
    heard, truncated = voice.truncate_to_heard(text, heard_ms=2000)
    assert truncated is True
    assert heard.endswith("…")
    assert len(heard.split()) < len(text.split())
    assert heard.startswith("Under Section 138")

    full, truncated_full = voice.truncate_to_heard(text, heard_ms=60000)
    assert truncated_full is False
    assert full == text


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Speech rows are shared state; each test gets its own database."""
    monkeypatch.setattr(sessions, "DB_PATH", tmp_path / "sessions.sqlite3")
    sessions.init()


def test_cancelling_a_speech_turn_marks_it_and_sets_the_event():
    registry = voice.SpeechRegistry()
    turn = registry.begin("session-1", "Here are the three steps you need to take.")
    registry.interrupt("session-1", heard_ms=900, reason="caller_spoke")

    stopped = registry.current("session-1")
    assert stopped is not None
    assert stopped.cancelled_at is not None
    assert stopped.reason == "caller_spoke"
    assert turn.cancelled.is_set()
    assert registry.interrupted_message("session-1")["truncated"] is True
    assert registry.interrupted_message("session-1")["interrupted"] is True


def test_a_new_speech_supersedes_the_previous_one():
    registry = voice.SpeechRegistry()
    first = registry.begin("session-2", "One long answer that is still playing.")
    second = registry.begin("session-2", "A shorter follow-up.")
    assert first.reason == "superseded"
    assert second.speech_id != first.speech_id
    assert registry.current("session-2").text == "A shorter follow-up."


def test_a_barge_in_reaches_a_worker_that_does_not_own_the_speech():
    """The audio may be streaming from another process than the interrupt."""
    import asyncio

    owner = voice.SpeechRegistry()
    other = voice.SpeechRegistry()
    turn = owner.begin("session-cross", "A long answer that is playing.")

    async def scenario():
        event = other.cancel_event("session-cross", turn.speech_id)
        owner.interrupt("session-cross", heard_ms=400, reason="caller_spoke")
        await asyncio.wait_for(event.wait(), timeout=3)
        return True

    assert asyncio.run(scenario()) is True
    assert other.current("session-cross").cancelled_at is not None


def test_backoff_is_reported_in_milliseconds():
    assert voice.resume_after_ms(voice.StopSpeakingPlan(backoff_seconds=0.8)) == 800
