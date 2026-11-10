"""Real-time voice control: barge-in, endpointing and spoken truncation.

Reversed from the shipped `@vapi-ai/web` client and the assistant configuration
schema it sends, because the caller-facing behaviour is defined by those
configs, not by marketing copy:

  * StopSpeakingPlan: `numWords` (words the caller must say before the
    assistant yields), `voiceSeconds` (a VAD spike, used when `numWords` is 0),
    `backoffSeconds` (how long the assistant waits before trying again),
    `acknowledgementPhrases` (never interrupt) and `interruptionPhrases`
    (always interrupt, regardless of word count).
  * StartSpeakingPlan: `waitSeconds` (a floor, so the answer does not start in
    the first millisecond after the model returns) plus punctuation-aware
    endpointing: a completed sentence is treated as finished quickly, a
    trailing number is not (people read long numbers in chunks), and an
    unpunctuated trailing clause waits longest.

Two things in that design are worth naming because they are the difference
between "feels like a person" and "feels like a walkie-talkie":

  1. Backchannels are not interruptions. "haan", "okay", "hmm" while the
     assistant talks must not cut it off, or the caller cannot be listened to.
  2. A cut-off answer must be cut off in the *record* too. If the assistant
     stops mid-sentence but the transcript keeps the full paragraph, the next
     model turn answers a sentence the caller never heard.

Everything here is pure and deterministic; the server and the browser share
the same numbers, and the browser is what actually stops the audio.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

# Backchannels. Deliberately multilingual: a Hindi speaker saying "haan" or
# "theek hai" is acknowledging, not taking the floor.
DEFAULT_ACKNOWLEDGEMENTS: tuple[str, ...] = (
    "okay", "ok", "yeah", "yes", "right", "sure", "alright", "got it",
    "understood", "i see", "i understand", "mm-hmm", "mhmm", "uh-huh", "hmm",
    "haan", "han", "ha", "theek", "theek hai", "thik hai", "acha", "achha",
    "hmm hmm", "ji", "ji haan", "sahi", "ok bhai", "haan ji", "pata hai",
    "बिल्कुल", "हाँ", "हां", "ठीक", "ठीक है", "अच्छा", "जी", "जी हाँ",
    "సరే", "అవును", "சரி", "ஆமாம்", "ঠিক", "হ্যাঁ", "ಹೌದು", "ಸರಿ",
)

# Floor-taking phrases: they mean "stop talking", even as one word.
DEFAULT_INTERRUPTIONS: tuple[str, ...] = (
    "stop", "wait", "hold", "no", "not", "but", "actually", "enough", "quiet",
    "pause", "cancel", "listen", "ruko", "rukiye", "nahi", "nahin", "mat",
    "suno", "sunо", "ek minute", "ek second", "bas", "chhodo", "chodo",
    "रुको", "रुकिए", "नहीं", "सुनो", "बस", "छोड़ो", "एक मिनट",
    "ఆపు", "వద్దు", "நிறுத்து", "வேண்டாம்", "থামুন", "না", "ನಿಲ್ಲಿಸಿ", "ಬೇಡ",
)

SENTENCE_END = re.compile(r"[.!?।॥…]['\")\]]?\s*$")
NUMBER_TAIL = re.compile(r"(?:\d[\d,.\s]*|[०-९]+)\s*$")


@dataclass(frozen=True)
class StopSpeakingPlan:
    """When the assistant yields the floor."""

    num_words: int = 0
    voice_seconds: float = 0.2
    backoff_seconds: float = 0.8
    acknowledgement_phrases: tuple[str, ...] = DEFAULT_ACKNOWLEDGEMENTS
    interruption_phrases: tuple[str, ...] = DEFAULT_INTERRUPTIONS

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_words": self.num_words,
            "voice_seconds": self.voice_seconds,
            "backoff_seconds": self.backoff_seconds,
            "interruption_phrases": len(self.interruption_phrases),
            "acknowledgement_phrases": len(self.acknowledgement_phrases),
        }


@dataclass(frozen=True)
class StartSpeakingPlan:
    """How long the assistant waits before taking the floor, and how it decides
    the caller has finished."""

    wait_seconds: float = 0.4
    # Prisma realtime has no interim transcripts: a `transcript` frame is final
    # and only arrives after the server VAD closed the segment. Waiting the
    # full unfinished-clause delay again on top of that doubled the silence the
    # caller heard, so a final segment only gets a short merge window — long
    # enough to catch a second clause of the same breath, short enough to keep
    # the turn moving.
    on_final_segment_seconds: float = 0.35
    on_punctuation_seconds: float = 0.3
    on_no_punctuation_seconds: float = 1.2
    on_number_seconds: float = 0.6
    # Spoken words per second, used to estimate what the caller actually heard
    # when they cut the answer off. 150 wpm is a calm conversational pace.
    words_per_second: float = 2.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "wait_seconds": self.wait_seconds,
            "on_final_segment_seconds": self.on_final_segment_seconds,
            "on_punctuation_seconds": self.on_punctuation_seconds,
            "on_no_punctuation_seconds": self.on_no_punctuation_seconds,
            "on_number_seconds": self.on_number_seconds,
        }


STOP_PLAN = StopSpeakingPlan()
START_PLAN = StartSpeakingPlan()


@dataclass
class InterruptionDecision:
    stop: bool
    reason: str
    matched: str | None = None
    speak_after_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "stop": self.stop,
            "reason": self.reason,
            "matched": self.matched,
            "speak_after_ms": self.speak_after_ms,
        }


def _normalise(text: str) -> str:
    return re.sub(r"[^\w\u0900-\u097F\u0B80-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\s']", " ", (text or "").lower())


def _phrase_hit(text: str, phrases: tuple[str, ...]) -> str | None:
    cleaned = f" {_normalise(text).strip()} "
    if not cleaned.strip():
        return None
    for phrase in phrases:
        needle = f" {_normalise(phrase).strip()} "
        if needle.strip() and needle in cleaned:
            return phrase
    return None


def _word_count(text: str) -> int:
    return len([word for word in _normalise(text).split() if word])


def decide_interruption(
    partial_text: str,
    *,
    vad_ms: int = 0,
    elapsed_ms: int = 0,
    plan: StopSpeakingPlan = STOP_PLAN,
) -> InterruptionDecision:
    """Decide whether the caller has taken the floor.

    Order matters: a stop word beats everything, a backchannel beats the word
    count and the VAD spike, and only then do the configured thresholds apply.
    `elapsed_ms` is how long the assistant has been speaking; the caller can
    always interrupt, but a very short utterance at the very start is usually
    the tail of the previous turn, not a new one.
    """
    interruption = _phrase_hit(partial_text, plan.interruption_phrases)
    if interruption:
        return InterruptionDecision(True, "interruption_phrase", interruption)

    if _phrase_hit(partial_text, plan.acknowledgement_phrases):
        return InterruptionDecision(False, "acknowledgement")

    if plan.num_words > 0 and _word_count(partial_text) >= plan.num_words:
        return InterruptionDecision(True, "words", None)

    if plan.num_words == 0:
        required_ms = int(plan.voice_seconds * 1000)
        if vad_ms >= required_ms and vad_ms > 0:
            return InterruptionDecision(True, "voice_activity", None)

    return InterruptionDecision(False, "listening")


def resume_after_ms(plan: StopSpeakingPlan = STOP_PLAN) -> int:
    """Backoff before the assistant may speak again after being cut off."""
    return int(plan.backoff_seconds * 1000)


def endpoint_delay_ms(text: str, plan: StartSpeakingPlan = START_PLAN) -> int:
    """How long to wait before treating a transcript as a finished turn.

    Mirrors the punctuation-aware endpointing in the reversed schema: a
    completed sentence is accepted quickly, a trailing number is held (the
    caller is probably still reading it), and an unfinished clause waits
    longest. Devanagari danda counts as a sentence end.
    """
    stripped = (text or "").strip()
    if not stripped:
        return int(plan.on_no_punctuation_seconds * 1000)
    if NUMBER_TAIL.search(stripped):
        return int(plan.on_number_seconds * 1000)
    if SENTENCE_END.search(stripped):
        return int(plan.on_punctuation_seconds * 1000)
    return int(plan.on_no_punctuation_seconds * 1000)


def truncate_to_heard(text: str, heard_ms: int, plan: StartSpeakingPlan = START_PLAN) -> tuple[str, bool]:
    """Cut the assistant's message to roughly what the caller heard.

    Returns (text, truncated). The estimate is deliberately crude and marked as
    such in the transcript; it exists so the next turn does not answer a
    sentence the caller never heard, not to attribute exact words.
    """
    if not text:
        return "", False
    budget = int(max(0, heard_ms) / 1000 * plan.words_per_second)
    if budget <= 0:
        return "", True
    words = text.split()
    if budget >= len(words):
        return text, False
    kept = " ".join(words[:budget]).rstrip(",;:")
    return f"{kept}…", True


@dataclass
class SpeechTurn:
    """One assistant utterance, as the server knows it."""

    speech_id: str
    session_id: str
    text: str
    started_at: float = field(default_factory=time.time)
    cancelled_at: float | None = None
    reason: str | None = None
    heard_ms: int = 0
    spoken: str = ""
    truncated: bool = False
    cancelled: Any = None  # asyncio.Event, set when the turn must stop

    def as_dict(self) -> dict[str, Any]:
        return {
            "speech_id": self.speech_id,
            "text": self.spoken or self.text,
            "cancelled": self.cancelled_at is not None,
            "reason": self.reason,
            "heard_ms": self.heard_ms,
            "truncated": self.truncated,
            "elapsed_ms": round((time.time() - self.started_at) * 1000),
        }


class SpeechRegistry:
    """Tracks the one utterance a session can be speaking at a time.

    The record lives in the shared session store, so the worker that receives
    the barge-in does not have to be the worker streaming the audio. A local
    `asyncio.Event` still exists for the common case and fires instantly; a
    second worker's synthesis socket watches the row and stops within a poll
    interval (`WATCH_INTERVAL_SECONDS`) instead of speaking over the caller.
    """

    WATCH_INTERVAL_SECONDS = 0.15

    def __init__(self) -> None:
        self._events: dict[str, Any] = {}
        self._watchers: set[Any] = set()
        # The object this worker created for a session, kept so a caller that
        # already holds it sees the cancellation. The row stays authoritative.
        self._live: dict[str, SpeechTurn] = {}

    @staticmethod
    def _key(session_id: str, speech_id: str) -> str:
        return f"{session_id}:{speech_id}"

    @staticmethod
    def _store():
        # Imported lazily: voice is pure decision logic and must stay usable
        # without the session database in unit tests.
        from app import sessions

        return sessions

    def begin(self, session_id: str, text: str) -> SpeechTurn:
        import asyncio
        import secrets

        previous = self.current(session_id)
        if previous and previous.cancelled_at is None:
            updated = self.interrupt(session_id, heard_ms=0, reason="superseded")
            if updated is not None and updated is not previous:
                # `current()` rebuilds the turn from the shared row, so the
                # object the caller already holds has to be told what happened.
                previous.cancelled_at = updated.cancelled_at
                previous.reason = updated.reason
                previous.spoken = updated.spoken
                previous.truncated = updated.truncated
        # Unique across workers: a counter alone collides as soon as there are
        # two processes.
        speech_id = f"sp_{int(time.time() * 1000)}_{secrets.token_hex(3)}"
        turn = SpeechTurn(
            speech_id=speech_id,
            session_id=session_id,
            text=text,
            cancelled=asyncio.Event(),
        )
        self._store().save_speech(session_id, speech_id, text, turn.started_at)
        self._events[self._key(session_id, speech_id)] = turn.cancelled
        self._live[session_id] = turn
        return turn

    def current(self, session_id: str) -> SpeechTurn | None:
        row = self._store().load_speech(session_id)
        live = self._live.get(session_id)
        if live is not None and (row is None or row["speech_id"] == live.speech_id):
            if row is not None:
                live.cancelled_at = row["cancelled_at"]
                live.reason = row["reason"]
                live.heard_ms = row["heard_ms"]
                live.spoken = row["spoken"]
                live.truncated = row["truncated"]
            return live
        if not row:
            self._live.pop(session_id, None)
            return None
        turn = SpeechTurn(
            speech_id=row["speech_id"],
            session_id=session_id,
            text=row["text"],
            started_at=row["started_at"],
            cancelled_at=row["cancelled_at"],
            reason=row["reason"],
            heard_ms=row["heard_ms"],
            spoken=row["spoken"],
            truncated=row["truncated"],
        )
        turn.cancelled = self._events.get(self._key(session_id, row["speech_id"]))
        return turn

    def interrupt(self, session_id: str, *, heard_ms: int = 0, reason: str = "caller") -> SpeechTurn | None:
        turn = self.current(session_id)
        if not turn or turn.cancelled_at is not None:
            return turn
        heard = max(0, int(heard_ms))
        spoken, truncated = truncate_to_heard(turn.text, heard)
        self._store().interrupt_speech(
            session_id, turn.speech_id, reason=reason, heard_ms=heard, spoken=spoken, truncated=truncated,
        )
        turn.cancelled_at = time.time()
        turn.reason = reason
        turn.heard_ms = heard
        turn.spoken, turn.truncated = spoken, truncated
        event = self._events.get(self._key(session_id, turn.speech_id))
        if event is not None:
            event.set()
        return turn

    def finish(self, session_id: str, speech_id: str) -> None:
        turn = self.current(session_id)
        if turn and turn.speech_id == speech_id:
            self._store().clear_speech(session_id, speech_id)
            self._events.pop(self._key(session_id, speech_id), None)
            self._live.pop(session_id, None)

    def cancel_event(self, session_id: str, speech_id: str) -> Any:
        """An event that fires when this utterance is cancelled, wherever from."""
        import asyncio

        event = self._events.get(self._key(session_id, speech_id))
        if event is not None:
            return event
        event = asyncio.Event()
        row = self._store().load_speech(session_id)
        if not row or row["speech_id"] != speech_id:
            return event
        if row["cancelled_at"] is not None:
            event.set()
            return event
        self._events[self._key(session_id, speech_id)] = event
        watcher = asyncio.create_task(self._watch(session_id, speech_id, event))
        self._watchers.add(watcher)
        watcher.add_done_callback(self._watchers.discard)
        return event

    async def _watch(self, session_id: str, speech_id: str, event: Any) -> None:
        import asyncio

        while not event.is_set():
            await asyncio.sleep(self.WATCH_INTERVAL_SECONDS)
            row = await asyncio.to_thread(self._store().load_speech, session_id)
            if not row or row["speech_id"] != speech_id or row["cancelled_at"] is not None:
                event.set()

    def interrupted_message(self, session_id: str) -> dict[str, Any] | None:
        turn = self.current(session_id)
        if not turn or turn.cancelled_at is None:
            return None
        return {
            "role": "assistant",
            "content": turn.spoken,
            "interrupted": True,
            "reason": turn.reason,
            "truncated": turn.truncated,
        }


class Metrics:
    """Small in-process latency record, so speed is measured rather than claimed.

    Keeps the last 200 samples per stage. A voice product that cannot show its
    own first-audio and interruption latencies is guessing about the thing the
    caller notices most.
    """

    def __init__(self, size: int = 200) -> None:
        from collections import deque

        self._samples: dict[str, Any] = {}
        self._size = size
        self._deque = deque

    def record(self, stage: str, ms: float) -> None:
        series = self._samples.setdefault(stage, self._deque(maxlen=self._size))
        series.append(max(0.0, float(ms)))

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for stage, series in self._samples.items():
            values = sorted(series)
            if not values:
                continue
            out[stage] = {
                "n": len(values),
                "p50": round(values[len(values) // 2], 1),
                "p95": round(values[min(len(values) - 1, int(len(values) * 0.95))], 1),
                "max": round(values[-1], 1),
            }
        return out


REGISTRY = SpeechRegistry()
METRICS = Metrics()
