"""Gnani Prisma (STT) and Timbre (TTS) protocol, verified against the vendor docs.

This module exists because the speech layer had drifted from what Gnani
actually documents, in ways that silently disabled the features this product
depends on:

  * The stream was opened with `lang_code` and `x-format` — both *legacy*
    spellings — and with **no `model` header at all**. A missing model is not an
    error: Prisma defaults to **v2.5**, which is single-language per session and
    has no language identification. So every call was pinned to whichever
    language the UI had guessed, and `detected_language_code` never arrived.
  * Language identification on the realtime stream is **`indic-auto`**, not
    `auto` (`auto` is the batch/REST value). It is a v3.0 feature and it reports
    the language per utterance.
  * Code-mixed speech is a **v3.0** capability, exposed through `format`:
    `codemix` keeps the mixed utterance as spoken, `transliterate` Romanises it.
  * Timbre's mixed-content path is `language: "auto"` ("auto-detect from
    script"), and `hi-en` is the dedicated Hinglish code. A v3.0 voice name sent
    with `timbre-v2.5` (or the reverse) is a hard 400, so the pairing has to be
    validated before synthesis, not discovered mid-sentence.
  * Prisma wants **exactly 1,024 bytes** of PCM16LE per binary frame, sent at
    real-time cadence. The browser worklet happens to produce exactly that, but
    a different capture path, a browser that batches, or the split point of a
    reconnect will not — so the proxy re-cuts the stream and pads the tail
    instead of closing the socket when a frame is the wrong size.

Everything here is pure: header and body construction, voice resolution, frame
re-chunking, text conditioning. No I/O, no globals that change at runtime.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import language as language_layer

# ---------------------------------------------------------------- Prisma (STT)

PRISMA_STREAM_URL = "wss://api.vachana.ai/stt/v3/stream"
PRISMA_MODELS: tuple[str, ...] = ("prisma-v3.0", "prisma-v2.5")
DEFAULT_PRISMA_MODEL = "prisma-v3.0"
STT_FORMATS: tuple[str, ...] = ("codemix", "transcribe", "verbatim", "transliterate")
SAMPLE_RATES: tuple[int, ...] = (8000, 16000, 44100, 48000)
# 512 samples of 16-bit mono. Prisma rejects any other frame size.
CHUNK_BYTES = 1024
STREAM_SESSION_SECONDS = 15 * 60

# Legal vocabulary the recogniser should prefer. Names, statutes and office
# words are what a caller repeats and what generic acoustic models flatten
# ("section 138" -> "section one thirty eight", "NALSA" -> "nalsa").
LEGAL_KEYTERMS: tuple[str, ...] = (
    "FIR", "पंचनामा", "chargesheet", "चार्जशीट", "summons", "समन", "warrant", "वारंट",
    "bail", "ज़मानत", "anticipatory bail", "maintenance", "भरण-पोषण", "alimony",
    "cheque bounce", "चेक बाउंस", "section 138", "धारा 138", "NI Act",
    "IPC", "BNS", "BNSS", "CrPC", "CPC", "Evidence Act", "BSA",
    "consumer forum", "उपभोक्ता फोरम", "District Consumer Commission",
    "RTI", "आरटीआई", "PIO", "First Appellate Authority",
    "NALSA", "DLSA", "legal aid", "कानूनी सहायता", "Lok Adalat", "लोक अदालत",
    "tehsildar", "तहसीलदार", "SDM", "DM", "collector", "sub-registrar",
    "salary", "वेतन", "tankha", "टंका", "PF", "ESI", "gratuity", "ग्रेच्युटी",
    "domestic violence", "घरेलू हिंसा", "protection order", "protection officer",
    "cyber crime", "साइबर क्राइम", "1930", "UPI", "OTP",
    "mutation", "दाखिल-खारिज", "registry", "रजिस्ट्री", "khata", "survey number",
    "tenant", "किरायेदार", "eviction", "बेदखली", "rent agreement",
    "pension", "पेंशन", "ration card", "राशन कार्ड", "Aadhaar", "आधार",
)


@dataclass
class SttSettings:
    """One Prisma connection's configuration.

    Header values are fixed for the life of a connection; only `language_code`,
    `format` and the substitution/bias fields can be changed mid-stream, which
    is why the whole set is built once here.
    """

    language_code: str
    candidates: list[str]
    model: str = DEFAULT_PRISMA_MODEL
    format: str = "codemix"
    sample_rate: int = 16000
    vad_threshold: float = 0.55
    min_silence_ms: int = 350
    min_speech_ms: int = 100
    min_segment_ms: int = 100
    idle_timeout_sec: int = 60
    keyterms: list[str] = field(default_factory=list)
    topic: str = ""
    previous_text: str = ""
    speaker_names: list[str] = field(default_factory=list)
    substitution: list[dict[str, Any]] = field(default_factory=list)

    def headers(self, api_key: str) -> dict[str, str]:
        """Modern documented header names only.

        The old spellings still work, but the new name wins when both are sent
        and the legacy set is what hid the missing model in the first place.
        """
        model = self.model if self.model in PRISMA_MODELS else DEFAULT_PRISMA_MODEL
        if model == "prisma-v2.5":
            # v2.5 cannot identify language or mix scripts. Sending indic-auto
            # or codemix to it is a wasted connection, so degrade honestly.
            if self.language_code not in {"en-IN", "hi-IN", "bn-IN", "te-IN", "ta-IN", "mr-IN", "gu-IN", "kn-IN", "ml-IN", "pa-IN"}:
                model = DEFAULT_PRISMA_MODEL
        headers: dict[str, str] = {
            "x-api-key-id": api_key,
            "model": model,
            "x-sample-rate": str(self.sample_rate if self.sample_rate in SAMPLE_RATES else 16000),
            "x-vad-threshold": f"{min(1.0, max(0.0, self.vad_threshold)):.2f}",
            "x-min-silence-ms": str(max(100, min(2000, self.min_silence_ms))),
            "x-min-speech-ms": str(max(50, min(2000, self.min_speech_ms))),
            "x-min-segment-ms": str(max(100, min(1000, self.min_segment_ms))),
            "x-idle-timeout-sec": str(max(5, min(900, self.idle_timeout_sec))),
        }
        if model == "prisma-v3.0":
            headers["language_code"] = self.language_code
            headers["format"] = self.format if self.format in STT_FORMATS else "codemix"
            if self.candidates and len(self.candidates) > 1:
                # candidate_languages is a secondary hint; language_code must remain a single valid code or indic-auto
                headers["candidate_languages"] = ",".join(self.candidates[:4])
            if self.keyterms:
                headers["keyterms"] = ",".join(self.keyterms[:200])
            if self.topic:
                headers["topic"] = self.topic[:1600]
            if self.previous_text:
                headers["previous_text"] = self.previous_text[-600:]
            if self.speaker_names:
                headers["speaker_names"] = ",".join(self.speaker_names[:4])
        else:
            headers["language_code"] = self.language_code if self.language_code in {
                "en-IN", "hi-IN", "bn-IN", "te-IN", "ta-IN", "mr-IN", "gu-IN", "kn-IN", "ml-IN", "pa-IN",
            } else "hi-IN"
            headers["format"] = "transcribe"
        if self.substitution:
            import json

            headers["enable_substitution"] = "true"
            headers["substitution_map"] = json.dumps(self.substitution[:10], ensure_ascii=False)
        return headers

    def as_public(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "language_code": self.language_code,
            "candidates": self.candidates,
            "format": self.format,
            "sample_rate": self.sample_rate,
            "min_silence_ms": self.min_silence_ms,
            "vad_threshold": self.vad_threshold,
            "keyterms": len(self.keyterms),
            "continuing_conversation": bool(self.previous_text),
        }


class FrameBuffer:
    """Cut a byte stream into Prisma's fixed 1,024-byte frames.

    The contract is per frame, not per buffer: whatever the capture path emits,
    the upstream socket only ever sees 512-sample frames. That removes the
    sample rate, the capture implementation and reconnect splits from the list
    of things that can break recognition.
    """

    def __init__(self, size: int = CHUNK_BYTES) -> None:
        self._size = size
        self._buffer = bytearray()

    def push(self, data: bytes) -> list[bytes]:
        self._buffer.extend(data)
        frames: list[bytes] = []
        while len(self._buffer) >= self._size:
            frames.append(bytes(self._buffer[: self._size]))
            del self._buffer[: self._size]
        return frames

    @property
    def buffered(self) -> int:
        """Bytes held back for the next frame. Zero means the stream is aligned."""
        return len(self._buffer)

    def flush(self) -> list[bytes]:
        """Pad the tail so the last partial frame is still a valid utterance
        end. Silence is the correct padding: it closes the segment."""
        if not self._buffer:
            return []
        tail = bytes(self._buffer) + b"\x00" * (self._size - len(self._buffer))
        self._buffer.clear()
        return [tail]


def transcript_of(message: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise a Prisma transcript frame.

    Realtime Prisma has no interim transcripts: every `transcript` frame is
    final and already endpointed by server VAD. That is why the client must not
    add its own long silence wait on top of a final transcript.
    """
    if not isinstance(message, dict) or message.get("type") != "transcript":
        return None
    text = message.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return {
        "type": "transcript",
        "text": text.strip(),
        "is_final": True,
        "detected_language_code": message.get("detected_language_code"),
        "segment_id": message.get("segment_id"),
        "segment_index": message.get("segment_index"),
        "audio_duration_ms": message.get("audio_duration_ms"),
        "latency_ms": message.get("latency"),
    }


# ----------------------------------------------------------------- Timbre (TTS)

TIMBRE_STREAM_URL = "wss://api.vachana.ai/api/v1/tts"
TIMBRE_MODELS: tuple[str, ...] = ("timbre-v3.0", "timbre-v2.5")
DEFAULT_TIMBRE_MODEL = "timbre-v3.0"
TIMBRE_LANGUAGES: tuple[str, ...] = (
    "auto", "en-IN", "hi-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN", "mr-IN", "pa-IN", "bn-IN", "gu-IN", "hi-en",
)
TTS_SAMPLE_RATES: tuple[int, ...] = (8000, 16000, 22050, 24000, 44100, 48000)

# Timbre v3.0: thirteen voices, each speaking all ten base languages.
V3_VOICES: tuple[str, ...] = (
    "Shivani", "Lakshmi", "Kalpana", "Chetana", "Vasudha", "Riyaa", "Tarika",
    "Avinash", "Shubham", "Prakash", "Devansh", "Yashraj", "Harshad",
)
# Timbre v2.5: one preferred language per voice, and names are not portable
# between models — sending a v3 name to v2.5 is a 400, not a fallback.
V25_VOICES: dict[str, tuple[str, ...]] = {
    "en-IN": ("Kaveri", "Trupti", "Devika", "Pranav", "Shlok", "Girish"),
    "hi-IN": ("Nalini", "Bhavna", "Yashvi", "Urmila", "Jwala", "Chitra", "Ambuja", "Deepak", "Roopesh", "Vikrant", "Hemraj", "Jalaj", "Omkar"),
    "hi-en": ("Poorvi",),
    "ta-IN": ("Asmita", "Trisha", "Brinda", "Vedika", "Noopur"),
    "te-IN": ("Suhana", "Lehara", "Lavanya", "Yukti", "Varuni"),
    "kn-IN": ("Saanvi", "Kavin"),
    "ml-IN": ("Reshma", "Riyaan"),
    "mr-IN": ("Zahira", "Ishaan"),
    "bn-IN": ("Kirra", "Dhruva"),
    "gu-IN": ("Falak", "Veera"),
    "pa-IN": ("Mehuli", "Zayan"),
}
_V25_ALL = {name for names in V25_VOICES.values() for name in names}

# Documented v3 audio tags. Anything else in angle brackets is stripped so the
# assistant cannot read "<coughs>" out loud.
TIMBRE_TAGS = frozenset({
    "coughs", "excited", "exhales", "giggle", "laugh", "pauses", "shouting",
    "sigh", "slow", "sniffles", "whispers", "yawns", "clears throat",
})
_TAG_RE = re.compile(r"<\s*([a-zA-Z ]{1,24})\s*>")


@dataclass
class SpeechChoice:
    model: str
    voice: str
    language: str
    substituted: str | None = None

    def as_public(self) -> dict[str, Any]:
        return {"model": self.model, "voice": self.voice, "language": self.language, "substituted": self.substituted}


def resolve_voice(model: str, voice: str, language: str) -> SpeechChoice:
    """Pick a (model, voice, language) triple that the provider will accept.

    A mis-paired voice is a 400 that arrives after the caller has already been
    told something is coming, so it is resolved here instead.
    """
    chosen_model = model if model in TIMBRE_MODELS else DEFAULT_TIMBRE_MODEL
    wanted = language if language in TIMBRE_LANGUAGES else "auto"

    if chosen_model == "timbre-v3.0":
        if voice in V3_VOICES:
            return SpeechChoice(chosen_model, voice, wanted)
        # A v2.5 name (or an unknown one) cannot be used with v3: fall back to
        # the default voice rather than failing the utterance.
        fallback = V3_VOICES[0]
        return SpeechChoice(chosen_model, fallback, wanted, substituted=f"voice '{voice}' is not a timbre-v3.0 voice; used '{fallback}'")

    # v2.5
    names = V25_VOICES.get(wanted if wanted != "auto" else "hi-IN", V25_VOICES["hi-IN"])
    if voice in names:
        return SpeechChoice(chosen_model, voice, wanted)
    fallback = names[0]
    return SpeechChoice(chosen_model, fallback, wanted, substituted=f"voice '{voice}' does not speak {wanted or 'this language'} on timbre-v2.5; used '{fallback}'")


def condition_text(text: str, model: str = DEFAULT_TIMBRE_MODEL) -> str:
    """Make text safe to synthesise.

    Two documented requirements: numbers and currency should reach the engine
    as spoken words, and audio tags are a v3 feature that v2.5 silently strips
    (tag-only text is even a 400). Angle-bracket content that is not a
    documented tag is removed rather than read aloud.
    """
    cleaned = str(text or "")
    cleaned = re.sub(r"₹\s*", " rupees ", cleaned)
    cleaned = re.sub(r"\bRs\.?\s*", " rupees ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)

    def keep(match: re.Match[str]) -> str:
        tag = " ".join(match.group(1).lower().split())
        if model == "timbre-v3.0" and tag in TIMBRE_TAGS:
            return f"<{tag}>"
        return " "

    cleaned = _TAG_RE.sub(keep, cleaned)
    return " ".join(cleaned.split()).strip()


def timbre_body(
    base: dict[str, Any] | None,
    *,
    model: str,
    voice: str,
    language: str,
    sample_rate: int = 24000,
    default_text: str = "",
) -> tuple[dict[str, Any], SpeechChoice]:
    """Build the synthesis request the server actually sends.

    The browser proposes a body; the server owns model, voice, language and
    audio format, because those are the fields whose mismatch is a hard failure
    and because they are configuration, not user input.
    """
    request = dict(base or {})
    choice = resolve_voice(model, voice, language)
    text = request.get("text") if isinstance(request.get("text"), str) else default_text
    rate = request.get("audio_config", {}).get("sample_rate") if isinstance(request.get("audio_config"), dict) else sample_rate
    rate = rate if isinstance(rate, int) and rate in TTS_SAMPLE_RATES else sample_rate
    try:
        speed = float(request.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0
    body = {
        "text": condition_text(text, choice.model),
        "voice": choice.voice,
        "model": choice.model,
        "language": choice.language,
        "speed": max(0.85, min(1.15, speed)),
        "audio_config": {
            "sample_rate": rate,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
            "container": "raw",
        },
    }
    return body, choice


def voices_for(model: str, language: str) -> list[str]:
    if model == "timbre-v2.5":
        return list(V25_VOICES.get(language if language != "auto" else "hi-IN", V25_VOICES["hi-IN"]))
    return list(V3_VOICES)


def capabilities() -> dict[str, Any]:
    """What the speech layer can honestly claim, for /api/config."""
    return {
        "stt": {
            "provider": "gnani-prisma",
            "stream_url": PRISMA_STREAM_URL,
            "models": list(PRISMA_MODELS),
            "formats": list(STT_FORMATS),
            "languages": list(language_layer.BASE_LANGUAGES),
            "auto_detect": True,
            "auto_detect_code": "indic-auto",
            "codemixed": True,
            "detected_language_per_utterance": True,
            "frame_bytes": CHUNK_BYTES,
            "session_cap_seconds": STREAM_SESSION_SECONDS,
            "keyterms": len(LEGAL_KEYTERMS),
        },
        "tts": {
            "provider": "gnani-timbre",
            "stream_url": TIMBRE_STREAM_URL,
            "models": list(TIMBRE_MODELS),
            "languages": list(TIMBRE_LANGUAGES),
            "auto_language": True,
            "voice_count": {"timbre-v3.0": len(V3_VOICES), "timbre-v2.5": len(_V25_ALL)},
        },
    }
