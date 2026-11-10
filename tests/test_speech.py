"""Contracts for language identity and the Gnani speech protocol.

These tests do not call Gnani. They pin the two things that decide whether a
caller has to answer "which language?" and whether the stream is even
configured in a way that can report a language at all:

  * what evidence decides the language, in what order;
  * the exact headers and request bodies Gnani documents.

The second half matters because a missing `model` header is not an error at the
provider: it silently downgrades the stream to the single-language model. A
test is the only thing that will notice.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from app import language, legal_engine, speech


# --------------------------------------------------------------- language


@pytest.mark.parametrize(
    "text,expected",
    [
        ("मेरा मालिक वेतन नहीं दे रहा", "hi-IN"),
        ("నా జీతం ఇవ్వడం లేదు", "te-IN"),
        ("என் சம்பளம் தரவில்லை", "ta-IN"),
        ("আমার বেতন দেয়নি", "bn-IN"),
        ("ನನ್ನ ಸಂಬಳ ಕೊಟ್ಟಿಲ್ಲ", "kn-IN"),
        ("എന്റെ ശമ്പളം തന്നിട്ടില്ല", "ml-IN"),
        ("મારો પગાર આપ્યો નથી", "gu-IN"),
        ("ਮੇਰੀ ਤਨਖਾਹ ਨਹੀਂ ਦਿੱਤੀ", "pa-IN"),
    ],
)
def test_each_script_resolves_to_its_language(text, expected):
    assert language.resolve(text)[0] == expected


def test_script_wins_over_a_stale_english_default():
    assert language.resolve("मालिक पैसे नहीं दे रहा")[1] == "script"


def test_romanised_hindi_is_hinglish_not_english():
    code, basis = language.resolve("mera maalik tankha nahi de raha, kya karun")
    assert code == "hi-en"
    assert basis == "romanised_hindi"


def test_plain_english_stays_english():
    assert language.resolve("my employer has not paid my salary")[0] == "en-IN"


def test_one_hindi_particle_inside_english_does_not_flip_the_language():
    # "hai" alone is not enough evidence, and a false positive here would give
    # an English-speaking caller a Hindi voice.
    assert language.resolve("the notice says he has to reply")[0] == "en-IN"


def test_a_caller_choice_beats_every_heuristic():
    code, basis = language.resolve("मेरा वेतन नहीं मिला", override="en-IN")
    assert (code, basis) == ("en-IN", "caller_selected")


def test_the_recogniser_beats_the_script():
    # Prisma heard Tamil even though the transcript came back in Devanagari.
    code, basis = language.resolve("मेरा वेतन", stt_code="ta-IN")
    assert (code, basis) == ("ta-IN", "speech_recognition")


def test_provider_auto_codes_are_treated_as_auto():
    assert language.normalize("indic-auto") == "auto"
    assert language.normalize("_") == "auto"
    assert language.normalize("Hindi") == "hi-IN"
    assert language.normalize("hi_IN") == "hi-IN"


def test_auto_never_locks_the_recogniser_to_two_languages():
    # `indic-auto` is what reports the language. Passing a candidate list
    # instead would silently pin every Telugu speaker to Hindi or English.
    code, candidates = language.stt_language("auto")
    assert code == language.PRISMA_AUTO
    assert candidates == []


def test_an_explicit_choice_is_sent_with_hints():
    code, candidates = language.stt_language("ta-IN")
    assert code == "ta-IN"
    assert candidates[0] == "ta-IN"
    assert "en-IN" in candidates


def test_hinglish_override_maps_onto_prisma_auto():
    code, candidates = language.stt_language("hi-en")
    assert code == language.PRISMA_AUTO
    assert candidates == []


def test_tts_mixed_content_uses_auto():
    assert language.tts_language("hi-en") == "auto"
    assert language.tts_language("auto") == "auto"
    assert language.tts_language("hi-IN") == "hi-IN"


def test_a_text_only_language_is_reported_as_text_only():
    # Odia has no Prisma/Timbre path in this build. `auto` asked Timbre to
    # voice Odia text with whatever it detected, which reads the caller's own
    # question back in another language. The honest answer is "unavailable",
    # and the client shows the text.
    assert language.tts_language("or-IN") == "unavailable"
    assert language.is_speakable("or-IN") is False


# ----------------------------------------------------------------- Prisma


def test_prisma_headers_use_documented_names_and_select_v3():
    settings = speech.SttSettings(language_code="indic-auto", candidates=[])
    headers = settings.headers("key")
    assert headers["model"] == "prisma-v3.0"
    assert headers["language_code"] == "indic-auto"
    assert headers["format"] == "codemix"
    # The legacy spellings are what hid the missing model before.
    assert "lang_code" not in headers
    assert "x-format" not in headers


def test_v25_cannot_be_asked_for_auto_and_is_upgraded():
    settings = speech.SttSettings(language_code="indic-auto", candidates=[], model="prisma-v2.5")
    headers = settings.headers("key")
    assert headers["model"] == "prisma-v3.0"


def test_v25_stays_v25_when_a_single_language_is_chosen():
    settings = speech.SttSettings(language_code="hi-IN", candidates=[], model="prisma-v2.5")
    headers = settings.headers("key")
    assert headers["model"] == "prisma-v2.5"
    assert headers["format"] == "transcribe"
    assert "keyterms" not in headers


def test_legal_keyterms_and_context_are_sent():
    settings = speech.SttSettings(
        language_code="indic-auto", candidates=[],
        keyterms=list(speech.LEGAL_KEYTERMS), previous_text="x" * 900,
    )
    headers = settings.headers("key")
    assert "FIR" in headers["keyterms"]
    assert len(headers["previous_text"]) == 600


def test_vad_values_are_clamped_to_the_documented_ranges():
    settings = speech.SttSettings(
        language_code="hi-IN", candidates=[], vad_threshold=4.0,
        min_silence_ms=5, sample_rate=1234,
    )
    headers = settings.headers("key")
    assert headers["x-vad-threshold"] == "1.00"
    assert headers["x-min-silence-ms"] == "100"
    assert headers["x-sample-rate"] == "16000"


def test_any_capture_buffer_becomes_1024_byte_frames():
    # The contract is per frame. A capture path that batches, or a reconnect
    # that splits a buffer, must not reach upstream as an invalid frame.
    buffer = speech.FrameBuffer()
    frames = buffer.push(b"\x01\x00" * 2048)
    assert len(frames) == 4
    assert all(len(frame) == 1024 for frame in frames)


def test_a_partial_frame_is_padded_not_dropped():
    buffer = speech.FrameBuffer()
    assert buffer.push(b"\x01\x00" * 100) == []
    tail = buffer.flush()
    assert len(tail) == 1 and len(tail[0]) == 1024


def test_transcript_frames_are_normalised_with_the_heard_language():
    message = {
        "type": "transcript", "text": " मेरा वेतन नहीं मिला ",
        "detected_language_code": "hi-IN", "latency": 320,
        "segment_id": "abc-seg-1", "is_final": False,
    }
    parsed = speech.transcript_of(message)
    assert parsed["text"] == "मेरा वेतन नहीं मिला"
    assert parsed["detected_language_code"] == "hi-IN"
    assert parsed["is_final"] is True
    assert parsed["latency_ms"] == 320


def test_empty_or_non_transcript_messages_are_passed_through_untouched():
    assert speech.transcript_of({"type": "transcript", "text": "   "}) is None
    assert speech.transcript_of({"type": "processing"}) is None


# ----------------------------------------------------------------- Timbre


def test_a_v3_voice_survives_and_the_format_is_owned_by_the_server():
    body, choice = speech.timbre_body(
        {"text": "नमस्ते", "voice": "Nalini", "sample_rate": 99999,
         "audio_config": {"sample_rate": 48000, "encoding": "mp3"}},
        model="timbre-v3.0", voice="Shivani", language="hi-IN",
    )
    assert body["voice"] == "Shivani"
    assert body["model"] == "timbre-v3.0"
    assert body["audio_config"]["encoding"] == "linear_pcm"
    assert body["audio_config"]["container"] == "raw"
    assert choice.substituted is None


def test_a_mismatched_voice_is_replaced_instead_of_failing_mid_answer():
    body, choice = speech.timbre_body(
        {"text": "hello"}, model="timbre-v3.0", voice="Nalini", language="hi-IN",
    )
    assert body["voice"] in speech.V3_VOICES
    assert "not a timbre-v3.0 voice" in (choice.substituted or "")


def test_v25_uses_a_voice_that_speaks_the_language():
    body, choice = speech.timbre_body(
        {"text": "नमस्ते"}, model="timbre-v2.5", voice="Nalini", language="ta-IN",
    )
    assert body["voice"] in speech.V25_VOICES["ta-IN"]
    assert choice.substituted


def test_speed_is_clamped_to_the_documented_range():
    body, _ = speech.timbre_body(
        {"text": "hello", "speed": 9}, model="timbre-v3.0", voice="Shivani", language="en-IN",
    )
    assert body["speed"] == 1.15


def test_unsupported_audio_tags_are_not_read_aloud():
    # v2.5 silently strips every tag, so tag-only text must not reach it.
    assert "<coughs>" not in speech.condition_text("ok <coughs> fine", "timbre-v2.5")
    # A tag nobody documented is never spoken, on either model.
    assert "<script>" not in speech.condition_text("ok <script> fine", "timbre-v3.0")
    assert "<coughs>" in speech.condition_text("ok <coughs> fine", "timbre-v3.0")


def test_currency_is_spoken_as_rupees():
    assert "rupees" in speech.condition_text("fine is ₹5,000")


def test_a_stream_rotation_happens_before_the_provider_cap_and_lands_on_a_frame():
    from app import main

    # Prisma closes a stream at fifteen minutes; the proxy rotates first, so the
    # seam is ours rather than wherever the provider hangs up.
    assert main.STT_ROTATE_SECONDS < 900
    buffer = speech.FrameBuffer()
    frames = buffer.push(b"\x00" * (speech.CHUNK_BYTES + 100))
    assert len(frames) == 1
    assert buffer.buffered == 100
    tail = buffer.flush()
    assert len(tail) == 1 and len(tail[0]) == speech.CHUNK_BYTES
    assert buffer.buffered == 0


def test_an_empty_request_keeps_the_last_sentence_out_of_the_proxy():
    body, _ = speech.timbre_body({"text": ""}, model="timbre-v3.0", voice="Shivani", language="auto")
    assert body["text"] == ""


# ------------------------------------------------------------ end to end

def test_reply_language_follows_a_marathi_caller_without_a_model(monkeypatch):
    monkeypatch.setattr(legal_engine, "model_configured", lambda: False)
    result = asyncio.run(legal_engine.answer_turn("माझा मालक पगार देत नाही", "auto"))
    # Devanagari alone cannot tell Marathi from Hindi, but the function words
    # can: a Marathi caller answered in Hindi for the whole call is a failure
    # they can hear.
    assert result["language"] == "mr-IN"
    assert result["language_basis"] == "script_markers"
    assert result["tts_language"] == "mr-IN"


def test_hindi_stays_hindi_when_the_markers_are_hindi() -> None:
    assert language.resolve("मेरा मालिक वेतन नहीं दे रहा है") == ("hi-IN", "script_markers")
    # "मी कुठे राहतो" is Marathi: two function words, no Hindi ones.
    assert language.resolve("मी कुठे राहतो")[0] == "mr-IN"


def test_a_telugu_caller_is_answered_in_telugu_without_a_model(monkeypatch):
    monkeypatch.setattr(legal_engine, "model_configured", lambda: False)
    result = asyncio.run(legal_engine.answer_turn("నా జీతం ఇవ్వలేదు, ఏమి చేయాలి", "auto"))
    assert result["language"] == "te-IN"
    assert result["language_basis"] == "script"
    assert result["tts_language"] == "te-IN"


def test_the_no_model_reply_is_in_the_callers_language(monkeypatch):
    monkeypatch.setattr(legal_engine, "model_configured", lambda: False)
    result = asyncio.run(legal_engine.answer_turn("నా జీతం ఇవ్వలేదు, ఏమి చేయాలి", "auto"))
    assert result["language"] == "te-IN"
    # The degraded sentence used to be English under a Telugu question.
    assert "ఆమోదించిన మూలం" in result["response"]
    # And no English suggestion chip under a Telugu answer.
    assert result["next_question"] is None


def test_every_degraded_sentence_exists_in_every_supported_language():
    """A missing key falls back to English inside a sentence, silently.

    `_floor` returns `FLOOR[en][key]` when the caller's language has no entry,
    which is right for an absent language and wrong for an absent *key*: ten
    languages answered the case-led path in English because only three of them
    defined `case_only_lead`. This is the test that makes adding a key to one
    language a failing build until every other language has it too.
    """
    expected = set(legal_engine.FLOOR["en-IN"])
    for code, table in legal_engine.FLOOR.items():
        missing = sorted(expected - set(table))
        assert not missing, f"{code} is missing {missing}; it would speak English"
        if code == "en-IN":
            continue
        for key in expected:
            assert table[key] != legal_engine.FLOOR["en-IN"][key], (
                f"{code}.{key} is the English sentence"
            )


def test_a_telugu_caller_hears_telugu_framing_around_an_english_quote():
    """The quote stays English (it is verbatim); everything around it does not.

    This is the case-led degraded path: no statute matched, one judgment did.
    It used to open with an English lead and close with an English caveat under
    a Telugu question.
    """
    excerpt = (
        "Registration of FIR is mandatory under Section 154 if the information "
        "discloses a cognizable offence."
    )
    context = SimpleNamespace(
        language="te-IN",
        case_relevance_note="none_on_point",
        cases=[{
            "id": "x-1", "title": "Lalita Kumari vs State of UP",
            "court": "Supreme Court of India", "court_level": "supreme",
            "citation": "(2014) 2 SCC 1", "decision_date": "2013-11-12",
            "jurisdiction": "IN", "source_url": "https://example.test/x-1",
            "excerpt": excerpt, "treatment_status": "not_checked",
        }],
    )
    sentence = legal_engine._case_led_sentence(context)
    assert legal_engine.FLOOR["te-IN"]["case_only_tail"] in sentence
    assert legal_engine.FLOOR["te-IN"]["case_only_caveat"].strip() in sentence
    assert legal_engine.FLOOR["en-IN"]["case_only_tail"] not in sentence
    # The passage itself is still exact: claim checking depends on it.
    assert excerpt in sentence
    # English still gets the English framing, unchanged.
    english = legal_engine._case_led_sentence(
        SimpleNamespace(language="en-IN", case_relevance_note="", cases=context.cases)
    )
    assert legal_engine.FLOOR["en-IN"]["case_only_tail"] in english


def test_the_acknowledgement_is_spoken_in_the_callers_language(monkeypatch):
    monkeypatch.setattr(legal_engine, "model_configured", lambda: False)

    async def collect():
        return [item async for item in legal_engine.answer_stream("నా జీతం ఇవ్వలేదు", "auto")]

    kind, payload = asyncio.run(collect())[0]
    assert kind == "ack"
    assert payload["language"] == "te-IN"
    assert payload["text"] == legal_engine.ACK["te-IN"]


def test_the_proxy_opens_a_v3_auto_stream_and_reframes_audio(monkeypatch):
    """The whole defect, end to end: what the upstream socket is actually told."""
    from starlette.websockets import WebSocketDisconnect

    import app.main as main

    captured: dict[str, object] = {}

    class FakeUpstream:
        def __init__(self) -> None:
            self.received: list[bytes] = []
            self.sent_transcript = False

        async def send(self, data):
            self.received.append(data)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent_transcript:
                await asyncio.sleep(3600)
            # Wait for the audio so the frame assertion below is deterministic.
            for _ in range(200):
                if len(self.received) >= 4:
                    break
                await asyncio.sleep(0.005)
            self.sent_transcript = True
            return json.dumps({
                "type": "transcript",
                "text": "నా జీతం ఇవ్వలేదు",
                "detected_language_code": "te-IN",
                "latency": 300,
            })

    class FakeSocket:
        """Ten 1,024-byte frames, then the caller releases the microphone."""

        def __init__(self) -> None:
            self.sent: list[str] = []
            self.frames = 0

        async def accept(self):
            return None

        async def send_text(self, text):
            self.sent.append(text)

        async def send_bytes(self, data):
            return None

        async def receive(self):
            await asyncio.sleep(0.005)
            self.frames += 1
            if self.frames > 10:
                raise WebSocketDisconnect()
            return {"type": "websocket.receive", "bytes": b"\x00\x00" * 512}

    fake = FakeUpstream()
    socket = FakeSocket()

    def connect(url, **kwargs):
        # `websockets.connect` is used as an async context manager, so this
        # returns the object rather than a coroutine.
        captured["url"] = url
        captured["headers"] = kwargs["additional_headers"]
        return fake

    monkeypatch.setattr(main.websockets, "connect", connect)
    monkeypatch.setattr(main, "GNANI_KEY", "test-key")

    asyncio.run(main._proxy_stt(socket, main._stt_settings(_QuerySocket("auto", "48000"), None), "test-key"))

    headers = captured["headers"]
    assert captured["url"] == speech.PRISMA_STREAM_URL
    assert headers["model"] == "prisma-v3.0"
    assert headers["language_code"] == "indic-auto"
    assert headers["format"] == "codemix"
    assert "lang_code" not in headers and "x-format" not in headers
    assert headers["x-sample-rate"] == "48000"
    assert "FIR" in headers["keyterms"]
    assert len(fake.received) == 10 and all(len(frame) == 1024 for frame in fake.received)

    delivered = [json.loads(item) for item in socket.sent]
    transcript = next(item for item in delivered if item["type"] == "transcript")
    assert transcript["detected_language_code"] == "te-IN"
    assert transcript["is_final"] is True


class _QuerySocket:
    def __init__(self, language: str, rate: str) -> None:
        self.query_params = {"language": language, "sample_rate": rate}


@pytest.mark.parametrize(
    "text",
    [
        "అతను నన్ను చంపేస్తాడు అని బెదిరిస్తున్నాడు",
        "அவன் என்னை கொலை செய்வான் என்று மிரட்டுகிறான்",
        "সে আমাকে মেরে ফেলবে বলে ভয় দেখাচ্ছে",
        "તે મને મારી નાખશે તેમ કહે છે",
        "ಅವನು ನನ್ನನ್ನು ಕೊಲ್ಲುತ್ತಾನೆ ಎಂದು ಬೆದರಿಸುತ್ತಿದ್ದಾನೆ",
        "അവൻ എന്നെ കൊല്ലും എന്ന് ഭീഷണിപ്പെടുത്തുന്നു",
        "ਉਹ ਮੈਨੂੰ ਮਾਰ ਦੇਵੇਗਾ",
        "तो मला मारून टाकेल अशी धमकी देतो",
    ],
)
def test_the_safety_net_understands_every_served_language(text):
    # Safety must not depend on a model being configured: the lexical net is
    # the path that still works when the provider is down.
    from app import triage

    assessment = triage.assess(text)
    assert assessment.level == "critical"
    assert triage.is_emergency(assessment)


@pytest.mark.parametrize(
    "text",
    [
        "నేను ఆత్మహత్య చేసుకోవాలనుకుంటున్నాను",
        "நான் தற்கொலை செய்து கொள்ள நினைக்கிறேன்",
        "আমি আত্মহত্যা করতে চাই",
        "હું આત્મહત્યા કરવા માંગુ છું",
        "ನಾನು ಆತ್ಮಹತ್ಯೆ ಮಾಡಿಕೊಳ್ಳಬೇಕು",
        "എനിക്ക് ആത്മഹത്യ ചെയ്യണം",
        "ਮੈਂ ਖੁਦਕੁਸ਼ੀ ਕਰਨਾ ਚਾਹੁੰਦਾ ਹਾਂ",
        "मला आत्महत्या करायची आहे",
    ],
)
def test_self_harm_is_recognised_in_every_served_language(text):
    from app import triage

    assessment = triage.assess(text)
    assert triage.is_emergency(assessment)
    assert assessment.has("block_drafting")


def test_config_api_never_asks_for_a_language():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        config = client.get("/api/config").json()
    assert config["speech"]["auto_detect"] is True
    assert config["speech"]["language_question_required"] is False
    assert config["speech"]["capabilities"]["stt"]["auto_detect_code"] == "indic-auto"
    assert "hi-en" in config["speech"]["capabilities"]["tts"]["languages"]
    assert config["language_policy"]["selection"] == "automatic"
