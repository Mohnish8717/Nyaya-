"""Language identity for a voice-first caller in India.

A caller does not announce a language. They open with "mera vetan nahi mila,
company bol rahi hai resign kar do" and expect the assistant to follow. Asking
"which language?" is the single most avoidable question in this product, so the
language is decided from evidence, in this order:

  1. what the caller explicitly chose (an override, honoured for the call);
  2. what the speech recogniser reported for *this* utterance
     (`detected_language_code`, Prisma v3 in auto mode);
  3. the script the text is written in;
  4. Romanised Indic markers when the script is Latin ("nahi", "chahiye",
     "kaise") — that is Hinglish, not English;
  5. English.

Three details decide whether this feels effortless or broken:

  * **Code-mixing is the normal case, not an edge case.** A Hindi sentence with
    English legal nouns ("section 138", "cheque bounce", "notice") stays one
    language: the reply keeps the caller's matrix language and leaves the
    legal term in English. `hi-en` exists for exactly that, and it is what a
    Latin-script Hindi transcript becomes.
  * **Devanagari is ambiguous.** Hindi and Marathi share a script, so script
    detection can only say "Devanagari". It answers `hi-IN` and lets the turn
    reader correct it when a model is configured; a Marathi caller who hears
    Hindi once can override.
  * **A detected code the provider cannot speak is not an error.** Odia has no
    Prisma/Timbre path in this build; it degrades to text rather than to a
    wrong-language voice.

Every function here is pure and cheap (no model, no network), so it can run on
the critical path before anything is generated.
"""
from __future__ import annotations

# The languages Prisma and Timbre both serve. Order is the UI's order.
BASE_LANGUAGES: tuple[str, ...] = (
    "en-IN", "hi-IN", "bn-IN", "te-IN", "ta-IN", "mr-IN", "gu-IN", "kn-IN", "ml-IN", "pa-IN",
)
# Text-only in this build: the LLM covers it, the speech models do not.
TEXT_ONLY_LANGUAGES: tuple[str, ...] = ("or-IN",)
HINGLISH = "hi-en"
AUTO = "auto"
# Prisma's streaming auto-detect value. It is not a language: the provider
# reports the language it heard in `detected_language_code`.
PRISMA_AUTO = "indic-auto"

TEXT_LANGUAGES: frozenset[str] = frozenset(BASE_LANGUAGES + TEXT_ONLY_LANGUAGES + (HINGLISH, AUTO, PRISMA_AUTO))
SPEECH_LANGUAGES: frozenset[str] = frozenset(BASE_LANGUAGES)

LABELS: dict[str, str] = {
    "en-IN": "English", "hi-IN": "Hindi", "bn-IN": "Bengali", "te-IN": "Telugu",
    "ta-IN": "Tamil", "mr-IN": "Marathi", "gu-IN": "Gujarati", "kn-IN": "Kannada",
    "ml-IN": "Malayalam", "or-IN": "Odia", "pa-IN": "Punjabi", "hi-en": "Hinglish (Hindi + English)",
    "auto": "Auto — follows you", "indic-auto": "Auto — follows you",
}

# Script ranges, checked in order. Devanagari is last because it is the one we
# cannot disambiguate.
_SCRIPTS: tuple[tuple[int, int, str], ...] = (
    (0x0980, 0x09FF, "bn-IN"),
    (0x0A00, 0x0A7F, "pa-IN"),
    (0x0A80, 0x0AFF, "gu-IN"),
    (0x0B80, 0x0BFF, "ta-IN"),
    (0x0C00, 0x0C7F, "te-IN"),
    (0x0C80, 0x0CFF, "kn-IN"),
    (0x0D00, 0x0D7F, "ml-IN"),
    (0x0900, 0x097F, "hi-IN"),
)

# Romanised Indic function words. Deliberately particles and everyday verbs,
# not legal terms, so an English sentence about "maintenance" does not become
# Hinglish. Two independent hits are required.
_ROMANISED = frozenset("""
    hai hain nahi nahin kya kaise kyun kab kahan kaun mera mere meri mujhe
    tumhara aapka humara uska uski unka karna karne karo kiya kiye raha rahi
    rahe chahiye chahiye mila mili mile gaya gayi gaye aaya aayi bhi aur par
    lekin kuch koi bahut zyada kam paisa rupaye vetan tankha kanoon kanun
    adalat vakil police thana shikayat notice bheja bhej do diya liya abhi
    phir sirf bas theek thik accha acha ji haan baat bol bolo batao bataiye
    mera ghar ghar kaam naukri company office bhai didi maa papa
""".split())

# Devanagari is shared by Hindi and Marathi, so the script alone cannot
# decide. These are function words, not content words, because a Marathi
# sentence about a Hindi-language notice is still Marathi. Two independent
# hits with a clear margin are required before the guess changes.
_MARATHI_MARKERS = frozenset("""
    आहे आहेत नाही नाहीत माझा माझी माझे माझ्या तुम्ही आम्ही त्यांनी यांनी
    काय का करायचे झाले होते होता पण खूप मला त्याला तिला साठी मध्ये जवळ
    सांगा बघा द्या घ्या जातो येतो राहतो कृपया आणि हा ही हे ते तो
    कुठे कसे केव्हा तुमचा तुमची माझ्याकडे आपल्याला हवे हवा नको
    पगार काम नोकरी कंपनी पोलीस तक्रार न्यायालय वकील कायदा अधिकार
""".split())
_HINDI_MARKERS = frozenset("""
    है हैं नहीं मेरा मेरी मेरे मेरेको आप तुम हम उसका उसकी उनका करना करने
    किया गया गई लेकिन बहुत ज्यादा पैसा वेतन अदालत वकील थाना शिकायत भेजा
    दिया लिया अभी फिर सिर्फ ठीक अच्छा जी हाँ बात बोलो बताओ कहाँ कब क्यों
    कैसे और यह वह जो काम नौकरी कंपनी पुलिस न्यायालय कानून अधिकार
""".split())

_SINGLE_WORD_ALIASES: dict[str, str] = {
    "en": "en-IN", "hi": "hi-IN", "hi_en": "hi-en", "hinglish": "hi-en",
    "bn": "bn-IN", "te": "te-IN", "ta": "ta-IN", "mr": "mr-IN", "gu": "gu-IN",
    "kn": "kn-IN", "ml": "ml-IN", "or": "or-IN", "od": "or-IN", "pa": "pa-IN",
    "auto": AUTO, "indic-auto": AUTO, "unknown": AUTO,
    # A model asked for the turn's language sometimes answers with the name
    # rather than the code. Accepting it keeps that from collapsing to auto
    # and silently switching the reply to English.
    "english": "en-IN", "hindi": "hi-IN", "bengali": "bn-IN", "bangla": "bn-IN",
    "telugu": "te-IN", "tamil": "ta-IN", "marathi": "mr-IN", "gujarati": "gu-IN",
    "kannada": "kn-IN", "malayalam": "ml-IN", "odia": "or-IN", "oriya": "or-IN",
    "punjabi": "pa-IN", "hinglish": "hi-en", "code-mixed": "hi-en", "mixed": "hi-en",
}


def normalize(code: object) -> str:
    """Map a language code onto one this build knows, without guessing.

    Accepts BCP-47 variants, provider spellings (`indic-auto`), ISO-639-1 and
    the word "Hindi". Anything unrecognised becomes `auto`, because a wrong
    concrete language is worse than following the caller.
    """
    if not isinstance(code, str):
        return AUTO
    candidate = code.strip().replace("_", "-")
    if not candidate:
        return AUTO
    lowered = candidate.lower()
    if lowered in _SINGLE_WORD_ALIASES:
        return _SINGLE_WORD_ALIASES[lowered]
    if candidate in TEXT_LANGUAGES:
        return candidate
    head = lowered.split("-")[0]
    if head in _SINGLE_WORD_ALIASES:
        resolved = _SINGLE_WORD_ALIASES[head]
        # `hi-en` is a real code, not "the Hindi variant of English"; only take
        # a suffix mapping when the head is a plain language.
        return resolved if lowered.split("-")[0] != "hi" or len(lowered.split("-")) == 1 else resolved
    if head in {"hi"} and "en" in lowered:
        return HINGLISH
    return AUTO


def script_of(text: str) -> str | None:
    """Return the language a script unambiguously implies, else None."""
    counts: dict[str, int] = {}
    for char in text or "":
        point = ord(char)
        for start, end, code in _SCRIPTS:
            if start <= point <= end:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return None
    # The dominant script wins; mixed-script text is common ("मेरा cheque
    # bounce ho gaya"), and the Indic script is the matrix, not the Latin one.
    return max(counts.items(), key=lambda item: item[1])[0]


def devanagari_variant(text: str) -> tuple[str, str]:
    """Separate Marathi from Hindi inside shared Devanagari.

    Returns (code, confidence). The default stays Hindi because it is far more
    common in this corpus, but a caller with two or more Marathi function
    words is not a Hindi caller, and answering in Hindi for a whole call is a
    failure the caller can hear.
    """
    words = [
        word.strip(".,!?;:\"'()\u0964\u0965").lower()
        for word in (text or "").split()
    ]
    marathi = sum(1 for word in words if word in _MARATHI_MARKERS)
    hindi = sum(1 for word in words if word in _HINDI_MARKERS)
    if marathi >= 2 and marathi > hindi:
        return "mr-IN", "markers"
    if hindi >= 2 and hindi > marathi:
        return "hi-IN", "markers"
    return "hi-IN", "default"


def looks_romanised_indic(text: str) -> bool:
    words = [word.strip(".,!?;:\"'()").lower() for word in (text or "").split()]
    hits = sum(1 for word in words if word in _ROMANISED)
    return hits >= 2 and hits / max(1, len(words)) >= 0.12


def resolve(
    text: str = "",
    *,
    stt_code: object = None,
    override: object = None,
) -> tuple[str, str]:
    """Decide the language of a turn. Returns (code, basis).

    `basis` is reported to the caller so a wrong guess is visible and
    correctable instead of silently shaping the whole conversation.
    """
    chosen = normalize(override)
    if chosen not in {AUTO, PRISMA_AUTO}:
        return chosen, "caller_selected"

    heard = normalize(stt_code)
    if heard not in {AUTO, PRISMA_AUTO}:
        return heard, "speech_recognition"

    script = script_of(text)
    if script:
        if script == "hi-IN":
            variant, confidence = devanagari_variant(text)
            return variant, "script" if confidence == "default" else "script_markers"
        return script, "script"

    if looks_romanised_indic(text):
        return HINGLISH, "romanised_hindi"

    return "en-IN", "default"


def stt_language(override: object = None, detected: object = None) -> tuple[str, list[str]]:
    """Build Prisma's `language_code` value plus the candidate hint list.

    Prisma v3 in `indic-auto` returns `detected_language_code` per utterance and
    accepts up to four candidates as a *hint* rather than a lock. Passing the
    hint is what keeps Hinglish accurate without ever asking the caller.
    """
    chosen = normalize(override)
    if chosen in SPEECH_LANGUAGES:
        # An explicit choice still gets the caller's likely second language as
        # a hint: a Tamil speaker says "notice" in English mid-sentence.
        hints = [chosen] + [code for code in ("en-IN", "hi-IN") if code != chosen]
        return chosen, hints[:4]
    heard = normalize(detected)
    if heard in SPEECH_LANGUAGES:
        return heard, [heard, "en-IN"]
    # No candidates here on purpose. `indic-auto` is what makes Prisma identify
    # the language of each utterance; a candidate list would pin the stream to
    # those languages and quietly exclude every other one.
    return PRISMA_AUTO, []


def reply_language(detected: str, caller_text: str) -> str:
    """The language the answer should come back in.

    The caller's own words win only when nothing better is known: a Hindi
    caller quoting an English notice still gets Hindi back.
    """
    if detected and detected not in {AUTO, PRISMA_AUTO}:
        return detected
    return resolve(caller_text)[0]


def tts_language(code: str) -> str:
    """Timbre language value. `auto` is the documented mixed-content path.

    Timbre v3 voices each speak all ten languages, and `auto` detects from the
    script. Hinglish has no v3 voice of its own, so the reply is synthesised in
    auto mode rather than forced into Hindi or English, which would mangle half
    of every mixed sentence.
    """
    resolved = normalize(code)
    if resolved in {AUTO, PRISMA_AUTO, HINGLISH}:
        return "auto"
    if resolved in SPEECH_LANGUAGES:
        return resolved
    # Odia has no Prisma or Timbre path in this build. Saying `auto` here asked
    # Timbre to voice Odia text with whatever it detected, which is how a
    # caller hears their question read back in the wrong language. The client
    # prints "text only" instead, which is true.
    return "unavailable"


def is_speakable(code: str) -> bool:
    return normalize(code) not in TEXT_ONLY_LANGUAGES
