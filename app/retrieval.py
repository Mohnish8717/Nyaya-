"""Query construction and rank fusion for legal retrieval.

Why this exists: the previous builder OR-joined every token longer than one
character. On the bundled 1,810-judgment index that matched 1,808 of 1,810
judgments for an ordinary sentence, because function words like "is" and
"the" match nearly every document. BM25 ordering then did all the filtering
over a candidate set that was the whole corpus.

This module instead builds a cascade: the strictest useful AND query first,
then a relaxed AND, then a broad OR, fusing whatever comes back with
reciprocal rank fusion. A small bilingual glossary turns Romanized Indic
terms ("kiraya", "tankha", "thana") into the English legal vocabulary the
corpus actually uses.

Dense retrieval is a hook, not a claim. No embedding model is assumed here;
`dense_search` stays disabled until a backend is configured, and nothing in
this module reports vector search as working when it is not.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import quote as urlquote

TOKEN_RE = re.compile(r"[\w\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u20B9]+", re.UNICODE)
FTS_RESERVED = {"AND", "OR", "NOT", "NEAR"}

# Function words in English, plus common Romanized Indic particles. Kept
# deliberately small: an aggressive stoplist would delete legal terms such as
# "not" that change meaning, so negation is handled separately below.
STOPWORDS = frozenset("""
a an the is are was were be been being am do does did done have has had having
i me my mine we us our you your he him his she her it its they them their
this that these those there here who whom whose which what when where why how
of in on at to for from by with without about into over after before under
and or but if then than so as such not no nor too very can could should would will shall may might must
also just only same own other some any each few more most much many
please kindly sir madam hi hello ok okay yeah
kya hai hain ka ki ke ko se mein me aur ya par bhi nahi nahin koi kuch kuchh mera meri mere hum tum aap apna apni apne kar karo karna karta karti tha thi the hua hui hoga hongi kyun kaise kahan kab wala wali wale
""".split())

# The same function words written in Devanagari. Without these a Hindi
# sentence's grammar was treated as its subject: "मेरा चेक बाउंस हो गया, अब क्या
# करूँ" yielded the terms {चेक, करूँ}, and "करूँ" -- "should I do" -- matched
# Hindi-language judgments about anything at all. Two terms, one of them noise,
# was enough to clear a 34% gate and return four unrelated decisions. A query
# whose only surviving words are these retrieves nothing, which is the honest
# answer.
INDIC_STOPWORDS = frozenset("""
मैं मुझे मेरा मेरी मेरे हम हमें हमारा हमारी आप आपका आपकी तुम तुम्हारा वह वो यह ये
उस उन्हें उनका का की के को से में पर और या नहीं ना है हैं था थी थे हो हैं होगा होगी हुआ हुई हुए
कर करना करने करता करती करते किया किए रहा रही रहे गया गई गए दिया दी
क्या क्यों कैसे कब कहाँ कौन सा भी तो ही बहुत कोई कुछ सकता सकती सकते चाहिए अब फिर
करूँ करूं होता होती होते जाता जाती जाते मुझको उन्होंने इसने उसने
मेरे को थोड़ा बहुत जरा यार भाई
""".split())
STOPWORDS = STOPWORDS | INDIC_STOPWORDS

# Romanized / colloquial Indian terms -> English legal vocabulary used by the
# corpus. Deterministic, reviewable, and safe to extend without a model.
GLOSSARY: dict[str, tuple[str, ...]] = {
    "kiraya": ("rent",), "kirayedar": ("tenant",), "makan": ("house", "premises"),
    "makaan": ("house", "premises"), "malik": ("landlord", "owner"),
    "tankha": ("wages", "salary"), "salary": ("wages",), "naukri": ("employment", "job"),
    "nokri": ("employment", "job"), "naukriwala": ("employer",),
    "thana": ("police", "station"), "policewala": ("police",),
    "shikayat": ("complaint",), "shikayat_": ("complaint",), "fir": ("first", "information", "report"),
    "vakil": ("lawyer", "advocate", "counsel"), "vakeel": ("lawyer", "advocate"),
    "adalat": ("court",), "court": ("court",), "judge": ("judge",),
    "kanoon": ("law",), "kanoon": ("law",), "hak": ("rights",), "adhikar": ("rights",),
    "soochna": ("information",), "sarkari": ("government", "public", "authority"),
    "muft": ("free", "legal", "aid"), "paisa": ("money", "amount"), "paise": ("money", "amount"),
    "dhokha": ("fraud", "cheating"), "dhoka": ("fraud", "cheating"),
    "shaadi": ("marriage",), "shadi": ("marriage",), "talaq": ("divorce",),
    "bachcha": ("child",), "bacche": ("child",), "baccha": ("child",),
    "zameen": ("land", "property"), "jameen": ("land", "property"),
    "kagaz": ("document",), "dastavez": ("document",), "daastavez": ("document",),
    "gharelu": ("domestic",), "hinsa": ("violence",), "maar": ("assault", "violence"),
    "dhamki": ("threat", "intimidation"), "mafi": ("compensation", "relief"),
    "naukri_chhodo": ("termination", "dismissal"), "chhutti": ("leave",),
    "cheque": ("cheque",), "khata": ("account",), "bank": ("bank",),
    "vanchit": ("deprived",), "shoshan": ("exploitation",), "bhatta": ("allowance",),
}

# Everyday words a caller uses, mapped to the term the judgment actually uses.
# This is the vocabulary gap that matters most in legal retrieval: a person says
# "my cheque bounced" and every s.138 judgment says "dishonour of cheque"; a
# caller says "he changed the lock", the judgment says "dispossession". Each
# entry is a reviewed one-way mapping, not a similarity guess — an expansion
# that is wrong adds noise to every query that contains the trigger word.
LEGAL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "bounced": ("dishonour", "dishonoured", "dishonor"),
    "bounce": ("dishonour", "dishonoured"),
    "bouncing": ("dishonour",),
    "sacked": ("termination", "retrenchment", "dismissal"),
    "fired": ("termination", "retrenchment", "dismissal"),
    "boss": ("employer", "management"),
    "salary": ("wages", "emoluments"),
    "lock": ("dispossession", "lockout", "premises"),
    "lockout": ("dispossession", "premises"),
    "eviction": ("dispossession", "tenancy"),
    "deposit": ("advance", "security"),
    "molested": ("outrage", "modesty", "assault"),
    "beaten": ("assault", "hurt", "injury"),
    "stolen": ("theft", "misappropriation"),
    "defective": ("defect", "goods"),
    "refund": ("refund", "restitution"),
}

# Words about the *request* rather than the legal subject. A caller's sentence
# is mostly framing — "I want the most relevant cases that cover the refusal of
# FIR" has four subject-bearing words and six framing words — and because the
# retriever anchors on the rarest terms, framing words were being selected as
# the strict BM25 anchor *because* they are rare in judgments. That one property
# is why a refused-FIR question returned income-tax "refusal of registration"
# cases: the anchor was "want" AND "relevant". Removing these is not cosmetic;
# it moves the anchor onto the caller's actual subject.
META_TERMS = frozenset("""
want wants wanted wanting need needs needed tell tells told show shows showing
give gives find finds finding look looking help know understands understand
question questions answer answers opinion opinions advice best most least
latest recent example examples cover covers covering explain explaining related
relevant similar case cases matter matters judgment judgments judgement judgements
precedent precedents something anything kindly
""".split())

# Words whose presence changes the meaning of a query and must not be dropped
# or merged away by the glossary step.
NEGATIONS = frozenset({"not", "no", "never", "without", "neither", "nor", "nahi", "nahin", "mat"})

# FTS5 does not stem, so "refused", "refuse" and "refusal" are three unrelated
# terms to the index. The caller says "refused", the judgment says "refusal",
# and the match silently fails. These suffixes are stripped only to build a
# prefix query (`"refus"*`), which matches every form of the word without a
# stemmer library and without touching the stored text.
DERIVATIONAL = ("ing", "ed", "es", "al", "s")

DENSE: Callable[[str], list[str]] | None = None


def register_dense(backend: Callable[[str], list[str]] | None) -> None:
    """Attach a dense retriever returning ordered record IDs.

    Intentionally inert by default. A caller may register a real embedding
    backend; until then retrieval is lexical only and must be reported as such.
    """
    global DENSE
    DENSE = backend


def dense_available() -> bool:
    return DENSE is not None


def normalize(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def tokens(text: Any) -> list[str]:
    return [t for t in TOKEN_RE.findall(normalize(text)) if t]


def content_terms(text: Any, limit: int = 12) -> list[str]:
    """Ordered unique content terms: no function words, no 1-2 char debris.

    Digits and section-like tokens (138, 498a, 22) are always kept: they are
    the highest-signal part of a legal query and are often the whole query.
    """
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens(text):
        if token in FTS_RESERVED:
            continue
        # Function words and negations carry no lexical signal under BM25, so
        # they are not query terms. Negations are surfaced separately instead:
        # the model needs to know the caller said "not", the index does not.
        if token in STOPWORDS or token in META_TERMS:
            continue
        if len(token) <= 1:
            continue
        if len(token) == 2 and not token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= limit:
            break
    return out


def resolve_prefixes(
    terms: Sequence[str],
    document_frequency: Callable[[str], int],
    prefix_exists: Callable[[str], bool],
    minimum: int = 4,
    maximum: int = 3,
) -> list[str]:
    """Replace a word the index has never seen with the longest prefix it has.

    Voice input is misspelled constantly -- "landloard", "securty", "depost" --
    and FTS5 has no fuzzy match, so every one of those terms retrieves nothing
    and a caller who described a tenancy dispute is told the corpus is empty.
    Truncating an unknown word until the index recognises the prefix is the
    deterministic version of "it probably meant landlord": `landloard` becomes
    `landl`, which matches `landlord` and every inflection of it, and the
    phrase is still a prefix query rather than a similarity guess.

    Only words of six characters or more that the index does not contain are
    touched, at most `maximum` per query, and the shortest prefix tried is
    `minimum` characters. A word the index knows is never rewritten.
    """
    resolved: list[str] = []
    corrections = 0
    for term in terms:
        if corrections >= maximum or len(term) < 6 or not term.isascii():
            resolved.append(term)
            continue
        if document_frequency(term):
            resolved.append(term)
            continue
        replacement = ""
        for size in range(len(term) - 1, minimum - 1, -1):
            prefix = term[:size]
            if prefix_exists(prefix):
                replacement = prefix
                break
        resolved.append(replacement or term)
        if replacement:
            corrections += 1
    return list(dict.fromkeys(resolved))


def case_identity(case: dict[str, Any]) -> str:
    """One judgment's identity, across records that describe it twice.

    The ingested corpus holds the same decision under two ids when the source
    gave two URLs for it (`.../doc/1127466` and `.../doc/1127466/`), and both
    copies were offered as separate precedents -- two cards, one judgment. The
    URL is the identity when it exists; title, date and court are the fallback.
    """
    url = str(case.get("source_url") or "").strip().lower().rstrip("/")
    # The URL alone is not an identity. Two copies of one judgment can carry
    # `.../doc/1127466` and `.../doc/1127466/`, while the Supreme Court import
    # records the *tar archive* it came from -- one URL shared by every judgment
    # of that year. Keying on the URL alone therefore both missed the real
    # duplicate and merged hundreds of distinct decisions into one, which
    # removed the correct answer from four retrieval-eval cases.
    return "|".join(
        [url, *(
            str(case.get(key) or "").strip().lower()
            for key in ("title", "decision_date")
        )]
    )


def dedupe_cases(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first occurrence of each judgment, preserving order."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for case in cases:
        identity = case_identity(case)
        if identity in seen:
            continue
        seen.add(identity)
        out.append(case)
    return out


def negations(text: Any) -> list[str]:
    """Negations present in the utterance, for prompt context only."""
    return [token for token in tokens(text) if token in NEGATIONS]


def glossary_terms(terms: Sequence[str]) -> list[str]:
    """Add English legal equivalents for Romanized Indic and everyday terms.

    Both maps are one-way and deterministic: a caller's vocabulary is expanded
    into the corpus's, never the reverse, so the expansion can only add recall
    and the reason for every added term is a line in this file.
    """
    extra: list[str] = []
    for term in terms:
        for mapped in (*GLOSSARY.get(term, ()), *LEGAL_SYNONYMS.get(term, ())):
            if mapped not in terms and mapped not in extra:
                extra.append(mapped)
    return extra


def _fts_token(term: str) -> str:
    cleaned = term.replace('"', '""')
    return f'"{cleaned}"'


def stem(term: str) -> str:
    """Crude suffix strip, used only to build an FTS5 prefix match.

    Deliberately conservative: at least four characters must remain, so "laws"
    does not become "law" while "refused" becomes "refus". A prefix query on
    that stem matches refuse, refused, refusal and refusing in one term.
    """
    for suffix in DERIVATIONAL:
        if term.endswith(suffix) and len(term) - len(suffix) >= 4:
            return term[: -len(suffix)]
    return ""


def term_group(term: str) -> str:
    """One query term, its inflected forms, as a group that can be ANDed."""
    base = stem(term)
    if not base:
        return _fts_token(term)
    return f'({_fts_token(term)} OR "{base}"*)'


def stem_expressions(terms: Sequence[str], anchor: int = 2) -> list[str]:
    """AND the two strongest terms together with their inflected forms.

    This is a separate arm, not a replacement for the exact-term cascade: exact
    terms keep their precision, and this one adds the recall the index cannot
    provide on its own. Placed after the synonym arm so a phrase match always
    outranks a stem match.
    """
    anchor = min(anchor, len(terms))
    if anchor < 2:
        return []
    return [" AND ".join(term_group(term) for term in terms[:anchor])]


def focus(text: Any, extra: Sequence[str] = ()) -> str:
    """The query string retrieval should actually receive.

    The caller's subject words plus any reviewed vocabulary the caller would
    never use ("cognizable offence", "154"). Framing words are already gone,
    so this cannot be made worse by a chatty sentence.
    """
    terms = content_terms(text, limit=10)
    for term in extra:
        for token in content_terms(term, limit=4):
            if token not in terms:
                terms.append(token)
    return " ".join(terms)


def match_expressions(terms: Sequence[str], expanded: Sequence[str] | None = None, strict: int = 2, mid: int = 3) -> list[str]:
    """Return FTS5 MATCH expressions in descending precision.

    `terms` is expected rarest-first (the retriever orders by document
    frequency), so a two-term AND is the precise query and longer ANDs are a
    relaxation, not the other way round. Requiring four common terms at once
    collapses recall to almost nothing, which is just as wrong as matching
    everything.

    Every returned expression is safe to pass to `case_fts MATCH ?`; an empty
    list means the query has no usable terms and retrieval must return nothing
    rather than everything.
    """
    same = [t for t in terms if t]
    expanded = list(expanded or [])
    if not same and not expanded:
        return []

    expressions: list[str] = []
    primary = same[:max(2, strict)] if len(same) >= 2 else []
    if primary:
        expressions.append(" AND ".join(_fts_token(t) for t in primary))
    secondary = same[:max(mid, len(primary) + 1)]
    if len(secondary) >= 3 and len(secondary) != len(primary):
        expressions.append(" AND ".join(_fts_token(t) for t in secondary))
    if len(same) >= 4:
        expressions.append(" AND ".join(_fts_token(t) for t in same[:4]))
    if same:
        expressions.append(" OR ".join(_fts_token(t) for t in same[:8]))
    combined = [*same[:6], *expanded[:4]]
    if expanded:
        expressions.append(" OR ".join(_fts_token(t) for t in combined))

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for expression in expressions:
        if expression and expression not in seen:
            seen.add(expression)
            unique.append(expression)
    return unique


def synonym_expressions(terms: Sequence[str], limit: int = 3) -> list[str]:
    """AND-join a caller's term with its legal synonyms, as an OR group.

    `"lock" AND ("dispossession" OR "lockout")` keeps the precision of requiring
    the caller's own word while letting the corpus's vocabulary satisfy the
    match. Flattening every synonym into one OR list instead would let any of
    them match alone, which is how a query about a changed lock returns criminal
    "lock-up" cases.
    """
    out: list[str] = []
    for term in terms:
        mapped: list[str] = []
        for synonym in (*GLOSSARY.get(term, ()), *LEGAL_SYNONYMS.get(term, ())):
            if synonym != term and synonym not in mapped:
                mapped.append(synonym)
        if not mapped:
            continue
        alternatives = " OR ".join(_fts_token(synonym) for synonym in mapped[:3])
        out.append(f"{_fts_token(term)} AND ({alternatives})")
        if len(out) >= limit:
            break
    return out


def rrf(ranked_lists: Iterable[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion over ordered ID lists."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def interleave(primary: Sequence[str], secondary: Sequence[str], limit: int) -> list[str]:
    """Merge two rankings without letting the second displace the first.

    This exists for query rewriting. A caller's own words are the best evidence
    of what they meant, so the ranking built from them keeps its order; the
    rewrite only contributes entries the first ranking did not already have, in
    between them. Replacing instead of merging was measured on the retrieval
    set: a Hindi question rewritten into English gained the money-laundering
    case and lost the pollution one, a net wash, because the encoder had
    answered the original well. Alternating keeps both (measured: 9/14 against
    8/14 for monologue retrieval).
    """
    out: list[str] = []
    seen: set[str] = set()
    for index in range(max(len(primary), len(secondary))):
        for ranking in (primary, secondary):
            if index < len(ranking):
                item = ranking[index]
                if item not in seen:
                    seen.add(item)
                    out.append(item)
                    if len(out) >= limit:
                        return out
    return out


def legacy_match(text: Any) -> str:
    """The previous OR-of-every-word expression, kept so the eval script can
    measure the change honestly instead of asserting an improvement."""
    words = list(dict.fromkeys(tokens(text)))
    words = [w for w in words if len(w) > 1][:24]
    return " OR ".join(_fts_token(w) for w in words)


def source_link(url: str) -> str:
    """Stable, displayable URL without leaking tracker-laden parameters."""
    return urlquote(url, safe=":/?&=#%+@~")


def explain(text: str, terms: Sequence[str] | None = None) -> dict[str, Any]:
    """Small introspection helper for tests and the trace panel."""
    terms = list(terms) if terms is not None else content_terms(text)
    return {
        "terms": terms,
        "negations": negations(text),
        "expanded": glossary_terms(terms),
        "expressions": match_expressions(terms, glossary_terms(terms)),
        "legacy": legacy_match(text),
        "dense": dense_available(),
    }
