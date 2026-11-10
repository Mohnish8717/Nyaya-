"""Deterministic feature reranking.

A neural cross-encoder is the usual second stage, but a CPU cross-encoder
costs 1-3 seconds per query, which is not a voice turn — so this build does
not ship one behind a name it cannot honour. Instead the reranker combines
signals the system already has, in a few microseconds:

  * how much of the caller's query the candidate actually contains;
  * whether the query's numbers and section-like tokens appear;
  * exact jurisdiction agreement;
  * title agreement, which is high-signal because the parties' names and the
    subject usually sit there;
  * the deciding court's tier, because a district-court order and a Supreme
    Court judgment are not the same authority even when both are on point.

Every weight is explicit and every contribution is reported, so a wrong
ordering can be explained rather than shrugged at.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

DIGITS = re.compile(r"\d+[a-zA-Z]?")

# How much weight a court's tier carries in ordering. This is about authority,
# not similarity: a district-court order directly on point is still a weaker
# answer than a Supreme Court judgment on the same question, and a caller
# reading "the court held" is entitled to know which court. It reorders by at
# most the authority term's weight, so it can never promote an irrelevant
# judgment over a relevant one.
AUTHORITY = {"supreme": 1.0, "high": 0.62, "district": 0.30, "unknown": 0.44}

# Minimum share of the reviewed issue vocabulary a judgment must contain when a
# procedure card supplies it. Half, not a third: one word out of three (usually
# "premises" or "landlord") is present in almost every property judgment,
# including ones about acquisition and motor insurance. Two of three is the
# smallest overlap that says the judgment is about this kind of dispute. A
# two-term focus therefore requires both terms.
FOCUS_GATE = 0.5


def tier_of(candidate: dict[str, Any]) -> str:
    level = str(candidate.get("court_level") or "").strip().lower()
    if level in AUTHORITY:
        return level
    # Older records carry the court name but not the tier; derive it rather
    # than calling a Supreme Court judgment "unknown".
    court = str(candidate.get("court") or "").lower()
    if "supreme court" in court:
        return "supreme"
    if "high court" in court:
        return "high"
    if any(marker in court for marker in ("district", "civil judge", "sessions", "magistrate")):
        return "district"
    return "unknown"


def _terms(query: str) -> list[str]:
    from .retrieval import content_terms

    return content_terms(query, limit=12)


def _search_terms(query: str, terms: list[str] | None) -> list[str]:
    """Coverage terms: the caller's subject words plus reviewed vocabulary.

    When retrieval is given extra vocabulary ("cognizable offence", "154") it
    must also be scored on it, otherwise a passage that carries the caller's
    topic in the bench's words alone scores as if it matched nothing.
    """
    return list(terms) if terms else _terms(query)


def _haystack(candidate: dict[str, Any]) -> str:
    title = str(candidate.get("title") or "").lower()
    body = str(candidate.get("excerpt") or candidate.get("text") or "").lower()
    return f"{title} {title} {body}"


def _contains(term: str, haystack: str) -> bool:
    """Whole-word (prefix) match, not substring.

    Substring matching made "fir" match "first" and "firm" — two of the most
    common words in any judgment — so a contract case about a coal purchase
    counted as covering the caller's FIR question and passed the relevance
    gate. The start-of-word boundary keeps inflected forms ("arrest" matching
    "arrested", which is wanted) while rejecting "first".
    """
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}", haystack) is not None


def term_weights(
    query: str,
    details: dict[str, dict[str, Any]],
    candidates: list[tuple[str, float]],
    terms: list[str] | None = None,
) -> tuple[dict[str, float], bool]:
    """Inverse document frequency over the candidate pool itself.

    A query word present in most candidates ("return", "court", "case") cannot
    separate them, while one present in two is doing the work. Estimating that
    from the pool is free, needs no corpus statistics, and follows the actual
    competition being ranked. Without it, an incidental match on "return" made
    a banking judgment score as relevant as a tenancy one.

    Returns the weights and whether any query term occurs in the pool at all.
    The second value is not a detail: a Hindi question against an English
    judgment set matches nothing lexically, and a reranker with no relevance
    signal must not reorder the list it was given.
    """
    terms = _search_terms(query, terms)
    pool = [details[case_id] for case_id, _ in candidates if case_id in details]
    if not terms or not pool:
        return {}, False
    total = len(pool)
    haystacks = [_haystack(candidate) for candidate in pool]
    weights: dict[str, float] = {}
    matched = False
    for term in terms:
        df = sum(1 for hay in haystacks if _contains(term, hay))
        if df:
            matched = True
        weights[term] = math.log((total + 1) / (df + 0.5))
    scale = sum(weights.values()) or 1.0
    return {term: value / scale for term, value in weights.items()}, matched


def score(
    query: str,
    candidate: dict[str, Any],
    base: float = 0.0,
    jurisdiction: str = "IN",
    weights: dict[str, float] | None = None,
    terms: list[str] | None = None,
) -> tuple[float, dict[str, float]]:
    terms = _search_terms(query, terms)
    title = str(candidate.get("title") or "").lower()
    haystack = _haystack(candidate)

    if weights:
        coverage = sum(value for term, value in weights.items() if _contains(term, haystack))
        title_hits = sum(value for term, value in weights.items() if _contains(term, title))
    else:
        coverage = sum(1 for term in terms if _contains(term, haystack)) / len(terms) if terms else 0.0
        title_hits = sum(1 for term in terms if _contains(term, title)) / len(terms) if terms else 0.0
    numbers = [token for token in DIGITS.findall(query.lower()) if len(token) > 1]
    numeric = sum(1 for token in numbers if _contains(token, haystack)) / len(numbers) if numbers else 0.0
    jurisdiction_match = 1.0 if candidate.get("jurisdiction") in {jurisdiction, "IN"} else 0.0
    authority = AUTHORITY[tier_of(candidate)]

    # Relevance first, authority second. The four relevance terms are scaled by
    # 0.92 and authority multiplies what remains, so its entire reach is 8% of a
    # candidate's relevance score. As an added constant it was strong enough to
    # promote a marginally relevant Supreme Court judgment over a directly
    # responsive High Court one -- a wrong answer that looked authoritative.
    #
    # The fused rank carries most of the weight (0.62, was 0.50). Two independent
    # rankers -- BM25 over 94,000 passages and int8 cosine -- already agreed on
    # this ordering, and a third, purely lexical coverage term that weighs the
    # caller's loose paraphrase as heavily as the court's vocabulary will break
    # that agreement. Measured on "the refusal of FIR": Lalita Kumari was fused
    # rank 1, the passage carried "section 154", "cognizable" and "FIR", but it
    # did not contain the word "refusal" -- so coverage demoted it below income-tax
    # cases that repeated the caller's word. Coverage still breaks ties between
    # otherwise similar candidates; it does not get to overturn both retrievers.
    relevance = {
        "base": 0.62 * base,
        "coverage": 0.20 * coverage,
        "numeric": 0.08 * numeric,
        "title": 0.10 * title_hits,
    }
    core = sum(relevance.values())
    parts = {key: round(value * 0.92, 4) for key, value in relevance.items()}
    parts["authority"] = round(0.08 * authority * core, 4)
    parts["jurisdiction"] = round(0.04 * jurisdiction_match, 4)
    return round(sum(parts.values()), 6), parts


def rerank(
    query: str,
    ranked: Iterable[tuple[str, float]],
    details: dict[str, dict[str, Any]],
    limit: int,
    jurisdiction: str = "IN",
    terms: list[str] | None = None,
    gate: float = 0.0,
    focus: list[str] | None = None,
    unjudgeable: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """Rerank fused candidates; returns ordered ids and per-id explanations.

    `gate` is the minimum weighted share of the query's vocabulary a candidate
    must contain to be shown at all. Retrieval always returns *something* when
    any word matches, and the caller has no way to tell "this decision is on
    point" from "this decision shares the word refusal". A gate that drops the
    latter is the difference between "closest approved passages" and a list of
    unrelated tax cases presented as authority.
    """
    candidates = list(ranked)
    if not candidates:
        return [], {"features": []}
    top_base = max(base for _, base in candidates) or 1.0
    focus_terms = [str(term) for term in (focus or []) if str(term).strip()]
    # Coverage is measured on the reviewed vocabulary when there is one. The
    # caller's framing is a retrieval hint, not a relevance test: both rankers
    # already agreed on this ordering using it, and re-scoring it here counted
    # "refusal" twice while ignoring whether the judgment used the words the
    # question is actually about. The top fused ranks differ by ~1.6%, so any
    # coverage term built on the caller's paraphrase decides the order outright.
    scoring_terms = focus_terms or list(terms or [])
    weights, shares_vocabulary = term_weights(query, details, candidates, scoring_terms)
    scored = []
    if not shares_vocabulary:
        # No candidate carries any of the vocabulary the question was searched
        # with. For a focus-driven query that is an abstention, not an ordering
        # problem: if not one judgment in the pool uses the issue's words, the
        # store does not have this issue, and returning the fused order would
        # present four arbitrary decisions as related authority.
        if focus_terms:
            return [], {
                "features": [],
                "focus_gate": FOCUS_GATE,
                "dropped": len(candidates),
                "reason": "no candidate carries the reviewed vocabulary for this issue",
            }
        # A vocabulary the corpus cannot contain at all -- a Telugu question
        # against an English store -- leaves the encoder as the only signal, and
        # reordering on authority alone demonstrably lost answers the dense arm
        # had ranked correctly (two of four Hindi retrieval cases). The fused
        # order is the honest result there.
        #
        # A vocabulary the corpus *does* hold, where not one candidate uses it,
        # is a different situation: the question was answerable by this index and
        # nothing in it matches. Returning the fused order there is how
        # "the weather in Mumbai" became four judgments. Abstain instead.
        if not unjudgeable:
            return [], {
                "features": [],
                "gate": gate,
                "dropped": len(candidates),
                "reason": "no candidate shares the question's vocabulary",
            }
        return [source_id for source_id, _ in candidates[:limit]], {
            "features": [],
            "skipped": "no shared vocabulary with the candidate pool",
        }
    for source_id, base in candidates:
        candidate = details.get(source_id)
        if not candidate:
            continue
        value, parts = score(query, candidate, base / top_base, jurisdiction, weights, scoring_terms)
        haystack = _haystack(candidate)
        coverage = sum(weight for term, weight in weights.items() if _contains(term, haystack))
        # The gate counts terms, it does not weight them. Pool-relative IDF
        # answers "which term separates these candidates", and for a focused
        # query the separating term is the caller's loose word rather than the
        # precise one, so a gate built on it dropped the judgment that carried
        # the reviewed vocabulary (cognizable / 154) and kept ones that only
        # repeated "refusal". A count asks the question the gate is actually
        # for: how much of the question is in this passage at all.
        present = sum(1 for term in terms if _contains(term, haystack))
        share = present / len(terms) if terms else 0.0
        # Reviewed issue vocabulary ("cognizable", "154") is the better gate
        # when it exists: the caller cannot be expected to say it, and a
        # judgment that does not use it is not deciding the caller's question
        # even if it repeats their framing.
        focus_present = sum(1 for term in focus_terms if _contains(term, haystack))
        focus_share = focus_present / len(focus_terms) if focus_terms else 0.0
        scored.append((source_id, value, parts, coverage, share, focus_share, present))
    dropped = 0
    if focus_terms:
        kept = [row for row in scored if row[5] >= FOCUS_GATE]
        dropped = len(scored) - len(kept)
        if not kept:
            return [], {
                "features": [],
                "focus_gate": FOCUS_GATE,
                "dropped": dropped,
                "reason": "no candidate carries the reviewed vocabulary for this issue",
            }
        scored = kept
    elif gate > 0:
        # Two terms of a short query, not one. The share test alone let a
        # two-word question through on a single incidental word: "what is the
        # weather in mumbai today" kept the criminal appeals because they say
        # "Mumbai" (1 of 2 = 0.50 >= 0.34). A candidate that reproduces only one
        # word of a two-word question reproduces neither the question nor its
        # subject.
        needed = min(2, len(terms))
        # Weighted coverage, not a term count. A count let two incidental common
        # words through: "plan a birthday party for my daughter" kept criminal
        # appeals because judgments say "party" and "daughter" constantly. The
        # weights are inverse document frequency over this candidate pool, so a
        # word in half the pool contributes little and the one word that
        # actually separates the candidates carries the decision.
        kept = [row for row in scored if row[3] >= gate and row[6] >= needed]
        dropped = len(scored) - len(kept)
        scored = kept
        if not scored:
            return [], {
                "features": [],
                "gate": gate,
                "dropped": dropped,
                "reason": "no candidate carries enough of the question's vocabulary",
            }
    scored.sort(key=lambda item: item[1], reverse=True)
    explanations = [
        {
            "id": source_id,
            "score": value,
            "weighted_coverage": round(coverage, 4),
            "terms_covered": round(share, 3),
            "focus_terms_covered": round(focus_share, 3),
            **parts,
        }
        for source_id, value, parts, coverage, share, focus_share, _ in scored[:limit]
    ]
    explanation: dict[str, Any] = {"features": explanations}
    if focus_terms:
        explanation["focus_gate"] = FOCUS_GATE
        explanation["dropped"] = dropped
    elif gate > 0:
        explanation["gate"] = gate
        explanation["dropped"] = dropped
    return [row[0] for row in scored[:limit]], explanation
