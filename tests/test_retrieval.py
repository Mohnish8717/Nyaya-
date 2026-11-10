"""Retrieval regressions.

The central claim this suite protects: a query built from an ordinary sentence
must not match almost the whole corpus, and a query with no content terms must
match nothing at all.
"""
from __future__ import annotations

import sqlite3

import pytest

from app import case_corpus, rerank, retrieval

SENTENCE = "my landlord is not returning my security deposit and he is threatening me"


def test_function_words_are_not_query_terms() -> None:
    terms = retrieval.content_terms(SENTENCE)
    for word in ("is", "and", "my", "me", "he", "not"):
        assert word not in terms, f"{word!r} should never be a retrieval term"
    assert "landlord" in terms and "deposit" in terms and "security" in terms


def test_negations_are_surfaced_but_not_indexed() -> None:
    # "not" is dropped from the query, but the turn still knows it was said.
    assert "not" not in retrieval.content_terms(SENTENCE)
    assert retrieval.negations(SENTENCE) == ["not"]
    assert retrieval.match_expressions(retrieval.content_terms("not")) == []


def test_strict_expression_comes_first() -> None:
    # Terms are expected rarest-first, so the AND of the two rarest is the
    # precise query and longer ANDs are relaxations.
    terms = ["deposit", "landlord", "security", "returning"]
    expressions = retrieval.match_expressions(terms)
    assert expressions[0] == '"deposit" AND "landlord"'
    assert any(" OR " in expression for expression in expressions)
    assert [e.count("AND") for e in expressions[:3]] == [1, 2, 3]


def test_no_terms_means_no_expression() -> None:
    assert retrieval.match_expressions([]) == []
    assert retrieval.match_expressions(retrieval.content_terms("the and of")) == []


def test_glossary_maps_romanized_terms() -> None:
    expanded = retrieval.glossary_terms(["kiraya", "tankha", "thana"])
    assert "rent" in expanded
    assert "wages" in expanded or "salary" in expanded
    assert "police" in expanded


def test_section_numbers_survive() -> None:
    terms = retrieval.content_terms("section 138 cheque bounce")
    assert "138" in terms


def test_full_corpus_selectivity() -> None:
    """The documented failure mode: the old OR-of-everything query matched
    1,808 of 1,810 judgments. The cascade must do far better."""
    if not case_corpus.CASE_DB.exists():
        pytest.skip("judgment index not built")

    with sqlite3.connect(case_corpus.CASE_DB) as db:
        total = db.execute("SELECT COUNT(DISTINCT source_id) FROM case_fts").fetchone()[0]

        def count(expression: str) -> int:
            return db.execute(
                "SELECT COUNT(DISTINCT source_id) FROM case_fts WHERE case_fts MATCH ?", (expression,)
            ).fetchone()[0]

        old = count(retrieval.legacy_match(SENTENCE))
        terms = retrieval.content_terms(SENTENCE)
        new = max(count(expression) for expression in retrieval.match_expressions(terms))

    assert old > total * 0.9, "the legacy builder is expected to match almost everything"
    assert new < total * 0.6, f"new builder still matches {new}/{total} judgments"
    assert new < old


def test_stopword_only_query_returns_nothing() -> None:
    if not case_corpus.CASE_DB.exists():
        pytest.skip("judgment index not built")
    assert case_corpus.retrieve_cases("is and the of my he") == []


def test_a_query_with_no_shared_vocabulary_skips_the_lexical_store() -> None:
    """A Hindi question against an English index used to produce one accidental
    survivor ("में" matched boilerplate) whose list fusion then scored at full
    store weight, outvoting the dense ranking that could actually answer it.
    Saying "this store has nothing to say" is the fix, and it is observable in
    the diagnostics rather than silently different."""
    if not case_corpus.CASE_DB.exists():
        pytest.skip("judgment index not built")
    hindi = "सड़क दुर्घटना में मुआवजा न मिलने पर बीमा कंपनी के खिलाफ अपील"
    with sqlite3.connect(case_corpus.CASE_DB) as db:
        total = case_corpus._total_rows(db) or 1
        ordered = case_corpus.rank_terms_by_rarity(db, retrieval.content_terms(hindi), total)
    assert ordered == [], "no term of this query occurs in an English index"

    _cases, stats = case_corpus.retrieve_cases_with_stats(hindi, "IN", 5, use_corpus=False)
    assert stats.get("lexical_skipped"), "the trace must say why lexical returned nothing"
    assert stats["expressions_tried"] == 0


def test_an_english_query_keeps_its_lexical_terms() -> None:
    if not case_corpus.CASE_DB.exists():
        pytest.skip("judgment index not built")
    with sqlite3.connect(case_corpus.CASE_DB) as db:
        total = case_corpus._total_rows(db) or 1
        ordered = case_corpus.rank_terms_by_rarity(db, retrieval.content_terms(SENTENCE), total)
    assert ordered, "a query the index shares words with must keep them"


def test_interleave_keeps_the_first_ranking_in_front() -> None:
    """The caller's own words are the best evidence of what they meant, so a
    rewritten search contributes entries between them, never over them."""
    merged = retrieval.interleave(["a", "b", "c"], ["c", "d", "e"], 5)
    assert merged[:3] == ["a", "c", "b"]
    assert merged == ["a", "c", "b", "d", "e"]
    assert retrieval.interleave(["a"], [], 5) == ["a"]
    assert retrieval.interleave([], [], 5) == []
    assert len(retrieval.interleave(["a", "b"], ["c", "d"], 2)) == 2


def test_reviewed_vocabulary_outranks_a_framing_word() -> None:
    """The reported regression, reduced to its two candidates.

    Fusion put Lalita Kumari first. The old coverage term was built on the
    caller's words, and Lalita's holding uses "section 154" and "cognizable"
    but never says "refusal" -- so coverage demoted the leading authority below
    an income-tax case that repeated the caller's word. Coverage is now measured
    on the reviewed issue vocabulary and the fused rank carries most of the
    score.
    """
    details = {
        "lalita": {
            "id": "lalita",
            "title": "Lalita Kumari v State of U.P.",
            "excerpt": "registration of an FIR under Section 154 for a cognizable offence is mandatory",
            "court_level": "supreme",
            "jurisdiction": "IN",
        },
        "tax": {
            "id": "tax",
            "title": "Refusal of registration of a firm",
            "excerpt": "the refusal to register the firm under the income tax act",
            "court_level": "high",
            "jurisdiction": "IN",
        },
    }
    ids, explanation = rerank.rerank(
        "the refusal of FIR",
        [("lalita", 1.0), ("tax", 0.984)],
        details,
        2,
        terms=["refusal", "fir", "cognizable", "154"],
        focus=["cognizable", "154"],
    )
    assert ids == ["lalita"]
    assert explanation["features"][0]["id"] == "lalita"


def test_the_focus_gate_abstains_rather_than_show_off_issue_cases() -> None:
    """Better an empty list than four judgments about something else."""
    details = {
        "tax": {
            "id": "tax",
            "title": "Refusal of registration",
            "excerpt": "refusal to register a partnership firm under the income tax act",
            "court_level": "high",
            "jurisdiction": "IN",
        }
    }
    ids, explanation = rerank.rerank(
        "the refusal of FIR",
        [("tax", 1.0)],
        details,
        2,
        terms=["refusal", "fir"],
        focus=["cognizable", "154"],
    )
    assert ids == []
    assert explanation["reason"].startswith("no candidate carries the reviewed vocabulary")


def test_retrieve_respects_limit() -> None:
    if not case_corpus.CASE_DB.exists():
        pytest.skip("judgment index not built")
    results = case_corpus.retrieve_cases("landlord security deposit rent", "IN", limit=3)
    assert len(results) <= 3
    for case in results:
        assert case["treatment_status"] == "not_checked"


def test_request_framing_is_not_a_query_term() -> None:
    """The failure this pins: "I want the most relevant cases that cover the
    refusal of FIR" produced the terms [want, relevant, cases, cover, refusal,
    fir]. Terms are ordered rarest-first and the two rarest become the strict
    BM25 anchor, so the anchor was "want" AND "relevant" — request framing beat
    the caller's actual subject, and a refused-FIR question returned income-tax
    "refusal of registration" cases."""
    terms = retrieval.content_terms("I want the most relevant cases that cover the refusal of FIR")
    assert terms == ["refusal", "fir"], terms
    for word in ("want", "relevant", "cases", "cover"):
        assert word not in terms


def test_a_meta_only_question_retrieves_nothing() -> None:
    # "tell me the most relevant cases" is a request, not a subject; returning
    # the corpus's top documents for it would be invented relevance.
    assert retrieval.content_terms("tell me the most relevant cases") == []
    assert retrieval.focus("tell me the most relevant cases") == ""


def test_stem_expression_covers_inflections() -> None:
    """FTS5 does not stem, so "refused" and "refusal" are unrelated terms to
    the index. The stem arm is what lets a caller's word meet the judgment's."""
    expressions = retrieval.stem_expressions(["refusal", "fir"])
    assert expressions == ['("refusal" OR "refus"*) AND "fir"'], expressions
    assert retrieval.stem_expressions(["fir"]) == []


def test_focus_adds_reviewed_vocabulary_without_losing_the_caller() -> None:
    focused = retrieval.focus("the police refused to register my FIR", ["cognizable", "154"])
    assert "fir" in focused and "cognizable" in focused and "154" in focused
    assert "the" not in focused.split()
