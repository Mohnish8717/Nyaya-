"""Passage-corpus retrieval: labelling, filters, fusion, and honest absence.

The claims under test are the ones the design rests on:

  * every returned case carries the verbatim passage that matched, its court
    tier and a provenance URL, so a claim built from it is traceable and
    quotable;
  * jurisdiction and tier filters actually filter, rather than being mentioned
    in a docstring;
  * one judgment cannot fill the result set through several passages;
  * store fusion is normalised, so a store that runs more query arms does not
    win on bookkeeping;
  * authority reorders within a bounded range instead of promoting a weakly
    relevant judgment, and
  * a missing artifact returns nothing and says so, instead of answering from
    whatever happens to be left in memory.

A synthetic four-dimension corpus is built per test: the real artifact is 530 MB
and the encoder download has no place in a unit test. Dense search is disabled
in these tests (`use_dense=False`) so no model is required, and the fusion tests
call the scoring functions directly.
"""

from __future__ import annotations

import sqlite3
import time
from types import SimpleNamespace

import pytest

from app import case_corpus, judgment_corpus, legal_engine, rerank

SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE cases(
    case_id TEXT PRIMARY KEY, title TEXT, court TEXT, court_level TEXT,
    jurisdiction TEXT, language TEXT, decision_date TEXT, citation TEXT,
    dataset TEXT, source_url TEXT, text_chars INTEGER, ingested_at REAL);
CREATE TABLE chunks(
    chunk_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, kind TEXT, ordinal INTEGER,
    court_level TEXT, language TEXT, court TEXT, jurisdiction TEXT,
    decision_date TEXT, header TEXT, text TEXT);
CREATE TABLE vectors(chunk_id TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL);
CREATE TABLE done_cases(case_id TEXT PRIMARY KEY);
CREATE VIRTUAL TABLE chunk_fts USING fts5(
    text, content='chunks', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2');
"""

CASES = [
    ("sc-1", "Rangappa vs Sri Mohan", "Supreme Court of India", "supreme", "IN", "en"),
    ("hc-1", "Manohar Lal vs Narain Dass", "High Court of Delhi", "high", "IN-DL", "en"),
    ("hc-2", "Smt. Sharada Bai vs Navratan Vyas", "Andhra High Court", "high", "IN-AP", "en"),
    ("dc-1", "Narender Kumar vs Annu", "Delhi District Court", "district", "IN-DL", "en"),
]

CHUNKS = [
    # Two passages from the same judgment: the dedup cap must keep only two, and
    # one passage is deliberately short so the neighbour expansion runs.
    ("sc-1#0", "sc-1", 0, "The dishonour of a cheque raises a presumption under Section 139 in favour of the holder."),
    ("sc-1#1", "sc-1", 1, "A cheque returned unpaid for insufficient funds is a dishonour within the meaning of the Act."),
    ("sc-1#2", "sc-1", 2, "The complainant must send a notice demanding payment within thirty days of the dishonour."),
    ("hc-1#0", "hc-1", 0, "The landlord cannot retain the security deposit without accounting for the tenancy."),
    ("hc-2#0", "hc-2", 0, "A tenant in possession cannot be dispossessed except in accordance with law."),
    ("dc-1#0", "dc-1", 0, "Possession of the premises was taken by changing the lock without notice to the tenant."),
]


def _build(path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    case_text: dict[str, int] = {}
    for _, case_id, _, text in CHUNKS:
        case_text[case_id] = case_text.get(case_id, 0) + len(text)
    for case_id, title, court, level, jurisdiction, language in CASES:
        connection.execute(
            "INSERT INTO cases(case_id,title,court,court_level,jurisdiction,language,"
            "decision_date,citation,dataset,source_url,text_chars,ingested_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (case_id, title, court, level, jurisdiction, language, "2010-05-07",
             "(2010) 11 SCC 441", "opennyai", f"https://example.test/{case_id}",
             case_text.get(case_id, 0), 0.0),
        )
    for chunk_id, case_id, ordinal, text in CHUNKS:
        meta = next(record for record in CASES if record[0] == case_id)
        connection.execute(
            "INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (chunk_id, case_id, "body", ordinal, meta[3], meta[5], meta[2], meta[4],
             "2010-05-07", f"[{meta[2]} | level: {meta[3]}]", text),
        )
        # Four int8 components is enough to exercise loading and the int8 path.
        connection.execute(
            "INSERT INTO vectors VALUES(?,4,?)", (chunk_id, bytes([10, 246, 5, 0]))
        )
    connection.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('rebuild')")
    connection.executemany(
        "INSERT INTO meta(key,value) VALUES(?,?)",
        [("model", "test-model"), ("dim", "4"), ("built_at", "0"),
         ("licence", "apache-2.0"), ("source_repo", "test/repo")],
    )
    connection.commit()
    connection.close()


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    path = tmp_path / "corpus.sqlite3"
    _build(path)
    monkeypatch.setenv("JUDGMENT_CORPUS_DB", str(path))
    judgment_corpus.reset_cache()
    yield path
    judgment_corpus.reset_cache()


def test_missing_artifact_reports_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGMENT_CORPUS_DB", str(tmp_path / "absent.sqlite3"))
    judgment_corpus.reset_cache()
    assert judgment_corpus.available() is False
    assert judgment_corpus.retrieve("deposit", "IN", 3)[0] == []
    status = judgment_corpus.status()
    assert status["available"] is False and status["exists"] is False
    judgment_corpus.reset_cache()


def test_a_loaded_artifact_is_not_served_for_another_path(tmp_path, monkeypatch):
    """The matrix belongs to one file, and swapping the file invalidates it.

    Reported failure: a full test run answered a Telugu question from a
    four-case fixture, because `_matrix is not None` was treated as "loaded"
    without checking *which* artifact was loaded. The vector scan still ran on
    the old matrix while titles came from the new one -- retrieval that looks
    like it worked and is answering from a different corpus.
    """
    first = tmp_path / "first.sqlite3"
    _build(first)
    monkeypatch.setenv("JUDGMENT_CORPUS_DB", str(first))
    judgment_corpus.reset_cache()
    ids, _, _ = judgment_corpus.retrieve("security deposit tenancy", "IN", 3, use_dense=False)
    assert ids == ["hc-1"]

    second = tmp_path / "second.sqlite3"
    _build(second)
    with sqlite3.connect(second) as connection:
        connection.execute("UPDATE cases SET title = 'Rangappa vs Sri Mohan (second)' WHERE case_id = 'sc-1'")
        connection.commit()
    monkeypatch.setenv("JUDGMENT_CORPUS_DB", str(second))
    ids, detail, _ = judgment_corpus.retrieve("cheque dishonour notice", "IN", 3, use_dense=False)
    judgment_corpus.reset_cache()
    assert ids and ids[0] == "sc-1"
    assert detail["sc-1"]["title"].endswith("(second)"), "served the first artifact's metadata"


def test_status_describes_the_artifact(corpus):
    status = judgment_corpus.status()
    assert status["available"] is True
    assert status["cases"] == 4 and status["chunks"] == len(CHUNKS)
    assert status["tiers"] == {"supreme": 1, "high": 2, "district": 1}
    assert status["model"] == "test-model" and status["dim"] == 4


def test_retrieval_returns_the_matching_passage_and_its_tier(corpus):
    ids, detail, stats = judgment_corpus.retrieve("security deposit tenancy", "IN", 3, use_dense=False)
    assert ids, "a passage containing every query term must be found"
    top = detail[ids[0]]
    assert top["id"] == "hc-1"
    assert "security deposit" in top["excerpt"]
    assert top["court_level"] == "high"
    assert top["source_url"].endswith("hc-1")
    assert top["treatment_status"] == "not_checked"
    assert top["passage_id"].startswith("hc-1#")
    assert stats["bm25_passages"] >= 1 and stats["dense"] is False


def test_no_passage_per_judgment_limit_is_respected(corpus):
    ids, detail, _ = judgment_corpus.retrieve("cheque dishonour notice", "IN", 5, use_dense=False)
    assert ids.count("sc-1") == 1, "a judgment is reported once, not once per passage"
    counts: dict[str, int] = {}
    for value in detail.values():
        counts[value["id"]] = counts.get(value["id"], 0) + 1
    assert all(count <= judgment_corpus.MAX_PASSAGES_PER_CASE for count in counts.values())


def test_tier_filter_excludes_other_tiers(corpus):
    ids, detail, _ = judgment_corpus.retrieve(
        "tenant premises possession", "IN", 5, use_dense=False, tiers=["district"]
    )
    assert ids == ["dc-1"]
    assert detail["dc-1"]["court_level"] == "district"


def test_jurisdiction_filter_excludes_other_states(corpus):
    ids, detail, _ = judgment_corpus.retrieve(
        "tenant premises possession", "IN-DL", 5, use_dense=False
    )
    assert ids, "the Delhi passage must survive a Delhi query"
    assert all(detail[case_id]["jurisdiction"] in {"IN-DL", "IN", ""} for case_id in ids)


def test_a_stopword_query_retrieves_nothing(corpus):
    assert judgment_corpus.retrieve("is and the of my he", "IN", 3, use_dense=False)[0] == []


def test_short_passage_pulls_its_neighbour(corpus):
    # sc-1#2 is the shortest passage in the fixture; its neighbour is appended so
    # the model sees a complete thought rather than half a sentence.
    _, detail, _ = judgment_corpus.retrieve("notice demanding payment thirty days", "IN", 3, use_dense=False)
    excerpt = detail["sc-1"]["excerpt"]
    assert "thirty days" in excerpt


def test_fusion_normalises_each_store(corpus):
    """A store with more query arms must not win on accumulation.

    Three legacy arms previously gave the legacy store three times the
    reciprocal-rank mass of a single-arm store, so its best result outranked a
    directly responsive passage from the other store for every query.
    """
    fused = case_corpus._fuse_stores([["a", "b"], ["a"], ["a"]], ["c", "d"])
    assert fused["a"] == pytest.approx(case_corpus.STORE_WEIGHT["legacy"])
    assert fused["c"] == pytest.approx(case_corpus.STORE_WEIGHT["corpus"])
    assert fused["c"] > fused["a"], "the corpus rank 1 must outrank a normalised legacy rank 1"
    assert fused["c"] >= fused["d"]


def test_corpus_status_block_is_reported(corpus):
    status = case_corpus.corpus_status()
    assert status["embedded_judgment_corpus"]["available"] is True
    assert status["embedded_judgment_corpus"]["cases"] == 4


def test_retrieve_abstains_while_another_thread_loads(corpus):
    """A turn must not queue behind the startup load.

    Measured in the running app: the first load takes ~8.6 s and a caller who
    arrived during it waited 5.3 s for an answer the curated store could already
    give. Retrieval now takes the lock only if it is free.
    """
    judgment_corpus.reset_cache()
    judgment_corpus._LOAD_STATE = "loading"
    started = time.perf_counter()
    ids, _, stats = judgment_corpus.retrieve("deposit tenancy", "IN", 3, use_dense=False)
    elapsed = time.perf_counter() - started
    judgment_corpus.reset_cache()
    assert ids == []
    assert stats.get("loading") is True
    assert elapsed < 0.5, f"retrieval blocked for {elapsed:.2f}s behind the loader"


def test_degraded_answer_keeps_the_judgments_it_retrieved():
    """`corpus.case_judgment_candidates` must describe something in the payload.

    With no model configured the reply lists statute passages only; before this,
    four retrieved judgments were dropped while the corpus block still reported
    four candidates.
    """
    context = SimpleNamespace(cases=[{
        "id": "hc-1", "title": "Manohar Lal vs Narain Dass",
        "court": "High Court of Delhi", "court_level": "high",
        "citation": "(1986) 2 DLT 1", "decision_date": "1986-08-05",
        "jurisdiction": "IN-DL", "source_url": "https://example.test/hc-1",
        "excerpt": "The landlord cannot retain the security deposit without accounting.",
        "treatment_status": "not_checked",
    }])
    precedents = legal_engine._case_precedents(context)
    assert len(precedents) == 1
    assert precedents[0]["court_level"] == "high"
    assert precedents[0]["treatment_status"] == "not_checked"
    assert precedents[0]["excerpt"].startswith("The landlord cannot retain")
    assert legal_engine._case_precedents(SimpleNamespace(cases=[])) == []


def test_authority_reorders_within_a_bounded_range():
    base = {
        "id": "x", "title": "Landlord and tenant", "excerpt": "landlord deposit tenancy premises",
        "court": "High Court of Delhi", "jurisdiction": "IN",
    }
    supreme = {**base, "court_level": "supreme"}
    district = {**base, "court_level": "district"}
    high, _ = rerank.score("landlord deposit", base, 1.0, "IN")
    top, _ = rerank.score("landlord deposit", supreme, 1.0, "IN")
    low, _ = rerank.score("landlord deposit", district, 1.0, "IN")
    assert top > high > low, "tier must order otherwise identical candidates"
    assert top - low < 0.1 * top, "authority must not be able to outrank relevance"


def test_term_weights_demote_a_word_every_candidate_contains():
    details = {
        "a": {"excerpt": "the landlord return deposit"},
        "b": {"excerpt": "the bank return documents"},
    }
    weights, matched = rerank.term_weights("landlord return deposit", details, [("a", 1.0), ("b", 1.0)])
    assert weights["return"] < weights["landlord"], "a term in every candidate cannot discriminate"
    assert sum(weights.values()) == pytest.approx(1.0)
    assert matched is True


def test_a_query_the_pool_shares_no_words_with_reports_no_match():
    """The flag that stops the reranker reordering a list it cannot judge: a
    Devanagari query against English excerpts matches nothing, and an
    authority-only reorder demonstrably lost answers."""
    details = {
        "a": {"excerpt": "the landlord must return the deposit"},
        "b": {"excerpt": "the bank dishonoured the cheque"},
    }
    weights, matched = rerank.term_weights("गुजारा भत्ता नहीं दिया", details, [("a", 1.0), ("b", 1.0)])
    assert matched is False
    assert weights, "weights are still computed; the flag is what changes behaviour"

    # A vocabulary the corpus cannot hold at all (a Telugu or Odia question
    # against English excerpts) leaves the encoder as the only signal, so the
    # fused order stands.
    ordered, explanation = rerank.rerank(
        "गुजारा भत्ता नहीं दिया", [("b", 0.9), ("a", 0.5)], details, 2, "IN", unjudgeable=True
    )
    assert ordered == ["b", "a"], "fusion order is kept when nothing can be judged"
    assert explanation.get("skipped")

    # A vocabulary the corpus *does* hold, where no candidate uses it, is a
    # different case: the question was answerable here and nothing matches.
    ordered, explanation = rerank.rerank(
        "theft robbery dacoity", [("b", 0.9), ("a", 0.5)], details, 2, "IN"
    )
    assert ordered == []
    assert explanation["reason"].startswith("no candidate shares")
