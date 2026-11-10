"""Local retrieval for imported Supreme Court and High Court judgment text.

The source corpora are many orders of magnitude larger than a laptop workspace.
This module indexes only normalized, explicitly ingested partitions and keeps
the court/date/citation metadata attached to every chunk.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from . import dense, judgment_corpus, rerank, retrieval

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime" / "legal-data"
CASE_DB = ROOT / ".runtime" / "cases.sqlite3"
NORMALIZED_CASES = RUNTIME / "judgments.jsonl"
PACKAGED_CASES = ROOT / "data" / "public_judgment_subset.jsonl"
CHUNK_CHARS = 3600
OVERLAP = 420
# Display cap for one quoted passage. The curated index chunks at 3,600
# characters, and returning the whole chunk put a 3,542-character excerpt in
# front of the caller and in the model's context. A prefix cut at a sentence
# boundary stays verbatim, so it can still be quoted exactly.
MAX_EXCERPT = 2000

# How many rows one MATCH expression contributes, and how many fused candidates
# the reranker sees. Both used to scale with the caller's `limit` (8x and 6x),
# which meant a five-result turn built its ranking out of a forty-row window of
# a match set holding thousands of charges: the target of a generic query was
# simply never in the pool. These are properties of the index, not of how many
# results the turn happens to show, so they do not scale with it.
EXPRESSION_ROWS = 96
RERANK_POOL = 60
# Judgments requested from the embedded corpus before fusion. Its own return
# is three times this, which is what the 60-candidate rerank pool needs. Sixteen
# rather than twelve because the leading authority on a question is often rank
# three or four in a passage index whose top hit is a shorter judgment that
# restates it, and fusion only sees what this asks for.
CORPUS_POOL = 16
# Minimum weighted share of the question's vocabulary a judgment must contain
# to be offered as a related decision. Measured against the retrieval set: at
# 0.0 the store returns whatever shares any word (a "refusal of FIR" question
# returned income-tax refusal-of-registration cases); this value keeps enough
# to answer while dropping candidates that only share an incidental word.
RELEVANCE_GATE = 0.34


def case_source_files() -> list[Path]:
    """Return imported partitions first, then the bundled sample.

    Order matters because it decides which copy wins when the same judgment
    appears in both: an imported record carries decision date, citation, bench
    and CNR, while the bundled sample carries none. The index deduplicates on
    a text fingerprint, so the richer copy is the one that survives.
    """
    return [path for path in (NORMALIZED_CASES, PACKAGED_CASES) if path.exists()]


def _chunks(text: str) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    result: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + CHUNK_CHARS)
        if end < len(cleaned):
            boundary = cleaned.rfind(". ", start + CHUNK_CHARS // 2, end)
            if boundary > start:
                end = boundary + 1
        result.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(start + 1, end - OVERLAP)
    return result


def index_imported_cases(force: bool = False) -> int:
    sources = case_source_files()
    if not sources:
        return 0
    stamp = "schema-v3|" + "|".join(f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}" for path in sources)
    CASE_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CASE_DB) as db:
        db.execute("CREATE TABLE IF NOT EXISTS index_state(key TEXT PRIMARY KEY, value TEXT)")
        existing = db.execute("SELECT value FROM index_state WHERE key='source_stamp'").fetchone()
        if not force and existing and existing[0] == stamp:
            # A row count is cached alongside the stamp because COUNT(*) on an
            # FTS5 table walks its shadow tables: measured at 150 ms warm and
            # over a second cold on this index, on every retrieval call.
            if db.execute("SELECT value FROM index_state WHERE key='row_count'").fetchone() is None:
                db.execute(
                    "INSERT OR REPLACE INTO index_state(key,value) VALUES('row_count',?)",
                    (str(int(db.execute("SELECT COUNT(*) FROM case_fts").fetchone()[0])),),
                )
            return int(db.execute("SELECT COUNT(DISTINCT source_id) FROM case_fts").fetchone()[0])
        db.execute("DROP TABLE IF EXISTS case_fts")
        db.execute("CREATE VIRTUAL TABLE case_fts USING fts5(source_id UNINDEXED, title, court, citation, decision_date UNINDEXED, dataset UNINDEXED, language UNINDEXED, jurisdiction UNINDEXED, source_url UNINDEXED, body, tokenize='unicode61 remove_diacritics 2')")
        batch: list[tuple[str, str, str, str, str, str, str, str, str, str]] = []
        source_ids: set[str] = set()
        seen_text: set[str] = set()
        seen_names: set[str] = set()
        for source_path in sources:
            with source_path.open(encoding="utf-8") as source_file:
                for line_number, line in enumerate(source_file, 1):
                    if not line.strip():
                        continue
                    try:
                        case = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid case JSON in {source_path.name} on line {line_number}") from exc
                    required = ("id", "title", "court", "text", "source_url", "decision_date", "dataset")
                    if any(not isinstance(case.get(key), str) for key in required):
                        continue
                    case_id = case["id"].strip()
                    if not case_id or case_id in source_ids:
                        continue
                    # Same judgment imported twice (bundled sample + imported
                    # partition) must not double-count or crowd retrieval.
                    # Hash the whole text, not a prefix: judgments share
                    # boilerplate openings, and a prefix hash would silently
                    # drop distinct decisions.
                    fingerprint = hashlib.sha1(" ".join(case["text"].split()).encode()).hexdigest()
                    if fingerprint in seen_text:
                        continue
                    # Text hashes differ between extraction pipelines (whitespace,
                    # hyphenation), so also match on the source file name. Imports
                    # are read first, so the metadata-rich copy wins.
                    name_key = Path(str(case.get("archive_member") or case.get("title") or "")).name
                    name_key = name_key[:-4].lower() if name_key.lower().endswith(".pdf") else name_key.lower()
                    if name_key.endswith("_en"):
                        name_key = name_key[:-3]
                    if name_key and name_key in seen_names:
                        continue
                    if name_key:
                        seen_names.add(name_key)
                    seen_text.add(fingerprint)
                    source_ids.add(case_id)
                    for chunk in _chunks(case["text"]):
                        batch.append((case_id, case["title"][:500], case["court"][:160], str(case.get("citation") or "")[:240], case["decision_date"][:40], case["dataset"][:40], str(case.get("language") or "unknown")[:30], str(case.get("jurisdiction") or "IN")[:40], case["source_url"][:1000], chunk))
                    if len(batch) >= 500:
                        db.executemany("INSERT INTO case_fts(source_id,title,court,citation,decision_date,dataset,language,jurisdiction,source_url,body) VALUES(?,?,?,?,?,?,?,?,?,?)", batch)
                        batch.clear()
        if batch:
            db.executemany("INSERT INTO case_fts(source_id,title,court,citation,decision_date,dataset,language,jurisdiction,source_url,body) VALUES(?,?,?,?,?,?,?,?,?,?)", batch)
        db.execute("INSERT OR REPLACE INTO index_state(key,value) VALUES('source_stamp',?)", (stamp,))
        db.execute("INSERT OR REPLACE INTO index_state(key,value) VALUES('source_count',?)", (str(len(source_ids)),))
        db.execute("INSERT OR REPLACE INTO index_state(key,value) VALUES('row_count',?)", (str(int(db.execute("SELECT COUNT(*) FROM case_fts").fetchone()[0])),))
        return len(source_ids)


# One weight per store, applied to a list already normalised to 0..1. The
# embedded corpus is the full 10,588-judgment set and carries the topical
# match; the curated partitions are a 2,046-judgment sample of recent Supreme
# Court decisions, useful for currency and citation but not a substitute for
# coverage. Weighting it at 0.75 keeps it in contention without letting four
# recent judgments displace the directly responsive ones on every query.
STORE_WEIGHT = {"legacy": 0.75, "corpus": 1.0}


def _fuse_stores(legacy_lists: list[list[str]], corpus_ids: list[str]) -> dict[str, float]:
    """Fuse stores by normalised rank, not by raw reciprocal-rank mass.

    Reciprocal rank fusion is defined over lists, so a store that runs three
    query arms (strict AND, relaxed AND, OR) accumulates three times the mass of
    a store that returns one ranking -- and wins on bookkeeping rather than
    relevance. Measured on a tenancy question, that put three Supreme Court
    banking judgments above the directly responsive High Court rent cases from
    the passage corpus, for every query.

    Each store is therefore reduced to one list, normalised so its best result
    scores 1.0, and only then combined. A judgment found by both stores -- not
    possible today, since their id spaces are disjoint -- would sum, which is
    the behaviour a real overlap should have.
    """
    scores: dict[str, float] = {}
    if legacy_lists:
        local = retrieval.rrf(legacy_lists)
        top = max(local.values()) or 1.0
        for case_id, value in local.items():
            scores[case_id] = scores.get(case_id, 0.0) + STORE_WEIGHT["legacy"] * (value / top)
    if corpus_ids:
        for rank, case_id in enumerate(corpus_ids, 1):
            # The same 1/(k+rank) shape the local fusion uses, scaled so rank 1
            # is 1.0 whatever the result-set length.
            scores[case_id] = scores.get(case_id, 0.0) + STORE_WEIGHT["corpus"] * (61.0 / (60.0 + rank))
    return scores


_TOTAL_ROWS = 0


def _total_rows(db: sqlite3.Connection) -> int:
    """Chunk count, cached per process: it cannot change under a running app."""
    global _TOTAL_ROWS
    if _TOTAL_ROWS:
        return _TOTAL_ROWS
    try:
        row = db.execute("SELECT value FROM index_state WHERE key='row_count'").fetchone()
        _TOTAL_ROWS = int(row[0]) if row else int(db.execute("SELECT COUNT(*) FROM case_fts").fetchone()[0])
    except sqlite3.Error:
        _TOTAL_ROWS = 0
    return _TOTAL_ROWS


def _excerpt(text: Any) -> str:
    """One bounded, verbatim passage for display and quoting."""
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= MAX_EXCERPT:
        return cleaned
    cut = cleaned[:MAX_EXCERPT]
    boundary = max(cut.rfind(". "), cut.rfind("? "))
    return cut[: boundary + 1].strip() if boundary > MAX_EXCERPT // 2 else cut.strip()


def _ranked_hits(db: sqlite3.Connection, expression: str, jurisdiction: str, limit: int) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Run one MATCH expression, returning case IDs in BM25 order plus rows."""
    try:
        sql = (
            "SELECT source_id,title,court,citation,decision_date,dataset,language,jurisdiction,source_url,body,"
            "bm25(case_fts,0,4.0,3.0,2.0,1.0,0,0,0,0,1.0) AS score FROM case_fts WHERE case_fts MATCH ?"
        )
        params: tuple[Any, ...] = (expression,)
        if jurisdiction != "IN":
            sql += " AND (jurisdiction='IN' OR jurisdiction=?)"
            params += (jurisdiction,)
        rows = db.execute(sql + " ORDER BY score LIMIT ?", params + (EXPRESSION_ROWS,)).fetchall()
    except sqlite3.OperationalError:
        return [], {}
    order: list[str] = []
    detail: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row["source_id"]
        if case_id not in detail:
            order.append(case_id)
            detail[case_id] = {
                "id": case_id,
                "title": row["title"],
                "court": row["court"],
                "court_level": rerank.tier_of({"court": row["court"]}),
                "citation": row["citation"],
                "decision_date": row["decision_date"],
                "dataset": row["dataset"],
                "language": row["language"],
                "jurisdiction": row["jurisdiction"],
                "source_url": row["source_url"],
                "excerpt": _excerpt(row["body"]),
                "source_type": "judgment_dataset_record",
                "treatment_status": "not_checked",
            }
    return order, detail


_DF_CACHE: dict[str, int] = {}


def _document_frequency(db: sqlite3.Connection, term: str) -> int:
    cached = _DF_CACHE.get(term)
    if cached is not None:
        return cached
    quoted = '"' + term.replace('"', '""') + '"'
    try:
        count = int(db.execute("SELECT COUNT(*) FROM case_fts WHERE case_fts MATCH ?", (quoted,)).fetchone()[0])
    except sqlite3.Error:
        count = 0
    _DF_CACHE[term] = count
    return count


WEAK_SUFFIXES = ("ing", "ed")


def rank_terms_by_rarity(db: sqlite3.Connection, terms: list[str], total_rows: int, keep_within: float = 0.35) -> list[str]:
    """Order terms rare-first, strong-before-weak, and drop near-ubiquitous ones.

    A term present in a third of the corpus adds a candidate row to every match
    without discriminating. Separately, English gerund/past-tense forms
    ("returning", "terminated") are rare in legal prose written this way and
    make poor anchors — "deposit" and "salary" carry the meaning. Ranking
    anchors first keeps a two-term strict query on the words that matter.
    """
    if not terms:
        return []
    scored = [(term, _document_frequency(db, term)) for term in terms[:10]]
    useful = [(term, df) for term, df in scored if df > 0 and df <= total_rows * keep_within]
    if not useful:
        # Two different situations look identical here. Either every term is
        # present but ubiquitous -- an OR over them still ranks sensibly, so
        # keep them -- or no term occurs in this index at all, which is what a
        # Hindi question against an English corpus looks like. That second case
        # is not "widen the query": it is "this store has nothing to say".
        # Returning the terms anyway produced a list of accidental matches
        # ("में" appearing in English boilerplate) that fusion then scored at
        # full store weight, outvoting the dense ranking that did answer the
        # question. Hindi queries missed for that reason, not because the
        # encoder was weak.
        return terms if any(df > 0 for _, df in scored) else []
    if len(useful) < max(2, (len(scored) + 2) // 3):
        # A surviving term or two out of a nine-word query is not a shared
        # vocabulary, it is an accident: "में" occurring in English boilerplate
        # once. Two such terms cannot rank anything, and keeping them let the
        # accident speak for the whole store, outvoting the dense ranking that
        # did understand the question. Requiring a third of the query keeps
        # every genuinely English query and drops the ones this index simply
        # does not hold.
        return []
    useful.sort(key=lambda item: (item[0].endswith(WEAK_SUFFIXES), item[1]))
    return [term for term, _ in useful]


def retrieve_cases(query: str, jurisdiction: str = "IN", limit: int = 5, **flags: Any) -> list[dict[str, Any]]:
    return retrieve_cases_with_stats(query, jurisdiction, limit, **flags)[0]


def _prefix_exists(db: sqlite3.Connection, prefix: str) -> bool:
    """Does any indexed word start with this prefix? One row, not a count."""
    quoted = '"' + prefix.replace('"', '""') + '"*'
    try:
        return db.execute(
            "SELECT 1 FROM case_fts WHERE case_fts MATCH ? LIMIT 1", (quoted,)
        ).fetchone() is not None
    except sqlite3.Error:
        return False


def _fetch_details(ids: list[str]) -> dict[str, dict[str, Any]]:
    """Load one representative chunk per unknown judgment, for dense-only hits."""
    if not ids or not CASE_DB.exists():
        return {}
    placeholders = ",".join("?" for _ in ids)
    out: dict[str, dict[str, Any]] = {}
    try:
        with sqlite3.connect(CASE_DB, timeout=5.0) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT source_id,title,court,citation,decision_date,dataset,language,jurisdiction,source_url,body "
                f"FROM case_fts WHERE source_id IN ({placeholders}) GROUP BY source_id",
                ids,
            ).fetchall()
    except sqlite3.Error:
        return {}
    for row in rows:
        out[row["source_id"]] = {
            "id": row["source_id"], "title": row["title"], "court": row["court"],
            "court_level": rerank.tier_of({"court": row["court"]}),
            "citation": row["citation"], "decision_date": row["decision_date"],
            "dataset": row["dataset"], "language": row["language"],
            "jurisdiction": row["jurisdiction"], "source_url": row["source_url"],
            "excerpt": _excerpt(row["body"]), "source_type": "judgment_dataset_record",
            "treatment_status": "not_checked",
        }
    return out


def retrieve_cases_with_stats(
    query: str,
    jurisdiction: str = "IN",
    limit: int = 5,
    use_dense: bool | None = None,
    use_rerank: bool | None = None,
    use_corpus: bool | None = None,
    focus: tuple[str, ...] | list[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Hybrid judgment retrieval across both local stores, then rerank.

    Two stores are consulted because they answer different questions: the
    curated partitions under `.runtime` carry citation, bench and CNR and are
    the only place a specific reported judgment can be traced to a URL, while
    `app/judgment_corpus.py` covers ~10,600 embedded judgments at passage
    granularity. Their rankings are fused, not concatenated, so a judgment that
    both stores find ranks above one only a single store found.

    `use_dense` and `use_rerank` exist so the retrieval eval can measure each
    stage instead of asserting that the combination helps.

    Returns the cases plus diagnostics, because "how many candidates did this
    query consider" is the number that shows whether retrieval actually
    narrowed, and it belongs in the trace rather than in a claim.
    """
    stats: dict[str, Any] = {"terms": [], "focus_terms": [], "expressions_tried": 0, "candidates": 0, "strict_pool": 0, "dense": False, "rerank": "none", "stores": []}
    has_local = CASE_DB.exists()
    # `use_corpus` exists for the same reason as `use_dense`: the retrieval eval
    # is baselined on the curated partitions, and a gate that silently changed
    # the store it measures would stop detecting regressions in the store it was
    # written for.
    has_corpus = judgment_corpus.available() if use_corpus is None else (use_corpus and judgment_corpus.available())
    if not has_local and not has_corpus:
        return [], stats
    terms = retrieval.content_terms(query, limit=12)
    # Reviewed bench vocabulary for this issue ("cognizable offence", "154").
    # A caller cannot be expected to say these words, and a judgment will not
    # say "refused to register an FIR", so the two vocabularies have to be
    # joined somewhere explicit. This is that place.
    # Keep the reviewed vocabulary intact, even where it repeats a caller word.
    # Filtering out terms already in the message shrank the gate to the one
    # synonym the caller happened not to use: "free legal aid counsel" became
    # "counsel", and a case mentioning counsel anywhere passed as legal aid.
    focus_terms = retrieval.content_terms(" ".join(focus), limit=8)
    # Focus first: `rank_terms_by_rarity` probes a bounded prefix, and appending
    # reviewed vocabulary after a long caller sentence dropped it before the
    # probe -- so the refused-FIR story was searched without "cognizable"/"154"
    # while its four-word form was searched with them.
    search_terms = list(dict.fromkeys([*focus_terms, *terms]))
    expanded = retrieval.glossary_terms(search_terms)
    # `terms` stays the store's own searchable vocabulary, filled in below when
    # a store could search at all. The caller's raw words go under
    # `query_terms`, because `terms == []` is what tells the reranker that
    # neither index holds this caller's vocabulary (see `unjudgeable`).
    stats["query_terms"] = terms
    stats["focus_terms"] = focus_terms
    if not search_terms and not expanded:
        return [], stats
    legacy_lists: list[list[str]] = []
    detail: dict[str, dict[str, Any]] = {}
    considered: set[str] = set()

    if has_local:
        stats["stores"].append("cases")
        try:
            with sqlite3.connect(CASE_DB, timeout=5.0) as db:
                db.execute("PRAGMA busy_timeout=5000")
                db.row_factory = sqlite3.Row
                total_rows = _total_rows(db) or 1
                # A misspelt word the index has never seen (voice input is full
                # of them) retrieves nothing at all; the longest prefix the
                # index does know retrieves the word it probably meant.
                resolved = retrieval.resolve_prefixes(
                    search_terms,
                    lambda term: _document_frequency(db, term),
                    lambda prefix: _prefix_exists(db, prefix),
                )
                if resolved != search_terms:
                    stats["corrected_terms"] = resolved
                    search_terms = resolved
                    expanded = retrieval.glossary_terms(search_terms)
                ordered_terms = rank_terms_by_rarity(db, search_terms, total_rows)
                stats["terms"] = ordered_terms
                if not ordered_terms:
                    # No shared vocabulary: the lexical store cannot answer this
                    # query at all, and saying so is better than letting a list
                    # of noise compete with the store that can.
                    stats["lexical_skipped"] = "no query term occurs in this index"
                cascade = retrieval.match_expressions(ordered_terms, expanded) if ordered_terms else []
                # Each arm answers a different way of not matching: the synonym
                # arm covers a caller's word against the bench's word, the stem
                # arm covers refused/refusal/refuse, and the relaxed cascade
                # covers a query whose exact terms never co-occur in one
                # passage. Order is precision-first.
                expressions = (
                    [
                        *cascade[:1],
                        *retrieval.synonym_expressions(ordered_terms),
                        *retrieval.stem_expressions(ordered_terms),
                        *cascade[1:],
                    ]
                    if cascade
                    else []
                )
                for index, expression in enumerate(expressions):
                    order, rows = _ranked_hits(db, expression, jurisdiction, limit)
                    detail.update(rows)
                    considered.update(order)
                    stats["expressions_tried"] = index + 1
                    if index == 0:
                        stats["strict_pool"] = len(order)
                    if order:
                        legacy_lists.append(order)
                    # Every arm runs. Stopping at the first arm that returned
                    # twice the requested limit meant the candidate set was
                    # whatever one expression happened to match, so a judgment
                    # that only the broad arm finds was never considered.
                    if len(considered) >= EXPRESSION_ROWS * 2:
                        break
        except sqlite3.Error:
            pass

    corpus_ids: list[str] = []
    corpus_dense_used = False
    if has_corpus:
        stats["stores"].append("corpus_embeddings")
        # Ask the corpus store for a pool, not for the final answer size.
        # It returns up to three judgments per requested slot, and the fusion
        # below ranks 60 candidates. Passing the caller's limit (4) meant the
        # store contributed twelve judgments, so a decision at rank 13 — the
        # leading authority on the exact question is often not rank 1 — never
        # entered the pool that the reranker is allowed to see at all.
        corpus_ids, corpus_detail, corpus_stats = judgment_corpus.retrieve(
            query, jurisdiction, max(limit, CORPUS_POOL), use_dense=use_dense, focus=focus_terms,
        )
        stats["judgment_corpus"] = corpus_stats
        if corpus_ids:
            detail.update(corpus_detail)
            considered.update(corpus_ids)
            corpus_dense_used = bool(corpus_stats.get("dense"))
    # Dense is a second opinion, not a replacement: a paraphrase or another
    # script contributes a ranking list that lexical search cannot produce,
    # and fusion keeps whichever ranking is confident. When the embedded
    # corpus already supplied a dense ranking over 94,000 passages, re-encoding
    # the query for the smaller local index would cost latency and add nothing.
    want_dense = dense.available() if use_dense is None else (use_dense and dense.available())
    # `ready()` keeps a turn from loading the encoder: the startup task owns that
    # load, and waiting for it inside a request cost 5.6 s on the first turn.
    # `search` applies its own similarity floor. Without one this arm returned
    # the nearest twenty judgments for any input whatsoever, so every store
    # failure turned into a result list.
    dense_ids = (
        dense.search(query, limit=max(limit * 5, 20))
        if want_dense and dense.ready() and not corpus_dense_used
        else []
    )
    if dense_ids:
        stats["dense"] = True
        legacy_lists.append(dense_ids)
        considered.update(dense_ids)
    elif corpus_dense_used:
        stats["dense"] = True

    if not legacy_lists and not corpus_ids:
        # Nothing ranked this query at all: no shared vocabulary and no vectors.
        # This check has to come after the dense arm, because a cross-script
        # query reaches this store *only* through dense.
        stats["candidates"] = len(considered)
        return [], stats

    stats["candidates"] = len(considered)
    fused = sorted(_fuse_stores(legacy_lists, corpus_ids).items(), key=lambda item: item[1], reverse=True)
    missing = [case_id for case_id, _ in fused[:RERANK_POOL] if case_id not in detail]
    if missing:
        detail.update(_fetch_details(missing))

    want_rerank = True if use_rerank is None else use_rerank
    # Can either index hold the caller's words at all? When neither can (a
    # Telugu or Odia question against English judgments), lexical agreement is
    # impossible and the encoder is the only ranking there is. When one can,
    # a candidate set that shares none of the question is not a near miss.
    unjudgeable = not (
        stats.get("terms")
        or (stats.get("judgment_corpus") or {}).get("bm25_passages")
    )
    stats["unjudgeable"] = unjudgeable
    if want_rerank:
        ordered_ids, explanation = rerank.rerank(
            query, fused[:RERANK_POOL], detail, limit, jurisdiction,
            terms=search_terms, gate=RELEVANCE_GATE, focus=list(focus_terms),
            unjudgeable=unjudgeable,
        )
        if ordered_ids:
            stats["rerank"] = "deterministic_features"
            stats["rerank_features"] = explanation["features"]
        # Retrieval always returns *something*; the gate decides whether that
        # something is on point. Saying "nothing relevant was found" is a
        # supported answer. Showing a tax case as a precedent is not.
        if explanation.get("dropped"):
            stats["dropped_irrelevant"] = explanation["dropped"]
        if explanation.get("reason"):
            stats["no_relevant_case"] = explanation["reason"]
    else:
        ordered_ids = [case_id for case_id, _ in fused[:limit]]
    cases = [detail[case_id] for case_id in ordered_ids if case_id in detail]
    # One judgment, one card. The source holds the same decision twice under two
    # ids when it published two URLs for it, and both copies were offered as
    # separate precedents.
    return retrieval.dedupe_cases(cases), stats


def source_url(source_id: str) -> str:
    if not CASE_DB.exists():
        return "https://github.com/vanga/indian-supreme-court-judgments"
    try:
        with sqlite3.connect(CASE_DB) as db:
            row = db.execute("SELECT source_url FROM case_fts WHERE source_id=? LIMIT 1", (source_id,)).fetchone()
            if row and row[0]:
                return str(row[0])
    except sqlite3.Error:
        pass
    return "https://github.com/vanga/indian-supreme-court-judgments"


def corpus_status() -> dict[str, Any]:
    count = 0
    try:
        with sqlite3.connect(CASE_DB) as db:
            row = db.execute("SELECT value FROM index_state WHERE key='source_count'").fetchone()
            count = int(row[0]) if row else 0
    except sqlite3.Error:
        pass
    return {
        "indexed_judgments": count,
        "bundled_judgment_subset": PACKAGED_CASES.exists(),
        "local_normalized_file": NORMALIZED_CASES.exists(),
        "embedded_judgment_corpus": judgment_corpus.status(),
    }

