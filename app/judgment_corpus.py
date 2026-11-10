"""Passage-level retrieval over the embedded judgment corpus.

`legal_pipeline/` turns the OpenNyAI judgment dump into one SQLite file with
three tables that matter here: `chunks` (3,600-character passages, each carrying
its court tier, jurisdiction and language), `vectors` (one int8 unit vector per
passage) and `chunk_fts` (an external-content FTS5 index over the same text, so
the passage is stored once). This module is the read side of that artifact.

Why passage-level instead of judgment-level: a judgment is 40,000 characters of
cause-title, appearances, facts, arguments and only then the holding. Embedding
it whole averages all of that into a vector that matches nothing in particular,
and returning it whole spends the model's context on the least useful part.
Retrieval therefore runs on passages, then reports the enclosing judgment
(title, court, citation, tier) with the passage that actually matched.

Design rules, each of which is a failure mode from the simpler version:

  * Two independent rankings (BM25 and cosine) are fused with reciprocal rank
    fusion rather than compared by raw score. BM25 scores and cosine scores are
    not on the same scale, and a threshold picked on one silently drops the
    other.
  * Court tier is carried on every passage and surfaced to the caller. A
    district-court order and a Supreme Court judgment are not the same
    authority, and the system must not present them as if they were.
  * At most two passages per judgment, so one long decision cannot fill the
    result set and hide the rest of the corpus.
  * A dense-only hit must clear a similarity floor. Without it, every query
    returns its nearest neighbours in a 94,000-passage space whether or not
    anything is relevant — the "retrieval always returns something" failure.
  * The excerpt returned is the passage verbatim, never a summary: downstream
    claim checking matches exact substrings against it.
  * Latency is bounded by construction: BM25 runs on an index, and the vector
    scan is a block-wise int8 dot product that peaks at a few megabytes rather
    than materialising 94,000 × 384 floats.

Everything degrades: no artifact, no numpy, or no encoder means this module
reports itself unavailable and `case_corpus` falls back to its own index instead
of pretending it searched.
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from . import retrieval

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = (
    ROOT / "legal_pipeline" / "out" / "corpus_embeddings.sqlite3",
    ROOT / ".runtime" / "corpus_embeddings.sqlite3",
)
ENABLED = os.getenv("JUDGMENT_CORPUS_ENABLED", "auto").strip().lower() not in {"0", "false", "off", "no"}

# Authority weight per court tier. Used only for ordering, never to hide a
# tier: a caller asking about a district court practice needs district orders.
TIER_WEIGHT = {"supreme": 1.0, "high": 0.62, "district": 0.30, "unknown": 0.44}
TIER_INDEX = {"supreme": 0, "high": 1, "district": 2, "unknown": 3}
TIER_NAME = {value: key for key, value in TIER_INDEX.items()}

MAX_PASSAGES_PER_CASE = 2
MAX_EXCERPT = 2000
# How much a judgment's passage depth can lift it over its single best passage.
# Bounded at 50%: depth reorders judgments that are already close, it cannot
# promote one that matched weakly over one that matched strongly.
DEPTH_BOOST = 0.5
# Similarity floor for a dense-only passage hit. 0.28 admitted almost any
# neighbour: a Telugu wage question scored 0.34 and a gibberish string 0.35,
# while genuine cross-script questions sit at 0.45-0.66 (Tamil eviction 0.45,
# Marathi rent 0.49, Bengali wages 0.50, Gujarati legal aid 0.55, English
# wages 0.57). 0.45 is the smallest floor that separates those populations on
# this artifact; it is a guard on one arm, not the relevance decision.
MIN_DENSE = float(os.getenv("JUDGMENT_CORPUS_MIN_DENSE", "0.45"))
SCAN_BLOCK = 16384
# How many matched passages are scored before the top-k cutoff. bm25() runs
# per row, so scoring the whole match set is what makes a relaxed OR over a
# 94,000-passage index cost hundreds of milliseconds; capping the pool first
# keeps the same top results for typical queries at a thousandth of the cost.
SCORE_POOL = int(os.getenv("JUDGMENT_CORPUS_SCORE_POOL", "800"))
DF_PROBE_CAP = 2000
DF_PROBE_TERMS = 6

_lock = threading.Lock()
_matrix: Any = None
_chunk_ids: list[str] = []
_case_ids: list[str] = []
_case_of: Any = None
_tier_of: Any = None
_jurisdiction_of: Any = None
_JURISDICTIONS: list[str] = []
_CHUNK_INDEX: dict[str, int] = {}
_loaded_stamp: tuple[str, float] | None = None
_df_cache: dict[str, int] = {}
_TOTAL_CHUNKS = 0
# "idle" (nobody has loaded and nobody is loading) | "loading" | "ready".
# A turn loads only from "idle": once the startup warmer owns the load, no turn
# may start a second one, because whichever thread loads pays the read.
_LOAD_STATE = "idle"
_count_cache: dict[str, tuple[tuple[str, float], dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# Artifact discovery and status
# ---------------------------------------------------------------------------
def reset_cache() -> None:
    """Drop every cached artefact-derived value.

    The caches are keyed by the artifact's path and modification time, so this
    exists for tests that build a new artifact at the same path.
    """
    global _matrix, _chunk_ids, _case_ids, _case_of, _tier_of, _jurisdiction_of
    global _JURISDICTIONS, _CHUNK_INDEX, _loaded_stamp, _TOTAL_CHUNKS, _LOAD_STATE
    with _lock:
        _matrix, _case_of, _tier_of, _jurisdiction_of = None, None, None, None
        _chunk_ids, _case_ids, _JURISDICTIONS, _CHUNK_INDEX = [], [], [], {}
        _loaded_stamp, _TOTAL_CHUNKS = None, 0
        _LOAD_STATE = "idle"
        _df_cache.clear()
        _count_cache.clear()


def db_path() -> Path:
    override = os.getenv("JUDGMENT_CORPUS_DB", "").strip()
    if override:
        return Path(override).expanduser()
    for candidate in DEFAULT_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_CANDIDATES[0]


def _connect(path: Path | None = None) -> sqlite3.Connection:
    """Read-only connection: the artifact is an input, never something to write."""
    target = path or db_path()
    connection = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=5000")
    # A 530 MB artifact served page-by-page makes the first query of a new
    # wording cost hundreds of milliseconds. A memory map plus a large page
    # cache turns that into ordinary reads; both are read-only-safe.
    connection.execute("PRAGMA mmap_size=1073741824")
    connection.execute("PRAGMA cache_size=-65536")
    return connection


def _meta(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        return dict(connection.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.Error:
        return {}


def _probe(path: Path) -> bool:
    """Does the artifact hold passages? A file question, not a load question.

    Deliberately independent of `_LOAD_STATE`: the startup warmer has to be able
    to ask it while it owns the load, which is exactly when `available()` says no.
    """
    try:
        with _connect(path) as connection:
            # EXISTS, not COUNT(*): the question is whether the artifact holds
            # passages, and counting 94,843 rows is a full scan of a cold file.
            return connection.execute("SELECT 1 FROM chunks LIMIT 1").fetchone() is not None
    except sqlite3.Error:
        return False


def available() -> bool:
    """True only when the artifact exists, holds passages, and is not loading."""
    if not ENABLED:
        return False
    path = db_path()
    if not path.exists():
        return False
    if _LOAD_STATE == "loading":
        # The startup warmer owns the artifact. Probing it now competes with the
        # warmer for a 530 MB file, and `retrieve` abstains for this turn anyway
        # because the load is not finished -- so the honest answer is "not yet",
        # and the turn is answered from the curated store in ~100 ms instead of
        # waiting 4.2 s for a probe it cannot use. Measured: first turn after a
        # restart was 4,265 ms, of which 4,240 ms was this stage.
        return False
    stamp = (str(path), path.stat().st_mtime)
    cached = _count_cache.get("probe")
    # Keyed by path as well as mtime: two artifacts can share a timestamp, and
    # answering "present" for the other one is how retrieval reported a corpus
    # it had not opened.
    if cached and cached[0] == stamp:
        return bool(cached[1].get("chunks"))
    present = _probe(path)
    _count_cache["probe"] = (stamp, {"chunks": int(present), "vectors": int(present)})
    return present


def status() -> dict[str, Any]:
    """What is on disk, described honestly enough for `/api/health`."""
    path = db_path()
    out: dict[str, Any] = {
        "enabled": ENABLED,
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "exists": path.exists(),
        "available": False,
    }
    if not path.exists():
        return out
    try:
        with _connect(path) as connection:
            meta = _meta(connection)
            cases = int(connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0])
            chunks = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            vectors = int(connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])
            tiers = {
                row[0] or "unknown": int(row[1])
                for row in connection.execute("SELECT court_level, COUNT(*) FROM cases GROUP BY 1")
            }
    except sqlite3.Error as exc:
        out["error"] = type(exc).__name__
        return out
    out.update({
        "available": chunks > 0,
        "cases": cases,
        "chunks": chunks,
        "vectors": vectors,
        "tiers": tiers,
        "model": meta.get("model", ""),
        "dim": int(meta.get("dim", 0) or 0),
        "built_at": int(meta.get("built_at", 0) or 0),
        "licence": meta.get("licence", ""),
        "source_repo": meta.get("source_repo", ""),
        "size_mb": round(path.stat().st_size / 1_048_576, 1),
    })
    return out


# ---------------------------------------------------------------------------
# Vector matrix (loaded once, scanned in bounded blocks)
# ---------------------------------------------------------------------------
def _load(connection: sqlite3.Connection, wait: bool = False) -> bool:
    """Load the int8 matrix and its per-passage labels, once per artifact.

    A turn never waits for this. Loading the 530 MB artifact measured 8.6 s
    cold, and a caller who arrived during the startup warm waited 5.3 s for an
    answer the curated store could already give. So a retrieval attempt takes
    the lock only if it is free: if another thread is loading, this turn answers
    from the other store and `stats['loading']` says why the corpus abstained.
    """
    global _matrix, _chunk_ids, _case_ids, _case_of, _tier_of, _jurisdiction_of
    global _JURISDICTIONS, _CHUNK_INDEX, _loaded_stamp, _TOTAL_CHUNKS, _df_cache, _LOAD_STATE
    import numpy

    path = db_path()
    stamp = (str(path), path.stat().st_mtime)
    if _matrix is not None and _loaded_stamp == stamp:
        return True
    if wait:
        _lock.acquire()
    elif not _lock.acquire(blocking=False):
        return False
    try:
        _LOAD_STATE = "loading"
        if _matrix is not None and _loaded_stamp == stamp:
            return True
        meta = _meta(connection)
        dim = int(meta.get("dim", 0) or 0)
        rows = connection.execute("SELECT chunk_id, vec FROM vectors").fetchall()
        if not rows or dim <= 0:
            return False
        blob = b"".join(bytes(row["vec"]) for row in rows)
        if len(blob) != len(rows) * dim:
            return False
        matrix = numpy.frombuffer(blob, dtype=numpy.int8).reshape(-1, dim)
        chunk_ids = [row["chunk_id"] for row in rows]

        position = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
        case_ids: list[str] = []
        case_index: dict[str, int] = {}
        jurisdictions: list[str] = []
        jurisdiction_index: dict[str, int] = {}
        case_of = numpy.zeros(len(chunk_ids), dtype="int32")
        tier_of = numpy.full(len(chunk_ids), TIER_INDEX["unknown"], dtype="int8")
        jurisdiction_of = numpy.zeros(len(chunk_ids), dtype="int32")
        for row in connection.execute(
            "SELECT chunk_id, case_id, court_level, jurisdiction FROM chunks"
        ):
            slot = position.get(row["chunk_id"])
            if slot is None:
                continue
            case_id = row["case_id"] or ""
            if case_id not in case_index:
                case_index[case_id] = len(case_ids)
                case_ids.append(case_id)
            case_of[slot] = case_index[case_id]
            tier_of[slot] = TIER_INDEX.get(str(row["court_level"] or "").lower(), TIER_INDEX["unknown"])
            jurisdiction = str(row["jurisdiction"] or "IN")
            if jurisdiction not in jurisdiction_index:
                jurisdiction_index[jurisdiction] = len(jurisdictions)
                jurisdictions.append(jurisdiction)
            jurisdiction_of[slot] = jurisdiction_index[jurisdiction]

        _matrix, _chunk_ids, _case_ids = matrix, chunk_ids, case_ids
        _case_of, _tier_of, _jurisdiction_of = case_of, tier_of, jurisdiction_of
        _JURISDICTIONS = jurisdictions
        _CHUNK_INDEX = position
        _loaded_stamp = stamp
        # Frequency caches are per-artifact; a rebuilt corpus invalidates them.
        _TOTAL_CHUNKS, _df_cache = 0, {}
        _LOAD_STATE = "ready"
        return True
    except Exception:
        # Leave the state idle so the next turn may retry; a failure here is
        # usually a transient read, and the curated store keeps answering.
        _LOAD_STATE = "idle"
        raise
    finally:
        _lock.release()


def claim_load() -> bool:
    """Reserve the artifact load for the startup warmer, before any turn.

    `warm` claims the state itself, but it runs in a worker thread that the
    event loop may not schedule before the first request arrives -- measured: a
    turn landing in that window found "idle", started its own load inside the
    request, and answered in 8.6 s where the curated store answers in ~100 ms.
    Claiming synchronously at startup closes the window entirely.
    """
    global _LOAD_STATE
    if _LOAD_STATE == "idle":
        _LOAD_STATE = "loading"
        return True
    return False


def warm() -> dict[str, Any]:
    """Load the matrix and encoder ahead of the first caller. Never raises."""
    global _LOAD_STATE
    started = time.perf_counter()
    # Claim the load before the first file read. `available()` alone takes
    # seconds on a cold 530 MB artifact, and a turn arriving in that window used
    # to see "nobody is loading" and start its own load inside the request --
    # measured at 2.5 s where the curated store answers in ~40 ms.
    if _LOAD_STATE == "idle":
        _LOAD_STATE = "loading"
    # `_probe`, not `available()`: the warmer owns the load, so `available()`
    # answers no by design -- asking it here made the warmer conclude the corpus
    # was missing, reset the state, and hand the 4 s load back to the first turn.
    if not _probe(db_path()):
        _LOAD_STATE = "idle"
        return {"warmed": False, "reason": "corpus unavailable"}
    try:
        with _connect() as connection:
            loaded = _load(connection, wait=True)
        from . import dense

        if dense.available():
            dense.embed(["warm the retrieval encoder"])
        # Probing representative queries here was tried and reverted: it cost
        # 9 s of startup and still left the first real query at ~580 ms, because
        # the pages a probe touches are not the pages a different question needs.
        # Loading the matrix and the encoder is the part that pays off.
        return {"warmed": bool(loaded), "ms": round((time.perf_counter() - started) * 1000)}
    except Exception as exc:  # pragma: no cover - environment dependent
        # Release the claim so a later turn may retry the read.
        _LOAD_STATE = "idle"
        return {"warmed": False, "reason": type(exc).__name__}


def _mask(jurisdiction: str, tiers: Sequence[str] | None):
    import numpy

    mask = numpy.ones(len(_chunk_ids), dtype=bool)
    if tiers:
        wanted = [TIER_INDEX.get(str(tier).lower(), TIER_INDEX["unknown"]) for tier in tiers]
        mask &= numpy.isin(_tier_of, wanted)
    if jurisdiction and jurisdiction != "IN" and _JURISDICTIONS:
        wanted_ids = [index for index, name in enumerate(_JURISDICTIONS) if name in {jurisdiction, "IN", ""}]
        if wanted_ids:
            mask &= numpy.isin(_jurisdiction_of, wanted_ids)
    return mask


def _dense_scores(vector: Any, mask: Any, top: int) -> list[tuple[str, float]]:
    """Block-wise int8 dot product; peak extra memory is one block, not the corpus."""
    import numpy

    matrix = _matrix
    total = matrix.shape[0]
    if total == 0:
        return []
    scores = numpy.empty(total, dtype="float32")
    for start in range(0, total, SCAN_BLOCK):
        stop = min(total, start + SCAN_BLOCK)
        scores[start:stop] = matrix[start:stop].astype("float32") @ vector
    scores /= 127.0
    scores[~mask] = -numpy.inf
    keep = min(max(top, 1), total)
    if keep < total:
        indices = numpy.argpartition(-scores, keep - 1)[:keep]
    else:
        indices = numpy.arange(total)
    indices = indices[numpy.argsort(-scores[indices])]
    return [
        (_chunk_ids[int(index)], float(scores[int(index)]))
        for index in indices
        if numpy.isfinite(scores[int(index)])
    ]


# ---------------------------------------------------------------------------
# Lexical pass
# ---------------------------------------------------------------------------
def _document_frequency(connection: sqlite3.Connection, term: str) -> int:
    """Capped document frequency, for ordering terms without a full scan.

    A term in ``DF_PROBE_CAP`` passages is reported as ``DF_PROBE_CAP``: the
    probe only needs to know that a term is common, because common terms are
    demoted to the relaxed queries. Counting exactly would read every posting.
    """
    if term in _df_cache:
        return _df_cache[term]
    quoted = '"' + term.replace('"', '""') + '"'
    try:
        count = int(connection.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM chunk_fts WHERE chunk_fts MATCH ? LIMIT ?)",
            (quoted, DF_PROBE_CAP),
        ).fetchone()[0])
    except sqlite3.Error:
        count = 0
    _df_cache[term] = count
    return count


def _total_chunks(connection: sqlite3.Connection) -> int:
    """Passage count from the plain table, not from the FTS index.

    `COUNT(*)` on an external-content FTS5 table costs an index walk measured
    in hundreds of milliseconds; `chunks` is an ordinary table.
    """
    global _TOTAL_CHUNKS
    if _TOTAL_CHUNKS:
        return _TOTAL_CHUNKS
    try:
        _TOTAL_CHUNKS = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    except sqlite3.Error:
        _TOTAL_CHUNKS = 0
    return _TOTAL_CHUNKS


def _prefix_exists(connection: sqlite3.Connection, prefix: str) -> bool:
    """Does any indexed word start with this prefix? One row, not a count."""
    quoted = '"' + prefix.replace('"', '""') + '"*'
    try:
        return connection.execute(
            "SELECT 1 FROM chunk_fts WHERE chunk_fts MATCH ? LIMIT 1", (quoted,)
        ).fetchone() is not None
    except sqlite3.Error:
        return False


def _ordered_terms(connection: sqlite3.Connection, terms: list[str]) -> list[str]:
    """Rarest first, since the cascade treats the two rarest as the precise query.

    A term present in a third of 94,000 passages adds a candidate everywhere and
    discriminates nowhere; it is dropped from the anchor position but may still
    appear in the relaxed OR.
    """
    if not terms:
        return []
    total = _total_chunks(connection) or 1
    scored = [(term, _document_frequency(connection, term)) for term in terms[:DF_PROBE_TERMS]]
    useful = [(term, df) for term, df in scored if df > 0 and df < total * 0.35]
    if not useful:
        return terms
    useful.sort(key=lambda item: item[1])
    return [term for term, _ in useful]


def _lexical(connection: sqlite3.Connection, terms: list[str], expanded: list[str], jurisdiction: str, tiers: Sequence[str] | None, limit: int) -> tuple[list[str], int]:
    """Cascading BM25 over passages; returns ordered chunk ids and tries used."""
    # Strict AND first, then the caller's word paired with its legal synonyms,
    # then the relaxed AND/OR cascade: each arm is looser than the one before.
    # The synonym arm sits second because it is the second most precise thing
    # available; placing it after the relaxed arms meant a query that already
    # matched loosely never reached it.
    cascade = retrieval.match_expressions(terms, expanded)
    # The stem arm is what makes a caller's "refused" meet a judgment's
    # "refusal": FTS5 does not stem, so without it those are unrelated words.
    expressions = [
        *cascade[:1],
        *retrieval.synonym_expressions(terms),
        *retrieval.stem_expressions(terms),
        *cascade[1:],
    ]
    if not expressions:
        return [], 0
    filters: list[str] = []
    params: list[Any] = []
    if tiers:
        filters.append(f"c.court_level IN ({','.join('?' for _ in tiers)})")
        params.extend(tiers)
    if jurisdiction and jurisdiction != "IN":
        filters.append("(c.jurisdiction = ? OR c.jurisdiction = 'IN' OR c.jurisdiction = '')")
        params.append(jurisdiction)
    where = (" AND " + " AND ".join(filters)) if filters else ""
    order: list[str] = []
    seen: set[str] = set()
    tried = 0
    for expression in expressions:
        tried += 1
        try:
            # Rank the whole match set, not a sample of it.
            #
            # The previous form took the first SCORE_POOL rowids out of the
            # index and ranked only those. `LIMIT` on an FTS5 scan returns
            # rows in index order, not in relevance order, so on a broad
            # expression it ranked an arbitrary window of the matches and the
            # genuinely best passages were almost never in it: the leading
            # authority on mandatory FIR registration sat outside an arbitrary
            # 96-row sample of a 20,000-row match set. bm25 in ORDER BY costs a
            # top-k sort over the matches, which is the price of correct recall
            # here; measured at 20-60 ms for the expressions this cascade uses.
            rows = connection.execute(
                "SELECT c.chunk_id AS chunk_id, bm25(chunk_fts, 1.0) AS score "
                "FROM chunk_fts JOIN chunks c ON c.rowid = chunk_fts.rowid "
                f"WHERE chunk_fts MATCH ?{where} "
                "ORDER BY score LIMIT ?",
                (expression, *params, max(limit * 6, 24)),
            ).fetchall()
        except sqlite3.OperationalError:
            continue
        for row in rows:
            chunk_id = row["chunk_id"]
            if chunk_id not in seen:
                seen.add(chunk_id)
                order.append(chunk_id)
        if len(order) >= max(limit * 8, 48):
            break
    return order, tried


# ---------------------------------------------------------------------------
# Public retrieval
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    jurisdiction: str = "IN",
    limit: int = 5,
    use_dense: bool | None = None,
    tiers: Sequence[str] | None = None,
    focus: Sequence[str] = (),
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Return ranked judgment ids, their details, and diagnostics.

    Ranked ids are what the caller-facing fusion consumes; the details carry the
    verbatim passage that matched, plus the judgment's court, tier, citation and
    provenance so a claim built from it can be traced to a quotable source.
    """
    stats: dict[str, Any] = {
        "store": "corpus_embeddings",
        "terms": [],
        "expressions_tried": 0,
        "bm25_passages": 0,
        "dense_passages": 0,
        "candidates": 0,
        "dense": False,
        "tiers": list(tiers or []),
    }
    # Checked before `available()`, which is itself a read of the artifact.
    if _LOAD_STATE == "loading":
        stats["loading"] = True
        return [], {}, stats
    if not available() or not query.strip():
        return [], {}, stats
    terms = retrieval.content_terms(query, limit=12)
    # The reviewed vocabulary is kept whole even where it repeats a caller word:
    # subtracting those terms made the focus gate as weak as the single synonym
    # the caller happened not to use.
    focus_terms = retrieval.content_terms(" ".join(focus), limit=8)
    # Focus first, then the caller's words. `_ordered_terms` probes only the
    # first six terms, and appending focus put it last: a caller sentence of six
    # words or more silently discarded the reviewed vocabulary, which is why
    # "police refused to register my FIR after a theft at home" returned PFAS
    # and Jharkhand service cases while the short version of the same question
    # returned Lalita Kumari. The ranker sorts by rarity anyway; this only
    # decides which terms get probed.
    search_terms = list(dict.fromkeys([*focus_terms, *terms]))
    expanded = retrieval.glossary_terms(search_terms)
    stats["focus_terms"] = focus_terms
    if not search_terms and not expanded:
        return [], {}, stats
    # The loaded matrix has to belong to the artifact this call is configured
    # for. `_matrix is not None` alone was not enough: tests that build a small
    # corpus at a temporary path leave a matrix and a chunk index behind, and
    # the next call against the real artifact then scanned the temp vectors but
    # reported the temp titles -- a Telugu question "answered" from a four-case
    # fixture. A stale matrix is a reload, not a hit.
    try:
        stamp: tuple[str, float] | None = (str(db_path()), db_path().stat().st_mtime)
    except OSError:
        stamp = None
    if _matrix is None or _loaded_stamp != stamp:
        if _LOAD_STATE == "loading":
            # The startup warmer owns the load; abstain rather than start a
            # second one or queue behind it. Measured: loading cost 3.6-8.6 s,
            # and the curated store can answer this turn in ~40 ms.
            stats["loading"] = True
            return [], {}, stats
        with _connect() as connection:
            if not _load(connection):
                stats["loading"] = _LOAD_STATE == "loading"
                return [], {}, stats

    ranked: list[list[str]] = []
    try:
        with _connect() as connection:
            # A misspelt word the index has never seen retrieves nothing; the
            # longest prefix it does know retrieves the word it probably meant.
            resolved = retrieval.resolve_prefixes(
                search_terms,
                lambda term: _document_frequency(connection, term),
                lambda prefix: _prefix_exists(connection, prefix),
            )
            if resolved != search_terms:
                stats["corrected_terms"] = resolved
                search_terms = resolved
                expanded = retrieval.glossary_terms(search_terms)
            ordered = _ordered_terms(connection, search_terms)
            stats["terms"] = ordered
            lexical_ids, tried = _lexical(connection, ordered, expanded, jurisdiction, tiers, limit)
            stats["expressions_tried"] = tried
            stats["bm25_passages"] = len(lexical_ids)
            if lexical_ids:
                ranked.append(lexical_ids)
    except sqlite3.Error:
        return [], {}, stats

    want_dense = use_dense if use_dense is not None else True
    if want_dense:
        try:
            from . import dense as dense_layer

            # `ready()`, not `available()`: this must not load the encoder inside
            # a turn. Lexical answers this one, vectors answer the next.
            if dense_layer.available() and dense_layer.ready():
                # Embed the focused query, not the caller's sentence. A small
                # multilingual encoder confronted with "I want the most
                # relevant cases that cover..." encodes the framing; given
                # "cognizable refusal 154 fir" it encodes the question. Same
                # model, same cost, and the fused ranking then has a semantic
                # arm that is actually about the topic.
                vector = dense_layer.embed([retrieval.focus(query, focus_terms) or query])[0]
                if len(vector) == _matrix.shape[1]:
                    hits = _dense_scores(vector, _mask(jurisdiction, tiers), max(limit * 8, 40))
                    floor = [chunk_id for chunk_id, score in hits if score >= MIN_DENSE]
                    stats["dense"] = bool(floor)
                    stats["dense_passages"] = len(floor)
                    stats["min_dense"] = MIN_DENSE
                    if floor:
                        ranked.append(floor)
        except Exception as exc:  # encoder missing or failing: lexical still answers
            stats["dense_error"] = type(exc).__name__

    if not ranked:
        return [], {}, stats

    # Chunk-level fusion, then collapse to judgments. Fusing passages rather
    # than judgments means the ranking reflects the best supporting passage,
    # not an average across a whole decision.
    scores = retrieval.rrf(ranked)
    stats["candidates"] = len(scores)
    per_case: dict[str, int] = {}
    best: dict[str, tuple[float, str]] = {}
    # How many passages of a judgment matched, not just how well its best one
    # did. A decision whose whole reasoning is about the question matches in
    # several passages; one that mentions the same words in passing matches in
    # one. Among near-identical BM25 scores that is the difference between the
    # leading authority and a judgment that merely cites it, and it is free:
    # every passage is already scored before this loop.
    depth: dict[str, int] = {}
    for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        slot = _CHUNK_INDEX.get(chunk_id)
        if slot is None:
            continue
        case_id = _case_ids[int(_case_of[slot])]
        depth[case_id] = depth.get(case_id, 0) + 1
        if per_case.get(case_id, 0) >= MAX_PASSAGES_PER_CASE:
            continue
        per_case[case_id] = per_case.get(case_id, 0) + 1
        if case_id not in best:
            best[case_id] = (score, chunk_id)

    deepest = max(depth.values()) if depth else 1
    ordered_cases = sorted(
        best.items(),
        key=lambda item: item[1][0] * (
            1.0 + DEPTH_BOOST * math.log1p(depth.get(item[0], 1)) / math.log1p(deepest)
        ),
        reverse=True,
    )
    wanted = ordered_cases[: max(limit * 3, 12)]
    detail = _details(
        [(case_id, chunk_id) for case_id, (_, chunk_id) in wanted],
        search_terms,
        focus_terms,
    )
    ordered_ids = [case_id for case_id, _ in wanted if case_id in detail]
    # One judgment, one card: the same decision can be stored twice under two
    # ids when the source published two URLs for it.
    unique = retrieval.dedupe_cases([detail[case_id] for case_id in ordered_ids])
    ordered_ids = [case["id"] for case in unique]
    stats["judgments"] = len(ordered_ids)
    return ordered_ids, detail, stats


def _match_window(text: str, terms: Sequence[str], size: int) -> str:
    """Start the excerpt at the sentence that matched, not at the chunk start.

    A passage was selected by scoring its whole text, but only the first 900
    characters were shown and re-scored. When the query terms sat later in the
    passage, the caller saw a preamble that mentioned none of them and the
    relevance gate saw the same, so a judgment that did answer the question was
    scored as if it did not. The window is anchored on the first word-start
    match and includes a little context before it, then snaps forward to a
    sentence boundary: an excerpt that opens "rs on 12 November, 2013 Supreme
    Court of India" reads as a corrupted document, not a holding.
    """
    if not terms:
        return text[:size]
    lowered = text.lower()
    first = -1
    for term in terms:
        found = re.search(rf"(?<![a-z0-9]){re.escape(term)}", lowered)
        if found and (first < 0 or found.start() < first):
            first = found.start()
    if first < 0:
        return text[:size]
    start = max(0, first - 220)
    if start > 0:
        boundary = max(
            lowered.rfind(". ", 0, first),
            lowered.rfind("? ", 0, first),
            lowered.rfind("\u0964 ", 0, first),
        )
        if boundary >= start:
            start = boundary + 2
    return text[start : start + size]


def _pick_passage(
    rows: list[Any],
    matched_id: str,
    terms: Sequence[str],
    focus: Sequence[str] = (),
) -> str:
    """The passage of a judgment that actually addresses the question.

    A judgment is retrieved because *some* passage matched, and that passage is
    not necessarily the useful one: a decision can match on "refusal" in a
    procedural paragraph while the holding about refusing to register an FIR
    sits four pages away. Scoring every passage of the chosen decision against
    the same query terms costs one indexed lookup and shows the caller the part
    that answers them.

    Reviewed issue vocabulary counts double. A caller's framing appears all over
    a judgment in unrelated sentences ("the refusal of the accused to give a
    specimen"), while "cognizable" and "154" mark the paragraph that is actually
    deciding the question. And a passage too short to hold reasoning is not the
    holding, even when it repeats every word -- a cause-title page does exactly
    that.
    """
    if (not terms and not focus) or len(rows) < 2:
        return matched_id
    best_id, best_score = matched_id, -1
    for row in rows:
        text = str(row["text"] or "").lower()
        if not text:
            continue
        score = sum(1 for term in terms if term in text)
        score += 2 * sum(1 for term in focus if term in text)
        if len(text) < 400:
            score -= 1
        if score > best_score:
            best_id, best_score = str(row["chunk_id"]), score
    return best_id


def _details(
    pairs: list[tuple[str, str]],
    terms: Sequence[str] = (),
    focus: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    """Load the passage that best addresses the query, plus its sibling if short."""
    if not pairs:
        return {}
    by_case: dict[str, str] = {}
    for case_id, chunk_id in pairs:
        by_case.setdefault(case_id, chunk_id)
    out: dict[str, dict[str, Any]] = {}
    try:
        with _connect() as connection:
            for case_id, chunk_id in by_case.items():
                if terms or focus:
                    siblings = connection.execute(
                        "SELECT chunk_id, text FROM chunks WHERE case_id = ? ORDER BY ordinal",
                        (case_id,),
                    ).fetchall()
                    chunk_id = _pick_passage(siblings, chunk_id, terms, focus)
                row = connection.execute(
                    "SELECT c.chunk_id, c.case_id, c.ordinal, c.court_level, c.language, c.court, "
                    "c.jurisdiction, c.decision_date, c.text, k.title, k.citation, k.source_url, "
                    "k.dataset, k.text_chars "
                    "FROM chunks c JOIN cases k ON k.case_id = c.case_id WHERE c.chunk_id = ?",
                    (chunk_id,),
                ).fetchone()
                if row is None:
                    continue
                text = " ".join(str(row["text"] or "").split())
                # A passage cut short at a section boundary carries the start of
                # an argument and not its conclusion; one neighbour fixes that
                # without pulling in the whole judgment.
                if len(text) < 1200:
                    neighbour = connection.execute(
                        "SELECT text FROM chunks WHERE case_id = ? AND ordinal = ? LIMIT 1",
                        (case_id, (row["ordinal"] or 0) + 1),
                    ).fetchone()
                    if neighbour and neighbour["text"]:
                        text += "\n" + " ".join(str(neighbour["text"]).split())
                excerpt = _match_window(text, terms, MAX_EXCERPT)
                level = str(row["court_level"] or "unknown")
                out[case_id] = {
                    "id": case_id,
                    "title": str(row["title"] or "")[:500],
                    "court": str(row["court"] or "")[:160],
                    "court_level": level,
                    "authority_weight": TIER_WEIGHT.get(level, TIER_WEIGHT["unknown"]),
                    "citation": str(row["citation"] or "")[:240],
                    "decision_date": str(row["decision_date"] or "")[:40],
                    "dataset": str(row["dataset"] or "")[:40],
                    "language": str(row["language"] or "en")[:30],
                    "jurisdiction": str(row["jurisdiction"] or "IN")[:40],
                    "source_url": str(row["source_url"] or "")[:1000],
                    "excerpt": excerpt,
                    "passage_id": str(row["chunk_id"]),
                    "judgment_chars": int(row["text_chars"] or 0),
                    "source_type": "judgment_dataset_record",
                    "treatment_status": "not_checked",
                }
    except sqlite3.Error:
        return out
    return out
