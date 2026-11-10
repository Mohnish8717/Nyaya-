"""Local dense retrieval for judgments.

Gnani's public API is speech only: docs.gnani.ai documents STT (Prisma), TTS
(Timbre) and voice cloning, and its own RAG examples use a third-party LLM.
There is no text-embeddings endpoint, so dense retrieval here cannot call
Gnani. It runs a multilingual sentence encoder locally on CPU through
onnxruntime — no GPU, no network at query time, and the model is loaded once
per process.

Why this exists: lexical BM25 cannot match "मेरे मालिक ने वेतन नहीं दिया" to a
judgment that says "wages were withheld by the employer". A paraphrase or a
different script has no shared token. The measured cross-lingual similarity
of this model on exactly that pair is 0.76 against 0.25 for an unrelated
sentence, which is enough to reorder candidates.

Storage is int8-quantised unit vectors in SQLite, which keeps ~6k chunks at
about 2.5 MB. Search is an exact dot product in numpy: at this corpus size
an approximate index would add failure modes and save nothing.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / ".runtime" / "dense_cases.sqlite3"
DEFAULT_MODEL = os.getenv("DENSE_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
ENABLED = os.getenv("DENSE_ENABLED", "auto").lower() not in {"0", "false", "off", "no"}
MAX_CHUNKS_PER_CASE = 6
# Similarity floor for this index, symmetrical with the passage corpus's
# `MIN_DENSE`. Without it the arm returned the nearest 20 judgments for any
# input at all, which is how a Telugu wage question came back with a service
# matter: the encoder compresses everything into 0.25-0.7, so "nearest" is not
# "related". Measured on this index: genuine topical questions score 0.50-0.70,
# unrelated prose 0.2-0.36, and an unrelated *question* ("what is the weather")
# can reach 0.52 -- which is why the floor is a guard on this one arm and not
# the relevance decision; `rerank` still requires shared vocabulary whenever
# the corpus could hold the caller's words.
MIN_SCORE = float(os.getenv("DENSE_MIN_SCORE", "0.40"))

_lock = threading.Lock()
_model: Any = None
_model_name: str | None = None
_matrix: Any = None
_ids: list[str] | None = None
_loaded_at = 0.0


def _imports() -> tuple[Any, Any] | None:
    try:
        import numpy  # noqa: F401
        from fastembed import TextEmbedding  # noqa: F401
    except Exception:
        return None
    return __import__("numpy"), TextEmbedding


def backend_status() -> dict[str, Any]:
    """Report what is actually available, never what would be nice."""
    imports = _imports()
    exists = DB_PATH.exists()
    return {
        "enabled": ENABLED,
        "dependency_installed": imports is not None,
        "index_built": exists,
        "model": DEFAULT_MODEL,
        "runtime": "onnxruntime-cpu",
        "provider_note": "Gnani publishes no text-embeddings endpoint; this runs locally.",
    }


def available() -> bool:
    return ENABLED and DB_PATH.exists() and _imports() is not None


def ready() -> bool:
    """Whether the encoder is already loaded here — never triggers a load.

    Loading the ONNX encoder takes seconds. A turn that calls into it before the
    startup warm finishes waits for the whole thing, which measured 5.6 s on the
    first request after boot. Retrieval asks this first and answers lexically
    for that turn instead; the next turn has vectors.
    """
    return _model is not None


def _load_model() -> Any:
    global _model, _model_name
    with _lock:
        if _model is None:
            imports = _imports()
            if imports is None:
                raise RuntimeError("dense dependencies are not installed")
            _, TextEmbedding = imports
            _model = TextEmbedding(model_name=DEFAULT_MODEL)
            _model_name = DEFAULT_MODEL
        return _model


def embed(texts: Sequence[str]) -> Any:
    """Embed a batch; returns an (n, dim) float32 unit-normalised array."""
    numpy, _ = _imports()  # type: ignore[misc]
    model = _load_model()
    vectors = numpy.array(list(model.embed(list(texts))), dtype="float32")
    norms = numpy.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build_index(force: bool = False, batch_size: int = 64) -> dict[str, Any]:
    """Embed every chunk in the case index. Idempotent unless forced."""
    numpy, _ = _imports()  # type: ignore[misc]
    from . import case_corpus

    if not case_corpus.CASE_DB.exists():
        raise RuntimeError("case index is missing; run the judgment indexer first")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = f"{case_corpus.CASE_DB.stat().st_size}:{case_corpus.CASE_DB.stat().st_mtime_ns}"
    with sqlite3.connect(DB_PATH) as db:
        db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS vectors(source_id TEXT, chunk INTEGER, vec BLOB, PRIMARY KEY(source_id, chunk))")
        existing = dict(db.execute("SELECT key,value FROM meta").fetchall())
        if not force and existing.get("stamp") == stamp and existing.get("model") == DEFAULT_MODEL:
            count = int(db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])
            return {"built": False, "vectors": count, "model": DEFAULT_MODEL}

        if force:
            db.execute("DELETE FROM vectors")
        with sqlite3.connect(case_corpus.CASE_DB) as cases:
            cases.row_factory = sqlite3.Row
            chunks = cases.execute(
                "SELECT source_id, title, court, decision_date, body FROM case_fts ORDER BY source_id, rowid"
            ).fetchall()
        # Cap chunks per judgment so one very long decision cannot dominate.
        per_case: dict[str, int] = {}
        selected: list[tuple[str, int, str]] = []
        heads: dict[str, str] = {}
        for row in chunks:
            source_id = row["source_id"]
            body = " ".join(str(row["body"] or "").split())
            if source_id not in heads:
                # A judgment's first chunk is mostly cause-title and counsel
                # appearances. The headnote is where the legal subject is
                # stated, so the head vector carries title + headnote + court
                # and gives a topical query something to match.
                head = f"{row['title']} {row['court']} {row['decision_date']}"
                marker = body.find("Headnotes")
                head += " " + (body[marker:marker + 1200] if marker >= 0 else body[:900])
                heads[source_id] = head
            used = per_case.get(source_id, 0)
            if used >= MAX_CHUNKS_PER_CASE:
                continue
            per_case[source_id] = used + 1
            selected.append((source_id, used, body))
        # chunk = -1 is the semantic head of the judgment.
        selected.extend((source_id, -1, head) for source_id, head in heads.items())
        started = time.perf_counter()
        written = 0
        for start in range(0, len(selected), batch_size):
            window = selected[start:start + batch_size]
            vectors = embed([item[2] for item in window])
            quantised = numpy.clip(numpy.round(vectors * 127), -127, 127).astype("int8")
            db.executemany(
                "INSERT OR REPLACE INTO vectors(source_id,chunk,vec) VALUES(?,?,?)",
                [(source_id, chunk, quantised[i].tobytes()) for i, (source_id, chunk, _) in enumerate(window)],
            )
            db.commit()
            written += len(window)
        db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('stamp',?)", (stamp,))
        db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('model',?)", (DEFAULT_MODEL,))
        db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('dim',?)", (str(vectors.shape[1]),))
        db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('count',?)", (str(written),))
        db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('built_at',?)", (str(int(time.time())),))
        global _matrix, _ids
        _matrix, _ids = None, None
        return {
            "built": True, "vectors": written, "model": DEFAULT_MODEL,
            "seconds": round(time.perf_counter() - started, 1),
            "judgments": len(per_case), "dim": int(vectors.shape[1]),
        }



def _load_matrix() -> tuple[Any, list[str]]:
    """Load the whole index once per process (a few MB at this size)."""
    global _matrix, _ids, _loaded_at
    numpy, _ = _imports()  # type: ignore[misc]
    with _lock:
        modified = DB_PATH.stat().st_mtime
        if _matrix is not None and _ids is not None and modified <= _loaded_at:
            return _matrix, _ids
        with sqlite3.connect(DB_PATH) as db:
            rows = db.execute("SELECT source_id, chunk, vec FROM vectors").fetchall()
            meta = dict(db.execute("SELECT key,value FROM meta").fetchall())
        if not rows:
            _matrix, _ids = numpy.zeros((0, 384), dtype="float32"), []
            _loaded_at = modified
            return _matrix, _ids
        dim = int(meta.get("dim", "384"))
        matrix = numpy.zeros((len(rows), dim), dtype="float32")
        ids: list[str] = []
        for i, (source_id, _chunk, blob) in enumerate(rows):
            matrix[i] = numpy.frombuffer(blob, dtype="int8").astype("float32") / 127.0
            ids.append(source_id)
        _matrix, _ids, _loaded_at = matrix, ids, modified
        return _matrix, _ids


def search(
    query: str,
    limit: int = 20,
    store: str = "cases",
    min_score: float | None = None,
) -> list[str]:
    """Return judgment IDs ordered by best-chunk cosine similarity."""
    import numpy

    if not available() or not query.strip():
        return []
    matrix, ids = _load_matrix()
    if matrix.size == 0:
        return []
    query_vector = embed([query])[0]
    scores = matrix @ query_vector
    best: dict[str, float] = {}
    for index, score in enumerate(scores.tolist()):
        source_id = ids[index]
        if score > best.get(source_id, -1.0):
            best[source_id] = score
    floor = MIN_SCORE if min_score is None else min_score
    ordered = [
        (source_id, score)
        for source_id, score in sorted(best.items(), key=lambda item: item[1], reverse=True)
        if score >= floor
    ][:limit]
    return [source_id for source_id, _ in ordered]


def search_with_scores(
    query: str,
    limit: int = 20,
    min_score: float | None = None,
) -> list[tuple[str, float]]:
    import numpy

    if not available() or not query.strip():
        return []
    matrix, ids = _load_matrix()
    if matrix.size == 0:
        return []
    scores = matrix @ embed([query])[0]
    best: dict[str, float] = {}
    for index, score in enumerate(scores.tolist()):
        source_id = ids[index]
        if score > best.get(source_id, -1.0):
            best[source_id] = score
    floor = MIN_SCORE if min_score is None else min_score
    return [
        (source_id, score)
        for source_id, score in sorted(best.items(), key=lambda item: item[1], reverse=True)
        if score >= floor
    ][:limit]


def index_stats() -> dict[str, Any]:
    if not DB_PATH.exists():
        return {**backend_status(), "vectors": 0}
    try:
        with sqlite3.connect(DB_PATH) as db:
            meta = dict(db.execute("SELECT key,value FROM meta").fetchall())
    except sqlite3.Error:
        return {**backend_status(), "vectors": 0}
    return {
        **backend_status(),
        "vectors": int(meta.get("count", 0)),
        "dim": int(meta.get("dim", 0) or 0),
        "built_at": int(meta.get("built_at", 0) or 0),
    }
