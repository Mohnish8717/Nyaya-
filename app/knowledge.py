"""Version-aware legal source store and SQLite FTS5 retrieval.

Source records are curated inputs. Retrieved text is evidence, never executable
instructions. The store keeps exact source text and stable IDs for citations.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from . import retrieval

ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = ROOT / "data" / "legal_sources.json"
RUNTIME_DIR = ROOT / ".runtime"
DB_PATH = RUNTIME_DIR / "legal-index.sqlite3"
QA_PATH = ROOT / "data" / "indian-law-dataset.json"
QA_DB_PATH = RUNTIME_DIR / "secondary-qa.sqlite3"
REQUIRED = {"id", "title", "authority", "jurisdiction", "kind", "url", "text", "tags", "review_status", "effective_from", "effective_to", "checked_at"}


def read_pack(path: Path = PACK_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        pack = json.load(f)
    validate_pack(pack)
    return pack


def validate_pack(pack: Any) -> None:
    if not isinstance(pack, list):
        raise ValueError("Source pack must be a JSON array.")
    seen: set[str] = set()
    for i, source in enumerate(pack):
        if not isinstance(source, dict):
            raise ValueError(f"Source {i} must be an object.")
        missing = REQUIRED - source.keys()
        if missing:
            raise ValueError(f"Source {i} is missing: {', '.join(sorted(missing))}")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:-]{2,99}", str(source["id"])):
            raise ValueError(f"Source {i} has an invalid stable ID.")
        if source["id"] in seen:
            raise ValueError(f"Duplicate source ID: {source['id']}")
        seen.add(source["id"])
        if not str(source["url"]).startswith("https://"):
            raise ValueError(f"Source {source['id']} must have an HTTPS source URL.")
        for field in ("title", "authority", "jurisdiction", "kind"):
            if not isinstance(source[field], str) or not source[field].strip():
                raise ValueError(f"Source {source['id']} needs a non-empty {field}.")
        if not isinstance(source["text"], str) or len(source["text"].strip()) < 20:
            raise ValueError(f"Source {source['id']} needs reviewed source text.")
        if not isinstance(source["tags"], list) or any(not isinstance(tag, str) for tag in source["tags"]):
            raise ValueError(f"Source {source['id']} tags must be an array.")
        if source["review_status"] not in {"approved", "draft", "expired", "rejected"}:
            raise ValueError(f"Source {source['id']} has an invalid review_status.")
        if not source.get("checked_at"):
            raise ValueError(f"Source {source['id']} must include the date it was checked.")
        for field in ("effective_from", "effective_to", "checked_at"):
            value = source.get(field)
            if value:
                try:
                    date.fromisoformat(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Source {source['id']} has an invalid {field}; use YYYY-MM-DD.") from exc
        if source.get("effective_from") and source.get("effective_to") and source["effective_from"] > source["effective_to"]:
            raise ValueError(f"Source {source['id']} has an effective date range that ends before it starts.")
        for rel in source.get("relations", []):
            if not isinstance(rel, dict) or "target" not in rel or "type" not in rel:
                raise ValueError(f"Source {source['id']} contains an invalid relation.")


SOURCES = read_pack()


def reload_source_pack() -> int:
    """Reload the reviewed pack atomically in process and rebuild its index."""
    global SOURCES
    previous = SOURCES
    fresh = read_pack()
    try:
        SOURCES = fresh
        return rebuild_index()
    except Exception:
        SOURCES = previous
        raise


def approved_sources() -> list[dict[str, Any]]:
    today = date.today().isoformat()
    out = []
    for source in SOURCES:
        if source.get("review_status") != "approved":
            continue
        if source.get("effective_from") and source["effective_from"] > today:
            continue
        if source.get("effective_to") and source["effective_to"] < today:
            continue
        out.append(source)
    return out


def rebuild_index() -> int:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    sources = approved_sources()
    with sqlite3.connect(DB_PATH) as db:
        db.execute("DROP TABLE IF EXISTS source_fts")
        db.execute("CREATE VIRTUAL TABLE source_fts USING fts5(source_id UNINDEXED, title, authority, tags, body, tokenize='unicode61 remove_diacritics 2')")
        db.executemany(
            "INSERT INTO source_fts(source_id,title,authority,tags,body) VALUES(?,?,?,?,?)",
            [(s["id"], s["title"], s["authority"], " ".join(s.get("tags", [])), s["text"]) for s in sources],
        )
        db.execute("CREATE TABLE IF NOT EXISTS source_edges(source_id TEXT, relation TEXT, target_id TEXT, PRIMARY KEY(source_id,relation,target_id))")
        db.execute("DELETE FROM source_edges")
        db.executemany(
            "INSERT OR IGNORE INTO source_edges(source_id,relation,target_id) VALUES(?,?,?)",
            [(s["id"], rel["type"], rel["target"]) for s in sources for rel in s.get("relations", []) if rel["target"] in {x["id"] for x in sources}],
        )
    _ensure_secondary_qa_index()
    return len(sources)


def _ensure_secondary_qa_index() -> int:
    """Index the supplied Q&A corpus as a discovery aid, never legal authority."""
    if not QA_PATH.exists():
        return 0
    stat = QA_PATH.stat()
    stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
    with sqlite3.connect(QA_DB_PATH) as db:
        db.execute("CREATE TABLE IF NOT EXISTS index_state(key TEXT PRIMARY KEY, value TEXT)")
        existing = db.execute("SELECT value FROM index_state WHERE key='source_stamp'").fetchone()
        if existing and existing[0] == stamp:
            return int(db.execute("SELECT COUNT(*) FROM qa_fts").fetchone()[0])
        db.execute("DROP TABLE IF EXISTS qa_fts")
        db.execute("CREATE VIRTUAL TABLE qa_fts USING fts5(question, answer, tokenize='unicode61 remove_diacritics 2')")
        with QA_PATH.open(encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError("The Indian law Q&A dataset must be a JSON list.")
        db.executemany(
            "INSERT INTO qa_fts(question,answer) VALUES(?,?)",
            [(str(row.get("Instruction", ""))[:800], str(row.get("Response", ""))[:16000])
             for row in rows if isinstance(row, dict) and row.get("Instruction") and row.get("Response")],
        )
        db.execute("INSERT OR REPLACE INTO index_state(key,value) VALUES('source_stamp',?)", (stamp,))
        return len(rows)


def retrieve_secondary_qa(query: str, limit: int = 3) -> list[dict[str, str]]:
    """Return non-authoritative question/answer candidates for terminology only."""
    terms = retrieval.content_terms(query, limit=10)
    expressions = retrieval.match_expressions(terms, retrieval.glossary_terms(terms))
    if not expressions or not QA_DB_PATH.exists():
        return []
    rows = []
    for expression in expressions:
        try:
            with sqlite3.connect(QA_DB_PATH, timeout=5.0) as db:
                db.row_factory = sqlite3.Row
                rows = db.execute(
                    "SELECT question,answer,bm25(qa_fts,2.0,1.0) AS score FROM qa_fts "
                    "WHERE qa_fts MATCH ? ORDER BY score LIMIT ?", (expression, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        if len(rows) >= limit:
            break
    return [{"question": row["question"], "answer": row["answer"], "authority": "secondary_unverified_dataset"} for row in rows]


def _jurisdiction_matches(source_jurisdiction: str, requested: str) -> bool:
    if source_jurisdiction in {"*", requested}:
        return True
    # National authority applies as a retrieval candidate for a subnational query.
    return source_jurisdiction == "IN" and requested.startswith("IN-")


def _fts_hits(expression: str, limit: int) -> list[str]:
    try:
        with sqlite3.connect(DB_PATH, timeout=5.0) as db:
            db.execute("PRAGMA busy_timeout=5000")
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT source_id, bm25(source_fts,0,4.0,2.0,2.0,1.0) AS score FROM source_fts "
                "WHERE source_fts MATCH ? ORDER BY score LIMIT ?",
                (expression, limit),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [row["source_id"] for row in rows]


def retrieve(query: str, jurisdiction: str = "IN", limit: int = 8, as_of: str | None = None) -> list[dict[str, Any]]:
    """Cascading lexical retrieval with fusion, jurisdiction/date filtering,
    and one-hop relation expansion.

    The cascade matters: a strict AND query runs first and, when it already
    returns enough candidates, the broader (noisier) OR queries are skipped
    entirely. Dense retrieval can be registered through `retrieval`, but this
    function never claims it when no backend is attached.
    """
    terms = retrieval.content_terms(query)
    expanded = retrieval.glossary_terms(terms)
    expressions = retrieval.match_expressions(terms, expanded)
    if not expressions:
        return []
    now = as_of or date.today().isoformat()
    by_id = {source["id"]: source for source in approved_sources()}

    def eligible(source_id: str) -> bool:
        source = by_id.get(source_id)
        if not source or not _jurisdiction_matches(source["jurisdiction"], jurisdiction):
            return False
        if source.get("effective_from") and source["effective_from"] > now:
            return False
        if source.get("effective_to") and source["effective_to"] < now:
            return False
        return True

    ranked_lists: list[list[str]] = []
    and_matched: set[str] = set()
    for expression in expressions:
        ids = [source_id for source_id in _fts_hits(expression, max(limit * 6, 24)) if eligible(source_id)]
        if ids:
            ranked_lists.append(ids)
            if " AND " in expression:
                and_matched.update(ids)
        if len(ids) >= limit * 2:
            break
    scores = retrieval.rrf(ranked_lists)
    ordered = sorted(scores, key=lambda source_id: scores[source_id], reverse=True)[:limit]

    # Guard against accidental OR matches: a source that only matched a broad
    # OR query must share at least two tags or title terms with the caller's
    # terms. One was too weak: an FIR-registration question shares the word
    # "police" with the constitutional article on arrest, so Article 22 was
    # returned as "the closest approved passage" for a question it does not
    # answer at all. Two shared subject words is the smallest overlap that
    # distinguishes "this passage is about the caller's issue" from "this
    # passage is about the same institution".
    query_terms = set(terms) | set(expanded)
    def _topical(source_id: str) -> bool:
        if source_id in and_matched:
            return True
        source = by_id.get(source_id)
        if not source:
            return False
        tags = {t.lower() for t in source.get("tags", [])}
        title_terms = set(retrieval.content_terms(source.get("title", "")))
        return len(query_terms & (tags | title_terms)) >= 2

    direct = [by_id[source_id] for source_id in ordered if _topical(source_id)]
    seen = {source["id"] for source in direct}
    if direct:
        try:
            with sqlite3.connect(DB_PATH, timeout=5.0) as db:
                for source in direct:
                    rows = db.execute("SELECT target_id FROM source_edges WHERE source_id=?", (source["id"],)).fetchall()
                    for (target_id,) in rows:
                        target = by_id.get(target_id)
                        if target and target_id not in seen and _jurisdiction_matches(target["jurisdiction"], jurisdiction):
                            direct.append(target)
                            seen.add(target_id)
        except sqlite3.Error:
            pass
    return direct[:limit]


def source_by_id(source_id: str) -> dict[str, Any] | None:
    return next((s for s in approved_sources() if s["id"] == source_id), None)


def validate_evidence_claims(claims: list[dict[str, Any]], retrieved: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    by_id = {s["id"]: s for s in retrieved}
    issues: list[str] = []
    for i, claim in enumerate(claims):
        source_id = claim.get("source_id")
        source = by_id.get(source_id)
        quote = " ".join(str(claim.get("quote", "")).split())
        source_text = " ".join(str(source.get("text", "")).split()) if source else ""
        if not source or not quote:
            issues.append(f"claim_{i}_missing_or_unretrieved_evidence")
        elif quote not in source_text:
            issues.append(f"claim_{i}_quote_not_found_verbatim")
    return not issues, issues


def source_evidence(source: dict[str, Any]) -> dict[str, Any]:
    """Public evidence descriptor; caller text and internal review notes are excluded."""
    return {
        "id": source["id"],
        "title": source["title"],
        "authority": source["authority"],
        "url": source["url"],
        "jurisdiction": source["jurisdiction"],
        "kind": source["kind"],
        "effective_from": source.get("effective_from"),
        "effective_to": source.get("effective_to"),
        "checked_at": source.get("checked_at"),
    }

