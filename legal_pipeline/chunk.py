"""Chunk judgments into embedding-ready text units.

Per judgment: one semantic `head` row (title + court + date + headnote slice,
exactly the trick the app's dense index uses so topical queries land on the
holding) plus `body` rows of ~CHUNK_TARGET_CHARS with small overlap, capped at
CHUNKS_PER_CASE_CAP so one giant decision cannot dominate the vector index.

Output stays in SQLite pieces large enough to stream, small enough that the
embed stage can checkpoint per-batch onto the final vectors.

The embedding stage never sees a whole judgment: it sees `head` and `body`
rows, each of which fits the encoder's 512-token window.
"""
from __future__ import annotations

import json
import sqlite3
import sys

import config
import util

MIN_TEXT_CHARS = 200


def _chunk_one(text: str, head_text: str) -> list[tuple[str, str]]:
    """(kind, chunk) pairs; head first so ids sort naturally."""
    out = [("head", head_text)]
    cleaned = " ".join(text.split())
    start = 0
    n = len(cleaned)
    ordinal = 0
    while start < n and ordinal < config.CHUNKS_PER_CASE_CAP:
        end = min(start + config.CHUNK_TARGET_CHARS, n)
        if end < n:
            # cut at the last sentence boundary inside the window so chunks do
            # not split a holding in half; fall back to a hard cut when the
            # window holds no full stop (tables, numbered lists).
            cut = cleaned.rfind(".", start + 200, end)
            if cut > 0:
                end = cut + 1
        piece = cleaned[start:end].strip()
        if piece:
            out.append(("body", piece))
        ordinal += 1
        if end >= n:
            break
        start = max(end - config.CHUNK_OVERLAP_CHARS, start + 1)
    return out


def _head_slice(text: str) -> str:
    """The first ~1.2 K chars around the headnote, else the cause title area."""
    low = text.lower()
    marker = low.find("headnotes")
    if marker < 0:
        marker = low.find("judgment")
    if marker >= 0:
        return text[marker:marker + 1200]
    return text[:900]


def build(resume: bool = True, limit: int | None = None) -> dict:
    """Stream judgments.jsonl into out/chunks.sqlite3 (idempotent, resumable)."""
    if not config.JUDGMENTS_JSONL.exists():
        sys.exit("judgments.jsonl missing — run `python pipeline.py extract` first")
    config.CHUNKS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.CHUNKS_DB, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chunks(
               chunk_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, kind TEXT NOT NULL,
               ordinal INTEGER, text TEXT NOT NULL, court TEXT, decision_date TEXT,
               jurisdiction TEXT, language TEXT, dataset TEXT, source_url TEXT
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_case ON chunks(case_id)")
    stats = {"cases": 0, "chunks": 0, "skipped_existing": 0, "empty": 0, "chars": 0}

    read = 0
    with config.JUDGMENTS_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if limit is not None and read >= limit:
                break
            read += 1
            case_id = str(rec.get("id") or "")
            text = str(rec.get("text") or "")
            if not case_id or len(" ".join(text.split())) < MIN_TEXT_CHARS:
                stats["empty"] += 1
                continue
            if resume and conn.execute(
                "SELECT 1 FROM chunks WHERE case_id=? LIMIT 1", (case_id,)
            ).fetchone():
                stats["skipped_existing"] += 1
                continue
            meta = (
                str(rec.get("court") or "")[:250],
                str(rec.get("decision_date") or "unknown")[:40],
                str(rec.get("jurisdiction") or "IN")[:40],
                str(rec.get("language") or util.script_language(text))[:30],
                str(rec.get("dataset") or "")[:40],
                str(rec.get("source_url") or "")[:400],
            )
            head_text = " ".join([
                str(rec.get("title") or "")[:300],
                meta[0], meta[1], _head_slice(text),
            ]).strip()
            rows = [
                (f"{case_id}#{ordinal}", case_id, kind, ordinal, chunk, *meta)
                for ordinal, (kind, chunk) in enumerate(_chunk_one(text, head_text))
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO chunks(chunk_id, case_id, kind, ordinal, text, "
                "court, decision_date, jurisdiction, language, dataset, source_url) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)
            stats["cases"] += 1
            stats["chunks"] += len(rows)
            stats["chars"] += sum(len(row[4]) for row in rows)
            if stats["cases"] % 200 == 0:
                conn.commit()
    conn.commit()
    conn.close()
    print(f"chunks: {stats['cases']} new cases, {stats['chunks']} new chunk rows "
          f"({stats['chars'] / 1e6:.1f} M chars), {stats['skipped_existing']} already chunked, "
          f"{stats['empty']} too short")
    return stats


def stats() -> dict:
    """Row counts for `status` without scanning the corpus."""
    if not config.CHUNKS_DB.exists():
        return {"cases": 0, "chunks": 0}
    with sqlite3.connect(config.CHUNKS_DB, timeout=10) as db:
        cases = int(db.execute("SELECT COUNT(DISTINCT case_id) FROM chunks").fetchone()[0])
        chunks = int(db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    return {"cases": cases, "chunks": chunks}


if __name__ == "__main__":
    build()
