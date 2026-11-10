"""Case-level BM25 index (SQLite FTS5).

Written to be instantly queryable AND schematically identical to the index the
Nyaya app's case store uses (`app/case_corpus.py`), so out/cases_index.sqlite3
can be handed to the app with no migration: same table name, same columns,
same tokenizer, same `index_state` keys.

Rebuild policy: the table is keyed to a stamp of judgments.jsonl. If the file
changed (new rows appended by extract), the table is rebuilt rather than left
silently stale — a partial index would answer some queries and quietly miss
the newest judgments.
"""
from __future__ import annotations

import json
import sqlite3
import sys

import config
import util

CREATE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS case_fts USING fts5(
    source_id UNINDEXED, title, court, citation, decision_date UNINDEXED,
    dataset UNINDEXED, language UNINDEXED, jurisdiction UNINDEXED,
    source_url UNINDEXED, body,
    tokenize='unicode61 remove_diacritics 2')
"""

INSERT_SQL = (
    "INSERT INTO case_fts(source_id,title,court,citation,decision_date,"
    "dataset,language,jurisdiction,source_url,body) VALUES(?,?,?,?,?,?,?,?,?,?)"
)


def build(limit: int | None = None) -> dict:
    if not config.JUDGMENTS_JSONL.exists():
        sys.exit("judgments.jsonl missing — run `python pipeline.py extract` first")
    stamp = util.stamp_for(config.JUDGMENTS_JSONL, "schema-v3")
    db = config.CASES_INDEX_DB
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("CREATE TABLE IF NOT EXISTS index_state(key TEXT PRIMARY KEY, value TEXT)")
    existing = conn.execute("SELECT value FROM index_state WHERE key='source_stamp'").fetchone()

    current = 0
    if existing:
        try:
            current = int(conn.execute("SELECT COUNT(*) FROM case_fts").fetchone()[0])
        except sqlite3.OperationalError:
            # state row without the table (interrupted first build): treat as
            # absent instead of crashing on a missing table.
            current = 0
            existing = None
    if existing and existing[0] == stamp and current:
        print(f"fts index: already current ({current} cases); nothing to do")
        conn.close()
        return {"cases": current, "rebuilt": False}

    if existing:
        conn.execute("DROP TABLE IF EXISTS case_fts")
    conn.execute(CREATE_SQL)

    rows: list[tuple] = []
    inserted = 0
    seen_ids: set[str] = set()
    with config.JUDGMENTS_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if limit is not None and inserted + len(rows) >= limit:
                break
            source_id = str(rec.get("id") or "")
            body = str(rec.get("text") or "")
            if not source_id or not body.strip() or source_id in seen_ids:
                continue
            seen_ids.add(source_id)
            rows.append((
                source_id, str(rec.get("title") or "")[:500],
                str(rec.get("court") or "")[:160], str(rec.get("citation") or "")[:240],
                str(rec.get("decision_date") or "unknown")[:40], str(rec.get("dataset") or "")[:40],
                str(rec.get("language") or "unknown")[:30], str(rec.get("jurisdiction") or "IN")[:40],
                str(rec.get("source_url") or "")[:1000], body,
            ))
            if len(rows) >= 500:
                conn.executemany(INSERT_SQL, rows)
                inserted += len(rows)
                rows.clear()
    if rows:
        conn.executemany(INSERT_SQL, rows)
        inserted += len(rows)

    conn.execute("INSERT OR REPLACE INTO index_state(key,value) VALUES('source_stamp',?)", (stamp,))
    conn.execute("INSERT OR REPLACE INTO index_state(key,value) VALUES('case_count',?)", (str(inserted),))
    # The app reads `source_count`; keep both keys so either consumer works.
    conn.execute("INSERT OR REPLACE INTO index_state(key,value) VALUES('source_count',?)", (str(inserted),))
    conn.commit()
    conn.close()
    print(f"fts index: {inserted} cases written to {db}")
    return {"cases": inserted, "rebuilt": True}


def stats() -> dict:
    if not config.CASES_INDEX_DB.exists():
        return {"cases": 0}
    try:
        with sqlite3.connect(config.CASES_INDEX_DB, timeout=10) as db:
            return {"cases": int(db.execute("SELECT COUNT(*) FROM case_fts").fetchone()[0])}
    except sqlite3.Error:
        return {"cases": 0}


if __name__ == "__main__":
    build()
