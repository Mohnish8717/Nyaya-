"""Convert the pipeline's own artifacts into the schemas the Nyaya app opens.

The pipeline stores vectors as sharded .npy files (good for a single writer, a
bad fit for the app's reader). The app wants:

  app/dense.py        -> vectors(source_id, chunk, vec INT8 BLOB) + meta
  app/case_corpus.py  -> case_fts(...) + index_state

This stage is a pure conversion: it reads what embed/fts already produced and
writes `out/app/dense_cases.sqlite3` and `out/app/cases.sqlite3`, which can be
copied into `.runtime/` unchanged. Nothing is re-embedded, so running it twice
costs only I/O.

    python pipeline.py export
"""
from __future__ import annotations

import sqlite3
import sys
import time

import numpy as np

import config


def _dense() -> dict:
    if not config.VECTORS_MANIFEST.exists() or not config.CHUNKS_DB.exists():
        sys.exit("export needs vectors/manifest.sqlite3 and chunks.sqlite3 — run embed first")
    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = sqlite3.connect(config.EXPORT_DENSE_DB, timeout=60)
    out.execute("PRAGMA journal_mode=WAL")
    out.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
    out.execute(
        "CREATE TABLE IF NOT EXISTS vectors("
        "source_id TEXT, chunk INTEGER, vec BLOB, PRIMARY KEY(source_id, chunk))"
    )
    out.execute("DELETE FROM vectors")
    out.execute("ATTACH DATABASE ? AS m", (str(config.VECTORS_MANIFEST),))
    out.execute("ATTACH DATABASE ? AS c", (str(config.CHUNKS_DB),))
    out.execute("CREATE TEMP TABLE IF NOT EXISTS ids(chunk_id TEXT PRIMARY KEY)")

    written = 0
    skipped = 0
    dim = 0
    shards = out.execute("SELECT shard FROM m.shards ORDER BY shard").fetchall()
    for (shard,) in shards:
        path = config.VECTORS_DIR / f"shard_{int(shard):06d}.npy"
        if not path.exists():
            print(f"  shard {shard} missing on disk; skipped", file=sys.stderr)
            continue
        matrix = np.load(path)
        if matrix.ndim != 2:
            print(f"  shard {shard} has unexpected shape {matrix.shape}; skipped", file=sys.stderr)
            continue
        dim = int(matrix.shape[1])
        ids = [row[0] for row in out.execute(
            "SELECT chunk_id FROM m.vector_map WHERE shard=? ORDER BY row", (int(shard),)
        )]
        if len(ids) != matrix.shape[0]:
            print(f"  shard {shard}: manifest says {len(ids)} rows, file has "
                  f"{matrix.shape[0]}; using the shorter one", file=sys.stderr)
        out.execute("DELETE FROM ids")
        out.executemany("INSERT OR IGNORE INTO ids VALUES(?)", [(cid,) for cid in ids])
        located = {
            row[0]: (row[1], row[2], row[3])
            for row in out.execute(
                "SELECT i.chunk_id, ch.case_id, ch.kind, ch.ordinal "
                "FROM ids i JOIN c.chunks ch ON ch.chunk_id = i.chunk_id"
            )
        }
        rows = []
        for index, cid in enumerate(ids):
            if index >= matrix.shape[0]:
                break
            info = located.get(cid)
            if info is None:
                skipped += 1
                continue
            case_id, kind, ordinal = info
            # The app's convention: chunk -1 is the semantic head of a case.
            chunk_no = -1 if kind == "head" else int(ordinal) - 1
            rows.append((case_id, chunk_no, matrix[index].tobytes()))
        out.executemany("INSERT OR REPLACE INTO vectors(source_id,chunk,vec) VALUES(?,?,?)", rows)
        out.commit()
        written += len(rows)

    meta = dict(out.execute("SELECT key,value FROM m.meta").fetchall())
    stamp = f"{config.CHUNKS_DB.stat().st_size}:{config.CHUNKS_DB.stat().st_mtime_ns}"
    for key, value in (
        ("stamp", stamp),
        ("model", meta.get("model", config.DENSE_MODEL)),
        ("dim", str(dim or meta.get("dim", ""))),
        ("count", str(written)),
        ("built_at", str(int(time.time()))),
        ("source", "legal_pipeline export"),
    ):
        out.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, value))
    out.commit()
    out.close()
    print(f"export: {written} vectors -> {config.EXPORT_DENSE_DB}"
          + (f" ({skipped} orphan chunks skipped)" if skipped else ""))
    return {"vectors": written, "dim": dim, "skipped": skipped, "path": str(config.EXPORT_DENSE_DB)}


def _cases() -> dict:
    if not config.CASES_INDEX_DB.exists():
        sys.exit("export needs cases_index.sqlite3 — run fts first")
    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    target = config.EXPORT_CASES_DB
    if target.exists():
        target.unlink()
    # The backup API copies a consistent snapshot even with WAL pages pending,
    # which a plain file copy would not.
    with sqlite3.connect(config.CASES_INDEX_DB, timeout=60) as src, \
            sqlite3.connect(target, timeout=60) as dst:
        src.backup(dst)
    with sqlite3.connect(target, timeout=30) as db:
        count = int(db.execute("SELECT COUNT(*) FROM case_fts").fetchone()[0])
    print(f"export: {count} cases -> {target}")
    return {"cases": count, "path": str(target)}


def build() -> dict:
    result = {"dense": _dense(), "cases": _cases()}
    print(
        "To use these in the app: copy them into .runtime/ as dense_cases.sqlite3 "
        "and cases.sqlite3 (see legal_pipeline/README.md)."
    )
    return result


if __name__ == "__main__":
    build()
