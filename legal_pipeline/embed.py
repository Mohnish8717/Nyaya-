"""Int8-quantised embeddings: chunks -> out/vectors (sharded .npy + manifest).

Same quantisation scheme the Nyaya app's dense index uses (unit vectors scaled
to int8 in [-127, 127]), so a shard can be loaded anywhere without conversion.

Scale discipline, because this is the stage that costs GPU/CPU hours:
- the chunk table is ATTACHed, not fetched: a LEFT JOIN against the manifest
  picks up exactly the un-embedded rows, so memory stays flat at batch size
  even with tens of millions of chunks;
- a shard is written only when it is full, so an interrupted run loses at most
  one buffer and never leaves a half-written .npy that the manifest points at;
- a re-run verifies every shard referenced by the manifest still exists and,
  if one is gone, drops those rows from the map so they are recomputed rather
  than silently never being embedded.

    python pipeline.py embed                 # resume
    python pipeline.py embed --limit 5000    # bounded trial run
"""
from __future__ import annotations

import sqlite3
import sys

import numpy as np

import config


def _quantise(vectors: np.ndarray) -> np.ndarray:
    """Unit-vector fp32 -> int8 in [-127, 127]; 4x smaller, same ranking."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    return np.clip(np.round(unit * 127), -127, 127).astype(np.int8)


def _open_manifest() -> sqlite3.Connection:
    config.VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.VECTORS_MANIFEST, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS vector_map("
        "chunk_id TEXT PRIMARY KEY, shard INTEGER NOT NULL, row INTEGER NOT NULL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_map_shard ON vector_map(shard)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS shards("
        "shard INTEGER PRIMARY KEY, dim INTEGER NOT NULL, count INTEGER NOT NULL)"
    )
    return conn


def _repair_missing_shards(manifest: sqlite3.Connection) -> list[int]:
    """Forget vectors whose .npy disappeared externally; they get recomputed."""
    dropped: list[int] = []
    for shard, in manifest.execute("SELECT DISTINCT shard FROM vector_map").fetchall():
        if not (config.VECTORS_DIR / f"shard_{int(shard):06d}.npy").exists():
            manifest.execute("DELETE FROM vector_map WHERE shard=?", (shard,))
            manifest.execute("DELETE FROM shards WHERE shard=?", (shard,))
            dropped.append(int(shard))
    if dropped:
        manifest.commit()
    return dropped


def build(limit: int | None = None, model_name: str | None = None, embedder=None) -> dict:
    """Embed pending chunks into shards.

    `embedder` exists so the stage can be exercised without downloading a
    model: any object with `.embed(list[str]) -> iterable of vectors` works.
    Production always leaves it None and loads config.DENSE_MODEL.
    """
    if not config.CHUNKS_DB.exists():
        sys.exit("chunks.sqlite3 missing — run `python pipeline.py chunk` first")

    model_name = model_name or config.DENSE_MODEL
    manifest = _open_manifest()
    repaired = _repair_missing_shards(manifest)
    manifest.execute("ATTACH DATABASE ? AS c", (str(config.CHUNKS_DB),))

    total_chunks = int(manifest.execute("SELECT COUNT(*) FROM c.chunks").fetchone()[0])
    pending = int(manifest.execute(
        "SELECT COUNT(*) FROM c.chunks ch "
        "LEFT JOIN vector_map m ON m.chunk_id = ch.chunk_id "
        "WHERE m.chunk_id IS NULL"
    ).fetchone()[0])
    stats = {
        "chunks_total": total_chunks, "pending": pending, "embedded": 0,
        "model": model_name, "repaired_shards": repaired,
    }
    if not pending:
        print(f"embeddings: all {total_chunks} chunks already embedded; nothing to do")
        manifest.close()
        return stats

    # A previous model produces incompatible vectors: mixing dims in one index
    # is nonsense, so a model change means a rebuild, stated rather than silent.
    previous_model = manifest.execute("SELECT value FROM meta WHERE key='model'").fetchone()
    if previous_model and previous_model[0] != model_name:
        print(
            f"embeddings: vector map was built with {previous_model[0]}, now {model_name} — "
            "clearing it so the index is one model, not a blend. "
            "(Use the same DENSE_MODEL to resume incrementally.)",
            file=sys.stderr,
        )
        manifest.execute("DELETE FROM vector_map")
        manifest.execute("DELETE FROM shards")
        manifest.commit()
        for path in config.VECTORS_DIR.glob("shard_*.npy"):
            path.unlink()
        pending = total_chunks

    if embedder is None:
        from fastembed import TextEmbedding

        embedder = TextEmbedding(model_name=model_name)
    model = embedder
    next_shard = int(manifest.execute("SELECT COALESCE(MAX(shard), -1) + 1 FROM shards").fetchone()[0])
    buffer: list[tuple[str, np.ndarray]] = []
    dim = int(manifest.execute("SELECT COALESCE(MAX(dim), 0) FROM shards").fetchone()[0])
    written_shards: list[int] = []

    def flush() -> None:
        nonlocal next_shard, buffer
        if not buffer:
            return
        # Write to a temp name first: a torn .npy that the manifest already
        # points at is worse than no shard at all.
        shard_path = config.VECTORS_DIR / f"shard_{next_shard:06d}.npy"
        partial = shard_path.with_suffix(".npy.part")
        with partial.open("wb") as fh:
            # np.save(path) appends ".npy" to a name that lacks it; writing
            # through a handle keeps the temp name exact.
            np.save(fh, np.stack([vec for _, vec in buffer]).astype(np.int8))
        partial.replace(shard_path)
        manifest.executemany(
            "INSERT OR REPLACE INTO vector_map VALUES(?,?,?)",
            [(cid, next_shard, row) for row, (cid, _) in enumerate(buffer)],
        )
        manifest.execute(
            "INSERT OR REPLACE INTO shards(shard, dim, count) VALUES(?,?,?)",
            (next_shard, int(dim), len(buffer)),
        )
        manifest.commit()
        written_shards.append(next_shard)
        next_shard += 1
        buffer = []

    cursor = manifest.execute(
        "SELECT ch.chunk_id, ch.text FROM c.chunks ch "
        "LEFT JOIN vector_map m ON m.chunk_id = ch.chunk_id "
        "WHERE m.chunk_id IS NULL ORDER BY ch.chunk_id"
    )
    batch: list[tuple[str, str]] = []
    stop = False
    while not stop:
        batch = cursor.fetchmany(config.EMBED_BATCH)
        if not batch:
            break
        texts = [" ".join(text.split())[: config.EMBED_MAX_CHARS] for _, text in batch]
        vectors = np.asarray(list(model.embed(texts)), dtype="float32")
        dim = int(vectors.shape[1])
        quantised = _quantise(vectors)
        buffer.extend((cid, quantised[i]) for i, (cid, _) in enumerate(batch))
        stats["embedded"] += len(batch)
        if len(buffer) >= config.SHARD_MAX_VECTORS:
            flush()
        if limit is not None and stats["embedded"] >= limit:
            stop = True
    flush()

    embedded_total = int(manifest.execute("SELECT COUNT(*) FROM vector_map").fetchone()[0])
    manifest.execute("INSERT OR REPLACE INTO meta VALUES('model',?)", (model_name,))
    manifest.execute("INSERT OR REPLACE INTO meta VALUES('dim',?)", (str(dim or 0),))
    manifest.execute("INSERT OR REPLACE INTO meta VALUES('count',?)", (str(embedded_total),))
    manifest.execute("INSERT OR REPLACE INTO meta VALUES('chunks_total',?)", (str(total_chunks),))
    manifest.execute("INSERT OR REPLACE INTO meta VALUES('quantisation',?)", ("int8 /127",))
    manifest.commit()
    for row in manifest.execute("SELECT shard, count FROM shards ORDER BY shard"):
        stats.setdefault("shards", []).append({"shard": int(row[0]), "count": int(row[1])})
    manifest.close()
    stats["embedded_total"] = embedded_total
    print(f"embeddings: {stats['embedded']} chunks embedded this run "
          f"({embedded_total}/{total_chunks} total) with {model_name}, dim {dim}")
    return stats


if __name__ == "__main__":
    build()
