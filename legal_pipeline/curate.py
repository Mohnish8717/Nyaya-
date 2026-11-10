"""Curated slice of the judicial corpus -> exactly ONE embeddings file.

Full high-court corpus is ~1.25 TiB and ~17.8 M judgments: that is a cluster
job, not a laptop job. This module exists for the other need — a small, real,
multilingual slice that can be embedded on one machine today and still behaves
like the real thing:

  * every source is capped (default 150 MB of embedded text in total), split by
    court tier, so the slice is Supreme Court + High Court + district court,
    not just whichever archive sorted first;
  * selection is language-stratified: a tar in `regional/tamil/` gets a budget
    proportional to how many judgments it holds, so a slice cannot silently
    become English-only while regional judgments are sitting on disk;
  * every chunk carries its tier (`supreme` / `high` / `district`) in a column
    AND in a one-line header that is part of the embedded text, so vectors from
    three tiers are separable and the reader model is told which tier it is
    quoting;
  * the output is a single SQLite file — cases, passages, vectors and the BM25
    index together — so there is one artifact to copy, not a shard directory.
    Passages are stored once; the FTS5 table is an external-content index over
    them, which keeps a 150 MB deliverable at 150 MB instead of ~250 MB.

Honest limits, stated rather than hidden: within an archive the slice is the
head of the member list (deterministic, not random), metadata comes from the
sibling metadata tars where present and falls back to the file name otherwise,
and a tier that has no data on disk simply reports a shortfall instead of being
back-filled from another tier.

    python pipeline.py curate --target-mb 150
    python pipeline.py curate --sources aws_sc aws_hc --target-mb 20   # trial
"""
from __future__ import annotations

import itertools
import json
import sqlite3
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Iterator

import chunk as chunk_mod
import config
import dl_kaggle
import dl_opennyai
import extract
import util

# Language folders as the Open Data archives name them (english + regional).
LANG_FOLDERS = {
    "english": "en", "hindi": "hi", "tamil": "ta", "telugu": "te", "kannada": "kn",
    "malayalam": "ml", "marathi": "mr", "bengali": "bn", "gujarati": "gu",
    "punjabi": "pa", "odia": "or", "oriya": "or", "urdu": "ur", "assamese": "as",
    "nepali": "ne", "sanskrit": "sa", "manipuri": "mni", "konkani": "kok",
    "kashmiri": "ks", "sindhi": "sd", "dogri": "doi", "bodo": "brx", "maithili": "mai",
}

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)""",
    """CREATE TABLE IF NOT EXISTS cases(
           case_id TEXT PRIMARY KEY, title TEXT, court TEXT, court_level TEXT,
           jurisdiction TEXT, language TEXT, decision_date TEXT, citation TEXT,
           dataset TEXT, source_url TEXT, text_chars INTEGER, ingested_at REAL)""",
    """CREATE TABLE IF NOT EXISTS chunks(
           chunk_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, kind TEXT, ordinal INTEGER,
           court_level TEXT, language TEXT, court TEXT, jurisdiction TEXT,
           decision_date TEXT, header TEXT, text TEXT)""",
    # Index only: FTS5 external content points at chunks.text, so the judgment
    # is not written twice. Query BM25 by joining chunk_fts.rowid to chunks.rowid.
    """CREATE TABLE IF NOT EXISTS vectors(
           chunk_id TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)""",
    """CREATE INDEX IF NOT EXISTS idx_chunks_case ON chunks(case_id)""",
    """CREATE INDEX IF NOT EXISTS idx_chunks_level ON chunks(court_level)""",
    """CREATE INDEX IF NOT EXISTS idx_cases_level ON cases(court_level)""",
    """CREATE TABLE IF NOT EXISTS done_cases(case_id TEXT PRIMARY KEY)""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
           text, content='chunks', content_rowid='rowid',
           tokenize='unicode61 remove_diacritics 2')""",
)


# ---------------------------------------------------------------------------
# Budgets and source discovery
# ---------------------------------------------------------------------------
def _lang_from_path(path: Path) -> str:
    for part in path.parts:
        code = LANG_FOLDERS.get(part.lower())
        if code:
            return code
    low = path.name.lower()
    for name, code in LANG_FOLDERS.items():
        if name in low:
            return code
    return "unknown"


def _aws_root(source: str) -> Path:
    return config.RAW / f"{source}_data"


def _aws_tars(source: str) -> tuple[list[Path], list[Path]]:
    root = _aws_root(source)
    if not root.is_dir():
        return [], []
    data: list[Path] = []
    meta: list[Path] = []
    for path in sorted(root.rglob("*.tar")):
        (meta if "metadata" in path.name.lower() else data).append(path)
    return data, meta


def available_sources() -> dict[str, dict[str, Any]]:
    """What is actually on disk, with a human-readable reason when it isn't."""
    out: dict[str, dict[str, Any]] = {}
    for key in ("aws_sc", "aws_hc"):
        data, _ = _aws_tars(key)
        out[key] = {"available": bool(data), "tars": len(data),
                    "reason": "" if data else f"no tars under {_aws_root(key)}"}
    try:
        folder = dl_kaggle.discover_data_dir()
        out["kaggle_sc"] = {"available": True, "tars": 0, "folder": str(folder), "reason": ""}
    except SystemExit:
        out["kaggle_sc"] = {"available": False, "tars": 0,
                            "reason": "not downloaded (see `download --source kaggle_sc`)"}
    opennyai = None
    for candidate in (dl_opennyai.RAW_DIR / "dl", dl_opennyai.RAW_DIR / "local"):
        if candidate.is_dir():
            opennyai = candidate
            break
    out["opennyai_dc"] = {
        "available": opennyai is not None, "tars": 0,
        "folder": str(opennyai) if opennyai else "",
        "reason": "" if opennyai else "not downloaded (see `download --source opennyai_dc`)",
    }
    return out


def _scan_aws(data_tars: list[Path]) -> dict[str, dict[str, int]]:
    """Per-language PDF counts and bytes, read from tar headers only.

    One header pass is what makes proportional language budgets possible
    without parsing a single PDF; the second pass then knows exactly how much
    to take from each language.
    """
    avail: dict[str, dict[str, int]] = {}
    for tar in data_tars:
        lang = _lang_from_path(tar)
        try:
            with tarfile.open(tar) as tf:
                for member in tf:
                    if not member.isfile() or not member.name.lower().endswith(".pdf"):
                        continue
                    slot = avail.setdefault(lang, {"count": 0, "bytes": 0})
                    slot["count"] += 1
                    slot["bytes"] += int(member.size or 0)
        except (tarfile.TarError, OSError) as exc:
            print(f"  {tar.name}: unreadable ({exc}); skipped", file=sys.stderr)
    return avail


class _MetaCache:
    """Metadata JSON for the year currently being processed.

    Metadata tars are one per year; holding one year at a time keeps memory
    bounded no matter how many years are on disk.
    """

    def __init__(self, meta_tars: list[Path]):
        self._tars = meta_tars
        self._year = None
        self._map: dict[str, dict] = {}
        self._loaded_tars: set[str] = set()

    def get(self, stem: str, tar: Path) -> dict:
        year = extract._tar_year(str(tar))
        if year != self._year:
            self._year = year
            self._map = {}
            self._loaded_tars = set()
        if not self._tars:
            return {}
        if stem in self._map:
            return self._map[stem]
        for meta_tar in self._tars:
            if extract._tar_year(str(meta_tar)) != year and year:
                continue
            key = str(meta_tar)
            if key in self._loaded_tars:
                continue
            self._loaded_tars.add(key)
            try:
                with tarfile.open(meta_tar) as tf:
                    for member in tf:
                        if not member.isfile() or not member.name.endswith(".json"):
                            continue
                        handle = tf.extractfile(member)
                        if not handle:
                            continue
                        try:
                            payload = json.loads(handle.read().decode("utf-8", "replace"))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                        if isinstance(payload, dict):
                            self._map[Path(member.name).stem] = payload
            except (tarfile.TarError, OSError) as exc:
                print(f"  metadata {meta_tar.name}: unreadable ({exc})", file=sys.stderr)
        return self._map.get(stem, {})


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def _iter_aws(source: str, budget_bytes: int, per_tar_cap: int | None) -> Iterator[dict]:
    data_tars, meta_tars = _aws_tars(source)
    if not data_tars:
        return
    level = config.LEVEL_FOR_DATASET.get(source, "unknown")
    avail = _scan_aws(data_tars)
    if not avail:
        return
    total = sum(slot["count"] for slot in avail.values()) or 1
    budget = {lang: budget_bytes * slot["count"] / total for lang, slot in avail.items()}
    # A language with almost no judgments would otherwise get a budget too small
    # to hold even one judgment, so "present" must mean "searchable". The floor
    # can never exceed the source budget itself, or a tiny trial run would
    # silently ignore its own cap.
    floor = min(20_000, budget_bytes)
    for lang in budget:
        budget[lang] = max(budget[lang], floor)
    spent = {lang: 0 for lang in budget}
    metas = _MetaCache(meta_tars)
    per_tar_limit = per_tar_cap if per_tar_cap is not None else config.CORPUS_PER_TAR_CAP
    seen_cases = 0

    for tar in data_tars:
        lang = _lang_from_path(tar)
        if spent.get(lang, 0) >= budget.get(lang, 0):
            continue
        taken_here = 0
        try:
            handle = tarfile.open(tar)
        except (tarfile.TarError, OSError) as exc:
            print(f"  {tar.name}: unreadable ({exc}); skipped", file=sys.stderr)
            continue
        with handle as tf:
            for member in tf:
                if not member.isfile() or not member.name.lower().endswith(".pdf"):
                    continue
                if per_tar_limit and taken_here >= per_tar_limit:
                    break
                if spent[lang] >= budget[lang]:
                    break
                stream = tf.extractfile(member)
                if not stream:
                    continue
                text = extract._pdf_to_text(stream.read())
                if len(text) < 200:
                    continue
                stem = Path(member.name).stem
                record = extract._aws_record(
                    source, stem, member.name, text, metas.get(stem, tar), str(tar)
                )
                record["court_level"] = level
                # Members inside a tar carry no language in their path, so the
                # per-file guess is "unknown"; the tar's language folder is the
                # reliable signal and must win.
                if record.get("language") in (None, "", "unknown"):
                    record["language"] = lang
                spent[lang] += len(text)
                taken_here += 1
                seen_cases += 1
                yield record
                if sum(spent.values()) >= budget_bytes:
                    return
    print(f"  {source}: selected {seen_cases} judgments across "
          f"{len(spent)} language folder(s)", file=sys.stderr)


def _iter_tabular(source: str, budget_bytes: int) -> Iterator[dict]:
    if source == "kaggle_sc":
        try:
            folder = dl_kaggle.discover_data_dir()
        except SystemExit:
            return
        default_level = config.LEVEL_FOR_DATASET.get("kaggle_sc", "supreme")
    else:
        folder = None
        for candidate in (dl_opennyai.RAW_DIR / "dl", dl_opennyai.RAW_DIR / "local"):
            if candidate.is_dir():
                folder = candidate
                break
        if folder is None:
            return
        default_level = "district"

    rows = extract._rows_from_folder(folder)
    sample = list(itertools.islice(rows, 50))
    cols = extract._scan_columns(sample)
    if not cols.get("text"):
        print(f"  {source}: no text column in {sorted(sample[0]) if sample else 'no rows'}; "
              "skipped", file=sys.stderr)
        return
    # chain() keeps the sample and the rest of the file one stream: a parquet
    # source must never be materialised as a list.
    for row in itertools.chain(sample, rows):
        if budget_bytes <= 0:
            return
        text = extract._norm_ws(row.get(cols["text"]) or "")
        if len(text) < 200:
            continue
        ident = (extract._norm_ws(row.get("id") or row.get("cnr") or "")[:60]
                 or util.sha1_text(text)[:20])
        court = extract._norm_ws(row.get(cols.get("court")) or "")
        record = {
            "id": f"{source}:{ident}",
            "title": extract._norm_ws(row.get(cols.get("title")) or "")[:300]
                     or f"{source} judgment {ident}",
            "court": court or source,
            "text": text[:400_000],
            "source_url": extract._norm_ws(row.get(cols.get("url")) or "")[:400]
                          or config.SOURCE_URL_FALLBACK.get(source, source),
            "decision_date": extract._norm_ws(row.get(cols.get("date")) or "")[:32] or "unknown",
            "dataset": source,
            "citation": extract._norm_ws(row.get(cols.get("citation")) or "")[:120],
            "language": util.script_language(text),
            "jurisdiction": util.jurisdiction_tag(row),
            "archive_member": f"{extract._norm_ws(row.get('__file'))}#{ident}"[:240],
            "court_level": util.court_level(court, default_level),
        }
        budget_bytes -= len(record["text"])
        yield record


# ---------------------------------------------------------------------------
# Writing the single artifact
# ---------------------------------------------------------------------------
def _open_db(path: Path, fresh: bool) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fresh:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(path) + suffix)
            if candidate.exists():
                candidate.unlink()
    conn = sqlite3.connect(path, timeout=60)
    # DELETE journal, not WAL: the deliverable has to be ONE file on disk.
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=NORMAL")
    for statement in SCHEMA:
        conn.execute(statement)
    conn.commit()
    return conn


def _store_vectors(conn: sqlite3.Connection, embedder, items: list[tuple[str, str]]) -> int:
    """Embed (chunk_id, text) pairs and store int8 blobs. Returns the dimension."""
    import numpy as np

    import embed as embed_mod

    texts = [text[: config.EMBED_MAX_CHARS] for _, text in items]
    vectors = np.asarray(list(embedder.embed(texts)), dtype="float32")
    dim = int(vectors.shape[1])
    quantised = embed_mod._quantise(vectors)
    conn.executemany(
        "INSERT OR REPLACE INTO vectors(chunk_id,dim,vec) VALUES(?,?,?)",
        [(chunk_id, dim, quantised[i].tobytes()) for i, (chunk_id, _) in enumerate(items)],
    )
    conn.commit()
    return dim


def _backfill_vectors(conn: sqlite3.Connection, embedder) -> int:
    """Fill vectors for chunks that were written but not embedded.

    A crash between "passages inserted" and "vectors committed" would
    otherwise leave a resumable file with unsearchable holes, because the case
    is already marked done.
    """
    cursor = conn.execute(
        "SELECT chunk_id, header || ' ' || text FROM chunks "
        "WHERE chunk_id NOT IN (SELECT chunk_id FROM vectors)"
    )
    filled = 0
    while True:
        batch = cursor.fetchmany(config.EMBED_BATCH * 4)
        if not batch:
            break
        _store_vectors(conn, embedder, [(row[0], row[1]) for row in batch])
        filled += len(batch)
    return filled


class _Writer:
    def __init__(self, conn: sqlite3.Connection, embedder, model_name: str):
        self.conn = conn
        self.embedder = embedder
        self.model_name = model_name
        self.buffer: list[tuple[str, str]] = []
        self.stats = {
            "cases": 0, "chunks": 0, "vectors": 0, "text_bytes": 0, "dim": 0,
            "per_source": {}, "per_level": {}, "per_language": {}, "per_dataset": {},
        }
        self._done: set[str] = {row[0] for row in conn.execute("SELECT case_id FROM done_cases")}

    def already_done(self, case_id: str) -> bool:
        return case_id in self._done

    def add(self, record: dict) -> None:
        case_id = str(record["id"])
        level = record.get("court_level") or "unknown"
        language = record.get("language") or "unknown"
        court = str(record.get("court") or "")
        jurisdiction = str(record.get("jurisdiction") or "IN")
        text = str(record.get("text") or "")
        header = util.level_header(level, court, jurisdiction, language, config.LEVEL_LABEL)

        head_text = " ".join([
            str(record.get("title") or "")[:300], court,
            str(record.get("decision_date") or ""), chunk_mod._head_slice(text),
        ]).strip()
        pieces = chunk_mod._chunk_one(text, head_text)
        self.conn.execute(
            "INSERT OR REPLACE INTO cases VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (case_id, str(record.get("title") or "")[:300], court[:250], level,
             jurisdiction, language, str(record.get("decision_date") or "")[:40],
             str(record.get("citation") or "")[:120], str(record.get("dataset") or ""),
             str(record.get("source_url") or "")[:400], len(text), time.time()),
        )
        for ordinal, (kind, piece) in enumerate(pieces):
            chunk_id = f"{case_id}#{ordinal}"
            self.conn.execute(
                "INSERT OR REPLACE INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (chunk_id, case_id, kind, ordinal, level, language, court[:250],
                 jurisdiction, str(record.get("decision_date") or "")[:40], header, piece),
            )
            self.buffer.append((chunk_id, f"{header} {piece}"))
            self.stats["chunks"] += 1
        # External-content FTS: index the passages just inserted, no copy of
        # the text. Rowids match, so BM25 hits join straight back to chunks.
        self.conn.execute(
            "INSERT INTO chunk_fts(rowid, text) SELECT rowid, text FROM chunks "
            "WHERE case_id=?", (case_id,)
        )
        self.conn.execute("INSERT OR IGNORE INTO done_cases(case_id) VALUES(?)", (case_id,))
        self._done.add(case_id)

        self.stats["cases"] += 1
        self.stats["text_bytes"] += len(text)
        for key, bucket in (("per_source", record.get("dataset")),
                            ("per_level", level), ("per_language", language)):
            target = self.stats[key]
            target[bucket] = target.get(bucket, 0) + 1
        if len(self.buffer) >= config.EMBED_BATCH * 4:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.stats["dim"] = _store_vectors(self.conn, self.embedder, self.buffer)
        self.stats["vectors"] += len(self.buffer)
        self.buffer = []


def _finalize(conn: sqlite3.Connection, writer: "_Writer", *, model_name: str,
              started: float, notes: list[str], meta_extra: dict[str, str],
              extra: dict[str, Any]) -> dict:
    """Write meta, close the file, print the measured result. Shared by every
    provider that fills the single corpus file (tar path and parquet path)."""
    meta = {
        "model": model_name,
        "dim": str(writer.stats["dim"]),
        "quantisation": "int8 /127",
        "text_mb": f"{writer.stats['text_bytes'] / 1e6:.2f}",
        "cases": str(writer.stats["cases"]),
        "chunks": str(writer.stats["chunks"]),
        "vectors": str(writer.stats["vectors"]),
        "per_source": json.dumps(writer.stats["per_source"], sort_keys=True),
        "per_level": json.dumps(writer.stats["per_level"], sort_keys=True),
        "per_language": json.dumps(writer.stats["per_language"], sort_keys=True),
        "languages_supported_by_model": json.dumps(list(config.CORPUS_LANGUAGES)),
        "built_at": str(int(time.time())),
        "seconds": f"{time.perf_counter() - started:.1f}",
        "sampling_notes": json.dumps(notes),
    }
    meta.update(meta_extra)
    conn.executemany("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", meta.items())
    conn.commit()
    conn.execute("VACUUM")
    conn.close()

    result = dict(writer.stats)
    result.update({
        "file": str(config.SINGLE_DB),
        "file_mb": round(config.SINGLE_DB.stat().st_size / 1e6, 2) if config.SINGLE_DB.exists() else 0,
        "model": model_name,
        "seconds": round(time.perf_counter() - started, 1),
    })
    result.update(extra)
    target = result.get("target_mb")
    suffix = f", target {target:.0f} MB)" if target else ")"
    print(
        f"\ncorpus done: {result['cases']} judgments, {result['chunks']} chunks, "
        f"{result['vectors']} vectors, {result['text_bytes'] / 1e6:.1f} MB of text "
        f"-> {result['file']} ({result['file_mb']} MB" + suffix
    )
    if target and result["file_mb"] > target * 1.3:
        print(
            f"  note: the file is {result['file_mb'] / target:.2f}x the target; "
            "re-run with a smaller --target-mb if that matters (the ratio above is "
            "the measured, not assumed, one).",
            file=sys.stderr,
        )
    print(f"  tiers: {writer.stats['per_level']}")
    print(f"  languages: {writer.stats['per_language']}")
    return result


def build(target_mb: float | None = None, sources: list[str] | None = None,
          model_name: str | None = None, embedder=None, per_tar_cap: int | None = None,
          fresh: bool = True) -> dict:
    target_mb = float(target_mb or config.CORPUS_TARGET_MB)
    if target_mb <= 0:
        raise SystemExit("--target-mb must be > 0")
    # The flag is the size of the file the user will check on disk; text is the
    # only part we can steer, so it takes a fixed share of that budget.
    file_target = int(target_mb * 1e6)
    target_bytes = int(file_target * config.CORPUS_TEXT_SHARE_OF_FILE)
    model_name = model_name or config.DENSE_MODEL

    available = available_sources()
    # Only sources that carry judgment text can be curated; DDL is metadata.
    text_sources = ["aws_sc", "aws_hc", "kaggle_sc", "opennyai_dc"]
    wanted = [key for key in (sources or text_sources) if key in text_sources]
    unknown = [key for key in (sources or []) if key not in text_sources]
    if unknown:
        raise SystemExit(
            f"curate only takes text sources; unknown {unknown}. Choose from {text_sources} "
            "(ddl_meta is metadata-only and is never embedded)."
        )
    chosen = [key for key in wanted if available.get(key, {}).get("available")]
    for key in wanted:
        if key not in chosen:
            print(f"  skip {key}: {available.get(key, {}).get('reason', 'unavailable')}",
                  file=sys.stderr)
    if not chosen:
        raise SystemExit(
            "No downloaded source to curate from. Run `download` first, or check "
            f"the raw folders under {config.RAW}."
        )

    # Tier budgets, normalised over the tiers that actually have data, then
    # split evenly between sources of the same tier.
    levels: dict[str, list[str]] = {}
    for key in chosen:
        level = config.LEVEL_FOR_DATASET.get(key) or "district"
        levels.setdefault(level, []).append(key)
    weight_total = sum(config.CORPUS_MIX.get(level, 0.1) for level in levels) or 1.0
    plan: dict[str, int] = {}
    for level, keys in levels.items():
        share = int(target_bytes * config.CORPUS_MIX.get(level, 0.1) / weight_total)
        for key in keys:
            plan[key] = share // len(keys)

    if embedder is None:
        from fastembed import TextEmbedding

        embedder = TextEmbedding(model_name=model_name)

    conn = _open_db(config.SINGLE_DB, fresh=fresh)
    writer = _Writer(conn, embedder, model_name)
    started = time.perf_counter()
    if not fresh:
        filled = _backfill_vectors(conn, embedder)
        if filled:
            writer.stats["vectors"] += filled
            print(f"  resume: backfilled {filled} missing vector(s)")

    print(f"curate: target {target_mb:.0f} MB file, so {target_bytes / 1e6:.1f} MB of text, "
          f"across {len(chosen)} source(s)")
    for key, budget in plan.items():
        print(f"  {key}: budget {budget / 1e6:.1f} MB ({config.LEVEL_FOR_DATASET.get(key, 'mixed')})")
    for key in chosen:
        budget = plan[key]
        iterator = (_iter_aws(key, budget, per_tar_cap) if key in ("aws_sc", "aws_hc")
                    else _iter_tabular(key, budget))
        taken = 0
        for record in iterator:
            if writer.already_done(str(record["id"])):
                continue
            writer.add(record)
            taken += 1
            if taken % 100 == 0:
                print(f"    {key}: {taken} judgments, "
                      f"{writer.stats['text_bytes'] / 1e6:.1f} MB total", file=sys.stderr)
    writer.flush()

    notes = [
        "Selection is the head of each archive's member list, not a random sample.",
        "Tier budgets are normalised over tiers present on disk; a tier with no data reports a shortfall.",
        "Language budgets are proportional to the judgments available in each language folder.",
        "Every chunk stores its tier and a header line that is part of the embedded text.",
    ]
    return _finalize(
        conn, writer, model_name=model_name, started=started, notes=notes,
        meta_extra={
            "target_file_mb": f"{target_mb:g}",
            "text_budget_mb": f"{target_bytes / 1e6:.2f}",
            "sources": json.dumps(chosen),
            "mode": "tar archives (local)",
        },
        extra={"target_mb": target_mb, "sources": chosen},
    )


def _report(result: dict) -> None:
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    _report(build())
