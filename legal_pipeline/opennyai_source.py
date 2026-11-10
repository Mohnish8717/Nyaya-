"""OpenNyAI InJudgements dataset -> the single embeddings file.

This dataset is different in kind from the two Open Data parquet files: its
parquet carries the **judgment text itself** (`Text`), so nothing has to be
scraped or PDF-parsed. That is why it is the cheapest real corpus to embed.

Published schema (from the dataset card's `dataset_info`, one `train` split):

    Titles                 string   cause title
    Court_Name             string   e.g. "High Court of Karnataka"
    Cites                  int64    outbound citations
    Cited_by               int64    inbound citations (importance signal)
    Doc_url                string   Indian Kanoon document URL
    Text                   string   full judgment text
    Doc_size               int64
    Case_Type              string   subject bucket (tax, criminal, civil, ...)
    Court_Type             string   tier: supreme / high court / district ...
    Court_Name_Normalized  string

11,970 judgments, ~629 MB of text, ~293 MB of parquet, apache-2.0, English.
District coverage is Delhi and Bengaluru only — the dataset's own card says so.

Access: the repo is gated, so a Hugging Face token with the terms accepted is
required (`huggingface-cli login` or `HF_TOKEN`). There is no unauthenticated
path, and this module reports that instead of failing obscurely.

    python pipeline.py opennyai                    # download if needed, then embed
    python pipeline.py opennyai --limit 500        # trial slice
    python pipeline.py opennyai --local ~/InJudgements_dataset
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import config
import curate
import dl_opennyai
import util

# Field names exactly as published; the lookup below tolerates case and
# underscore/spacing differences because column names travel through parquet
# writers that are not always consistent.
FIELDS = {
    "title": "Titles",
    "court": "Court_Name",
    "court_type": "Court_Type",
    "court_normalized": "Court_Name_Normalized",
    "text": "Text",
    "url": "Doc_url",
    "cites": "Cites",
    "cited_by": "Cited_by",
    "case_type": "Case_Type",
    "doc_size": "Doc_size",
}

_TIER = (
    ("supreme", re.compile(r"supreme", re.I)),
    # Court_Type values arrive both as "High Court" and "high_court".
    ("high", re.compile(r"high[\s_]*court|highcourt", re.I)),
    ("district", re.compile(r"district|sessions|family|magistrate|civil[\s_]+judge|munsif", re.I)),
)

MIN_TEXT_CHARS = 200


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _resolve_columns(names: list[str]) -> dict[str, str]:
    """Map published names onto actual parquet columns, tolerantly."""
    index = {_norm(name): name for name in names}
    out: dict[str, str] = {}
    for key, published in FIELDS.items():
        out[key] = index.get(_norm(published), "")
    return out


def _tier(value: Any, court: Any) -> str:
    """Tier from Court_Type, else from the court name, else honest 'unknown'.

    'tribunal' is deliberately NOT folded into 'district': a tribunal order is
    not a district-court judgment, and mislabelling it would let the reader
    model treat it as one.
    """
    text = str(value or "").strip()
    for level, pattern in _TIER:
        if pattern.search(text):
            return level
    if text and re.search(r"tribunal|commission|authority|board", text, re.I):
        return "unknown"
    return util.court_level(str(court or ""), default="unknown")


def data_dir() -> Path | None:
    """Where a snapshot would live, if one is already present."""
    for candidate in (dl_opennyai.RAW_DIR / "dl", dl_opennyai.RAW_DIR / "local"):
        if candidate.is_dir() and any(candidate.rglob("*.parquet")):
            return candidate
    return None


def ensure_data(local: str | None = None, download: bool = True) -> Path:
    """Local parquet folder, downloading the full snapshot when absent."""
    if local:
        folder = Path(local).expanduser()
        if not folder.is_dir():
            raise SystemExit(f"--local must be an existing folder, got {folder}")
        if not any(folder.rglob("*.parquet")):
            raise SystemExit(f"{folder} holds no .parquet file")
        return folder
    existing = data_dir()
    if existing is not None:
        print(f"opennyai: using existing data at {existing}", file=sys.stderr)
        return existing
    if not download:
        raise SystemExit(
            "No OpenNyAI data on disk. Run without --no-download to fetch it "
            "(~293 MB, public download)."
        )
    print("opennyai: downloading the full dataset (~293 MB parquet) …", file=sys.stderr)
    dl_opennyai.download_command([])
    folder = data_dir()
    if folder is None:
        raise SystemExit(
            "Download reported success but no parquet was found under "
            f"{dl_opennyai.RAW_DIR}. Check the network/proxy and re-run."
        )
    return folder


def _value(row: dict, columns: dict[str, str], key: str) -> Any:
    name = columns.get(key, "")
    return row.get(name) if name else None


def _record_from_row(row: dict, columns: dict[str, str], origin: str,
                     stats: dict[str, int]) -> dict | None:
    """One parquet/dataset row -> one unified record, or None if unusable.

    Shared by the parquet reader and the `datasets` reader so the two paths
    cannot drift: the same row must produce the same record either way.
    """
    text = str(_value(row, columns, "text") or "")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(text) < MIN_TEXT_CHARS:
        stats["skipped_short"] = stats.get("skipped_short", 0) + 1
        return None
    title = str(_value(row, columns, "title") or "").strip()
    court = str(_value(row, columns, "court") or "").strip()
    if not court:
        court = str(_value(row, columns, "court_normalized") or "").strip()
    url = str(_value(row, columns, "url") or "").strip()
    cited_by = _value(row, columns, "cited_by")
    return {
        "id": "opennyai:" + (util.sha1_text(url)[:20] if url
                             else util.sha1_text(text)[:20]),
        "title": (title or f"Judgment {court}")[:300],
        "court": court[:160] or "InJudgements",
        "text": text[:400_000],
        "source_url": (url or config.SOURCE_URL_FALLBACK["opennyai_dc"])[:400],
        "decision_date": "unknown",
        "dataset": "opennyai_dc",
        "citation": "",
        "language": util.script_language(text),
        "jurisdiction": util.jurisdiction_tag({"court": court}),
        "archive_member": f"{origin}#{record_key(url, text)}"[:240],
        "court_level": _tier(_value(row, columns, "court_type"), court),
        # Kept for the meta report; the writer ignores unknown keys.
        "case_type": str(_value(row, columns, "case_type") or ""),
        "cited_by": int(cited_by or 0),
    }


def iter_records(folder: Path, limit: int | None = None,
                 stats: dict[str, int] | None = None) -> Iterator[dict]:
    """Stream the parquet as unified records (text included, no PDFs).

    `stats["skipped_short"]` counts rows dropped for having no usable text, so
    the run can report what it discarded instead of silently shrinking.
    """
    stats = stats if stats is not None else {}
    import pyarrow.parquet as pq

    files = sorted(folder.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No .parquet under {folder}")
    emitted = 0
    for path in files:
        handle = pq.ParquetFile(str(path))
        columns = _resolve_columns(handle.schema_arrow.names)
        if not columns["text"]:
            raise SystemExit(
                f"{path.name}: no Text column; observed {handle.schema_arrow.names}. "
                "This is not the InJudgements layout."
            )
        wanted = [c for c in columns.values() if c]
        for batch in handle.iter_batches(batch_size=256, columns=wanted):
            for row in batch.to_pylist():
                record = _record_from_row(row, columns, path.name, stats)
                if record is None:
                    continue
                yield record
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def iter_records_via_datasets(limit: int | None = None,
                              stats: dict[str, int] | None = None) -> Iterator[dict]:
    """The same records, read through `datasets.load_dataset` instead of parquet.

    This is the path the dataset's own card documents:

        from datasets import load_dataset
        ds = load_dataset("opennyaiorg/InJudgements_dataset")

    It streams, so the 629 MB of text is never materialised in memory or in the
    Arrow cache; rows arrive one at a time and go through the same mapping as
    the parquet reader. A token is still required (the repo is gated for file
    access) — `datasets` reads HF_TOKEN / the `huggingface-cli login` cache.
    """
    stats = stats if stats is not None else {}
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "`datasets` is not installed. Either install it (pip install datasets) "
            "or drop --via datasets to read the parquet directly."
        )
    try:
        stream = load_dataset(config.HF_REPO, split="train", streaming=True)
    except Exception as exc:  # noqa: BLE001 - gated repo / network
        raise SystemExit(
            f"load_dataset({config.HF_REPO!r}) failed: {str(exc).splitlines()[0]}\n"
            "If this is a 401/gated error, accept the terms on the dataset page and "
            "run `huggingface-cli login` (or export HF_TOKEN), then retry."
        )
    columns = _resolve_columns(list(stream.features.keys()))
    if not columns["text"]:
        raise SystemExit(
            f"InJudgements stream has no Text column; observed {list(stream.features)}."
        )
    emitted = 0
    for row in stream:
        record = _record_from_row(row, columns, "datasets-stream", stats)
        if record is None:
            continue
        yield record
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def record_key(url: str, text: str) -> str:
    return util.sha1_text(url)[:12] if url else util.sha1_text(text)[:12]


def build(local: str | None = None, download: bool = True, limit: int | None = None,
          model_name: str | None = None, embedder=None, fresh: bool = True,
          via: str = "parquet") -> dict:
    """Download (when needed) and embed InJudgements into the one corpus file.

    via="parquet"   reads the downloaded parquet shards with pyarrow — resumable,
                    low memory, and what the CLI default uses.
    via="datasets"  streams through `datasets.load_dataset`, the route the
                    dataset card documents; same records, same writer.
    """
    if via not in ("parquet", "datasets"):
        raise SystemExit(f"via must be 'parquet' or 'datasets', got {via!r}")
    folder = None
    if via == "datasets":
        if local:
            raise SystemExit("--local needs --via parquet (it names a parquet folder).")
        rows = iter_records_via_datasets
        where = f"streaming {config.HF_REPO} through datasets.load_dataset"
    else:
        folder = ensure_data(local=local, download=download)
        rows = iter_records
        where = str(folder)
    model_name = model_name or config.DENSE_MODEL
    if embedder is None:
        from fastembed import TextEmbedding

        embedder = TextEmbedding(model_name=model_name)

    conn = curate._open_db(config.SINGLE_DB, fresh=fresh)
    writer = curate._Writer(conn, embedder, model_name)
    started = time.perf_counter()
    if not fresh:
        filled = curate._backfill_vectors(conn, embedder)
        if filled:
            writer.stats["vectors"] += filled

    print(f"opennyai: reading {where}")
    seen = 0
    counters: dict[str, int] = {}
    stream = rows(limit=limit, stats=counters) if via == "datasets" else rows(
        folder, limit=limit, stats=counters)
    for record in stream:
        seen += 1
        if writer.already_done(record["id"]):
            continue
        writer.add(record)
        if seen % 250 == 0:
            print(f"    {seen} judgments, {writer.stats['chunks']} chunks", file=sys.stderr)
            conn.commit()
    writer.flush()

    notes = [
        "Text comes from the dataset's own Text column; no PDF scraping or parsing.",
        "Court_Type supplies the tier; when it is missing the court name is used, and an unrecognised court stays 'unknown'.",
        "decision_date is not published by this dataset, so it is stored as 'unknown' rather than guessed.",
        "District coverage is Delhi and Bengaluru only, as the dataset card states.",
        "Licence: apache-2.0; cite OpenNyAI if you publish results.",
    ]
    result = curate._finalize(
        conn, writer, model_name=model_name, started=started, notes=notes,
        meta_extra={
            "mode": f"opennyai InJudgements ({via})",
            "source_repo": config.HF_REPO,
            "rows_seen": str(seen),
            "skipped_short": str(counters.get("skipped_short", 0)),
            "sources": json.dumps(["opennyai_dc"]),
            "licence": "apache-2.0",
        },
        extra={"sources": ["opennyai_dc"], "rows_seen": seen,
               "skipped_short": counters.get("skipped_short", 0)},
    )
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, default=str))
