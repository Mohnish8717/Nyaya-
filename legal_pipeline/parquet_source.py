"""Parquet-first ingest: structured metadata selects, PDFs supply the words.

Why this exists: the Open Data parquet files are the cheapest way to know what
the corpus contains — case identity, court, bench, decision date, citation,
judge, disposal — without touching a 1.25 TiB tar forest. What they do *not*
contain is the judgment text: `path` (Supreme Court) and `pdf_link` (High
Court) are storage pointers, and the maintainers' own pipeline writes text only
into the PDF archives. So this module does the honest two-step:

  1. read the parquet **with column projection** (the bulky `raw_html` column is
     never transferred), pick rows spread across years and courts, and take the
     tier/court/date/citation labels straight from structured fields;
  2. fetch only the PDFs those rows point at, inside the same transfer budget,
     parse them, and feed the passages to the same single-file writer `curate`
     uses.

The result is `out/corpus_embeddings.sqlite3` — one file, tier-labelled, with
`supreme` and `high` passages, no district courts, and a hard byte cap on
everything pulled over the wire.

    python pipeline.py parquet --target-mb 150
    python pipeline.py parquet --target-mb 20 --source aws_sc --years 2020,2023 --plan-only

Honest limits: the PDF a row points at is the one fetched (English for the
Supreme Court, whatever the High Court partition publishes), so this path's
language mix is the mix of those files; regional-language PDFs remain available
through the tar path (`curate`). Rows whose PDF is missing (`pdf_exists` false
or an unreadable pointer) are counted and skipped, never invented.
"""
from __future__ import annotations

import itertools
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import config
import curate
import dl_aws
import extract
import util

SBUCKET, HBUCKET = config.AWS_BUCKETS["aws_sc"], config.AWS_BUCKETS["aws_hc"]
COURT_MAP_PATH = Path(__file__).resolve().parent.parent / "data" / "hc_court_jurisdictions.json"

_LANG_WORDS = {
    "english": "en", "hindi": "hi", "tamil": "ta", "telugu": "te", "kannada": "kn",
    "malayalam": "ml", "marathi": "mr", "bengali": "bn", "gujarati": "gu",
    "punjabi": "pa", "odia": "or", "oriya": "or", "urdu": "ur", "assamese": "as",
}


class Budget:
    """Byte budget for everything pulled over the wire, with a hard refusal."""

    def __init__(self, total: int):
        self.total = max(int(total), 0)
        self.spent = 0

    def left(self) -> int:
        return max(self.total - self.spent, 0)

    def charge(self, n: int, what: str) -> None:
        self.spent += max(int(n), 0)
        if self.spent > self.total * 1.001:
            raise _BudgetDone(f"{what}: budget exhausted")


class _BudgetDone(Exception):
    """Raised when the transfer budget is spent; caught as a normal stop."""


# ---------------------------------------------------------------------------
# Partition discovery (listing only: no object bodies are read)
# ---------------------------------------------------------------------------
def _list_folders(client, bucket: str, prefix: str) -> list[str]:
    paginator = client.get_paginator("list_objects_v2")
    out: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        out.extend(item["Prefix"] for item in page.get("CommonPrefixes", []))
    return out


def _list_objects(client, bucket: str, prefix: str) -> list[dict]:
    paginator = client.get_paginator("list_objects_v2")
    out: list[dict] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        out.extend(page.get("Contents", []))
    return out


def _sc_partitions(client, years: list[int] | None) -> list[dict]:
    available = [int(p.strip("/").split("=")[-1])
                 for p in _list_folders(client, SBUCKET, "metadata/parquet/")
                 if re.search(r"year=\d{4}$", p.strip("/"))]
    if years:
        wanted = sorted(years)
    else:
        # Historical breadth plus the newest year: an even spread alone stops
        # a decade short of the present, which is where live questions live.
        wanted = sorted(set(_spread(sorted(available), 8)) | ({available[-1]} if available else set()))
    wanted = [int(y) for y in wanted]
    parts = []
    for year in wanted:
        if available and year not in available:
            print(f"  aws_sc: no parquet for {year}; skipped", file=sys.stderr)
            continue
        parts.append({
            "source": "aws_sc", "level": "supreme", "bucket": SBUCKET,
            "key": f"metadata/parquet/year={year}/metadata.parquet",
            "label": f"SC {year}", "year": year, "court": "",
        })
    return parts


def _hc_partitions(client, years: list[int] | None, court: str | None,
                   max_partitions: int = 12) -> list[dict]:
    year_folders = [p for p in _list_folders(client, HBUCKET, "metadata/parquet/")
                    if re.search(r"year=\d{4}/$", p)]
    available = sorted(int(p.strip("/").split("=")[-1]) for p in year_folders)
    if not available:
        return []
    # High Court volume is concentrated in recent years; the oldest partitions
    # hold a handful of rows, so an even spread across 1950-2026 samples mostly
    # empty files. Take the newest years, which is also what a live caller asks
    # about.
    wanted = sorted(years) if years else available[-3:]
    parts: list[dict] = []
    for year in wanted:
        if year not in available:
            print(f"  aws_hc: no parquet for {year}; skipped", file=sys.stderr)
            continue
        courts = [p.strip("/").split("=")[-1]
                  for p in _list_folders(client, HBUCKET, f"metadata/parquet/year={year}/")
                  if "court=" in p]
        if court:
            courts = [c for c in courts if c.replace("~", "_") == court.replace("~", "_")]
        for code in _spread(sorted(courts), max(2, max_partitions // max(len(wanted), 1))):
            for bench in _list_folders(client, HBUCKET,
                                       f"metadata/parquet/year={year}/court={code}/"):
                parts.append({
                    "source": "aws_hc", "level": "high", "bucket": HBUCKET,
                    "key": f"{bench}metadata.parquet",
                    "label": f"HC {code} {year}", "year": year, "court": code,
                })
                break  # one bench per court is enough to label and sample
    return parts


def _spread(items: list, n: int) -> list:
    """Take n items spread evenly across the list, keeping order."""
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[min(int(i * step), len(items) - 1)] for i in range(n)]


def plan(client=None, sources: list[str] | None = None, years: list[int] | None = None,
         court: str | None = None) -> list[dict]:
    client = client or dl_aws.s3()
    wanted = sources or ["aws_sc", "aws_hc"]
    parts: list[dict] = []
    if "aws_sc" in wanted:
        parts += _sc_partitions(client, years)
    if "aws_hc" in wanted:
        parts += _hc_partitions(client, years, court)
    return parts


def describe(parts: list[dict], client=None) -> None:
    """Sizes of the partitions, so a run is chosen rather than gambled."""
    client = client or dl_aws.s3()
    total = 0
    for part in parts:
        objs = [o for o in _list_objects(client, part["bucket"], part["key"]) if o["Key"] == part["key"]]
        size = int(objs[0]["Size"]) if objs else 0
        total += size
        print(f"  {part['label']:<18} {part['key']}")
        print(f"  {'':<18} {size / 1e6:8.2f} MB on S3 "
              f"(only the projected columns are transferred)")
    print(f"  {len(parts)} partition(s), {total / 1e6:.1f} MB if read whole; "
          "projection excludes raw_html, the largest column")


# ---------------------------------------------------------------------------
# Reading rows (projected) and resolving PDF pointers
# ---------------------------------------------------------------------------
def _pick(names: list[str], *candidates: str) -> str:
    lower = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return ""


def _columns(source: str, names: list[str]) -> dict[str, str]:
    if source == "aws_sc":
        return {
            "title": _pick(names, "title"),
            "description": _pick(names, "description"),
            "citation": _pick(names, "citation", "nc_display"),
            "case_id": _pick(names, "case_id", "cnr"),
            "date": _pick(names, "decision_date", "date_of_decision"),
            "court": _pick(names, "court"),
            "language": _pick(names, "available_languages", "language"),
            "pointer": _pick(names, "path", "pdf_link", "pdf_url"),
            "judge": _pick(names, "judge", "author_judge"),
        }
    return {
        "title": _pick(names, "title"),
        "description": _pick(names, "description"),
        "citation": _pick(names, "citation"),
        "case_id": _pick(names, "cnr", "case_id"),
        "date": _pick(names, "decision_date"),
        "court": _pick(names, "court"),
        "language": _pick(names, "language"),
        "pointer": _pick(names, "pdf_link", "path", "pdf_url"),
        "judge": _pick(names, "judge"),
        "court_code": _pick(names, "court_code"),
        "exists": _pick(names, "pdf_exists"),
    }


def _selected_column_bytes(parquet_file, columns: list[str]) -> int:
    """Compressed size of just the columns we read, from the footer metadata."""
    wanted = {c for c in columns if c}
    names = parquet_file.schema_arrow.names
    indexes = {names.index(c) for c in wanted if c in names}
    total = 0
    for group in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(group)
        for index in indexes:
            total += int(row_group.column(index).total_compressed_size)
    return total


def _iter_partition(fs, part: dict, budget: Budget, parquet_budget: Budget,
                    max_rows: int) -> Iterator[dict]:
    """Yield at most `max_rows` dicts from one parquet partition, projection-only."""
    import pyarrow.parquet as pq

    # `path` lets a local mirror (and the smoke test) be read through the same
    # code path as S3; otherwise the bucket/key pair is what pyarrow needs.
    path = part.get("path") or f"{part['bucket']}/{part['key']}"
    try:
        handle = pq.ParquetFile(path, filesystem=fs)
    except Exception as exc:  # noqa: BLE001 - a bad partition must not kill the run
        print(f"  {part['label']}: unreadable parquet ({exc})", file=sys.stderr)
        return
    names = handle.schema_arrow.names
    columns = _columns(part["source"], names)
    wanted = [c for c in columns.values() if c]
    total_rows = int(handle.metadata.num_rows or 0)
    if not total_rows:
        return
    selected_bytes = _selected_column_bytes(handle, wanted)
    if selected_bytes > int(config.PARQUET_PARTITION_WARN_MB * 1e6):
        print(f"  {part['label']}: selected columns are {selected_bytes / 1e6:.1f} MB; "
              "reading a bounded prefix only", file=sys.stderr)

    rows_seen = 0
    try:
        for batch in handle.iter_batches(batch_size=256, columns=wanted):
            rows = batch.to_pylist()
            for row in rows:
                rows_seen += 1
                # Charge the projected bytes in proportion to rows consumed.
                # The total budget is charged too, so parquet reads can never
                # push the run past the cap it advertised.
                # Proportional charge, with a one-page floor: parquet stores
                # columns in ~1 MB pages, so even a 256-row batch pulls a page.
                share = max(int(selected_bytes * len(rows) / total_rows), 64 * 1024)
                budget.charge(share, f"{part['label']} parquet")
                parquet_budget.charge(share, f"{part['label']} parquet")
                yield {**{k: row.get(v) for k, v in columns.items()}, "__partition": part}
            if rows_seen >= max_rows:
                return
    except _BudgetDone:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"  {part['label']}: read stopped ({exc})", file=sys.stderr)


def resolve_pointer(pointer: Any, bucket: str) -> tuple[str, str]:
    """(bucket, key) for a parquet pointer, or ("", "") when it is unusable."""
    text = str(pointer or "").strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return "", ""
    if text.startswith("s3://"):
        rest = text[5:]
        if "/" not in rest:
            return "", ""
        bkt, key = rest.split("/", 1)
        return bkt, key.lstrip("/")
    if text.startswith("http://") or text.startswith("https://"):
        match = re.match(r"https?://([^/]+)/(.+)", text)
        if not match:
            return "", ""
        host, key = match.group(1), match.group(2)
        bkt = host.split(".")[0]
        return bkt, key
    # A bare key ("data/pdf/...") in the source bucket.
    return bucket, text.lstrip("/")


def fetch_pdf(client, bucket: str, key: str, budget: Budget) -> bytes:
    """One PDF, byte-accounted, capped so a scan cannot swallow the run."""
    cap = int(config.PARQUET_MAX_PDF_MB * 1e6)
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
        size = int(obj.get("ContentLength") or 0)
        if size and size > cap:
            return b""
        data = obj["Body"].read(cap + 1)
    except Exception:  # noqa: BLE001 - missing object is a skip, not a crash
        return b""
    if len(data) > cap:
        return b""
    budget.charge(len(data), f"pdf {key}")
    return data


# ---------------------------------------------------------------------------
# Rows -> records
# ---------------------------------------------------------------------------
def _court_entry(code: str) -> dict:
    if not code or not COURT_MAP_PATH.exists():
        return {}
    try:
        return json.loads(COURT_MAP_PATH.read_text(encoding="utf-8")).get(code.replace("~", "_"), {}) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _language_from_row(value: Any) -> str:
    text = str(value or "").lower()
    found = [code for word, code in _LANG_WORDS.items() if word in text]
    if not found:
        return ""
    # One PDF was fetched, so report the language of that document: English
    # when it is available, otherwise the first listed language.
    return "en" if "en" in found else found[0]


def _record(part: dict, row: dict, text: str) -> dict:
    source = part["source"]
    level = part["level"]
    code = str(row.get("court_code") or part.get("court") or "")
    entry = _court_entry(code)
    court = str(row.get("court") or entry.get("name") or "").strip()
    if not court:
        court = "Supreme Court of India" if source == "aws_sc" else f"High Court {code}".strip()
    case_id = str(row.get("case_id") or "").strip()
    pointer = str(row.get("pointer") or "").strip()
    ident = case_id or util.sha1_text(pointer or text)[:20]
    language = _language_from_row(row.get("language")) or util.script_language(text)
    jurisdiction = "IN" if source == "aws_sc" else (entry.get("jurisdiction") or "IN")
    bucket, key = resolve_pointer(pointer, part["bucket"])
    url = (f"https://{bucket}.s3.ap-south-1.amazonaws.com/{key}" if key else
           config.SOURCE_URL_FALLBACK.get(source, source))
    title = str(row.get("title") or "").strip()
    if not title:
        title = str(row.get("description") or "").strip()[:200] or f"{court} {ident}"
    date = str(row.get("date") or "").strip()[:40] or "unknown"
    return {
        "id": f"{source}:{ident}",
        "title": title[:300],
        "court": court[:160],
        "text": text,
        "source_url": url[:400],
        "decision_date": date,
        "dataset": source,
        "citation": str(row.get("citation") or "").strip()[:120],
        "language": language,
        "jurisdiction": jurisdiction,
        "archive_member": (key or pointer)[:240],
        "court_level": level,
    }


def pull(target_mb: float | None = None, sources: list[str] | None = None,
         years: list[int] | None = None, court: str | None = None,
         model_name: str | None = None, embedder=None, fresh: bool = True,
         rows_per_partition: int | None = None, client=None, fs=None,
         plan_only: bool = False, partitions: list[dict] | None = None) -> dict:
    """Read parquet, fetch pointed-at PDFs, write the single corpus file."""
    target_mb = float(target_mb or config.CORPUS_TARGET_MB)
    rows_per_partition = int(rows_per_partition or config.PARQUET_ROWS_PER_PARTITION)
    client = client or dl_aws.s3()
    budget = Budget(int(target_mb * 1e6))
    # Parquet may not eat the budget the PDFs need.
    parquet_budget = Budget(int(target_mb * 1e6 * config.PARQUET_RESERVE_SHARE))

    parts = partitions if partitions is not None else plan(
        client=client, sources=sources, years=years, court=court)
    if not parts:
        raise SystemExit(
            "No parquet partitions found. Check the source buckets and --years "
            "(for the High Court, --court narrows to one code such as 29_3)."
        )
    print(f"parquet: {len(parts)} partition(s), transfer cap {target_mb:.0f} MB "
          f"(parquet reserve {config.PARQUET_RESERVE_SHARE:.0%})")
    if plan_only:
        describe(parts, client=client)
        return {"partitions": len(parts), "plan_only": True}

    if fs is None:
        from pyarrow.fs import S3FileSystem

        fs = S3FileSystem(anonymous=True, region=config.AWS_REGION)

    model_name = model_name or config.DENSE_MODEL
    if embedder is None:
        from fastembed import TextEmbedding

        embedder = TextEmbedding(model_name=model_name)

    conn = curate._open_db(config.SINGLE_DB, fresh=fresh)
    writer = curate._Writer(conn, embedder, model_name)
    started = time.perf_counter()

    # Per-partition row generators, drained round-robin so one year or court
    # cannot dominate the slice.
    per_partition_cap = max(rows_per_partition * 6, rows_per_partition)
    gens: dict[str, Iterator[dict]] = {}
    for part in parts:
        gens[part["label"]] = _iter_partition(fs, part, budget, parquet_budget,
                                               per_partition_cap)
    fetched = skipped = 0
    exhausted = False
    while gens and not exhausted:
        progressed = False
        for label in list(gens):
            if budget.left() <= 0:
                exhausted = True
                break
            try:
                batch = list(itertools.islice(gens[label], rows_per_partition))
            except _BudgetDone:
                exhausted = True
                break
            if not batch:
                gens.pop(label, None)
                continue
            progressed = True
            for row in batch:
                part = row["__partition"]
                # pdf_exists is a real boolean in the parquet; `value or ""`
                # would turn False into "" and lose the signal entirely.
                exists = row.get("exists")
                if exists is False or str(exists).strip().lower() in {"false", "0"}:
                    skipped += 1
                    continue
                pointer = row.get("pointer")
                bucket, key = resolve_pointer(pointer, part["bucket"])
                if not key:
                    skipped += 1
                    continue
                try:
                    blob = fetch_pdf(client, bucket, key, budget)
                except _BudgetDone:
                    exhausted = True
                    break
                if not blob:
                    skipped += 1
                    continue
                text = extract._pdf_to_text(blob)
                if len(text) < 200:
                    skipped += 1
                    continue
                record = _record(part, row, text)
                if writer.already_done(record["id"]):
                    continue
                writer.add(record)
                fetched += 1
                if fetched % 25 == 0:
                    print(f"    {fetched} judgments, {budget.spent / 1e6:.1f} MB transferred, "
                          f"{budget.left() / 1e6:.1f} MB left", file=sys.stderr)
        if not progressed:
            break
    writer.flush()

    notes = [
        "Parquet supplies identity, court, date and citation; the judgment text comes from the PDF each row points at.",
        "The parquet `raw_html` column is never transferred: reads are column-projected.",
        "Rows are taken round-robin across years and courts, capped per partition.",
        "The transfer budget counts projected parquet bytes plus each PDF actually downloaded.",
        "Regional-language PDFs are not derived from this path; use the tar path (`curate`) for those.",
    ]
    return curate._finalize(
        conn, writer, model_name=model_name, started=started, notes=notes,
        meta_extra={
            "mode": "parquet-first (metadata + pointed-at PDFs)",
            "transfer_cap_mb": f"{target_mb:g}",
            "transfer_mb": f"{budget.spent / 1e6:.2f}",
            "parquet_mb": f"{parquet_budget.spent / 1e6:.2f}",
            "partitions": str(len(parts)),
            "rows_per_partition": str(rows_per_partition),
            "skipped_rows": str(skipped),
            "sources": json.dumps(sorted({p["source"] for p in parts})),
        },
        extra={"target_mb": target_mb, "transferred_mb": round(budget.spent / 1e6, 2),
               "partitions": len(parts), "skipped_rows": skipped,
               "sources": sorted({p["source"] for p in parts})},
    )


if __name__ == "__main__":
    print(json.dumps(pull(), indent=2, default=str))
