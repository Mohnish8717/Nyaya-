#!/usr/bin/env python3
"""Install judgment text from the public AWS Open Data corpora, with metadata.

Why this exists: the bundled subset had 1,810 records and *zero* decision
dates, citations, benches or CNRs, because it was produced without joining the
structured metadata the bucket publishes alongside the PDFs. This importer
joins them and refuses to write a record that silently loses those fields.

Both datasets are enormous (Supreme Court ~52 GB / ~35k judgments across
77 years; High Courts ~1.25 TiB / ~17.8M judgments across 25 courts). Nothing
here downloads "everything": every run states a download budget, checks free
disk before starting, and can be pointed at one year or one court/bench.

Examples
--------
    # Supreme Court, English, five recent years (~1.4 GB of archives)
    python scripts/install_judgments.py --dataset sc --years 2021 2022 2023 2024 2025 --max-download-gb 3

    # One High Court bench for one year, metadata joined
    python scripts/install_judgments.py --dataset hc --years 2024 --court 29_3 --bench karhcdharwad --max-download-gb 3

Output is appended to .runtime/legal-data/judgments.jsonl, which the app
indexes on startup. Already-ingested source keys are skipped, so a rerun after
an interruption resumes instead of duplicating.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    import boto3
    import fitz
    import pyarrow.parquet as pq
    from botocore import UNSIGNED
    from botocore.config import Config
except ImportError as exc:  # pragma: no cover - operator guidance
    raise SystemExit(
        "Install ingestion dependencies first: python -m pip install -r requirements-ingest.txt pyarrow"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime" / "legal-data"
OUTFILE = RUNTIME / "judgments.jsonl"
STATE_DB = RUNTIME / "ingestion-state.sqlite3"
COURT_MAP = ROOT / "data" / "hc_court_jurisdictions.json"

SC_BUCKET = "indian-supreme-court-judgments"
HC_BUCKET = "indian-high-court-judgments"
GIB = 1024 ** 3
HEADROOM = 2 * GIB
LANGUAGE_SUFFIXES = ("_EN", "_HIN", "_PUN", "_TAM", "_TEL", "_BEN", "_GUJ", "_KAN", "_MAL", "_MAR", "_ORI", "_URD")


def s3():
    return boto3.client("s3", region_name="ap-south-1", config=Config(signature_version=UNSIGNED, retries={"max_attempts": 3}))


def disk_guard(budget_bytes: int) -> None:
    free = shutil.disk_usage(ROOT).free
    if free < budget_bytes + HEADROOM:
        raise SystemExit(
            f"Refusing to start: {free / GIB:.1f} GiB free, need {budget_bytes / GIB:.1f} GiB budget "
            f"plus {HEADROOM / GIB:.0f} GiB headroom. Lower --max-download-gb or free space."
        )


def list_objects(client, bucket: str, prefix: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        out.extend(page.get("Contents", []))
        if not page.get("IsTruncated"):
            return out
        token = page.get("NextContinuationToken")


def state() -> sqlite3.Connection:
    STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(STATE_DB)
    db.execute("CREATE TABLE IF NOT EXISTS ingested(source_key TEXT PRIMARY KEY, record_id TEXT, at REAL)")
    return db


def seen(db: sqlite3.Connection, key: str) -> bool:
    return db.execute("SELECT 1 FROM ingested WHERE source_key=?", (key,)).fetchone() is not None


def mark(db: sqlite3.Connection, key: str, record_id: str) -> None:
    db.execute("INSERT OR IGNORE INTO ingested(source_key,record_id,at) VALUES(?,?,?)", (key, record_id, time.time()))


def normalize_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    text = str(value or "").strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]  # already ISO, possibly with a time component
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d %B %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def pdf_text(blob: bytes) -> str:
    try:
        with fitz.open(stream=blob, filetype="pdf") as doc:
            text = "\n".join(page.get_text("text") for page in doc)
        return " ".join(text.split())
    except Exception:
        return ""


def stem_of(name: str) -> str:
    base = Path(name).name
    if base.lower().endswith(".pdf"):
        base = base[:-4]
    for suffix in LANGUAGE_SUFFIXES:
        if base.upper().endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def _stem(value: Any) -> str:
    name = Path(str(value or "").strip()).name
    return name[:-4] if name.lower().endswith(".pdf") else name


def download_metadata(client, bucket: str, key: str, work: Path) -> dict[str, dict[str, Any]]:
    """Return {filename stem: metadata row} for a parquet metadata object.

    The two corpora key their metadata differently: the Supreme Court uses a
    `path` such as `2024_10_108_125`, while the High Courts only expose a
    `pdf_link` such as `court/cnrorders/<bench>/orders/<id>_2024-03-18.pdf`.
    Both are indexed by stem so the join works for either, and a row that
    matches nothing shows up as `without_metadata_row` rather than silently
    producing a blank record.
    """
    path = work / (hashlib.sha1(key.encode()).hexdigest() + ".parquet")
    client.download_file(bucket, key, str(path))
    table = pq.read_table(path)
    rows = table.to_pylist()
    path.unlink(missing_ok=True)
    by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        for candidate in (row.get("path"), _stem(row.get("pdf_link")), row.get("file_name"), row.get("filename")):
            stem = str(candidate or "").strip()
            if stem:
                by_path[stem] = row
    return by_path


def sc_record(meta: dict[str, Any], text: str, member: str, archive_key: str, year: str) -> dict[str, Any]:
    path = str(meta.get("path") or stem_of(member))
    return {
        "id": f"sc:{path}",
        "title": str(meta.get("title") or Path(member).stem)[:600],
        "court": "Supreme Court of India",
        "citation": str(meta.get("citation") or "")[:240],
        "case_id": str(meta.get("case_id") or "")[:120],
        "cnr": str(meta.get("cnr") or "")[:60],
        "bench": str(meta.get("judge") or "")[:300],
        "petitioner": str(meta.get("petitioner") or "")[:300],
        "respondent": str(meta.get("respondent") or "")[:300],
        "disposal": str(meta.get("disposal_nature") or "")[:80],
        "decision_date": normalize_date(meta.get("decision_date", "")),
        "language": "en",
        "jurisdiction": "IN",
        "dataset": "sc",
        "source_url": f"https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com/data/tar/year={year}/english/english.tar",
        "source_archive": archive_key,
        "archive_member": member,
        "retrieval_warning": "Similar facts do not guarantee a similar result; later treatment is not checked.",
        "text": text,
    }


def hc_record(meta: dict[str, Any], text: str, member: str, archive_key: str, year: str, court_code: str) -> dict[str, Any]:
    court_name = str(meta.get("court") or meta.get("court_name") or f"High Court {court_code}")
    jurisdiction = court_jurisdiction(court_code)
    cnr = str(meta.get("cnr") or "")
    decision = normalize_date(meta.get("decision_date", ""))
    stable = cnr or hashlib.sha1(f"{archive_key}::{member}".encode()).hexdigest()[:24]
    return {
        "id": f"hc:{court_code}:{stable}",
        "title": str(meta.get("title") or Path(member).stem)[:600],
        "court": court_name[:160],
        "citation": str(meta.get("citation") or "")[:240],
        "cnr": cnr[:60],
        "bench": str(meta.get("judge") or meta.get("bench") or "")[:300],
        "petitioner": str(meta.get("petitioner") or "")[:300],
        "respondent": str(meta.get("respondent") or "")[:300],
        "disposal": str(meta.get("disposal_nature") or "")[:80],
        "decision_date": decision,
        "language": str(meta.get("language") or "en")[:30],
        "jurisdiction": jurisdiction,
        "dataset": "hc",
        "source_url": f"https://indian-high-court-judgments.s3.ap-south-1.amazonaws.com/data/tar/year={year}/court={court_code}/",
        "source_archive": archive_key,
        "archive_member": member,
        "retrieval_warning": "High Court judgments bind only within that court's jurisdiction; treatment is not checked.",
        "text": text,
    }


def court_jurisdiction(court_code: str) -> str:
    if COURT_MAP.exists():
        mapping = json.loads(COURT_MAP.read_text())
        entry = mapping.get(court_code)
        if entry and entry.get("jurisdiction"):
            return entry["jurisdiction"]
    return "IN"


def iter_pdf_members(handle: tarfile.TarFile) -> Iterable[tarfile.TarInfo]:
    for member in handle:
        if member.isfile() and member.name.lower().endswith(".pdf"):
            yield member


def ingest_archive(
    client,
    bucket: str,
    key: str,
    metadata: dict[str, dict[str, Any]],
    db: sqlite3.Connection,
    out,
    builder,
    work: Path,
    **builder_kwargs,
) -> dict[str, int]:
    stats = {"files": 0, "written": 0, "no_text": 0, "no_metadata": 0}
    local = work / (hashlib.sha1(key.encode()).hexdigest() + ".tar")
    client.download_file(bucket, key, str(local))
    try:
        with tarfile.open(local, mode="r:*") as archive:
            for member in iter_pdf_members(archive):
                stats["files"] += 1
                source_key = f"{key}::{member.name}"
                if seen(db, source_key):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                text = pdf_text(handle.read())
                if len(text) < 250:
                    stats["no_text"] += 1
                    continue
                meta = metadata.get(stem_of(member.name), {})
                if not meta:
                    stats["no_metadata"] += 1
                record = builder(meta, text, member.name, key, **builder_kwargs)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                mark(db, source_key, record["id"])
                stats["written"] += 1
                if stats["written"] % 200 == 0:
                    out.flush()
                    print(f"    {stats['written']} written...", flush=True)
    finally:
        local.unlink(missing_ok=True)
    db.commit()
    return stats


def run_sc(args, client, db, out, work: Path) -> dict[str, int]:
    totals = {"files": 0, "written": 0, "no_text": 0, "no_metadata": 0}
    for year in args.years:
        meta_key = f"metadata/parquet/year={year}/metadata.parquet"
        print(f"[sc {year}] metadata {meta_key}", flush=True)
        metadata = download_metadata(client, SC_BUCKET, meta_key, work)
        archives = [
            obj for obj in list_objects(client, SC_BUCKET, f"data/tar/year={year}/english/")
            if obj["Key"].endswith(".tar")
        ]
        for obj in archives:
            print(f"[sc {year}] {obj['Key']} ({obj['Size'] / 1e6:.0f} MB), metadata rows {len(metadata)}", flush=True)
            stats = ingest_archive(
                client, SC_BUCKET, obj["Key"], metadata, db, out, sc_record, work, year=str(year)
            )
            print(f"[sc {year}] {stats}", flush=True)
            for key in totals:
                totals[key] += stats[key]
    return totals


def run_hc(args, client, db, out, work: Path) -> dict[str, int]:
    totals = {"files": 0, "written": 0, "no_text": 0, "no_metadata": 0}
    if not args.court:
        raise SystemExit("High Court ingestion needs --court, for example 29_3 (Karnataka).")
    budget = int(args.max_download_gb * GIB)
    spent = 0
    for year in args.years:
        prefix = f"data/tar/year={year}/court={args.court}/"
        if args.bench:
            prefix += f"bench={args.bench}/"
        objects = list_objects(client, HC_BUCKET, prefix)
        archives = [obj for obj in objects if obj["Key"].endswith(".tar")]
        archives.sort(key=lambda obj: obj["Size"])
        if args.parts:
            archives = archives[: args.parts]
        for obj in archives:
            if spent + obj["Size"] > budget:
                print(f"  budget reached before {obj['Key']}; stopping", flush=True)
                return totals
            spent += obj["Size"]
            bench = next((part.split("=", 1)[1] for part in obj["Key"].split("/") if part.startswith("bench=")), "")
            meta_key = f"metadata/parquet/year={year}/court={args.court}/bench={bench}/metadata.parquet"
            try:
                metadata = download_metadata(client, HC_BUCKET, meta_key, work)
            except Exception as exc:
                print(f"  no metadata parquet at {meta_key} ({type(exc).__name__}); records will be flagged", flush=True)
                metadata = {}
            print(f"[hc {year}] {obj['Key']} ({obj['Size'] / 1e6:.0f} MB), metadata rows {len(metadata)}", flush=True)
            stats = ingest_archive(
                client, HC_BUCKET, obj["Key"], metadata, db, out, hc_record, work,
                year=str(year), court_code=args.court,
            )
            print(f"[hc {year}] {stats}", flush=True)
            for key in totals:
                totals[key] += stats[key]
    return totals


def validate(db: sqlite3.Connection, before: set[str]) -> dict[str, Any]:
    """Read back what was just written and report metadata completeness."""
    if not OUTFILE.exists():
        return {"records": 0}
    written = []
    with OUTFILE.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("id") not in before:
                written.append(record)
    total = len(written)
    missing_date = sum(1 for r in written if not r.get("decision_date"))
    missing_citation = sum(1 for r in written if not r.get("citation"))
    missing_cnr = sum(1 for r in written if not r.get("cnr"))
    return {
        "records": total,
        "missing_decision_date": missing_date,
        "missing_citation": missing_citation,
        "missing_cnr": missing_cnr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Install budgeted judgment data with metadata.")
    parser.add_argument("--dataset", choices=["sc", "hc"], required=True)
    parser.add_argument("--years", nargs="+", required=True)
    parser.add_argument("--court", help="High Court court code, e.g. 29_3")
    parser.add_argument("--bench", help="High Court bench directory, e.g. karhcdharwad")
    parser.add_argument("--parts", type=int, default=0, help="Limit High Court archives (smallest first)")
    parser.add_argument("--max-download-gb", type=float, default=2.0)
    parser.add_argument("--allow-missing-meta", type=float, default=0.05,
                        help="Fail if more than this fraction of new records lack a decision date")
    args = parser.parse_args()

    budget = int(args.max_download_gb * GIB)
    disk_guard(budget)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    work = RUNTIME / "tmp"
    work.mkdir(exist_ok=True)
    client = s3()
    db = state()

    before = set()
    if OUTFILE.exists():
        with OUTFILE.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    before.add(json.loads(line).get("id"))
                except json.JSONDecodeError:
                    continue

    started = time.time()
    with OUTFILE.open("a", encoding="utf-8") as out:
        totals = run_sc(args, client, db, out, work) if args.dataset == "sc" else run_hc(args, client, db, out, work)

    report = validate(db, before)
    report.update({
        "dataset": args.dataset,
        "years": args.years,
        "files_seen": totals["files"],
        "written": totals["written"],
        "skipped_no_text": totals["no_text"],
        "without_metadata_row": totals["no_metadata"],
        "seconds": round(time.time() - started, 1),
        "outfile": str(OUTFILE.relative_to(ROOT)),
    })
    print("\n" + json.dumps(report, indent=2))

    if report["records"] and report["missing_decision_date"] / report["records"] > args.allow_missing_meta:
        print("\nFAILED: too many new records have no decision date. Metadata join is broken.", file=sys.stderr)
        return 2
    print("\nNow reindex: python scripts/reindex_judgments.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
