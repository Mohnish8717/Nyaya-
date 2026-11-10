#!/usr/bin/env python3
"""Download and index selected public judgment partitions from AWS Open Data.

This is deliberately partitioned and budgeted: it never syncs the full corpus
by default. Run it only on a machine with enough storage for the selected years.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sqlite3
import sys
import tarfile
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Iterable

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    import fitz
except ImportError as exc:
    raise SystemExit("Install ingestion dependencies first: python -m pip install -r requirements-ingest.txt") from exc

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime" / "legal-data"
OUTFILE = RUNTIME / "judgments.jsonl"
INGEST_DB = RUNTIME / "ingestion-state.sqlite3"
SC_BUCKET = "indian-supreme-court-judgments"
HC_BUCKET = "indian-high-court-judgments"
COURT_MAP_PATH = ROOT / "data" / "hc_court_jurisdictions.json"


def hc_jurisdiction(court_code: str | None) -> str:
    """Return the state/UT for a High Court code, or refuse.

    The previous table held two codes and silently defaulted every other court
    to jurisdiction "IN", which made a state's judgments look like national
    law. Unknown codes now stop the import instead of being mislabelled.
    """
    if not court_code:
        raise SystemExit("High Court ingestion needs --court-code.")
    if not COURT_MAP_PATH.exists():
        raise SystemExit(f"Missing court map: {COURT_MAP_PATH}")
    mapping = json.loads(COURT_MAP_PATH.read_text(encoding="utf-8"))
    entry = mapping.get(court_code)
    if not entry or not entry.get("jurisdiction"):
        raise SystemExit(
            f"Unknown High Court code {court_code!r}. Add it to data/hc_court_jurisdictions.json "
            "rather than importing it as national law."
        )
    return entry["jurisdiction"]


def s3_client():
    return boto3.client("s3", region_name="ap-south-1", config=Config(signature_version=UNSIGNED, retries={"max_attempts": 5}))


def list_objects(client, bucket: str, prefix: str) -> list[dict[str, Any]]:
    paginator = client.get_paginator("list_objects_v2")
    objects: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects.extend(page.get("Contents", []))
    return objects


def year_prefixes(dataset: str, year: str, court: str | None, bench: str | None) -> tuple[str, str, str, str]:
    if dataset == "sc":
        root = f"year={year}/"
        return SC_BUCKET, f"data/tar/{root}english/", f"metadata/tar/{root}", "Supreme Court of India"
    if not court:
        raise ValueError("High Court ingestion needs --court-code, for example 29_3 for Karnataka.")
    court = court.replace("~", "_")
    root = f"year={year}/court={court}/"
    if bench:
        root += f"bench={bench}/"
    return HC_BUCKET, f"data/tar/{root}", f"metadata/tar/{root}", f"High Court {court}" + (f" / {bench}" if bench else "")


def find_archives(client, bucket: str, data_prefix: str, metadata_prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    archive_suffixes = (".tar", ".tar.gz")
    data = [obj for obj in list_objects(client, bucket, data_prefix) if obj["Key"].endswith(archive_suffixes)]
    metadata = [obj for obj in list_objects(client, bucket, metadata_prefix) if obj["Key"].endswith(archive_suffixes)]
    return data, metadata


def check_budget(objects: Iterable[dict[str, Any]], max_bytes: int, dataset: str, year: str) -> int:
    total = sum(int(obj.get("Size", 0)) for obj in objects)
    if total > max_bytes:
        gb = total / (1024 ** 3)
        limit = max_bytes / (1024 ** 3)
        raise SystemExit(f"Refusing to download {dataset} {year}: archives require about {gb:.2f} GiB, above --max-download-gb={limit:.2f}. Select a narrower court/year or raise the budget explicitly.")
    return total


def load_metadata_archives(client, bucket: str, archives: list[dict[str, Any]], work: Path) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for obj in archives:
        path = work / (hashlib.sha1(obj["Key"].encode()).hexdigest() + ".tar")
        client.download_file(bucket, obj["Key"], str(path))
        with tarfile.open(path, mode="r:*") as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith(".json"):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                try:
                    value = json.loads(handle.read().decode("utf-8", errors="replace"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(value, dict):
                    continue
                by_name[Path(member.name).stem.lower()] = value
        path.unlink(missing_ok=True)
    return by_name


def text_from_pdf(blob: bytes) -> str:
    try:
        with fitz.open(stream=blob, filetype="pdf") as doc:
            return "\n".join(page.get_text("text") for page in doc).strip()
    except Exception:
        return ""


def choose(mapping: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def read_seen(db: sqlite3.Connection, source_key: str) -> bool:
    return db.execute("SELECT 1 FROM ingested_files WHERE source_key=?", (source_key,)).fetchone() is not None


def record_seen(db: sqlite3.Connection, source_key: str, case_id: str) -> None:
    db.execute("INSERT OR IGNORE INTO ingested_files(source_key,case_id) VALUES(?,?)", (source_key, case_id))


def process_data_archives(client, bucket: str, archives: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], dataset: str, court: str, court_code: str | None, year: str, work: Path, db: sqlite3.Connection, source_root: str) -> int:
    license_label = "CC-BY-4.0" if dataset == "sc" else "CC-BY-4.0"
    dataset_title = "Indian Supreme Court Judgments" if dataset == "sc" else "Indian High Court Judgments"
    docs_url = "https://github.com/vanga/indian-supreme-court-judgments" if dataset == "sc" else "https://github.com/vanga/indian-high-court-judgments"
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUTFILE.open("a", encoding="utf-8") as output:
        for archive_obj in archives:
            archive_key = archive_obj["Key"]
            local_archive = work / (hashlib.sha1(archive_key.encode()).hexdigest() + ".tar")
            client.download_file(bucket, archive_key, str(local_archive))
            with tarfile.open(local_archive, mode="r:*") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.lower().endswith(".pdf"):
                        continue
                    source_key = f"{archive_key}::{member.name}"
                    if read_seen(db, source_key):
                        continue
                    stream = archive.extractfile(member)
                    if stream is None:
                        continue
                    text = text_from_pdf(stream.read())
                    if len(text) < 250:
                        continue
                    basename = Path(member.name).stem.lower()
                    meta = metadata.get(basename, {})
                    decision_date = choose(meta, "decision_date", "date_of_decision", "judgment_date")
                    case_number = choose(meta, "cnr", "case_id", "case_number", default=basename)
                    stable = f"{dataset}:{hashlib.sha1((case_number + '|' + decision_date + '|' + archive_key).encode()).hexdigest()}"
                    title = choose(meta, "title", "case_title", "description", default=Path(member.name).name)
                    court_name = choose(meta, "court_name", default=court)
                    citation = choose(meta, "citation", "neutral_citation")
                    case_path = choose(meta, "pdf_link", "path", default=docs_url)
                    if case_path.startswith("s3://"):
                        case_path = "https://" + case_path[5:].replace("/", ".s3.amazonaws.com/", 1)
                    elif not case_path.startswith("https://"):
                        case_path = docs_url
                    item = {
                        "id": stable, "title": title, "court": court_name, "citation": citation,
                        "decision_date": decision_date, "dataset": dataset, "dataset_title": dataset_title,
                        "dataset_license": license_label, "source_url": case_path, "dataset_docs_url": docs_url,
                        "source_archive": f"s3://{bucket}/{archive_key}", "archive_member": member.name,
                        "jurisdiction": "IN" if dataset == "sc" else hc_jurisdiction(court_code),
                        "language": choose(meta, "language", "lang", default="unknown"), "text": text,
                        "retrieval_warning": "Judgment's later treatment, appeal history, and binding status have not been verified.",
                    }
                    output.write(json.dumps(item, ensure_ascii=False) + "\n")
                    record_seen(db, source_key, stable)
                    written += 1
                    if written % 250 == 0:
                        output.flush(); db.commit()
            local_archive.unlink(missing_ok=True)
            db.commit()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sc", "hc"), required=True)
    parser.add_argument("--years", nargs="+", required=True, help="Years to ingest; each is preflighted against the download budget.")
    parser.add_argument("--court-code", help="High Court code from the dataset, e.g. 29_3. Required for --dataset hc.")
    parser.add_argument("--bench", help="Optional High Court bench identifier to narrow a partition.")
    parser.add_argument("--max-download-gb", type=float, default=1.0, help="Maximum compressed archive bytes to download per selected year (default: 1 GiB).")
    args = parser.parse_args()
    if args.max_download_gb <= 0:
        parser.error("--max-download-gb must be positive")
    client = s3_client()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    INGEST_DB.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0
    with sqlite3.connect(INGEST_DB) as db:
        db.execute("CREATE TABLE IF NOT EXISTS ingested_files(source_key TEXT PRIMARY KEY, case_id TEXT NOT NULL)")
        for year in args.years:
            bucket, data_prefix, metadata_prefix, court_label = year_prefixes(args.dataset, year, args.court_code, args.bench)
            data_archives, metadata_archives = find_archives(client, bucket, data_prefix, metadata_prefix)
            if not data_archives:
                print(f"No judgment tar archives found for {args.dataset} {year} at {data_prefix}", file=sys.stderr)
                continue
            selected = data_archives + metadata_archives
            need = check_budget(selected, int(args.max_download_gb * 1024 ** 3), args.dataset, year)
            print(f"Ingesting {args.dataset} {year}: {len(data_archives)} judgment archives + {len(metadata_archives)} metadata archives, {need / 1024**3:.2f} GiB compressed")
            with tempfile.TemporaryDirectory(prefix="legal-case-ingest-") as temp:
                work = Path(temp)
                metadata = load_metadata_archives(client, bucket, metadata_archives, work)
                source_root = data_prefix
                total_written += process_data_archives(client, bucket, data_archives, metadata, args.dataset, court_label, args.court_code, year, work, db, source_root)
            db.commit()
    print(f"Imported {total_written:,} searchable judgments to {OUTFILE}. Run scripts/reindex_judgments.py before restarting the app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

