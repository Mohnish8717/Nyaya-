"""AWS Open Data judgment buckets (CC-BY-4.0), unsigned access.

Both buckets (SC, 25 High Courts) use the same layout family:
    data/tar/year=YYYY/[english|regional]/*.tar          (PDFs inside)
    metadata/tar/year=YYYY/[...]/metadata.tar            (JSON per judgment)

We stream tars to out/raw/<source>/ keeping directory names, remembering each
completed object in a resume manifest, and refusing to overshoot budgets.
"""
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from pathlib import Path

import config
import util

AWS_REGION = "ap-south-1"
ARCHIVE_SUFFIXES = (".tar", ".tar.gz")
_YEAR = re.compile(r"^year=(\d{4})$")
_COURT = re.compile(r"^court=([^/]+)$")


def s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client(
        "s3", region_name=AWS_REGION,
        config=Config(signature_version=UNSIGNED, retries={"max_attempts": 5}),
    )


def list_objects(client, bucket: str, prefix: str) -> list[dict]:
    out: list[dict] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        out.extend(page.get("Contents", []))
    return out


def list_folders(client, bucket: str, prefix: str) -> list[str]:
    """One level of "directories" (delimiter listing) — one request per 1000
    folders instead of paging every object under the prefix."""
    out: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        out.extend(item["Prefix"] for item in page.get("CommonPrefixes", []))
    return out


def available_years(bucket: str) -> list[int]:
    """Year folders under data/tar/, read from the live bucket listing."""
    seen: set[int] = set()
    for prefix in list_folders(s3(), bucket, "data/tar/"):
        for part in prefix.strip("/").split("/"):
            m = _YEAR.match(part)
            if m:
                seen.add(int(m.group(1)))
    return sorted(seen)


def available_courts(bucket: str, year: int) -> list[str]:
    return [p.strip("/").split("=")[-1] for p in list_folders(s3(), bucket, f"data/tar/year={year}/")]


def year_report(bucket: str, year: int, court: str | None = None) -> dict:
    """Total bytes under one {year[, court]} slice: the honest unit the user
    needs before choosing what to download. Both the judgment tars and the
    metadata tars are counted, because both must be transferred."""
    suffixes = (f"data/tar/year={year}/", f"metadata/tar/year={year}/")
    if court:
        suffixes = tuple(prefix + f"court={court}/" for prefix in suffixes)
    archives = 0
    total = 0
    for prefix in suffixes:
        objs = [o for o in list_objects(s3(), bucket, prefix) if o["Key"].endswith(ARCHIVE_SUFFIXES)]
        archives += len(objs)
        total += sum(int(o["Size"]) for o in objs)
    return {"dataset": bucket, "year": year, "court": court, "archives": archives, "bytes": total}


def archives_for_slice(bucket: str, year: int, court: str | None) -> Iterator[dict]:
    prefix = f"data/tar/year={year}/"
    if court:
        prefix += f"court={court}/"
    for obj in list_objects(s3(), bucket, prefix):
        if obj["Key"].endswith(ARCHIVE_SUFFIXES):
            yield {"key": obj["Key"], "size": int(obj["Size"])}


def download_command(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Download selected archives from an open-data bucket.")
    ap.add_argument("--source", choices=("aws_sc", "aws_hc"), required=True)
    ap.add_argument("--years", help="comma list or start-end range, e.g. 1950,1951 or 2020-2024")
    ap.add_argument("--court", help="High Court code only, e.g. 29_3 (Karnataka)")
    ap.add_argument("--max-download-gb", type=float, default=config.DEFAULT_MAX_DOWNLOAD_GB,
                    help="hard byte budget for this run (default %.1f)" % config.DEFAULT_MAX_DOWNLOAD_GB)
    args = ap.parse_args(argv)

    bucket = config.AWS_BUCKETS[args.source]
    raw_dir = config.RAW / (args.source + "_data")
    years = parse_years(args.years)
    if args.court and args.source != "aws_hc":
        raise SystemExit("--court applies to the High Court bucket only.")
    if not years:
        raise SystemExit(
            "--years is required: nothing here syncs a bucket blind. "
            f"Run `python pipeline.py inspect --source {args.source}` for per-year sizes, "
            "then pass e.g. --years 2023 or --years 2020-2024."
        )
    state = util.JobState(raw_dir / ".state.sqlite3")
    budget = util.Budget.from_gb_flag(args.max_download_gb, args.source)

    # One pre-flight listing: sizes are known up front, so an oversized run
    # aborts before transferring a single byte of data.tar.
    plan: list[tuple[dict, dict]] = []
    for year in years:
        for arc in archives_for_slice(bucket, year, args.court):
            if state.done(arc["key"]):
                continue
            size = budget.check(arc["size"], f"{arc['key']}")
            plan.append((arc, {"bucket": bucket, "size": size}))
    if not plan:
        print("No new archives in this slice (all already in the resume manifest).")
        return
    print(f"Will stream {len(plan)} archives, budgeted {budget.spent / 1e9:.2f} GB")
    client = s3()
    for arc, _meta in plan:
        dest = raw_dir / arc["key"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        _stream_object(client, bucket, arc["key"], dest, arc["size"])
        state.mark(arc["key"], json.dumps({"size": arc["size"]}))
        print(f"done {arc['key']}")


def parse_years(raw: str | None) -> list[int]:
    if not raw:
        return []
    years: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if "-" in token:
            start, end = token.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        elif token:
            years.append(int(token))
    if not years or not (1950 <= min(years) <= 2030):
        raise SystemExit(f"--years must be legal calendar years, got {raw!r}")
    return years


def inspect_command(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Report archive sizes for a slice before downloading.")
    ap.add_argument("--source", choices=("aws_sc", "aws_hc"), required=True)
    ap.add_argument("--years", help="narrow the report; default lists every available year")
    ap.add_argument("--court")
    args = ap.parse_args(argv)
    bucket = config.AWS_BUCKETS[args.source]

    if not args.years:
        years = available_years(bucket)
        if not years:
            print(f"{bucket}: no year folders found (unexpected layout); pass --years explicitly.")
            return
        print(f"{bucket}: {len(years)} year folders ({years[0]}–{years[-1]}). "
              "Per-year sizes below; totals combine data+metadata tars.")
        rows = []
        for year in years:
            rows.append(year_report(bucket, year, args.court))
        total = sum(r["bytes"] for r in rows)
        for r in rows:
            print(f"  {r['year']:>4}  {r['bytes'] / 1e9:8.2f} GB  ({r['archives']} archives)")
        print(f"  total ~{total / 1e9:.1f} GB — narrow with --years and (HC) --court")
        return
    for year in parse_years(args.years):
        print(json.dumps(year_report(bucket, year, args.court), indent=2))


def _stream_object(client, bucket: str, key: str, dest: Path, expected: int) -> None:
    """Streamed GET: never holds an archive in memory, verifies size at close."""
    partial = dest.with_suffix(dest.suffix + ".part")
    got = 0
    with client.get_object(Bucket=bucket, Key=key)["Body"] as body, \
            partial.open("wb") as fh:
        for chunk in iter(lambda: body.read(1 << 22), b""):
            fh.write(chunk)
            got += len(chunk)
    if expected and got != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{key}: streamed {got} B, listing said {expected} B")
    partial.replace(dest)


if __name__ == "__main__":
    print(__doc__)
