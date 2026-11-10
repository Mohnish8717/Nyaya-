"""Command-line entry for the standalone judicial-data pipeline.

One contract, one order, every stage resumable:

  sources   - datasets, sizes, licences (no network)
  inspect   - per-partition byte previews before any download
  download  - budgeted, resumable fetches for each supported source
  extract   - every text source -> out/judgments.jsonl (+ sources_report.json)
  chunk     - judgments.jsonl -> out/chunks.sqlite3 (head + capped body chunks)
  embed     - chunks -> out/vectors/*.npy + manifest.sqlite3 (int8)
  fts       - judgments.jsonl -> out/cases_index.sqlite3 (app-compatible FTS5)
  ddl       - metadata-only source -> out/ddl_cases.sqlite3 (never embedded)
  export    - vectors/FTS -> out/app/{dense_cases,cases}.sqlite3 (drop-in)
  opennyai  - InJudgements parquet (has the text) -> the same single embeddings file
  parquet   - structured metadata + only the PDFs it points at -> the same single file
  curate    - capped, level-labelled, multilingual slice -> ONE out/corpus_embeddings.sqlite3
  status    - what exists, what is current, what is missing
  all       - extract -> chunk -> embed -> fts -> export

Nothing here downloads by itself; every stage refuses to run on missing input
rather than producing an empty artifact that looks finished.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chunk as chunk_mod
import config
import dl_aws
import dl_ddl
import dl_kaggle
import dl_opennyai
import export as export_mod
import extract
import fts_index
import util


def cmd_sources(_args) -> None:
    for key, meta in config.SOURCES.items():
        print(f"[{key}]")
        for field in ("label", "level", "size", "license", "location"):
            print(f"  {field:>9}: {meta.get(field, '')}")


def cmd_inspect(args) -> None:
    chosen = args.source
    if chosen in ("aws_sc", "aws_hc"):
        argv = ["--source", chosen]
        if args.years:
            argv += ["--years", args.years]
        dl_aws.inspect_command(argv)
        return
    if chosen == "ddl_meta":
        links = dl_ddl.read_links_file() or dl_ddl.gather_links("")
        if not links:
            print("ddl_meta: no links discovered; see README for the ddl_links.txt flow.")
            return
        total = 0
        for url in links[: args.limit]:
            try:
                size = dl_ddl.content_length(url)
            except Exception as exc:  # noqa: BLE001 - report, never abort the report
                print(f"  {url} -> HEAD failed ({exc})")
                continue
            total += size
            print(f"  {url.rsplit('/', 1)[-1]}  {size / 1e9:.2f} GB")
        print(f"  total {total / 1e9:.2f} GB across enumerated files")
        return
    print(f"{chosen}: no remote size probe; run `download` with --local <path> after fetching")
    for field in ("label", "size", "location", "cadence"):
        print(f"  {field:>8}: {config.SOURCES[chosen].get(field, '')}")


def cmd_download(args) -> None:
    if args.source in ("aws_sc", "aws_hc"):
        argv = ["--source", args.source]
        if args.years:
            argv += ["--years", args.years]
        if args.court:
            argv += ["--court", args.court]
        if args.max_download_gb is not None:
            argv += ["--max-download-gb", str(args.max_download_gb)]
        dl_aws.download_command(argv)
        return
    # Kaggle and OpenNyAI are single-dataset pulls with their own clients; only
    # the URL-list DDL pull takes a byte budget flag.
    argv: list[str] = []
    if args.source == "ddl_meta":
        argv += ["--max-download-gb", str(args.max_download_gb or config.DEFAULT_MAX_DOWNLOAD_GB)]
    if args.local:
        argv += ["--local", args.local]
    {
        "kaggle_sc": dl_kaggle.download_command,
        "opennyai_dc": dl_opennyai.download_command,
        "ddl_meta": dl_ddl.download_command,
    }[args.source](argv)


def cmd_text(args) -> None:
    argv: list[str] = []
    for source in (getattr(args, "source", None) or []):
        argv += ["--source", source]
    extract.extract(argv)


def cmd_chunk(args) -> None:
    chunk_mod.build(limit=getattr(args, "limit", None))


def cmd_embed(args) -> None:
    import embed as embed_mod

    embed_mod.build(limit=getattr(args, "limit", None))


def cmd_fts(args) -> None:
    fts_index.build(limit=getattr(args, "limit", None))


def cmd_curate(args) -> None:
    import curate

    result = curate.build(
        target_mb=args.target_mb,
        sources=args.source,
        model_name=args.model,
        per_tar_cap=args.per_tar_cap,
        fresh=not args.resume,
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_opennyai(args) -> None:
    import opennyai_source

    result = opennyai_source.build(
        local=args.local,
        download=not args.no_download,
        limit=args.limit,
        model_name=args.model,
        fresh=not args.resume,
        via=getattr(args, "via", "parquet"),
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_parquet(args) -> None:
    import parquet_source

    years = dl_aws.parse_years(args.years) if args.years else None
    result = parquet_source.pull(
        target_mb=args.target_mb,
        sources=args.source,
        years=years,
        court=args.court,
        model_name=args.model,
        rows_per_partition=args.rows_per_partition,
        fresh=not args.resume,
        plan_only=args.plan_only,
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_ddl(_args) -> None:
    import ddl_store

    print(json.dumps(ddl_store.load_ddl_cases(), indent=2, default=str))


def cmd_export(_args) -> None:
    print(json.dumps(export_mod.build(), indent=2, default=str))


def cmd_status(_args) -> None:
    print("pipeline status — artifacts under out/:")
    rows = [
        ("judgments.jsonl", config.JUDGMENTS_JSONL, "rows"),
        ("chunks.sqlite3", config.CHUNKS_DB, "chunk rows"),
        ("vectors/ (shards)", config.VECTORS_DIR, "shards"),
        ("vectors/manifest.sqlite3", config.VECTORS_MANIFEST, "vectors"),
        ("cases_index.sqlite3", config.CASES_INDEX_DB, "cases"),
        ("ddl_cases.sqlite3", config.DDL_DB, "MB"),
        ("app/dense_cases.sqlite3", config.EXPORT_DENSE_DB, "vectors"),
        ("app/cases.sqlite3", config.EXPORT_CASES_DB, "cases"),
        ("corpus_embeddings.sqlite3", config.SINGLE_DB, "corpus"),
        ("sources_report.json", config.REPORT_PATH, ""),
    ]
    for label, path, unit in rows:
        if not path.exists():
            print(f"  {label:<26} missing")
            continue
        detail = ""
        try:
            if unit == "rows":
                detail = f"{sum(1 for _ in util.jsonl_iter(path))} rows"
            elif unit == "shards":
                count = len(list(path.glob("shard_*.npy")))
                detail = f"{count} shard{'s' if count != 1 else ''}"
            elif unit == "MB":
                size = path.stat().st_size
                detail = f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{size / 1e3:.0f} KB"
            elif unit:
                import sqlite3

                with sqlite3.connect(path, timeout=5) as db:
                    if unit == "chunk rows":
                        detail = f"{db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]} chunks"
                    elif unit == "vectors":
                        detail = f"{db.execute('SELECT COUNT(*) FROM vectors').fetchone()[0]} vectors" \
                            if _has_table(db, "vectors") else \
                            f"{db.execute('SELECT COUNT(*) FROM vector_map').fetchone()[0]} vectors"
                    elif unit == "cases":
                        detail = f"{db.execute('SELECT COUNT(*) FROM case_fts').fetchone()[0]} cases"
                    elif unit == "corpus":
                        cases = db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
                        vectors = db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
                        detail = f"{cases} cases, {vectors} vectors"
        except Exception as exc:  # noqa: BLE001 - status must never raise
            detail = f"unreadable ({exc})"
        print(f"  {label:<26} yes ({detail})" if detail else f"  {label:<26} yes")


def _has_table(db, name: str) -> bool:
    return bool(db.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?", (name,)
    ).fetchone())


def cmd_all(args) -> None:
    for step in (cmd_text, cmd_chunk, cmd_embed, cmd_fts, cmd_export):
        step(args)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sources", help="list datasets, sizes, licences")

    p = sub.add_parser("inspect", help="field/size/range report before downloading")
    p.add_argument("--source", default="aws_hc", choices=list(config.SOURCES))
    p.add_argument("--years")
    p.add_argument("--limit", type=int, default=8)

    p = sub.add_parser("download", help="budgeted, resumable fetch")
    p.add_argument("--source", required=True, choices=list(config.SOURCES))
    p.add_argument("--years")
    p.add_argument("--court")
    p.add_argument("--max-download-gb", type=float)
    p.add_argument("--local")

    for name in ("extract", "chunk", "embed", "fts", "export", "ddl", "status", "all"):
        p = sub.add_parser(name, help=f"{name} stage")
        if name in ("extract", "all"):
            p.add_argument("--source", action="append",
                           choices=("aws_sc", "aws_hc", "kaggle_sc", "opennyai_dc"))
        if name in ("chunk", "embed", "fts", "all"):
            p.add_argument("--limit", type=int, help="process at most N judgments/chunks (trial run)")

    p = sub.add_parser("opennyai", help="InJudgements parquet text -> one embeddings file")
    p.add_argument("--local", help="existing InJudgements folder (skips the download)")
    p.add_argument("--via", choices=("parquet", "datasets"), default="parquet",
                   help="read the downloaded parquet (default) or stream through "
                        "datasets.load_dataset")
    p.add_argument("--no-download", action="store_true", help="fail instead of downloading")
    p.add_argument("--limit", type=int, help="stop after N judgments (trial run)")
    p.add_argument("--model", help=f"encoder (default {config.DENSE_MODEL})")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted run instead of rebuilding the file")

    p = sub.add_parser("parquet", help="structured metadata + pointed-at PDFs, one file")
    p.add_argument("--target-mb", type=float, default=config.CORPUS_TARGET_MB,
                   help="hard cap on everything transferred (default %g MB)" % config.CORPUS_TARGET_MB)
    p.add_argument("--source", action="append", choices=("aws_sc", "aws_hc"),
                   help="default: both; district courts are not part of this path")
    p.add_argument("--years", help="comma list or range, e.g. 2019-2023")
    p.add_argument("--court", help="High Court code only, e.g. 29_3")
    p.add_argument("--rows-per-partition", type=int, default=config.PARQUET_ROWS_PER_PARTITION,
                   help="rows read per parquet partition per round (default %d)"
                        % config.PARQUET_ROWS_PER_PARTITION)
    p.add_argument("--model", help=f"encoder (default {config.DENSE_MODEL})")
    p.add_argument("--plan-only", action="store_true",
                   help="list the partitions and their sizes, transfer nothing")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted run instead of rebuilding the file")

    p = sub.add_parser("curate", help="capped multilingual slice -> one embeddings file")
    p.add_argument("--target-mb", type=float, default=config.CORPUS_TARGET_MB,
                   help="size of the single output file (default %g MB)" % config.CORPUS_TARGET_MB)
    p.add_argument("--source", action="append",
                   choices=("aws_sc", "aws_hc", "kaggle_sc", "opennyai_dc"))
    p.add_argument("--model", help=f"encoder (must match the app; default {config.DENSE_MODEL})")
    p.add_argument("--per-tar-cap", type=int,
                   help="judgments per archive (default %d; 0 = no cap)" % config.CORPUS_PER_TAR_CAP)
    p.add_argument("--resume", action="store_true",
                   help="keep the existing file and continue, instead of rebuilding it")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    handler = {
        "sources": cmd_sources, "inspect": cmd_inspect, "download": cmd_download,
        "extract": cmd_text, "chunk": cmd_chunk, "embed": cmd_embed, "fts": cmd_fts,
        "export": cmd_export, "curate": cmd_curate, "parquet": cmd_parquet,
        "opennyai": cmd_opennyai, "ddl": cmd_ddl, "status": cmd_status,
        "all": cmd_all,
    }[args.cmd]
    handler(args)


if __name__ == "__main__":
    main()
