#!/usr/bin/env python3
"""One command: OpenNyAI InJudgements -> ONE embeddings file.

This is the whole path in a single script. It touches nothing but
`opennyaiorg/InJudgements_dataset` — no AWS buckets, no Kaggle, no DDL — and it
writes exactly one artifact:

    legal_pipeline/out/corpus_embeddings.sqlite3

    python build_opennyai_embeddings.py              # download (~293 MB) + embed everything
    python build_opennyai_embeddings.py --preflight  # check the environment, change nothing
    python build_opennyai_embeddings.py --limit 300  # trial run: 300 judgments
    python build_opennyai_embeddings.py --resume     # continue where it stopped
    python build_opennyai_embeddings.py --verify     # re-check an existing file
    python build_opennyai_embeddings.py --via datasets   # stream via datasets.load_dataset
    python build_opennyai_embeddings.py --local ~/InJudgements_dataset

The dataset is gated: its *file list* is public, the parquet is not. Accept the
terms on the dataset page, then `huggingface-cli login` (or export HF_TOKEN).
The script says exactly that instead of failing with an HTTP traceback.

What lands in the file (one SQLite file, three tables you query + one index):

    cases     one row per judgment: title, court, court_level, jurisdiction,
              language, decision_date, source_url, dataset
    chunks    the passages that were embedded; every `header` starts with the
              tier, e.g. "[HIGH COURT — High Court of Karnataka | level: high
              | jurisdiction: IN-KA | language: kn]", so a vector cannot pose as
              a Supreme Court holding
    vectors   int8 unit vectors, 384-d, one per chunk (384 bytes each)
    chunk_fts BM25 over chunks.text (external content: the text is stored once)
    meta      model, dimension, counts, and the honest caveats

Honest caveats written into `meta` rather than hidden: this dataset publishes no
decision date (stored as "unknown", never guessed), district-court coverage is
Delhi and Bengaluru only, and it is English-language text.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402  (path set up above)

DB = config.SINGLE_DB
NEEDED_MODULES = (
    ("pyarrow", "pyarrow.parquet"),
    ("fastembed", "fastembed"),
    ("huggingface_hub", "huggingface_hub"),
    ("numpy", "numpy"),
)
# parquet (~300 MB) + the sqlite file (~450-700 MB at full size) + headroom.
NEEDED_BYTES = 2 * 1024 ** 3


# ---------------------------------------------------------------------------
# Preflight: everything that can fail, checked before anything is downloaded
# ---------------------------------------------------------------------------
def _ok(label: str, detail: str = "") -> None:
    print(f"  [ ok ] {label}{(' — ' + detail) if detail else ''}")


def _bad(label: str, detail: str = "") -> None:
    print(f"  [fail] {label}{(' — ' + detail) if detail else ''}")


def _warn(label: str, detail: str = "") -> None:
    print(f"  [warn] {label}{(' — ' + detail) if detail else ''}")


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 0


def preflight(via: str, local: str | None, strict: bool = True) -> bool:
    """Report every precondition. Returns True when the run can proceed."""
    print("preflight")
    good = True

    if sys.version_info >= (3, 9):
        _ok("python", f"{sys.version.split()[0]}")
    else:
        _bad("python", f"{sys.version.split()[0]} < 3.9")
        good = False

    missing: list[str] = []
    for pip_name, module in NEEDED_MODULES:
        try:
            __import__(module)
        except Exception as exc:  # noqa: BLE001 - any import error counts
            missing.append(f"{pip_name} ({type(exc).__name__})")
    if missing:
        _bad("dependencies", ", ".join(missing))
        print(f"         fix: {sys.executable} -m pip install -r {HERE / 'requirements.txt'}")
        good = False
    else:
        _ok("dependencies", "pyarrow, fastembed, huggingface_hub, numpy")

    if via == "datasets":
        try:
            import datasets  # noqa: F401

            _ok("datasets", "installed (streaming mode)")
        except Exception:  # noqa: BLE001
            _bad("datasets", "not installed; `pip install datasets` or drop --via datasets")
            good = False

    if local:
        folder = Path(local).expanduser()
        if folder.is_dir() and any(folder.rglob("*.parquet")):
            _ok("local data", f"{folder} ({len(list(folder.rglob('*.parquet')))} parquet)")
        else:
            _bad("local data", f"{folder} holds no .parquet")
            good = False
    elif via == "datasets":
        _ok("download", "streamed by datasets.load_dataset; no pre-download")
    else:
        existing = _existing_parquet()
        if existing:
            _ok("download", f"already on disk at {existing}; will reuse it")
        else:
            import dl_opennyai

            if dl_opennyai.has_token():
                _ok("download", f"token found; will fetch {config.HF_REPO} (~293 MB)")
            else:
                _bad("download", "no Hugging Face token")
                print("         This dataset is gated. Two steps, once per machine:")
                print(f"           1. accept the terms: https://huggingface.co/datasets/{config.HF_REPO}")
                print("           2. authenticate:    huggingface-cli login   (or export HF_TOKEN=hf_...)")
                good = False

    free = _free_bytes(config.OUT)
    if free >= NEEDED_BYTES:
        _ok("disk", f"{free / 1024 ** 3:.1f} GB free under {config.OUT}")
    else:
        _bad("disk", f"{free / 1024 ** 3:.1f} GB free, need ~{NEEDED_BYTES / 1024 ** 3:.0f} GB")
        good = False

    if _reachable("https://huggingface.co"):
        _ok("network", "huggingface.co reachable")
    else:
        _warn("network", "huggingface.co not reachable from here (proxy? offline?)")

    if strict and not good:
        print("\npreflight failed: fix the [fail] lines above, then re-run.")
    return good


def _existing_parquet() -> Path | None:
    import dl_opennyai

    for candidate in (dl_opennyai.RAW_DIR / "dl", dl_opennyai.RAW_DIR / "local"):
        if candidate.is_dir() and any(candidate.rglob("*.parquet")):
            return candidate
    return None


def _reachable(url: str, timeout: float = 8.0) -> bool:
    try:
        request = urllib.request.Request(url, method="HEAD",
                                         headers={"User-Agent": "legal-pipeline/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status < 500
    except (urllib.error.URLError, OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run(args) -> dict:
    import opennyai_source

    started = time.perf_counter()
    result = opennyai_source.build(
        local=args.local,
        download=not args.no_download,
        limit=args.limit,
        model_name=args.model,
        fresh=not args.resume,
        via=args.via,
    )
    print(f"\nbuilt in {time.perf_counter() - started:.0f}s")
    return result


# ---------------------------------------------------------------------------
# Verify the artifact we just wrote (or one from an earlier run)
# ---------------------------------------------------------------------------
def verify(path: Path, sample_query: str = "cheque dishonour") -> bool:
    print(f"\nverify {path}")
    if not path.exists():
        _bad("file", "missing")
        return False
    size = path.stat().st_size
    if size < 1024:
        _bad("file", f"{size} B is not a built corpus")
        return False
    ok = True
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity == "ok":
            _ok("integrity", "ok")
        else:
            _bad("integrity", str(integrity)[:200])
            ok = False

        counts = {table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                  for table in ("cases", "chunks", "vectors")}
        if counts["cases"] and counts["chunks"] and counts["vectors"]:
            _ok("rows", f"{counts['cases']:,} judgments, {counts['chunks']:,} passages, "
                        f"{counts['vectors']:,} vectors")
        else:
            _bad("rows", f"empty table in {counts}")
            ok = False

        if counts["vectors"] == counts["chunks"]:
            _ok("coverage", "every passage has a vector")
        else:
            holes = conn.execute(
                "SELECT count(*) FROM chunks ch LEFT JOIN vectors v"
                " ON v.chunk_id = ch.chunk_id WHERE v.chunk_id IS NULL"
            ).fetchone()[0]
            _bad("coverage", f"{holes} passage(s) without a vector; re-run with --resume")
            ok = False

        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        dim = meta.get("dim", "")
        if dim.isdigit() and int(dim) > 0:
            _ok("model", f"{meta.get('model', '?')} ({dim}-d, {meta.get('quantisation', '?')})")
        else:
            _bad("model", "meta has no dimension")
            ok = False

        levels = dict(conn.execute(
            "SELECT court_level, count(*) FROM cases GROUP BY court_level ORDER BY 2 DESC"
        ).fetchall())
        described = ", ".join(f"{k}={v:,}" for k, v in levels.items())
        if levels:
            _ok("tiers", described)
        else:
            _bad("tiers", "no court_level labels")
            ok = False

        languages = dict(conn.execute(
            "SELECT language, count(*) FROM cases GROUP BY language ORDER BY 2 DESC LIMIT 6"
        ).fetchall())
        _ok("languages", ", ".join(f"{k}={v:,}" for k, v in languages.items()))

        # BM25 must actually answer: a corpus whose index is empty retrieves nothing.
        try:
            hits = conn.execute(
                "SELECT count(*) FROM chunk_fts WHERE chunk_fts MATCH ?", (sample_query,)
            ).fetchone()[0]
        except sqlite3.OperationalError as exc:
            _bad("bm25", f"query failed: {exc}")
            hits, ok = 0, False
        if hits:
            _ok("bm25", f"{hits:,} passages match {sample_query!r}")
        else:
            _warn("bm25", f"no passage matches {sample_query!r} (small slice?)")

        # One join back from the index to real, labelled text.
        row = conn.execute(
            "SELECT c.court_level, c.court, substr(ch.text, 1, 70)"
            " FROM chunk_fts JOIN chunks ch ON ch.rowid = chunk_fts.rowid"
            " JOIN cases c ON c.case_id = ch.case_id"
            " WHERE chunk_fts MATCH ? LIMIT 1", (sample_query,)
        ).fetchone()
        if row:
            _ok("join", f"{row[0]} / {row[1]}: {row[2].strip()}…")
        elif hits:
            _bad("join", "FTS hits do not join back to cases")
            ok = False
    finally:
        conn.close()

    _ok("file", f"{path} — {size / 1e6:.1f} MB")
    print("\nquery it:")
    print(f"  sqlite3 {path}")
    print("  SELECT c.court_level, c.title, substr(ch.text,1,160)")
    print("  FROM chunk_fts JOIN chunks ch ON ch.rowid = chunk_fts.rowid")
    print("  JOIN cases c ON c.case_id = ch.case_id")
    print("  WHERE chunk_fts MATCH 'cheque dishonour' ORDER BY rank LIMIT 5;")
    return ok


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download opennyaiorg/InJudgements_dataset and turn it into one embeddings file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--preflight", action="store_true",
                        help="check python, deps, token, disk, network; build nothing")
    parser.add_argument("--verify", action="store_true",
                        help="verify an existing output file; build nothing")
    parser.add_argument("--limit", type=int, help="embed at most N judgments (trial run)")
    parser.add_argument("--model", help=f"encoder (default {config.DENSE_MODEL})")
    parser.add_argument("--via", choices=("parquet", "datasets"), default="parquet",
                        help="read the downloaded parquet (default) or stream through "
                             "datasets.load_dataset")
    parser.add_argument("--local", help="an already-downloaded InJudgements folder")
    parser.add_argument("--no-download", action="store_true",
                        help="never hit the network; fail if the data is not on disk")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run instead of rebuilding")
    parser.add_argument("--output", help=f"override the output path (default {DB})")
    parser.add_argument("--sample-query", default="cheque dishonour",
                        help="lexical query used by --verify")
    args = parser.parse_args(argv)

    print("OpenNyAI InJudgements -> embeddings")
    print(f"  source   {config.HF_REPO} (apache-2.0, ~293 MB parquet, 11,970 judgments)")
    print(f"  encoder  {args.model or config.DENSE_MODEL}")
    print(f"  output   {args.output or DB}\n")

    if args.output:
        target = Path(args.output).expanduser().resolve()
        config.SINGLE_DB = target
    else:
        target = DB

    if args.verify:
        return 0 if verify(target, args.sample_query) else 1

    ready = preflight(args.via, args.local)
    if args.preflight:
        return 0 if ready else 2
    if not ready:
        return 2
    if args.no_download and args.via != "datasets" and not _existing_parquet():
        print("\n--no-download was given but no local parquet exists.")
        return 2

    print()
    try:
        result = run(args)
    except SystemExit as exc:  # preflight-style exits inside the modules
        print(f"\n{exc}")
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted; the file is resumable: re-run with --resume")
        return 130
    except Exception as exc:  # noqa: BLE001 - one clear line beats a traceback
        print(f"\nbuild failed: {type(exc).__name__}: {exc}")
        print("If this is the model download, check the network; the encoder is fetched "
              "once (~120 MB) and cached.")
        return 1

    print("\nsummary")
    print(json.dumps({k: result.get(k) for k in
                      ("file", "cases", "chunks", "vectors", "text_bytes",
                       "per_level", "per_language", "seconds")}, indent=2, default=str))
    return 0 if verify(target, args.sample_query) else 1


if __name__ == "__main__":
    raise SystemExit(main())
