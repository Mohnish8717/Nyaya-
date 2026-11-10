"""Unified extract + normalize.

Every text-bearing raw artifact becomes one out/judgments.jsonl row in the
shared schema (config.REQUIRED + OPTIONAL). Sources without judgment text
(DDL) are handled separately by ddl_store.load_ddl_cases() into
ddl_cases.sqlite3 — never into embedding inputs.

Guarantees:
- full-text + archive-name dedupe against everything ever accepted
- every dropped record is counted with a reason
- resume: each (source, record-id) is processed once, kept in a manifest
- field detection is measured from real rows and reported, never hardcoded
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tarfile
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import config
import util
import dl_kaggle
import dl_opennyai

_TITLE_HINT = re.compile(r"title|case[_ ]?name|petitio|appellan|respondent|subject|caption", re.I)
_DATE_HINT = re.compile(r"date|year|dt$|day$", re.I)
_URL_HINT = re.compile(r"url|link|href|indiankanoon|source", re.I)
_COURT_HINT = re.compile(r"court|bench|tribunal|forum", re.I)
_CITE_HINT = re.compile(r"cit|citation|cnr|cin|ref", re.I)


def _norm_ws(text: Any) -> str:
    return " ".join(str(text or "").split())


# ---------------------------------------------------------------------------
# AWS tar/PDF path (aws_sc, aws_hc)
# ---------------------------------------------------------------------------
_LANGS = {
    "hindi": "hi", "hind": "hi", "tamil": "ta", "telugu": "te", "kannada": "kn",
    "malayalam": "ml", "marathi": "mr", "bengali": "bn", "gujarati": "gu",
    "punjabi": "pa", "odia": "or", "urdu": "ur", "assamese": "as",
}


def _language_for_member(member: str) -> str:
    parts = Path(member).parts
    low = member.lower()
    for token, code in _LANGS.items():
        if token in low:
            return code
    if any(p.lower() == "english" for p in parts):
        return "en"
    return "unknown"


def _pdf_to_text(raw: bytes) -> str:
    import fitz

    if not raw:
        return ""
    try:
        with fitz.open(stream=raw, filetype="pdf") as doc:
            text = "\n".join(page.get_text("text") or "" for page in doc)
        return "\n".join(_norm_ws(line) for line in text.splitlines() if _norm_ws(line))
    except Exception as exc:
        print(f"  pdf parse failed ({exc}); member skipped", file=sys.stderr)
        return ""


def _tar_year(tar_path: str) -> str:
    m = re.search(r"year=(\d{4})", tar_path)
    return m.group(1) if m else ""


def _hc_jurisdiction(source: str, meta: dict, tar_path: str) -> str:
    """Jurisdiction code: meta state, else the checked HC court map when one is
    present next to the pipeline, else the conservative national default.
    Refusing to invent mappings is deliberate (mirrors the app's rule)."""
    state = _norm_ws(meta.get("state_code") or meta.get("state") or "")
    if state and len(state) >= 4:
        return state[:32]
    m = re.search(r"court=([\w~]+)", tar_path)
    if m:
        for candidate in (config.ROOT / "hc_court_jurisdictions.json",
                          config.ROOT.parent / "data" / "hc_court_jurisdictions.json"):
            if candidate.exists():
                try:
                    entry = json.loads(candidate.read_text(encoding="utf-8")).get(m.group(1).replace("~", "_"))
                except (json.JSONDecodeError, OSError):
                    entry = None
                if entry and entry.get("jurisdiction"):
                    return str(entry["jurisdiction"])[:32]
    return "IN"


def _aws_record(source: str, stem: str, member: str, text: str, meta: dict, tar_path: str = "") -> dict:
    title = _norm_ws(meta.get("title") or meta.get("case_title") or "")[:300] or stem.replace("_", " ")[:220]
    court = _norm_ws(meta.get("court") or meta.get("court_name") or
                     ("Supreme Court of India" if source == "aws_sc" else "High Court"))[:160]
    decision = _norm_ws(meta.get("judgment_date") or meta.get("decision_date") or meta.get("date") or "")[:32]
    year = _norm_ws(meta.get("year") or "")[:8] or _tar_year(tar_path)
    if not decision and year:
        decision = f"{year}-01-01"
    if not decision:
        decision = "unknown"
    source_url = _norm_ws(meta.get("source_url") or meta.get("url") or "")[:400]
    if not source_url:
        # REQUIRED field; fall back to the dataset the record came from so the
        # row keeps provenance instead of being dropped for an empty url.
        source_url = config.SOURCE_URL_FALLBACK.get(source, source)
    return {
        "id": f"{source}:{stem[:160]}",
        "title": title,
        "court": court,
        "text": text,
        "source_url": source_url,
        "decision_date": decision,
        "dataset": source,
        "citation": _norm_ws(meta.get("citation") or meta.get("case_number") or meta.get("cnr") or "")[:120],
        "language": _norm_ws(meta.get("language") or "") or _language_for_member(member),
        "jurisdiction": (_norm_ws(meta.get("jurisdiction") or "") or
                         ("IN" if source == "aws_sc" else _hc_jurisdiction(source, meta, tar_path)))[:32],
        "archive_member": member[:240],
    }


def _extract_aws(source: str, state: util.JobState, stats: dict) -> Iterator[dict]:
    raw_root = config.RAW / (source + "_data")
    if not raw_root.is_dir():
        print(f"  {source} raw folder missing; run download first", file=sys.stderr)
        return
    meta_map: dict[str, dict] = {}
    for meta_tar in sorted(raw_root.rglob("*.tar")):
        if "metadata" not in meta_tar.name.lower():
            continue
        try:
            with tarfile.open(meta_tar) as tf:
                for member in tf:
                    f = tf.extractfile(member)
                    if not f or Path(member.name).suffix != ".json":
                        continue
                    try:
                        payload = json.loads(f.read().decode("utf-8", "replace"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if isinstance(payload, dict):
                        meta_map[Path(member.name).stem] = payload
        except (tarfile.TarError, OSError) as exc:
            print(f"  metadata tar skipped ({exc})", file=sys.stderr)
    stats["metadata_join"] = len(meta_map)

    for data_tar in sorted(raw_root.rglob("*.tar")):
        if "metadata" in data_tar.name.lower():
            continue
        try:
            tf = tarfile.open(data_tar)
        except (tarfile.TarError, OSError) as exc:
            print(f"  data tar skipped ({exc})", file=sys.stderr)
            continue
        with tf:
            for member in tf:
                name = Path(member.name).name
                if not name.lower().endswith(".pdf"):
                    continue
                f = tf.extractfile(member)
                if not f:
                    continue
                stem = Path(name).stem
                key = f"aws:{source}:{data_tar.name}:{stem}"
                if state.done(key):
                    continue
                text = _pdf_to_text(f.read())
                if not text or len(text) < 200:
                    stats["empty_text"] += 1
                    state.mark(key)  # also skip empties on re-run
                    continue
                state.mark(key)
                # _accept_record() owns the kept/rejected counters; counting
                # here as well double-reported every accepted judgment.
                yield _aws_record(source, stem, member.name, text, meta_map.get(stem, {}), str(data_tar))


# ---------------------------------------------------------------------------
# Tabular path (kaggle_sc, opennyai_dc): columns detected from real rows
# ---------------------------------------------------------------------------
def _scan_columns(sample: list[dict]) -> dict[str, str]:
    """Identify the judgment-text column and its companions from real rows.

    The longest column is the judgment; the companions are the *shortest*
    matching candidates (a 40-char "court" beats a 10 KB "court_observations"),
    which is why matches are ranked by total length rather than by name.
    """
    if not sample:
        return {}
    cols = [c for c in sample[0] if not c.startswith("__")]
    lengths = {c: sum(len(str(r.get(c) or "")) for r in sample) for c in cols}
    if not lengths:
        return {}
    text_col = max(lengths, key=lambda c: lengths[c])

    def pick(rx: re.Pattern, avoid: set[str] = frozenset()) -> str:
        candidates = [c for c in cols if c != text_col and c not in avoid and rx.search(c)]
        if not candidates:
            return ""
        return min(candidates, key=lambda c: lengths[c])

    court = pick(_COURT_HINT)
    if not court:
        court = pick(re.compile(r"supreme|high court|district|adala|bench", re.I))
    return {
        "text": text_col,
        "title": pick(_TITLE_HINT, avoid={court}),
        "date": pick(_DATE_HINT, avoid={court}),
        "url": pick(_URL_HINT),
        "court": court,
        "citation": pick(_CITE_HINT, avoid={court}),
    }


def _rows_from_folder(folder: Path) -> Iterator[dict]:
    files = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in (".csv", ".parquet")]
    if not files and any(p.suffix.lower() == ".zip" for p in folder.rglob("*.zip")):
        raise SystemExit(f"Nested zips under {folder}: unzip them first (pip zip target not auto-pulled).")
    if not files:
        raise SystemExit(f"No csv/parquet files under {folder}")
    for path in files:
        if path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq

            with pq.ParquetFile(str(path)) as pqf:
                for batch in pqf.iter_batches(batch_size=512):
                    for row in batch.to_pylist():
                        yield {"__file": path.name, **row}
        else:
            with path.open(newline="", encoding="utf-8", errors="replace") as fh:
                for row in csv.DictReader(fh):
                    yield {"__file": path.name, **row}


def extract_tabular(source: str, folder: Path, state: util.JobState, out_fh, seen: util.Dedupe, stats: dict) -> None:
    rows = _rows_from_folder(folder)
    sample = list(islice(rows, 50))
    cols = _scan_columns(sample)
    if not cols.get("text"):
        sys.exit(
            f"{source}: no text column could be identified. Observed columns: "
            f"{sorted(sample[0].keys()) if sample else 'none'}"
        )
    stats["columns_detected"] = cols
    for row in sample:
        row = {**row}
        row.setdefault("__file", "")
        _accept_tabular_row(source, cols, row, state, out_fh, seen, stats)
    for row in rows:
        _accept_tabular_row(source, cols, row, state, out_fh, seen, stats)


def _accept_tabular_row(source: str, cols: dict, row: dict, state: util.JobState,
                        out_fh, seen: util.Dedupe, stats: dict) -> None:
    text = _norm_ws(row.get(cols["text"]) or "")
    if not text or len(text) < 200:
        stats["empty_text"] += 1
        return
    ident = _norm_ws(row.get("id") or row.get("cnr") or row.get("ddl_case_id") or "")[:60] \
        or util.sha1_text(text)[:20]
    key = f"tab:{source}:{ident}"
    if state.done(key):
        return
    # archive_member is the dedupe identity. Pointing it at the source file
    # would mark every row after the first in that file as a duplicate of it;
    # the ident is what makes each row unique.
    member = f"{_norm_ws(row.get('__file'))}#{ident}"[:240]
    record = {
        "id": f"{source}:{ident}",
        "title": _norm_ws(row.get(cols.get("title")) or "")[:300] or f"{source} judgment {ident}",
        "court": _norm_ws(row.get(cols.get("court")) or "")[:160] or source,
        "text": text[:400_000],
        "source_url": _norm_ws(row.get(cols.get("url")) or "")[:400]
                      or config.SOURCE_URL_FALLBACK.get(source, source),
        "decision_date": _norm_ws(row.get(cols.get("date")) or "")[:32] or "unknown",
        "dataset": source,
        "citation": _norm_ws(row.get(cols.get("citation")) or "")[:120],
        "language": util.script_language(text),
        "jurisdiction": util.jurisdiction_tag(row),
        "archive_member": member,
    }
    state.mark(key)
    _accept_record(record, out_fh, seen, stats)


# ---------------------------------------------------------------------------
# Accept / dedupe
# ---------------------------------------------------------------------------
def _accept_record(record: dict, out_fh, seen: util.Dedupe, stats: dict) -> None:
    missing = [k for k in config.REQUIRED if not _norm_ws(record.get(k))]
    if missing or not _norm_ws(record.get("text")):
        stats["empty_text"] += 1
        if missing:
            stats.setdefault("missing_required", []).append(missing)
        return
    fp = util.sha1_text(record["text"])
    if not seen.add("text", fp):
        stats["dup_text"] += 1
        return
    name = util.name_key(record.get("archive_member") or "", record.get("title"))
    if name and not seen.add("name", name):
        stats["dup_name"] += 1
        return
    util.jsonl_write_row(out_fh, record)
    stats["records"] += 1
    stats["_bytes"] = stats.get("_bytes", 0) + len(record["text"])


def extract(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Normalize downloaded text sources into one judgments.jsonl.")
    ap.add_argument("--source", action="append",
                    choices=("aws_sc", "aws_hc", "kaggle_sc", "opennyai_dc"))
    args = ap.parse_args(argv)
    sources = args.source or ["aws_sc", "aws_hc", "kaggle_sc", "opennyai_dc"]

    config.OUT.mkdir(parents=True, exist_ok=True)

    # Dedupe lives on disk: constant memory, survives re-runs, so an import of
    # millions of judgments does not need the hashes resident in RAM.
    seen = util.Dedupe(config.OUT / "dedupe.sqlite3")
    existing_rows = 0
    if seen.count("text") == 0 and config.JUDGMENTS_JSONL.exists():
        # First run against an existing corpus: backfill once so a previously
        # accepted judgment cannot be emitted a second time.
        print("backfilling dedupe index from existing judgments.jsonl …", file=sys.stderr)
        for prev in util.jsonl_iter(config.JUDGMENTS_JSONL):
            existing_rows += 1
            seen.add("text", util.sha1_text(prev.get("text") or ""))
            nm = util.name_key(prev.get("archive_member") or "", prev.get("title"))
            if nm:
                seen.add("name", nm)
        seen.commit()
    elif config.JUDGMENTS_JSONL.exists():
        existing_rows = int(seen.count("text"))

    report = util.load_json(config.REPORT_PATH)
    stats: dict[str, Any] = {
        "schema": list(config.UNIFIED_FIELDS),
        "sources": (report.get("sources") or {}) if report else {},
        "rows_before": existing_rows,
    }

    # Copy the accepted corpus forward, then append into the copy and swap.
    # Writing the temp file from scratch used to discard every earlier row.
    tmp = config.JUDGMENTS_JSONL.with_name(config.JUDGMENTS_JSONL.name + ".tmp")
    if config.JUDGMENTS_JSONL.exists():
        shutil.copyfile(config.JUDGMENTS_JSONL, tmp)
    else:
        tmp.write_text("", encoding="utf-8")

    appended = 0
    with tmp.open("a", encoding="utf-8") as out_fh:
        for source in sources:
            sub: dict[str, Any] = {"records": 0, "empty_text": 0, "dup_text": 0,
                                   "dup_name": 0, "missing_required": []}
            state = util.JobState(config.RAW / (source + ".extract.sqlite3"))
            stats["sources"][source] = sub
            if source in ("aws_sc", "aws_hc"):
                for record in _extract_aws(source, state, sub):
                    _accept_record(record, out_fh, seen, sub)
            elif source in ("kaggle_sc", "opennyai_dc"):
                folder = dl_kaggle.discover_data_dir() if source == "kaggle_sc" else _opennyai_dir()
                if folder is None:
                    print(f"  {source} skipped: not downloaded yet", file=sys.stderr)
                    continue
                extract_tabular(source, folder, state, out_fh, seen, sub)
            appended += sub["records"]
            seen.commit()
        out_fh.flush()
    tmp.replace(config.JUDGMENTS_JSONL)
    seen.close()

    stats["rows_after"] = existing_rows + appended
    for sub in stats["sources"].values():
        sub.pop("_bytes", None)
        if not sub.get("missing_required"):
            sub.pop("missing_required", None)
    util.write_json(config.REPORT_PATH, stats)
    for name, sub in stats["sources"].items():
        print(f"  {name}: kept {sub.get('records', 0)}, empty {sub.get('empty_text', 0)}, "
              f"dup_text {sub.get('dup_text', 0)}, dup_name {sub.get('dup_name', 0)}")
    print(f"appended {appended} new rows; corpus now {stats['rows_after']} rows "
          f"({config.JUDGMENTS_JSONL})")


def _opennyai_dir() -> Path | None:
    for candidate in (dl_opennyai.RAW_DIR / "dl", dl_opennyai.RAW_DIR / "local"):
        if candidate.is_dir():
            return candidate
    return None
