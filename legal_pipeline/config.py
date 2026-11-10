"""Standalone judicial-text pipeline.

Deliberately isolated from the Nyaya app: this folder only reads public
datasets and writes to ``out/`` beside this file, so it can be moved to any
workspace and run with no project-specific imports.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
RAW = OUT / "raw"
REPORT_PATH = OUT / "sources_report.json"

# ---------------------------------------------------------------------------
# Unified judgment schema — the contract every text-producing source must emit,
# and the exact schema the Nyaya app's case indexer consumes.
# It is a dict; `required` fields must be non-empty strings, `optional` may be "".
# ---------------------------------------------------------------------------
REQUIRED = ("id", "title", "court", "text", "source_url", "decision_date", "dataset")
OPTIONAL = ("citation", "language", "jurisdiction", "archive_member", "court_level")
UNIFIED_FIELDS = REQUIRED + OPTIONAL

# ---------------------------------------------------------------------------
# Court level: the label that travels with every chunk so a reader model knows
# whether a passage is Supreme Court, a High Court, or a district court. It is
# also written into the embedded text, so nearest-neighbour search can separate
# the three instead of mixing a district sessions order with a Constitution
# Bench judgment.
# ---------------------------------------------------------------------------
LEVELS = ("supreme", "high", "district", "unknown")
LEVEL_FOR_DATASET = {
    "aws_sc": "supreme",
    "kaggle_sc": "supreme",
    "aws_hc": "high",
    # The OpenNyAI sample mixes levels, so its level is read per row from the
    # court column rather than assumed.
    "opennyai_dc": "",
    "ddl_meta": "district",
}
LEVEL_LABEL = {
    "supreme": "SUPREME COURT OF INDIA",
    "high": "HIGH COURT",
    "district": "DISTRICT COURT",
    "unknown": "COURT (LEVEL UNRECORDED)",
}

# ---------------------------------------------------------------------------
# Curated development corpus: a slice small enough to embed on one machine.
# `--target-mb` is the size of the ONE deliverable file, because that is the
# number a person can check on disk. Text is stored once (chunk passages); the
# BM25 table is an index over those passages, not a second copy; vectors are
# ~0.5 KB each. Measured ratio is roughly file = 1.4 x text, so the text budget
# is a fixed fraction of the file target and the actual size is printed.
# 150 MB of file is ~35 K chunks: hours on CPU, under an hour on one GPU.
# ---------------------------------------------------------------------------
CORPUS_TARGET_MB = float(os.getenv("PIPELINE_CORPUS_TARGET_MB") or 150.0)
CORPUS_TEXT_SHARE_OF_FILE = 0.70
CORPUS_MIX = {"supreme": 0.35, "high": 0.45, "district": 0.20}
# Hard ceiling on judgments taken from a single archive so one court-year
# cannot dominate the slice; the cap is per (tar, language) pair.
CORPUS_PER_TAR_CAP = int(os.getenv("PIPELINE_CORPUS_PER_TAR_CAP", "150"))
# Languages the encoder is expected to handle. Used purely for reporting and to
# stop a slice from silently being English-only when regional tars exist.
CORPUS_LANGUAGES = (
    "en", "hi", "ta", "te", "kn", "ml", "mr", "bn", "gu", "pa", "or", "ur",
)
SINGLE_DB = OUT / "corpus_embeddings.sqlite3"

# ---------------------------------------------------------------------------
# Sources. Fields/sizes are what the upstream maintainers document; anything
# row-level is measured at runtime by the `inspect` command, never assumed.
# ---------------------------------------------------------------------------
SOURCES: dict[str, dict[str, str]] = {
    "aws_sc": {
        "label": "Supreme Court of India judgments (AWS Open Data, Dattam Labs)",
        "level": "supreme-court",
        "license": "CC-BY-4.0",
        "location": "S3 bucket indian-supreme-court-judgments, ap-south-1, no signing",
        "size": "~35K judgments, ~52 GiB of archives, 1950–2025, English + regional",
        "text_fields": "judgment PDFs per member; metadata JSON: case number, decision date, bench, petitioners, respondents, state, disposal",
        "cadence": "twice monthly",
    },
    "aws_hc": {
        "label": "25 High Courts judgments (AWS Open Data, Dattam Labs)",
        "level": "high-court",
        "license": "CC-BY-4.0",
        "location": "S3 bucket indian-high-court-judgments, ap-south-1, no signing",
        "size": "~17.8M judgments, ~1.25 TiB of S3 tar archives across all years",
        "text_fields": "judgment PDFs per member; metadata parquet/json: court/bench/citation-like case info per member",
        "cadence": "quarterly",
    },
    "kaggle_sc": {
        "label": "SC Judgments India 1950–2024 (Kaggle, scraped from Indian Kanoon)",
        "level": "supreme-court",
        "license": "dataset terms on Kaggle; public judgment text",
        "location": "kaggle datasets adarshsingh0903/legal-dataset-sc-judgments-india-19502024",
        "size": "full text for ~98% coverage of SC judgments 1950–2024; exact size reported by `inspect` after Kaggle auth",
        "text_fields": "detected at runtime: longest text column = judgment; title/date/hearing columns mapped heuristically",
        "cadence": "static upload",
    },
    "opennyai_dc": {
        "label": "OpenNyAI representative judgment sample (incl. Delhi & Bengaluru district courts)",
        "level": "district-court (sample) + SC/HC",
        "license": "apache-2.0",
        "location": "Hugging Face datasets opennyaiorg/InJudgements_dataset (parquet)",
        "size": "~293 MB, most-cited judgments 1950–2017 (district coverage: Delhi + Bengaluru only)",
        "text_fields": "detected at runtime: judgment text, indiankanoon URL, court, date",
        "cadence": "static sample",
    },
    "ddl_meta": {
        "label": "Dev Data Lab eCourts district case metadata 2010–2018 (NO judgment text)",
        "level": "district-court metadata",
        "license": "CC BY-NC-SA 4.0 — non-commercial use only",
        "location": "devdatalab.org/judicial-data, download links at that page's data table",
        "size": "81.2M cases; links are behind that page's data table (see README), no text fields exist",
        "text_fields": "case dates, acts/sections, type, disposition, judge position — used for statistical lookup, never embedded",
        "cadence": "static release",
    },
}

AWS_BUCKETS = {
    "aws_sc": "indian-supreme-court-judgments",
    "aws_hc": "indian-high-court-judgments",
}
AWS_REGION = "ap-south-1"

KAGGLE_HANDLE = "adarshsingh0903/legal-dataset-sc-judgments-india-19502024"
HF_REPO = "opennyaiorg/InJudgements_dataset"
DDL_PAGE = "https://www.devdatalab.org/judicial-data"
DDL_LINKS_FILE = ROOT / "ddl_links.txt"

# Provenance fallback for REQUIRED.source_url: a judgment parsed from a tar
# often has no per-row url in its metadata, and dropping the row would lose the
# passage entirely. The dataset itself is still the honest provenance.
SOURCE_URL_FALLBACK = {
    "aws_sc": "https://registry.opendata.aws/indian-supreme-court-judgments/",
    "aws_hc": "https://registry.opendata.aws/indian-high-court-judgments/",
    "kaggle_sc": "https://www.kaggle.com/datasets/adarshsingh0903/legal-dataset-sc-judgments-india-19502024",
    "opennyai_dc": "https://huggingface.co/datasets/opennyaiorg/InJudgements_dataset",
}

# Output artifacts (all under out/):
JUDGMENTS_JSONL = OUT / "judgments.jsonl"
CHUNKS_DB = OUT / "chunks.sqlite3"
VECTORS_DIR = OUT / "vectors"
VECTORS_MANIFEST = VECTORS_DIR / "manifest.sqlite3"
CASES_INDEX_DB = OUT / "cases_index.sqlite3"
DDL_DB = OUT / "ddl_cases.sqlite3"

# App-compatible exports: the exact schemas the Nyaya app already opens.
#   dense_cases.sqlite3 -> app/dense.py   (vectors(source_id, chunk, vec INT8))
#   cases.sqlite3       -> app/case_corpus.py (case_fts, same columns)
EXPORT_DIR = OUT / "app"
EXPORT_DENSE_DB = EXPORT_DIR / "dense_cases.sqlite3"
EXPORT_CASES_DB = EXPORT_DIR / "cases.sqlite3"

# Budgets. Every download and every import step refuses to exceed them;
# raise via CLI flags rather than silently consuming TBs.
def _gb(env_key: str, default: float) -> float:
    try:
        return float(os.getenv(env_key, default))
    except ValueError:
        return default


DEFAULT_MAX_DOWNLOAD_GB = _gb("PIPELINE_MAX_DOWNLOAD_GB", 4.0)

# Chunking: ~800-token windows, headnote-first per judgment, capped per case so
# one multi-hundred-page decision cannot dominate the vector index.
CHUNK_TARGET_CHARS = 3200
CHUNK_OVERLAP_CHARS = 150
CHUNKS_PER_CASE_CAP = int(os.getenv("PIPELINE_CHUNKS_PER_CASE", "8"))

# Embeddings: int8-quantised unit vectors, same quantisation the app uses.
DENSE_MODEL = os.getenv("DENSE_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_BATCH = 64
SHARD_MAX_VECTORS = 200_000
# Encoder context is 512 tokens; ~2400 chars of Latin/IAS text is roughly 600
# tokens, so texts are truncated to the model's window instead of being cut
# mid-instruction by the tokenizer.
EMBED_MAX_CHARS = int(os.getenv("PIPELINE_EMBED_MAX_CHARS", "2400"))

# ---------------------------------------------------------------------------
# Parquet-first ingest. The published parquet is structured metadata only — it
# carries case identity, court, dates, citation, judge and a PDF pointer, never
# the judgment text. So parquet decides *what* to read and supplies the labels,
# and the pointed-at PDFs supply the words. Columns are read with projection so
# the bulky `raw_html` column is never transferred.
# ---------------------------------------------------------------------------
PARQUET_COLUMNS = {
    "aws_sc": ("title", "description", "citation", "case_id", "cnr", "decision_date",
               "court", "available_languages", "path", "year", "judge"),
    "aws_hc": ("court_code", "title", "description", "cnr", "decision_date",
               "court", "pdf_link", "pdf_exists", "judge", "date_of_registration"),
}
# Share of the transfer budget that parquet reads may consume; PDFs get the rest.
PARQUET_RESERVE_SHARE = 0.15
PARQUET_ROWS_PER_PARTITION = int(os.getenv("PIPELINE_PARQUET_ROWS", "12"))
# A single judgment PDF above this is skipped rather than allowed to eat the run.
PARQUET_MAX_PDF_MB = float(os.getenv("PIPELINE_PARQUET_MAX_PDF_MB", "12"))
# A partition whose selected columns exceed this is read page-by-page anyway;
# the number is a pre-flight warning threshold, not a hard limit.
PARQUET_PARTITION_WARN_MB = 40.0

DDL_FIELDS = (
    "ddl_case_id", "year", "state", "district", "court",
    "case_type", "act", "section", "disposition", "disposition_detail",
    "date_filing", "date_registration", "date_decision",
    "judge_position", "female_defendant", "female_petitioner", "female_advocate",
    "criminal", "source_dataset",
)
