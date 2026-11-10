"""Shared plumbing: budgets, resume manifests, hashing, JSON/JSONL IO.

Everything here is stdlib-only so downloaders and builders need no project
dependencies beyond the few declared in requirements.txt.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator


class BudgetExceeded(RuntimeError):
    """Raised before any archive overruns the caller's download budget."""


class Budget:
    """Byte budget that refuses work instead of partially consuming it."""

    def __init__(self, max_bytes: float):
        self.max_bytes = int(max_bytes)
        self.spent = 0

    def check(self, size: int, what: str) -> int:
        if self.spent + size > self.max_bytes:
            raise BudgetExceeded(
                f"refusing {what}: needs {size / 1e9:.2f} GB, "
                f"already allocated {self.spent / 1e9:.2f} GB of "
                f"{self.max_bytes / 1e9:.2f} GB budget. Raise --max-download-gb "
                "or narrow --years/--courts."
            )
        self.spent += size
        return size

    @classmethod
    def from_gb_flag(cls, gb: float, what: str = "") -> "Budget":
        if gb <= 0:
            raise ValueError(f"--max-download-gb must be > 0 ({what})")
        return cls(int(gb * 1e9))


def sha1_text(text: str) -> str:
    """Full-text fingerprint: judgments share boilerplate openings, so hashing
    only a prefix would silently drop distinct decisions."""
    return hashlib.sha1(" ".join(text.split()).encode("utf-8", "ignore")).hexdigest()


def name_key(archive_member: str, title: str = "") -> str:
    """File-stem identity used to catch the same judgment imported twice once
    by different pipelines that assign different ids but the same source file."""
    raw = Path(archive_member or title or "").name
    low = raw.lower()
    if low.endswith(".pdf"):
        low = low[:-4]
    for suffix in ("_en", "-en", "_english"):
        if low.endswith(suffix):
            low = low[: -len(suffix)]
    return low


class JobState:
    """Tiny resumable-manifest store: one row per completed unit of work."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("CREATE TABLE IF NOT EXISTS done(key TEXT PRIMARY KEY, value TEXT, at REAL)")
        self.db.commit()

    def done(self, key: str) -> bool:
        return bool(self.db.execute("SELECT 1 FROM done WHERE key=?", (key,)).fetchone())

    def value(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM done WHERE key=?", (key,)).fetchone()
        return row[0] if row and row[0] is not None else None

    def mark(self, key: str, value: Any = "") -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO done(key,value,at) VALUES(?,?,?)",
            (key, json.dumps(value) if not isinstance(value, str) else value, time.time()),
        )
        self.db.commit()

    def keys(self, prefix: str = "") -> list[str]:
        like = prefix + "%"
        return [row[0] for row in self.db.execute("SELECT key FROM done WHERE key LIKE ?", (like,))]

    def count(self, prefix: str = "") -> int:
        like = prefix + "%"
        return int(self.db.execute("SELECT COUNT(*) FROM done WHERE key LIKE ?", (like,)).fetchone()[0])


class Dedupe:
    """Constant-memory duplicate detection backed by SQLite.

    A Python set of 8 M text hashes plus 8 M file names costs close to a
    gigabyte and dies with the process; a UNIQUE index on disk costs nothing
    per row and survives re-runs, which is what makes a TB-scale import
    resumable. ``add`` returns True when the key was new.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS seen(kind TEXT, key TEXT, PRIMARY KEY(kind,key))")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_seen_kind ON seen(kind)")
        self.db.commit()

    def add(self, kind: str, key: str) -> bool:
        if not key:
            return True
        cur = self.db.execute("INSERT OR IGNORE INTO seen(kind,key) VALUES(?,?)", (kind, key))
        return cur.rowcount > 0

    def contains(self, kind: str, key: str) -> bool:
        return bool(self.db.execute("SELECT 1 FROM seen WHERE kind=? AND key=?", (kind, key)).fetchone())

    def count(self, kind: str = "text") -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM seen WHERE kind=?", (kind,)).fetchone()[0])

    def commit(self) -> None:
        self.db.commit()

    def close(self) -> None:
        self.db.close()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


class Cast:
    """Tolerant field coercion for messy third-party rows."""

    @staticmethod
    def text(value: Any, limit: int | None = None) -> str:
        if value is None:
            return ""
        out = str(value)
        if limit is not None and len(out) > limit:
            out = out[:limit]
        return out

    @staticmethod
    def int(value: Any) -> int:
        try:
            return int(str(value).strip()[:11])
        except (ValueError, TypeError, AttributeError):
            try:
                return int(float(str(value).strip()[:11]))
            except (ValueError, TypeError, AttributeError):
                return 0


def jsonl_iter(path: Path) -> Iterator[dict[str, Any]]:
    """Read a .jsonl file yielding decoded rows, skipping blank lines."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate partial last line after an interrupted run


def jsonl_write_row(fh: Any, record: dict[str, Any]) -> None:
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def stamp_for(path: Path, extra: str = "") -> str:
    if not path.exists():
        return "missing"
    return f"{extra}|{path.stat().st_size}:{path.stat().st_mtime_ns}"


_STATE_CODES = {
    "karnataka": "IN-KA", "kerala": "IN-KL", "delhi": "IN-DL", "maharashtra": "IN-MH",
    "bombay": "IN-MH", "gujarat": "IN-GJ", "punjab": "IN-PB", "haryana": "IN-HR",
    "rajasthan": "IN-RJ", "uttar pradesh": "IN-UP", "allahabad": "IN-UP",
    "madhya pradesh": "IN-MP", "bihar": "IN-BR", "patna": "IN-BR", "odisha": "IN-OR",
    "orissa": "IN-OR",    "west bengal": "IN-WB", "calcutta": "IN-WB", "andhra pradesh": "IN-AP",
    "telangana": "IN-TG", "hyderabad": "IN-TG", "tamil nadu": "IN-TN", "madras": "IN-TN",
}


def jurisdiction_tag(row: dict) -> str:
    """Jurisdiction code from any state/court-ish value in the row, else IN.

    Named jurisdictions only: an unmapped court keeps the national default
    rather than being guessed from an unrelated nearby word."""
    for value in row.values():
        low = str(value or "").lower()
        if len(low) < 120:
            for name, code in _STATE_CODES.items():
                if f" {name}" in f" {low}" or low == name:
                    return code
    return "IN"


# Backwards-compatible alias: an earlier version of this module exposed the
# jurisdiction helper under a misleading name.
language_tag = jurisdiction_tag

_SCRIPTS = (
    ("\u0900", "\u097f", "hi"),   # Devanagari — also Marathi/Nepali; hi is the
                                    # honest coarse tag until a reader refines it
    ("\u0980", "\u09ff", "bn"),
    ("\u0a00", "\u0a7f", "pa"),
    ("\u0a80", "\u0aff", "gu"),
    ("\u0b00", "\u0b7f", "or"),
    ("\u0b80", "\u0bff", "ta"),
    ("\u0c00", "\u0c7f", "te"),
    ("\u0c80", "\u0cff", "kn"),
    ("\u0d00", "\u0d7f", "ml"),
    ("\u0600", "\u06ff", "ur"),
)


_LEVEL_PATTERNS = (
    ("supreme", r"supreme\s+court|supreme\s+court\s+of\s+india|\bsci\b"),
    ("high", r"high\s+court|highcourt"),
    ("district", r"district|sessions\s+court|family\s+court|magistrate|civil\s+judge|"
                 r"small\s+causes|tribunal|munsif|adalat"),
)


def court_level(court: str, default: str = "unknown") -> str:
    """Which tier a court belongs to, from its name.

    Only named matches count: "High Court of Karnataka" is high, "Supreme
    Court of India" is supreme, "District & Sessions Judge, Pune" is district.
    Anything unrecognised keeps the caller's default rather than being guessed.
    """
    text = str(court or "")
    for level, pattern in _LEVEL_PATTERNS:
        if re.search(pattern, text, re.I):
            return level
    return default


def level_header(level: str, court: str, jurisdiction: str, language: str,
                 labels: dict[str, str] | None = None) -> str:
    """The one-line label prepended to embedded text.

    Retrieval hands the reader model the verbatim passage and this header, so
    the passage's tier and jurisdiction are never inferred by the model: a
    district-court order cannot be quoted as if it bound the whole country.
    """
    label = (labels or {}).get(level, level.upper())
    court = str(court or "").strip()
    if court and court.lower() not in label.lower():
        label = f"{label} — {court}"
    return f"[{label} | level: {level} | jurisdiction: {jurisdiction or 'IN'} | language: {language or 'unknown'}]"


def script_language(text: str) -> str:
    """Coarse language tag from the dominant Unicode script of a passage.

    Used to fill the `language` field for tabular sources that ship no language
    column, so the embedding stage can report real coverage instead of
    labelling everything "unknown". It is script-level, not linguistic: a
    Marathi judgment is tagged hi because both use Devanagari. Stated plainly
    wherever it is reported.
    """
    sample = text[:4000]
    counts: dict[str, int] = {}
    for char in sample:
        code = ord(char)
        if code < 128:
            continue
        for low, high, tag in _SCRIPTS:
            if low <= char <= high:
                counts[tag] = counts.get(tag, 0) + 1
                break
    if not counts:
        return "en" if sample.strip() else "unknown"
    best = max(counts.items(), key=lambda item: item[1])[1]
    total_non_ascii = sum(counts.values())
    # Latin script dominant: ASCII letters outnumber the non-Latin scripts.
    if total_non_ascii * 8 < len(sample):
        return "en"
    return max(counts, key=lambda tag: counts[tag]) if best else "en"
