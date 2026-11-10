"""Write the DDL eCourts case metadata into its own queryable store.

DDL files are large CSVs (cases_YYYY per year, plus act/section, case-type and
judges metadata). Columns vary slightly across releases, so the loader maps
them by name-intersection with config.DDL_FIELDS and reports exactly which
source columns it found. Per-file manifests make re-runs incremental.
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path
from typing import Any

import config


def _open_csv_like(path: Path):
    if path.suffix.lower() == ".zip":
        import io
        import zipfile

        zf = zipfile.ZipFile(path)
        member = next(m for m in zf.namelist() if m.lower().endswith(".csv"))
        return io.TextIOWrapper(zf.open(member, "r"), encoding="utf-8", errors="replace")
    return path.open(newline="", encoding="utf-8", errors="replace")


def _column_map(header: list[str]) -> dict[str, str]:
    """Map DDL header names -> canonical fields (case-insensitive, fuzzy)."""
    low = {c.strip().lower(): c for c in header}
    out: dict[str, str] = {}
    def take(canonical: str, *candidates: str) -> None:
        for cand in candidates:
            if cand in low and canonical not in out:
                out[canonical] = low[cand]
    take("ddl_case_id", "ddl_case_id", "ddlcaseid", "cino", "cnr")
    take("year", "year")
    take("state", "state_name", "state", "state_code")
    take("district", "district_name", "district", "dist_name")
    take("court", "court_no", "court_num", "court")
    take("case_type", "type_name", "case_type", "type")
    take("act", "act", "act_name")
    take("section", "section", "sections", "no_of_sections")
    take("disposition", "disp_name", "disposition_name", "disposition")
    take("disposition_detail", "disposition_of_case", "disp_of_case")
    take("date_filing", "date_filing", "file_date", "date_of_filing")
    take("date_registration", "date_registration", "reg_date")
    take("date_decision", "date_decision", "date_of_decision")
    take("judge_position", "judge_position", "position")
    take("female_defendant", "female_defendant")
    take("female_petitioner", "female_petitioner")
    take("female_advocate", "female_adv")
    take("criminal", "criminal", "case_type_id")
    return out


def load_ddl_cases(folder: Path = None) -> dict[str, Any]:
    folder = folder or (config.RAW / "ddl" / "dl")
    if not folder.is_dir():
        folder = config.RAW / "ddl" / "local"
    if not folder.is_dir():
        sys.exit(
            "DDL data not found. Fetch it per README: save file URLs into ddl_links.txt, "
            "run `python pipeline.py download --source ddl_meta`, or pass --local <folder>."
        )

    config.DDL_DB.parent.mkdir(parents=True, exist_ok=True)
    db = config.DDL_DB.parent / config.DDL_DB.name

    conn = sqlite3.connect(db, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS ddl_cases(
               ddl_case_id TEXT, year INTEGER, state TEXT, district TEXT, court TEXT,
               case_type TEXT, act TEXT, section TEXT, disposition TEXT, disposition_detail TEXT,
               date_filing TEXT, date_registration TEXT, date_decision TEXT,
               judge_position TEXT, female_defendant INTEGER, female_petitioner INTEGER,
               female_advocate INTEGER, criminal INTEGER, source_file TEXT,
               source_dataset TEXT,
               PRIMARY KEY(ddl_case_id, source_file)
           ) WITHOUT ROWID"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ddl_lookup ON ddl_cases(state, year)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ddl_acts ON ddl_cases(act, section)")
    conn.execute("CREATE TABLE IF NOT EXISTS done(key TEXT PRIMARY KEY, at REAL)")
    per_file_stats: dict[str, Any] = {}

    files = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in (".zip", ".csv")]
    if not files:
        sys.exit(f"No zip/csv files under {folder}")
    only_cases = [f for f in files if "case" in f.name.lower() or "files" in f.name.lower()]
    for path in (only_cases or files):
        key = f"ddl:{path.name}:{path.stat().st_size}"
        if bool(conn.execute("SELECT 1 FROM done WHERE key=?", (key,)).fetchone()):
            per_file_stats[path.name] = "already loaded"
            continue
        mapped: dict[str, str] | None = None
        batch: list[tuple[Any, ...]] = []
        inserted = 0
        with _open_csv_like(path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if mapped is None and reader.fieldnames:
                    mapped = _column_map(reader.fieldnames)
                    if not mapped.get("ddl_case_id"):
                        sys.exit(
                            f"{path.name}: no case id column found in {reader.fieldnames[:8]}… — "
                            "unexpected DDL layout; extend _column_map rather than guessing"
                        )
                vals = tuple(
                    row.get(mapped.get(field, ""), "") if mapped.get(field) else ""
                    for field in config.DDL_FIELDS if field not in ("source_dataset",)
                ) + (path.name, "ddl_meta")
                if not vals[0]:
                    continue
                batch.append(vals)
                if len(batch) >= 5_000:
                    conn.executemany(_insert_sql, batch)
                    inserted += len(batch)
                    batch.clear()
        if batch:
            conn.executemany(_insert_sql, batch)
            inserted += len(batch)
        conn.execute("INSERT OR REPLACE INTO done(key, at) VALUES(?, julianday('now'))", (key,))
        conn.commit()
        per_file_stats[path.name] = {"inserted": inserted, "file": path.name}
    conn.close()
    return {"files": per_file_stats, "db": str(db)}


_insert_sql = (
    "INSERT OR IGNORE INTO ddl_cases("
    "ddl_case_id, year, state, district, court, case_type, act, section, disposition, "
    "disposition_detail, date_filing, date_registration, date_decision, judge_position, "
    "female_defendant, female_petitioner, female_advocate, criminal, source_file, "
    "source_dataset) "
    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)
