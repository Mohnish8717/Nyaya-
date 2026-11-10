#!/usr/bin/env python3
"""Build/update the local searchable index for normalized judgment JSONL."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.case_corpus import index_imported_cases

if __name__ == "__main__":
    print(f"Indexed {index_imported_cases(force=True):,} case records.")

