#!/usr/bin/env python3
"""Fetch the supplied Indian law Q&A dataset with Hugging Face Datasets.

The 24,607-row corpus is stored with the project so the local prototype can
index it offline. It is a secondary discovery aid, never a legal authority.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "indian-law-dataset.json"
TEMP = OUTPUT.with_suffix(".json.tmp")
os.environ.setdefault("HF_HOME", str(ROOT / ".runtime" / "huggingface"))

from datasets import load_dataset


def main() -> int:
    dataset = load_dataset("viber1/indian-law-dataset", split="train")
    rows = []
    for row in dataset:
        question = str(row.get("Instruction", "")).strip()
        answer = str(row.get("Response", "")).strip()
        rows.append({"Instruction": question, "Response": answer})
    TEMP.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TEMP.replace(OUTPUT)
    print(f"Saved {len(rows):,} question-answer rows ({OUTPUT.stat().st_size:,} bytes) to {OUTPUT}")
    print("License: Apache-2.0. Use only as unverified retrieval context, never as a cited legal source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

