"""Kaggle source: SC Judgments India 1950–2024 (Indian Kanoon full text).

Kaggle exposes no stable direct link and its column names are not documented on
the public page, so this module never hardcodes fields: `extract` maps columns
by content detection and writes what it observed into the report.

Credentials: the official Kaggle CLI/API (~/.kaggle/kaggle.json or
KAGGLE_USERNAME/KAGGLE_KEY). A previously downloaded local copy also works via
--local <dir>.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import config
import util

RAW_DIR = config.RAW / "kaggle"


def _have_kaggle() -> bool:
    return shutil.which("kaggle") is not None


def _creds_ok() -> bool:
    import os
    home = Path("~/.kaggle/kaggle.json").expanduser()
    return home.exists() or bool(os.getenv("KAGGLE_KEY") and os.getenv("KAGGLE_USERNAME"))


def download_command(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fetch the Kaggle SC judgments dataset.")
    ap.add_argument("--local", help="path to an already-downloaded dataset folder (CSV/parquet files)")
    args = ap.parse_args(argv)

    if args.local:
        src = Path(args.local)
        if not src.is_dir():
            raise SystemExit(f"--local must be an existing folder, got {src}")
        dest = RAW_DIR / "local"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        shutil.copytree(src, dest, dirs_exist_ok=True)
        print(f"Copied local Kaggle dataset to {dest}")
        return

    if not _have_kaggle():
        sys.exit("kaggle CLI is not installed: pip install -r requirements.txt, then `kaggle --version`.")
    if not _creds_ok():
        sys.exit(
            "Kaggle credentials missing. Put kaggle.json in ~/.kaggle/ or export "
            "KAGGLE_USERNAME and KAGGLE_KEY before running the Kaggle source."
        )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    state = util.JobState(RAW_DIR / ".state.sqlite3")
    key = "dataset_downloaded"
    if state.done(key):
        print("Kaggle dataset already downloaded (resume manifest); nothing to do.")
        return
    cmd = ["kaggle", "datasets", "download", config.KAGGLE_HANDLE, "--path", str(RAW_DIR / "dl"), "--unzip"]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    state.mark(key)
    print(f"Downloaded and unzipped into {RAW_DIR / 'dl'}")


def discover_data_dir() -> Path:
    """The CLI unzips flat files somewhere under out/raw/kaggle; find them."""
    for base in (RAW_DIR / "dl", RAW_DIR / "local", RAW_DIR):
        if not base.is_dir():
            continue
        files = sorted(base.rglob("*"))
        if any(f.suffix.lower() in (".csv", ".parquet") and not f.name.startswith(".") for f in files):
            return base
    raise SystemExit(
        "No Kaggle data found yet — run `python pipeline.py download --source kaggle_sc` first, "
        "or pass --local with an existing download."
    )
