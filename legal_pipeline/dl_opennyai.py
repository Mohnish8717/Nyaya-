"""OpenNyAI representative sample (Hugging Face parquet): the only bulk source
with real district-court judgment text (Delhi + Bengaluru district courts,
1950–2017) plus SC/HC coverage — apache-2.0.

The dataset card is gated, so Hugging Face requires a login/token for direct
download; a local folder passed via --local (or PW local fallback) is accepted.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import config
import util

RAW_DIR = config.RAW / "opennyai"


def has_token() -> bool:
    """Whether Hugging Face has a credential it can send.

    Only a boolean is returned. For a gated repo the token is not optional, so
    checking first turns a traceback into a two-line instruction.
    """
    try:
        from huggingface_hub import get_token
    except ImportError:
        return False
    try:
        return bool(get_token())
    except Exception:  # noqa: BLE001 - cache unreadable is "no token"
        return False


def download_command(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fetch the OpenNyAI judgments sample.")
    ap.add_argument("--local", help="path to an already-downloaded dataset folder (parquet/json)")
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
        print(f"Copied local OpenNyAI dataset to {dest}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "dl"
    existing = sorted(target.rglob("*.parquet")) if target.is_dir() else []
    if existing and sum(p.stat().st_size for p in existing) > 1_000_000:
        total = sum(p.stat().st_size for p in existing)
        print(f"OpenNyAI parquet already present: {len(existing)} file(s), "
              f"{total / 1e6:.1f} MB in {target}; nothing to do.")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is not installed: pip install -r requirements.txt")
    # The whole dataset is ~293 MB of parquet (11,970 judgments, ~629 MB of
    # text), which is small enough to take in full — no sampling at this layer.
    # No `allow_patterns` filter: the repo also holds the card and splits
    # metadata, and a partial snapshot is harder to reason about than a full one.
    if not has_token():
        sys.exit(
            f"{config.HF_REPO} is a gated dataset: its file list is public but the\n"
            "parquet needs a Hugging Face token. Two steps, once per machine:\n"
            f"  1. accept the terms: https://huggingface.co/datasets/{config.HF_REPO}\n"
            "  2. authenticate:    huggingface-cli login   (or export HF_TOKEN=hf_...)\n"
            "Then re-run. `--local <folder>` skips the download entirely."
        )
    try:
        got = snapshot_download(
            repo_id=config.HF_REPO,
            repo_type="dataset",
            local_dir=str(target),
        )
    except Exception as exc:  # noqa: BLE001 - HF raises several error types
        message = str(exc)
        if "gated" in message.lower() or "401" in message or "restricted" in message.lower():
            sys.exit(
                f"Hugging Face refused the download: {message.splitlines()[0]}\n\n"
                "This dataset is gated for file access (its file list is public, the\n"
                "parquet is not). Three steps, in order:\n"
                f"  1. open https://huggingface.co/datasets/{config.HF_REPO} and accept the terms\n"
                "  2. create a read token at https://huggingface.co/settings/tokens\n"
                "  3. authenticate:  huggingface-cli login   (or export HF_TOKEN=hf_...)\n"
                "Then re-run this command. `--local <folder>` skips the download entirely."
            )
        sys.exit(
            f"Hugging Face download failed: {exc}\n"
            f"Repo: https://huggingface.co/datasets/{config.HF_REPO}\n"
            "That is a transport/proxy failure rather than an access one; retry, or "
            "pass --local to reuse an existing copy."
        )
    files = sorted(Path(got).rglob("*.parquet")) if Path(got).is_dir() else []
    total = sum(p.stat().st_size for p in files)
    if not files:
        sys.exit(f"Snapshot downloaded to {got} but no .parquet was found.")
    print(f"Downloaded {len(files)} parquet file(s), {total / 1e6:.1f} MB, to {got}")
