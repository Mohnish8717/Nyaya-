"""Dev Data Lab district-case metadata (CC BY-NC-SA 4.0, non-commercial).

DDL publishes county-level case records with no judgment text; this module
pulls only the tabular metadata and lands it in its own store, never in the
embedding inputs.

DDL does not expose a stable link list. Two accepted ways in, in order:

1. `ddl_links.txt` next to this script — one raw URL per line (get them from
   the "Data download / Act sections / cases metadata / Judges metadata" table
   on the judicial-data page after accepting their data terms).
2. `--link <url>` or `--local <folder>` on this command. Every accepted input
   is HEAD-probed for size before consuming budget.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import urllib.request
from pathlib import Path

import config
import util

RAW_DIR = config.RAW / "ddl"
_LINK_RE = re.compile(r'https?://[^\s\'"<>]+\.(?:zip|csv|dta|stata)\b', re.I)


def gather_links(page_html: str) -> list[str]:
    """Collect downloadable-data URLs embedded on the DDL judicial-data page.

    Only content-bearing files (zip/csv/dta) are considered; html/js/css junk is
    filtered. If the page listed no obvious data files, the caller needs to
    supply ddl_links.txt — we never invent candidate URLs.
    """
    if not page_html:
        with urllib.request.urlopen(config.DDL_PAGE, timeout=30) as resp:
            page_html = resp.read().decode("utf-8", "replace")
    seen: set[str] = set()
    for m in _LINK_RE.finditer(page_html):
        url = m.group(0)
        if url not in seen:
            seen.add(url)
    return sorted(seen)


def read_links_file() -> list[str]:
    if not config.DDL_LINKS_FILE.exists():
        return []
    out = []
    for line in config.DDL_LINKS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and line.startswith("http"):
            out.append(line)
    return out


def content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers.get("Content-Length", 0))


def download_command(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fetch Dev Data Lab case metadata (non-commercial licence).")
    ap.add_argument("--max-download-gb", type=float, default=config.DEFAULT_MAX_DOWNLOAD_GB)
    ap.add_argument("--link", action="append", help="explicit file URL; repeatable")
    ap.add_argument("--local", help="folder with already-downloaded DDL zip/csv files")
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
        print(f"Copied local DDL data to {dest}")
        return

    links = list(args.link or ())
    if not links:
        links = read_links_file()
    if not links:
        page_links = gather_links("")
        if page_links:
            print(f"No ddl_links.txt; the judicial-data page itself lists {len(page_links)} files:")
            for url in page_links:
                print(" ", url)
            print("Copy the ones you want (one per line) into ddl_links.txt and re-run.")
        print(
            f"\nDDL data lives behind {config.DDL_PAGE} — open the page, accept the CC BY-NC-SA "
            "terms, then either save the file links into ddl_links.txt or unzip them and pass --local."
        )
        sys.exit(0)

    # Plain HTTP HEAD/GET only; DDL needs no cloud signing.
    budget = util.Budget.from_gb_flag(args.max_download_gb, "ddl")
    order = sorted(set(links))
    for url in order:
        size = content_length(url)
        name = url.rsplit("/", 1)[-1]
        dest = RAW_DIR / "dl" / name
        if dest.exists() and dest.stat().st_size == size:
            print(f"have {name}")
            continue
        budget.check(size, url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"streaming {name} ({size / 1e9:.2f} GB)")
        _stream_download(url, dest, size)
    print("DDL pull complete.")


def _stream_download(url: str, dest: Path, expected: int) -> None:
    partial = dest.with_suffix(dest.suffix + ".part")
    got = 0
    with urllib.request.urlopen(url, timeout=60) as resp, partial.open("wb") as fh:
        while True:
            chunk = resp.read(1 << 22)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
    if expected and got != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{url}: got {got} B, expected {expected} B")
    partial.replace(dest)
    print(f"done {dest.name}")
