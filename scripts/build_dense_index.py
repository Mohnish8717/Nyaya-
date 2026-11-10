"""Build the local dense index for judgments.

    python scripts/build_dense_index.py            # incremental / idempotent
    python scripts/build_dense_index.py --force    # rebuild from scratch

Runs entirely on CPU through onnxruntime. The first run downloads the
multilingual encoder (~120 MB) once; later runs use the fitted cache.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import dense  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    status = dense.backend_status()
    if not status["dependency_installed"]:
        print(json.dumps({
            "error": "dense dependencies missing",
            "fix": "pip install -r requirements-retrieval.txt",
        }, indent=2))
        return 1
    result = dense.build_index(force=args.force)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
