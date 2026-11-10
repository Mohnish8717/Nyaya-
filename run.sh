#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
python -m pip install -r requirements.txt
# Dense (hybrid) retrieval is local-on-CPU; optional but part of the intended
# recall quality, so a failed install degrades instead of stopping the boot.
python -m pip install -r requirements-retrieval.txt || \
  echo "[run.sh] dense retrieval deps missing — continuing with lexical retrieval only" >&2
if [ ! -f .env ]; then cp .env.example .env; fi
set -a
source .env
set +a
exec uvicorn app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --reload
