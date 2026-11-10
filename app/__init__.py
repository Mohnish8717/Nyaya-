"""Package root: configuration boundary.

`.env` is read here, at package-import time, so every entry point gets the
same behavior — `uvicorn app.main:app`, `run.sh`, Docker, tests — and no
module's import-time `os.getenv` can silently miss a key that is sitting in
the environment file. Values already present in the real environment always
win over the file (operators can override per process).
"""
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Search upward so a nested checkout still finds the real .env; the
    # explicit encoding matches what run.sh sources.
    load_dotenv(encoding="utf-8")
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", encoding="utf-8")
except ImportError:  # pragma: no cover - python-dotenv is a hard dependency
    pass
