# A single-process deployment of the legal assistant.
#
# One worker on purpose: a voice turn keeps its barge-in state in the shared
# session database, and the speech proxy holds one upstream stream per caller,
# so replicas are scaled behind the same volume rather than inside one process.
# `UVICORN_WORKERS` is therefore not exposed here.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Dependencies first, so a source edit does not reinstall them.
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY app ./app
COPY prompts ./prompts
COPY data ./data
COPY static ./static

# The session store and index live here. Created and owned by the runtime user
# so a mounted volume inherits a writable path rather than root-owned files.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/.runtime \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
