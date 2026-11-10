from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit

import httpx
import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .pdf_generator import generate_legal_pdf
from . import reminders as reminder_layer
from . import (
    case_corpus, dense, feedback, judgment_corpus, knowledge, language as language_layer,
    model_gateway, privacy, procedural, retrieval, sessions, speech, tools as tool_layer,
    voice, workflow,
)
from .legal_engine import (
    INDIAN_JURISDICTIONS,
    JURISDICTION_LABELS,
    LANGUAGE_LABELS,
    SERVICE_CATALOG,
    STT_LANGUAGES,
    TEXT_LANGUAGES,
    TTS_LANGUAGES,
    analyze,
    answer_stream,
    answer_turn,
)
from .model_gateway import BASE_URL as LLM_BASE_URL
from .model_gateway import MODEL as LLM_MODEL
from .model_gateway import configured as llm_configured
from .model_gateway import last_error as llm_last_error

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
GNANI_KEY = os.getenv("GNANI_API_KEY_ID", "").strip()
PRISMA_MODEL = os.getenv("PRISMA_MODEL", speech.DEFAULT_PRISMA_MODEL)
TIMBRE_MODEL = os.getenv("TIMBRE_MODEL", speech.DEFAULT_TIMBRE_MODEL)
TIMBRE_VOICE = os.getenv("TIMBRE_DEFAULT_VOICE", speech.V3_VOICES[0])
# `codemix` keeps a Hindi sentence with English legal nouns as one utterance.
# `transcribe` normalises harder and loses that mix, so it is the fallback for
# the single-language v2.5 model only.
STT_FORMAT = os.getenv("PRISMA_FORMAT", "codemix")
TTS_SAMPLE_RATE = int(os.getenv("TIMBRE_SAMPLE_RATE", "24000") or 24000)
REVIEW_PROVIDER_URL = os.getenv("REVIEW_PROVIDER_URL", "").strip()
REVIEW_PROVIDER_TOKEN = os.getenv("REVIEW_PROVIDER_TOKEN", "").strip()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
TRUST_PROXY = os.getenv("TRUST_PROXY", "0") == "1"
DAILY_TURN_BUDGET = int(os.getenv("DAILY_TURN_BUDGET", "0") or 0)
try:
    _vad_threshold = float(os.getenv("PRISMA_VAD_THRESHOLD", "0.55"))
    PRISMA_VAD_THRESHOLD = _vad_threshold if 0.0 <= _vad_threshold <= 1.0 else 0.55
except ValueError:
    PRISMA_VAD_THRESHOLD = 0.55
try:
    # Server VAD is the end-of-turn signal, so this is the single biggest lever
    # on how long the caller waits before the assistant takes its turn.
    PRISMA_MIN_SILENCE_MS = int(os.getenv("PRISMA_MIN_SILENCE_MS", "350"))
except ValueError:
    PRISMA_MIN_SILENCE_MS = 350


class _JsonLogs(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"level": record.levelname, "event": record.getMessage(), "ts": round(time.time(), 3)}
        for key, value in getattr(record, "fields", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


LOG = logging.getLogger("nyaya")
if not LOG.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(_JsonLogs())
    LOG.addHandler(_handler)
    LOG.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    LOG.propagate = False


def log(event: str, **fields: Any) -> None:
    # Nothing that reaches a log file may contain an Aadhaar, PAN, phone or
    # account number, even when a caller pasted one into a turn.
    LOG.info(event, extra={"fields": privacy.redact_deep(fields)})


class TurnRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=80)
    text: str = Field(min_length=1, max_length=5000)
    language: str = Field(default="auto", max_length=16)
    jurisdiction: str = Field(default="auto", max_length=32)
    service: str | None = Field(default=None, max_length=64)
    consent_confirmed: bool = False


class StartRequest(BaseModel):
    language: str = "auto"
    consent_confirmed: bool = False
    consent_version: str = Field(default=sessions.CONSENT_NOTICE_VERSION, max_length=40)


class ActionRequest(BaseModel):
    session_id: str
    action: str
    payload: dict[str, Any] = Field(default_factory=dict)
    user_confirmed: bool = False


class ReviewRequest(BaseModel):
    session_id: str
    language: str
    user_requested: bool = False
    consent_to_share: bool = False
    summary: str = Field(default="", max_length=3000)
    include_identifiers: bool = False


class ReviewPreviewRequest(BaseModel):
    session_id: str
    summary: str = Field(default="", max_length=3000)
    include_identifiers: bool = False


class ExportPdfRequest(BaseModel):
    title: str = Field(default="Legal Document", max_length=300)
    content: str = Field(min_length=1, max_length=50000)
    document_type: str = Field(default="general_legal_draft", max_length=100)
    warning: str = Field(default="", max_length=600)


@asynccontextmanager
async def lifespan(_: FastAPI):
    sessions.init()
    knowledge.rebuild_index()
    indexed = case_corpus.index_imported_cases()
    corpus = judgment_corpus.status()
    log(
        "startup",
        sources=len(knowledge.approved_sources()), judgments=indexed,
        procedures=len(procedural.PROCEDURES), model_configured=llm_configured(),
        gnani_configured=bool(GNANI_KEY),
        embedded_corpus=corpus.get("available"), corpus_judgments=corpus.get("cases", 0),
        corpus_passages=corpus.get("chunks", 0),
    )
    if not corpus.get("available"):
        log(
            "corpus_unavailable",
            fix="build it with legal_pipeline/build_opennyai_embeddings.py",
            path=corpus.get("path", ""),
        )
    # The dense half of hybrid retrieval must exist by the time the first
    # caller speaks; building it does not need any API key, so a fresh
    # checkout self-heals instead of silently serving lexical-only recall.
    # Claimed synchronously, not inside the task: the task is scheduled after the
    # server starts accepting requests, and a turn arriving first used to load the
    # 530 MB artifact inside its own request (measured 8.6 s against ~100 ms).
    judgment_corpus.claim_load()
    _BACKGROUND_TASKS.add(asyncio.create_task(_prepare_dense()))
    _BACKGROUND_TASKS.add(asyncio.create_task(_prepare_corpus()))
    yield


_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


async def _prepare_corpus() -> None:
    """Load the embedded corpus and its encoder before the first turn.

    Never raises: a missing artifact leaves `case_corpus` answering from the
    curated partitions, and `/api/health` reports `embedded_judgment_corpus`.
    """
    try:
        result = await asyncio.to_thread(judgment_corpus.warm)
        log("corpus_ready", **{key: value for key, value in result.items() if key != "reason"}, reason=result.get("reason", ""))
    except Exception as exc:
        log("corpus_prepare_failed", error=type(exc).__name__, detail=str(exc)[:200])


async def _prepare_dense() -> None:
    """Build (or warm) the local dense index without blocking first speech.

    Never raises: the retrieval stage fails open to lexical BM25, and
    `/api/health` reports the honest state via `dense_retrieval`.
    """
    if os.getenv("DENSE_AUTOBUILD", "1").strip().lower() in {"0", "false", "off"}:
        return
    try:
        if dense.available():
            await asyncio.to_thread(dense.search, "warm the retrieval encoder", 1)
            log("dense_ready", index=dense.DB_PATH.name)
            return
        status = dense.backend_status()
        if not status.get("dependency_installed"):
            log("dense_unavailable", fix="pip install -r requirements-retrieval.txt", **{"model": str(status.get("model", ""))})
            return
        started = time.perf_counter()
        result = await asyncio.to_thread(dense.build_index)
        await asyncio.to_thread(dense.search, "warm the retrieval encoder", 1)
        log(
            "dense_built",
            built=result.get("built"), vectors=result.get("vectors"),
            ms=round((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:
        log("dense_prepare_failed", error=type(exc).__name__, detail=str(exc)[:200])


app = FastAPI(title="Nyaya — voice-first legal navigator", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(Exception)
async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Log an unhandled error *and* feed the failure-to-rule loop.

    The store in app/feedback.py is what turns repeated production errors
    into candidate rules or eval fixtures; this handler makes an unhandled
    exception a first-class input to that loop instead of a stack trace that
    disappears into a log file.
    """
    log(
        "unhandled_error",
        method=request.method, path=request.url.path,
        error=type(exc).__name__, detail=str(exc)[:300],
    )
    feedback.record(
        "unhandled_error",
        note=f"{type(exc).__name__}: {exc}"[:300],
        method=request.method, path=request.url.path,
    )
    return JSONResponse({"error": "This turn failed unexpectedly. Start a new session if it repeats."}, status_code=500)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), geolocation=(), microphone=(self)"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
        "connect-src 'self' ws: wss:; media-src 'self' blob:; object-src 'none'; "
        "manifest-src 'self'; worker-src 'self'; "
        "base-uri 'self'; frame-ancestors 'none'"
    )
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.api_route("/manifest.webmanifest", methods=["GET", "HEAD"], include_in_schema=False)
async def web_manifest():
    """Served from the root so the manifest's own `scope` is the whole app."""
    return FileResponse(STATIC / "manifest.webmanifest", media_type="application/manifest+json")


@app.api_route("/sw.js", methods=["GET", "HEAD"], include_in_schema=False)
async def service_worker():
    """A service worker at /static/ could only control /static/, so it is served
    from the root with an explicit scope allowance."""
    return FileResponse(
        STATIC / "sw.js",
        media_type="text/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
    )


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "mode": "voice-first-legal-navigator",
        "model_configured": llm_configured(),
        "grounded_only_mode": not llm_configured(),
        "time": int(time.time()),
        "source_count": len(knowledge.approved_sources()),
        "procedures": procedural.status(),
        "dense_retrieval": dense.available(),
        "sessions_backend": "sqlite",
        "safety_triage": True,
        "fact_ledger": True,
        "semantic_reading": llm_configured(),
        "barge_in": True,
        "workflows": {"total": len(workflow.WORKFLOWS), "ids": [item["id"] for item in workflow.WORKFLOWS]},
        "tools": [spec["function"]["name"] for spec in tool_layer.TOOL_SPECS],
        "pii_redaction": True,
        "llm_last_error": llm_last_error(),
        # One consultation call per turn, and what it cost: prompt/completion
        # tokens from the provider's own `usage`, the local minute window, and
        # its rate-limit headers. A 429 is diagnosable here instead of
        # inferring it from slow turns.
        "consultation": {"calls_per_turn": 1},
        "llm_usage": model_gateway.usage_snapshot(),
        **case_corpus.corpus_status(),
    }


@app.get("/api/config")
async def config():
    # The browser never receives provider keys.
    return {
        "product": "Nyaya",
        "mode": "route → curated procedure or retrieved authority → evidence-gated answer",
        "speech": {
            "stt": PRISMA_MODEL,
            "tts": TIMBRE_MODEL,
            "voice": TIMBRE_VOICE,
            "voice_catalog": speech.voices_for(TIMBRE_MODEL, "hi-IN"),
            "gnani_configured": bool(GNANI_KEY),
            "stt_languages": sorted(STT_LANGUAGES),
            "tts_languages": sorted(TTS_LANGUAGES),
            "stt_format": STT_FORMAT,
            "auto_detect": True,
            "language_question_required": False,
            "speech_languages": list(language_layer.BASE_LANGUAGES),
            "text_only_languages": list(language_layer.TEXT_ONLY_LANGUAGES),
            "codemixed_speech": PRISMA_MODEL == "prisma-v3.0",
            "capabilities": speech.capabilities(),
        },
        "language_policy": {
            "selection": "automatic",
            "order": ["caller_override", "speech_recognition", "script", "romanised_hindi", "reader", "default"],
            "override_available": True,
            "note": "The caller is never asked which language they speak. Prisma v3 reports the language of each utterance and the reply follows it.",
        },
        "voice_control": {
            "stop_speaking": voice.STOP_PLAN.as_dict(),
            "start_speaking": voice.START_PLAN.as_dict(),
            "truncates_interrupted_reply": True,
            "interruption_requires_server_decision": True,
        },
        "conversation_languages": sorted(TEXT_LANGUAGES),
        "language_labels": LANGUAGE_LABELS,
        # Sorted by the name a person reads in the picker, not by the ISO code.
        "jurisdictions": sorted(
            ({"code": code, "label": JURISDICTION_LABELS.get(code, code)} for code in INDIAN_JURISDICTIONS),
            key=lambda item: item["label"],
        ),
        "reasoning": {
            "mode": "deterministic routing + cascading retrieval + exact-quote validation",
            "model_configured": llm_configured(),
            "model": LLM_MODEL or None,
            "grounded_only_available": True,
            "legal_generation": "only with retrieved evidence; otherwise passage display with no conclusion",
        },
        "actions": {
            "review_provider_configured": bool(REVIEW_PROVIDER_URL),
            "external_actions_enabled": bool(REVIEW_PROVIDER_URL),
            "requires_explicit_user_request_and_consent": True,
        },
        "procedures": procedural.status(),
        "corpus": {
            "jurisdiction": "IN",
            "primary_sources": len(knowledge.approved_sources()),
            "secondary_qa": "viber1/indian-law-dataset (unverified discovery aid; never cited as law)",
            "dense_retrieval": dense.available(),
            **case_corpus.corpus_status(),
        },
        "consent_notice_version": sessions.CONSENT_NOTICE_VERSION,
    }


@app.get("/api/workflows")
async def workflows():
    return SERVICE_CATALOG


@app.get("/api/procedures")
async def procedures():
    return {"version": procedural.KB.get("version"), "procedures": procedural.list_public(), "policy": procedural.KB.get("policy", {})}


@app.get("/api/procedures/{procedure_id}")
async def procedure_detail(procedure_id: str):
    procedure = procedural.BY_ID.get(procedure_id)
    if not procedure:
        raise HTTPException(404, "Unknown procedure.")
    return procedural.public(procedure)


@app.post("/api/session")
async def start_session(payload: StartRequest, request: Request):
    if not payload.consent_confirmed:
        raise HTTPException(400, "Consent must be confirmed before starting a session.")
    # Every session creation is a row and a token, so an unmetered endpoint is a
    # cheap way to fill the store from one address. The per-turn budget below
    # cannot cover this: a caller who never sends a turn never reaches it.
    allowed, _ = sessions.LIMITER.allow(f"session-create:{_client(request)}", 60, 60)
    if not allowed:
        raise HTTPException(429, "Too many new conversations from this address. Try again in a minute.")
    # A language is never required to start: anything unrecognised becomes
    # auto, and the first utterance decides. Rejecting the call to demand a code
    # was the wrong door to put in front of a person with a legal problem.
    language = language_layer.normalize(payload.language)
    session_id, token = sessions.create(language, payload.consent_version)
    log("session_start", session_id=session_id[:6], language=language)
    return {
        "session_id": session_id,
        # Shown once, at creation. The client keeps it in memory and sends it
        # as a bearer token; it is never written into a URL, a log line or the
        # service-worker cache.
        "session_token": token,
        "language": language,
        "consent": {"given_at": time.time(), "notice_version": payload.consent_version},
        "retention": "local session store; expires after 4 hours or on deletion",
    }


def _client(request: Request) -> str:
    if TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return ""


def _authorize(session_id: str, consent: bool, request: Request, route_hint: str = "", token: str | None = None) -> dict[str, Any]:
    if not consent:
        raise HTTPException(400, "Consent must be confirmed before processing a turn.")
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session expired. Start a new session.")
    supplied = _bearer(request) if token is None else token
    if not sessions.verify_token(session_id, supplied):
        # The id alone is not a credential: it appears in paths, query strings
        # and logs. Without the token a leaked id reads nothing and does
        # nothing.
        raise HTTPException(401, "This conversation is not authorised. Start a new session.")
    allowed, remaining = sessions.LIMITER.allow(f"session:{session_id}", 20, 60)
    if not allowed:
        raise HTTPException(429, "This conversation reached its short-term limit. Wait a minute and continue.")
    allowed_ip, _ = sessions.LIMITER.allow(f"ip:{_client(request)}", 40, 60)
    if not allowed_ip:
        raise HTTPException(429, "Too many requests from this address in the last minute.")
    if not sessions.LIMITER.daily_allow(DAILY_TURN_BUDGET):
        raise HTTPException(429, "The server's daily request budget is exhausted. Try again tomorrow.")
    return session


def _require(session_id: str, request: Request) -> dict[str, Any]:
    """A session plus proof that the caller holds its token."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session expired. Start a new session.")
    if not sessions.verify_token(session_id, _bearer(request)):
        raise HTTPException(401, "This conversation is not authorised. Start a new session.")
    return session


def _finish(session: dict[str, Any], payload: TurnRequest, result: dict[str, Any], started: float, turn_id: str) -> dict[str, Any]:
    result["session_id"] = payload.session_id
    result["turn_id"] = turn_id
    result["latency_ms"] = round((time.perf_counter() - started) * 1000)
    sessions.record_turn(
        payload.session_id, payload.text, result.get("response", ""),
        result.get("route", {}).get("name", "unclear"), result.get("language", payload.language),
    )
    state = result.get("conversation_state") or {}
    route_meta = result.get("route") or {}
    previous = session.get("last_state") if isinstance(session.get("last_state"), dict) else {}
    sessions.save_state(
        payload.session_id,
        state.get("ledger") or {"facts": {}, "asked": [], "contradictions": []},
        {
            "route": route_meta.get("name"),
            "procedure_id": (result.get("procedure") or {}).get("id") or previous.get("procedure_id"),
            "risk_level": (result.get("safety") or {}).get("level"),
            "answer_status": result.get("answer_status"),
            "next_question": result.get("next_question"),
            "workflow": state.get("workflow"),
            # The model's own phrasing of this turn's legal question. Next turn
            # folds it into the search when the caller only points at the
            # matter ("what about the money?"), so the call's vocabulary
            # accumulates instead of resetting to four words.
            "search_query": result.get("search_query") or previous.get("search_query"),
            "at": round(time.time(), 3),
        },
    )
    log(
        "turn", turn_id=turn_id, session=payload.session_id[:6], route=route_meta.get("name"),
        mode=result.get("mode"), status=result.get("answer_status"), ms=result["latency_ms"],
        sources=len(result.get("sources", [])), claims=len(result.get("claims", [])),
        rejected=result.get("verification", {}).get("rejected_claims", 0), lang=result.get("language"),
        risk=(result.get("safety") or {}).get("level"), facts=len(state.get("known_facts") or []),
        conflicts=len(state.get("conflicts") or []),
    )
    # Failure-to-rule loop: a turn that had to fall back, or lost claims in
    # verification, is a production signal worth a candidate rule or eval
    # fixture — recorded only once per turn, redacted, bounded.
    if result.get("mode") == "grounded_only":
        feedback.record(
            "degraded_turn",
            note=str((result.get("source_gap") or "generation degraded to grounded-only"))[:200],
            route=route_meta.get("name"), status=result.get("answer_status"),
        )
    rejected = int((result.get("verification") or {}).get("rejected_claims") or 0)
    if rejected:
        feedback.record(
            "claim_rejected",
            note=f"{rejected} legal claim(s) removed by exact-quote verification",
            route=route_meta.get("name"), rejected=rejected,
        )
    for audited in result.get("tool_calls") or []:
        if isinstance(audited, dict) and audited.get("over_budget"):
            feedback.record(
                "tool_over_budget",
                note="tool exceeded its wall-clock budget",
                tool=audited.get("tool"), ms=audited.get("ms"),
            )
            break
    return result


@app.post("/api/turn")
async def turn(payload: TurnRequest, request: Request):
    started = time.perf_counter()
    turn_id = secrets.token_hex(6)
    session = _authorize(payload.session_id, payload.consent_confirmed, request)
    payload.language = language_layer.normalize(payload.language)
    if payload.service and payload.service not in {item["id"] for item in SERVICE_CATALOG["capabilities"]}:
        raise HTTPException(400, "Requested service is not in the service catalog.")
    result = await answer_turn(
        payload.text, payload.language, payload.jurisdiction, payload.service,
        session["history"], session=session,
    )
    return _finish(session, payload, result, started, turn_id)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/turn/stream")
async def turn_stream(payload: TurnRequest, request: Request):
    """Server-sent turn events so a voice reply can start before the turn ends."""
    started = time.perf_counter()
    turn_id = secrets.token_hex(6)

    session = _authorize(payload.session_id, payload.consent_confirmed, request)
    payload.language = language_layer.normalize(payload.language)
    if payload.service and payload.service not in {item["id"] for item in SERVICE_CATALOG["capabilities"]}:
        raise HTTPException(400, "Requested service is not in the service catalog.")

    async def events():
        yield _sse("open", {"turn_id": turn_id, "at": round(time.time(), 3)})
        try:
            async for kind, data in answer_stream(
                payload.text, payload.language, payload.jurisdiction, payload.service,
                session["history"], session=session,
            ):
                if await request.is_disconnected():
                    log("turn_cancelled", turn_id=turn_id)
                    return
                if kind == "final":
                    result = _finish(session, payload, data, started, turn_id)
                    yield _sse("final", result)
                else:
                    yield _sse(kind, data)
        except asyncio.CancelledError:
            log("turn_cancelled", turn_id=turn_id)
            raise
        except Exception as exc:  # never leave the client hanging without an event
            log("turn_error", turn_id=turn_id, error=type(exc).__name__)
            yield _sse("error", {"message": "The assistant could not complete this turn.", "kind": type(exc).__name__})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.get("/api/debug/analysis")
async def debug_analysis(text: str, request: Request, jurisdiction: str = "auto"):
    """Deterministic analysis only: no model, no generation. Useful for tests
    and for explaining why a turn was routed the way it was.

    It is unauthenticated by design (it reads nothing stored), but it does run
    retrieval, so it carries its own budget rather than being a free way to
    drive the index from outside.
    """
    allowed, _ = sessions.LIMITER.allow(f"debug:{_client(request)}", 60, 60)
    if not allowed:
        raise HTTPException(429, "Too many analysis requests from this address.")
    context = await analyze(text, "auto", jurisdiction)
    return {
        "route": context.route,
        "safety": {
            "level": context.risk.get("level"),
            "categories": context.risk.get("categories"),
            "policies": context.risk.get("policies"),
        },
        "jurisdiction": context.jurisdiction,
        "procedure": context.procedure["id"] if context.procedure else None,
        "sources": [source["id"] for source in context.sources],
        "cases": [case["id"] for case in context.cases],
        "facts": context.ledger.as_dict(),
        "query": retrieval.explain(text),
        "trace": context.trace,
    }


@app.post("/api/action/confirm")
async def confirm_action(payload: ActionRequest, request: Request):
    _require(payload.session_id, request)
    if not payload.user_confirmed:
        raise HTTPException(400, "This action requires an explicit user confirmation.")
    if payload.action not in {"save_artifact", "create_reminder"}:
        raise HTTPException(400, "Action is not enabled in this build.")
    payload_text = json.dumps(payload.payload, ensure_ascii=False)
    if len(payload_text) > 12000:
        raise HTTPException(413, "This local artifact is too large to save in the session.")
    if payload.action == "create_reminder":
        due_date = payload.payload.get("date")
        try:
            date.fromisoformat(str(due_date))
        except ValueError as exc:
            raise HTTPException(400, "A reminder needs a date in YYYY-MM-DD format.") from exc
        item = {
            "action": payload.action, "date": due_date, "note": str(payload.payload.get("note", ""))[:300],
            "created_at": time.time(), "delivery": "calendar_file",
        }
        sessions.add_reminder(payload.session_id, item)
        reminder_calendar = reminder_layer.to_ics(item)
    else:
        item = {"action": payload.action, "payload": payload.payload, "created_at": time.time(), "storage": "local_session_store"}
        sessions.add_artifact(payload.session_id, item)
    log("action", session=payload.session_id[:6], action=payload.action)
    response = {
        "accepted": True,
        "action": payload.action,
        "status": "saved_in_session_only",
        "message": "Saved in this session only. No external person or service was contacted.",
        "item": item,
    }
    if payload.action == "create_reminder":
        # Delivered as a file the caller saves, because there is no notification
        # service here and a reminder that is only stored is not a reminder.
        response["calendar"] = reminder_calendar
        response["calendar_url"] = f"/api/session/{payload.session_id}/reminders/0.ics"
        response["delivery"] = "Save the calendar file; your own phone will remind you. This is not a calculated legal deadline."
    return response


@app.get("/api/session/{session_id}/reminders/{index}.ics")
async def reminder_calendar_file(session_id: str, index: int, request: Request):
    """A saved reminder as a calendar file the caller actually receives."""
    session = _require(session_id, request)
    items = session.get("reminders") or []
    if not isinstance(items, list) or index < 0 or index >= len(items):
        raise HTTPException(404, "No such reminder.")
    try:
        body = reminder_layer.to_ics(items[index])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return Response(
        content=body,
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="reminder-{index}.ics"'},
    )


@app.get("/api/session/{session_id}/artifacts")
async def list_session_artifacts(session_id: str, request: Request):
    _require(session_id, request)
    data = sessions.artifacts(session_id)
    if data is None:
        raise HTTPException(404, "Session expired.")
    return data


@app.post("/api/export/pdf")
async def export_pdf(payload: ExportPdfRequest, request: Request):
    """Generate and download a publication-grade, court-standard A4 PDF."""
    allowed, _ = sessions.LIMITER.allow(f"pdf-export:{_client(request)}", 30, 60)
    if not allowed:
        raise HTTPException(429, "Too many PDF downloads from this address. Wait a minute and continue.")
    try:
        pdf_bytes = generate_legal_pdf(
            title=payload.title,
            content=payload.content,
            document_type=payload.document_type,
            warning=payload.warning or (
                "Statutory Notice: This draft is generated for assistance and reference. "
                "It must be reviewed, verified, and finalized in consultation with an enrolled advocate before filing in court."
            ),
        )
    except Exception as exc:
        log("pdf_export_failed", title=payload.title[:40], error=str(exc))
        raise HTTPException(500, f"Could not generate court PDF: {exc}") from exc

    safe_title = re.sub(r"[^a-zA-Z0-9_\-]+", "_", payload.title).strip("_")[:50] or "court_legal_document"
    filename = f"{safe_title}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-cache, no-store, must-revalidate",
        },
    )


@app.get("/api/session/{session_id}/artifact/{index}/pdf")
async def export_session_artifact_pdf(session_id: str, index: int, request: Request):
    """Download a previously saved session artifact as a publication-grade court PDF."""
    _require(session_id, request)
    artifacts_data = sessions.artifacts(session_id)
    if not artifacts_data or "artifacts" not in artifacts_data:
        raise HTTPException(404, "No artifacts found.")
    items = artifacts_data["artifacts"]
    if index < 0 or index >= len(items):
        raise HTTPException(404, "Artifact index out of range.")
    item = items[index]
    payload = item.get("payload", {})
    title = payload.get("title") or item.get("title") or "legal_draft"
    content = payload.get("content") or item.get("content") or ""
    doc_type = payload.get("type") or item.get("type") or "general_legal_draft"
    warning = payload.get("warning") or item.get("warning") or ""
    if not content:
        raise HTTPException(400, "Artifact has no text content to render.")
    try:
        pdf_bytes = generate_legal_pdf(
            title=title,
            content=content,
            document_type=doc_type,
            warning=warning,
        )
    except Exception as exc:
        raise HTTPException(500, f"Could not generate court PDF: {exc}") from exc

    safe_title = re.sub(r"[^a-zA-Z0-9_\-]+", "_", title).strip("_")[:50] or "court_legal_document"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_title}.pdf"',
            "Cache-Control": "no-cache, no-store, must-revalidate",
        },
    )


@app.get("/api/session/{session_id}/state")
async def session_state(session_id: str, request: Request):
    """What the assistant currently believes the caller said, and why.

    A caller can inspect and dispute this; nothing in the ledger is hidden
    from the person it describes.
    """
    _require(session_id, request)
    data = sessions.state(session_id)
    if data is None:
        raise HTTPException(404, "Session expired.")
    return data


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Delete a conversation and everything saved in it."""
    _require(session_id, request)
    allowed, _ = sessions.LIMITER.allow(f"delete:{_client(request)}", 10, 60)
    if not allowed:
        raise HTTPException(429, "Too many deletion requests.")
    sessions.delete(session_id)
    log("session_deleted", session=session_id[:6])
    return {"deleted": True, "session_id": session_id}


@app.post("/api/review/preview")
async def review_preview(payload: ReviewPreviewRequest, request: Request):
    """Exactly what would be shared, before anything is shared."""
    _require(payload.session_id, request)
    return privacy.share_preview(payload.summary, payload.include_identifiers)


@app.post("/api/review/request")
async def request_review(payload: ReviewRequest, request: Request):
    session = _require(payload.session_id, request)
    if not payload.user_requested:
        raise HTTPException(400, "A human connection can only be initiated after the user requests it.")
    if not payload.consent_to_share:
        raise HTTPException(400, "The user must consent to share the summary before a referral is sent.")
    # A person must be found in one language, so an auto session resolves to the
    # language the caller actually spoke in rather than refusing the request.
    requested_language = language_layer.normalize(payload.language)
    if requested_language in {"auto", language_layer.PRISMA_AUTO}:
        requested_language = language_layer.normalize(session.get("language"))
    if requested_language in {"auto", language_layer.PRISMA_AUTO}:
        raise HTTPException(400, "Tell me which language the person should speak, and I will only connect someone who speaks it.")
    payload.language = requested_language
    if not REVIEW_PROVIDER_URL:
        return JSONResponse({
            "connected": False,
            "status": "review_provider_not_configured",
            "language": payload.language,
            "message": "You asked for a person, but a live reviewer connection is not configured in this build. No case information was sent.",
            "fallback": next((s["url"] for s in knowledge.approved_sources() if s["id"] == "nalsa-faq-urgent-legal-assistance"), None),
        }, status_code=503)
    request_id = secrets.token_hex(8)
    # Only what the caller approved leaves the server: no session id, no
    # conversation history, and identity numbers masked unless the caller
    # explicitly chose to include them.
    preview = privacy.share_preview(payload.summary, payload.include_identifiers)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            headers = {"Authorization": f"Bearer {REVIEW_PROVIDER_TOKEN}"} if REVIEW_PROVIDER_TOKEN else {}
            response = await client.post(REVIEW_PROVIDER_URL, headers=headers, json={
                "request_id": request_id,
                "language": payload.language,
                "required_language": payload.language,
                "summary": preview["body"],
                "identifiers_masked": preview["masked"],
            })
            response.raise_for_status()
            result = response.json()
    except Exception as exc:
        log("review_failed", error=type(exc).__name__)
        raise HTTPException(502, f"Reviewer connection failed: {type(exc).__name__}") from exc
    language_matched = result.get("language") == payload.language or result.get("language_matched") is True
    connected = bool(result.get("connected")) and language_matched
    if not connected:
        return {
            "connected": False, "language": payload.language, "request_id": request_id,
            "identifiers_masked": preview["masked"],
            "message": "No qualified person in the requested language was confirmed. No connection was completed.",
        }
    return {
        "connected": True, "language": payload.language, "request_id": request_id,
        "identifiers_masked": preview["masked"],
    }


async def _proxy_ws(
    websocket: WebSocket,
    upstream_url: str,
    headers: dict[str, str],
    label: str,
    cancel: asyncio.Event | None = None,
    transform: Any = None,
):
    await websocket.accept()
    try:
        async with websockets.connect(upstream_url, additional_headers=headers, max_size=65536, ping_interval=20, open_timeout=10) as upstream:
            async def client_to_upstream():
                while True:
                    item = await websocket.receive()
                    if item.get("type") == "websocket.disconnect":
                        return
                    if item.get("bytes") is not None:
                        await upstream.send(item["bytes"])
                    elif item.get("text") is not None:
                        if len(item["text"].encode("utf-8")) > 60000:
                            await websocket.close(code=1009, reason="Speech request is too large")
                            return
                        payload = transform(item["text"]) if transform is not None else item["text"]
                        if payload is not None:
                            await upstream.send(payload)

            async def upstream_to_client():
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)

            tasks = [asyncio.create_task(client_to_upstream()), asyncio.create_task(upstream_to_client())]
            if cancel is not None:
                async def stop_on_barge_in():
                    # Barge-in stops synthesis at the source: leaving Timbre
                    # generating audio nobody hears wastes the caller's latency
                    # budget and can bleed buffered speech after the stop.
                    await cancel.wait()
                    with contextlib.suppress(Exception):
                        await upstream.close()
                    with contextlib.suppress(Exception):
                        await websocket.close(code=1000, reason="interrupted")

                tasks.append(asyncio.create_task(stop_on_barge_in()))
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if task.exception():
                    raise task.exception()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log("ws_error", label=label, error=type(exc).__name__)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": f"{label} unavailable: {type(exc).__name__}"}))
            await websocket.close(code=1011)
        except Exception:
            pass


def _valid_websocket_origin(websocket: WebSocket) -> bool:
    origin = websocket.headers.get("origin")
    host = websocket.headers.get("host")
    if not origin:
        # Non-browser clients may omit Origin, but they must not get the
        # operator's provider key for free: require a loopback peer.
        client = websocket.client.host if websocket.client else ""
        return client in {"127.0.0.1", "::1", "localhost", "testclient"}
    parsed = urlsplit(origin)
    if not host or parsed.netloc != host:
        return False
    return websocket.url.scheme == "wss" or parsed.scheme in {"http", "https"}


class SpeechStartRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=80)
    text: str = Field(min_length=1, max_length=6000)


class InterruptRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=80)
    speech_id: str | None = Field(default=None, max_length=40)
    partial_text: str = Field(default="", max_length=600)
    vad_ms: int = Field(default=0, ge=0, le=60000)
    heard_ms: int = Field(default=0, ge=0, le=600000)
    reason: str = Field(default="caller_spoke", max_length=40)


class SpeechMetricRequest(BaseModel):
    speech_id: str = Field(min_length=1, max_length=40)
    first_audio_ms: float | None = Field(default=None, ge=0, le=120000)
    stopped_ms: float | None = Field(default=None, ge=0, le=600000)


@app.post("/api/voice/speech")
async def begin_speech(payload: SpeechStartRequest, request: Request):
    """Register an utterance before synthesis starts.

    The client needs this before it can be interrupted: the server has to know
    which text is playing, so a barge-in can truncate the *record* and stop the
    upstream synthesis stream rather than only muting the speaker.
    """
    _require(payload.session_id, request)
    turn = voice.REGISTRY.begin(payload.session_id, payload.text)
    return {
        "speech_id": turn.speech_id,
        "speak_after_ms": int(voice.START_PLAN.wait_seconds * 1000),
        "backoff_ms": voice.resume_after_ms(voice.STOP_PLAN),
        "stop_speaking": voice.STOP_PLAN.as_dict(),
        "start_speaking": voice.START_PLAN.as_dict(),
    }


@app.post("/api/voice/interrupt")
async def interrupt_speech(payload: InterruptRequest, request: Request):
    """Decide and apply a barge-in.

    The decision is server-side so the same thresholds govern every client and
    every language; the client only reports what it observed (VAD and partial
    transcript). A confirmed VAD onset with `caller_spoke` stops playback even
    with no words yet, which is what makes the stop feel instant.
    """
    started = time.perf_counter()
    _require(payload.session_id, request)
    current = voice.REGISTRY.current(payload.session_id)
    if payload.speech_id and current and current.speech_id != payload.speech_id:
        return {
            "stop": False, "reason": "stale_speech", "speech_id": current.speech_id,
            "resume_after_ms": voice.resume_after_ms(voice.STOP_PLAN),
        }
    decision = voice.decide_interruption(
        payload.partial_text, vad_ms=payload.vad_ms, plan=voice.STOP_PLAN,
    )
    # A confirmed VAD onset stops playback even before any word exists, but a
    # transcript that is only a backchannel still wins: "haan" is listening,
    # not taking the floor. Explicit user actions always stop.
    forced = payload.reason in {"button", "superseded"} or (
        payload.reason == "caller_spoke" and not payload.partial_text.strip()
    )
    stopped = decision.stop or forced
    stop_reason = decision.reason if decision.stop else payload.reason
    turn = voice.REGISTRY.interrupt(
        payload.session_id, heard_ms=payload.heard_ms, reason=stop_reason,
    ) if stopped else None
    voice.METRICS.record("interrupt_decision_ms", (time.perf_counter() - started) * 1000)
    log(
        "interrupt", session=payload.session_id[:6], stop=stopped,
        reason=decision.reason, ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return {
        "stop": stopped,
        "reason": stop_reason if stopped else decision.reason,
        "matched": decision.matched,
        "speech": turn.as_dict() if turn else None,
        "resume_after_ms": voice.resume_after_ms(voice.STOP_PLAN),
    }


@app.post("/api/voice/metrics")
async def report_speech_metrics(payload: SpeechMetricRequest):
    """Client-reported speech timings, aggregated for the health panel."""
    if payload.first_audio_ms is not None:
        voice.METRICS.record("first_audio_ms", payload.first_audio_ms)
    if payload.stopped_ms is not None:
        voice.METRICS.record("barge_in_stop_ms", payload.stopped_ms)
    return {"ok": True}


@app.get("/api/voice/metrics")
async def speech_metrics():
    return {
        "stages": voice.METRICS.summary(),
        "plans": {
            "stop_speaking": voice.STOP_PLAN.as_dict(),
            "start_speaking": voice.START_PLAN.as_dict(),
        },
        "note": "First-audio and barge-in samples come from the browser; interrupt_decision_ms is server-side.",
    }


def _stt_settings(websocket: WebSocket, session: dict[str, Any] | None) -> speech.SttSettings:
    """Everything Prisma needs for one call, in one place.

    `language=auto` is the default: v3 in `indic-auto` mode identifies the
    language of every utterance and returns it, so the caller is never asked.
    """
    override = websocket.query_params.get("language", "auto")
    try:
        sample_rate = int(websocket.query_params.get("sample_rate", "48000"))
    except ValueError:
        sample_rate = 48000
    language_code, candidates = language_layer.stt_language(override)
    history = (session or {}).get("history") or []
    # Prior turns are context, not a transcript: `previous_text` improves names,
    # amounts and legal terms across utterances without re-asking anything.
    previous = " ".join(
        str(item.get("content", ""))
        for item in history[-4:]
        if isinstance(item, dict) and item.get("role") == "user"
    )[-600:]
    return speech.SttSettings(
        language_code=language_code,
        candidates=candidates,
        model=PRISMA_MODEL,
        format=STT_FORMAT,
        sample_rate=sample_rate,
        vad_threshold=PRISMA_VAD_THRESHOLD,
        min_silence_ms=PRISMA_MIN_SILENCE_MS,
        keyterms=list(speech.LEGAL_KEYTERMS),
        topic="Indian legal information: police complaint, court procedure, legal aid, documents",
        previous_text=previous,
    )


async def _stt_pump_down(websocket: WebSocket, upstream: Any, buffer: speech.FrameBuffer) -> None:
    """Caller audio in, re-cut to Prisma's fixed 1,024-byte frames."""
    while True:
        item = await websocket.receive()
        if item.get("type") == "websocket.disconnect":
            # Pad out the trailing partial frame so the last word is not clipped.
            for frame in buffer.flush():
                with contextlib.suppress(Exception):
                    await upstream.send(frame)
            return
        data = item.get("bytes")
        if data:
            for frame in buffer.push(data):
                await upstream.send(frame)
        elif item.get("text"):
            # `commit` and `update_settings` control frames pass through as-is.
            await upstream.send(item["text"])


async def _stt_pump_up(websocket: WebSocket, upstream: Any, heard: list[str]) -> None:
    """Upstream messages out, with transcript frames normalised.

    Normalising here is what lets the browser stop guessing: every transcript
    carries `is_final: true` (Prisma realtime has no interim transcripts) and, in
    v3 auto mode, the language the provider actually heard.
    """
    async for message in upstream:
        if isinstance(message, bytes):
            await websocket.send_bytes(message)
            continue
        payload: Any = None
        with contextlib.suppress(json.JSONDecodeError):
            payload = json.loads(message)
        transcript = speech.transcript_of(payload) if isinstance(payload, dict) else None
        if transcript:
            heard.append(transcript["text"])
            await websocket.send_text(json.dumps(transcript, ensure_ascii=False))
        else:
            await websocket.send_text(message)


# Prisma closes a stream after fifteen minutes. Rotating a little before the
# cap means the seam happens on our terms, at a frame boundary, instead of
# wherever the provider decides to hang up.
STT_ROTATE_SECONDS = max(60.0, min(840.0, float(os.getenv("STT_ROTATE_SECONDS", "840"))))


async def _stt_rotate_after(seconds: float) -> None:
    await asyncio.sleep(seconds)


async def _proxy_stt(websocket: WebSocket, settings: speech.SttSettings, api_key: str) -> None:
    """Proxy a live microphone into Prisma, surviving the 15-minute cap.

    A session with a lawyer runs long. Prisma closes a stream after fifteen
    minutes by design, so the proxy re-dials and continues instead of letting
    the microphone go quietly dead mid-sentence; the last transcript is carried
    into `previous_text` so the new stream keeps its context.

    The re-dial is also scheduled, not only reactive: at `STT_ROTATE_SECONDS`
    the proxy stops its own pumps while the frame buffer is byte-aligned, pads
    any tail into one complete frame, and opens a new stream. A controlled
    seam loses at most the in-flight frame; the provider's own cut used to land
    in the middle of a word with no warning.
    """
    await websocket.accept()
    buffer = speech.FrameBuffer()
    heard: list[str] = []
    attempt = 0
    rotations = 0
    while attempt < 3:
        attempt += 1
        if heard:
            settings.previous_text = " ".join(heard)[-600:]
        try:
            async with websockets.connect(
                speech.PRISMA_STREAM_URL,
                additional_headers=settings.headers(api_key),
                max_size=65536, ping_interval=20, open_timeout=10,
            ) as upstream:
                down = asyncio.create_task(_stt_pump_down(websocket, upstream, buffer))
                up = asyncio.create_task(_stt_pump_up(websocket, upstream, heard))
                rotate = asyncio.create_task(_stt_rotate_after(STT_ROTATE_SECONDS))
                done, pending = await asyncio.wait(
                    {down, up, rotate}, return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                if rotate in done:
                    align_started = time.perf_counter()
                    tail = buffer.flush()
                    for frame in tail:
                        with contextlib.suppress(Exception):
                            await upstream.send(frame)
                    rotations += 1
                    # A rotation is not a failure: it must not consume one of
                    # the three attempts a real error is allowed.
                    attempt -= 1
                    log(
                        "stt_rotating", rotations=rotations, aligned_ms=round((time.perf_counter() - align_started) * 1000, 2),
                        padded_frames=len(tail), heard=len(heard),
                    )
                    continue
                for task in done:
                    error = task.exception()
                    if error is None:
                        continue
                    if isinstance(error, WebSocketDisconnect):
                        return
                    if type(error).__name__.startswith("ConnectionClosed"):
                        continue
                    raise error
                if down in done:
                    # The caller released the microphone: the call is over.
                    return
                log("stt_reconnect", attempt=attempt, reason="upstream_closed", heard=len(heard))
        except WebSocketDisconnect:
            return
        except Exception as exc:
            log("stt_error", error=type(exc).__name__, attempt=attempt)
            if attempt == 3:
                with contextlib.suppress(Exception):
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Speech recognition was interrupted. Tap the microphone to continue, or type your message.",
                    }))
                    await websocket.close(code=1011)
                return
    # Three failed re-dials in a row. Say so instead of letting the microphone
    # go quiet.
    with contextlib.suppress(Exception):
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "This call has been listening for a long time. Tap the microphone to continue.",
        }))
        await websocket.close(code=1011)


@app.websocket("/api/voice/prisma")
async def prisma_stream(websocket: WebSocket):
    if not GNANI_KEY:
        await websocket.accept()
        await websocket.send_text(json.dumps({"type": "error", "message": "Gnani credentials are not configured. Set GNANI_API_KEY_ID on the server."}))
        await websocket.close(code=1011)
        return
    if not _valid_websocket_origin(websocket):
        await websocket.close(code=1008, reason="Cross-origin websocket connection rejected")
        return
    # A microphone opened before consent cannot have a session yet, so this
    # socket cannot always require a token. It does get an address budget: an
    # unauthenticated socket that streams to a paid recogniser is otherwise a
    # free transcription service for anyone who can reach the port.
    address = websocket.client.host if websocket.client else "unknown"
    allowed, _ = sessions.LIMITER.allow(f"stt:{address}", 12, 60)
    if not allowed:
        await websocket.close(code=1013, reason="Too many speech connections")
        return
    session_id = websocket.query_params.get("session_id", "")
    session = sessions.get(session_id) if session_id else None
    if session and not sessions.verify_token(session_id, websocket.query_params.get("token", "")):
        # A browser socket cannot send an Authorization header, so the token
        # rides in the query string for this upgrade only.
        await websocket.close(code=1008, reason="Session token rejected")
        return
    settings = _stt_settings(websocket, session)
    log(
        "stt_opened", language=settings.language_code, model=settings.model,
        format=settings.format, sample_rate=settings.sample_rate,
        min_silence_ms=settings.min_silence_ms,
    )
    await _proxy_stt(websocket, settings, GNANI_KEY)


def _timbre_transform(default_language: str):
    """Rewrite the browser's synthesis request into a valid Timbre request.

    The browser does not know the voice catalogue, and a mis-paired voice or an
    unsupported language is a 400 that lands after the caller has been promised
    an answer. Model, voice, language and audio format are therefore decided
    here: `auto` for mixed content, the caller's language otherwise, and a
    catalogue-checked voice for the model actually in use.
    """

    def transform(raw: str) -> str | None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        language = language_layer.tts_language(str(payload.get("language") or default_language or "auto"))
        if language == "unavailable":
            # The catalogue has no voice for this language. Voicing it anyway
            # reads the caller's own sentence back in a different language;
            # dropping the request leaves the written answer standing.
            log("tts_unavailable", requested=str(payload.get("language")))
            return None
        body, choice = speech.timbre_body(
            payload, model=TIMBRE_MODEL, voice=TIMBRE_VOICE,
            language=language, sample_rate=TTS_SAMPLE_RATE,
        )
        if not body["text"]:
            return None
        if choice.substituted:
            log("tts_voice_substituted", detail=choice.substituted)
        log("tts_request", language=choice.language, voice=choice.voice, model=choice.model, chars=len(body["text"]))
        return json.dumps(body, ensure_ascii=False)

    return transform


@app.websocket("/api/voice/timbre")
async def timbre_stream(websocket: WebSocket):
    if not GNANI_KEY:
        await websocket.accept()
        await websocket.send_text(json.dumps({"type": "error", "message": "Gnani credentials are not configured. Set GNANI_API_KEY_ID on the server."}))
        await websocket.close(code=1011)
        return
    if not _valid_websocket_origin(websocket):
        await websocket.close(code=1008, reason="Cross-origin websocket connection rejected")
        return
    headers = {"X-API-Key-ID": GNANI_KEY}
    # Bind this synthesis stream to the speech turn the caller is hearing, so a
    # barge-in can stop it server-side instead of only muting the browser.
    cancel = None
    session_id = websocket.query_params.get("session_id", "")
    speech_id = websocket.query_params.get("speech_id", "")
    if session_id and not sessions.verify_token(session_id, websocket.query_params.get("token", "")):
        await websocket.close(code=1008, reason="Session token rejected")
        return
    if session_id and speech_id:
        # Local when this worker owns the utterance, a watched row when the
        # barge-in lands on another one.
        cancel = voice.REGISTRY.cancel_event(session_id, speech_id)
    await _proxy_ws(
        websocket, speech.TIMBRE_STREAM_URL, headers, "Timbre TTS",
        cancel=cancel, transform=_timbre_transform(websocket.query_params.get("language", "auto")),
    )


@app.get("/api/sources")
async def list_sources():
    return [knowledge.source_evidence(source) for source in knowledge.approved_sources()]


@app.post("/api/admin/sources/reload")
async def reload_sources(request: Request):
    if not ADMIN_TOKEN:
        raise HTTPException(503, "Source reload requires ADMIN_TOKEN to be configured on the server.")
    if request.headers.get("authorization") != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(401, "Administrator authorization required.")
    previous = knowledge.SOURCES
    try:
        count = knowledge.reload_source_pack()
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        knowledge.SOURCES = previous
        knowledge.rebuild_index()
        raise HTTPException(400, f"Source pack validation failed: {exc}") from exc
    log("sources_reloaded", count=count)
    return {"reloaded": True, "approved_source_count": count, "note": "Only records marked approved are indexed. Editing a record to approved still requires an independent legal review process."}
