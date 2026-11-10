"""Provider-neutral multilingual LLM gateway.

Gnani Evon can be served locally with vLLM/SGLang or through an OpenAI-compatible
endpoint. The gateway never receives tools that can contact people, file forms,
or take irreversible actions. It returns a structured proposal which the legal
evidence and action layers validate before the UI can expose it.
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
API_KEY = os.getenv("LLM_API_KEY", "") or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "")
if not BASE_URL and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "gemini" in MODEL.lower()):
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
TIMEOUT_SECONDS = max(2.0, min(45.0, float(os.getenv("LLM_TIMEOUT_SECONDS", "12"))))
# Whole-turn ceiling across retries. A voice turn that takes longer than this
# has already failed the caller, so it is better to fall back to passages.
BUDGET_SECONDS = max(3.0, min(90.0, float(os.getenv("LLM_BUDGET_SECONDS", "18"))))
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
PROMPT_PATH = ROOT / "prompts" / "legal_agent.md"

# `auto` asks for a JSON Schema (guided decoding on vLLM/SGLang) and falls
# back to json_object if the server rejects it. Set STRUCTURED_MODE=json_object
# to skip the probe entirely on a server that does not support schemas.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?\u0964])\s+")
STRUCTURED_MODE = os.getenv("LLM_STRUCTURED_MODE", "auto").strip().lower()
_SCHEMA_REJECTED = False
# Probed once per process. Paying a failed tool call on every turn was a
# hidden full-generation cost on servers that reject `tools`.
_TOOLS_UNSUPPORTED = os.getenv("LLM_DISABLE_NATIVE_TOOLS", "1" if "groq.com" in BASE_URL.lower() else "0") == "1"


# ---------------------------------------------------------------------------
# Provider budget: tokens per minute, not per request
# ---------------------------------------------------------------------------
# A 429 is not an outage, it is a quota window, and it is organisation-wide. Two
# calls on every turn met that window twice as fast as one, which is how a
# single throttled question turned into a 37-second turn that degraded anyway.
# The unified consultation makes one call; this window stops a burst -- several
# callers, a retry, a long context -- from walking into the same wall.
#
# `LLM_TPM_BUDGET` is the provider's published limit for this model. It cannot
# be guessed from the model name, so an unset value disables local throttling
# and leaves the provider's 429 and `Retry-After` as the only signal.
TPM_BUDGET = max(0, int(os.getenv("LLM_TPM_BUDGET", "0") or 0))
_TPM_WINDOW: "collections.deque[tuple[float, int]]" = collections.deque()
_LAST_USAGE: dict[str, int] = {}
_LAST_RATE_LIMIT: dict[str, str] = {}


def _record_usage(response: httpx.Response, prompt_chars: int) -> None:
    """Count what the provider actually charged, not what we estimated."""
    try:
        body = response.json()
    except ValueError:
        return
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    _LAST_USAGE.clear()
    _LAST_USAGE.update({
        "prompt_tokens": prompt, "completion_tokens": completion,
        "total_tokens": total, "at": int(time.time()),
    })
    if total:
        _TPM_WINDOW.append((time.monotonic(), total))
    elif prompt_chars:
        # A server that omits `usage` still spent something; a character-based
        # estimate keeps the window honest instead of empty.
        _TPM_WINDOW.append((time.monotonic(), max(1, prompt_chars // 4)))
    _trim_window()


def _record_rate_headers(response: httpx.Response) -> None:
    """Keep the provider's own numbers for health and diagnosis."""
    for header in (
        "retry-after", "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens",
        "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
    ):
        value = response.headers.get(header)
        if value:
            _LAST_RATE_LIMIT[header] = value[:40]


def _trim_window(now: float | None = None) -> None:
    now = now if now is not None else time.monotonic()
    while _TPM_WINDOW and now - _TPM_WINDOW[0][0] >= 60.0:
        _TPM_WINDOW.popleft()


def _spent_this_minute() -> int:
    _trim_window()
    return sum(tokens for _, tokens in _TPM_WINDOW)


def usage_snapshot() -> dict[str, Any]:
    """What the last call cost, and what this minute has spent so far."""
    return {
        "last": dict(_LAST_USAGE),
        "minute_tokens": _spent_this_minute(),
        "tpm_budget": TPM_BUDGET or None,
        "rate_limit_headers": dict(_LAST_RATE_LIMIT),
    }


async def _respect_tpm(estimate: int, deadline: float) -> None:
    """Wait for room in the provider's minute window instead of collecting 429.

    The wait is bounded by the turn's own deadline: a caller would rather hear
    the grounded answer now than silence while a quota window refills.
    """
    if TPM_BUDGET <= 0:
        return
    _trim_window()
    spent = _spent_this_minute()
    if spent + estimate <= TPM_BUDGET or not _TPM_WINDOW:
        return
    now = time.monotonic()
    # When the oldest entry ages out, `estimate` tokens have room again.
    wait = max(0.0, 60.0 - (now - _TPM_WINDOW[0][0]))
    remaining = deadline - asyncio.get_running_loop().time()
    await asyncio.sleep(min(wait, max(0.0, remaining - 0.4)))


# One keep-alive client per event loop. A voice turn makes one to three model
# calls; paying a fresh TCP and TLS handshake on each one is pure latency.
_CLIENT: httpx.AsyncClient | None = None
_CLIENT_LOOP: Any = None


def _client() -> httpx.AsyncClient:
    global _CLIENT, _CLIENT_LOOP
    loop = asyncio.get_running_loop()
    if _CLIENT is None or _CLIENT.is_closed or _CLIENT_LOOP is not loop:
        _CLIENT = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=32, max_connections=64),
            timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=min(2.0, TIMEOUT_SECONDS)),
        )
        _CLIENT_LOOP = loop
    return _CLIENT


def is_gemini_live() -> bool:
    return "gemini-3.8-live" in MODEL.lower() or "gemini-live" in MODEL.lower()


def configured() -> bool:
    if is_gemini_live():
        return bool(API_KEY)
    if not (BASE_URL and MODEL):
        return False
    if ("googleapis.com" in BASE_URL or "groq.com" in BASE_URL or "api.openai.com" in BASE_URL) and not API_KEY:
        return False
    return True


def tool_calling_available() -> bool:
    return not _TOOLS_UNSUPPORTED


tools_supported = tool_calling_available


def _sentence_spans(text: str, consumed: int) -> tuple[list[str], int]:
    """Sentences completed after `consumed` characters of a growing string."""
    chunk = text[consumed:]
    parts = SENTENCE_SPLIT.split(chunk)
    if len(parts) <= 1:
        return [], consumed
    return parts[:-1], consumed + len(chunk) - len(parts[-1])


def partial_reply(buffer: str) -> str | None:
    """Extract the `reply` string value from a JSON document still arriving.

    Generation streams JSON, so the prose is inside a string with escapes.
    Reading it incrementally lets the voice start on sentence one instead of
    waiting for the object to close.
    """
    key = buffer.find('"reply"')
    if key < 0:
        return None
    colon = buffer.find(":", key + 7)
    if colon < 0:
        return None
    start = buffer.find('"', colon + 1)
    if start < 0:
        return None
    out: list[str] = []
    index = start + 1
    while index < len(buffer):
        char = buffer[index]
        if char == "\\":
            if index + 1 >= len(buffer):
                break
            escaped = buffer[index + 1]
            if escaped == "u":
                if index + 6 > len(buffer):
                    break
                try:
                    out.append(chr(int(buffer[index + 2:index + 6], 16)))
                except ValueError:
                    break
                index += 6
                continue
            out.append({"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}.get(escaped, escaped))
            index += 2
            continue
        if char == '"':
            break
        out.append(char)
        index += 1
    return "".join(out)


ENTAILMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdicts"],
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "supported"],
                "properties": {
                    "index": {"type": "integer"},
                    "supported": {"type": "boolean"},
                },
            },
        },
    },
}

# An exact substring proves the words came from the source. It does not prove
# the source says what the claim says: an adjacent paragraph, a dissent, or a
# provision about someone else quotes just as exactly. This check asks the
# small model the entailment question and nothing else, so a claim that only
# survived on string matching has a second, independent chance to fail.
ENTAILMENT_PROMPT = (
    "You verify legal quotations. Each item has a CLAIM and a QUOTE taken from an "
    "official source. Decide one thing: does the QUOTE, on its own, state the CLAIM? "
    "Answer true only when the quotation itself carries the proposition. Answer false "
    "when the quote is from a different provision, a different party, a different "
    "remedy or a different stage, or when the quote would need outside law to make the "
    "claim. Do not use outside knowledge. Do not repair or soften the claim. "
    "Answer JSON only, one verdict per item, in the order received."
)


# Retrieval returns passages that share the question's words. Whether a
# passage is about the caller's *question* is a separate judgement, and it is
# the one a human lawyer makes before citing anything: "this judgment mentions
# s.154" is not "this judgment decides when an FIR must be registered". This
# check costs one small call over the top few passages, and its answer can only
# remove candidates, never add or rewrite one.
CASE_JUDGE_PROMPT = (
    "You decide whether retrieved judgments address a caller's legal question, "
    "the way a junior advocate would check a bundle before the senior cites from it. "
    "Each item has a QUESTION and a PASSAGE from a judgment. "
    "Score `relevance` from 0 to 3: "
    "3 = the passage decides or squarely discusses the question the caller asked; "
    "2 = it discusses the same legal question but on different facts, or states the "
    "governing rule in passing; "
    "1 = it is about the same subject area but does not answer this question; "
    "0 = different issue, different party's dispute, or an incidental use of a word. "
    "Set `addresses` true only for 2 and 3. "
    "Sharing words is not enough: a passage about the refusal of a tax registration "
    "does not address a refused FIR, and a citation of a section for another purpose "
    "does not address it. Use only the passage given; do not use outside knowledge. "
    "Answer JSON only, one verdict per item, in the order received."
)

CASE_JUDGE_SCHEMA = {
    "name": "case_relevance",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdicts"],
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["index", "addresses", "relevance"],
                    "properties": {
                        "index": {"type": "integer"},
                        "addresses": {"type": "boolean"},
                        "relevance": {"type": "integer"},
                    },
                },
            },
        },
    },
}


def case_judge_available() -> bool:
    if os.getenv("LLM_CASE_JUDGE", "auto").strip().lower() in {"off", "0", "false", "no"}:
        return False
    return configured()


async def case_relevance(
    question: str,
    candidates: list[dict[str, Any]],
    timeout_cap: float | None = None,
) -> dict[int, int] | None:
    """Score how squarely each retrieved judgment addresses the question.

    Returns index -> 0..3, or None when the check did not run. `None` is not a
    verdict: an unavailable judge is not evidence that a judgment is irrelevant, 
    and a returned map only contains answers the model actually gave. Anything
    it skipped stays in the result, unjudged.

    Scores rather than a boolean because the caller has to *order* four similar
    judgments, not just delete three. A binary keep/drop over retrieval order
    left the best-reasoned decision wherever the lexical ranker happened to put
    it; a 0-3 score is the same call with the information the ordering needs.
    """
    if not candidates or not case_judge_available():
        return None
    items = [
        {
            "index": index,
            "question": " ".join(str(question or "").split())[:600],
            "passage": " ".join(str(candidate.get("excerpt") or "").split())[:1200],
        }
        for index, candidate in enumerate(candidates[:8])
    ]
    answer = await _json_completion(
        CASE_JUDGE_PROMPT,
        [],
        {"items": items},
        # One JSON object of eight short verdicts. 320 tokens cut the answer off
        # mid-list, and a truncated verdict list is read as "no view" -- which
        # silently kept every candidate and did the work for nothing.
        max_tokens=int(os.getenv("LLM_CASE_JUDGE_MAX_TOKENS", "700")),
        temperature=0.0,
        schema=CASE_JUDGE_SCHEMA,
        timeout_cap=timeout_cap or float(os.getenv("LLM_CASE_JUDGE_TIMEOUT_SECONDS", "4")),
        model=os.getenv("LLM_CASE_JUDGE_MODEL", "") or None,
    )
    if not isinstance(answer, dict):
        return None
    raw = answer.get("verdicts")
    if not isinstance(raw, list):
        return None
    scores: dict[int, int] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if not 0 <= index < len(items):
            continue
        try:
            value = int(item.get("relevance"))
        except (TypeError, ValueError):
            # A server that ignored the new field still gives the boolean, and
            # "addresses" is a 2, not a 3: it answered the question, it did not
            # necessarily decide it.
            value = 2 if item.get("addresses") is True else 0
        scores[index] = max(0, min(3, value))
    return scores or None


async def judge_cases(
    question: str,
    candidates: list[dict[str, Any]],
    timeout_cap: float | None = None,
) -> dict[int, bool] | None:
    """Boolean view of `case_relevance`, kept for callers that only filter."""
    scores = await case_relevance(question, candidates, timeout_cap)
    if scores is None:
        return None
    return {index: score >= 2 for index, score in scores.items()}


def entailment_mode() -> str:
    """off | on | auto. `auto` runs when a model is configured."""
    mode = os.getenv("LLM_ENTAILMENT", "auto").strip().lower()
    if mode in {"off", "0", "false", "no"}:
        return "off"
    if mode in {"on", "1", "true", "yes"}:
        return "on"
    return "auto"


def entailment_available() -> bool:
    mode = entailment_mode()
    return mode != "off" and (configured() or mode == "on")


async def verify_claims(
    claims: list[dict[str, str]],
    timeout_cap: float | None = None,
) -> dict[int, bool] | None:
    """Ask whether each quote states its claim. `None` means it did not run.

    A verdict is never inferred from a failure: if the small model is slow or
    down the claims keep their substring pass and are reported as not reviewed,
    rather than being accepted as if they had been checked.
    """
    if not claims or not entailment_available():
        return None
    items = [
        {"index": index, "claim": claim.get("text", ""), "quote": claim.get("quote", "")}
        for index, claim in enumerate(claims[:12])
    ]
    answer = await _json_completion(
        ENTAILMENT_PROMPT,
        [],
        {"items": items},
        max_tokens=int(os.getenv("LLM_ENTAILMENT_MAX_TOKENS", "400")),
        temperature=0.0,
        schema=ENTAILMENT_SCHEMA,
        timeout_cap=timeout_cap or float(os.getenv("LLM_ENTAILMENT_TIMEOUT_SECONDS", "4")),
        model=os.getenv("LLM_ENTAILMENT_MODEL", "") or None,
    )
    if not isinstance(answer, dict):
        return None
    raw = answer.get("verdicts")
    if not isinstance(raw, list):
        return None
    verdicts: dict[int, bool] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(claims):
            verdicts[index] = item.get("supported") is True
    # Items the checker skipped stay accepted but unverified, so a partial
    # answer cannot quietly delete a correct claim.
    return verdicts or None


def endpoint() -> str:
    url = BASE_URL
    if not url:
        if "gemini" in MODEL.lower() or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        return "/v1/chat/completions"
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1") or url.endswith("/openai"):
        return url + "/chat/completions"
    if "googleapis.com" in url:
        return url.rstrip("/") + "/chat/completions"
    return url + "/v1/chat/completions"


def _auth_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
        if "googleapis.com" in endpoint():
            headers["x-goog-api-key"] = API_KEY
    return headers


def _system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _parse_message(content: Any) -> dict[str, Any] | None:
    if isinstance(content, list):
        content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    if not isinstance(content, str):
        return None
    raw = content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _gemini_live_completion(
    context: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
    extra_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Execute a consultation turn directly via Gemini 3.8 Live bidirectional WebSocket."""
    if not API_KEY:
        return None
    try:
        import websockets
    except ImportError:
        return None

    ws_url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={API_KEY}"
    func_decls = [
        {
            "name": "submit_consultation_reading",
            "description": "Submit structured legal consultation reading, extracted facts, and verified claims.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "reply": {"type": "STRING", "description": "Spoken conversational reply in caller language."},
                    "language": {"type": "STRING"},
                    "intent": {"type": "STRING", "enum": ["procedural", "statute", "precedent", "draft", "human", "emergency", "unclear"]},
                    "confidence": {"type": "NUMBER"},
                    "issue": {"type": "STRING"},
                    "procedure_id": {"type": "STRING"},
                    "search_query": {"type": "STRING"},
                    "needs_precedents": {"type": "BOOLEAN"},
                    "next_question": {"type": "STRING"},
                    "risk": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "category": {"type": "STRING"},
                                "severity": {"type": "STRING", "enum": ["normal", "high", "critical"]},
                                "evidence": {"type": "STRING"}
                            }
                        }
                    },
                    "reported_facts": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "slot": {"type": "STRING"},
                                "value": {"type": "STRING"},
                                "evidence": {"type": "STRING"},
                                "original_expression": {"type": "STRING"},
                                "normalized": {"type": "STRING"},
                                "precision": {"type": "STRING"},
                                "status": {"type": "STRING"},
                                "confidence": {"type": "STRING"},
                                "change_reason": {"type": "STRING"}
                            },
                            "required": ["slot", "value", "evidence"]
                        }
                    },
                    "legal_claims": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "text": {"type": "STRING"},
                                "source_id": {"type": "STRING"},
                                "quote": {"type": "STRING"}
                            },
                            "required": ["text", "source_id", "quote"]
                        }
                    },
                    "case_context": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "source_id": {"type": "STRING"},
                                "why_similar": {"type": "STRING"},
                                "limits": {"type": "STRING"}
                            }
                        }
                    },
                    "artifact": {
                        "type": "OBJECT",
                        "description": "Formal legal document draft, court application, legal notice, RTI, or petition prepared for the caller.",
                        "properties": {
                            "type": {"type": "STRING", "description": "Type of document, e.g. application_156_3, complaint_to_sp, legal_notice_138, rti_application, consumer_complaint, affidavit, or general_legal_draft."},
                            "title": {"type": "STRING", "description": "Formal legal title of the document or application."},
                            "content": {"type": "STRING", "description": "Full formatted legal draft with numbered clauses, memo of parties, statement of facts, prayers, and verification."},
                            "intended_for_submission": {"type": "BOOLEAN", "description": "True if intended to be filed in court or sent to an authority."},
                            "warning": {"type": "STRING", "description": "Statutory warning that draft must be reviewed by an advocate before filing."}
                        }
                    },
                    "follow_up": {"type": "STRING"},
                    "source_gap": {"type": "STRING"}
                },
                "required": ["reply", "intent", "issue"]
            }
        }
    ]

    if tools:
        for t in tools:
            fn = t.get("function") or {}
            fname = fn.get("name")
            if fname and fname != "submit_consultation_reading":
                fparams = fn.get("parameters") or {}
                gemini_props = {}
                for pk, pv in (fparams.get("properties") or {}).items():
                    prop_dict: dict[str, Any] = {
                        "type": str(pv.get("type", "string")).upper(),
                        "description": str(pv.get("description", "")),
                    }
                    if pv.get("enum"):
                        prop_dict["enum"] = pv["enum"]
                    gemini_props[pk] = prop_dict
                func_decls.append({
                    "name": fname,
                    "description": fn.get("description", ""),
                    "parameters": {
                        "type": "OBJECT",
                        "properties": gemini_props,
                        "required": fparams.get("required", []),
                    }
                })

    setup_msg = {
        "setup": {
            "model": "models/gemini-3.8-live",
            "generationConfig": {
                "responseModalities": ["AUDIO"]
            },
            "systemInstruction": {
                "parts": [{"text": _system_prompt()}]
            },
            "tools": [{"functionDeclarations": func_decls}]
        }
    }

    user_content = json.dumps(context, ensure_ascii=False)
    turns: list[dict[str, Any]] = []
    if extra_messages:
        for em in extra_messages:
            role = em.get("role")
            if role == "tool":
                turns.append({
                    "role": "user",
                    "parts": [{"text": f"Tool Result ({em.get('name')}):\n{em.get('content')}"}]
                })
            elif role == "assistant":
                turns.append({
                    "role": "model",
                    "parts": [{"text": em.get("content") or "Processing tool call..."}]
                })
    turns.append({
        "role": "user",
        "parts": [{"text": user_content + "\n\nImportant: You MUST call `submit_consultation_reading` with your complete reading, reply, facts, and legal claims (and draft artifact if requested)."}]
    })

    merged_turns: list[dict[str, Any]] = []
    for t in turns:
        if merged_turns and merged_turns[-1]["role"] == t["role"]:
            merged_turns[-1]["parts"].extend(t["parts"])
        else:
            merged_turns.append(t)

    client_turn = {
        "clientContent": {
            "turns": merged_turns,
            "turnComplete": True
        }
    }

    try:
        async with websockets.connect(ws_url, close_timeout=2.0) as ws:
            await ws.send(json.dumps(setup_msg))
            await asyncio.wait_for(ws.recv(), timeout=5.0)
            await ws.send(json.dumps(client_turn))

            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=TIMEOUT_SECONDS)
                data = json.loads(msg)
                tool_call = data.get("toolCall")
                if tool_call:
                    calls = tool_call.get("functionCalls", [])
                    other_calls = []
                    for call in calls:
                        cname = call.get("name")
                        cargs = call.get("args") or {}
                        if cname == "submit_consultation_reading":
                            if cargs.get("reply"):
                                return cargs
                        else:
                            other_calls.append({
                                "id": call.get("id") or f"call_{cname}",
                                "name": cname,
                                "arguments": cargs,
                            })
                    if other_calls:
                        return {"_tool_calls": other_calls}
                server_content = data.get("serverContent")
                if server_content and server_content.get("turnComplete"):
                    break
    except Exception as exc:
        global LAST_ERROR
        LAST_ERROR = f"gemini_live_{type(exc).__name__}"
    return None


async def generate_turn_stream(
    context: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
) -> AsyncIterator[tuple[str, Any]]:
    """Stream one turn as it is generated.

    Yields `("sentence", text)` for each completed sentence of the `reply`
    field, then `("final", payload)`. A server that does not stream, or a turn
    that turns into tool calls, yields only the final payload: the caller then
    falls back to the non-streaming path instead of speaking something it
    cannot complete.
    """
    if not configured():
        return
    if is_gemini_live():
        result = await _gemini_live_completion(context, tools=tools)
        if result and result.get("reply"):
            for sentence in SENTENCE_SPLIT.split(result["reply"]):
                cleaned = sentence.strip()
                if cleaned:
                    yield "sentence", cleaned
            yield "final", result
            return
    buffer = ""
    consumed = 0
    async for delta in _stream_completion(context, tools):
        buffer += delta
        text = partial_reply(buffer)
        if not text:
            continue
        sentences, consumed = _sentence_spans(text, consumed)
        for sentence in sentences:
            cleaned = sentence.strip()
            if cleaned:
                yield "sentence", cleaned
    final = _parse_message(buffer)
    yield "final", final


def _clean_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for item in (history or [])[-8:]:
        if isinstance(item, dict) and item.get("role") and item.get("content") is not None:
            cleaned.append({
                "role": str(item["role"]),
                "content": str(item["content"])[:600],
            })
    return cleaned


async def _stream_completion(
    context: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
) -> AsyncIterator[str]:
    """Raw content deltas from an OpenAI-compatible streaming endpoint."""
    if not configured():
        return
    global _TOOLS_UNSUPPORTED
    headers = _auth_headers()
    stream_model = "gemini-2.5-flash" if "gemini-3.8-live" in MODEL.lower() else MODEL
    payload: dict[str, Any] = {
        "model": stream_model,
        "temperature": 0.2,
        "top_p": 0.85,
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2500" if "gpt-oss" in MODEL.lower() else "1800")),
        "stream": True,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _system_prompt()},
            *_clean_history(context.get("history", [])),
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
    }
    effort = os.getenv("LLM_REASONING_EFFORT", "low" if "gpt-oss" in MODEL.lower() else "")
    if effort:
        payload["reasoning_effort"] = effort
    if tools and not _TOOLS_UNSUPPORTED:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    timeout = httpx.Timeout(TIMEOUT_SECONDS, connect=min(2.0, TIMEOUT_SECONDS))
    try:
        async with _client().stream("POST", endpoint(), headers=headers, json=payload, timeout=timeout) as response:
            if response.status_code in {400, 404, 422} and tools:
                _TOOLS_UNSUPPORTED = True
            if response.status_code >= 400:
                return
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    return
                try:
                    event = json.loads(data)
                    choice = event["choices"][0]
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
                delta = (choice.get("delta") or {}).get("content")
                if isinstance(delta, str) and delta:
                    yield delta
    except (httpx.HTTPError, asyncio.TimeoutError):
        return


async def generate_turn(
    context: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
    extra_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Produce one natural-language turn from facts and retrieved evidence.

    The context object is application-generated. Caller text and all corpus
    passages inside it are explicitly untrusted data in the system prompt.

    When `tools` is supplied the provider's native tool calling is used; any
    tool calls come back under `_tool_calls` and are executed by the engine,
    never here. Servers without tool support answer with the same shape in
    `tool_requests`, which the engine accepts as an equivalent.
    """
    if not configured():
        return None
    if is_gemini_live():
        try:
            result = await _gemini_live_completion(context, tools=tools, extra_messages=extra_messages)
        except TypeError:
            result = await _gemini_live_completion(context, tools=tools)
        if result:
            return result
    if tools and _TOOLS_UNSUPPORTED:
        tools = None
    # 1,200 was tight enough that a complete answer with two quotations and a
    # next step could be cut mid-sentence; the model then dropped content
    # rather than the engine noticing. A consultation-style answer -- the rule,
    # the closest decision, how it applies here, and one question -- needs more
    # than that, so the ceiling is 3,200 (4,200 for the reasoning models that
    # spend tokens before the JSON). The budget still bounds the turn.
    default_gen_tokens = "800"
    return await _json_completion(
        _system_prompt(),
        context.get("history", []),
        context,
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", default_gen_tokens)),
        tools=tools,
        extra_messages=extra_messages,
        budget=float(os.getenv("LLM_GENERATE_BUDGET_SECONDS", "18")),
    )


async def _json_completion(
    system: str,
    history: list[dict[str, str]],
    context: dict[str, Any],
    max_tokens: int,
    temperature: float = 0.2,
    schema: dict[str, Any] | None = None,
    timeout_cap: float | None = None,
    tools: list[dict[str, Any]] | None = None,
    extra_messages: list[dict[str, Any]] | None = None,
    model: str | None = None,
    budget: float | None = None,
) -> dict[str, Any] | None:
    """One structured completion with a bounded retry.

    A single transient 429/503 should not degrade the whole turn to
    passages-only, and an unparseable JSON answer should get exactly one
    cheaper second chance. Everything stays inside BUDGET_SECONDS so a voice
    reply cannot hang behind a slow provider.
    """
    global _SCHEMA_REJECTED, LAST_ERROR, _TOOLS_UNSUPPORTED
    headers = _auth_headers()
    wants_schema = bool(schema) and STRUCTURED_MODE != "json_object" and not _SCHEMA_REJECTED
    target_model = model or MODEL
    if "gemini-3.8-live" in target_model.lower():
        target_model = "gemini-2.5-flash"
    payload = {
        "model": target_model,
        "temperature": temperature,
        "top_p": 0.85,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "nyaya_reading", "strict": True, "schema": schema},
        } if wants_schema else {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            *_clean_history(history),
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            *(extra_messages or []),
        ],
    }
    effort = os.getenv("LLM_REASONING_EFFORT", "low" if "gpt-oss" in target_model.lower() else "")
    if effort:
        payload["reasoning_effort"] = effort
    if tools and not _TOOLS_UNSUPPORTED:
        # Native tool calling and a forced JSON schema are not reliably
        # compatible on self-hosted parsers; the prompt already demands JSON
        # output, so the schema constraint is dropped for tool turns only.
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        payload.pop("response_format", None)
    # `budget` is the wall-clock ceiling for the whole call including retries,
    # not the per-attempt socket timeout. A voice turn needs it bounded: three
    # attempts against a slow provider under a 45-second budget produced a
    # 33.7-second turn that then failed, which is worse than a fast, named
    # degradation.
    deadline = asyncio.get_running_loop().time() + (budget or BUDGET_SECONDS)
    last_error: str | None = None
    # A 429 gets exactly one short retry. Measured: three retries turned one
    # throttled question into a 37-second turn that degraded anyway, and a
    # quota wall does not clear inside a voice turn.
    retried_429 = False
    prompt_chars = len(system) + len(json.dumps(context, ensure_ascii=False))
    estimate = max(1, prompt_chars // 4) + max_tokens
    for attempt in range(3):
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0.5:
            break
        await _respect_tpm(estimate, deadline)
        timeout_seconds = min(timeout_cap or TIMEOUT_SECONDS, remaining)
        retry_after = 0.0
        try:
            timeout = httpx.Timeout(timeout_seconds, connect=min(2.0, timeout_seconds))
            response = await asyncio.wait_for(
                _client().post(endpoint(), headers=headers, json=payload, timeout=timeout),
                timeout=timeout_seconds,
            )
            _record_rate_headers(response)
            if response.status_code in {400, 404, 422} and payload.get("tools"):
                # This server will not accept tool definitions. Remember it, so
                # the next turn does not pay for the same failed attempt.
                _TOOLS_UNSUPPORTED = True
                payload.pop("tools", None)
                payload.pop("tool_choice", None)
                payload["response_format"] = {"type": "json_object"}
                last_error = f"http_{response.status_code}"
                retry_after = 0.0
            elif response.status_code == 400 and payload.get("response_format", {}).get("type") == "json_schema":
                # Server without guided decoding: remember, downgrade, retry
                # once. Paying this 400 on every turn would be slow and noisy.
                _SCHEMA_REJECTED = True
                payload["response_format"] = {"type": "json_object"}
                last_error = "schema_rejected_switched_to_json_object"
                retry_after = 0.0
            elif response.status_code == 400 and ("json_validate_failed" in response.text or "adjust your prompt" in response.text):
                # Provider constrained validator failed (e.g. token cutoff on reasoning models):
                # Drop response_format constraint, bump tokens, and let _parse_message decode the JSON.
                payload.pop("response_format", None)
                if "max_tokens" in payload:
                    payload["max_tokens"] = max(payload["max_tokens"], 2000)
                last_error = "json_validator_failed_switched_to_prompt_json"
                retry_after = 0.0
            elif response.status_code in RETRY_STATUS:
                last_error = f"http_{response.status_code}"
                if response.status_code == 503 and "3.8" in str(payload.get("model", "")):
                    payload["model"] = "gemini-2.5-flash"
                    retry_after = 0.05
                else:
                    retry_after = _retry_after(response, attempt)
                    if response.status_code == 429:
                        if retried_429:
                            break
                        retried_429 = True
                        retry_after = max(0.8, min(retry_after, 5.0))
            else:
                response.raise_for_status()
                _record_usage(response, prompt_chars)
                message = response.json()["choices"][0]["message"]
                result = _parse_message(message.get("content")) or {}
                calls = _tool_calls(message)
                if calls:
                    result["_tool_calls"] = calls
                if result:
                    LAST_ERROR = None
                    return result
                last_error = "unparseable_json"
                retry_after = 0.2
        except asyncio.TimeoutError:
            last_error = "timeout"
            retry_after = 0.2
        except httpx.HTTPStatusError as exc:
            # A 4xx (bad key, unknown model, rejected schema) will not get
            # better by asking again.
            last_error = f"http_{exc.response.status_code}"
            break
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            last_error = type(exc).__name__
            retry_after = 0.2
        wait = min(retry_after, max(0.0, deadline - asyncio.get_running_loop().time() - 0.5))
        if wait > 0:
            # Jitter, so several callers retrying one throttled provider do not
            # arrive together and throttle each other again.
            await asyncio.sleep(wait + random.uniform(0.0, 0.15))
    if last_error:
        LAST_ERROR = last_error
    return None


LAST_ERROR: str | None = None


def last_error() -> str | None:
    """Why the most recent completion failed, for health and diagnostics."""
    return LAST_ERROR


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise native `tool_calls` into name + parsed arguments."""
    raw = message.get("tool_calls")
    if not isinstance(raw, list):
        return []
    normalised: list[dict[str, Any]] = []
    for index, item in enumerate(raw[:6]):
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else {}
        name = str(function.get("name") or item.get("name") or "")
        if not name:
            continue
        arguments = function.get("arguments", item.get("arguments"))
        normalised.append({
            "id": str(item.get("id") or f"call_{index}"),
            "name": name[:64],
            "arguments": arguments,
        })
    return normalised


def _retry_after(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after", "") or response.headers.get("x-ratelimit-reset-tokens", "")
    header = str(header).strip()
    if header.endswith("ms"):
        try:
            return max(0.2, min(5.0, float(header[:-2]) / 1000.0))
        except ValueError:
            pass
    elif header.endswith("s"):
        try:
            return max(0.5, min(5.0, float(header[:-1])))
        except ValueError:
            pass
    try:
        return max(0.5, min(5.0, float(header)))
    except ValueError:
        return min(3.0, 0.5 * (2 ** attempt) + random.uniform(0.1, 0.3))

