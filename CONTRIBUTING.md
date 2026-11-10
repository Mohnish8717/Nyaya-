# Contributing & Engineering Guidelines

Read this guide before contributing or making architectural changes. It records the core architecture, the claims and safety guarantees the product makes, and the verification commands that enforce them. Treat these as load-bearing engineering invariants: if you change behavior described here, update the matching specification and tests in the same pull request.

## What this project is

Nyaya — a voice-first, multilingual legal **information** agent for citizens across India. It is not a lawyer, not a judge, and never delivers advice beyond what approved sources verbatim support. The core thesis that every subsystem protects: **no legal proposition reaches a caller without a verbatim quote from an approved, in-date, jurisdiction-matching source; when evidence is missing the system says so instead of guessing.**

## Hard Invariants (Break one and the product is broken)

1. **Claims discipline.** Legal propositions live only in `legal_claims`, each with an exact `quote` from a supplied source. Exact-substring verification runs in the engine; a claim that fails is removed, never softened. Downgraded turns report `entailment: "not_reviewed"` — never claim a review that did not happen.
2. **Fail closed, degrade visibly.** No model → deterministic citations mode. The interpreter, generation, and entailment stages can each fail without breaking a turn. Every degradation is named in the trace; nothing pretends to be smarter than it was.
3. **Monotone safety.** The lexical safety net (`app/triage.py`) always runs and can never be overruled downward by a model. Emergencies get the safety script first, skip generation, and block drafting.
4. **Grounding or silence.** Facts enter the ledger only with a verbatim span from the caller. The `record_fact` tool enforces this at the boundary. Ungrounded model guesses never become session state.
5. **No new powers.** No shell, network, filesystem, or arbitrary action tool. A legal agent's worst reachable failure is retrieving the wrong passage. Human contact requires an explicit two-step caller action (`/api/action` → confirm, and `consent_to_share`).
6. **PII at the boundary.** `privacy.for_model()` masks identity numbers before any model sees them; `privacy.redact_deep()` masks logs and the failure store. Keep it that way for every new model-facing surface, including compacted digests.
7. **Sessions are not credentials.** The per-session bearer token is required on every private route. Never turn a session id into an auth bypass.

## Architecture Map

- `app/main.py` — FastAPI surface: auth, rate limits, CSP headers, turn pipeline, speech websockets, and exception hooks.
- `app/legal_engine.py` — The orchestrator: lexical safety → route → speculative retrieval → single-round consultation call returning answer and reading together with tool calls → validate reading → settle route (`routing.settle`) → write grounded facts → verify claims (exact quotes + entailment judge) → fallback to grounded-only. A rejected claim immediately terminates the generated answer to avoid latency spikes.
- `app/understanding.py` + `prompts/legal_agent.md` — Reading validation out of the consultation payload: closed vocabularies, verbatim grounding, monotone risk, and emergency intent capping when ungrounded.
- `app/model_gateway.py` — OpenAI-compatible provider gateway: bounded token budgets, 429 backoff retry, streaming first sentences, and local tokens-per-minute window (`LLM_TPM_BUDGET`). `usage_snapshot()` reports provider token metrics at `/api/health`.
- `app/tools.py` — Tool registry: JSON-Schema-declared specifications, strict `additionalProperties: false`, runtime validators, side-effect classes (`read_only` / `guarded_write`), and per-tool budgets pinned by `tests/test_tool_schema.py`.
- `app/facts.py` — Structured fact ledger (session memory): every fact retains verbatim spans, source turns, status (`caller_confirmed`, `corrected`, `estimated`, `reported`), confidence, and prior values.
- `app/voice.py` — Real-time barge-in and speech turns backed in SQLite WAL so any worker can stop synthesis audio.
- `app/sessions.py` — SQLite WAL store, distributed rate limiter, and conversation compaction digest (`earlier` column).
- `app/knowledge.py`, `app/case_corpus.py`, `app/judgment_corpus.py`, `app/dense.py`, `app/retrieval.py` — The statutory source pack and two judgment stores: curated partitions under `.runtime` and embedded passage corpus (`JUDGMENT_CORPUS_DB`) with 10,588 judgments / 94,843 passages, BM25 + int8 cosine embeddings fused via Reciprocal Rank Fusion.
- `app/pdf_generator.py` — Universal court-grade legal drafting engine in ReportLab with A4 legal margins, two-pass running header/footer rules, automatic line height, auto-fit table layout, and statutory legal disclaimers.
- `data/procedural_kb.json`, `data/workflows.json` — Human-curated procedure cards and intake state machines gated by `review_status`.
- `evals/run_evals.py`, `evals/retrieval_eval.py`, `evals/golden.jsonl`, `evals/retrieval_baseline.json` — Behavioural evaluation benchmarks and CI retrieval gates.
- `app/feedback.py` + `scripts/feedback_candidates.py` — Failure-to-rule pipeline: production errors and claim rejections are logged redacted into `.runtime/failures.jsonl` for offline review.

## Verification Checklist

Always verify the full test and evaluation suite before opening a pull request:

```bash
# 1. Full unit + API suite (359 tests, no external models required)
.venv/bin/python -m pytest -q

# 2. Behavioral verification (18 real-world legal edge cases)
.venv/bin/python evals/run_evals.py

# 3. Retrieval quality benchmark vs baseline
.venv/bin/python evals/retrieval_eval.py --gate

# 4. Browser speech-queue logic & syntax check
node --check static/app.js && node --test tests/js
```

## Engineering Rules

- Prefer editing existing modules over creating new ones; keep `app/` flat and cohesive.
- Tests must be written adjacent to the behavior they pin under `tests/`.
- Latency budgets live in `tests/test_latency.py`; turn budgets in `app/legal_engine.py`.
- Keep documentation up to date: `README.md`, `docs/architecture.md`, `DESIGN.md`, and this file are the standing source of truth.
- `requirements.txt` tracks abstract dependency ranges; `requirements.lock` tracks exact resolved deployment floors.

