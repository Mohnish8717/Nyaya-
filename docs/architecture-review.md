# Nyaya — architecture review

Date: 2026-11-01. Scope: the whole repository as it exists now (`app/` 1,206 lines, `static/` 498, `scripts/` 268, 2 approved legal sources, 1,810 bundled judgments).

## Method

This architecture review and competitive analysis evaluates the full repository and retrieval pipeline (`app/`, `static/`, `scripts/`, approved legal sources, and bundled judgments).

The analysis follows strict empirical evidence discipline:

- one bounded question per phase, not broad assumptions;
- every conclusion tied to an exact file/line or an empirical measurement;
- observation separated from inference from unknown;
- coverage limits stated explicitly (an empty search page does not prove absence).

Measurements were run against the bundled indices. Claims below that are inferences are marked as such; unverified items are collected at the end.

---

## A. Products that are functionally the same

"Intended implementation" means: a citizen-facing, multilingual, voice-first Indian legal-information assistant that retrieves authority, answers in the caller's language, and hands off to a human rather than acting autonomously.

### Closest analogues — Indian, citizen-facing

| Product | What it is | Overlap with Nyaya |
|---|---|---|
| **Nyaya Setu / "Dishika"** (government, Bhashini) | Multilingual legal-assistance chatbot in the Nyaya Setu app, 22 Indian languages, voice + translation | Almost the same product thesis: citizen, multilingual, voice, non-judgmental first contact. Government-backed distribution and Bhashini ASR/TTS are the two things it has that this build cannot match. |
| **NyayGuru** | Free plain-language Indian-law chat assistant | Same audience and register. Weaker on evidence discipline (no source-gated claims). |
| **Nyaay AI** | Legal AI platform built with judiciary collaboration — research, documentation, case workflows | Overlaps the retrieval and drafting surfaces; institution-facing rather than citizen-facing. |
| **IndLawAI** | AI legal research/information app for citizens, students, professionals | Overlaps "explain the law with sources" and judgment lookup. |
| **Nyaypeeth** | Consumer "personal AI legal assistant" for Indian laws and rights | Same consumer framing, text-first. |

### Professional-tier comparables (different user, same machinery)

Lawyer-side assistants and legal-research platforms — NyayAssist, CaseMine, Jurisphere, Riverus/LegitQuest, Provakil — solve the same retrieval, citation and drafting problems for firms rather than citizens. They are the realistic benchmark for the *retrieval quality bar*, not for the product: they can assume a trained reader who will check the citation.

### Global analogues

DoNotPay and the US guided-self-help cluster (Upsolve, JustFix, Hello Divorce) for the "get a citizen to a correct next step" job; JusticeBot (Canada) and Amica (Australia) for decision-tree-plus-ML legal guidance with a documented audit trail; CoCounsel / Harvey / Spellbook / Luminance for grounded drafting with citation.

### What is actually differentiated here

None of the consumer-grade Indian products above advertise **claim-level abstention**: a legal proposition is displayed only if it carries a verbatim quote from an approved, in-date, jurisdiction-matching source, and the system degrades to non-legal help when that fails. That is the defensible idea in this repo. It is also why the thin source pack (2 approved records) is the binding constraint — the differentiator only pays off at a coverage level this build does not have.

---

## B. What is right, and should not be churned

1. **No canned answers.** Prompts constrain scope and evidence; they contain no answer templates. This is the correct architecture for the product claim.
2. **Evidence gate before display.** Claims must name a retrieved source and a verbatim quote; failures trigger one constrained repair, then a claim-free reply. Fail-closed in the right direction.
3. **Tool surface is an allow-list, not a plugin API.** `data/service_catalog.json` is scope, not a script; the model cannot execute anything.
4. **Judgments are not law.** Case records carry `treatment_status: "not_checked"`, and the prompt forbids presenting similarity as authority.
5. **Consent and non-action defaults.** Nothing is filed, sent, paid, or contacted; the human handoff requires a separate consent step and a language match.
6. **Documentation is unusually honest** about what the system cannot do (`docs/architecture.md`, `data/DATA_SOURCES.md`). Most legal-AI repos overclaim; this one does not.

---

## C. Findings

Severity: **P0** = produces wrong or unusable behaviour now; **P1** = blocks production or hides failures; **P2** = debt.

### P0-1. Lexical retrieval matches almost everything (measured)

`app/knowledge.py:185`, `app/case_corpus.py:98`, `app/knowledge.py:155` build a disjunctive FTS query from every word longer than one character:

```python
words = [word for word in words if len(word) > 1][:24]
match = " OR ".join('"' + word + '"' for word in words)
```

Measured on the bundled index (28,617 chunks / 1,810 judgments), for the query *"my landlord is not returning my security deposit and he is threatening me"*:

- tokens used: `my, landlord, is, not, returning, security, deposit, and, he, threatening, me`
- distinct judgments matched: **1,808 of 1,810 (99.9%)**
- matched by content words only (`landlord`, `security`, `deposit`): 554
- matched by pure stopwords (`the`, `and`, `was`, `for`): **1,810 (100%)**

So `LIMIT` + BM25 ordering is doing all the filtering, on a candidate set that is essentially the entire corpus. The same construction feeds the "authoritative" source lookup, where it is currently masked only because there are two approved sources.

**Fix:** stopword/negation-aware term selection, `AND` for required terms with `OR` for expansion, phrase and field boosts, a title/court boost, and a top-k reranker. Keep the interface; the internals are replaceable by design.

### P0-2. High Court judgments silently become national authority

`scripts/sync_public_judgments.py:36`:

```python
HC_JURISDICTION_BY_CODE = {"11_24": "IN-SK", "29_3": "IN-KA"}
```

and line 169:

```python
"jurisdiction": "IN" if dataset == "sc" else HC_JURISDICTION_BY_CODE.get(court_code or "", "IN")
```

Only 2 of 25 High Court codes are mapped. Any other court is written with jurisdiction `IN`, which `_jurisdiction_matches` (`app/knowledge.py:168`) and `retrieve_cases` (`app/case_corpus.py:108`) then treat as applicable to *every* state query. A Karnataka or Bombay judgment can surface as if it were national law — the exact error class the retrieval policy is trying to prevent. Silent: no warning, no count of unmapped courts.

**Fix:** full court-code → jurisdiction table from the upstream metadata, refuse to ingest an unmapped code, and assert in tests that no HC record carries `IN`.

### P0-3. The source pack is two records

`data/legal_sources.json` contains `constitution-india-article-22` and `nalsa-faq-urgent-legal-assistance`, both approved. Everything the product promises about "verified law" rests on those two passages. In practice most turns cannot end in `answer_status: "source_linked"`, so the system's best path is rarely reachable and the honest fallback ("I could not verify this") becomes the normal answer.

**Fix:** this is content work, not code: statutes with section-level granularity, amendment intervals, and per-state rules, each reviewed. It is also the only path to a meaningful evaluation set.

### P0-4. "Verified" labels overstate what was checked

The validator (`app/legal_engine.py:62`) confirms that the quote is a whitespace-normalised substring of the source. That is provenance, not entailment — correctly stated in the docs and in `verification.source_scope`, but the UI renders `SOURCE-LINKED INFORMATION` (`static/app.js:141`) as a badge, which a distressed caller will read as "checked and true".

**Fix:** label the mechanism, not the outcome ("Quote matched to source" + "interpretation not reviewed"), and add a claim-level `entailment: unreviewed` field that the UI cannot drop.

### P1-1. No end-to-end latency budget; no streaming

`answer_turn` makes up to **three sequential** LLM calls (plan → generate → optional repair → optional claim-free regenerate), each with a 12 s timeout (`app/model_gateway.py:21`). Worst case is ~36–48 s with no aggregate deadline, no retry/backoff, no circuit breaker, and no cancellation when the browser disconnects. The client shows typing dots and has no abort.

For a voice-first product this is the most damaging UX defect: time-to-first-audio equals full generation plus a TTS round trip. `docs/architecture.md` describes streaming the first validated sentence; the implementation does not.

**Fix:** one deadline per turn shared across stages, `asyncio.timeout` + client `AbortController`, SSE streaming for the reply, and a two-phase answer (safe acknowledgement + validated claims) as the docs already envision.

### P1-2. Failures are invisible

`app/model_gateway.py:101` catches `httpx.HTTPError, KeyError, TypeError, ValueError` and returns `None`. There is no logging anywhere in the app: no request ID, no stage timing, no record of which claims were rejected or why, no metric for abstention rate. When the model endpoint returns a 400 because a server rejects `response_format`, the user sees "The configured model did not return a valid structured response" and the operator sees nothing.

**Fix:** structured logging with a per-turn trace ID, stage timings, rejection reasons, and counters. This is also the prerequisite for any evaluation.

### P1-3. Session state cannot survive a restart or a second worker

`SESSIONS` (`app/main.py:39`) is a module-level dict. Consequences: all in-flight conversations are lost on restart; running two workers breaks sessions non-deterministically; there is no per-session lock, so two concurrent turns interleave history; and 5,000 sessions × 12 messages × up to 6,000 chars is roughly 700 MB of worst-case retention in process memory with no eviction except on new-session creation.

**Fix:** Redis (or Postgres) with TTL, per-session lock, and a hard per-session byte cap. The `SessionStore` seam does not exist yet; add it before any multi-user deployment.

### P1-4. Rate limiting is per session, so it is free to bypass

`_check_session` (`app/main.py:153`) allows 20 turns/minute per session. Creating a session is unauthenticated, so an attacker resets the budget at will, and every turn costs LLM inference plus a Gnani call. There is no IP, subnet, tenant, or global budget, and no cost accounting.

**Fix:** token-bucket per IP + per session, a daily global spend cap, and 429 with `Retry-After`.

### P1-5. Consent is a client-supplied boolean

The browser hard-codes `consent_confirmed: true` (`static/app.js:104`) after the modal, and the server only checks that the field is true (`app/main.py:137`, `:169`). An API client can assert consent it never obtained. For a product that documents consent as a safety gate, the gate is currently decorative: there is no consent record, timestamp, version of the notice shown, or revocation.

**Fix:** server-issued, versioned consent records tied to the notice text, with an audit trail — required for any real deployment.

### P1-6. Speech proxy is open to non-browser clients holding no credentials

`_valid_websocket_origin` (`app/main.py:369`) returns `True` when `Origin` is absent, on the reasoning that the provider key is server-side. But the *point* of the proxy is that clients without the key can use it; with `Origin` omitted, any local process (or any host that can reach the port) gets free Prisma/Timbre traffic billed to the operator. The comparison also uses `netloc` only, which ignores scheme and breaks behind a TLS-terminating proxy.

**Fix:** require an `Origin` (or a signed short-lived token minted with the session), compare full origin against an allow-list, and bind to loopback by default in production docs.

### P1-7. Startup blocks the event loop

`lifespan` (`app/main.py:70`) calls `knowledge.rebuild_index()` and `case_corpus.index_imported_cases()` synchronously inside the async startup hook. With the bundled 175 MB / 28.6k-chunk index this is tolerable; with an imported partition it becomes minutes of synchronous SQLite writes and PDF-adjacent I/O before the app accepts a request.

**Fix:** build indices out-of-process (a CLI, already half-present in `scripts/reindex_judgments.py`) and have startup verify a version stamp instead of rebuilding.

### P1-8. SQLite concurrency is unmanaged and fails open

Every query opens a new connection with no `busy_timeout` and no WAL (`app/knowledge.py:104`, `app/case_corpus.py:97`). `sqlite3.OperationalError` is caught and converted to an empty result — so a lock contention event becomes "no evidence found", which the generator then honestly reports as a source gap. The system degrades to a *plausible wrong answer* instead of an error.

**Fix:** WAL + `busy_timeout`, a connection pool or a dedicated reader, and distinguish "no evidence" from "index unavailable" in the response.

### P1-9. Operational basics are absent

No tests (0 test files), no CI, no lockfile, unpinned ranges in `requirements.txt`, no container, no `.env` loading in the app itself (only `run.sh` sources it, so `uvicorn app.main:app` starts silently unconfigured), and `run.sh` runs `pip install` on every start. `float(os.getenv(...))` at import (`app/model_gateway.py:21`) crashes with a raw traceback on a malformed value.

### P2 — smaller, still real

- **Service chips resend the previous message** (`static/app.js`, `runService`): clicking "Compare judgments" after a turn re-submits `state.lastUserText`, duplicating the user turn in history and paying for a second full generation.
- **TTS playback has no stop or interrupt**: two "Speak" clicks overlap, and there is no barge-in while the mic is live. Playback assumes 24 kHz regardless of what the provider returns.
- **`#safetyCheck`** (`static/index.html:97`) is collected and never read.
- **UI chrome is English-only** while the product is multilingual: no `lang` attribute update, no translated rail/composer/modal strings, no RTL handling.
- **Modals** have no focus trap, no Escape handler, and no focus restoration; `aria-modal` is set but focus can leave the dialog.
- **Empty session id in a URL** (`GET /api/session/{id}/artifacts`) means session identifiers land in access logs, which contradicts the "no transcript on disk" claim in spirit.
- **`/api/review/request` forwards the internal `session_id`** to the third-party review provider — unnecessary identifier sharing.
- **Dead code**: `knowledge.validate_evidence_claims` duplicates `legal_engine._validate_claims` and is never called; `case_corpus.source_url` is unused; `.fact-card`, `.issue-detail-wrap`, `.draft-card`, `.followup-chip`, `.divider-label` in `styles.css` are leftovers from a removed facts form.
- **Language codes are not canonicalised**: a model reply of `"hi"` or `"Hindi"` silently falls back to `auto` instead of mapping to `hi-IN`, so `tts_language` can go `unavailable` for a supported language.
- **Prompt/context duplication**: history is sent both as `messages` and inside the JSON context, inflating tokens every turn.

---

## D. What would make this the biggest, not just cleaner

Ordered by leverage. Each step is independently shippable.

**1. Coverage before cleverness.** Expand `legal_sources.json` to section-level statute coverage for the central codes that generate the most citizen questions (IPC→BNS, CrPC→BNSA, CPC, Contract, Rent/tenancy, Consumer, Family, Labour), each with effective intervals and a reviewer. Ten thousand lines of unreviewed text is worse than 200 reviewed passages; but 2 is not a product. This single change moves the success rate more than any code work below.

**2. Retrieval that can be measured.** Stopword-aware query building, `AND` semantics, BM25 field weights, a dense retriever for paraphrase and transliteration, reciprocal-rank fusion, then a cross-encoder reranker over the top 50. Build the eval harness *first*: 200–300 labelled queries (per language, per jurisdiction, plus adversarial: wrong state, overruled case, fabricated section, malicious pasted text) with recall@k, nDCG, and claim-entailment review. Without this, every other change is unfalsifiable.

**3. A real evidence pipeline.** Treat judgments as a data product: bronze manifests with checksums, silver page-aware text with OCR for scanned PDFs, gold paragraph/section chunks with court, bench, date, posture, citation, and language, then a treatment graph (`appealed_by`, `overruled_by`, `distinguished_by`) sourced from verified records — never from an LLM extraction alone. Full corpus in object storage, index built by a batch job, service reading a versioned index.

**4. Latency architecture for voice.** Streaming planner → parallel retrieval → streamed generation → sentence-level TTS pipelined against token output; speculative retrieval on the ASR partial; barge-in that cancels generation and playback; a hard p95 budget from end-of-speech to first audible token, enforced in CI with per-language load tests.

**5. A service boundary that can grow.** Split into a stateless API, a retrieval service, an inference gateway (model router by language and task — a small model for planning, a legal-tuned model for generation, a verifier for entailment), Redis for session and rate limits, Postgres for consent/source/workflow audit, and a queue for ingestion and evaluation runs. Add accounts, encryption at rest, retention/deletion, PII redaction before third-party inference, and abuse monitoring.

**6. Trust machinery that matches the claims.** A claim ledger (proposition, exact passages, source version + checksum, retrieval rank, model/prompt version, reviewer status), timestamped consent records, an abstention dashboard, and a reviewer console for the human handoff that is currently a single optional HTTP POST.

---

## E. What "perfect" cannot mean here

Legal information has no correct-answer oracle; the honest targets are: **calibrated abstention** (state the rate at which you decline, and be right about it), **provenance completeness** (every displayed proposition traceable to a versioned source), **jurisdiction and currency correctness**, and **measured multilingual parity**. Anything advertised beyond that — "near-perfect accuracy", "never wrong" — is a liability, not a feature, and the existing README is right to refuse it.

## F. Unknowns (not established by this review)

- Whether `11_24` and `29_3` actually correspond to Sikkim and Karnataka in the upstream dataset. The bundled records are self-consistent (`High Court of Sikkim`, `IN-SK`), so the bundled data is not evidence about the mapping table.
- Real Prisma/Timbre behaviour: no provider credentials were present, so the WebSocket frame contract (1024-byte frames from `static/pcm-worklet.js`), transcript event shape, and Timbre sample rate were read from code, not observed against the live API.
- Model quality: no LLM endpoint was configured, so no turn was executed. Every statement above about generation is about the orchestration around the model.
- Retrieval numbers are measured on the bundled 1,810-judgment index only; they do not predict behaviour on a full corpus, which would be worse for stopword noise, not better.
