# Nyaya — multilingual, model-led citizen legal advocate

This project is a runnable prototype for a voice-first Indian legal-information assistant. A turn is routed before anything is generated: a matched, reviewed **procedure** is answered from a curated file (no model, no invention), a safety turn gets a safety script, and a law question goes to retrieval and an evidence-gated answer. The model writes language, not authority — there is no issue classifier that chooses canned legal replies, no fixed questionnaire, and no mandatory fact-confirmation card. The prompts define scope and evidence rules; they do not contain answer templates.

With no model configured the app still works: it shows retrieved passages with their exact source, and says plainly that it has not drawn a legal conclusion. That degraded mode is a deliberate design choice, not a fallback accident. The sentences it says in that mode — the acknowledgement, the safety script and the "I could not verify this" / "no model is connected" replies — are written for each of the ten served languages and Hinglish, so a Telugu caller is not answered in English at the one moment the app has least to offer.

The system combines Gnani **Prisma** streaming speech-to-text and **Timbre** streaming text-to-speech with a provider-neutral LLM gateway, legal-source retrieval, a searchable court-judgment ingestion path, and explicit action controls. A local/self-hosted Gnani Evon endpoint is supported through the OpenAI-compatible API used by vLLM and SGLang.

### Where the real-time contracts came from

The voice behaviour here is not modelled on screenshots. It is reversed from the shipped `@vapi-ai/web` package (2.7.1): `StopSpeakingPlan` (`numWords`, `voiceSeconds`, `backoffSeconds`, acknowledgement vs interruption phrases), `StartSpeakingPlan` (`waitSeconds` plus punctuation / trailing-number / unfinished-clause endpointing), `Workflow` with `ConversationNode` and `ToolNode`, and `ToolCall`/`ToolMessage`. [app/voice.py](app/voice.py), [app/workflow.py](app/workflow.py) and [app/tools.py](app/tools.py) implement those semantics on Gnani Prisma + Timbre, with the safety and guardrail differences this domain requires.

The speech turn that a barge-in has to stop, and the rate-limit counters, live in the shared session database rather than in process memory: the worker that receives the interrupt does not have to be the worker streaming the audio, and two workers no longer double the real rate limit. One limit is stated rather than hidden: first-audio and stop latencies are measured against a stub provider on this machine — no Timbre endpoint is configured here, so anything about real synthesis latency is unverified.

## Run locally

Requirements: Python 3.10+, a modern browser, and model credentials/endpoints in `.env`.

```bash
./run.sh
```

Open <http://127.0.0.1:8000>. `run.sh` creates `.venv`, installs the runtime requirements plus the optional local dense-retrieval dependencies, loads `.env`, and launches FastAPI.

The boot path is the same without the script: the app itself loads `.env` at import (verified by running `uvicorn app.main:app` directly), and on first start it builds and warms the local dense retrieval index in the background — no API key is needed for that. If the dense dependencies are missing it logs the exact fix and continues with lexical retrieval only; `/api/health` always shows which mode is real (`model_configured`, `gnani_configured`, `dense_retrieval`).

## Configure the models

Copy `.env.example` to `.env` (the run script does this when `.env` is absent), then configure your server-side Gnani key and an OpenAI-compatible LLM endpoint:

```dotenv
GNANI_API_KEY_ID=your-gnani-key
PRISMA_MODEL=prisma-v2.5
TIMBRE_MODEL=timbre-v3.0
TIMBRE_DEFAULT_VOICE=Shivani

# Example for a locally served Evon model on port 8001:
LLM_BASE_URL=http://127.0.0.1:8001/v1
LLM_MODEL=gnani/gnani-evon-v3.3-30B-A3B
LLM_API_KEY=
LLM_TIMEOUT_SECONDS=12
```

The model key stays on the server. The gateway accepts any OpenAI-compatible chat completion endpoint; it does not call the ChatGPT API unless you explicitly configure such an endpoint. Evon is available as open weights; the model card shows vLLM/SGLang-compatible serving. A 30B-total-parameter model still needs suitable inference hardware or a hosted endpoint.

Gnani's current docs list **ten** supported Indian languages for Prisma and Timbre. Prisma v3 realtime identifies the language of each utterance when the stream is opened with `language_code=indic-auto`, and reports it as `detected_language_code`; `format=codemix` keeps a Hindi sentence with English legal nouns intact. That is what makes the language question unnecessary: the caller speaks, the stream says what it heard, and the reply follows. Typed messages take the same path, with script detection and a Romanised-Hindi check standing in when no model is available. Hinglish is a first-class outcome (`hi-en` in text, `auto` in Timbre). Evon lists Odia in addition to the ten, but Prisma and Timbre do not, so Odia is supported as text and reported as **text only** — the reply is not voiced, because voicing it read the caller's own sentence back in whichever language the synthesiser guessed. No vendor list means every Indian language. See the current [Prisma realtime docs](https://docs.gnani.ai/api/STT/stt-websocket), [Prisma v3 model page](https://docs.gnani.ai/api/STT/models/prisma-v3), [Timbre realtime language list](https://docs.gnani.ai/api/TTS/tts-websocket), and [Evon model card](https://huggingface.co/gnani/gnani-evon-v3.3-30B-A3B).

For microphone input you just tap the orb and talk — no language to pick, including when you mix Hindi and English in one sentence. The language selector is an override for the rare case the recogniser misreads the script. The transcript appears in the composer so you can correct names, dates, amounts, or negations, and it is followed by a short merge window rather than a second full silence wait, because Prisma has already endpointed. The app then sends the transcript directly to the LLM; it does not make you confirm a pre-filled facts form before receiving help.

## What happens on a turn

1. **The model reads, code checks.** `app/understanding.py` makes one structured call that reads the turn as a person would: intent (emergency, handoff, procedure, statute, precedent, draft, unclear), risk findings with a verbatim quote each, stated facts with the span they came from, the single most useful next question, and the caller's language. Code then does only what judgement should not: it rejects anything outside the published vocabularies, drops facts that are not grounded in the caller's own words, caps a critical finding the reader cannot quote, and unions risk with the lexical net in `app/triage.py` so severity can be raised but never lowered — including when the provider is down. A threat phrased in words no keyword list holds still reaches the safety route; a document requested during an active threat still gets `block_drafting`.
2. **Reviewed procedures need no model.** Eight common processes (FIR, free legal aid, consumer complaint, cybercrime, RTI, unpaid wages, domestic-violence protection, cheque bounce) live in `data/procedural_kb.json`. The steps, sources and review status come from that file; the model cannot add, drop or reorder a step, and a `draft` procedure is labelled as unreviewed in the UI.
3. **Everything else retrieves.** `app/retrieval.py` removes function words, maps Romanized terms (`kiraya` → `rent`) through a small glossary, orders terms by document frequency, and cascades strict-AND → relaxed-AND → OR, fused with reciprocal rank fusion. On the bundled index this considers ~21 judgments instead of ~1,804 for a typical sentence.
4. **The model writes; the evidence decides.** With a model configured, `app/legal_engine.py` retrieves from the validated reading and generates an answer, then validates every legal claim against an exact quote from a retrieved approved source. A failed quote triggers one constrained repair, then a claim-free answer. Without a model, the app displays passages and states that no conclusion was drawn.
5. **Voice-first shaping, with real barge-in.** Every turn returns `spoken` (short, markdown-free, safe to read aloud) separately from the full text, plus exactly one `next_question`. The stream endpoint speaks an acknowledgement before the reader is even called, streams safe sentences of an intake reply as they are generated, overlaps retrieval with the reading, and reports its own `timing` (route, first speech, total) per turn. A legal answer is never streamed early: no unverified proposition reaches audio. While the answer is playing, local microphone RMS is checked against the stop plan: a backchannel like "haan" never cuts the assistant off, a stop word or 200 ms of speech does, instantly and locally, while the server closes the Timbre synthesis stream and truncates the stored answer to what was actually heard. When the caller pauses, punctuation-aware endpointing sends the turn without asking them to press a button. [app/voice.py](app/voice.py) holds the thresholds; `/api/voice/metrics` reports measured first-audio and stop latency.
6. **A checklist behind a conversation, not a form.** [data/workflows.json](data/workflows.json) defines finite-state intakes (unpaid wages, police FIR refusal). The model talks and can request a step change, but [app/workflow.py](app/workflow.py) decides which transitions exist and which facts each step requires, so a step re-asks instead of skipping a document or a safety check. Mid-intake the model phrases the one missing fact in the caller's language, the reviewed steps are shown once rather than under every answer, a terse or misread answer keeps the intake, and a second matter is held for the completion turn. An emergency aborts an *active* intake. A question that matches no workflow is not rejected: it is read, retrieved and answered like any other legal question.
7. **Tools, with a closed boundary.** [app/tools.py](app/tools.py) exposes five OpenAI-format tools: search approved law, search judgments, fetch a reviewed procedure, record a caller-stated fact, advance a workflow. Retrieval runs mid-turn and its passages join the evidence set before claim validation; every write is checked against a verbatim quote from the caller. There is no network, shell, file or action tool.
8. **State is a ledger, not a transcript.** `app/facts.py` keeps caller-stated facts (state, date, amount, notice, stage) with their source and turn, asks only the most material missing one, and records a conflict instead of silently choosing. It survives restarts and is inspectable at `GET /api/session/{id}/state`.
9. **Identifiers never travel by accident.** `app/privacy.py` masks Aadhaar, PAN, phone, email and account numbers before the message reaches a model — not only in the logs — so a model that never receives an account number cannot repeat one in a draft. It tells the caller when their message contained one, and masks them in a review handoff unless the sender ticks an explicit include option; the modal shows exactly what would be sent before anything is.
10. **Every claim is inspectable, and so is every spoken sentence.** A claim is accepted only when its quote is an exact substring of the retrieved source *and* that quote contains every number the claim asserts; when a model is configured, a second cheap call then asks the entailment question the substring test cannot answer — does this passage state this proposition? Finally the prose the caller actually hears is passed through the same numeric check, so a model cannot state "section 138" or "within 30 days" in a sentence with no claim behind it. Anything unsupported gets one repair, then the deterministic grounded reply. The response carries a `trace` (stage timings, route reasons, candidate counts) and a `verification` block reporting `entailment: model_checked` or `not_reviewed` — a quote match is provenance, not a reviewed legal conclusion.
11. **A session id is not a credential.** Creating a session returns a separate bearer token once; only its hash is stored, and every endpoint that touches a conversation requires it (the speech sockets carry it in the query string because a browser socket cannot send a header). Sessions live in SQLite for four hours; `DELETE /api/session/{id}` removes the conversation and everything saved in it. Rate limits are per session, per address and optionally a global daily cap, counted in the shared database.
12. **A reminder is delivered, not just stored.** There is no notification service, so a saved reminder is returned as an RFC 5545 calendar file (`GET /api/session/{id}/reminders/{n}.ics`). The event says it is a reminder to act, not a deadline the assistant calculated.
13. A person is contacted only after the user selects the connection action and separately consents to sharing the displayed summary. Nothing is filed, sent, paid, or submitted.
14. **Court-grade document generation.** When a formal legal notice, representation, or application is drafted, `app/pdf_generator.py` produces publication-grade legal PDFs formatted to Indian court standards using ReportLab: standard A4 dimensions with formal legal margins (0.75" / 54pt), automated two-pass running headers ("IN THE COURT OF..." / document title) and footers (page counters "Page X of Y" + statutory disclaimers), dynamic leading to eliminate overlapping text lines, auto-fit table layout, and formatted verification/signature blocks. All drafts can be immediately edited in the web interface and downloaded as signed PDFs.

A structured model completion is retried once on a transient provider error or unparseable JSON, inside an 18-second whole-turn budget; after that the turn degrades to retrieved passages instead of hanging a spoken reply. The reading stage asks for a JSON Schema (guided decoding on vLLM/SGLang), falls back to `json_object` if the server rejects schemas, and keeps its own 7-second deadline so a voice turn is not waiting on nuance. If the reader is unavailable the lexical net handles the turn and the trace says `source: lexical` — the app never pretends a model read the call.

The AI can also respond to requests outside the examples in the sidebar. It can explain retrieved law, compare similar case records, help interpret pasted notice text, organize a reported timeline/evidence, list caller-provided dates without asserting statutory deadlines, translate or rephrase, prepare a situation-specific legal-aid summary or questions, and draft editable documents. It cannot perform court filing or a live precedent-treatment check in this build.

## Check the promises, not just the code

```bash
.venv/bin/python -m pytest tests -q      # 359 unit and API tests
.venv/bin/python evals/run_evals.py      # 18 behavioural cases on the real turn path
node --test tests/js                     # browser speech-queue logic, no browser needed
node --check static/app.js               # client parses cleanly
.venv/bin/python -m scripts.latency_bench --runs 7   # measured turn budget, no provider needed
```

All of it runs in CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) on every push, along with a container build that has to answer `/api/health`. The pipeline is deterministic because the degraded mode is a supported mode: no model and no Gnani key are needed for any of it.

The eval cases are the product promises, not happy paths: a threat in English and in Devanagari, a draft requested during a threat (must be withheld), an OTP debit (must be urgent, must not produce a document), an eviction, custody, a handoff request, an instruction injected into the caller's message, a first-turn opener, and two fact-capture turns. A failure prints the exact expectation that broke. It runs with no model configured and re-checks the generated path unchanged once a model is set.

## The three requested datasets

The Hugging Face dataset is included in `data/indian-law-dataset.json` (24,607 rows, Apache-2.0). Startup indexes it for discovery, but the assistant is instructed never to cite its answers as law.

The full Supreme Court corpus (about 52 GB upstream) and 25 High Court corpus (about 1.25 TiB upstream) are too large to copy in full. The searchable index currently holds **2,046 judgments with joined metadata**: 1,679 Supreme Court English judgments (2024–2025), 244 Karnataka High Court judgments (2024) and 123 Sikkim High Court judgments (2024).

Every one now carries a decision date and a CNR; 82% carry a citation (the High Court corpora do not publish citations, which is why that figure is not 100%). Before the importer was fixed, the bundled sample carried **no** decision dates, citations, benches or CNRs at all, so nothing could be cited or date-filtered.

Alongside that sample, `legal_pipeline/` builds a second, much larger store: **10,588 judgments — 1,600 Supreme Court, 8,228 High Court and 760 district court — chunked into 94,843 embedded passages**, each labelled with its court tier, jurisdiction and language. It is one SQLite file (531 MB) with the passages, their int8 vectors and an FTS5 index, produced from `opennyaiorg/InJudgements_dataset` by:

```bash
cd legal_pipeline && pip install -r requirements.txt
export HF_TOKEN=...                     # the dataset is gated; accept its terms once
python build_opennyai_embeddings.py      # download + embed into one file
```

Point the app at it with `JUDGMENT_CORPUS_DB` if it lives elsewhere. Both stores are searched and their rankings are fused; [`docs/retrieval.md`](docs/retrieval.md) records the rules that fusion follows and the failures each one prevents.

This remains a sample, not coverage of 25 High Courts or 77 years. [`docs/data-and-compute.md`](docs/data-and-compute.md) has the storage and compute arithmetic for the full corpora; [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md) has attribution and licensing. Exact install commands are in the section above.

Install optional ingestion dependencies:

```bash
python -m pip install -r requirements-ingest.txt
```

Refresh the Hugging Face data from the dataset hub:

```bash
python scripts/install_hf_law_dataset.py
```

Install judgment text **with its structured metadata** (decision date, citation, bench, CNR, parties). This is the recommended path: the earlier subset carried none of those fields, so it could not be cited or date-filtered.

```bash
python -m pip install -r requirements-ingest.txt

# Supreme Court, recent years, metadata joined
python scripts/install_judgments.py --dataset sc --years 2024 2025 --max-download-gb 2

# One High Court bench, metadata joined
python scripts/install_judgments.py --dataset hc --years 2024 --court 29_3 --bench karhcdharwad --parts 1 --max-download-gb 2

python scripts/reindex_judgments.py
```

The importer enforces a download budget, refuses to start if the budget does not fit on disk plus 2 GiB headroom, skips already-ingested archive members so a rerun resumes, and **fails the run** if more than 5% of new records arrive without a decision date — the silent-metadata bug cannot return unnoticed.

High Court codes come from `data/hc_court_jurisdictions.json`, which covers all 25 courts and 45 benches. An unknown code stops the import instead of being mislabelled as national law.

The legacy partitioner is still available for a pure archive sync without metadata:

```bash
python scripts/sync_public_judgments.py --dataset sc --years 2020 --max-download-gb 1
python scripts/sync_public_judgments.py --dataset hc --years 2024 --court-code 29_3 --max-download-gb 1
```

High Court codes come from the upstream dataset. The normalizer extracts searchable PDF text into `.runtime/legal-data/judgments.jsonl`; the app indexes it with SQLite FTS5. Scanned PDFs with no text layer need a separately evaluated OCR pipeline. Do not set an all-years budget or mirror the 1.25 TiB corpus to a laptop.

## Legal evidence policy

Only source records in `data/legal_sources.json` marked `approved` are eligible to support legal claims. Exact quote matching verifies that a claim cites text actually retrieved; it does **not** prove that the reasoning, interpretation, completeness, or application to the caller is correct. This starter contains a very small reviewed-source set (Constitution Article 22 and NALSA FAQ). It does not contain complete, current legislation for every state or service.

Judgments are examples of what a particular court decided. This project has no citator that checks appeals, overruling, later treatment, or binding status. The model must name the court/date and explain that similar facts can produce a different result; it cannot present a case as current governing law based only on retrieval similarity. No “near-perfect” accuracy or legal outcome is promised.

The upstream Evon model card explicitly recommends evaluation on the target language/domain and warns against high-stakes legal use without safety evaluation. This prototype is a development system, not a public legal service. Before deployment, add current official state and central statutes, treatment-aware case metadata, expert-reviewed evaluation, secure user accounts, encrypted persistence, abuse monitoring, and a staffed consent-based legal-aid network.

## API and project map

- `app/main.py`: FastAPI routes, SSE streaming, approval-gated actions, structured logs, and Gnani WebSocket proxies.
- `app/understanding.py`: the model's reading of a turn, and the validator that keeps it inside closed vocabularies and grounded evidence.
- `app/routing.py`: route selection from that reading, over a lexical net that keeps working when the provider does not.
- `app/procedural.py`: validated loader and matcher for the curated procedure set.
- `app/retrieval.py`: query construction, Romanized and colloquial-legal expansion, synonym AND-groups, rank fusion, dense hook.
- `app/judgment_corpus.py`: passage-level retrieval over the embedded corpus — pooled BM25, int8 cosine scan, per-store normalisation, tier and jurisdiction filters.
- `app/legal_engine.py`: `analyze` → `compose`, voice shaping, claim and prose validation, trace.
- `app/pdf_generator.py`: court-grade ReportLab PDF generation with standard legal margins, dynamic leading, running headers/footers, and auto-fit tables.
- `app/sessions.py`: SQLite session store, session tokens, shared rate limiting and the live speech turn.
- `app/reminders.py`: RFC 5545 calendar delivery for a saved reminder.
- `app/knowledge.py`: approved legal sources, cascading retrieval, secondary Q&A.
- `app/case_corpus.py`: judgment indexing, rarity-ordered cascading retrieval, and the fusion of both stores.
- `prompts/`: model scope and planning behaviour, not fixed user-facing answers.
- `tests/`: 359 tests covering routing, retrieval, the passage corpus, procedures, claims, the prose guard, auth, sessions, latency paths, PDF drafting, and the API; `tests/js/` covers the browser speech logic.
- `scripts/eval_retrieval.py`: measures candidate-pool reduction and term focus.
- `data/legal_sources.json`: reviewed primary source excerpts.
- `data/procedural_kb.json`: curated procedures with review status and official entry points.
- `data/service_catalog.json`: capability boundaries and side-effect policy.

Routes: `/`, `/manifest.webmanifest`, `/sw.js`, `/api/health`, `/api/config`, `/api/workflows`, `/api/procedures`, `/api/sources`, `/api/session`, `/api/turn`, `/api/turn/stream` (SSE), `/api/debug/analysis`, `/api/action/confirm`, `/api/review/request`, `/api/session/{id}/artifacts`, `/api/session/{id}/state`, `/api/session/{id}/reminders/{n}.ics`, `DELETE /api/session/{id}`, and WebSockets `/api/voice/prisma` and `/api/voice/timbre`. Everything that touches a conversation needs `Authorization: Bearer <session_token>`.

## One interface, desktop and installed app

There is one layout, not a separate mobile build. Above ~900px it is a three-column page with a sidebar and a right rail; below that it becomes a single column with a fixed bottom bar for **Talk · Guides · Saved · New**, because the sidebar held the only controls for a new conversation and saved work and it was simply hidden on a phone.

The same route is installable as an app: `static/manifest.webmanifest` declares standalone display, `static/sw.js` caches the shell and serves it offline, and `app/main.py` serves both from the root — a worker under `/static/` could only control `/static/`. Installing is offered by a real button driven by `beforeinstallprompt`, and on iOS, which has no such event, the button explains Share → Add to Home Screen instead of hiding.

Three details decide whether an installed page feels like an app or a shrunk website:

- **Safe areas:** `viewport-fit=cover` plus `env(safe-area-inset-*)` on the shell, the fixed bar and the dialogs, so the notch and home indicator do not sit on the composer.
- **The keyboard:** iOS shrinks the visual viewport without resizing fixed elements. `--keyboard-inset` is measured from `window.visualViewport` and lifts the bar, or it would cover the composer you are typing into.
- **Focus zoom:** any control under 16px makes iOS zoom the page on focus, so the composer, selects and draft fields are 16px on small screens.

Nothing under `/api/` is ever cached or intercepted by the service worker: a cached turn would be a cached legal answer. Static assets are matched on the full URL including `?v=`, so bumping that number in `index.html` is what reaches a returning user.

Icons are generated, not committed as opaque binaries: `python scripts/make_icons.py` writes the 192/512/maskable/apple PNGs and the favicon from code, using the same orb-from-the-logo mark.

## Tests and measurement

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q          # 359 tests, no model or provider keys needed
python scripts/eval_retrieval.py   # candidate-pool and term-focus table
```

The test suite runs without an LLM endpoint or Gnani credentials: routing, procedures, retrieval, claim validation and the API all have deterministic paths. `tests/test_api.py` asserts the degraded mode directly, so a regression that starts pretending to have a verified answer without a model will fail the suite.

Architecture details: [`docs/architecture.md`](docs/architecture.md). Runtime design: [`docs/v1-overview.md`](docs/v1-overview.md). Dataset and source notes: [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md). Storage, compute and retrieval sizing at full corpus scale: [`docs/data-and-compute.md`](docs/data-and-compute.md).
