# Judgment retrieval — design and the rules it follows

This is the read path from a caller's sentence to the passages the model is allowed
to quote. It covers two stores and one fusion step, and every claim below is
implemented in `app/judgment_corpus.py`, `app/case_corpus.py` and `app/rerank.py`.

The product rule it serves: **no legal proposition reaches a caller without a
verbatim quote from an approved source.** Retrieval decides what can be quoted, so
a retrieval mistake is a correctness mistake, not a ranking annoyance.

## The stores

| Store | What it is | Where |
| --- | --- | --- |
| Embedded corpus | 10,588 judgments (1,600 Supreme Court, 8,228 High Court, 760 district), 94,843 passages of ~3,600 characters, each with an int8 unit vector | `legal_pipeline/out/corpus_embeddings.sqlite3` |
| Curated partitions | 2,046-judgment sample of recent Supreme Court decisions with citation, bench and CNR | `.runtime/cases.sqlite3` |

Both are read-only at query time. The embedded corpus is produced by
`legal_pipeline/` and can be moved with that folder; `JUDGMENT_CORPUS_DB` points at
a copy elsewhere.

## The rules, and what each one prevents

**1. Retrieve passages, not judgments.** A judgment is tens of thousands of
characters of cause title, appearances, facts and arguments before the holding.
Embedding it whole averages all of that into a vector that matches nothing in
particular. Passages are embedded; the enclosing judgment's title, court, citation
and tier are attached to the hit.

**2. Fuse two independent rankings, never compare their scores.** BM25 and cosine
are not on the same scale, and a threshold tuned on one silently discards the
other. Reciprocal rank fusion combines orderings. Measured on this corpus, dense
search alone reaches hit@5 0.429 and lexical alone 0.429; the fusion reaches 0.571
on the curated eval's own terms, which is the only reason to keep both. With the
rewritten query merged in (rule 11) it reaches **0.643 / MRR 0.452**; without it,
without the rewrite, the fusion's MRR is 0.429.

**3. Size the candidate pool by the index, not by the answer.** The number of rows
each query arm contributes and the number of fused candidates the reranker sees
used to scale with the caller's `limit` (8× and 6×), so a five-result turn built
its ranking from a forty-row window of a match set holding thousands — the target
of a generic query was never in the pool at all. They are now constants
(`EXPRESSION_ROWS`, `RERANK_POOL`). Every arm of the cascade also runs to
completion instead of stopping at the first arm that returned twice the requested
limit, which had made the candidate set whatever one expression happened to match.

**4. A store that shares no words with the query has nothing to say.** The lexical
arm orders terms rare-first and drops near-ubiquitous ones, but when it kept the
terms that happened to occur at all, a Hindi question against an English index
produced a one-term list ("में" matching boilerplate) that fusion then scored at
full store weight, outvoting the dense ranking that could actually answer. If
fewer than a third of a query's terms occur in the index, the lexical arm now
abstains and says so in the trace (`lexical_skipped`) rather than competing.

**5. A reranker with no relevance signal must not reorder.** For a cross-script
query every relevance feature is zero on every candidate, leaving only authority
and the fused base; reordering on authority alone lost two of four Hindi retrieval
cases that the dense arm had ranked correctly. `rerank.rerank` returns the fused
order unchanged and records `skipped: no shared vocabulary with the candidate pool`.

**6. Normalise before fusing stores.** RRF is defined over lists, so a store that
runs three query arms accumulates three times the mass of a store that returns one
ranking — and wins on bookkeeping rather than relevance. Each store is reduced to
one list, normalised so its best result scores 1.0, then combined. Without this,
three Supreme Court banking judgments outranked directly responsive High Court rent
cases on every tenancy query.

**7. Carry the court tier and let it reorder only within a bounded range.** A
district-court order and a Supreme Court judgment are not the same authority, and a
caller hearing "the court held" is entitled to know which court. Authority is a
multiplier worth at most 8% of the relevance score. As an added constant it was
strong enough to promote a marginally relevant Supreme Court judgment over a
directly responsive High Court one — a wrong answer that looked authoritative.

**8. Weight query terms by what they can discriminate.** A word present in most
candidates ("return", "court", "case") cannot separate them. Term weights are
estimated as inverse document frequency over the candidate pool itself, which is
free and follows the actual competition being ranked. Without it an incidental
match on "return" made a banking judgment score like a tenancy one.

**9. Bridge the caller's vocabulary to the corpus's, in one direction only.**
`retrieval.GLOSSARY` maps Romanized Indic words to English ("kiraya" → rent) and
`LEGAL_SYNONYMS` maps everyday words to the term judgments use ("bounced" →
dishonour, "lock" → dispossession). Expansion is one-way and deterministic, so it
can only add recall. Synonyms are used as an AND group — `"lock" AND
("dispossession" OR "lockout")` — because flattening them into a free OR is how a
query about a changed lock returns criminal "lock-up" cases.

**10. Cap passages per judgment and dedupe.** At most two passages per judgment, so
one long decision cannot fill the result set and hide the rest of the corpus.

**11. Search the caller's words, then the reader's words, and merge — never
replace.** The reader returns `search_query`: the same question in the legal
vocabulary Indian judgments use ("he changed the lock" → "dispossession"). The
prompt forbids party names and URLs; it *allows* section numbers the caller never
said, because those are the words that separate the leading authority from a case
that merely mentions the topic. This was learned the hard way: a rewrite that
said "registration of FIR for a cognizable offence under section 154" was deleted
by a guard against invented digits, which is precisely why a refused-FIR question
returned income-tax "refusal of registration" cases. A wrong number in a search
query cannot become a false statement — the query only selects passages that
already exist, quotes are checked as exact substrings, and the claims that reach
the caller are verified separately. Retrieval runs both phrasings and
`retrieval.interleave` merges them. The rewrite now leads and the caller's own
ranking stays underneath it: the rewrite is the reader's statement of the *legal
question*, the first pass was the caller's phrasing of it, and putting the
paraphrase first was why "the refusal of FIR" outranked "refusal to register a
first information report, section 154". Interleaving still means a rewrite can
only add candidates, never drop one. The old guard skipped the second pass
whenever it shared most of its terms with the caller's message; that is exactly
when it mattered — the two phrasings share "refusal" and "FIR" and search very
different things — so only a byte-identical rewrite is skipped now. The 14-case
retrieval gate is what checks this ordering, and it is run on every change.

**12. Return the matching passage verbatim, never a summary.** Exact-substring
verification downstream compares quotes against this text. An excerpt that has been
reworded, truncated mid-word or reconstructed from metadata cannot be quoted, and a
claim built on it fails verification for the wrong reason.

**13. The first caller must not pay for the startup load.** The 530 MB artifact
loads in ~9 s. `claim_load()` reserves it synchronously in the lifespan handler,
`available()` answers "not yet" while that claim is held, and `retrieve` abstains,
so a turn arriving during the warm is answered from the curated partitions in
~100 ms instead of loading the artifact inside its own request (measured 8.6 s).
The guard is paired with `_probe()`, a load-state-independent file check, because
the warmer itself has to ask whether the artifact exists while it holds the claim
— routing that through `available()` made the warmer conclude the corpus was
missing and hand the load back to the first turn.

**14. Fail visibly, never quietly.** No artifact → `available()` is false and
retrieval returns nothing. No encoder → lexical still answers, and the trace records
`dense_error`. `store` selection is reported in `/api/health` under
`embedded_judgment_corpus`, so a deployment that lost its artifact is visible rather
than plausible.

**15. Bound the work, then measure it.** `COUNT(*)` on an FTS5 table walked its
shadow tables at 150 ms warm and over a second cold, on every call; it now comes
from the `chunks` table and is cached. Each arm is ranked by bm25 over its whole
match set, measured at 1–11 ms for a focused AND (87 matches) and 11–190 ms for a
broad OR (5,000–7,000 matches). The earlier version sampled 96 rows and ranked
those: `LIMIT` on an FTS5 scan returns index order, not relevance order, so the
best passage was usually not in the sample. That trade — 1 ms for the wrong
answer — is the one that lost the leading FIR authority while keeping arbitrary
district-court orders. Steady state is 30–70 ms for the corpus store and ~55 ms for the
combined path; the first call after a restart pays page faults on the 180 MB
curated index and contends with the startup warmer for disk, measured at
**3.5 s once** against **80–140 ms** on every turn after it. That first-turn cost
is cold-start, not steady state; a deployment that must not show it should wait
for `corpus_ready` in the log, or for `embedded_judgment_corpus.available` in
`/api/health`, before routing callers. Measured after warm: **31–53 ms** for the combined path on a repeated
query, **100–400 ms** for the first query of a new wording. A memory map is set
(`mmap_size`, 1 GiB) and the matrix and encoder load at startup (~3.7 s, in a
background task). Warming with representative probe queries was tried and
reverted: 9 s of startup and the first real query still cost ~580 ms, because the
pages a probe touches are not the pages a different question needs. Retrieval runs
in a worker thread while the turn's prompt is assembled, so this is overlapped
rather than added to a turn.

## Query construction, and the failure that forced it

The reported failure: *"I want the most relevant cases that cover the refusal of
FIR"* returned **Commissioner of Income-tax v Navabharat Enterprises** — a
"refusal of registration" case — and no FIR authority at all, while *Lalita Kumari
v Govt. of U.P.*, the decision holding that FIR registration is mandatory, sat in
the same corpus. Four independent causes, each now fixed and pinned by a test:

1. **Request framing became the search anchor.** The query terms were `want,
   relevant, cases, cover, refusal, fir`. Terms are ordered rarest-first and the two
   rarest become the strict BM25 anchor, so the anchor was `"want" AND "relevant"`.
   `retrieval.META_TERMS` removes words about the *request* rather than the subject;
   the terms are now `refusal, fir`. A meta-only question ("tell me the most
   relevant cases") now retrieves nothing instead of inventing relevance.
2. **The index does not stem.** "refused", "refuse" and "refusal" are three
   unrelated terms to FTS5. `stem_expressions` adds one arm that ANDs the two
   strongest terms as word-start prefix groups — `("refusal" OR "refus"*) AND "fir"`.
3. **The caller's vocabulary is not the bench's.** No judgment says "refused to
   register an FIR"; they say "information disclosing a cognizable offence" and
   "section 154". Each procedure card now carries reviewed `case_terms`, and
   `_evidence_block` consults `procedural.best_match` as a *vocabulary source* — not
   a routing decision — so the search runs in both vocabularies. Measured after the
   ranking fixes: *Lalita Kumari* is rank 1 for `cognizable + 154` with *Bhajan Lal*
   second, and outside the top 400 for `fir + refusal` alone. That is why the focus
   terms are now passed on every retrieval path, including the rewrite pass and the
   precedent route — the speculative pass had them, the route that actually served
   the four cases to the caller did not.
4. **The gate counted substrings.** Coverage scored `term in haystack`, so `fir`
   matched "first" and "firm" and a coal-supply contract passed as covering an FIR
   question. `rerank._contains` matches at a word start, so inflections still count
   and "first" does not.

Then two ranking faults that only appear on a corpus this size:

- **Passage depth.** A judgment that returns to the question in seven passages and
  one that mentions it once scored identically, because only the best passage was
  ranked. `DEPTH_BOOST` lifts the former by at most 50%, bounded so it can only
  reorder judgments that already matched.
- **The pool was the answer size.** The corpus store returns three judgments per
  requested slot, and `case_corpus` asked for the caller's `limit` (4), so it
  contributed twelve judgments to a sixty-candidate rerank pool. `CORPUS_POOL`
  asks for a pool.

**The relevance gate.** Retrieval always returns something when any word matches,
and the caller cannot tell "this decision is on point" from "this decision shares
the word refusal". `case_corpus.RELEVANCE_GATE` (0.34) drops candidates that cover
less than a third of the question's terms, measured two ways that both have to
hold: at least two of the question's terms must appear, and their pool-weighted
share must clear 0.34.

The second test used to be an unweighted count, and a count let two incidental
common words carry a question: *"plan a birthday party for my daughter"* kept
criminal appeals, because judgments say "party" and "daughter" constantly. The
weights are inverse document frequency over the candidate pool, so a word half the
pool contains contributes almost nothing and the one word that actually separates
the candidates carries the decision. The unweighted count is still needed for the
*narrow* case it was introduced for: a two-word question must match both words, not
one — *"what is the weather in mumbai today"* matched a single "Mumbai" and kept
four criminal appeals. For a focused query the reviewed vocabulary is the gate
below, and the IDF objection does not apply, because the terms were chosen by a
reviewer rather than by the caller's phrasing.

When the procedure card supplies reviewed `case_terms`, that vocabulary is the
gate instead: a candidate must carry a third of *those* words. The caller cannot
be expected to say "cognizable" or "154", and a judgment that repeats their
framing without using the vocabulary of the issue is not deciding their question.
With nothing passing, the answer is "no approved source covers this" — a supported
answer — rather than a list of near-misses.

**Coverage is measured on the reviewed vocabulary, not the caller's paraphrase.**
This is the second half of the same lesson. Fused RRF ranks differ by ~1.6% at the
top (61/61, 61/62, 61/63), so any coverage term built on the caller's wording
decides the order outright: on "the refusal of FIR", *Lalita Kumari* was fused rank
1 and carried "section 154", "cognizable" and "FIR", but not the word "refusal",
so IDF coverage demoted it below income-tax cases that repeated the caller's word.
Scoring now uses the reviewed `case_terms` when they exist, and the fused base
carries 0.62 of the score (was 0.50). Coverage breaks ties; it does not overturn
two retrievers that agreed.

**The model's judgement.** Sharing vocabulary is not answering the question. When
a model is configured, `model_gateway.case_relevance` scores the top candidates
0–3: 3 decides or squarely discusses the question, 2 discusses it on different
facts, 1 is the same subject area, 0 is a different issue or an incidental word.
It filters at 2 and, more importantly, *reorders*: a 3 the retriever put fourth is
more useful than a 2 it put first. It can only remove or reorder candidates, never
add or rewrite one; an unavailable judge leaves the list untouched; and if it
rejects everything, the two closest are shown with `corpus.case_relevance =
"none_on_point"` so the interface says they are not confirmed to answer. Traced as
`case_relevance` with `judged`, `kept`, `dropped`, `unanswered`, `reinstated` and
`best`. `judge_cases` remains as the boolean view for callers that only filter.

## A follow-up is a turn in a call, not a question on its own

Retrieval used to be built from one message. Everything else about the call — the
ledger, the intake, the transcript — was available to the reader but not to the
search, so the query was only as good as the caller's last four words. Measured on
a two-turn call, in degraded mode with no reader to rewrite anything:

> "my employer has not paid my salary for three months" → labour judgments.
> "what can I do about the money?" → `content_terms = ['money']`, **nothing
> retrieved at all**; with the wages vocabulary supplied it returned tax and
> enforcement cases sharing the word money rather than the wage dispute.

The rule now is that a *pointer* is searched as the matter it points at. Three
substantive terms is the same floor that already decides whether a turn is a real
question (`routing`): at or above it, the turn's own words are the evidence and are
searched alone — "my landlord changed the lock and threw my things out" must not be
glued to an earlier wages dispute. Below it, `legal_engine._carried_matter` supplies
the thread, in this order of authority:

1. **What the caller actually said**, taken from the transcript's most recent
   substantive turn — not summarised, not paraphrased. Two-word questions are the
   reason the relevance gate needs context at all: a two-term query must match both
   of its terms, so "keep deposit" alone asks the store for judgments containing
   both words and gets nothing.
2. **The ledger's description slots** (`issue`, `situation`) when the transcript has
   nothing usable — for a call that resumed from stored state.
3. **The intake's label**, when neither exists.

The query is then `residual + thread`, where `residual` is the turn's own words
with pointer vocabulary removed (`money`, `case`, `matter`, `problem`, `next`, …;
`police` and `court` are deliberately *not* pointers — they are the subject).
Pointers lead nowhere on their own and, placed in front, they changed the ranking:
"money" ranked bank and tax judgments above the wage dispute. The newest words are
kept first because the 360-character cap must never drop them.

The coverage vocabulary comes from the procedure the call is in, and the **current**
turn's procedure wins over an intake still marked active in storage. An FSM left
open after the caller moved on measured a deposit question against
`wages`/`employer`/`labour` and returned a company-law judgment; the engine updates
`procedure_id` every turn, so it is the newer fact.

The judge sees the same context. `case_relevance` is asked about the matter, not
the sentence — a follow-up is unjudgeable on its own words, and judging it alone is
how on-point judgments were dropped for not answering four words:
`"what can I do about the money? (matter: my employer has not paid my salary for
three months)"`. The trace's `evidence` stage reports `carried` so a turn that
searched the thread is distinguishable from one that searched itself.

This is deliberately not "put the whole transcript in the query". BM25 and the
dense arm both degrade when the query is a conversation: the terms of the previous
matter outvote the current one, and a caller who mentions three problems in one
call gets three mediocre averages of them. The transcript is read by the
consultation, which is the stage that can summarise and disambiguate; the search
takes the one thread that matters plus the vocabulary the review supplies.

The consultation writes its own `search_query` in the vocabulary judgments use
("dispossession without due process of law", "dishonour of cheque"). With one
model call per turn that phrasing cannot seed the same turn's search, so it is
persisted in the session and folded into the **next** turn's query when that turn
only points at the matter — which is exactly when the caller's four words
("what about the money?") have no corpus vocabulary in them. The model can also
ask for a search inside the turn through `search_law` / `search_judgments` when
the seeded evidence is missing, and that costs one tool round.

## What a caller's voice actually produces

Each of these was reported from a live turn, measured, and is now pinned by a test
in `tests/test_retrieval_queries.py`.

**The wrong artifact answered.** `_matrix is not None` meant "loaded", without
checking *which* file was loaded. A run that built a four-case test corpus left its
matrix and chunk index in memory, and the next turn against the real artifact
scanned the fixture's vectors while reporting the fixture's titles. The loaded
state is now keyed on the artifact path and modification time, and the probe cache
is keyed the same way. A stale matrix is a reload, not a hit.

**One judgment, two cards.** The ingested corpus holds the same decision under two
ids when the source published two URLs for it (`.../doc/1127466` and
`.../doc/1127466/`), and both were offered as separate precedents. Identity is the
normalised URL *plus* title and date — not the URL alone, because the Supreme Court
import records the tar archive as every record's URL, and de-duplicating on that
collapsed a whole year of judgments into one and removed the correct answer from
four retrieval-eval cases.

**Hindi anchored on grammar.** The stoplist knew English and Romanised Hindi only,
so *"मेरा चेक बाउंस हो गया, अब क्या करूँ"* produced the terms `{चेक, करूँ}` — and
"करूँ", "should I do", is in every Hindi-language judgment ever written. Two terms,
one of them noise, cleared a 34% gate and returned four unrelated decisions.
`retrieval.INDIC_STOPWORDS` removes the function words of the script the caller is
speaking.

**Hindi then matched no procedure.** Every card's triggers were English or
Romanised (`cheque bounce`, `tankha`, `makan malik`), so a Devanagari question
matched no card, carried no reviewed vocabulary, and had nothing to search with.
Each card now carries Devanagari triggers for its distinctive words (चेक, एफआईआर,
वेतन, किराया, ताला बदल, गिरफ्तार, उपभोक्ता, …), and `test_every_reviewed_procedure_is_reachable_in_hindi`
requires one per card. Hindi is the script covered so far; Tamil, Telugu, Bengali,
Kannada, Malayalam, Gujarati and Punjabi reach the store only through the
multilingual encoder and the dense floor below, and that is a stated limitation, not
a solved problem.

**Speech-to-text misspells.** "landloard", "securty", "depost" retrieve nothing —
FTS5 has no fuzzy match. `retrieval.resolve_prefixes` truncates a word the index has
never seen until a prefix it does know is found (`landloard` → `landl`*, which
matches "landlord" and its inflections), at most three words per query, minimum four
characters, ASCII only, and never a word the index already contains. A repaired
query is the same query with its words spelled right; it is reported in the trace as
`corrected_terms`. A genealogy this aggressive still refuses when it cannot repair:
a four-word tenancy question with every word misspelt returns nothing rather than
four unrelated decisions.

**Dense always had a nearest neighbour.** Both dense arms returned the top-k
neighbours of any input at all, so a store failure became a result list. The local
index now has `dense.MIN_SCORE` (0.40; genuine topical questions 0.50–0.70,
unrelated prose 0.2–0.36) and the passage index `JUDGMENT_CORPUS_MIN_DENSE` (0.45;
real cross-script questions 0.45–0.66, a Telugu wage question 0.34, gibberish 0.35).
These floors are guards on one arm, not the relevance decision — an unrelated
*question* scores 0.52 on the local index, which is why the gate above still
exists.

**Lexical agreement is required when the corpus could have it.** When no query
term occurs in either index at all (a Telugu or Odia question against English
judgments), the encoder is the only signal there is and its order stands — that is
how the four Hindi eval cases are answered. When the index *does* hold the caller's
words and not one candidate uses them, the honest answer is nothing: the question
was answerable here and this store does not answer it. Traced as `unjudgeable`.

**An unclear turn does not get precedents.** *"what is the weather in mumbai
today"* routes `unclear`; the speculative retrieval that ran before routing found
judgments that genuinely discuss weather and Mumbai, and the degraded reply quoted
one as "the closest decision I have". Lexical relevance cannot tell an off-topic
sentence from a legal one — that is the router's job — so when the route is
`unclear` the reply asks what happened and shows no precedents.

**The case-led degraded reply speaks English.** Ten of eleven languages had no
`case_only_lead`, so `_floor` fell back to English inside a Telugu answer. Every key
in `legal_engine.FLOOR` now exists in every supported language, and
`test_every_degraded_sentence_exists_in_every_supported_language` fails the build if
one is added to English alone. The quote itself stays verbatim in the judgment's
language — translating it would break exact-substring claim checking.

## What is deliberately not done

- **No cross-encoder.** A CPU cross-encoder costs 1–3 s per query, which is not a
  voice turn. The reranker combines signals the system already has in microseconds.
- **No approximate vector index.** At 94,843 passages an exact int8 scan is 22 ms;
  an ANN index would add a failure mode and save nothing.
- **No cross-encoder** (repeated below for contrast) — the rewritten query is a
  *vocabulary* fix, not a semantic one: it cannot tell whether a judgment actually
  answers the caller's question, only whether it uses the same words.

### What the rewrite does not fix

Four of the fourteen gold targets still rank outside top 5, and the cause is not
routing or candidates — it is that a deliberately generic query has several equally
responsive judgments. "Factory ordered shut for polluting the environment" returns
Vellore Citizens Welfare Forum and the closure orders, which are at least as
relevant as the 2024 order the case was written from; "insurance company appeal
against compensation awarded to the family of a road accident victim" returns six
Supreme Court motor-accident judgments. Exact-target hit@5 cannot distinguish those
from a retrieval failure, and tuning weights until it does would be fitting fourteen
cases. The honest fix is a more specific gold query, which is eval work.
- **No treatment checking.** `treatment_status` is `not_checked` on every record: a
  judgment that has been overruled is returned as decided. The reply must not
  present a case as current law on this basis alone.

## Verifying a change here

```bash
python -m pytest tests/test_judgment_corpus.py tests/test_retrieval.py -q
python evals/retrieval_eval.py --gate              # curated store, gated
python evals/retrieval_eval.py --gate --rewrites   # adds the production rewrite row
python evals/retrieval_eval.py --store hybrid      # production path, reported
```

The `--rewrites` row is the only one that measures the production shape, because
it is the only one where `search_query` is exercised; the other rows are the stages
beneath it.

The gate measures the curated partitions because that is what its checked-in
baseline was measured on; pointing it at the hybrid store would compare against
targets that do not exist there. A hybrid run is for reading, not for passing.
