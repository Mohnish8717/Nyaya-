# Nyaya v1.0 — what changed and why

This document is the design note for the current build. `architecture.md`
remains the evidence and safety policy; this file explains the runtime shape.

## The problem v1.0 fixes

The earlier build had one path for every request: plan → retrieve → generate.
That is wrong in three separate ways.

1. **Not every question is a law question.** "How do I file an FIR" has a
   correct, finite, reviewable answer. Generating it invites invention and
   costs a model round trip.
2. **Retrieval could not narrow.** The query was an OR of every word longer
   than one character: 1,808 of 1,810 bundled judgments matched an ordinary
   sentence, so ranking was doing all the filtering over the whole corpus.
3. **A turn with no model produced nothing.** The most common deployment
   failure mode turned into a 503 instead of an honest, useful answer.

## The runtime

```
turn
 ├─ lexical net (microseconds)       emergency short-circuits to the script
 ├─ understanding.read()            one model call, validated (v1.1)
 │    └─ intent, risk+evidence, facts+spans, next question, language
 ├─ routing.route_turn(reading)      reader decides, net is the floor
 │    ├─ emergency   → safety script, 112 / 181, one question
 │    ├─ human       → handoff offer, nothing sent
 │    ├─ procedural  → data/procedural_kb.json (reviewed steps)
 │    ├─ statute     → approved sources
 │    ├─ precedent   → judgment index, treatment not_checked
 │    ├─ draft       → sources + artifact gate
 │    └─ unclear     → one clarifying question
 ├─ retrieval (rare-term ordered cascade + RRF)
 ├─ compose
 │    ├─ model configured → generate → exact-quote validation → repair → downgrade
 │    └─ no model         → verbatim passages, explicitly no conclusion
 └─ shape: spoken + full + next_question + trace + verification
```

`analyze` makes one short, schema-constrained reading call and no generation;
`compose` is the stage that writes the answer. The split still holds where it
matters: an emergency detected by the lexical net never waits for a model, and
a reviewed procedure is answered from the curated file without one. That is
what lets `/api/turn/stream` emit `routed` and `evidence` events before
generation, and what keeps those two paths in milliseconds.

## Retrieval numbers (measured, bundled 1,810-judgment index)

`scripts/eval_retrieval.py`, mean over 8 representative citizen queries:

| | legacy OR | v1.0 cascade |
|---|---|---|
| candidates considered | 1,804 | 21 |
| reduction | — | 99% |

Term focus (share of returned top-5 containing both required terms) ranges
40–100% across those queries. This is a precision proxy, not a relevance
judgement: there is no labelled relevance set yet, and inventing one would
defeat the point of measuring.

## What is still not built

- Dense retrieval is an interface (`retrieval.register_dense`) with no backend;
  `dense_retrieval` is reported as `false` everywhere it is surfaced.
- Claim verification is exact-quote provenance plus, when a model is configured,
  a model-checked entailment verdict. Legal *review* of the interpretation is
  still absent, and the UI says so.
- Procedures are `draft` until a reviewer signs them off; the UI says so.
- No accounts, no citator, no OCR pipeline, no filing integration.
