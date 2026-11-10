# Data, storage and compute sizing

Everything below separates **measured** (from this machine or the published
dataset docs), **derived** (arithmetic from measured), and **assumption**
(explicit, adjustable). No number here is a benchmark of a system that was not
run.

## 1. What the corpora actually are

| | Source | Records | Published size |
|---|---|---|---|
| Supreme Court, 1950–2025 | `s3://indian-supreme-court-judgments` (public, no credentials) | ~35,000 | **52.24 GB** total, incl. regional languages, PDFs and metadata |
| High Courts, 25 courts / 45 benches | `s3://indian-high-court-judgments` (public, daily updates) | ~17.8M | **~1.25 TiB** of tar archives |
| Indian law Q&A | `viber1/indian-law-dataset` | 24,607 | 14 MB |

*Measured* per-year Supreme Court English archives (S3 `HEAD`/listing):

| Year | `english.tar` | Metadata parquet |
|---|---|---|
| 2010 | 603 MB | ~1 MB |
| 2015 | 508 MB | ~1 MB |
| 2019 | 392 MB | ~1 MB |
| 2021 | 373 MB | ~1 MB |
| 2023 | 400 MB | 0.96 MB |
| 2024 | 203 MB | 0.89 MB |
| 2025 | 177 MB | 0.97 MB |

So the 52 GB figure is the whole bucket across 77 years *and* both languages;
a recent English-only window is ~0.2–0.6 GB per year.

*Measured* metadata: the 2024 parquet has 782 rows and 18 columns —
`title, petitioner, respondent, judge, citation, case_id, cnr, decision_date,
disposal_nature, court, available_languages, path, year`. This is exactly what
the bundled subset was missing: it shipped 1,810 records with **0** decision
dates, citations, benches or CNRs.

## 2. This machine's limits (measured)

```
disk:  228 GB total, 13–14 GB free
RAM:   8 GB
CPU:   8 cores
```

That rules out a full install by three orders of magnitude for the High Courts
and by ~4× for the Supreme Court. `scripts/install_judgments.py` therefore
enforces a `--max-download-gb` budget and refuses to start unless the budget
fits with 2 GiB headroom.

## 3. Storage model at full scale

Assumption: **25 KB of extracted text + JSON per judgment** (derived from the
bundled sample: 1,810 records → 90 MB, median judgment text 30.7k chars).

| Tier | SC | High Courts | Notes |
|---|---|---|---|
| Bronze — raw tar/PDF | 52 GB | 1.25 TiB | keep immutable, checksummed |
| Silver — text + metadata (parquet, one row/judgment) | 1.8 GB | ~450 GB (25 KB × 17.8M) | compresses ~3× in parquet |
| Gold — chunks, **offsets not copies** | ~1 GB | ~270 GB | see below |
| BM25 index | 2–3 GB | 150–250 GB | OpenSearch/Quickwit, not SQLite |
| Dense vectors (only a curated subset) | 35k × 8 = 280k | 2M × 8 = 16M | see the embedding section |

**Why offsets matter.** At 3,600 chars with 420 overlap, an average judgment
yields ~8 chunks that *duplicate* the text they contain. Copying chunk bodies
multiplies the corpus by ~1.15×; storing `(document_id, char_start, char_end)`
does not, and the text stays in one place.

**Embedding the full High Court corpus is not viable.**
142M chunks × 1024 dims × 1 byte (int8) ≈ **145 GB of vectors alone**, before
the ANN graph. Quantised to 256 dims it is still ~36 GB and loses the
precision multilingual retrieval needs. The workable version is a curated
subset — recent years, or judgments that cite a statute in scope — around
**2M judgments / 16M chunks**, which is ~16 GB of int8 vectors.

## 4. Compute model

### Ingest (one-off, full corpus)

Assumption: 8 pages per judgment, PyMuPDF text extraction at 50 pages/s/core.

| Step | Work | 8 cores | 32 cores |
|---|---|---|---|
| PDF → text (SC) | 280k pages | ~12 min | ~3 min |
| PDF → text (HC) | ~142M pages | **~99 h** | ~25 h |
| Parquet build | trivially parallel | <1 h | <15 min |
| Embeddings (subset) | 8B tokens | CPU: not viable | 1×A100: ~1 h |
| Embeddings (full HC) | 71B tokens | CPU: ~355 days | 1×A100: ~8–10 h |

The number that decides your hardware is **HC PDF extraction**: ~100 core-hours.
That is the real reason to use a rentable batch cluster rather than a laptop.

### Query (per turn, the number that matters for voice)

| Stage | Target | Realistic |
|---|---|---|
| STT finalisation | 100–250 ms | provider-dependent |
| Route + safety gate | **5 ms** (measured: procedure turn end-to-end) | local |
| BM25 over 20M docs | 10–30 ms | needs OpenSearch/Quickwit |
| Dense ANN (16M × int8) | 5–15 ms | HNSW, ~20–30 GB RAM |
| Rerank top 50 | 30–80 ms | GPU; **1–3 s on CPU** |
| First sentence from LLM | 250–400 ms | hosted 30B, or smaller model |
| TTS first chunk | 150–250 ms | streaming |
| **End of speech → first audio** | **≤ 1.2 s p50** | achievable only with a reranker on GPU |

A cross-encoder on CPU alone blows the voice budget. If no GPU is available,
the honest design is: skip reranking, keep BM25 + dense RRF, and say so in the
trace — which is exactly what this build does today (lexical only, no rerank).

## 5. The stack that would actually be "most efficient"

Not "RAG with everything"; each layer earns its place:

1. **Bronze → Silver**: raw archive → parquet of text + the 18 metadata
   columns, partitioned `year/court/bench`. DuckDB or Polars, resumable.
2. **Gold**: paragraph-level segments with `(doc_id, seq, char_start, char_end)`
   plus a claim/authority graph built from citations (`overruled_by`,
   `appealed_by`, `followed`) — from verified sources, never LLM extraction.
3. **One search engine, two indexes**: OpenSearch does BM25 **and** kNN, so
   metadata filters (court, jurisdiction, date range) apply before vector
   search, not after. Postgres + pgvector + `tsvector` is the smaller-team
   equivalent up to a few million judgments.
4. **Multilingual embeddings**, int8: Indic queries and English text must land
   in one space, so `bge-m3`-class models, not an English-only encoder.
5. **Reranker on GPU**, top-50 → top-5, with a hard timeout that degrades to
   fused scores instead of blocking the turn.
6. **Two-stage selection for voice**: retrieve 200, fuse, rerank 50, then let
   the planner pick 3–8 passages. Context size is a latency and cost lever,
   not a quality lever, beyond that.
7. **Cache**: immutable query→passage results for anonymous topical queries;
   never cache personalised turns. Source text embeddings are immutable and
   cacheable forever; queries are not.

## 6. What is installed *here*, and why

Given 13 GB free / 8 GB RAM:

```bash
python scripts/install_judgments.py --dataset sc --years 2024 2025 --max-download-gb 2
python scripts/install_judgments.py --dataset hc --years 2024 --court 29_3 --bench karhcdharwad --parts 1 --max-download-gb 2
```This is a **demonstration slice**, not the corpus: it proves the metadata join, the jurisdiction mapping for all 25 High Courts, the deduplication against the bundled sample, and the ingest-time validation that refuses to write records without a decision date.

Result of running it (measured, 380 MB + 39 MB downloaded, ~5.5 minutes):

| | Before | After |
|---|---|---|
| Judgment records | 1,810 | **2,046** |
| With a decision date | 0 | **2,046 (100%)** |
| With a citation | 0 | 1,679 (82%) |
| High Courts represented | 1 (Sikkim) | 2 (Sikkim, Karnataka) |
| Jurisdictions | IN, IN-SK | IN, IN-SK, IN-KA |
| Index chunks | 28,617 | 28,902 |

The High Court corpora publish no citation column, which is why that row is not 100%. The importer also caught its own bug: the High Court parquet keys records by `pdf_link` while the Supreme Court uses `path`, so the first HC run produced 244 records with blank metadata and the run **failed instead of writing them**.

Scaling up is a matter of budget and hardware, not of code.

For the full corpora:

- **Supreme Court, everything**: ~26 GB of archives, ~1.8 GB text, a few hours
  of extraction on 8 cores — feasible on an external 64 GB disk.
- **High Courts, everything**: 1.25 TiB raw + ~450 GB text + ~250 GB index,
  ~100 core-hours of extraction, a GPU for embeddings, and a real search
  cluster. This is a machine with ≥2 TB NVMe, 64–128 GB RAM, and a GPU, or a
  batch cluster writing into object storage.
