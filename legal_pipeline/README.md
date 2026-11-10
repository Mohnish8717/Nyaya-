# legal_pipeline — standalone judicial-text ingest + embeddings

Self-contained folder: move it anywhere, `cd` in, install deps, run. It imports
nothing from the Nyaya app, and everything it writes lands under `out/`
beside this README.

```bash
cd legal_pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py sources        # what exists, licences, sizes
python pipeline.py status         # what's already built
python smoke.py                   # end-to-end self-check on synthetic data (no download, no model)
```

## One command: InJudgements -> embeddings

If the only corpus you want is `opennyaiorg/InJudgements_dataset`, this is the
whole path in one script. It touches no other dataset and writes exactly one
file, `out/corpus_embeddings.sqlite3`:

```bash
python build_opennyai_embeddings.py --preflight    # check deps, token, disk, network
python build_opennyai_embeddings.py                # download (~293 MB) + embed everything
python build_opennyai_embeddings.py --limit 300    # trial run first, if you like
python build_opennyai_embeddings.py --verify       # re-check the artifact it just wrote
```

It downloads only the parquet, streams it row by row (no PDFs, no HTML, no
scraping), labels every passage with its tier, and finishes by *verifying* the
file it produced — integrity, row counts, vector coverage, and a real BM25
query joined back to a labelled judgment — so "done" is measured, not asserted.

`--via datasets` reads the same data through the dataset card's own snippet
(`load_dataset(..., streaming=True)`) and feeds the identical writer, so the two
readers cannot drift. `--local <folder>` works on an already-downloaded copy;
`--resume` continues an interrupted run.

The dataset is **gated**: its file list is public, its parquet is not. Accept
the terms on the dataset page, then `huggingface-cli login` (or export
`HF_TOKEN`). The script says exactly that instead of printing an HTTP traceback.

## The 150 MB path (one file, three tiers, all languages)

The full corpus is a cluster job. If what you need is a slice you can embed on
one machine and ship as a single artifact, this is the whole flow:

```bash
python pipeline.py download --source aws_sc  --years 2020 --max-download-gb 6
python pipeline.py download --source aws_hc  --years 2021 --court 29_4 --max-download-gb 6
python pipeline.py download --source kaggle_sc          # or --local <folder>

python pipeline.py curate --target-mb 150
```

Result: **`out/corpus_embeddings.sqlite3`, one file**, ~150 MB, holding the
slice, the passages, their int8 vectors and the BM25 index. `status` reports
it; `--resume` continues an interrupted run and back-fills any passage that was
written but not embedded; `--target-mb 20` and `--per-tar-cap 0` are the knobs
for a quick trial.

What the slice guarantees, and how:

- **every tier is represented and labelled.** Supreme Court, High Court and
district-court sources get budget shares (`CORPUS_MIX`, default 35/45/20,
normalised over the tiers actually on disk). Each passage carries
`court_level` (`supreme` / `high` / `district`) in a column *and* in a header
line prepended to the embedded text:
  `[HIGH COURT — High Court of Karnataka | level: high | jurisdiction: IN-KA | language: kn]`
  so a district order can never be retrieved as if it bound the whole country,
  and the vector itself separates the tiers.
- **all languages, not just English.** Open Data archives are laid out per
  language folder (`english/`, `regional/hindi/`, …). `curate` reads one header
  pass per archive to count what each language holds, then budgets each
  language in proportion — a slice cannot silently become English-only while
  Tamil judgments sit on the same disk. The header line records the language,
  and the encoder (`paraphrase-multilingual-MiniLM-L12-v2`, 50+ languages) is
  the same multilingual model the app uses.
- **capped per source, capped per archive.** `--target-mb` is the size of the
  file you will check on disk; the text budget is 70% of it and the measured
  ratio is printed, so the number is checked rather than assumed.
  `CORPUS_PER_TAR_CAP` (default 150 judgments) stops one court-year from
  filling the slice.
- **honest sampling note.** Selection inside an archive is the head of its
  member list — deterministic, not random — and that is recorded in the file's
  `meta` table alongside tier/language/dataset counts.

## OpenNyAI InJudgements (the one dataset that ships the text)

`opennyaiorg/InJudgements_dataset` is the cheapest real corpus here because its
parquet **contains the judgment text** — no PDFs, no scraping, no HTML.
Published schema (one `train` split, 11,970 rows, ~629 MB of text, ~293 MB of
parquet, apache-2.0, English):

| column | type | used as |
|---|---|---|
| `Titles` | string | record title |
| `Court_Name` | string | court (also drives jurisdiction) |
| `Cites`, `Cited_by` | int64 | importance signal; not needed to embed |
| `Doc_url` | string | provenance URL and dedupe identity |
| `Text` | string | **the judgment, embedded** |
| `Doc_size` | int64 | not used |
| `Case_Type` | string | subject bucket (kept in the source record) |
| `Court_Type` | string | the tier: `supreme` / `high` / `district` |
| `Court_Name_Normalized` | string | court fallback when `Court_Name` is empty |

```bash
python pipeline.py opennyai                      # download (~293 MB) then embed everything
python pipeline.py opennyai --limit 500          # trial slice
python pipeline.py opennyai --local ~/InJudgements_dataset
python pipeline.py opennyai --no-download        # use what is already on disk
```

The full 293 MB is taken deliberately: 11,970 judgments produce roughly 200 K
chunks, which is one to two hours on CPU and minutes on a GPU. The output is
the same `out/corpus_embeddings.sqlite3`, written by the same writer, with the
same tiers and headers.

Two honest caveats, both from the dataset itself:

- **no decision date.** The schema has no date column, so `decision_date` is
  stored as `unknown` rather than guessed from the text or the URL.
- **district coverage is Delhi and Bengaluru only.** The card says so; the
  other courts in it are Supreme Court, High Courts and tribunals. A tribunal
  is labelled `unknown`, never folded into `district`, because a tribunal order
  is not a district-court judgment.

**Access is gated, verified against the live repo.** Its *file list* is public
(`hf://datasets/opennyaiorg/InJudgements_dataset/data/train-0000*-of-00002-*.parquet`,
two shards), but fetching those bytes without a token returns
`401 Cannot access gated repo`. So a Hugging Face token with the terms accepted
is required, and the command says exactly that instead of failing opaquely:

```bash
huggingface-cli login          # or: export HF_TOKEN=hf_...
python pipeline.py opennyai    # then the ~293 MB download proceeds
```

Accepting the terms is a click on the dataset page; it cannot be scripted, and
nothing here attempts to work around it. `--local <folder>` remains the offline
route for anyone who obtained the parquet another way.

## Parquet-first mode (metadata selects, PDFs supply the words)

> **The parquet has no judgment text.** Both published files are *structured
> metadata*: Supreme Court columns are `title, petitioner, respondent,
> description, judge, author_judge, citation, case_id, cnr, decision_date,
> disposal_nature, court, available_languages, raw_html, path, nc_display,
> scraped_at, year`; High Court columns are `court_code, title, description,
> judge, pdf_link, cnr, date_of_registration, decision_date, disposal_nature,
> court, raw_html, pdf_exists`. `path` and `pdf_link` are pointers to the PDF,
> which is the only place the judgment text exists. Anything that claims to
> embed from parquet alone is embedding metadata, not law.

So `parquet` mode uses the structured fields for *selection and labels* and
fetches only the PDFs those rows point at:

```bash
python pipeline.py parquet --plan-only                 # partitions + sizes, transfers nothing
python pipeline.py parquet --target-mb 150             # both tiers, 150 MB hard cap
python pipeline.py parquet --source aws_sc --target-mb 40 --years 2015-2024
python pipeline.py parquet --source aws_hc --court 29_3 --target-mb 60 --years 2023
```

District courts are not part of this path: it takes `aws_sc` and `aws_hc`
only. The output is the same single file, `out/corpus_embeddings.sqlite3`,
with the same `court_level` labels and the same schema, so downstream code does
not care which mode produced it.

How the 150 MB cap is enforced (`--target-mb` = everything transferred):

- parquet is read **with column projection** through `pyarrow.fs.S3FileSystem`,
  so `raw_html` — by far the largest column — is never transferred;
- parquet may consume at most 15% of the cap (`PARQUET_RESERVE_SHARE`), and its
  charge is counted inside the same total, so there is no second budget to
  overshoot. Parquet bytes are *estimated* from the projected columns'
  compressed sizes in the footer; PDF bytes are counted exactly;
- each PDF is counted at its real byte length; a single PDF above
  `PARQUET_MAX_PDF_MB` (12) is skipped rather than allowed to eat the run;
- rows are drained round-robin across years and courts, `--rows-per-partition`
  at a time, then stop when the cap is hit. `meta` records
  `transfer_mb`, `parquet_mb`, `skipped_rows` and the partition count, so the
  claim is measured, not asserted.

Honest limits of this mode: the fetched PDF is the one the row points at (the
English judgment for the Supreme Court), so its language mix is that of those
files — regional-language PDFs remain available through the tar path above.
Rows with a missing pointer, `pdf_exists = false`, an oversized PDF or a scan
with no text layer are counted as `skipped_rows`, never fabricated.

Querying the single file:

```sql
-- lexical, tier-filtered, with the passage text and its label
SELECT c.title, c.court_level, ch.header, ch.text, bm25(chunk_fts) AS score
FROM chunk_fts
JOIN chunks ch ON ch.rowid = chunk_fts.rowid
JOIN cases  c  ON c.case_id = ch.case_id
WHERE chunk_fts MATCH '"section 138" AND cheque'
  AND ch.court_level = 'high'
ORDER BY score LIMIT 5;
```

```sql
-- dense: load vectors into numpy once (dim from meta), score, then read passages
SELECT chunk_id, vec FROM vectors;                     -- int8, 384 per row
SELECT key, value FROM meta WHERE key IN ('model','dim','per_level','per_language');
```

## Sources (one line each; details with `python pipeline.py inspect --source <key>`)

| key          | court level       | text? | licence | size (as documented upstream)                                |
|--------------|-------------------|-------|---------|--------------------------------------------------------------|
| `aws_sc`     | Supreme Court     | yes   | CC-BY-4.0 | ~35K judgments, ~52 GiB of tars, 1950–2025                 |
| `aws_hc`     | 25 High Courts    | yes   | CC-BY-4.0 | ~17.8M judgments, ~1.25 TiB of tars (narrow by year+court!) |
| `kaggle_sc`  | Supreme Court     | yes   | Kaggle terms | SC texts from Indian Kanoon, ~98% coverage 1950–2024     |
| `opennyai_dc`| District (sample) + SC/HC | yes | apache-2.0 | ~293 MB, most-cited 1950–2017 (district: Delhi+Bengaluru) |
| `ddl_meta`   | district-case metadata | **no text** | CC BY-NC-SA 4.0 (non-commercial) | 81.2M cases 2010–2018, stats only |

Why one sample has no rival for district text: no bulk public district-court
judgment corpus exists yet (eCourts district judgments are per-case PDFs behind
CAPTCHAs; the vanga/indian-district-court-judgments scraper is WIP). So `aws_sc
+ aws_hc` provide bulk text at the upper levels, `opennyai_dc` is the only
real district-level text, and `ddl_meta` adds the statistical layer on the 81M
lower-judiciary cases without polluting the embedding index.

## Run order (each step is resumable; nothing re-downloads done work)

```bash
# 1. choose + download (budgeted; raise --max-download-gb deliberately)
python pipeline.py inspect --source aws_hc --years 2020-2024
python pipeline.py download --source aws_sc --years 1950-2019 --max-download-gb 40
python pipeline.py download --source aws_hc --years 2020-2024 --court 29_4 --max-download-gb 60
python pipeline.py download --source kaggle_sc          # needs KAGGLE_USERNAME/KAGGLE_KEY (or --local)
python pipeline.py download --source opennyai_dc        # needs huggingface-cli login (or --local)
python pipeline.py download --source ddl_meta           # needs ddl_links.txt URLs (CC BY-NC-SA!), or --local

# 2. text -> unified schema
python pipeline.py extract        # judgments.jsonl + sources_report.json (field detection log)
python pipeline.py ddl            # metadata-only store out/ddl_cases.sqlite3

# 3. searchable artifacts
python pipeline.py chunk          # out/chunks.sqlite3  (head + capped body chunks)
python pipeline.py embed          # out/vectors/shard_*.npy + manifest (int8, resumable)
python pipeline.py fts            # out/cases_index.sqlite3 (BM25, app-compatible schema)
python pipeline.py export         # out/app/{dense_cases,cases}.sqlite3 (drop-in for the app)

# one shot for 2+3 once downloads are complete:
python pipeline.py all

# bounded trial before committing hours of CPU/GPU to a full run:
python pipeline.py chunk --limit 2000 && python pipeline.py embed --limit 2000
```

Every stage is idempotent: re-running `extract` appends only genuinely new
judgments (de-duplication is a SQLite UNIQUE index under `out/dedupe.sqlite3`,
not an in-memory set), `chunk` skips cases already chunked, `embed` picks up
exactly the chunks missing from its manifest, and `fts` rebuilds only when
`judgments.jsonl` changed.

## What lands in out/

- `judgments.jsonl` — one row per unique judgment: `{id, title, court, text,
  source_url, decision_date, dataset, citation, language, jurisdiction,
  archive_member}`. This is the embedding input and the FTS input.
- `sources_report.json` — columns detected per source, kept/dropped counts
  (`empty_text`, `dup_text`, `dup_name`), metadata-join counts.
- `chunks.sqlite3` — per-chunk rows (`head` + capped `body`), with provenance.
- `vectors/shard_*.npy` + `vectors/manifest.sqlite3` — int8 unit vectors,
  `chunk_id -> (shard, row)` mapping, model name recorded.
- `cases_index.sqlite3` — FTS5 BM25 over full judgments, same table shape the
  Nyaya app's case DB uses; query e.g.:
  ```sql
  SELECT title, decision_date, source_url FROM case_fts
  WHERE case_fts MATCH '"section 138" AND cheque'
    AND jurisdiction='IN-KA' ORDER BY bm25(case_fts,0,4.0,3.0,2.0,1.0,0,0,0,0,1.0);
  ```
- `ddl_cases.sqlite3` — DDL metadata only (acts/sections, dispositions, dates);
  indexed on `(state, year)` and `(act, section)`. **Never embedded.**
- `app/dense_cases.sqlite3` — the vectors reshaped into the exact schema
  `app/dense.py` opens: `meta(key,value)` and
  `vectors(source_id, chunk, vec)` where `vec` is 384 int8 values and `chunk
  = -1` is that case's semantic head. Copy to `.runtime/dense_cases.sqlite3`.
- `corpus_embeddings.sqlite3` — the curated single-file slice: `cases`,
  `chunks` (`header` + verbatim `text`), `vectors` (int8), `chunk_fts`
  (external-content FTS over `chunks.text`, so text is stored once) and `meta`.
  Written by `curate`, not by the full-corpus stages.
- `app/cases.sqlite3` — a consistent snapshot of `cases_index.sqlite3`
  (`case_fts` + `index_state` incl. `source_count`). Copy to
  `.runtime/cases.sqlite3` and the app's BM25 + dense paths answer from it.

## Text → embedding format (what each stage actually converts)

A judgment is never handed to the encoder whole. A 200-page decision is
~500 K chars and the multilingual encoder reads 512 tokens:

| stage | input | output | why |
|---|---|---|---|
| `extract` | PDF tar member, CSV/parquet row | one JSONL row, whitespace-normalised, `text` capped at 400 K chars | one schema for four upstream layouts |
| `chunk` | that row | `head` row (title + court + date + ~1.2 K chars around the headnote) + up to `PIPELINE_CHUNKS_PER_CASE` body rows of ~3.2 K chars with 150-char overlap, cut at sentence boundaries | fits the 512-token window; the head row is what a topical query lands on |
| `embed` | chunk rows | unit vector scaled to int8 [-127, 127] in `shard_*.npy`, mapped by `chunk_id` | 4× smaller than fp32, identical ranking after `/127` |
| `export` | shards + manifest | `vectors(source_id, chunk, vec INT8 BLOB)` | the app's reader expects SQLite blobs, not `.npy` |

Texts are truncated to `PIPELINE_EMBED_MAX_CHARS` (2400) before encoding, which
is the encoder's window rather than an arbitrary cut. `DENSE_MODEL` must match
the app's `DENSE_MODEL`; `embed` refuses to blend two models in one manifest —
it clears and rebuilds instead, and says so.

## Scale notes

- 1.25 TiB of High Court tars is ~17.8 M judgments: budget disk before the
  fashion. Rule of thumb — raw tars 1.25 TiB, `judgments.jsonl` ~0.6 of the
  compressed text, chunks ~1.2× the JSONL, vectors ~1.5 KB per chunk
  (384 int8 + overhead), FTS ~1.1× the JSONL. Narrow with `--years` and
  `--court`; a single court-year is usually a few GB.
- memory stays flat: extract streams tar members, embed streams via an
  attached SQLite cursor, dedupe and job state live on disk. Nothing holds
  corpus-scale data in Python objects.
- a vector index this exact (brute-force dot product) is fine to roughly
  ~10 M chunks per process; beyond that shard by jurisdiction and query the
  relevant shards, which is what `shards` in the manifest is for.

## Budgets and honesty rules

- every downloader checks total bytes BEFORE streaming (the `inspect` step
  prints per-year sizes first); nothing spans silently over `--max-download-gb`.
- resume manifests live beside the raw data (`out/raw/**/.state.sqlite3`), so
  separate runs, different slices, do not duplicate or re-charge anything.
- DDL licence is CC BY-NC-SA — non-commercial; it belongs in the stats store
  only.
- text extraction refuses rows under 200 chars and reports them; synthetic
  placeholders are never written (checks in extract.py `_accept_record`).

## Upstream licenses

aws_sc / aws_hc: CC-BY-4.0 (Dattam Labs) • kaggle_sc: Kaggle's terms apply •
opennyai_dc: apache-2.0 (OpenNyAI) • ddl_meta: CC BY-NC-SA 4.0 (Dev Data Lab) —
non-commercial use only; cite their paper if you use those metadata.
