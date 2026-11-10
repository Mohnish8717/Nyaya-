# Data sources and how the app uses them

## Hugging Face Indian law Q&A

- Dataset: `viber1/indian-law-dataset`, 24,607 rows.
- Local file: `indian-law-dataset.json`.
- License: Apache-2.0 per the Hugging Face dataset card.
- Use: searchable secondary context for terminology and research leads. The app labels it unverified and does not cite it or use it to support a legal proposition.
- Refresh: `python scripts/install_hf_law_dataset.py` after installing `requirements-ingest.txt`.

## Supreme Court of India judgments

- Dataset: Vanga / Dattam Labs AWS Open Data dataset, 1950–present, English and regional language records; upstream documentation describes approximately 35,000 cases and about 52 GB.
- License: CC BY 4.0 according to the repository README. Retain dataset attribution when redistributing imported text.
- Use: retrieved judgments may illustrate what a named court decided. This build does not check appeal history, overruling, or subsequent treatment, so it never treats semantic similarity alone as current law.
- Bundled subset: 1,679 English records from the 2024 and 2025 partitions are included in `public_judgment_subset.jsonl`.
- Upstream: <https://github.com/vanga/indian-supreme-court-judgments>.

## 25 High Courts

- Dataset: Vanga / Dattam Labs AWS Open Data dataset, 25 High Courts / 45 benches, approximately 17.8 million orders and judgments, about 1.25 TiB of tar archives, with daily updates.
- License: CC BY 4.0 according to the repository README. Retain dataset attribution when redistributing imported text.
- Use: retrieve by court, date, metadata, and judgment text. A High Court decision's geographic and precedential scope must be handled separately; a matching fact pattern does not establish that the same rule applies.
- Bundled subset: 131 records from the 2024 Sikkim High Court partition (`court=11_24`) are included in `public_judgment_subset.jsonl`. No other High Court is bundled.
- Upstream: <https://github.com/vanga/indian-high-court-judgments>.

## Primary legal sources

Only records in `legal_sources.json` marked `approved` can support a legal claim in this local prototype. Each generated legal claim must point to a retrieved record and a verbatim quote that the server verifies. The judgment corpora are case-law sources, not complete, current legislation. Add official statutory sources and amendment/version metadata before making current-law claims about a new jurisdiction or subject.

## Local storage boundary

The included Hugging Face Q&A file and selected judgment subset ship in the project. The full judgment archives cannot fit in the current workspace (only about 16 GiB was available during project work). `scripts/sync_public_judgments.py` downloads selected, size-budgeted Supreme Court years or a selected High Court/court/year partition, extracts searchable PDF text into `.runtime/legal-data/judgments.jsonl`, and deduplicates repeat imports. The app indexes this runtime file together with the bundled subset. The runtime index is local and ignored by Git. Do not run an all-years sync on a laptop; use object storage plus a scalable search index for full-corpus deployment.
