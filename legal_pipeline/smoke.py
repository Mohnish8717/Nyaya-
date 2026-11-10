"""End-to-end self-check with synthetic data: no network, no model download.

This is the run-readiness proof for the pipeline. It fabricates the smallest
possible version of every supported input (an AWS-style tar pair with a real
PDF, a Kaggle-style CSV with two judgments, a DDL-style metadata CSV), runs
extract -> chunk -> embed -> fts -> ddl -> export against a temporary output
directory, and asserts the contracts that would otherwise only break on real
data:

  * both PDFs survive, are joined to metadata, and get a provenance url even
    when the metadata has none (the REQUIRED.source_url fallback);
  * the two rows in one CSV are both kept (the dedupe key is per row, not per
    file);
  * a second extract run appends nothing and does not erase the first run;
  * every chunk is embedded exactly once, head rows export as chunk -1, and
    the vector blob is 384 int8 values the app can read;
  * the exported SQLite files have the app's exact tables.

    python -m legal_pipeline.smoke        # or: python legal_pipeline/smoke.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
import dl_ddl  # noqa: E402
import dl_kaggle  # noqa: E402
import dl_opennyai  # noqa: E402
import ddl_store  # noqa: E402
import chunk as chunk_mod  # noqa: E402
import embed as embed_mod  # noqa: E402
import export as export_mod  # noqa: E402
import extract  # noqa: E402
import fts_index  # noqa: E402
import util  # noqa: E402

BODY_A = ("This appeal arises from a claim for unpaid wages. The employer withheld "
          "salary for three months and the workman approached the labour court. "
          "Section 33C of the Industrial Disputes Act provides the remedy. ") * 4
BODY_B = ("The petitioner challenges an order refusing to register a first information "
          "report. The magistrate directed the station house officer to register the "
          "complaint and investigate. The petition is disposed of with directions. ") * 4
BODY_C = ("The complainant alleges that a cheque issued towards a legally enforceable "
          "debt was dishonoured for insufficient funds and that the statutory notice "
          "was served. Section 138 of the Negotiable Instruments Act applies. ") * 4
BODY_D = ("Tenant asserts illegal eviction by lock change without notice to quit. The "
          "rent controller found that due process was not followed and ordered "
          "restoration of possession to the tenant. ") * 4
BODY_E = ("Application for regular bail in a case alleging cheating and criminal breach "
          "of trust. The court considered the period already undergone, the completion "
          "of investigation and the absence of any further custodial requirement. ") * 4
BODY_F = ("Revision against an order of maintenance. The family court assessed the "
          "income of the respondent from the documentary evidence and fixed an "
          "interim amount payable every month to the applicant. ") * 4


def _pdf_bytes(text: str) -> bytes:
    import fitz  # PyMuPDF

    doc = fitz.open()
    page = doc.new_page()
    # Wrap so the page holds the whole judgment the way a real one would.
    y = 60
    for start in range(0, len(text), 90):
        page.insert_text((50, y), text[start:start + 90], fontsize=9)
        y += 13
        if y > 780:
            page = doc.new_page()
            y = 60
    payload = doc.tobytes()
    doc.close()
    return payload


def _make_tar(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            import io

            tf.addfile(info, io.BytesIO(data))


def _rewrite_paths(root: Path) -> None:
    out = root / "out"
    config.OUT = out
    config.RAW = out / "raw"
    config.REPORT_PATH = out / "sources_report.json"
    config.JUDGMENTS_JSONL = out / "judgments.jsonl"
    config.CHUNKS_DB = out / "chunks.sqlite3"
    config.VECTORS_DIR = out / "vectors"
    config.VECTORS_MANIFEST = config.VECTORS_DIR / "manifest.sqlite3"
    config.CASES_INDEX_DB = out / "cases_index.sqlite3"
    config.DDL_DB = out / "ddl_cases.sqlite3"
    config.EXPORT_DIR = out / "app"
    config.EXPORT_DENSE_DB = config.EXPORT_DIR / "dense_cases.sqlite3"
    config.EXPORT_CASES_DB = config.EXPORT_DIR / "cases.sqlite3"
    config.SINGLE_DB = out / "corpus_embeddings.sqlite3"
    # Downloaders cache their root at import time; move it too.
    dl_kaggle.RAW_DIR = config.RAW / "kaggle"
    dl_opennyai.RAW_DIR = config.RAW / "opennyai"
    dl_ddl.RAW_DIR = config.RAW / "ddl"


class StubEmbedder:
    """Deterministic 384-dim vectors: same shape/type as the real encoder."""

    dim = 384

    def embed(self, texts):
        import hashlib

        import numpy as np

        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            yield rng.standard_normal(self.dim).astype("float32")


def _check(condition: bool, message: str, failures: list[str]) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def run() -> int:
    failures: list[str] = []
    root = Path(tempfile.mkdtemp(prefix="legal_pipeline_smoke_"))
    try:
        _rewrite_paths(root)

        # --- synthetic AWS Open Data layout -------------------------------
        raw = config.RAW / "aws_sc_data"
        _make_tar(
            raw / "data/tar/year=2020/english/judgments_1.tar",
            {"wage_claim_1234_en.pdf": _pdf_bytes(BODY_A),
             "fir_refusal_5678_en.pdf": _pdf_bytes(BODY_B)},
        )
        _make_tar(
            raw / "metadata/tar/year=2020/metadata_1.tar",
            {"wage_claim_1234_en.json": json.dumps({
                "case_number": "WP 1234/2020",
                "judgment_date": "2020-03-11",
                "court": "Supreme Court of India",
                "title": "Workman v. Employer",
            }).encode()},
        )

        # --- synthetic Kaggle csv (two judgments in one file) -------------
        kaggle = config.RAW / "kaggle" / "local"
        kaggle.mkdir(parents=True, exist_ok=True)
        header = "doc_id,title,court,decision_date,source_url,text\n"
        rows = [
            ("k1", "Cheque dishonour appeal", "Karnataka High Court", "2019-07-02",
             "https://indiankanoon.org/doc/1", BODY_C.replace("\n", " ")),
            ("k2", "Illegal eviction petition", "Delhi High Court", "2018-01-09",
             "https://indiankanoon.org/doc/2", BODY_D.replace("\n", " ")),
        ]
        with (kaggle / "judgments.csv").open("w", encoding="utf-8") as fh:
            fh.write(header)
            for row in rows:
                fh.write(",".join(f'"{v}"' for v in row) + "\n")

        # --- synthetic DDL metadata ---------------------------------------
        ddl = config.RAW / "ddl" / "local"
        ddl.mkdir(parents=True, exist_ok=True)
        (ddl / "cases_2018.csv").write_text(
            "ddl_case_id,state_name,district_name,type_name,act,section,"
            "disp_name,date_filing,date_decision,criminal\n"
            "c1,Karnataka,Bengaluru,Civil,Industrial Disputes Act,33C,Allowed,2018-01-01,2018-06-01,0\n"
            "c2,Delhi,New Delhi,Criminal,Negotiable Instruments Act,138,Convicted,2018-02-01,2018-09-01,1\n",
            encoding="utf-8",
        )

        print("extract: AWS tar/PDF + metadata join")
        extract.extract(["--source", "aws_sc"])
        print("extract: Kaggle csv column detection + per-row dedupe")
        extract.extract(["--source", "kaggle_sc"])
        records = list(util.jsonl_iter(config.JUDGMENTS_JSONL))
        by_id = {r["id"]: r for r in records}
        _check(len(records) == 4, f"4 judgments kept (got {len(records)})", failures)
        aws = [r for r in records if r["dataset"] == "aws_sc"]
        _check(len(aws) == 2, f"both PDFs extracted (got {len(aws)})", failures)
        joined = by_id.get("aws_sc:wage_claim_1234_en")
        _check(bool(joined), "metadata joined to the matching PDF", failures)
        if joined:
            _check(joined["decision_date"] == "2020-03-11", "metadata decision_date used", failures)
            _check(joined["title"] == "Workman v. Employer", "metadata title used", failures)
        lonely = by_id.get("aws_sc:fir_refusal_5678_en")
        if lonely:
            year_fallback = lonely["decision_date"] == "2020-01-01"
            _check(year_fallback, "missing metadata date falls back to the tar year", failures)
            _check(lonely["source_url"].startswith("https://registry.opendata.aws/"),
                   "missing source_url falls back to dataset provenance", failures)
        kaggle_rows = [r for r in records if r["dataset"] == "kaggle_sc"]
        _check(len(kaggle_rows) == 2, f"both CSV rows kept (got {len(kaggle_rows)})", failures)
        if kaggle_rows:
            _check(all(r["text"] for r in kaggle_rows), "CSV text column detected", failures)
            _check(any(r["jurisdiction"] == "IN-KA" for r in kaggle_rows),
                   "jurisdiction derived from the court column", failures)
            _check(all(r["language"] for r in kaggle_rows), "language tagged from script", failures)

        print("extract: second run must append nothing and keep the corpus")
        before = len(records)
        extract.extract(["--source", "aws_sc", "--source", "kaggle_sc"])
        after = list(util.jsonl_iter(config.JUDGMENTS_JSONL))
        _check(len(after) == before, f"re-extract is a no-op (got {len(after)})", failures)

        print("chunk")
        chunk_stats = chunk_mod.build()
        _check(chunk_stats["cases"] == before, f"{before} cases chunked", failures)
        _check(chunk_stats["chunks"] >= before, "at least one chunk per case", failures)

        print("embed (stub encoder, no model download)")
        embed_stats = embed_mod.build(embedder=StubEmbedder())
        _check(embed_stats["embedded"] == chunk_stats["chunks"],
               f"every chunk embedded once ({embed_stats['embedded']}/{chunk_stats['chunks']})",
               failures)
        rerun = embed_mod.build(embedder=StubEmbedder())
        _check(rerun["embedded"] == 0, "re-embed is a no-op", failures)

        print("fts")
        fts_stats = fts_index.build()
        _check(fts_stats["cases"] == before, f"{before} cases in FTS", failures)

        print("ddl")
        ddl_stats = ddl_store.load_ddl_cases(ddl)
        _check(bool(ddl_stats["files"]), "DDL csv loaded", failures)
        import sqlite3

        with sqlite3.connect(config.DDL_DB) as db:
            ddl_rows = db.execute("SELECT COUNT(*) FROM ddl_cases").fetchone()[0]
            _check(ddl_rows == 2, f"2 DDL rows stored (got {ddl_rows})", failures)

        print("export to the app's schemas")
        export_mod.build()
        with sqlite3.connect(config.EXPORT_DENSE_DB) as db:
            total = db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            heads = db.execute("SELECT COUNT(*) FROM vectors WHERE chunk=-1").fetchone()[0]
            blob = db.execute("SELECT vec FROM vectors LIMIT 1").fetchone()[0]
            meta = dict(db.execute("SELECT key,value FROM meta").fetchall())
        _check(total == chunk_stats["chunks"], f"all {total} vectors exported", failures)
        _check(heads == before, f"{heads} head vectors at chunk -1", failures)
        _check(len(blob) == StubEmbedder.dim, "vector blob is one int8 value per dimension", failures)
        _check(meta.get("dim") == str(StubEmbedder.dim), "manifest dim recorded", failures)
        with sqlite3.connect(config.EXPORT_CASES_DB) as db:
            cases = db.execute("SELECT COUNT(*) FROM case_fts").fetchone()[0]
            state = dict(db.execute("SELECT key,value FROM index_state").fetchall())
        _check(cases == before, f"{cases} cases in the app case index", failures)
        _check("source_count" in state, "index_state has the key the app reads", failures)

        print("curate: one combined, level-labelled, multilingual file")
        import curate

        # Two extra archives, added only now so the extract-stage counts above
        # stay exactly what that stage is being tested for: one High Court tar
        # (level high) and one regional tar whose language comes from its
        # folder path, which members inside a tar can never express.
        _make_tar(
            config.RAW / "aws_hc_data/data/tar/year=2021/court=29_4/english/hc_1.tar",
            {"hc_bail_9101_en.pdf": _pdf_bytes(BODY_E)},
        )
        _make_tar(
            config.RAW / "aws_sc_data/data/tar/year=2020/regional/hindi/judgments_hi.tar",
            {"maintenance_7777_hi.pdf": _pdf_bytes(BODY_F)},
        )
        curated = curate.build(target_mb=1, sources=["aws_sc", "aws_hc", "kaggle_sc"],
                               embedder=StubEmbedder(), per_tar_cap=0)
        _check(Path(curated["file"]).exists(), "one embeddings file written", failures)
        sidecars = [Path(str(config.SINGLE_DB) + suffix) for suffix in ("-wal", "-shm")]
        _check(not any(p.exists() for p in sidecars), "no WAL sidecar files beside it", failures)
        _check(curated["text_bytes"] <= int(1_000_000 * config.CORPUS_TEXT_SHARE_OF_FILE),
               "text stayed inside the budget for a 1 MB file", failures)
        _check(curated["vectors"] == curated["chunks"], "every chunk has a vector", failures)
        _check(curated["per_level"].get("supreme", 0) >= 1, "supreme-tier judgments labelled", failures)
        _check(curated["per_level"].get("high", 0) >= 1, "high-court judgments labelled", failures)
        _check("hi" in curated["per_language"], "regional-language tar labelled hi", failures)

        with sqlite3.connect(config.SINGLE_DB) as db:
            headers = dict(db.execute(
                "SELECT court_level, header FROM chunks GROUP BY court_level"
            ).fetchall())
            has_fts = bool(db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='chunk_fts'"
            ).fetchone())
            fts_hits = db.execute(
                "SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH 'cheque'"
            ).fetchone()[0]
            joinable = db.execute(
                "SELECT COUNT(*) FROM chunk_fts f JOIN chunks ch ON ch.rowid=f.rowid "
                "JOIN cases c ON c.case_id=ch.case_id WHERE chunk_fts MATCH 'cheque'"
            ).fetchone()[0]
            meta_single = dict(db.execute("SELECT key,value FROM meta").fetchall())
            blob = db.execute("SELECT vec FROM vectors LIMIT 1").fetchone()[0]
        _check(all("level:" in (h or "") for h in headers.values()),
               "every tier's chunks carry a level header", failures)
        _check("level: supreme" in headers.get("supreme", ""),
               "header states the tier in words", failures)
        _check(has_fts, "BM25 index lives in the same file", failures)
        _check(fts_hits >= 1, f"lexical search finds a real passage ({fts_hits} hit(s))", failures)
        _check(joinable == fts_hits, "FTS hits join back to cases through chunk rowids", failures)
        _check(meta_single.get("model") is not None and meta_single.get("dim"),
               "model and dimension recorded", failures)
        _check(len(blob) == StubEmbedder.dim, "vector blob is int8 per dimension", failures)

        print("curate: a tiny target must actually cap the slice")
        small = curate.build(target_mb=0.01, sources=["aws_sc", "aws_hc", "kaggle_sc"],
                             embedder=StubEmbedder(), per_tar_cap=0)
        tiny_budget = int(10_000 * config.CORPUS_TEXT_SHARE_OF_FILE)
        _check(small["cases"] < curated["cases"],
               f"caps the slice ({small['cases']} < {curated['cases']} judgments)", failures)
        # One judgment cannot be split, so a source may exceed its share by at
        # most a single record; the total must still be of that order.
        _check(small["text_bytes"] <= tiny_budget * 2,
               f"text stays near the tiny budget ({small['text_bytes']} B)", failures)

        print("parquet-first: metadata selects, pointed-at PDFs supply text")
        import io

        import pyarrow as pa
        import pyarrow.parquet as pq
        from pyarrow.fs import LocalFileSystem

        import parquet_source

        class FakeS3:
            """Only the two objects the parquet rows point at exist."""

            def __init__(self, payloads: dict[str, bytes]):
                self.payloads = payloads

            def get_object(self, Bucket, Key):  # noqa: N803 - boto3 signature
                if Key not in self.payloads:
                    raise KeyError(Key)
                data = self.payloads[Key]
                return {"Body": io.BytesIO(data), "ContentLength": len(data)}

        parq_dir = root / "parquet"
        parq_dir.mkdir(parents=True, exist_ok=True)
        sc_path = parq_dir / "sc.parquet"
        pq.write_table(pa.table({
            "title": ["Wage claim appeal", "Maintenance revision", "Pointerless row"],
            "description": ["service matter", "family matter", "missing pdf"],
            "citation": ["(2020) 5 SCC 1", "(2021) 2 SCC 9", ""],
            "case_id": ["sc-1", "sc-2", "sc-3"],
            "cnr": ["CNR1", "CNR2", "CNR3"],
            "decision_date": ["2020-04-01", "2021-05-02", "2021-06-03"],
            "court": ["Supreme Court of India"] * 3,
            "available_languages": ["english,hindi", "english", "english"],
            "path": ["s3://sc-bucket/data/pdf/one.pdf", "s3://sc-bucket/data/pdf/two.pdf", ""],
            "year": [2020, 2021, 2021],
        }), sc_path)
        hc_path = parq_dir / "hc.parquet"
        pq.write_table(pa.table({
            "court_code": ["29_3", "7_26"],
            "title": ["Bail application", "Eviction petition"],
            "description": ["criminal", "civil"],
            "cnr": ["HCNR1", "HCNR2"],
            "decision_date": ["2021-02-02", "2021-03-03"],
            "court": ["High Court of Karnataka", "High Court of Delhi"],
            # The second pointer is deliberately valid: the row must be skipped
            # because pdf_exists is False, not because the object is absent.
            "pdf_link": ["s3://hc-bucket/data/pdf/hc1.pdf", "s3://hc-bucket/data/pdf/hc2.pdf"],
            "pdf_exists": [True, False],
            "judge": ["A. Judge", "B. Judge"],
            "date_of_registration": ["2020-12-01", "2020-12-02"],
        }), hc_path)

        partitions = [
            {"source": "aws_sc", "level": "supreme", "bucket": "sc-bucket",
             "key": "metadata/parquet/year=2020/metadata.parquet", "label": "SC test",
             "year": 2020, "court": "", "path": str(sc_path)},
            {"source": "aws_hc", "level": "high", "bucket": "hc-bucket",
             "key": "metadata/parquet/year=2021/court=29_3/metadata.parquet",
             "label": "HC test", "year": 2021, "court": "29_3", "path": str(hc_path)},
        ]
        fake = FakeS3({
            "data/pdf/one.pdf": _pdf_bytes(BODY_A),
            "data/pdf/two.pdf": _pdf_bytes(BODY_F),
            "data/pdf/hc1.pdf": _pdf_bytes(BODY_E),
            "data/pdf/hc2.pdf": _pdf_bytes(BODY_D),
        })
        pulled = parquet_source.pull(
            target_mb=5, embedder=StubEmbedder(), client=fake, fs=LocalFileSystem(),
            partitions=partitions, rows_per_partition=4,
        )
        _check(pulled["cases"] == 3, f"3 rows with a real PDF embedded (got {pulled['cases']})", failures)
        _check(pulled["skipped_rows"] == 2,
               f"empty pointer and pdf_exists=false skipped, nothing else "
               f"({pulled['skipped_rows']})", failures)
        _check(pulled["per_level"].get("supreme", 0) >= 1, "parquet rows labelled supreme", failures)
        _check(pulled["per_level"].get("high", 0) >= 1, "parquet rows labelled high", failures)
        _check(pulled["transferred_mb"] <= 5, "transfer stayed inside the 5 MB cap", failures)
        _check(pulled["vectors"] == pulled["chunks"], "parquet path vectorised every chunk", failures)
        with sqlite3.connect(config.SINGLE_DB) as db:
            courts = dict(db.execute(
                "SELECT court_level, COALESCE((SELECT header FROM chunks c WHERE c.court_level=ch.court_level LIMIT 1),'') "
                "FROM cases ch GROUP BY court_level").fetchall())
            high_court = db.execute(
                "SELECT court, jurisdiction FROM cases WHERE court_level='high' LIMIT 1"
            ).fetchone()
            meta_parq = dict(db.execute("SELECT key,value FROM meta").fetchall())
        _check("level: supreme" in courts.get("supreme", ""),
               "supreme header present in the parquet-built file", failures)
        _check(bool(high_court) and high_court[1] == "IN-KA",
               f"High Court row mapped to its jurisdiction ({high_court[1] if high_court else '-'})", failures)
        _check(meta_parq.get("mode", "").startswith("parquet"), "mode recorded in meta", failures)
        _check(meta_parq.get("parquet_mb") is not None, "parquet transfer measured", failures)

        print("opennyai: published parquet schema maps straight to embeddings")
        import opennyai_source

        oa_dir = root / "opennyai"
        oa_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({
            "Titles": ["Cheque dishonour appeal", "Wage claim appeal",
                       "Tribunal order", "Too short row"],
            "Court_Name": ["High Court of Karnataka", "Supreme Court of India",
                           "Customs, Excise and Gold Tribunal", "High Court of Delhi"],
            "Cites": [12, 40, 3, 1],
            "Cited_by": [8, 120, 0, 0],
            "Doc_url": ["https://indiankanoon.org/doc/101",
                        "https://indiankanoon.org/doc/102",
                        "https://indiankanoon.org/doc/103",
                        "https://indiankanoon.org/doc/104"],
            "Text": [BODY_C, BODY_A, BODY_B, "tiny"],
            "Doc_size": [90000, 120000, 40000, 4],
            "Case_Type": ["Financial", "Industrial & Labour", "Tax", "Civil"],
            "Court_Type": ["High Court", "Supreme Court", "Tribunal", "High Court"],
            "Court_Name_Normalized": ["High Court of Karnataka", "Supreme Court of India",
                                      "Customs Excise and Gold Tribunal", "High Court of Delhi"],
        }), oa_dir / "train-00000.parquet")

        oa = opennyai_source.build(local=str(oa_dir), download=False,
                                   embedder=StubEmbedder(), fresh=True)
        _check(oa["cases"] == 3, f"3 rows with real text embedded (got {oa['cases']})", failures)
        _check(oa["rows_seen"] == 3, f"3 usable rows reached the writer ({oa['rows_seen']})", failures)
        _check(oa["skipped_short"] == 1,
               f"the sub-200-char row was counted, not silently dropped "
               f"({oa['skipped_short']})", failures)
        _check(oa["per_level"].get("high") == 1, "Court_Type mapped one row to high", failures)
        _check(oa["per_level"].get("supreme") == 1, "Court_Type mapped one row to supreme", failures)
        _check(oa["per_level"].get("unknown") == 1,
               "tribunal kept as unknown, not relabelled district", failures)
        _check(oa["vectors"] == oa["chunks"], "every chunk vectorised", failures)
        with sqlite3.connect(config.SINGLE_DB) as db:
            oa_rows = dict(db.execute(
                "SELECT court_level, court FROM cases GROUP BY court_level").fetchall())
            hc_row = db.execute(
                "SELECT title, jurisdiction, source_url FROM cases WHERE court_level='high'"
            ).fetchone()
            unknown_header = db.execute(
                "SELECT header FROM chunks WHERE court_level='unknown' LIMIT 1"
            ).fetchone()
            oa_meta = dict(db.execute("SELECT key,value FROM meta").fetchall())
        _check(bool(hc_row) and hc_row[0] == "Cheque dishonour appeal",
               "Titles became the record title", failures)
        _check(bool(hc_row) and hc_row[1] == "IN-KA",
               f"court name gave jurisdiction ({hc_row[1] if hc_row else '-'})", failures)
        _check(bool(hc_row) and hc_row[2] == "https://indiankanoon.org/doc/101",
               "Doc_url became the provenance url", failures)
        _check(bool(unknown_header) and "LEVEL UNRECORDED" in (unknown_header[0] or ""),
               "tribunal header says the level is unrecorded", failures)
        _check(oa_meta.get("mode", "").startswith("opennyai"), "mode recorded in meta", failures)
        _check(oa_meta.get("licence") == "apache-2.0", "dataset licence recorded", failures)

        print("\nstatus report")
        import pipeline

        pipeline.main(["status"])
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("\nALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
